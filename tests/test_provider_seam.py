"""Direct cloud-provider seam — provider selection, sampling-param
hygiene, `resolved_model` fallback, and lazy-import behaviour.

Covers the `LLM_PROVIDER` / `ReviewerConfig.llm_provider` /
`AgentConfig.provider` seam added on top of the existing
OpenAI-compatible-only path (`cora.core.agent.make_review_agent`).
Every test here is offline — no network, no real Anthropic/AWS
credentials required. Tests that construct a real `AnthropicModel` /
`BedrockConverseModel` skip when the optional `anthropic` / `bedrock`
extras aren't installed, matching the existing
`pytest.importorskip("pydantic_ai.mcp")` style in `test_agent_factory.py`.
"""

from __future__ import annotations

import sys

import pytest

from cora.config import ReviewerConfig
from cora.core import config as _c
from cora.core.agent import AgentConfig, build_model_settings, make_review_agent
from cora.core.budget import model_name_from_result
from cora.providers.reporter import NullReporter
from cora.providers.retrieval import NullRetrievalProvider
from cora.review._preflight import build_run, preflight


pytest.importorskip("pydantic_ai")


# ── AgentConfig / make_review_agent provider selection ────────────────


def test_agent_config_default_provider_is_openai_compatible():
    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
    )
    assert c.provider == "openai-compatible"


def test_make_review_agent_default_provider_builds_openai_chat_model():
    from pydantic_ai.models.openai import OpenAIChatModel

    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
    )
    agent = make_review_agent(c)
    assert isinstance(agent.model, OpenAIChatModel)


def test_make_review_agent_unknown_provider_raises():
    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        provider="bogus",
    )
    with pytest.raises(ValueError, match="unknown AgentConfig.provider"):
        make_review_agent(c)


def test_make_review_agent_anthropic_provider_builds_anthropic_model():
    pytest.importorskip("anthropic")
    from pydantic_ai.models.anthropic import AnthropicModel

    c = AgentConfig(
        # Unmodified localhost placeholder — must NOT be forwarded as
        # a base_url override (see `_provider_base_url_override`).
        endpoint_base_url=_c.DEFAULT_LITELLM_BASE,
        api_key="sk-ant-test",
        model_alias="claude-sonnet-5",
        system_prompt="test",
        provider="anthropic",
    )
    agent = make_review_agent(c)
    assert isinstance(agent.model, AnthropicModel)
    assert agent.model._provider.base_url == "https://api.anthropic.com"  # noqa: SLF001


def test_make_review_agent_anthropic_respects_explicit_base_url_override():
    pytest.importorskip("anthropic")

    c = AgentConfig(
        endpoint_base_url="https://anthropic-proxy.example.com",
        api_key="sk-ant-test",
        model_alias="claude-sonnet-5",
        system_prompt="test",
        provider="anthropic",
    )
    agent = make_review_agent(c)
    assert (
        agent.model._provider.base_url  # noqa: SLF001
        == "https://anthropic-proxy.example.com"
    )


def test_make_review_agent_bedrock_provider_builds_bedrock_model(monkeypatch):
    pytest.importorskip("boto3")
    from pydantic_ai.models.bedrock import BedrockConverseModel

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

    c = AgentConfig(
        endpoint_base_url=_c.DEFAULT_LITELLM_BASE,
        api_key="",
        model_alias="claude-opus-4-8",
        system_prompt="test",
        provider="bedrock",
    )
    agent = make_review_agent(c)
    assert isinstance(agent.model, BedrockConverseModel)
    # Bare alias gets the `anthropic.` prefix Bedrock model IDs require.
    assert agent.model.model_name == "anthropic.claude-opus-4-8"


def test_make_review_agent_bedrock_does_not_double_prefix(monkeypatch):
    pytest.importorskip("boto3")

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

    c = AgentConfig(
        endpoint_base_url=_c.DEFAULT_LITELLM_BASE,
        api_key="",
        model_alias="anthropic.claude-opus-4-8",
        system_prompt="test",
        provider="bedrock",
    )
    agent = make_review_agent(c)
    assert agent.model.model_name == "anthropic.claude-opus-4-8"


def test_anthropic_provider_missing_extra_raises_friendly_import_error(monkeypatch):
    """Simulate the `anthropic` extra not being installed — `sys.modules`
    entries set to `None` make the next `import` of that name raise
    `ImportError`, regardless of what's actually installed in the test
    environment. Guards the packaging contract: a helpful message
    pointing at `cora[anthropic]`, not a raw traceback."""
    monkeypatch.setitem(sys.modules, "pydantic_ai.models.anthropic", None)
    monkeypatch.setitem(sys.modules, "pydantic_ai.providers.anthropic", None)

    c = AgentConfig(
        endpoint_base_url=_c.DEFAULT_LITELLM_BASE,
        api_key="sk-ant-test",
        model_alias="claude-sonnet-5",
        system_prompt="test",
        provider="anthropic",
    )
    with pytest.raises(ImportError, match=r"cora\[anthropic\]"):
        make_review_agent(c)


def test_bedrock_provider_missing_extra_raises_friendly_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydantic_ai.models.bedrock", None)

    c = AgentConfig(
        endpoint_base_url=_c.DEFAULT_LITELLM_BASE,
        api_key="",
        model_alias="claude-opus-4-8",
        system_prompt="test",
        provider="bedrock",
    )
    with pytest.raises(ImportError, match=r"cora\[bedrock\]"):
        make_review_agent(c)


# ── build_model_settings: sampling-param + extra_body hygiene ─────────


def test_build_model_settings_openai_compatible_keeps_temperature_and_extra():
    settings = build_model_settings(
        ReviewerConfig(llm_provider="openai-compatible"),
        max_tokens=1234,
        timeout_s=30,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    assert settings["max_tokens"] == 1234
    assert settings["temperature"] == 0.2
    assert settings["timeout"] == 30
    assert settings["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_build_model_settings_none_cfg_defaults_to_openai_compatible():
    """A None cfg (legacy call shape some call sites still support)
    must behave exactly like an explicit default-provider config —
    the compatibility contract's byte-identical-default requirement."""
    settings = build_model_settings(None, max_tokens=999, timeout_s=10)
    assert settings["temperature"] == 0.2
    assert settings["max_tokens"] == 999


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_build_model_settings_direct_providers_strip_sampling_and_extra_body(provider):
    settings = build_model_settings(
        ReviewerConfig(llm_provider=provider),
        max_tokens=1234,
        timeout_s=30,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    assert settings["max_tokens"] == 1234
    assert settings["timeout"] == 30
    assert "temperature" not in settings
    assert "extra_body" not in settings


# ── ReviewerConfig.llm_provider / from_env wiring ──────────────────────


def test_reviewer_config_llm_provider_defaults_to_openai_compatible():
    assert ReviewerConfig().llm_provider == "openai-compatible"


def test_from_env_llm_provider_default_unset():
    cfg = ReviewerConfig.from_env({})
    assert cfg.llm_provider == "openai-compatible"


def test_from_env_anthropic_api_key_fallback():
    """`LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=…` with no
    `LLM_GATEWAY_KEY` — the acceptance-criteria call shape — must
    resolve `llm_api_key` from `ANTHROPIC_API_KEY`."""
    cfg = ReviewerConfig.from_env(
        {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant-live"}
    )
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_api_key == "sk-ant-live"


def test_from_env_llm_gateway_key_wins_over_anthropic_api_key():
    cfg = ReviewerConfig.from_env(
        {
            "LLM_PROVIDER": "anthropic",
            "LLM_GATEWAY_KEY": "gw-key",
            "ANTHROPIC_API_KEY": "sk-ant-live",
        }
    )
    assert cfg.llm_api_key == "gw-key"


def test_from_env_bedrock_provider_no_api_key_required():
    cfg = ReviewerConfig.from_env({"LLM_PROVIDER": "bedrock"})
    assert cfg.llm_provider == "bedrock"
    assert cfg.llm_api_key is None


def test_from_env_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        ReviewerConfig.from_env({"LLM_PROVIDER": "not-a-real-provider"})


# ── resolved_model fallback (budget.model_name_from_result) ────────────


class _FakeResponse:
    def __init__(self, model_name):
        self.model_name = model_name


class _FakeResult:
    def __init__(self, model_name):
        self.response = _FakeResponse(model_name)


def test_model_name_from_result_reads_response_model_name():
    assert model_name_from_result(_FakeResult("claude-sonnet-5")) == "claude-sonnet-5"


def test_model_name_from_result_none_when_response_missing():
    class _NoResponse:
        pass

    assert model_name_from_result(_NoResponse()) is None


def test_model_name_from_result_none_when_model_name_empty():
    assert model_name_from_result(_FakeResult("")) is None


def test_model_name_from_result_never_raises_on_garbage_input():
    assert model_name_from_result(object()) is None
    assert model_name_from_result(None) is None


# ── ReviewRun.endpoint_base_url — provider-aware base URL shaping ──────


def _run(**cfg_overrides):
    base = dict(repo="owner/repo", pr_number="1", llm_api_key="key")
    base.update(cfg_overrides)
    cfg = ReviewerConfig(**base)
    return build_run(
        cfg,
        reporter=NullReporter(),
        retrieval=NullRetrievalProvider(),
        git=None,
        second_opinion=None,
    )


def test_endpoint_base_url_openai_compatible_appends_v1():
    run = _run()  # default provider, default llm_base_url
    assert run.endpoint_base_url == f"{_c.DEFAULT_LITELLM_BASE}/v1"


def test_endpoint_base_url_openai_compatible_strips_trailing_slash_before_v1():
    run = _run(llm_base_url="http://litellm.test/")
    assert run.endpoint_base_url == "http://litellm.test/v1"


def test_endpoint_base_url_anthropic_ignores_unmodified_default_placeholder():
    """No `LITELLM_BASE_URL` override — the localhost placeholder must
    not leak through to the direct-SDK path (see `ReviewRun.endpoint_base_url`
    and `agent._provider_base_url_override`, which both key off the
    same `DEFAULT_LITELLM_BASE` sentinel)."""
    run = _run(llm_provider="anthropic")
    assert run.endpoint_base_url == ""


def test_endpoint_base_url_anthropic_respects_explicit_override():
    run = _run(
        llm_provider="anthropic",
        llm_base_url="https://anthropic-proxy.example.com",
    )
    assert run.endpoint_base_url == "https://anthropic-proxy.example.com"


def test_endpoint_base_url_bedrock_ignores_unmodified_default_placeholder():
    run = _run(llm_provider="bedrock")
    assert run.endpoint_base_url == ""


# ── preflight: bedrock doesn't require an API key ──────────────────────


def _preflight_skip_reasons(cfg, monkeypatch):
    import cora.core.pr_context as prc_mod

    monkeypatch.setattr(
        prc_mod,
        "fetch_pr_metadata",
        lambda pr: {
            "title": "t",
            "body": "b",
            "labels": [],
            "author": {"login": "alice", "is_bot": False},
            "baseRefName": "main",
            "headRefName": "feat/x",
            "isCrossRepository": False,
            "additions": 1,
            "deletions": 1,
            "changedFiles": 1,
        },
    )
    monkeypatch.setattr(prc_mod, "_pr_head_sha", lambda: None)
    run = _run(**cfg)
    result = preflight(run)
    return result.terminated_reason if result is not None else None


def test_preflight_bedrock_does_not_require_api_key(monkeypatch):
    reason = _preflight_skip_reasons(
        {"llm_provider": "bedrock", "llm_api_key": None}, monkeypatch
    )
    assert reason != "secret-missing"


def test_preflight_anthropic_still_requires_api_key(monkeypatch):
    reason = _preflight_skip_reasons(
        {"llm_provider": "anthropic", "llm_api_key": None}, monkeypatch
    )
    assert reason == "secret-missing"


def test_preflight_openai_compatible_still_requires_api_key(monkeypatch):
    """Unchanged default-path behaviour — the compatibility contract."""
    reason = _preflight_skip_reasons({"llm_api_key": None}, monkeypatch)
    assert reason == "secret-missing"
