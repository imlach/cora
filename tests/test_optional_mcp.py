"""MCP is an augmentation, not a dependency of the agent loop.

Deep mode used to probe a required MCP server before doing anything, so
an adopter who hadn't stood one up got `mcp-connect-failed` on every deep
review — the mode was unreachable without private infrastructure. Empty
now self-disarms; configured-but-unreachable still fails loudly, because
silently dropping tools someone asked for is the worse failure.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

pytest.importorskip("pydantic_ai")

from cora.config import ReviewerConfig
from cora.core import config as c


# ── The default self-disarms ─────────────────────────────────────────


def test_mcp_url_defaults_to_disarmed():
    assert c.DEFAULT_MCP_URL == ""
    assert ReviewerConfig().mcp_url == ""
    assert ReviewerConfig.from_env({}).mcp_url == ""
    # An explicit value still threads through untouched.
    assert (
        ReviewerConfig.from_env({"MCP_URL": "https://mcp.example/mcp"}).mcp_url
        == "https://mcp.example/mcp"
    )


# ── The tool palette reflects what actually loaded ───────────────────


def test_loaded_tool_names_drops_read_tools_without_a_server():
    from cora.core.deep_review import _loaded_tool_names

    allowed = set(c.READ_TOOLS) | set(c.LOCAL_REPO_TOOLS)

    with_mcp = _loaded_tool_names(allowed, read_enabled=True)
    without = _loaded_tool_names(allowed, read_enabled=False)

    # Local repo tools are in-process — always there.
    assert set(c.LOCAL_REPO_TOOLS) <= set(without)
    # MCP-served read tools are not, and the footer's "unused"
    # denominator must not claim otherwise.
    assert not (set(c.READ_TOOLS) - set(c.LOCAL_REPO_TOOLS)) & set(without)
    assert set(with_mcp) > set(without)


# ── deep_review_call ─────────────────────────────────────────────────


class _FakeUsage:
    input_tokens = 1
    output_tokens = 2
    total_tokens = 3
    tool_calls = 0


class _FakeResult:
    output = "Verdict: looks good\n\nbody"

    def usage(self):
        return _FakeUsage()

    def all_messages(self):
        return []


def _fake_agent():
    class FakeRun:
        result = _FakeResult()

        def all_messages(self):
            return []

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @contextlib.asynccontextmanager
        async def iter(self, prompt, **kwargs):
            yield FakeRun()

    return FakeAgent()


def _wire(monkeypatch, *, probe):
    monkeypatch.setattr("cora.core.deep_review._probe_mcp_server", probe)
    monkeypatch.setattr("cora.core.mcp_probe.probe_mcp_server", probe, raising=False)
    monkeypatch.setattr(
        "cora.core.agent.make_review_agent", lambda config: _fake_agent()
    )

    async def _noop_iter(*a, **k):
        return None

    monkeypatch.setattr("cora.core.loop_logging.iter_with_turn_logging", _noop_iter)
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", lambda: {}
    )


def _run_deep(mcp_url: str, logs: list[str] | None = None):
    from cora.core.budget import Budget
    from cora.core.deep_review import deep_review_call

    return asyncio.run(
        deep_review_call(
            endpoint_base_url="https://llm.example/v1",
            llm_gateway_key="key",
            model_alias="review",
            system_prompt="system",
            initial_user_prompt="prompt",
            budget=Budget(max_input=0, max_output=0, max_iterations=4),
            timeout_s=30,
            pr_number="42",
            repo="o/r",
            mcp_url=mcp_url,
            mcp_headers={},
            allowed_tools=set(c.READ_TOOLS) | set(c.LOCAL_REPO_TOOLS),
            gha_log=(logs.append if logs is not None else print),
            cfg=ReviewerConfig(),
        )
    )


def test_empty_mcp_url_runs_deep_on_local_tools_without_probing(monkeypatch):
    """The whole point: a working agent loop with a smaller tool
    surface, instead of a review that always skips."""
    probed: list = []

    async def _probe(*a, **k):  # pragma: no cover — must not fire
        probed.append(a)
        raise AssertionError("probed an unconfigured MCP server")

    _wire(monkeypatch, probe=_probe)
    logs: list[str] = []
    body, reason, tools, _messages = _run_deep("", logs)

    assert probed == []
    assert reason is None
    assert body == "Verdict: looks good\n\nbody"
    # Local tools present, MCP-served read tools absent.
    assert set(c.LOCAL_REPO_TOOLS) <= set(tools)
    assert not (set(c.READ_TOOLS) - set(c.LOCAL_REPO_TOOLS)) & set(tools)
    assert any("no MCP server configured" in line for line in logs)


def test_whitespace_only_mcp_url_is_also_disarmed(monkeypatch):
    async def _probe(*a, **k):  # pragma: no cover — must not fire
        raise AssertionError("probed a blank MCP server")

    _wire(monkeypatch, probe=_probe)
    _, reason, _tools, _ = _run_deep("   ")
    assert reason is None


def test_configured_but_unreachable_still_fails_loudly(monkeypatch):
    """A review that quietly drops the tools you configured is worse
    than one that asks to be retried — this path is unchanged."""

    async def _probe(*a, **k):
        return False

    _wire(monkeypatch, probe=_probe)
    body, reason, _tools, _ = _run_deep("https://mcp.example/mcp")
    assert body == ""
    assert reason == "mcp-connect-failed"


def test_configured_and_reachable_registers_the_server(monkeypatch):
    calls: list = []

    async def _probe(url, *a, **k):
        calls.append(url)
        return True

    _wire(monkeypatch, probe=_probe)
    _body, reason, tools, _ = _run_deep("https://mcp.example/mcp")
    assert calls == ["https://mcp.example/mcp"]
    assert reason is None
    assert set(c.READ_TOOLS) & set(tools)


# ── continue_on_t1 ───────────────────────────────────────────────────


def test_t1_does_not_probe_an_unconfigured_server(monkeypatch):
    from cora.core.budget import Budget
    from cora.core.continuation import continue_on_t1

    async def _probe(*a, **k):  # pragma: no cover — must not fire
        raise AssertionError("T1 probed an unconfigured MCP server")

    monkeypatch.setattr("cora.core.continuation._probe_mcp_server", _probe)
    monkeypatch.setattr(
        "cora.core.agent.make_review_agent", lambda config: _fake_agent()
    )

    async def _noop_iter(*a, **k):
        return None

    monkeypatch.setattr("cora.core.loop_logging.iter_with_turn_logging", _noop_iter)
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", lambda: {}
    )

    body, reason, tools = asyncio.run(
        continue_on_t1(
            endpoint_base_url="https://llm.example/v1",
            llm_gateway_key="key",
            t1_model_alias="main",
            system_prompt="system",
            prior_messages=[],
            initial_user_prompt="prompt",
            budget=Budget(max_input=0, max_output=0, max_iterations=4),
            timeout_s=30,
            pr_number="42",
            repo="o/r",
            mcp_url="",
            mcp_headers={},
            allowed_tools=set(c.LOCAL_REPO_TOOLS),
            cfg=ReviewerConfig(),
        )
    )
    assert reason is None
    assert body == "Verdict: looks good\n\nbody"
    assert set(c.LOCAL_REPO_TOOLS) <= set(tools)
