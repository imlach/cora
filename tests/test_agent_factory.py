"""Smoke tests for the Pydantic-AI Agent factory used by the
reviewer (deep mode) and the triage agent.

`pytest.importorskip("pydantic_ai")` guards the whole file so it's
still runnable in environments without the framework installed.
"""
from __future__ import annotations

import pytest

from cora.core.agent import AgentConfig, Deps, make_review_agent

pytest.importorskip("pydantic_ai")


def test_deps_constructs_with_no_op_defaults():
    """Deps should be constructable with only the required fields;
    observability callables default to no-ops so tests / local-dev
    invocations don't need to wire real Loki / GHA outputs."""
    d = Deps(repo="owner/repo", pr_number="1234")
    assert d.repo == "owner/repo"
    assert d.pr_number == "1234"
    assert d.mode == "quick"
    assert d.eval_mode is False
    assert d.gh_app_token is None
    # No-op callables are wired by default — calling them shouldn't
    # raise. Using the documented call shapes.
    d.loki_push("test line", None)
    d.loki_push("test line", {"consumer": "test"})
    d.gha_log("test message")


def test_deps_accepts_real_callables():
    """Confirm the callable fields take real functions, not just the
    default no-ops. This is the path real deployment wiring uses."""
    captured: list[str] = []

    def fake_loki(line: str, labels: dict | None = None) -> None:
        captured.append(f"loki:{line}")

    def fake_gha(msg: str) -> None:
        captured.append(f"gha:{msg}")

    d = Deps(
        repo="owner/repo",
        pr_number="1234",
        loki_push=fake_loki,
        gha_log=fake_gha,
    )
    d.loki_push("L", None)
    d.gha_log("G")
    assert captured == ["loki:L", "gha:G"]


def test_agent_config_minimum_required_fields():
    """AgentConfig requires endpoint + api_key + model + system_prompt;
    everything else has sensible defaults."""
    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="You are a test reviewer.",
    )
    assert c.mcp_servers == []  # default empty list
    assert c.retries == 1


def test_make_review_agent_returns_pydantic_ai_agent():
    """Factory should produce an importable Pydantic-AI Agent.
    No network call here — Agent construction is offline; .run()
    is what hits the endpoint."""
    from pydantic_ai import Agent

    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="You are a test reviewer.",
    )
    agent = make_review_agent(c)
    assert isinstance(agent, Agent)
    # Sanity: the deps_type is wired so a future RunContext[Deps]
    # access in a tool body has the right typed shape.
    assert agent._deps_type is Deps


def test_make_review_agent_accepts_mcp_server_entries():
    """AgentConfig accepts mcp_servers entries and the factory wires
    them as `MCPToolset` toolsets on the returned Agent.

    Construction is offline; the toolsets don't open transports until
    `agent.run` enters its async context.
    """
    pytest.importorskip("pydantic_ai.mcp")
    from pydantic_ai.mcp import MCPToolset

    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        mcp_servers=[
            ("http://mcp.test/mcp", {"Authorization": "Bearer x"}),
            ("http://wfg.test/mcp", {}),
        ],
    )
    agent = make_review_agent(c)
    assert agent is not None
    # `agent.toolsets` includes the auto-added function-toolset (for
    # `tools=[]`) + each user-provided toolset. Count the MCP ones —
    # one per `mcp_servers` entry. The default config doesn't filter,
    # so each entry is a bare `MCPToolset` (no wrapper).
    declared = list(agent.toolsets)
    mcp_count = sum(1 for t in declared if isinstance(t, MCPToolset))
    assert mcp_count == 2


def test_make_review_agent_applies_mcp_allowed_tools_filter():
    """When `mcp_allowed_tools` is set, each MCP toolset wraps with a
    FilteredToolset so only allowed tools reach the agent."""
    pytest.importorskip("pydantic_ai.mcp")
    from pydantic_ai.mcp import MCPToolset

    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        mcp_servers=[
            ("http://mcp.test/mcp", {"Authorization": "Bearer x"}),
        ],
        mcp_allowed_tools={"search_knowledge", "grep_repo"},
    )
    agent = make_review_agent(c)
    declared = list(agent.toolsets)
    # With the allowlist, the MCPToolset is wrapped by FilteredToolset
    # — no bare MCPToolset remains. Look for "Filtered" in the class
    # name (the wrapping toolset's concrete class is a framework-
    # internal type; name-based check keeps the test stable across
    # framework restructures).
    assert not any(isinstance(t, MCPToolset) for t in declared)
    assert any("Filtered" in type(t).__name__ for t in declared)


def test_make_review_agent_accepts_local_tools():
    """`local_tools` flow into the Agent's `tools=` parameter — typed
    Python functions get schema inferred from signatures (the
    deep-mode grep_repo / git_show registration pattern)."""

    async def my_local_tool(query: str) -> str:
        """Test local tool.

        Args:
            query: A test query string.
        """
        return f"result: {query}"

    c = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        local_tools=[my_local_tool],
    )
    agent = make_review_agent(c)
    assert agent is not None


# ── Session-affinity header ──────────────────────────────────────────


def test_session_header_is_absent_unless_configured():
    """No configured value must mean no header at all — an unconfigured
    deployment's requests stay byte-identical to before the feature."""
    from cora.core.agent import SESSION_HEADER, AgentConfig, make_review_agent

    for session_id in (None, "", "   "):
        agent = make_review_agent(
            AgentConfig(
                endpoint_base_url="https://llm.example/v1",
                api_key="k",
                model_alias="review",
                system_prompt="sys",
                session_id=session_id,
            )
        )
        client = agent.model.client._client
        assert SESSION_HEADER not in client.headers


def test_session_header_rides_every_model_call_when_set():
    """The http client is the seam because it covers ALL model calls,
    including the ones pydantic-ai issues internally."""
    from cora.core.agent import SESSION_HEADER, AgentConfig, make_review_agent

    agent = make_review_agent(
        AgentConfig(
            endpoint_base_url="https://llm.example/v1",
            api_key="k",
            model_alias="review",
            system_prompt="sys",
            session_id="pr-3305-run-7",
        )
    )
    assert agent.model.client._client.headers[SESSION_HEADER] == "pr-3305-run-7"


def test_session_header_env_treats_empty_as_unset():
    from cora.config import ReviewerConfig

    assert ReviewerConfig().session_header is None
    assert ReviewerConfig.from_env({}).session_header is None
    assert ReviewerConfig.from_env({"AGENT_REVIEW_SESSION_HEADER": ""}).session_header is None
    assert (
        ReviewerConfig.from_env(
            {"AGENT_REVIEW_SESSION_HEADER": "pr-42"}
        ).session_header
        == "pr-42"
    )
