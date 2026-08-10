"""A second bad call to the same MCP tool must not end the review.

Pydantic-AI's per-tool retry budget is cumulative across a run and only
clears when that tool succeeds. With `retries=1` and the framework's
default `'retry'` behaviour, an early recoverable MCP error the model
routed around by picking a *different* tool leaves the budget at zero —
so an unrelated bad argument to the first tool, many turns later, raises
`UnexpectedModelBehavior("Tool 'x' exceeded max retries count of 1")`
and terminates the whole agent loop (`agent-loop-errored`).

The scenario runs end-to-end against a real in-memory MCP server: the
`mcp` SDK serves the tools, fastmcp's client speaks to it over an
in-process transport, and a scripted `FunctionModel` plays the model.
Neither half is a stand-in for the code under test — the retry
accounting exercised here is the same code a live review runs.
"""
from __future__ import annotations

import asyncio

import pytest

from cora.core.agent import AgentConfig, make_review_agent, mcp_tool_error_behavior

pytest.importorskip("pydantic_ai")
pytest.importorskip("pydantic_ai.mcp")
pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("fastmcp.client")


NOTES = {"ref/cilium-policy.md": "FQDN wildcards are per-label."}


def _build_server():
    """An MCP server with one strict tool and one forgiving one.

    `read_note` rejects an unknown id the way a real note-reading tool
    does — an exception, surfaced to the client as a tool error.
    `grep_repo` always succeeds, standing in for the successful work
    that separates the two bad calls.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("cora-test")

    @server.tool()
    def read_note(note_id: str) -> str:
        if note_id not in NOTES:
            raise ValueError(f"unknown note id: {note_id}")
        return NOTES[note_id]

    @server.tool()
    def grep_repo(pattern: str) -> str:
        return f"1 match for {pattern}"

    return server


def _scripted_model(parts):
    """A `FunctionModel` that emits `parts[i]` on the i-th request.

    Deterministic by construction: the point of the test is the
    framework's retry accounting across a fixed trajectory, so the
    model's choices are fixed too. Requests past the end repeat the
    last part rather than raising, so an unexpected extra turn shows up
    as a wrong assertion, not an IndexError.
    """
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.function import FunctionModel

    calls = {"n": 0}

    def respond(_messages, _info) -> ModelResponse:
        index = min(calls["n"], len(parts) - 1)
        calls["n"] += 1
        return ModelResponse(parts=[parts[index]])

    return FunctionModel(respond)


def _two_bad_calls_script():
    """Bad `read_note`, successful `grep_repo`, bad `read_note`, verdict.

    This is the observed failure shape: the model's first mistake is
    recovered by reaching for a different tool, so `read_note` never
    succeeds and never gets its budget back.
    """
    from pydantic_ai.messages import TextPart, ToolCallPart

    return [
        ToolCallPart(
            tool_name="read_note",
            args={"note_id": "4321"},
            tool_call_id="call-1",
        ),
        ToolCallPart(
            tool_name="grep_repo",
            args={"pattern": "FQDN"},
            tool_call_id="call-2",
        ),
        ToolCallPart(
            tool_name="read_note",
            args={"note_id": "ref/does-not-exist.md"},
            tool_call_id="call-3",
        ),
        TextPart(content="Verdict: approve"),
    ]


async def _run_scenario(tool_error_behavior: str):
    from fastmcp.client import Client
    from fastmcp.client.transports import FastMCPTransport
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset(
        Client(FastMCPTransport(_build_server())),
        tool_error_behavior=tool_error_behavior,
    )
    agent = Agent(
        _scripted_model(_two_bad_calls_script()),
        toolsets=[toolset],
        output_type=str,
        # The reviewer's budget, unchanged — the fix is the error
        # routing, not a bigger allowance.
        retries=1,
    )
    return await agent.run("review this PR")


def test_two_separated_bad_mcp_calls_do_not_abort_the_run():
    """The regression: two invalid calls to the same MCP tool, with
    successful work between them, complete the run."""
    result = asyncio.run(_run_scenario(mcp_tool_error_behavior()))

    assert result.output == "Verdict: approve"

    # Both errors reached the model as tool results it could act on,
    # rather than one of them killing the loop.
    failed = [
        part
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if getattr(part, "part_kind", None) == "tool-return"
        and getattr(part, "outcome", "success") == "failed"
    ]
    assert [p.tool_name for p in failed] == ["read_note", "read_note"]
    assert "unknown note id" in str(failed[0].content)

    # The intervening call still succeeded — the failed ones don't
    # poison the rest of the tool surface.
    succeeded = [
        part
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if getattr(part, "part_kind", None) == "tool-return"
        and getattr(part, "outcome", "success") == "success"
    ]
    assert [p.tool_name for p in succeeded] == ["grep_repo"]


def test_default_retry_behaviour_is_what_aborted_the_run():
    """Negative control: the identical trajectory under pydantic-ai's
    default `'retry'` behaviour dies on the second bad call.

    Without this the test above would still pass if the framework
    stopped counting retries for some unrelated reason, and the fix
    would look load-bearing when it wasn't.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    with pytest.raises(UnexpectedModelBehavior) as excinfo:
        asyncio.run(_run_scenario("retry"))

    assert "read_note" in str(excinfo.value)
    assert "max retries" in str(excinfo.value)


def test_mcp_tool_error_behavior_prefers_failed():
    """On a pydantic-ai that has `ToolFailed` (2.16+), the resolver
    picks `'failed'`; older builds degrade to `'retry'` rather than
    passing a value they would mishandle."""
    from cora.core.agent import MCP_TOOL_ERROR_BEHAVIOR

    behavior = mcp_tool_error_behavior()
    try:
        from pydantic_ai.exceptions import ToolFailed  # noqa: F401
    except ImportError:  # pragma: no cover - depends on installed version
        assert behavior == "retry"
    else:
        assert behavior == MCP_TOOL_ERROR_BEHAVIOR == "failed"


def test_factory_stamps_tool_error_behavior_on_every_mcp_toolset():
    """Every MCP session the factory wires carries the resolved
    behaviour — the read server, the actions server and the web-fetch
    gate all reach the model through the same path."""
    from pydantic_ai.mcp import MCPToolset

    config = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        mcp_servers=[
            ("http://mcp.test/mcp", {"Authorization": "Bearer x"}),
            ("http://actions.test/mcp", {}),
            ("http://wfg.test/mcp", {}),
        ],
    )
    agent = make_review_agent(config)

    toolsets = [t for t in agent.toolsets if isinstance(t, MCPToolset)]
    assert len(toolsets) == 3
    assert {t.tool_error_behavior for t in toolsets} == {mcp_tool_error_behavior()}


def test_allowlist_filter_preserves_tool_error_behavior():
    """The `.filtered(...)` wrapper narrows the tool surface without
    resetting the error routing underneath it."""
    config = AgentConfig(
        endpoint_base_url="http://litellm.test/v1",
        api_key="dummy",
        model_alias="review",
        system_prompt="test",
        mcp_servers=[("http://mcp.test/mcp", {})],
        mcp_allowed_tools={"read_note", "search_knowledge"},
    )
    agent = make_review_agent(config)

    wrapped = [t for t in agent.toolsets if "Filtered" in type(t).__name__]
    assert len(wrapped) == 1
    assert wrapped[0].wrapped.tool_error_behavior == mcp_tool_error_behavior()
