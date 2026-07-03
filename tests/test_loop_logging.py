"""Smoke tests for `agent_review/loop_logging.py`.

The helpers run inside Pydantic-AI's `agent.iter()` loop in real
production usage, but their formatting + counter-bumping logic is
pure-Python and can be exercised offline.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic_ai")


def test_truncate_args_caps_at_limit():
    """Args longer than the cap get truncated with a marker. Short
    args pass through unchanged. Newlines collapse to spaces so the
    log line stays single-line."""
    from cora.core.loop_logging import _ARGS_LOG_CHAR_CAP, _truncate_args

    short = "pattern=foo"
    assert _truncate_args(short) == short

    long_args = "x" * (_ARGS_LOG_CHAR_CAP + 50)
    truncated = _truncate_args(long_args)
    assert len(truncated) <= _ARGS_LOG_CHAR_CAP + len("…[truncated at NNN]") + 10
    assert "truncated" in truncated

    multiline = "line1\nline2\r\nline3"
    assert "\n" not in _truncate_args(multiline)
    assert "\r" not in _truncate_args(multiline)


def test_truncate_args_handles_none_and_dict():
    """Nil args render explicitly; dict args render via str()."""
    from cora.core.loop_logging import _truncate_args

    assert _truncate_args(None) == "<none>"
    assert "pattern" in _truncate_args({"pattern": "needle", "glob": "**/*.py"})


def test_log_wall_hit_emits_structured_line():
    """`log_wall_hit` writes a single logfmt-parseable line so log
    dashboards can extract the fields with plain `| logfmt`."""
    from cora.core.loop_logging import log_wall_hit

    captured: list[str] = []
    log_wall_hit(
        phase="T0",
        pr_number="1234",
        terminated_reason="max_iterations",
        turn_counter=[7],
        tool_call_counter={"grep_repo": 5, "git_show": 3},
        log=captured.append,
    )
    assert len(captured) == 1
    line = captured[0]
    # `agent_review iter` prefix matches the legacy `_progress` shape
    # so existing dashboard `|~ "iter"` filters keep working.
    assert line.startswith("agent_review iter pr_number=1234")
    assert "phase=T0" in line
    assert "event=wall_hit" in line
    assert "terminated_reason=max_iterations" in line
    assert "turns=7" in line
    # Sum of per-tool counts.
    assert "tool_calls=8" in line


def test_iter_with_turn_logging_extracts_tool_calls_and_summary():
    """`iter_with_turn_logging` drives the agent_run iteration and
    must:
      - emit `turn N start` on `ModelRequestNode`
      - on `CallToolsNode`, walk `model_response.parts` to:
        - log a `tool_call name=… args=…` per `ToolCallPart`
        - bump `tool_call_counter[name]`
        - call `on_tool_call(name)` if provided
        - log a `model_response thinking_chars=… text_chars=… …`
          summary
    """
    from cora.core.loop_logging import iter_with_turn_logging
    from pydantic_ai._agent_graph import ModelRequestNode, CallToolsNode
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ThinkingPart,
        ToolCallPart,
    )

    # Build a fake CallToolsNode with a ModelResponse carrying one
    # tool call + a thinking block + a small text part.
    response = ModelResponse(
        parts=[
            ThinkingPart(content="let me check the relevant config..."),
            ToolCallPart(
                tool_name="grep_repo",
                args={"pattern": "TODO"},
                tool_call_id="abc123",
            ),
            TextPart(content="here's what I found"),
        ],
        model_name="review",
        finish_reason="tool_use",
    )
    call_tools_node = CallToolsNode(model_response=response)

    # Minimal stand-in for a ModelRequestNode — only the type
    # check matters; the helper doesn't touch fields on it.
    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    captured_logs: list[str] = []
    budget_bumps: list[str] = []
    counter: dict[str, int] = {}
    turn = [0]

    async def fake_agent_run():
        yield FakeModelReqNode()
        yield call_tools_node

    # The helper uses `async for node in agent_run:` so a plain
    # async generator is enough — no real AgentRun needed.
    asyncio.run(
        iter_with_turn_logging(
            fake_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter=counter,
            log=captured_logs.append,
            on_tool_call=budget_bumps.append,
        )
    )

    assert turn[0] == 1
    assert counter == {"grep_repo": 1}
    assert budget_bumps == ["grep_repo"]
    # All lines should match the `agent_review iter pr_number=… phase=…
    # event=…` prefix so existing dashboard panels parse them as one
    # logfmt stream.
    for line in captured_logs:
        assert line.startswith("agent_review iter pr_number=1234 phase=T0")
    assert any("event=turn_start turn=1" in line for line in captured_logs)
    tool_call_lines = [
        line for line in captured_logs if "event=tool_call" in line
    ]
    assert len(tool_call_lines) == 1
    assert "name=grep_repo" in tool_call_lines[0]
    assert "pattern" in tool_call_lines[0]
    summary_lines = [
        line for line in captured_logs if "event=model_response" in line
    ]
    assert len(summary_lines) == 1
    line = summary_lines[0]
    assert "thinking_chars=" in line
    assert "text_chars=" in line
    assert "tool_calls=1" in line
    assert "finish=tool_use" in line


def test_iter_with_turn_logging_soft_fails_on_tool_call_hook():
    """A raising `on_tool_call` hook shouldn't break the iteration —
    Budget tracking is soft-fail."""
    from cora.core.loop_logging import iter_with_turn_logging
    from pydantic_ai._agent_graph import CallToolsNode
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    def explode(_name):
        raise RuntimeError("boom")

    response = ModelResponse(
        parts=[
            ToolCallPart(tool_name="grep_repo", args={}, tool_call_id="x"),
        ],
        model_name="review",
    )
    node = CallToolsNode(model_response=response)

    counter: dict[str, int] = {}

    async def fake_agent_run():
        yield node

    # Should not raise.
    asyncio.run(
        iter_with_turn_logging(
            fake_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=[1],
            tool_call_counter=counter,
            log=lambda _msg: None,
            on_tool_call=explode,
        )
    )
    assert counter == {"grep_repo": 1}
