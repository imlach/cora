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
    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ThinkingPart,
        ToolCallPart,
    )

    from cora.core.loop_logging import iter_with_turn_logging

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
    from pydantic_ai._agent_graph import CallToolsNode
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from cora.core.loop_logging import iter_with_turn_logging

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


def _request_node(*parts):
    """A `ModelRequestNode` carrying the given request parts — the shape
    the framework hands back after a turn's tools have run."""
    from pydantic_ai._agent_graph import ModelRequestNode
    from pydantic_ai.messages import ModelRequest

    return ModelRequestNode(request=ModelRequest(parts=list(parts)))


def test_iter_with_turn_logging_counts_tool_errors_and_recovery():
    """A failed tool result emits `tool_error` and bumps the counter; a
    later success on the same tool emits `tool_recovered` carrying how
    many errors it closes out.

    Both halves matter: the error count alone can't tell a model that
    self-corrected from one that never got the tool to work.
    """
    from pydantic_ai.messages import ToolReturnPart

    from cora.core.loop_logging import iter_with_turn_logging

    captured: list[str] = []
    errors: dict[str, int] = {}

    async def fake_agent_run():
        yield _request_node(
            ToolReturnPart(
                tool_name="read_note",
                content="unknown note id: 4321",
                tool_call_id="c1",
                outcome="failed",
            )
        )
        yield _request_node(
            ToolReturnPart(
                tool_name="read_note",
                content="the note body",
                tool_call_id="c2",
            )
        )

    asyncio.run(
        iter_with_turn_logging(
            fake_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=[0],
            tool_call_counter={},
            tool_error_counter=errors,
            log=captured.append,
        )
    )

    assert errors == {"read_note": 1}
    error_lines = [ln for ln in captured if "event=tool_error" in ln]
    assert len(error_lines) == 1
    assert "name=read_note" in error_lines[0]
    assert "kind=failed_result" in error_lines[0]
    assert "errors=1" in error_lines[0]
    assert "unknown note id" in error_lines[0]

    recovered = [ln for ln in captured if "event=tool_recovered" in ln]
    assert len(recovered) == 1
    assert "name=read_note after_errors=1" in recovered[0]


def test_iter_with_turn_logging_counts_retry_prompts_as_tool_errors():
    """A framework `RetryPromptPart` — bad arguments, unknown tool name
    — is the same class of recoverable error as a failed MCP result and
    counts alongside it."""
    from pydantic_ai.messages import RetryPromptPart

    from cora.core.loop_logging import iter_with_turn_logging

    captured: list[str] = []
    errors: dict[str, int] = {}

    async def fake_agent_run():
        yield _request_node(
            RetryPromptPart(
                tool_name="search_knowledge",
                content="query is required",
                tool_call_id="c1",
            )
        )

    asyncio.run(
        iter_with_turn_logging(
            fake_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=[0],
            tool_call_counter={},
            tool_error_counter=errors,
            log=captured.append,
        )
    )

    assert errors == {"search_knowledge": 1}
    assert any("kind=retry_prompt" in ln for ln in captured)
    assert not any("event=tool_recovered" in ln for ln in captured)


def test_log_wall_hit_reports_tool_errors_when_supplied():
    """The break marker carries the error total, and omits the field
    entirely when the caller has no counter to report."""
    from cora.core.loop_logging import log_wall_hit

    with_errors: list[str] = []
    log_wall_hit(
        phase="T0",
        pr_number="1234",
        terminated_reason="wall_time",
        turn_counter=[4],
        tool_call_counter={"grep_repo": 6},
        tool_error_counter={"read_note": 2, "grep_repo": 1},
        log=with_errors.append,
    )
    assert "tool_errors=3" in with_errors[0]

    without: list[str] = []
    log_wall_hit(
        phase="T0",
        pr_number="1234",
        terminated_reason="wall_time",
        turn_counter=[4],
        tool_call_counter={"grep_repo": 6},
        log=without.append,
    )
    assert "tool_errors=" not in without[0]
