"""`_reconcile_unprocessed_tool_calls` — make a wall-hit T0 history
resumable on T1 by stubbing dangling tool calls (ported from the
reference deployment; the T1 escalation safety net dies without it)."""

from __future__ import annotations

import cora.core.continuation as cont


def test_reconcile_empty_history_is_noop():
    assert cont._reconcile_unprocessed_tool_calls([]) == []


def test_reconcile_is_noop_when_history_is_well_formed():
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    history = [
        ModelRequest(parts=[UserPromptPart(content="review")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="git_show", args={}, tool_call_id="tc1")],
            model_name="m",
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="git_show", content="diff", tool_call_id="tc1")]
        ),
    ]
    out = cont._reconcile_unprocessed_tool_calls(history, log=lambda _m: None)
    assert out is history  # every tool call already has a return → unchanged


def test_reconcile_appends_stub_returns_for_dangling_calls():
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    # T0 wall-hit right after requesting a tool call, before its return.
    history = [
        ModelRequest(parts=[UserPromptPart(content="review")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="grep_repo", args={}, tool_call_id="tc1")],
            model_name="m",
        ),
    ]
    out = cont._reconcile_unprocessed_tool_calls(history, log=lambda _m: None)
    assert len(out) == len(history) + 1
    stub_parts = out[-1].parts
    assert any(
        isinstance(p, ToolReturnPart) and p.tool_call_id == "tc1" for p in stub_parts
    )
    # trajectory preserved — original messages untouched, only appended to
    assert out[: len(history)] == history
