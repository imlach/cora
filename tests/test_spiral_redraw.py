"""Uncommitted-draw detection and the in-loop re-draw.

The failure this guards: a turn spends its whole completion budget
without committing to a tool call or a verdict. Two shapes reach the
loop — thinking-only (pydantic-ai raises) and truncated prose (it does
not) — and both used to end the review. Detection is on the response
boundary so one code path covers both; recovery is one re-send of the
identical payload, which is enough because the draw is a lottery rather
than a deterministic property of the prompt.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

pytest.importorskip("pydantic_ai")


# ── The signal ───────────────────────────────────────────────────────


def _has_verdict(text: str) -> bool:
    return "Verdict:" in text


def test_is_uncommitted_draw_truth_table():
    from cora.core.spiral import is_uncommitted_draw

    def draw(**kw):
        base = {
            "finish_reason": "length", "tool_calls": 0, "text": "", "has_verdict": _has_verdict
        }
        return is_uncommitted_draw(**{**base, **kw})

    # The signature itself: ceiling hit, nothing committed.
    assert draw() is True
    assert draw(text="I was in the middle of thinking about") is True

    # A tool call IS a commitment — the loop has somewhere to go next.
    assert draw(tool_calls=1) is False
    # So is a parseable verdict; the tail being clipped doesn't undo it.
    assert draw(text="Verdict: looks good\n\nbody that got cut o") is False
    # Any other finish reason is a normal turn.
    assert draw(finish_reason="stop") is False
    assert draw(finish_reason="tool_call") is False
    assert draw(finish_reason=None) is False
    # Case/shape tolerance — the field arrives from the wire.
    assert draw(finish_reason="LENGTH") is True

    # With no probe we cannot judge the text, so text counts as usable:
    # never re-draw over what might be a real answer.
    assert draw(text="some prose", has_verdict=None) is False
    assert draw(text="   ", has_verdict=None) is True


# ── Prefix / salvage extraction ──────────────────────────────────────


def _history(*, final_parts, finish_reason="length"):
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

    return [
        ModelRequest(parts=[UserPromptPart(content="review this PR")]),
        ModelResponse(parts=list(final_parts), finish_reason=finish_reason),
    ]


def test_committed_prefix_drops_only_a_trailing_response():
    from pydantic_ai.messages import ThinkingPart

    from cora.core.spiral import committed_prefix

    hist = _history(final_parts=[ThinkingPart(content="...")])
    prefix = committed_prefix(hist)
    assert len(prefix) == 1
    assert prefix == hist[:-1]

    # A history that already ends on a request is the payload — leave it.
    assert committed_prefix(hist[:-1]) == hist[:-1]
    assert committed_prefix([]) == []
    assert committed_prefix(None) == []


def test_has_model_response_is_the_real_did_this_tier_get_anywhere_test():
    """The `per_call_fresh_start` predicate. pydantic-ai appends the
    outgoing request BEFORE awaiting the model, so a history from a
    first-call timeout is `[ModelRequest]` — non-empty. Any `not
    messages` guard over it is dead code, which is what the T1 fresh-
    start entry was built on."""
    from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart

    from cora.core.spiral import has_model_response

    first_call_timeout = [ModelRequest(parts=[UserPromptPart(content="review")])]
    assert first_call_timeout  # the trap: truthy
    assert has_model_response(first_call_timeout) is False

    assert has_model_response([]) is False
    assert has_model_response(_history(final_parts=[TextPart(content="hi")])) is True


def test_extract_final_text_salvages_prose_but_not_reasoning():
    from pydantic_ai.messages import TextPart, ThinkingPart

    from cora.core.spiral import extract_final_text

    assert extract_final_text(_history(final_parts=[TextPart(content="half a re")])) == (
        "half a re"
    )
    # Thinking is not an answer — nothing to post.
    assert extract_final_text(_history(final_parts=[ThinkingPart(content="hm")])) == ""
    assert extract_final_text([]) == ""


def test_tool_call_names_counts_from_offset():
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from cora.core.spiral import tool_call_names

    msgs = [
        ModelResponse(parts=[ToolCallPart(tool_name="grep_repo", args={}, tool_call_id="1")]),
        ModelResponse(parts=[ToolCallPart(tool_name="git_show", args={}, tool_call_id="2")]),
    ]
    assert tool_call_names(msgs) == ["grep_repo", "git_show"]
    # The re-draw only owns the dispatches it made itself.
    assert tool_call_names(msgs, start=1) == ["git_show"]
    assert tool_call_names(msgs, start=5) == []


def test_is_completion_ceiling_exception_is_narrower_than_the_coarse_check():
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from cora.core.spiral import (
        is_completion_ceiling_exception,
        is_token_limit_exception,
    )

    thinking_only = UnexpectedModelBehavior(
        "Model token limit (12000) exceeded before any response was generated."
    )
    truncated_tool = UnexpectedModelBehavior(
        "Model token limit (12000) exceeded while generating a tool call, "
        "resulting in incomplete arguments."
    )
    unrelated = UnexpectedModelBehavior("Received empty model response")

    assert is_completion_ceiling_exception(thinking_only)
    assert is_completion_ceiling_exception(truncated_tool)
    # The distinction that matters: the coarse classifier claims every
    # UnexpectedModelBehavior, so it must not gate spending another call.
    assert not is_completion_ceiling_exception(unrelated)
    assert is_token_limit_exception(unrelated)


# ── Boundary detection inside the node loop ──────────────────────────


def _drive(response, *, verdict_probe):
    """Run one ModelRequest→CallToolsNode pair through the helper."""
    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def fake_run():
        yield FakeModelReqNode()
        yield CallToolsNode(model_response=response)

    logs: list[str] = []
    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            fake_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=logs.append,
            verdict_probe=verdict_probe,
        )
    )
    return logs


def _response(parts, finish_reason="length"):
    from pydantic_ai.messages import ModelResponse

    return ModelResponse(parts=parts, model_name="review", finish_reason=finish_reason)


def test_thinking_only_length_stop_raises_at_the_boundary():
    """The common shape. Detection here beats pydantic-ai's own raise,
    which lands one node later — same turn, but from inside framework
    code where the committed prefix is harder to reach."""
    from pydantic_ai.messages import ThinkingPart

    from cora.core.loop_logging import ReasoningSpiralDetected

    with pytest.raises(ReasoningSpiralDetected) as exc_info:
        _drive(
            _response([ThinkingPart(content="x" * 40)]), verdict_probe=_has_verdict
        )
    exc = exc_info.value
    assert exc.turn == 1
    assert exc.thinking_chars == 40
    assert exc.text_chars == 0


def test_truncated_prose_length_stop_raises_too():
    """The shape pydantic-ai never raises on: a TextPart is 'output', so
    the run would have ended normally with a verdict-less body."""
    from pydantic_ai.messages import TextPart

    from cora.core.loop_logging import ReasoningSpiralDetected

    with pytest.raises(ReasoningSpiralDetected):
        _drive(
            _response([TextPart(content="I'll start by checking whether")]),
            verdict_probe=_has_verdict,
        )


def test_committed_turns_are_left_alone():
    from pydantic_ai.messages import TextPart, ThinkingPart, ToolCallPart

    # A verdict, clipped in the tail — a real answer.
    _drive(
        _response([TextPart(content="Verdict: minor\n\nbody cut o")]),
        verdict_probe=_has_verdict,
    )
    # A tool call — the loop has its next move.
    _drive(
        _response(
            [
                ThinkingPart(content="y" * 40),
                ToolCallPart(tool_name="grep_repo", args={}, tool_call_id="t1"),
            ]
        ),
        verdict_probe=_has_verdict,
    )
    # A normal stop, however much thinking preceded it.
    _drive(
        _response([ThinkingPart(content="z" * 999)], finish_reason="stop"),
        verdict_probe=_has_verdict,
    )


def test_detection_is_disarmed_without_a_probe():
    """Triage callers and the re-draw's own second pass pass no probe —
    the helper must behave exactly as it did before the feature."""
    from pydantic_ai.messages import ThinkingPart

    logs = _drive(_response([ThinkingPart(content="x" * 40)]), verdict_probe=None)
    assert any("event=model_response" in line for line in logs)
    assert not any("event=spiral_redraw" in line for line in logs)


def test_log_spiral_redraw_emits_the_iter_stream_shape():
    from cora.core.loop_logging import log_spiral_redraw

    logs: list[str] = []
    log_spiral_redraw(
        phase="T1",
        pr_number="1234",
        turn=3,
        outcome="recovered",
        log=logs.append,
        redraw_tool_calls=2,
    )
    line = logs[0]
    # Same prefix + field names the existing dashboards parse.
    assert line.startswith("agent_review iter pr_number=1234 phase=T1 ")
    assert "event=spiral_redraw" in line
    assert "turn=3" in line
    assert "outcome=recovered" in line
    assert "redraw_tool_calls=2" in line


# ── The re-draw, end to end through deep_review_call ──────────────────


class _FakeUsage:
    input_tokens = 11
    output_tokens = 22
    total_tokens = 33
    tool_calls = 0


class _FakeResult:
    def __init__(self, output, messages):
        self.output = output
        self._messages = messages

    def usage(self):
        return _FakeUsage()

    def all_messages(self):
        return list(self._messages)


def _fake_agent(*, history, redraw_result, capture):
    """Agent stand-in: `agent.iter` yields a run whose `all_messages()`
    is the spiralled history; `agent.run` is the re-draw."""

    class FakeRun:
        def all_messages(self):
            return list(history)

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @contextlib.asynccontextmanager
        async def iter(self, prompt, **kwargs):
            yield FakeRun()

        async def run(self, prompt, **kwargs):
            capture["called"] = capture.get("called", 0) + 1
            capture["prompt"] = prompt
            capture["history"] = kwargs.get("message_history")
            capture["usage_limits"] = kwargs.get("usage_limits")
            if isinstance(redraw_result, BaseException):
                raise redraw_result
            return redraw_result

    return FakeAgent()


def _wire(monkeypatch, agent, *, spiral_turn=2):
    from cora.core.loop_logging import ReasoningSpiralDetected

    async def _probe_ok(*a, **k):
        return True

    async def _iter_spirals(*a, **k):
        raise ReasoningSpiralDetected(
            spiral_turn, thinking_chars=9000, text_chars=0, out_tokens=12000
        )

    monkeypatch.setattr("cora.core.deep_review._probe_mcp_server", _probe_ok)
    monkeypatch.setattr(
        "cora.core.mcp_probe.probe_mcp_server", _probe_ok, raising=False
    )
    monkeypatch.setattr("cora.core.agent.make_review_agent", lambda config: agent)
    monkeypatch.setattr(
        "cora.core.loop_logging.iter_with_turn_logging", _iter_spirals
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", dict
    )


def _run_deep(cfg, logs=None):
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
            mcp_url="http://mcp",
            mcp_headers={},
            allowed_tools=set(),
            max_iterations=6,
            gha_log=(logs.append if logs is not None else print),
            cfg=cfg,
        )
    )


def test_redraw_resends_the_identical_payload_and_recovers(monkeypatch):
    from pydantic_ai.messages import ThinkingPart

    from cora.config import ReviewerConfig

    history = _history(final_parts=[ThinkingPart(content="spiralled")])
    recovered = _FakeResult("Verdict: looks good\n\nreal review body", history[:-1])
    capture: dict = {}
    _wire(monkeypatch, _fake_agent(history=history, redraw_result=recovered, capture=capture))

    logs: list[str] = []
    body, reason, _tools, _messages = _run_deep(ReviewerConfig(), logs)

    assert reason is None
    assert body == "Verdict: looks good\n\nreal review body"
    # The re-draw is a re-send, not a reframe: no new user prompt, and
    # the history is exactly the payload that produced the spiral.
    assert capture["called"] == 1
    assert capture["prompt"] is None
    assert capture["history"] == history[:-1]
    # It inherits what the main loop left, not a fresh allowance.
    assert capture["usage_limits"].request_limit == 4
    assert any("event=spiral_redraw" in line and "outcome=recovered" in line for line in logs)


def test_redraw_that_spirals_again_salvages_the_truncated_turn(monkeypatch):
    """Losing the tail beats losing the review. This is also what the
    path did before the re-draw existed, so the fallback is a no-op
    against the previous behaviour rather than a new outcome."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import TextPart

    from cora.config import ReviewerConfig

    history = _history(final_parts=[TextPart(content="Half a review, no verd")])
    capture: dict = {}
    _wire(
        monkeypatch,
        _fake_agent(
            history=history,
            redraw_result=UnexpectedModelBehavior(
                "Model token limit (12000) exceeded before any response was generated."
            ),
            capture=capture,
        ),
    )

    logs: list[str] = []
    body, reason, _tools, _messages = _run_deep(ReviewerConfig(), logs)

    assert reason is None
    assert body == "Half a review, no verd"
    assert any(
        "event=spiral_redraw" in line and "outcome=spiralled_again" in line
        for line in logs
    )


def test_redraw_with_nothing_to_salvage_soft_fails(monkeypatch):
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ThinkingPart

    from cora.config import ReviewerConfig

    history = _history(final_parts=[ThinkingPart(content="only reasoning")])
    _wire(
        monkeypatch,
        _fake_agent(
            history=history,
            redraw_result=UnexpectedModelBehavior("Model token limit (12000) exceeded"),
            capture={},
        ),
    )

    body, reason, _tools, _messages = _run_deep(ReviewerConfig())
    assert body == ""
    # Same skip class the loop already routes through — a re-push retries.
    assert reason.startswith("agent-loop-errored")


def test_killswitch_disarms_detection_entirely(monkeypatch):
    """`AGENT_REVIEW_SPIRAL_REDRAW=false` must leave the loop exactly as
    it was: no probe passed down, so nothing raises and no extra call."""
    from pydantic_ai.messages import ThinkingPart

    from cora.config import ReviewerConfig

    captured_kwargs: dict = {}

    async def _iter_records(*a, **k):
        captured_kwargs.update(k)

    async def _probe_ok(*a, **k):
        return True

    history = _history(final_parts=[ThinkingPart(content="x")])
    completed = _FakeResult("Verdict: minor\n\nbody", history)

    class FakeRun:
        result = completed

        def all_messages(self):
            return list(history)

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @contextlib.asynccontextmanager
        async def iter(self, prompt, **kwargs):
            yield FakeRun()

        async def run(self, prompt, **kwargs):  # pragma: no cover — must not fire
            raise AssertionError("re-draw ran with the killswitch off")

    monkeypatch.setattr("cora.core.deep_review._probe_mcp_server", _probe_ok)
    monkeypatch.setattr(
        "cora.core.mcp_probe.probe_mcp_server", _probe_ok, raising=False
    )
    monkeypatch.setattr("cora.core.agent.make_review_agent", lambda config: FakeAgent())
    monkeypatch.setattr(
        "cora.core.loop_logging.iter_with_turn_logging", _iter_records
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", dict
    )

    _body, reason, _tools, _messages = _run_deep(
        ReviewerConfig.from_env({"AGENT_REVIEW_SPIRAL_REDRAW": "false"})
    )
    assert reason is None
    assert captured_kwargs["verdict_probe"] is None


def test_killswitch_is_typo_safe_and_defaults_on():
    from cora.config import ReviewerConfig

    assert ReviewerConfig().spiral_redraw is True
    assert ReviewerConfig.from_env({}).spiral_redraw is True
    # Only the literal "false" disables — same shape as the other
    # default-on reliability switches.
    assert ReviewerConfig.from_env({"AGENT_REVIEW_SPIRAL_REDRAW": "0"}).spiral_redraw
    assert ReviewerConfig.from_env({"AGENT_REVIEW_SPIRAL_REDRAW": "FALSE"}).spiral_redraw is False
