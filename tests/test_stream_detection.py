"""Streaming detection: telling a thinking model from a dead wire.

A non-streaming call is opaque until it completes, so both look the
same from outside — nothing arrives for minutes, and both end up
recorded as the same per-call timeout. Inside the stream they are
opposites: a steady delta rate versus silence. These tests drive the
state machine off a fake delta stream, so they need no model, no
network and no real `AgentStream`.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

pytest.importorskip("pydantic_ai")


# ── Fake stream plumbing ─────────────────────────────────────────────


class _Delta:
    """Stand-in for a `*PartDelta`, carrying only the discriminator the
    classifier reads. Duck-typed on purpose — the engine must not need
    real framework objects to count deltas."""

    def __init__(self, kind: str, content: str = ""):
        self.part_delta_kind = kind
        self.content_delta = content


class _Event:
    def __init__(self, kind: str, content: str = ""):
        self.delta = _Delta(kind, content)


class _FakeStream:
    """Yields the scripted events, then optionally hangs forever — the
    hang is the stall."""

    def __init__(self, events, *, hang_after_all=False):
        self._events = list(events)
        self._hang = hang_after_all
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i < len(self._events):
            item = self._events[self._i]
            self._i += 1
            if item is None:  # a gap: silence in the middle of a stream
                await asyncio.sleep(3600)
            return item
        if self._hang:
            await asyncio.sleep(3600)
        raise StopAsyncIteration


class _FakeModelRequestNode:
    """A `ModelRequestNode` stand-in that can open a stream. Subclasses
    the real class so the helper's `isinstance` checks still fire."""

    def __init__(self, stream):
        self._stream = stream
        self.finished = False

    @contextlib.asynccontextmanager
    async def stream(self, _ctx):
        try:
            yield self._stream
        except BaseException:
            # Mirrors pydantic-ai: raising out of the block cancels the
            # request and commits nothing. Exiting cleanly finalises.
            raise
        else:
            self.finished = True


def _node(stream):
    from pydantic_ai._agent_graph import ModelRequestNode

    node = _FakeModelRequestNode.__new__(
        type("_Node", (_FakeModelRequestNode, ModelRequestNode), {})
    )
    _FakeModelRequestNode.__init__(node, stream)
    return node


class _FakeRun:
    """An `agent_run` stand-in exposing the `ctx` the streaming path
    needs to hand to `node.stream(...)`."""

    ctx = object()

    def __init__(self, nodes):
        self._nodes = list(nodes)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._nodes:
            raise StopAsyncIteration
        return self._nodes.pop(0)


def _drive(node, *, logs=None, **kwargs):
    from cora.core.loop_logging import iter_with_turn_logging

    settings = dict(
        phase="T0",
        pr_number="1234",
        turn_counter=[0],
        tool_call_counter={},
        log=(logs.append if logs is not None else (lambda _m: None)),
        stream_detect=True,
        stall_timeout_s=0.05,
        thinking_budget_tokens=10,
    )
    settings.update(kwargs)
    return asyncio.run(iter_with_turn_logging(_FakeRun([node]), **settings))


# ── The four paths ───────────────────────────────────────────────────


def test_normal_stream_completes_and_finalises_the_node():
    """Thinking then an answer: the budget never trips, and the stream
    exits cleanly so the framework finalises the response as usual."""
    node = _node(
        _FakeStream(
            [_Event("thinking", "hm") for _ in range(5)]
            + [_Event("text", "Verdict: looks good")]
        )
    )
    _drive(node)
    assert node.finished is True


def test_thinking_past_the_budget_aborts_as_a_spiral():
    from cora.core.loop_logging import ReasoningSpiralDetected

    node = _node(_FakeStream([_Event("thinking", "x") for _ in range(50)]))
    logs: list[str] = []
    with pytest.raises(ReasoningSpiralDetected) as exc_info:
        _drive(node, logs=logs)

    assert exc_info.value.turn == 1
    assert exc_info.value.out_tokens > 10
    # Aborted mid-flight — the framework must NOT finalise a partial
    # response, or the history would carry a dead turn forward.
    assert node.finished is False
    line = next(ln for ln in logs if "event=spiral_detected" in ln)
    assert "turn=1" in line
    assert "budget_tokens=10" in line


def test_committing_to_text_lifts_the_thinking_budget():
    """Once the model is writing, a long response is not a spiral. The
    budget only governs the pre-commitment phase."""
    node = _node(
        _FakeStream(
            [_Event("thinking", "x") for _ in range(5)]
            + [_Event("text", "word ") for _ in range(200)]
        )
    )
    _drive(node)
    assert node.finished is True


def test_committing_to_a_tool_call_lifts_the_thinking_budget():
    node = _node(
        _FakeStream(
            [_Event("thinking", "x") for _ in range(5)]
            + [_Event("tool_call")]
            + [_Event("thinking", "x") for _ in range(50)]
        )
    )
    _drive(node)
    assert node.finished is True


def test_silence_aborts_as_a_stall_not_a_spiral():
    """The distinction the whole feature exists for: nothing arriving
    at all, versus reasoning arriving steadily."""
    from cora.core.loop_logging import StreamStallDetected

    node = _node(_FakeStream([_Event("thinking", "x"), None]))
    logs: list[str] = []
    with pytest.raises(StreamStallDetected) as exc_info:
        _drive(node, logs=logs)

    assert exc_info.value.turn == 1
    assert exc_info.value.streamed_tokens == 1
    assert node.finished is False
    line = next(ln for ln in logs if "event=stall_detected" in ln)
    assert "idle_s=0" in line
    assert "thinking_tokens=1" in line


def test_a_stall_after_visible_text_salvages_it():
    """Losing the wire mid-answer should not lose the answer — the
    re-draw resumes from it instead of restarting blind."""
    from cora.core.loop_logging import StreamStallDetected

    node = _node(
        _FakeStream(
            [_Event("text", "Verdict: minor"), _Event("text", "\n\nthe body so far"), None]
        )
    )
    with pytest.raises(StreamStallDetected) as exc_info:
        _drive(node)
    assert exc_info.value.partial_text == "Verdict: minor\n\nthe body so far"


def test_a_slow_but_steady_stream_is_not_a_stall():
    """2 tok/s is slow, not dead. The stall timeout has to sit above one
    slow token or it would abort every thinking turn."""

    class _Slow(_FakeStream):
        async def __anext__(self):
            await asyncio.sleep(0.01)
            return await super().__anext__()

    node = _node(_Slow([_Event("thinking", "x") for _ in range(5)]))
    _drive(node, stall_timeout_s=0.5)
    assert node.finished is True


def test_streaming_degrades_when_the_run_cannot_supply_a_context():
    """An older pydantic-ai, a stand-in agent_run, a triage caller: no
    `ctx` means fall back to awaiting the call whole, not crash."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class _NoCtxRun:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "_done", False):
                raise StopAsyncIteration
            self._done = True
            return _node(_FakeStream([_Event("thinking", "x") for _ in range(50)]))

    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            _NoCtxRun(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=lambda _m: None,
            stream_detect=True,
            thinking_budget_tokens=10,
        )
    )
    # Reached the end without streaming and without raising.
    assert turn[0] == 1
    assert isinstance(_node(_FakeStream([])), ModelRequestNode)


def test_stream_detect_off_never_opens_a_stream():
    node = _node(_FakeStream([_Event("thinking", "x") for _ in range(50)]))
    _drive(node, stream_detect=False)
    # Never entered the stream context, so it was never finalised there.
    assert node.finished is False


# ── Delta classification ─────────────────────────────────────────────


def test_delta_classification_covers_the_real_discriminators():
    from pydantic_ai.messages import (
        PartDeltaEvent,
        TextPartDelta,
        ThinkingPartDelta,
        ToolCallPartDelta,
    )

    from cora.core.loop_logging import _delta_kind, _delta_text

    thinking = PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hm"))
    text = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))
    tool = PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"a":'))

    assert _delta_kind(thinking) == "thinking"
    assert _delta_kind(text) == "text"
    assert _delta_kind(tool) == "tool"
    assert _delta_text(text) == "hello"
    # Reasoning is not an answer — it must never reach the salvage path.
    assert _delta_text(tool) == ""


# ── Config surface ───────────────────────────────────────────────────


def test_streaming_is_opt_in_and_its_tunables_thread():
    from cora.config import ReviewerConfig
    from cora.core import config as c

    assert ReviewerConfig().stream_detection is False
    assert ReviewerConfig.from_env({}).stream_detection is False
    # Reasoning stays on unless an operator explicitly says otherwise.
    assert ReviewerConfig().spiral_degrade_thinking is False

    cfg = ReviewerConfig.from_env(
        {
            "AGENT_REVIEW_STREAM_DETECTION": "true",
            "AGENT_REVIEW_STALL_TIMEOUT_S": "12.5",
            "AGENT_REVIEW_THINKING_BUDGET_TOKENS": "4000",
            "AGENT_REVIEW_SPIRAL_DEGRADE_THINKING": "true",
        }
    )
    assert cfg.stream_detection is True
    assert cfg.stall_timeout_s == 12.5
    assert cfg.thinking_budget_tokens == 4000
    assert cfg.spiral_degrade_thinking is True
    # The budget must abort before the completion ceiling does —
    # the point is to stop paying for a spiral, not to watch one land.
    assert c.THINKING_BUDGET_TOKENS < c.DEEP_MAX_OUTPUT_TOKENS


def test_reasoning_off_kwarg_matches_the_verified_template_gate():
    """The reference deployment's chat template gates on
    `enable_thinking is defined and enable_thinking is false`, so an
    ABSENT key means "default" — which for a reasoning model is on. The
    key has to be present and false."""
    from cora.core.deep_review import _no_thinking_extra_body, _thinking_extra_body

    assert _no_thinking_extra_body() == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    # The ON-direction helper stays silent when off; that silence is
    # exactly why a separate helper is needed for the OFF direction.
    assert _thinking_extra_body(False) == {}
