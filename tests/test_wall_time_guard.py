"""Tests pinning the wall-time guard contract for the agent review loop.

An observed review hung for 20 minutes despite `WALL_TIME_S=540`,
because the deadline mechanism described in `agent_review/config.py`
was documentation-only — `WALL_TIME_S` was read but never enforced
inside the loop. Only `UsageLimits(request_limit=N)` capped turns;
with T0=12 + T1=6 turns × 180s per_call_timeout, the script could run
~54 minutes against the workflow's 20-min `timeout-minutes` ceiling.
GHA SIGTERMed the job mid-call and no comment landed on the PR.

These tests pin the contract: when a loop_deadline_monotonic is
passed in and elapses, the iteration terminates within a small grace
window — both for the clean in-loop case (`WallTimeExceeded`) and the
outer-ceiling case (`asyncio.TimeoutError` from `asyncio.wait_for`).
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("pydantic_ai")


def test_iter_with_turn_logging_raises_walltime_when_deadline_past():
    """Once `time.monotonic() >= loop_deadline_monotonic`, the next
    node iteration must raise `WallTimeExceeded` — not silently
    continue. The fake `async for` yields nodes faster than the
    deadline allows so the trip is deterministic."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import (
        WallTimeExceeded,
        iter_with_turn_logging,
    )

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def slow_agent_run():
        for _ in range(20):
            yield FakeModelReqNode()
            await asyncio.sleep(0.05)

    start = time.monotonic()
    deadline = start + 0.2  # 200ms budget; node loop produces ~50ms/node

    with pytest.raises(WallTimeExceeded) as exc_info:
        asyncio.run(
            iter_with_turn_logging(
                slow_agent_run(),
                phase="T0",
                pr_number="1848",
                turn_counter=[0],
                tool_call_counter={},
                log=lambda _msg: None,
                loop_deadline_monotonic=deadline,
            )
        )

    elapsed = time.monotonic() - start
    # Must terminate within budget + a small grace window — not after
    # the fake generator's 20 × 50ms = 1s of nodes have all yielded.
    assert elapsed < 1.0, f"deadline trip took {elapsed:.2f}s, expected <1.0s"
    # Overshoot field should be a non-negative float (how far past the
    # deadline we noticed); used by callers for tuning the headroom.
    assert exc_info.value.overshoot_s >= 0


def test_iter_with_turn_logging_no_deadline_runs_to_completion():
    """When `loop_deadline_monotonic` is None (default), the helper
    never checks a deadline — the existing call sites that don't pass
    one (tests, future triage adapter) keep working unchanged."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def short_agent_run():
        for _ in range(3):
            yield FakeModelReqNode()

    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            short_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=lambda _msg: None,
            # Deliberately omit loop_deadline_monotonic — back-compat
            # path that existing callers still rely on.
        )
    )
    assert turn[0] == 3


def test_iter_with_turn_logging_deadline_in_future_completes():
    """A deadline far enough in the future should not trip on a short
    run — confirms the comparator direction (`now >= deadline`, not
    `now <= deadline`)."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def short_agent_run():
        for _ in range(3):
            yield FakeModelReqNode()

    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            short_agent_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=lambda _msg: None,
            loop_deadline_monotonic=time.monotonic() + 60,
        )
    )
    assert turn[0] == 3


def test_wall_time_exceeded_carries_overshoot():
    """The exception's overshoot_s field is what the wall_hit log
    line uses to record how far past the deadline the trip noticed —
    keeps headroom tuning honest (a consistently large overshoot
    means the in-loop check is firing late and POST_HEADROOM_S
    needs a bump)."""
    from cora.core.loop_logging import WallTimeExceeded

    exc = WallTimeExceeded(overshoot_s=3.14)
    assert exc.overshoot_s == 3.14
    assert "3.1s" in str(exc)


def test_asyncio_wait_for_caps_iter_when_in_loop_check_misses():
    """The in-loop check fires between graph nodes. If a single
    node hangs (LLM call that doesn't honour the `timeout=` setting,
    a wedged MCP transport), the in-loop check never gets a chance
    to fire — `asyncio.wait_for` is the backstop. This test mimics
    that pathology by sleeping forever inside an async generator
    that the wrapper code in `deep_review_call` would wrap with
    `wait_for`.
    """
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def hung_agent_run():
        yield FakeModelReqNode()
        # Single node that never returns — simulates a wedged
        # LLM call where the framework's timeout doesn't actually
        # cancel the in-flight request.
        await asyncio.sleep(60)
        yield FakeModelReqNode()

    inner = iter_with_turn_logging(
        hung_agent_run(),
        phase="T0",
        pr_number="1848",
        turn_counter=[0],
        tool_call_counter={},
        log=lambda _msg: None,
        loop_deadline_monotonic=time.monotonic() + 60,
    )

    start = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(inner, timeout=0.3))
    elapsed = time.monotonic() - start
    # The outer wait_for cancels at ~0.3s + small overhead — without
    # it, the inner `await asyncio.sleep(60)` would hold the loop
    # for a full minute before the deadline check ran.
    assert elapsed < 1.0, f"wait_for cap took {elapsed:.2f}s, expected <1.0s"


def test_deep_review_signature_accepts_loop_deadline():
    """`deep_review_call` must accept `loop_deadline_monotonic` as a
    keyword arg — pins the public contract so the entrypoint in
    `agent_review.py` doesn't silently regress (TypeError on call)
    after a future refactor."""
    import inspect

    from cora.core.deep_review import deep_review_call

    sig = inspect.signature(deep_review_call)
    assert "loop_deadline_monotonic" in sig.parameters
    # Must default to None so existing callers (tests, triage adapter)
    # don't have to pass it.
    assert sig.parameters["loop_deadline_monotonic"].default is None


def test_deep_review_preserves_wall_time_when_cleanup_raises(monkeypatch):
    """When the inner wall-hit handler sets terminated_reason="wall_time"
    and the framework's `async with agent:` / `async with agent.iter()`
    cleanup THEN raises (CancelledError from MCP transport teardown
    after asyncio.wait_for fires — empty `str()`), the outer
    `except Exception` must NOT overwrite the legitimate wall-time
    reason with `agent-loop-errored:`.

    A live self-review hit this — the wall-time guard fired
    cleanly at ~8:35, then 14s later the agent_run cleanup raised
    something with empty `str()` that the outer handler caught and
    surfaced as `agent-loop-errored:` (instead of the clean
    `wall_time` the inner handler had recorded).
    """
    import asyncio
    from types import SimpleNamespace

    from cora.core import deep_review as dr

    # Fake agent that succeeds in connecting but whose
    # `agent.iter()` immediately raises WallTimeExceeded inside the
    # body, then raises a second empty-string exception during
    # cleanup. Tests the early_terminated_reason preservation.
    from cora.core.loop_logging import WallTimeExceeded

    class FakeAgentRun:
        def __init__(self):
            self.result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            # Mimic the post-wait_for cleanup that raises
            # CancelledError (str("") == ""). Use bare ValueError
            # with empty message instead of CancelledError because
            # CancelledError doesn't inherit from Exception in 3.11+.
            raise ValueError("")

        def all_messages(self):
            return []

    class FakeAgent:
        def __aenter__(self):
            async def _ae():
                return self
            return _ae()

        def __aexit__(self, *a):
            async def _ax():
                return None
            return _ax()

        def iter(self, *_args, **_kwargs):
            return FakeAgentRun()

    # Stub the probe so deep_review_call gets past the MCP check.
    async def _probe_ok(*_a, **_kw):
        return True

    # Stub iter_with_turn_logging to immediately raise WallTimeExceeded.
    async def _iter_raises_wall_time(*_a, **_kw):
        raise WallTimeExceeded(overshoot_s=0.5)

    monkeypatch.setattr(dr, "_probe_mcp_server", _probe_ok)
    # Replace iter_with_turn_logging at the import site inside the
    # function.
    monkeypatch.setattr(
        "cora.core.loop_logging.iter_with_turn_logging",
        _iter_raises_wall_time,
    )
    # Replace make_review_agent so it returns our FakeAgent.
    monkeypatch.setattr(
        "cora.core.agent.make_review_agent",
        lambda _cfg: FakeAgent(),
    )

    budget = SimpleNamespace(
        add_tool_call=lambda _name: None,
        add_usage=lambda _u: None,
        set_resolved_model=lambda _m: None,
    )

    body, reason, _tools, _msgs = asyncio.run(
        dr.deep_review_call(
            endpoint_base_url="http://fake/v1",
            llm_gateway_key="fake",
            model_alias="review",
            system_prompt="",
            initial_user_prompt="",
            budget=budget,
            timeout_s=10,
            pr_number="1853",
            repo="owner/repo",
            mcp_url="http://fake-mcp",
            mcp_headers={},
            allowed_tools=set(),
            max_iterations=12,
            loop_deadline_monotonic=time.monotonic() + 60,
            gha_log=lambda _msg: None,
        )
    )
    assert body == ""
    # The key assertion: the legitimate wall_time reason is preserved
    # even though the FakeAgentRun's __aexit__ raised an empty-string
    # ValueError during cleanup. Pre-fix, this would have been
    # `agent-loop-errored: ` (the empty trail).
    assert reason == "wall_time", f"expected 'wall_time', got {reason!r}"


def test_continue_on_t1_signature_accepts_loop_deadline():
    """`continue_on_t1` must accept `loop_deadline_monotonic` — amain
    computes a *fresh* T1 deadline (`now + T1_WALL_TIME_S`, with a
    floor of `start + T0_WALL_TIME_S + T1_WALL_TIME_S` so an early-
    finishing T0 yields the slack to T1) and passes it in."""
    import inspect

    from cora.core.continuation import continue_on_t1

    sig = inspect.signature(continue_on_t1)
    assert "loop_deadline_monotonic" in sig.parameters
    assert sig.parameters["loop_deadline_monotonic"].default is None


def test_iter_with_turn_logging_accepts_context_refresher_kwargs():
    """`iter_with_turn_logging` must accept the push-injection kwargs
    (`context_refresher`, `extend_deadline_fn`) and default both to
    None so existing callers (tests, triage adapter) stay green."""
    import inspect

    from cora.core.loop_logging import iter_with_turn_logging

    sig = inspect.signature(iter_with_turn_logging)
    assert "context_refresher" in sig.parameters
    assert sig.parameters["context_refresher"].default is None
    assert "extend_deadline_fn" in sig.parameters
    assert sig.parameters["extend_deadline_fn"].default is None


def test_iter_injects_user_prompt_into_next_model_request():
    """When the refresher returns a body on a CallToolsNode boundary,
    `iter_with_turn_logging` must append a `UserPromptPart` to the
    NEXT `ModelRequestNode.request.parts` — that's the natural carrier
    of the user turn in pydantic-ai's graph.
    """

    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        UserPromptPart,
    )

    from cora.core.loop_logging import iter_with_turn_logging

    # Build a minimal CallToolsNode → ModelRequestNode → done sequence.
    # CallToolsNode carries a model_response so the iter helper emits
    # its model_response log line; the ModelRequestNode carries an
    # empty request whose `parts` we'll inspect after iteration.

    fake_response = ModelResponse(parts=[])
    call_node = CallToolsNode(model_response=fake_response)
    next_request = ModelRequest(parts=[])
    next_node = ModelRequestNode(request=next_request)

    async def fake_iter():
        # Caller sees turn 1 ModelRequestNode start, then the
        # CallToolsNode response, then the injection-target ModelRequestNode.
        yield ModelRequestNode(request=ModelRequest(parts=[]))
        yield call_node
        yield next_node

    class StubRefresher:
        def __init__(self):
            self.last_source = "head"
            self.calls = 0

        async def refresh(self, *, turn):
            self.calls += 1
            return "INJECTION-BODY-XYZ" if self.calls == 1 else None

        def can_extend(self):
            return True

        def record_extension(self):
            self.recorded = True

    refresher = StubRefresher()
    extended = []

    def extend_fn(seconds):
        extended.append(seconds)

    asyncio.run(
        iter_with_turn_logging(
            fake_iter(),
            phase="T0",
            pr_number="9999",
            turn_counter=[0],
            tool_call_counter={},
            log=lambda _m: None,
            context_refresher=refresher,
            extend_deadline_fn=extend_fn,
        )
    )

    # The injection body should now be in the second ModelRequestNode's
    # request.parts as a UserPromptPart.
    assert len(next_request.parts) == 1
    part = next_request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert "INJECTION-BODY-XYZ" in str(part.content)
    # The extend callback must have fired with the +90s contract.
    assert extended == [90.0]


def test_iter_skips_injection_when_refresher_returns_none():
    """If the refresher returns None (no delta), no part is added to
    any ModelRequest and the extend callback is not invoked."""
    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import ModelRequest, ModelResponse

    from cora.core.loop_logging import iter_with_turn_logging

    next_request = ModelRequest(parts=[])

    async def fake_iter():
        yield ModelRequestNode(request=ModelRequest(parts=[]))
        yield CallToolsNode(model_response=ModelResponse(parts=[]))
        yield ModelRequestNode(request=next_request)

    class NullRefresher:
        last_source = None

        async def refresh(self, *, turn):
            return None

        def can_extend(self):
            return True

        def record_extension(self):
            pass

    extended = []
    asyncio.run(
        iter_with_turn_logging(
            fake_iter(),
            phase="T0",
            pr_number="9999",
            turn_counter=[0],
            tool_call_counter={},
            log=lambda _m: None,
            context_refresher=NullRefresher(),
            extend_deadline_fn=lambda s: extended.append(s),
        )
    )
    assert next_request.parts == []
    assert extended == []


def test_iter_skips_extension_when_cap_hit():
    """Once the refresher reports `can_extend() == False`, further
    injections still land but the extend callback is NOT called —
    enforcing the documented +3-injection cap."""
    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

    from cora.core.loop_logging import iter_with_turn_logging

    next_request = ModelRequest(parts=[])

    async def fake_iter():
        yield ModelRequestNode(request=ModelRequest(parts=[]))
        yield CallToolsNode(model_response=ModelResponse(parts=[]))
        yield ModelRequestNode(request=next_request)

    class CappedRefresher:
        last_source = "ci"

        async def refresh(self, *, turn):
            return "BODY"

        def can_extend(self):
            return False  # cap already hit

        def record_extension(self):
            pass

    extended = []
    asyncio.run(
        iter_with_turn_logging(
            fake_iter(),
            phase="T0",
            pr_number="9999",
            turn_counter=[0],
            tool_call_counter={},
            log=lambda _m: None,
            context_refresher=CappedRefresher(),
            extend_deadline_fn=lambda s: extended.append(s),
        )
    )
    # Injection landed.
    assert len(next_request.parts) == 1
    assert isinstance(next_request.parts[0], UserPromptPart)
    # But extension was skipped because cap was hit.
    assert extended == []


def test_deep_review_signature_accepts_context_refresher():
    """deep_review_call must accept `context_refresher` as a kwarg
    with default None for back-compat."""
    import inspect

    from cora.core.deep_review import deep_review_call

    sig = inspect.signature(deep_review_call)
    assert "context_refresher" in sig.parameters
    assert sig.parameters["context_refresher"].default is None


def test_continue_on_t1_signature_accepts_context_refresher():
    """continue_on_t1 mirrors deep_review_call — same instance
    threaded through for dedupe-state continuity across T0 → T1."""
    import inspect

    from cora.core.continuation import continue_on_t1

    sig = inspect.signature(continue_on_t1)
    assert "context_refresher" in sig.parameters
    assert sig.parameters["context_refresher"].default is None


def test_default_wall_constants_sum_with_headroom():
    """`DEFAULT_WALL_TIME_S` is the derived envelope reported in the
    start_line + summary, so it MUST equal `T0 + T1 + POST_HEADROOM_S`.
    Drift between these would surface as the dashboard reporting a
    wall budget that doesn't match what the loop actually enforces."""
    from cora.core.config import (
        DEFAULT_T0_WALL_TIME_S,
        DEFAULT_T1_WALL_TIME_S,
        DEFAULT_WALL_TIME_S,
        POST_HEADROOM_S,
    )

    assert DEFAULT_WALL_TIME_S == (
        DEFAULT_T0_WALL_TIME_S + DEFAULT_T1_WALL_TIME_S + POST_HEADROOM_S
    )
    # Sanity floor: T1 must be at least as large as T0 (T1 runs on the
    # bigger model and is where reasoning typically completes). If a
    # future tuning inverts this, that's likely a mistake — surface it.
    assert DEFAULT_T1_WALL_TIME_S >= DEFAULT_T0_WALL_TIME_S


def test_t1_deadline_floor_protects_against_slow_t0():
    """The whole point of the split: even if T0 burns its FULL cap
    (or somehow runs slightly past it), T1's deadline computed as
    `max(now + T1, start + T0 + T1)` still gives T1 a fresh full
    window. Reproduces the observed pathology where the prior shared-
    deadline shape left T1 with -54s and skipped it."""
    t0_wall_time_s = 360
    t1_wall_time_s = 600
    start = 1000.0  # synthetic monotonic origin

    # T0 ran exactly to its cap: now == start + T0
    now_at_t0_end = start + t0_wall_time_s
    t1_deadline = max(
        now_at_t0_end + t1_wall_time_s,
        start + t0_wall_time_s + t1_wall_time_s,
    )
    t1_budget = t1_deadline - now_at_t0_end
    assert t1_budget == t1_wall_time_s, (
        f"T1 must get full {t1_wall_time_s}s on a wall-hit T0, got {t1_budget}"
    )

    # T0 finished early at 100s in — T1 should absorb the slack via
    # the `start + T0 + T1` floor (so T1's deadline is the same as if
    # T0 had run to cap, giving T1 the longer effective window).
    now_at_t0_end = start + 100
    t1_deadline = max(
        now_at_t0_end + t1_wall_time_s,
        start + t0_wall_time_s + t1_wall_time_s,
    )
    t1_budget = t1_deadline - now_at_t0_end
    expected = t0_wall_time_s - 100 + t1_wall_time_s  # 260 + 600 = 860
    assert t1_budget == expected, (
        f"early-finish T0 should yield slack to T1: expected {expected}s, "
        f"got {t1_budget}"
    )

    # T0 somehow ran past its cap (in-loop guard is best-effort) — T1
    # still gets a fresh window via the `now + T1` floor.
    now_at_t0_end = start + t0_wall_time_s + 30  # 30s overshoot
    t1_deadline = max(
        now_at_t0_end + t1_wall_time_s,
        start + t0_wall_time_s + t1_wall_time_s,
    )
    t1_budget = t1_deadline - now_at_t0_end
    assert t1_budget == t1_wall_time_s, (
        f"T0 overshoot must not steal from T1: expected {t1_wall_time_s}s, "
        f"got {t1_budget}"
    )


# ---------------------------------------------------------------------------
# Per-call timeout guard — the wall that `ModelSettings(timeout=…)` was
# supposed to be. Observed failure shape: a backend streamed 771 tokens
# in 324s without a gap big enough to trip httpx's read_timeout (which
# is what the OpenAI client's `timeout=` actually becomes for streaming
# responses), so the 180s per-call cap was never enforced. These tests
# pin the new per-turn `asyncio.wait_for` wrap that closes the gap.
# ---------------------------------------------------------------------------


def test_per_call_timeout_fires_on_slow_model_step():
    """When a single graph-node step (model call) takes longer than
    `per_call_timeout_s`, the helper raises `PerCallTimeoutExceeded`
    rather than letting the slow call run to completion. Reproduces
    the observed pathology where turn 1 streamed 324s of `<think>`
    tokens unchallenged."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import (
        PerCallTimeoutExceeded,
        iter_with_turn_logging,
    )

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    # Async-gen: first yields a ModelRequestNode (which sets
    # last_was_model_request=True), then the NEXT __anext__() takes
    # 0.5s before yielding — that's the "model call" step. With a
    # 0.05s per-call cap, the wait_for fires.
    async def slow_step_run():
        yield FakeModelReqNode()
        await asyncio.sleep(0.5)
        yield FakeModelReqNode()

    start = time.monotonic()
    with pytest.raises(PerCallTimeoutExceeded) as exc_info:
        asyncio.run(
            iter_with_turn_logging(
                slow_step_run(),
                phase="T0",
                pr_number="1856",
                turn_counter=[0],
                tool_call_counter={},
                log=lambda _msg: None,
                per_call_timeout_s=0.05,
            )
        )

    elapsed = time.monotonic() - start
    # Must terminate at the cap, not after the fake "model call"'s 0.5s.
    assert elapsed < 0.4, f"per-call trip took {elapsed:.2f}s, expected <0.4s"
    # Exception carries the trip diagnostics the wall_hit log line uses.
    assert exc_info.value.turn == 1, f"expected turn 1, got {exc_info.value.turn}"
    assert exc_info.value.cap_s == 0.05
    assert exc_info.value.elapsed_s >= 0.05


def test_first_call_extra_timeout_only_applies_to_first_call():
    """`first_call_extra_timeout_s` extends ONLY the first model call —
    the one that may hit a cold scale-to-zero card mid warm-image restore.
    Here the first call (0.2s) clears the 0.1+0.3=0.4 first-call cap, but
    the second identical call trips the base 0.1 cap, proving the allowance
    is one-shot (not a global per-call bump)."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import (
        PerCallTimeoutExceeded,
        iter_with_turn_logging,
    )

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def run():
        yield FakeModelReqNode()
        await asyncio.sleep(0.2)  # first model call — under the 0.4 first-call cap
        yield FakeModelReqNode()
        await asyncio.sleep(0.2)  # second model call — over the 0.1 base cap → trips
        yield FakeModelReqNode()

    with pytest.raises(PerCallTimeoutExceeded) as exc_info:
        asyncio.run(
            iter_with_turn_logging(
                run(),
                phase="T0",
                pr_number="cold",
                turn_counter=[0],
                tool_call_counter={},
                log=lambda _msg: None,
                per_call_timeout_s=0.1,
                first_call_extra_timeout_s=0.3,
            )
        )
    # Tripped on the SECOND call at the base cap — the first was covered.
    assert exc_info.value.turn == 2, f"expected turn 2, got {exc_info.value.turn}"
    assert exc_info.value.cap_s == 0.1


def test_per_call_timeout_none_disables_check():
    """`per_call_timeout_s=None` (the default) bypasses the wrap
    entirely — back-compat for callers that don't pass it (triage,
    existing tests, future direct uses)."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def short_run():
        for _ in range(3):
            yield FakeModelReqNode()
            await asyncio.sleep(0.02)

    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            short_run(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=lambda _msg: None,
            # per_call_timeout_s deliberately omitted
        )
    )
    assert turn[0] == 3


def test_per_call_timeout_does_not_trip_fast_steps():
    """A per-call cap comfortably larger than per-step latency should
    not trip — confirms we're not accidentally counting cumulative
    elapsed against the cap (per-step semantics, not total)."""
    from pydantic_ai._agent_graph import ModelRequestNode

    from cora.core.loop_logging import iter_with_turn_logging

    class FakeModelReqNode(ModelRequestNode):
        def __init__(self):
            pass

    async def fast_steps():
        for _ in range(5):
            yield FakeModelReqNode()
            await asyncio.sleep(0.02)

    turn = [0]
    asyncio.run(
        iter_with_turn_logging(
            fast_steps(),
            phase="T0",
            pr_number="1234",
            turn_counter=turn,
            tool_call_counter={},
            log=lambda _msg: None,
            per_call_timeout_s=1.0,  # 50× the per-step delay
        )
    )
    assert turn[0] == 5


def test_per_call_timeout_exception_message_shape():
    """The exception's `__str__` is what lands in the `::warning::`
    GHA annotation — keep the format readable for operators scanning
    workflow logs without diving into the log store."""
    from cora.core.loop_logging import PerCallTimeoutExceeded

    exc = PerCallTimeoutExceeded(turn=3, elapsed_s=324.0, cap_s=180.0)
    assert exc.turn == 3
    assert exc.elapsed_s == 324.0
    assert exc.cap_s == 180.0
    s = str(exc)
    assert "turn 3" in s
    assert "324" in s
    assert "180" in s
