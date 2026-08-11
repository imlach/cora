"""Structured per-turn logging for the Pydantic-AI Agent loop.

The reviewer (`deep_review.py`) and the T1 continuation
(`continuation.py`) both iterate `agent.iter()` node-by-node so the
loop is visible to instrumentation. This module owns the log-line
format so both consumers emit the same shape — a single point of
truth for the GHA log + Loki dashboard panels that key off these
lines.

One log line per event, logfmt-parseable:

```
agent_review iter pr_number=N phase=T0 event=<event> turn=N <kvps>…
```

The `agent_review iter` prefix + `event=…` field matches the
legacy `_progress` callback's emission contract, so existing
`{kind="loop"}` Grafana panels parse the new events with the same
logfmt pipeline. The caller wraps the same line into both
`::notice::` (GHA workflow UI) and `loki_push` (structured stream)
— one source of truth, two destinations.

Events emitted (`event=…` label):

- `turn_start` — model is about to be called (a `ModelRequestNode`
  traversal). Just `turn=N`.
- `tool_call` — model emitted a tool-call part. Adds `name=…
  args=…` (args truncated to `_ARGS_LOG_CHAR_CAP`, wrapped in
  double quotes for logfmt-safe whitespace).
- `model_response` — model returned a complete response. Adds
  `thinking_chars=…`, `text_chars=…`, `tool_calls=…`,
  `in_tokens=…`, `out_tokens=…`, `finish=…`, `elapsed_s=…`. The
  `thinking_chars > text_chars` + `finish=length` combination is
  the canonical reasoning-budget exhaustion signature for a
  reasoning model.
- `tool_error` — a tool call came back recoverable-bad: a failed
  result (an MCP server rejecting the arguments) or a framework
  retry prompt. Adds `turn=…`, `name=…`, `kind=…`
  (`failed_result` / `retry_prompt`), `errors=…` (running count for
  that tool) and a truncated `detail=…`. Not a terminal event —
  the model reads it and picks its next call.
- `tool_recovered` — first success on a tool after one or more
  `tool_error`s. Adds `turn=…`, `name=…`, `after_errors=…`. Read
  with `tool_error`: errors without recoveries are a tool the model
  never learned to call, which is a prompt or tool-description
  problem rather than a flaky server.
- `wall_hit` — `UsageLimitExceeded` (or future timeout/budget
  trip). Adds `terminated_reason=…`, `turns=…`, `tool_calls=…`, and
  `tool_errors=…` when the caller kept a counter.
- `spiral_redraw` — a turn hit the completion ceiling without
  committing and the identical payload was re-sent once. Adds
  `turn=…`, `outcome=…` (`recovered` / `spiralled_again` /
  `errored` / `skipped`). Emitted from the tier callers, which own
  the agent context the re-draw runs inside.
- `spiral_detected` — streaming only. Reasoning deltas passed
  `thinking_budget_tokens` with no text or tool-call delta, so the
  call was aborted mid-flight. Adds `turn=…`, `thinking_tokens=…`,
  `text_tokens=…`, `budget_tokens=…`.
- `stall_detected` — streaming only. No delta of any kind for
  `stall_timeout_s`. Adds `turn=…`, `idle_s=…`, `thinking_tokens=…`,
  `text_tokens=…`, `salvaged_chars=…`. The pair is the point:
  `spiral_detected` is a model generating hard, `stall_detected` is a
  wire that stopped. A whole-call timeout cannot tell them apart, and
  reported both as the same `per_call_timeout`.
- `continuation_start` — T1 picked up T0's wall-hit. Adds
  `prior_messages=…`, `prior_text_chars=…`, `prior_tool_calls=…`,
  `request_limit=…`. Emitted from `continuation.py` rather than
  inside the iteration helper.

Tool *result* sizes are not logged separately — derivable from the
next turn's `in_tokens` delta. Live tool dispatch sits between
graph nodes; extracting result bytes per-call would require an
out-of-band message-history walk.

Tool-call counters + the optional `on_tool_call` Budget hook are
bumped from inside this helper too (was previously a separate
`event_stream_handler`, but `agent.iter()` doesn't accept that
kwarg — only `agent.run()` does — so we centralise both functions
here).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

# `(line: str) -> None` — written by the caller's combined-emit
# wrapper that prints to GHA + pushes to Loki.
_LogFn = Callable[[str], None]


class WallTimeExceeded(Exception):
    """Raised by `iter_with_turn_logging` when the loop_deadline
    passed in by the caller has elapsed. Caught by `deep_review_call`
    / `continue_on_t1` and mapped to `terminated_reason="wall_time"` —
    same downstream finalize / T1-continuation path as
    `UsageLimitExceeded → terminated_reason="max_iterations"`.

    Carries the elapsed seconds past the deadline so the wall_hit
    log line can record overshoot for tuning."""

    def __init__(self, overshoot_s: float):
        super().__init__(f"wall-time deadline exceeded by {overshoot_s:.1f}s")
        self.overshoot_s = overshoot_s


class PerCallTimeoutExceeded(Exception):
    """Raised by `iter_with_turn_logging` when a single graph-node
    step (typically a model call) exceeds `per_call_timeout_s`.

    Distinct from `WallTimeExceeded` because the operational signal
    is different: wall-time means we ran out of overall budget;
    per-call means a single turn hung past its individual cap. The
    observed pathology — a backend streaming 771 tokens in 324s
    (~2.4 tok/s, vs typical ~50) — would surface as `per_call_timeout`
    here, distinguishable from "the whole review budget was used up
    by lots of normal-speed turns".

    Pydantic-AI passes `ModelSettings(timeout=N)` to the OpenAI client,
    which httpx treats as a read_timeout for streaming responses (gap
    between SSE chunks). vLLM streaming `<think>` tokens steadily —
    even at 2 tok/s — never trips that. The wrap here is a wall on
    total per-turn time, independent of how the bytes arrive.

    Caught by `deep_review_call` / `continue_on_t1` and mapped to
    `terminated_reason="per_call_timeout"`. Counted as a wall-hit
    reason in `amain` so T1 escalation still triggers (T1 runs on a
    different endpoint with more KV-cache headroom —
    and may complete where T0 stalled)."""

    def __init__(self, turn: int, elapsed_s: float, cap_s: float):
        super().__init__(
            f"turn {turn} exceeded per-call cap "
            f"({elapsed_s:.1f}s > {cap_s:.1f}s)"
        )
        self.turn = turn
        self.elapsed_s = elapsed_s
        self.cap_s = cap_s


class StreamStallDetected(Exception):
    """Raised when a streaming model call goes silent for longer than
    `stall_timeout_s` — no delta of any kind arrived.

    This is the distinction a whole-call timeout cannot draw. A model
    thinking at full speed and a dead wire look identical from outside
    the call: both produce nothing for minutes. Inside the stream they
    are opposites — one is a steady delta rate, the other is silence.
    Separating them is the entire reason for streaming here.

    Raised from inside `node.stream(...)`, which cancels the in-flight
    request task rather than committing a partial response, so the
    backend stops generating and the history stays clean for a
    re-draw."""

    def __init__(
        self,
        turn: int,
        *,
        idle_s: float,
        streamed_tokens: int,
        partial_text: str = "",
    ):
        super().__init__(
            f"turn {turn} produced no stream delta for {idle_s:.0f}s "
            f"after {streamed_tokens} streamed tokens"
        )
        self.turn = turn
        self.idle_s = idle_s
        self.streamed_tokens = streamed_tokens
        # Visible text streamed before the wire went quiet — threaded
        # into the re-draw so it resumes rather than restarting blind.
        self.partial_text = partial_text


class ReasoningSpiralDetected(Exception):
    """Raised by `iter_with_turn_logging` when a turn ends
    `finish_reason='length'` having produced no tool call and no
    verdict-shaped text — its whole completion budget went into
    reasoning (see `spiral.is_uncommitted_draw`).

    The opposite operational signal to `PerCallTimeoutExceeded`: the
    call SUCCEEDED and came back with data, it just came back with
    nothing the loop can use. Raised at the response boundary so the
    caller can re-draw while its agent context — and therefore its MCP
    sessions — is still open.

    Fires ahead of pydantic-ai's own `UnexpectedModelBehavior` for the
    thinking-only case (the `CallToolsNode` carrying the response is
    yielded before `CallToolsNode.run()` raises), and additionally
    covers the truncated-prose case, which produces a `TextPart` and so
    never raises at all."""

    def __init__(
        self,
        turn: int,
        *,
        thinking_chars: int,
        text_chars: int,
        out_tokens: int,
        partial_text: str = "",
    ):
        super().__init__(
            f"turn {turn} hit the completion ceiling without committing "
            f"({out_tokens} out tokens, {thinking_chars} thinking chars, "
            f"{text_chars} text chars, no tool call, no verdict)"
        )
        self.turn = turn
        self.thinking_chars = thinking_chars
        self.text_chars = text_chars
        self.out_tokens = out_tokens
        # Visible text the turn had emitted before it was aborted. Only
        # ever populated on the streaming path — a response that came
        # back whole is already in the history, so the caller reads it
        # from there. Threaded into the re-draw so it resumes rather
        # than restarting blind.
        self.partial_text = partial_text


# Tool-call args land in the log with this much detail; full args
# can be massive (grep_repo patterns + large globs) and would dwarf
# the actual signal. Same cap as the legacy `args_fingerprint`.
_ARGS_LOG_CHAR_CAP = 200


def _truncate_args(args: Any) -> str:
    """Render a tool-call args object (dict or string) into a single-
    line log-safe string, truncated to `_ARGS_LOG_CHAR_CAP` chars."""
    if args is None:
        return "<none>"
    try:
        text = str(args).replace("\n", " ").replace("\r", " ")
    except Exception:  # noqa: BLE001
        return "<unstringifiable>"
    if len(text) > _ARGS_LOG_CHAR_CAP:
        return text[:_ARGS_LOG_CHAR_CAP] + f"…[truncated at {_ARGS_LOG_CHAR_CAP}]"
    return text


def _delta_kind(event: Any) -> str:
    """Classify one stream event as `thinking` / `text` / `tool` / `other`.

    Duck-typed on the delta's `part_delta_kind` discriminator (and the
    part's `part_kind` for the start events), with a class-name
    fallback — same style as `spiral._part_kind`, so a fake delta
    stream in a unit test needs no pydantic-ai objects and a framework
    rename doesn't silently reclassify everything as `other`.
    """
    payload = getattr(event, "delta", None) or getattr(event, "part", None) or event
    kind = (
        getattr(payload, "part_delta_kind", None)
        or getattr(payload, "part_kind", None)
        or ""
    )
    name = str(kind) or type(payload).__name__
    lowered = name.lower()
    if "thinking" in lowered:
        return "thinking"
    if "tool" in lowered:
        return "tool"
    if "text" in lowered:
        return "text"
    return "other"


def _delta_text(event: Any) -> str:
    """The visible-text content of a stream event, or "".

    `TextPartDelta` carries `content_delta`; a `PartStartEvent` carries
    the seed part with `content`. Anything else contributes nothing."""
    payload = getattr(event, "delta", None) or getattr(event, "part", None) or event
    for attr in ("content_delta", "content"):
        value = getattr(payload, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _emit(
    *,
    pr_number: str,
    phase: str,
    event: str,
    log: _LogFn,
    **fields: Any,
) -> None:
    """Render one event as a logfmt line and write it once.

    Shape: `agent_review iter pr_number=N phase=T0 event=… <kvps>`.
    The caller's `log` callable is responsible for fanning the line
    to GHA (`::notice::…`) and Loki (`loki_push(line, labels=…)`)
    — this module never knows about either backend.
    """
    parts = [
        f"agent_review iter pr_number={pr_number}",
        f"phase={phase}",
        f"event={event}",
    ]
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    log(" ".join(parts))


def _log_tool_results(
    request: Any,
    *,
    phase: str,
    pr_number: str,
    turn: int,
    log: _LogFn,
    tool_errors: dict[str, int],
    open_errors: dict[str, int],
) -> None:
    """Emit `tool_error` / `tool_recovered` events for the tool results
    carried on one `ModelRequest`.

    Two part shapes are a recoverable tool error: a `ToolReturnPart`
    with `outcome='failed'` (an MCP server error routed back as a
    result — see `agent.MCP_TOOL_ERROR_BEHAVIOR`) and a
    `RetryPromptPart` (a framework retry: bad arguments, unknown tool
    name). Neither ends the run on its own.

    `tool_recovered` fires on the first success after one or more
    errors on the same tool. The pair is what tells a run that
    self-corrected apart from one that kept calling a tool it could
    never use — the second is a prompt or tool-description problem, and
    without the recovery half both look identical in the error count.
    """
    from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

    for part in getattr(request, "parts", None) or []:
        if isinstance(part, ToolReturnPart):
            name = getattr(part, "tool_name", None) or "<unknown>"
            if getattr(part, "outcome", "success") != "failed":
                opened = open_errors.pop(name, 0)
                if opened:
                    _emit(
                        pr_number=pr_number,
                        phase=phase,
                        event="tool_recovered",
                        log=log,
                        turn=turn,
                        name=name,
                        after_errors=opened,
                    )
                continue
            kind = "failed_result"
        elif isinstance(part, RetryPromptPart):
            name = getattr(part, "tool_name", None) or "<unknown>"
            kind = "retry_prompt"
        else:
            continue

        tool_errors[name] = tool_errors.get(name, 0) + 1
        open_errors[name] = open_errors.get(name, 0) + 1
        _emit(
            pr_number=pr_number,
            phase=phase,
            event="tool_error",
            log=log,
            turn=turn,
            name=name,
            kind=kind,
            errors=tool_errors[name],
            # Quoted + truncated for the same reason `tool_call` args
            # are: the server's message is free text and would
            # otherwise break the logfmt tokeniser.
            detail=f'"{_truncate_args(getattr(part, "content", None))}"',
        )


async def iter_with_turn_logging(
    agent_run,
    *,
    phase: str,
    pr_number: str,
    turn_counter: list[int],
    tool_call_counter: dict[str, int],
    log: _LogFn,
    on_tool_call: Callable[[str], None] | None = None,
    loop_deadline_monotonic: float | None = None,
    per_call_timeout_s: float | None = None,
    # One-time extra added to `per_call_timeout_s` for the FIRST model
    # call only — absorbs a scale-from-zero backend's cold start
    # without extending every later call. The clock starts at
    # dispatch but a scaled-to-zero backend isn't serving for up to its
    # readiness budget; later calls hit the warm backend and
    # use the base cap. 0.0 = no allowance (triage + warm-only callers + tests).
    first_call_extra_timeout_s: float = 0.0,
    # Push-based context injection. When set, the
    # refresher's `refresh(turn=...)` is awaited after each
    # `CallToolsNode` yield; a non-None return is wrapped in a
    # `UserPromptPart` and appended to the next `ModelRequestNode`'s
    # request. Defaults to None for back-compat with triage callers +
    # tests that don't construct one.
    context_refresher=None,
    # Caller-supplied callback that bumps the SHARED wall deadline
    # when an injection lands (the model needs extra room to act on
    # the new context without running headlong into the cap). Per-
    # injection extension is `INJECTION_DEADLINE_EXTENSION_S`; cap is
    # enforced by the refresher's `can_extend()` check below.
    extend_deadline_fn: Callable[[float], None] | None = None,
    # `(text) -> bool`: does this response body already carry a parseable
    # verdict? Supplying it ARMS uncommitted-draw detection — a turn that
    # hits the completion ceiling with no tool call and no verdict raises
    # `ReasoningSpiralDetected` instead of being carried forward. None
    # (the default) disarms it, which is what triage callers, tests, and
    # the re-draw's own second pass want.
    verdict_probe: Callable[[str], bool] | None = None,
    # Streaming detection (opt-in). Consumes each model call as a delta
    # stream instead of awaiting it whole, which is the only way to tell
    # a thinking model from a dead wire before the per-call cap fires.
    # Degrades to the non-streaming path when the run or node doesn't
    # expose the streaming surface, so an older pydantic-ai, a fake
    # agent_run in a test, or a triage caller all keep working.
    stream_detect: bool = False,
    # Inter-delta silence that counts as a stall. Sized well above a
    # slow token (a model at 2 tok/s still emits every 500ms) and well
    # below the per-call cap, so a stall is caught in seconds instead of
    # minutes.
    stall_timeout_s: float = 30.0,
    # Reasoning deltas a single turn may stream before it must have
    # committed to something. Breaching it with no text or tool-call
    # delta IS the spiral, caught while it's happening rather than
    # inferred from a corpse.
    thinking_budget_tokens: int = 10_000,
    # Cumulative recoverable-tool-error counter, keyed by tool name.
    # Supplied by the reviewer so the finish-line + wall-hit lines can
    # report it next to `tool_call_counter`; the per-occurrence
    # `tool_error` / `tool_recovered` events fire either way. None (the
    # default) keeps triage callers and tests unchanged.
    tool_error_counter: dict[str, int] | None = None,
):
    """Drive `agent_run` node-by-node and emit per-turn log lines.

    Pydantic-AI's graph yields:
      - `UserPromptNode` — initial user prompt setup
      - `ModelRequestNode` — about to call the model
      - `CallToolsNode` — model returned; response is on
        `node.model_response`, tool calls (if any) get dispatched
        next

    We mark a new turn on each `ModelRequestNode` and on each
    `CallToolsNode` walk `model_response.parts` to:
      - Log each `ToolCallPart` (with truncated args)
      - Bump `tool_call_counter[name]`
      - Invoke the optional `on_tool_call(name)` hook (Budget bump
        in the reviewer's deep path)
      - Emit the per-turn `model_response` summary line with the
        thinking/text/tool-calls/tokens breakdown

    Each `ModelRequestNode` carries the *results* of the turn before
    it, so that branch also walks `request.parts` to emit
    `tool_error` / `tool_recovered` (see `_log_tool_results`).

    Two independent timeouts guard the loop:

    - `loop_deadline_monotonic` — overall wall deadline. Checked at
      the top of each node iteration. Once `time.monotonic()` is past
      it, `WallTimeExceeded` is raised. Cheap (fires between nodes),
      coarse (won't interrupt a single hung node).
    - `per_call_timeout_s` — per-step cap. Each `__anext__()` on the
      agent_run iterator (which awaits the model call when the prior
      node was a `ModelRequestNode`) is wrapped in `asyncio.wait_for`.
      A single turn exceeding the cap raises `PerCallTimeoutExceeded`.
      This is the wall that `ModelSettings(timeout=N)` was supposed
      to be — but `ModelSettings.timeout` becomes httpx's read_timeout
      for streaming responses, only firing on inactivity. Observed:
      vLLM streamed 771 tokens in 324s and never went silent, so the
      ModelSettings cap never tripped.

    Caller still owns exception handling around the iteration
    (`UsageLimitExceeded`, `WallTimeExceeded`, `PerCallTimeoutExceeded`,
    `asyncio.TimeoutError` → break-marker log + return).
    """
    from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import (
        TextPart,
        ThinkingPart,
        ToolCallPart,
        UserPromptPart,
    )

    # Imported lazily here so the module-level import of loop_logging
    # (used by triage callers that don't construct a refresher) doesn't
    # pull pr_context's subprocess imports until actually needed.
    from cora.core.context_refresher import (
        INJECTION_DEADLINE_EXTENSION_S,
    )

    turn_start: dict[int, float] = {}
    # Cumulative errors per tool (caller-visible when supplied) and the
    # subset not yet followed by a success on that tool — the latter is
    # what `tool_recovered` reports and clears.
    tool_errors = tool_error_counter if tool_error_counter is not None else {}
    open_errors: dict[str, int] = {}
    # Manual iteration so we can wrap each `__anext__()` with the
    # per-call timeout. `agent_run.__aiter__()` returns the same
    # iterator `async for` would drive; `StopAsyncIteration` signals
    # natural completion.
    agent_iter = agent_run.__aiter__()
    # Track whether the most-recently-yielded node was a
    # ModelRequestNode — the *next* `__anext__()` is what awaits the
    # model call, so that's the step where the per-call timeout
    # actually matters. Wrapping every step (including the cheap
    # tool-dispatch awaits) is harmless but obscures the signal in
    # the log line.
    last_was_model_request = False
    # Counts model-call steps so the one-time cold-start allowance lands
    # on the first call only (the one that may hit a cold scale-to-zero
    # backend).
    model_call_count = 0
    # Local mutable deadline mirror. Starts at the kwarg's value;
    # when `extend_deadline_fn` is invoked on an injection landing,
    # we bump this too so the in-loop check picks up the new ceiling
    # on the very next iteration. The kwarg parameter is a plain
    # float (back-compat for existing tests + triage callers) so the
    # bump can't reach into the caller's `loop_deadline_monotonic`
    # variable — see the caller-side wiring in deep_review_call /
    # continue_on_t1 where the same closure also bumps the outer
    # `asyncio.wait_for` budget.
    _current_deadline: float | None = loop_deadline_monotonic
    # When the refresher returns a body on a CallToolsNode boundary,
    # we stash it here and graft it onto the very next ModelRequestNode
    # (which is the natural carrier of the tool-returns + next user
    # turn). The pending state lives across `__anext__` calls.
    pending_injection: str | None = None

    def _can_stream(node: Any) -> bool:
        """Streaming needs both halves of the surface: a node that can
        open a stream and a run that can hand it the graph context.
        Missing either — an older pydantic-ai, a stand-in `agent_run` in
        a test, a triage caller — falls back to awaiting the whole call,
        which is the pre-feature behaviour."""
        return (
            stream_detect
            and callable(getattr(node, "stream", None))
            and getattr(agent_run, "ctx", None) is not None
        )

    async def _consume_stream(node: Any, *, turn: int) -> None:
        """Drive one model call as a delta stream, aborting on a stall
        or a runaway reasoning budget.

        Raising out of this block is what aborts. Pydantic-ai's
        `ModelRequestNode.stream` cancels the in-flight request task on
        an exception and does NOT append a partial response, so the
        backend stops generating and the history stays exactly at the
        payload we would re-send. Exiting normally lets the framework
        finalise the response as usual, so the `CallToolsNode` that
        follows is byte-identical to the non-streaming path.

        Token counts are streamed-delta counts, not tokeniser output:
        the backend emits one delta per token, so the two agree closely
        enough for a budget and neither the engine nor the log claims
        more precision than that."""
        thinking_deltas = 0
        text_deltas = 0
        text_parts: list[str] = []
        committed = False
        async with node.stream(agent_run.ctx) as stream:
            events = stream.__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(
                        events.__anext__(), timeout=stall_timeout_s
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError as timeout_exc:
                    partial = "".join(text_parts)
                    _emit(
                        pr_number=pr_number,
                        phase=phase,
                        event="stall_detected",
                        log=log,
                        turn=turn,
                        idle_s=f"{stall_timeout_s:.0f}",
                        thinking_tokens=thinking_deltas,
                        text_tokens=text_deltas,
                        salvaged_chars=len(partial),
                    )
                    # Chain the TimeoutError: the stall IS that timeout,
                    # and keeping the cause makes the traceback say where
                    # the wait expired rather than starting at the domain
                    # exception.
                    raise StreamStallDetected(
                        turn,
                        idle_s=stall_timeout_s,
                        streamed_tokens=thinking_deltas + text_deltas,
                        partial_text=partial,
                    ) from timeout_exc
                kind = _delta_kind(event)
                if kind == "thinking":
                    thinking_deltas += 1
                elif kind == "tool":
                    # A tool call is a commitment; the turn has somewhere
                    # to go and the reasoning budget stops applying.
                    committed = True
                elif kind == "text":
                    committed = True
                    text_deltas += 1
                    text_parts.append(_delta_text(event))
                if (
                    not committed
                    and thinking_budget_tokens > 0
                    and thinking_deltas > thinking_budget_tokens
                ):
                    _emit(
                        pr_number=pr_number,
                        phase=phase,
                        event="spiral_detected",
                        log=log,
                        turn=turn,
                        thinking_tokens=thinking_deltas,
                        text_tokens=0,
                        budget_tokens=thinking_budget_tokens,
                    )
                    raise ReasoningSpiralDetected(
                        turn,
                        thinking_chars=0,
                        text_chars=0,
                        out_tokens=thinking_deltas,
                    )

    while True:
        if _current_deadline is not None:
            now = time.monotonic()
            if now >= _current_deadline:
                raise WallTimeExceeded(overshoot_s=now - _current_deadline)
        try:
            if per_call_timeout_s is not None and last_was_model_request:
                # The model-call step. Record start so we can attribute
                # the timeout to the right turn number on a trip.
                model_call_count += 1
                # First model call gets the cold-start allowance on top of
                # the base cap; every later call uses the base cap.
                step_cap = per_call_timeout_s
                if model_call_count == 1 and first_call_extra_timeout_s:
                    step_cap += first_call_extra_timeout_s
                turn_at_call = turn_counter[0]
                step_start = time.monotonic()
                try:
                    node = await asyncio.wait_for(
                        agent_iter.__anext__(),
                        timeout=step_cap,
                    )
                except TimeoutError as exc:
                    raise PerCallTimeoutExceeded(
                        turn=turn_at_call,
                        elapsed_s=time.monotonic() - step_start,
                        cap_s=step_cap,
                    ) from exc
            else:
                node = await agent_iter.__anext__()
        except StopAsyncIteration:
            break
        last_was_model_request = isinstance(node, ModelRequestNode)
        if isinstance(node, ModelRequestNode):
            # This request carries the tool results for the turn that
            # just ended, so it reports under that turn number — the
            # counter is bumped for the new turn further down.
            _log_tool_results(
                getattr(node, "request", None),
                phase=phase,
                pr_number=pr_number,
                turn=turn_counter[0],
                log=log,
                tool_errors=tool_errors,
                open_errors=open_errors,
            )
            # If a refresh on the prior CallToolsNode produced an
            # injection body, splice it onto this ModelRequest's
            # parts NOW — before the framework's `ModelRequestNode.run()`
            # executes and appends the request to history. The
            # request is a public dataclass (`ModelRequest`) with a
            # public `parts: list[ModelRequestPart]` attribute;
            # appending a `UserPromptPart` is exactly how a
            # user-prompt would have arrived on a fresh
            # `agent.run(prompt, message_history=…)` call.
            if pending_injection is not None:
                request = getattr(node, "request", None)
                parts_field = getattr(request, "parts", None)
                if isinstance(parts_field, list):
                    parts_field.append(UserPromptPart(content=pending_injection))
                    source = (
                        context_refresher.last_source
                        if context_refresher is not None
                        else "unknown"
                    ) or "unknown"
                    extended_s: float | None = None
                    if (
                        extend_deadline_fn is not None
                        and context_refresher is not None
                        and context_refresher.can_extend()
                    ):
                        try:
                            extend_deadline_fn(INJECTION_DEADLINE_EXTENSION_S)
                            # Also bump the in-loop mirror so the
                            # WallTimeExceeded check on the next
                            # iteration uses the new ceiling.
                            if _current_deadline is not None:
                                _current_deadline = (
                                    _current_deadline + INJECTION_DEADLINE_EXTENSION_S
                                )
                            context_refresher.record_extension()
                            extended_s = INJECTION_DEADLINE_EXTENSION_S
                        except Exception:  # noqa: BLE001
                            pass
                    _emit(
                        pr_number=pr_number,
                        phase=phase,
                        event="context_injected",
                        log=log,
                        turn=turn_counter[0] + 1,  # the turn we're about to start
                        source=source,
                        chars=len(pending_injection),
                        extended_s=(
                            f"{extended_s:.0f}"
                            if extended_s is not None
                            else None
                        ),
                    )
                # Clear whether or not we landed it — a missing
                # parts list (e.g. node was an End we mis-classified)
                # means the injection has nowhere to go and we drop
                # it rather than holding it indefinitely.
                pending_injection = None
            turn_counter[0] += 1
            turn_start[turn_counter[0]] = time.monotonic()
            _emit(
                pr_number=pr_number,
                phase=phase,
                event="turn_start",
                log=log,
                turn=turn_counter[0],
            )
            if _can_stream(node):
                # Consume the model call HERE rather than letting the
                # next `__anext__()` await it whole. The per-call cap
                # moves with it: it's the same wall on the same work,
                # and `last_was_model_request` stays False so the now-
                # trivial next step isn't wrapped a second time.
                model_call_count += 1
                step_cap = per_call_timeout_s
                if (
                    step_cap is not None
                    and model_call_count == 1
                    and first_call_extra_timeout_s
                ):
                    step_cap += first_call_extra_timeout_s
                stream_start = time.monotonic()
                try:
                    coro = _consume_stream(node, turn=turn_counter[0])
                    if step_cap is not None:
                        await asyncio.wait_for(coro, timeout=step_cap)
                    else:
                        await coro
                except TimeoutError as exc:
                    raise PerCallTimeoutExceeded(
                        turn=turn_counter[0],
                        elapsed_s=time.monotonic() - stream_start,
                        cap_s=step_cap or 0.0,
                    ) from exc
                last_was_model_request = False
        elif isinstance(node, CallToolsNode):
            resp = getattr(node, "model_response", None)
            if resp is None:
                continue
            parts = list(getattr(resp, "parts", []) or [])

            # Per-tool-call lines + counter bumps. Walk the parts in
            # order so the log reflects the model's emission order
            # (helpful when comparing against the eventual review
            # body for "did the model use the right tool first").
            for part in parts:
                if isinstance(part, ToolCallPart):
                    name = getattr(part, "tool_name", None) or "<unknown>"
                    args = getattr(part, "args", None)
                    args_trunc = _truncate_args(args)
                    tool_call_counter[name] = tool_call_counter.get(name, 0) + 1
                    _emit(
                        pr_number=pr_number,
                        phase=phase,
                        event="tool_call",
                        log=log,
                        turn=turn_counter[0],
                        name=name,
                        # Wrap args in quotes for logfmt safety —
                        # they often contain spaces / regex
                        # metacharacters that would otherwise break
                        # the kvp tokeniser.
                        args=f'"{args_trunc}"',
                    )
                    if on_tool_call is not None:
                        try:
                            on_tool_call(name)
                        except Exception:  # noqa: BLE001
                            pass

            # Per-turn summary.
            thinking_chars = sum(
                len(getattr(p, "content", "") or "")
                for p in parts
                if isinstance(p, ThinkingPart)
            )
            text = "".join(
                str(getattr(p, "content", "") or "")
                for p in parts
                if isinstance(p, TextPart)
            )
            text_chars = len(text)
            tool_calls = sum(1 for p in parts if isinstance(p, ToolCallPart))
            usage = getattr(resp, "usage", None)
            in_t = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
            out_t = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            finish = getattr(resp, "finish_reason", None) or "?"
            elapsed_f = (
                time.monotonic() - turn_start.get(turn_counter[0], 0)
                if turn_counter[0] in turn_start
                else 0.0
            )
            _emit(
                pr_number=pr_number,
                phase=phase,
                event="model_response",
                log=log,
                turn=turn_counter[0],
                thinking_chars=thinking_chars,
                text_chars=text_chars,
                tool_calls=tool_calls,
                in_tokens=in_t,
                out_tokens=out_t,
                finish=finish,
                elapsed_s=f"{elapsed_f:.1f}",
            )

            # Uncommitted draw — the turn burned its completion ceiling
            # and produced nothing the loop can carry forward. Raise
            # before the refresher poll below: there is no point
            # gathering fresh context for a turn we're about to discard.
            if verdict_probe is not None:
                from cora.core import spiral as _spiral

                if _spiral.is_uncommitted_draw(
                    finish_reason=finish,
                    tool_calls=tool_calls,
                    text=text,
                    has_verdict=verdict_probe,
                ):
                    raise ReasoningSpiralDetected(
                        turn_counter[0],
                        thinking_chars=thinking_chars,
                        text_chars=text_chars,
                        out_tokens=out_t,
                    )

            # Push-based context injection — runs after each model
            # response (CallToolsNode boundary, same as the wall-time
            # check). Refresher returns a wrapped body or None per
            # its dedupe + cadence rules. We don't splice into the
            # current node — we stash the body in `pending_injection`
            # and graft it onto the NEXT ModelRequestNode's
            # `request.parts` when that node arrives on the next
            # `__anext__()` (see the ModelRequestNode branch above).
            #
            # Pydantic-AI's CallToolsNode returns a `ModelRequestNode`
            # (with tool-return + optional user-prompt parts already
            # built) as the result of its `.run()`. Our `__anext__`
            # receives that ModelRequestNode BEFORE its own `.run()`
            # executes — `node.request.parts` is still mutable at
            # that point, and `ModelRequestNode.run()` appends the
            # full request (including our extra part) to
            # `ctx.state.message_history` and ships it to the model.
            # This stays on the documented `ModelRequest.parts` API
            # surface, no private-attribute reach-around.
            if context_refresher is not None:
                injection = await context_refresher.refresh(turn=turn_counter[0])
                if injection is not None:
                    pending_injection = injection


def log_wall_hit(
    *,
    phase: str,
    pr_number: str,
    terminated_reason: str,
    turn_counter: list[int],
    tool_call_counter: dict[str, int],
    log: _LogFn,
    tool_error_counter: dict[str, int] | None = None,
) -> None:
    """Structured break-marker log on `UsageLimitExceeded` and friends.
    The caller fires a `::warning::` line separately when it wants the
    GHA UI to surface this as an annotation (the GHA UI elevates
    `::warning::` to top-of-summary; `::notice::` stays inline).

    `tool_error_counter` is the same dict `iter_with_turn_logging`
    filled; omitting it drops the `tool_errors` field rather than
    reporting a zero it can't vouch for."""
    total_tools = sum(tool_call_counter.values())
    total_errors = (
        sum(tool_error_counter.values()) if tool_error_counter is not None else None
    )
    _emit(
        pr_number=pr_number,
        phase=phase,
        event="wall_hit",
        log=log,
        terminated_reason=terminated_reason,
        turns=turn_counter[0],
        tool_calls=total_tools,
        tool_errors=total_errors,
    )


def log_spiral_redraw(
    *,
    phase: str,
    pr_number: str,
    turn: int,
    outcome: str,
    log: _LogFn,
    **fields: Any,
) -> None:
    """Marker for one uncommitted-draw re-draw attempt.

    `outcome` is the disposition, one of:
      - `recovered`      — the re-draw produced a usable body
      - `spiralled_again`— the re-draw hit the ceiling the same way
      - `errored`        — the re-draw raised (MCP blip, wall-time, …)
      - `skipped`        — detection fired but no re-draw ran (disabled,
                           or no committed prefix to re-send)

    A rising `spiralled_again` share means the ceiling is genuinely too
    low for the workload rather than the draw being unlucky — that is
    the signal to raise `AGENT_REVIEW_MAX_COMPLETION_TOKENS` (and
    `AGENT_REVIEW_PER_CALL_TIMEOUT_S` with it), not to add re-draws."""
    _emit(
        pr_number=pr_number,
        phase=phase,
        event="spiral_redraw",
        log=log,
        turn=turn,
        outcome=outcome,
        **fields,
    )


def log_continuation_start(
    *,
    pr_number: str,
    t1_model_alias: str,
    prior_messages: int,
    prior_text_chars: int,
    prior_tool_calls: int,
    request_limit: int,
    log: _LogFn,
) -> None:
    """Marker for the T0→T1 handoff. Distinct event so a dashboard
    can chart escalation rate without grepping for the in-loop turn
    boundaries.
    """
    fields: dict = {
        "t1_model": t1_model_alias,
        "prior_messages": prior_messages,
        "prior_text_chars": prior_text_chars,
        "prior_tool_calls": prior_tool_calls,
        "request_limit": request_limit,
    }
    _emit(
        pr_number=pr_number,
        phase="T1",
        event="continuation_start",
        log=log,
        **fields,
    )


def log_tier_verdict(
    *,
    pr_number: str,
    tier: str,
    verdict: str | None,
    has_blocker: bool,
    body_chars: int,
    log: _LogFn,
) -> None:
    """Per-tier final-verdict marker — one event per tier that
    actually ran.

    Escalation foundation: when a T2 (alternate-family second
    opinion) is enabled, the
    disagreement-resolver reads these events to compare verdicts
    across tiers. Until a T2 is deployed, only T0 (always) and T1 (on
    wall-hit continuation) emit; a "tier verdicts" telemetry view
    renders one row per tier and the disagreement rate
    stays empty because there's nothing to disagree with.

    `has_blocker` distinguishes the categorical-failure case (T2
    finds a concrete blocker per the `🚨 **Blocker:**` marker) from
    a soft `needs changes` verdict on subjective concerns. A future
    escalation policy can use this to gate a higher tier on the
    `gap=2 AND has_blocker=true` combination.

    `verdict` is the lowercase verdict word
    (`looks good` / `minor` / `needs changes`) or None if the tier
    ran but produced no parseable verdict (rare today — leak retry
    catches most of these; future-proofing for a tier that genuinely
    returned no text).
    """
    _emit(
        pr_number=pr_number,
        phase=tier,
        event="tier_verdict",
        log=log,
        tier=tier,
        verdict=(verdict or "none").replace(" ", "_"),
        has_blocker=str(has_blocker).lower(),
        body_chars=body_chars,
    )
