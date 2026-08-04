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
- `wall_hit` — `UsageLimitExceeded` (or future timeout/budget
  trip). Adds `terminated_reason=…`, `turns=…`, `tool_calls=…`.
- `spiral_redraw` — a turn hit the completion ceiling without
  committing and the identical payload was re-sent once. Adds
  `turn=…`, `outcome=…` (`recovered` / `spiralled_again` /
  `errored` / `skipped`). Emitted from the tier callers, which own
  the agent context the re-draw runs inside.
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
from typing import Any, Callable


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
    from pydantic_ai._agent_graph import ModelRequestNode, CallToolsNode
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
                except asyncio.TimeoutError as exc:
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
) -> None:
    """Structured break-marker log on `UsageLimitExceeded` and friends.
    The caller fires a `::warning::` line separately when it wants the
    GHA UI to surface this as an annotation (the GHA UI elevates
    `::warning::` to top-of-summary; `::notice::` stays inline)."""
    total_tools = sum(tool_call_counter.values())
    _emit(
        pr_number=pr_number,
        phase=phase,
        event="wall_hit",
        log=log,
        terminated_reason=terminated_reason,
        turns=turn_counter[0],
        tool_calls=total_tools,
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
