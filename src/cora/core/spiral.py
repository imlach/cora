"""Reasoning-spiral detection + recovery for the `review` reasoning model.

The `review` alias serves a reasoning model. A turn
occasionally spends its ENTIRE per-call output budget inside `<think>` and
emits no text/tool-call — `finish_reason='length'` with thinking-only
parts. Pydantic-AI raises `UnexpectedModelBehavior` ("Model token limit
(N) exceeded before any response was generated", `_agent_graph.py` ~L1104)
at that point. Quick mode soft-fails; deep mode's loop errors out
(`agent-loop-errored`). The spiral is high-variance — the SAME PR has
reasoned 58,740 chars one run and 8,082 the next — so bumping the per-call
cap is a treadmill.

Recovery posture ("B with limits"): the model's partial reasoning IS
available at failure time via pydantic-ai's `capture_run_messages()` /
`agent_run.all_messages()`. Instead of failing, re-issue ONE bounded call
seeded with that reasoning tail + a "produce the final review now"
directive — so the model commits using work it already did. The recovery
turn KEEPS reasoning ENABLED (no `enable_thinking=False`) but is bounded:
a tight output cap (`SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS`) + a concise
"keep further reasoning brief" lead-in. One attempt only; on a second
spiral/failure the caller falls back to today's soft-fail (no regression).

This is the same shape as `continuation.continue_on_t1` (re-run with
`message_history=<captured>` + a lead-in prompt), just a different trigger
(reasoning spiral vs T0 wall-hit), same model/tier, with a tight budget.

Two rungs, cheapest first. `is_uncommitted_draw` + `committed_prefix`
drive the in-loop **re-draw**: re-send the identical payload once, no
prompt perturbation, because the same draw usually succeeds on a second
roll. Only when that also fails does the bounded conclude-now recovery
above (`build_recovery_leadin`, opt-in) spend a differently-shaped call.
The re-draw needs no exception at all — it fires on the response
boundary, which also catches the truncated-prose case that never makes
pydantic-ai raise.

Duck-typed on each part's `part_kind` discriminator (the stable literal
pydantic-ai stamps on every message part) with a class-name fallback —
mirrors `loop_logging.py` / `transcript.py`'s getattr-on-parts style — so
the module needs **no** `pydantic_ai` import and stays cheap to import +
unit-testable with plain stand-in objects. The optional exception
classifier (`is_token_limit_exception`) does a lazy local import so the
module-level import path stays dependency-free; detection prefers the
captured messages, which is robust to pydantic-ai version drift.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def _part_kind(part: Any) -> str:
    """The part's `part_kind` literal, with a class-name fallback for
    stand-ins / future renames. Same shape as `transcript._part_kind`."""
    kind = getattr(part, "part_kind", None)
    if kind:
        return str(kind)
    name = type(part).__name__
    mapping = {
        "TextPart": "text",
        "ThinkingPart": "thinking",
        "ToolCallPart": "tool-call",
    }
    return mapping.get(name, name)


def _last_model_response(messages: Iterable[Any]) -> Any | None:
    """Return the last `ModelResponse` in a captured-message list, or None.

    A `ModelResponse` is duck-typed as a message whose `parts` are
    response-side kinds (text / thinking / tool-call) rather than the
    request-side kinds (system-prompt / user-prompt / tool-return). We
    identify it by `kind == "response"` (pydantic-ai stamps `kind` on the
    message) with a class-name fallback for stand-ins."""
    last: Any | None = None
    for msg in messages or []:
        kind = getattr(msg, "kind", None)
        if kind == "response" or type(msg).__name__ == "ModelResponse":
            last = msg
    return last


def has_model_response(messages: Iterable[Any]) -> bool:
    """True iff the captured history contains at least one
    `ModelResponse` — i.e. the model answered at least once.

    This is the emptiness test that means "did this tier get anywhere".
    `not messages` is NOT: pydantic-ai appends the outgoing
    `ModelRequest` to the history *before* awaiting the model
    (`_agent_graph.ModelRequestNode.run`), so even a first-call timeout
    leaves a one-element history behind and any `not messages` guard is
    dead code that silently never fires."""
    return _last_model_response(messages) is not None


def is_reasoning_spiral(messages: Iterable[Any]) -> bool:
    """True iff the last `ModelResponse` is thinking-only.

    "Thinking-only" = ≥1 `ThinkingPart` and NO `TextPart` and NO
    `ToolCallPart` — the canonical reasoning-budget-exhaustion signature
    that makes pydantic-ai raise before any usable response was generated.

    Robust to version drift: detection is on the captured messages, not on
    the exception type/string. Returns False for an empty/None history or
    a final response that produced any text or tool call.
    """
    resp = _last_model_response(messages)
    if resp is None:
        return False
    parts = list(getattr(resp, "parts", []) or [])
    has_thinking = any(_part_kind(p) == "thinking" for p in parts)
    has_text = any(_part_kind(p) == "text" for p in parts)
    has_tool_call = any(_part_kind(p) == "tool-call" for p in parts)
    return has_thinking and not has_text and not has_tool_call


def is_uncommitted_draw(
    *,
    finish_reason: Any,
    tool_calls: int,
    text: str,
    has_verdict: Callable[[str], bool] | None = None,
) -> bool:
    """True iff a turn spent its whole completion budget without
    committing to anything the loop can use.

    The signature is `finish_reason='length'` AND no tool call AND no
    verdict-shaped text: the model was cut off by the token ceiling
    mid-reasoning, so continuing the loop would carry a dead turn
    forward. Deliberately independent of *how* the budget was spent —
    a thinking-only response and a truncated prose ramble are the same
    problem, and only the first of those makes pydantic-ai raise.

    Text that already parses as a verdict is a usable answer whose tail
    got clipped, not a spiral — the loop keeps it. With no `has_verdict`
    probe we cannot tell, so any text at all is treated as usable (the
    conservative direction: never re-draw over a real answer).
    """
    if str(finish_reason or "").lower() != "length":
        return False
    if tool_calls:
        return False
    if not text.strip():
        return True
    if has_verdict is None:
        return False
    return not has_verdict(text)


def committed_prefix(messages: Iterable[Any]) -> list[Any]:
    """The captured history with a trailing `ModelResponse` dropped —
    i.e. exactly the payload that was sent to produce it.

    Re-issuing this prefix re-rolls the same draw with no prompt
    perturbation, which is the point: an identical re-send of a payload
    that spiralled typically completes normally, so the cheapest
    recovery is to ask again rather than to ask differently. A history
    that doesn't end in a response is returned unchanged.
    """
    msgs = list(messages or [])
    if msgs and msgs[-1] is _last_model_response(msgs):
        return msgs[:-1]
    return msgs


def tool_call_names(messages: Iterable[Any], *, start: int = 0) -> list[str]:
    """Tool names called in `messages[start:]`, in emission order.

    Lets a caller that recovered via a plain `agent.run` (which drives
    its own loop, bypassing the per-turn instrumentation) fold the
    dispatches it made back into the shared tool-call accounting."""
    names: list[str] = []
    for msg in list(messages or [])[start:]:
        for part in getattr(msg, "parts", []) or []:
            if _part_kind(part) == "tool-call":
                names.append(str(getattr(part, "tool_name", "") or "<unknown>"))
    return names


def extract_partial_reasoning(messages: Iterable[Any], *, char_cap: int) -> str:
    """Concatenate the `ThinkingPart` content of the last `ModelResponse`,
    returning the TAIL truncated to `char_cap`.

    The tail is where conclusions form (the head is exploratory), so we
    keep the last `char_cap` chars rather than the first. Returns "" when
    there's no final response or it carried no thinking content.
    """
    resp = _last_model_response(messages)
    if resp is None:
        return ""
    reasoning = "".join(
        str(getattr(p, "content", "") or "")
        for p in (getattr(resp, "parts", []) or [])
        if _part_kind(p) == "thinking"
    )
    if char_cap >= 0 and len(reasoning) > char_cap:
        return reasoning[-char_cap:]
    return reasoning


def extract_final_text(messages: Iterable[Any]) -> str:
    """Concatenate the `TextPart` content of the last `ModelResponse`.

    The salvage path: a turn cut off by the completion ceiling mid-prose
    still said something, and a truncated review beats no review. Returns
    "" for a thinking-only response, which has nothing to salvage."""
    resp = _last_model_response(messages)
    if resp is None:
        return ""
    return "".join(
        str(getattr(p, "content", "") or "")
        for p in (getattr(resp, "parts", []) or [])
        if _part_kind(p) == "text"
    )


def is_completion_ceiling_exception(exc: BaseException) -> bool:
    """True for pydantic-ai's two completion-ceiling raises: "token limit
    … exceeded before any response was generated" (thinking-only) and
    "… exceeded while generating a tool call" (truncated tool args).

    Matched on the shared "token limit" phrasing rather than the class,
    because both are `UnexpectedModelBehavior` and only the message
    separates them from unrelated model-behaviour failures. Narrower on
    purpose than `is_token_limit_exception` below, which is a coarse
    type-first classifier for the soft-fail path — this one decides
    whether to spend another call, so a false positive costs money."""
    return "token limit" in str(exc).lower()


# Concise directive prepended to the recovered reasoning tail. Frames the
# re-run as "you already did the analysis — commit now, briefly" so the
# bounded recovery turn produces the verdict body instead of spiralling
# again. Keeps the required verdict-line / output-format guidance intact
# (the downstream leak/verdict parser in `leak.py` is unchanged).
_RECOVERY_DIRECTIVE = (
    "You have already analyzed this PR — your reasoning so far is below. "
    "Produce your final review now in the required output format "
    "(the standard `Verdict:` line and review body). You have already "
    "done the analysis; keep any further reasoning brief."
)


def build_recovery_leadin(reasoning_tail: str) -> str:
    """The recovery prompt: the conclude-now directive followed by the
    prior reasoning tail, framed so the model treats the reasoning as its
    own working state. Paired with `message_history=<captured>` on the
    re-run (same shape as `continuation._CONTINUATION_PROMPT`)."""
    return f"{_RECOVERY_DIRECTIVE}\n\n<prior reasoning>\n{reasoning_tail}"


def is_token_limit_exception(exc: BaseException) -> bool:
    """Best-effort classifier for pydantic-ai's "token limit exceeded
    before any response" raise. Prefer `is_reasoning_spiral` on the
    captured messages — this is a coarse secondary signal only, and is
    lazy-imported so the module stays dependency-free at import time.

    Matches by type (`UnexpectedModelBehavior`) AND/OR the canonical
    message substring, since the message text is what carries the
    distinct signal and survives across class hierarchies.
    """
    try:
        from pydantic_ai.exceptions import UnexpectedModelBehavior
    except Exception:  # noqa: BLE001 — keep cheap/dependency-free
        UnexpectedModelBehavior = ()  # type: ignore[assignment]
    if isinstance(exc, UnexpectedModelBehavior):
        return True
    text = str(exc).lower()
    return "token limit" in text and "before any response" in text
