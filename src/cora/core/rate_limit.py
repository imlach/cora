"""LiteLLM 429 (rate-limit) backoff helper.

LiteLLM enforces `max_parallel_requests` on the reviewer backends.
When concurrent reviews
exceed the cap, the surplus call gets a 429 with `Retry-After` set
from the deployment's `cooldown_time` (30s in the reference
deployment). The intent is for
the reviewer job to **back off in-runner** — the PR check stays in
"queued / running" while we wait for a backend slot — instead of
catching the exception and posting a skipped review.

`run_with_rate_limit_backoff` wraps a single LLM-calling coroutine and
retries it on 429, sleeping between attempts. Total waiting time is
capped (`max_total_wait_s`) so a permanently saturated backend doesn't
hold the GHA runner forever; the workflow `timeout-minutes` is the
ultimate ceiling above that.

Non-429 errors raise through unchanged so existing handlers
(`agent-run failed: ...` skip path, `UsageLimitExceeded` wall-hit
escalation) fire normally.

Service-ification follow-up: this whole back-off-in-runner pattern is
a workaround for not having a queue. A reviewer-as-service design
with a real queue is the durable
fix — a queue lets the service hold concurrency at the right level
natively, freeing GHA runners as soon as the review is accepted.
"""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

# Total wait budget across all retries on a single call. Sized so a
# queue ~3 reviews deep (each holding the backend for ~2 min) can drain
# before we give up. The workflow `timeout-minutes` must cover this
# plus the actual run; see the consumer workflow.
DEFAULT_MAX_TOTAL_WAIT_S = 600

# Per-attempt floor. Mirrors LiteLLM's `cooldown_time: 30` in the
# reference deployment — a shorter sleep would tight-loop against a backend
# that's still cooling down. Also the default when no `Retry-After`
# header is parseable from the exception surface.
MIN_BACKOFF_S = 30

# Hard ceiling on a single sleep so a misconfigured `Retry-After` (or
# a future change to LiteLLM's cooldown) can't pin the runner for an
# absurd duration.
MAX_BACKOFF_S = 180

# Pattern fallback — pydantic-ai sometimes surfaces upstream HTTP
# errors as a string-wrapped UnexpectedModelBehavior rather than a
# typed ModelHTTPError. Match either "status_code=429" or a bare
# " 429 " in the message; conservative on purpose so we don't retry
# random errors that happen to contain "429" in unrelated text.
_RATE_LIMIT_MESSAGE_RE = re.compile(r"(?:status[_ ]code[ =:]*429|\b429\b\s+(?:Too Many|Rate))", re.IGNORECASE)


def _is_rate_limit(exc: BaseException) -> bool:
    """Best-effort 429 detection across the exception shapes pydantic-ai /
    openai client / httpx might surface."""
    # Typed shape (pydantic-ai ModelHTTPError, openai APIStatusError).
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and code == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int) and code == 429:
            return True
    # String fallback — UnexpectedModelBehavior wraps the upstream
    # response text and loses the typed status_code.
    return bool(_RATE_LIMIT_MESSAGE_RE.search(str(exc)))


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Parse `Retry-After` from response headers when the client surfaces
    them. Returns None if absent — caller falls back to MIN_BACKOFF_S.

    Pydantic-AI's ModelHTTPError stores the body but not headers as a
    documented attribute; this only fires for the openai / httpx
    exception shape that does carry headers. Best-effort throughout."""
    headers = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:  # noqa: BLE001
        return None
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def run_with_rate_limit_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_total_wait_s: int = DEFAULT_MAX_TOTAL_WAIT_S,
    log: Callable[[str], None] = print,
    description: str = "LLM call",
) -> T:
    """Run `fn()` with rate-limit (429) backoff.

    On 429: parse `Retry-After` if present, sleep (clamped to
    `[MIN_BACKOFF_S, MAX_BACKOFF_S]`), retry. Total time spent waiting
    across all attempts is capped at `max_total_wait_s` — when the
    next sleep would exceed the cap, the last 429 raises through and
    the caller's existing handler decides what to do (skip review,
    fall through to T1, etc.).

    Non-429 exceptions propagate immediately on first occurrence.
    """
    waited = 0.0
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit(exc):
                raise
            attempt += 1
            retry_after = _retry_after_seconds(exc) or MIN_BACKOFF_S
            sleep_s = min(max(retry_after, MIN_BACKOFF_S), MAX_BACKOFF_S)
            if waited + sleep_s > max_total_wait_s:
                log(
                    f"::warning::{description}: 429 backoff exhausted after "
                    f"{attempt} attempts ({waited:.0f}s waited, "
                    f"cap={max_total_wait_s}s); giving up"
                )
                raise
            log(
                f"::notice::{description}: rate-limited (429); sleeping "
                f"{sleep_s:.0f}s before retry (attempt {attempt}, "
                f"total waited so far {waited:.0f}s)"
            )
            await asyncio.sleep(sleep_s)
            waited += sleep_s
