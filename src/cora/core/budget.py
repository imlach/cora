"""Per-PR token + tool-call accounting for the agentic reviewer +
triage agent.

Both consumers — the review orchestrator (deep mode) and
the triage agent — populate `Budget` through small
adapter classes in `deep_review.py` / the triage entrypoint so the
finish-line / check-run summary code in `summary.py` +
`check_run.py` reads a uniform shape.

Lives in the shared core package because both consumers already
share other helpers from it (`check_run`, `summary`, `comment`); the
module is reviewer-flavoured but the contents are agent-shared.

Exposes:
- `Budget` — input/output token + iteration counter + LiteLLM header
  capture + resolved-model chip
- `PER_CALL_TIMEOUT_S` — single-call timeout (180s) used by both
  entrypoints as the `timeout` field of `ModelSettings`. Matches the
  gateway's longest single-leg per-model timeout.

The `resolved_model` chip is populated from `x-litellm-*` response
headers captured by `litellm_capture.py` — pydantic-ai's Agent wraps
the openai client, so the capture path runs through a custom
`httpx.AsyncClient` event hook threaded into `OpenAIProvider`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


# Client-side per-call cap on a single LLM completion. MUST be ≥
# the gateway's longest single-leg per-model timeout (180s in the
# reference deployment) so the openai client doesn't give up on a
# legitimately slow successful call before LiteLLM hands the response
# back. The old 120s was lower than LiteLLM's 90s × ~30% margin — the
# moment the backend timeout was bumped to 180s, the client had to
# follow. 180s matches the single-leg cap; the consumer workflow's
# `timeout-minutes` absorbs the rest. Don't bump further without
# also bumping the outer workflow timeout — the iteration budget
# halves otherwise.
import os as _os
PER_CALL_TIMEOUT_S = int(_os.environ.get("AGENT_REVIEW_PER_CALL_TIMEOUT_S", "180"))

# One-time additional budget for the FIRST T0 model call, to absorb a
# scale-from-zero backend's cold start. The per-call clock starts when the
# call is dispatched, but a scaled-to-zero backend isn't serving for up
# to its readiness budget (90s in the reference deployment) while it
# warms up — so on a cold hit ~half the 180s base cap is gone
# before the model emits a token, and a thinky PR times out. Applied ONLY
# to T0's first call (later T0 calls + all of T1/T2 hit the now-warm
# backend, so they keep the base cap). Bounded to the readiness window:
# if the backend is genuinely stuck the gateway still errors at 90s and
# its fallback cascade takes over, so this can't hang on a dead backend.
# The pretrigger keeps cold hits rare; this is the backstop for when it
# didn't land in time.
T0_COLD_START_ALLOWANCE_S = int(
    _os.environ.get("AGENT_REVIEW_T0_COLD_START_ALLOWANCE_S", "90")
)


@dataclass
class Budget:
    """Tracks accumulated cost across the agent loop. First cap hit
    forces the next call to be a final terminal completion."""

    max_input: int
    max_output: int
    max_iterations: int
    input_used: int = 0
    output_used: int = 0
    # Total tool dispatches across every tier of a single PR review.
    # T0 (deep_review.py) and T1 (continuation.py) both feed this via
    # `on_tool_call=budget.add_tool_call`, so the post-loop value is
    # the union — not T0-only. The cap field above (`max_iterations`)
    # is T0-only by design: T1's request cap is enforced by
    # pydantic-ai's `UsageLimits` (caught as `UsageLimitExceeded`),
    # not by `reason_if_over`. Don't confuse the counter (combined)
    # with the cap (T0-only).
    iterations: int = 0
    tool_calls: Counter = field(default_factory=Counter)
    # The concrete model that LiteLLM resolved the request to (the
    # primary backend, or a fallback when the chain falls
    # through). Captured per turn via the `litellm_capture.py` httpx
    # hook; last-write-wins so the value after the loop reflects
    # whichever backend produced the final answer.
    resolved_model: str | None = None
    # Full diagnostic dump of x-litellm-* headers (last call only).
    litellm_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        """Construct the per-review Budget from a `ReviewerConfig`'s caps
        (`max_input_tokens` / `max_output_tokens` / `max_tool_iterations`).
        Duck-typed to keep this module free of a `cora.config` import —
        the config's field defaults mirror the engine constants, so a
        default-constructed config yields the stock engine budget."""
        return cls(
            max_input=cfg.max_input_tokens,
            max_output=cfg.max_output_tokens,
            max_iterations=cfg.max_tool_iterations,
        )

    def add_usage(self, usage) -> None:
        """Forward an OpenAI-shaped usage object into the counters.
        Both entrypoints adapt Pydantic-AI's `RunUsage` shape into
        `prompt_tokens` / `completion_tokens` via small adapter
        classes before passing it in."""
        self.input_used += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output_used += int(getattr(usage, "completion_tokens", 0) or 0)

    def add_tool_call(self, name: str) -> None:
        """Record one tool dispatch. Increments both the per-name
        Counter (read by the finish-line summary) and `iterations`
        (the cap used by `reason_if_over`). Wired into both T0
        (`deep_review.py`) and T1 (`continuation.py`) via
        `on_tool_call=budget.add_tool_call`, so the `iterations`
        total at finish reflects the combined T0+T1 dispatches —
        intentional, dashboards read it as the merged turn count."""
        self.tool_calls[name] += 1
        self.iterations += 1

    def set_resolved_model(self, name: str | None) -> None:
        if name:
            self.resolved_model = name

    def record_litellm_headers(self, headers) -> None:
        """Capture every `x-litellm-*` header from the LiteLLM response
        for diagnostic visibility. Overwrites on each call — last value
        wins so the post-loop summary reflects the final turn.

        Accepts either an `httpx.Headers` (case-insensitive) or a
        plain dict drained from `litellm_capture.drain_captured_headers`
        — both expose `.items()` the same way."""
        if not headers:
            return
        # OpenAI SDK headers are an httpx.Headers (case-insensitive).
        # Iterating yields lowercased keys.
        try:
            self.litellm_headers = {
                k: v for k, v in headers.items() if k.lower().startswith("x-litellm-")
            }
        except Exception:  # noqa: BLE001  # pragma: no cover — best-effort diag only
            pass

    def reason_if_over(self) -> str | None:
        if self.input_used >= self.max_input:
            return f"input tokens ({self.input_used} ≥ {self.max_input})"
        if self.output_used >= self.max_output:
            return f"output tokens ({self.output_used} ≥ {self.max_output})"
        if self.iterations >= self.max_iterations:
            return f"tool iterations ({self.iterations} ≥ {self.max_iterations})"
        return None


def _has_token_fields(usage) -> bool:
    """True if `usage` already exposes token counters — i.e. it IS the
    RunUsage object, not a method that returns one. Used to decide
    whether `resolve_run_usage` must call the accessor."""
    return any(
        hasattr(usage, name)
        for name in ("input_tokens", "output_tokens", "request_tokens", "response_tokens")
    )


def resolve_run_usage(result):
    """Return the pydantic-ai RunUsage for a completed agent run,
    tolerating pydantic-ai's method→property migration of
    `AgentRunResult.usage`.

    Through pydantic-ai 1.10x the accessor is mid-flight on a v2 staging:
    older builds expose `usage()` as a method, newer ones expose `usage`
    as a property returning the RunUsage directly. Calling the new
    property (`result.usage()`) raises `'RunUsage' object is not callable`
    — which silently zeroed token accounting because every call site
    soft-fails the adapter (observed as `~0 in / ~0 out tokens` in
    review footers). Read
    the attribute and only call it when handed a bare callable (the old
    method, or a test fake); never call the usage object itself, whose
    transitional deprecated-callable shim would emit a spurious warning."""
    usage = getattr(result, "usage", None)
    if usage is not None and callable(usage) and not _has_token_fields(usage):
        usage = usage()
    return usage


def model_name_from_result(result) -> str | None:
    """Best-effort `resolved_model` fallback for provider paths that
    never emit `x-litellm-*` headers (the direct Anthropic/Bedrock SDK
    paths have no gateway in front of them to set any). Reads
    pydantic-ai's `ModelResponse.model_name` off the run's final
    response — the provider's own answer to "what model actually
    served this" — so the check-run "Backend" chip keeps working
    without a gateway. Returns `None` on any shape mismatch (older
    pydantic-ai, a test double, ...) so callers can chain it after
    `resolved_model_from(captured)` without an extra guard."""
    try:
        response = getattr(result, "response", None)
        return getattr(response, "model_name", None) or None
    except Exception:  # noqa: BLE001 — diagnostic-only, never fatal
        return None


def usage_tokens(usage, *names: str) -> int:
    """First present, non-zero token counter from `usage`, trying `names`
    in order. pydantic-ai renamed `request_tokens`/`response_tokens` →
    `input_tokens`/`output_tokens` (the old names survive as deprecated
    aliases); pass the new name first so we read the deprecated alias —
    and emit its DeprecationWarning — only when the new field is absent."""
    for name in names:
        value = getattr(usage, name, 0) or 0
        if value:
            return int(value)
    return 0
