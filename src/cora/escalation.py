"""Escalation — the generalised tier ladder.

A review can run on a cheap/fast model and *escalate* to a stronger one on
a trigger (think **Sonnet → Opus**). A two-tier deep deployment resuming a
wall-hit trajectory on a bigger endpoint is one instance; a single-model
adopter just uses one tier and never escalates.

- `Tier` — one rung (a model + its iteration budget).
- `EscalationPolicy` — the ladder + which triggers escalate + the connector.
- `EscalationConnector` — how a finished tier hands its working context to
  the next. The base `handoff` shapes the message history; `escalate`
  drives the *full* next-tier dispatch. `ReprefillConnector` (default)
  passes the message history forward and re-runs the next tier via the
  supplied tier-runner — no infra required, so a generic two-tier adopter
  (Sonnet→Opus) works out of the box. Deep mode's default is
  `cora.core.kv_continuation.KvContinuationConnector`, which resumes the
  prior trajectory instead of re-prefilling.
- `EscalationContext` / `EscalationOutcome` — the loose-typed bag the driver
  hands the connector and the result it returns. Kept here (not in the
  engine) so the seam itself carries no engine dependency.
- `run_escalation` — the driver: asks the policy whether tier N's result
  escalates and, if so, calls the connector's `escalate`.

Configuration: `ReviewerConfig.escalation_policy` swaps the whole policy
(custom ladder, triggers, connector); `ReviewerConfig.escalation_triggers`
tunes just the trigger set on the default ladder.

This module is dependency-free (no pydantic_ai / openai / engine imports) so
it imports cheaply; the message history and the tier-runner callback are
typed loosely (`list`, `Callable`) for the same reason — the connector
implementations that *do* touch the engine live under `cora.core`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cora.result import ReviewResult

# `terminated_reason` values that mean "the tier was actively working and ran
# out of room" — the deep-mode continuation trigger. Distinct from infra
# failures (mcp-connect-failed) or empty responses, which shouldn't escalate.
WALL_HIT_REASONS = frozenset(
    {"max_iterations", "wall_time", "per_call_timeout", "budget_exhausted"}
)

# The triggers a policy can escalate on. `wall_hit` is the deep-mode
# default; `blocker` / `low_confidence` let an adopter escalate a small
# model to a larger one to double-check a blocker / a no-verdict outcome.
# `no_tool_use` catches a deep review that verdicted without a single tool
# call — under identical inputs the same model can produce a 0-call review
# asserting "I confirmed X" and a 35-call review actually confirming it, so
# a 0-call deep verdict is unverified by construction.
ESCALATE_TRIGGERS = frozenset(
    {"wall_hit", "blocker", "low_confidence", "no_tool_use"}
)


def escalation_triggers(
    result: ReviewResult, *, blocker_word: str = "needs changes"
) -> frozenset[str]:
    """Which escalation triggers a tier's result trips (independent of any
    policy). The policy intersects this with its `escalate_on` set.

    `blocker_word` is the block-severity verdict word — pass
    `cfg.verdict_words[2]` when a deployment runs a custom verdict
    vocabulary."""
    hits: set[str] = set()
    if result.terminated_reason in WALL_HIT_REASONS:
        hits.add("wall_hit")
    if result.verdict == blocker_word:
        hits.add("blocker")
    if result.verdict is None:
        hits.add("low_confidence")
    # Deep mode only: quick mode runs without tools, so zero calls is its
    # normal shape, not a signal.
    if result.mode == "deep" and not sum(result.budget.tool_calls.values()):
        hits.add("no_tool_use")
    return frozenset(hits)


@dataclass
class Tier:
    """One rung of the escalation ladder — the model a review runs on."""

    model: str
    endpoint: str | None = None   # None → same gateway, different model alias
    max_iterations: int = 0       # 0 → quick / no-tools


# A tier-runner: dispatches one tier given `(tier, prior_messages,
# initial_user_prompt)` and returns `(body, terminated_reason, tools)`.
# The driver/connectors are handed this callback by `run_review` so the
# escalation seam never imports the engine's dispatch functions directly.
TierRunner = Callable[
    ["Tier", list, "str | None"],
    Awaitable[tuple[str, "str | None", list[str]]],
]


@dataclass
class EscalationContext:
    """The working context a finished tier hands to its successor. Loose
    typing keeps this module engine-free — the engine fills the fields and
    the connector reads them.

    `prev_context` is the previous tier's message history; `entry` names how
    the next tier is being entered (`"wall_hit"` resumes the trajectory;
    `"fresh"` starts the next tier from `initial_user_prompt`). Forced
    entries (`classifier_large_start` / per-call fresh start) map onto
    `entry="fresh"` with a distinguishing `tag`."""

    next_tier: Tier
    prev_context: list
    initial_user_prompt: str | None
    entry: str = "wall_hit"
    tag: str | None = None
    terminated_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationOutcome:
    """What a connector's `escalate` returns. `body` empty → the next tier
    produced nothing, so the caller keeps the prior tier's result and
    `terminated_reason`. `tier_ran` is the model alias to append to the
    `tiers_run` trail (None → nothing ran)."""

    body: str
    terminated_reason: str | None
    tools: list[str]
    tier_ran: str | None = None


class EscalationConnector(ABC):
    """Hands a finished tier's working context to the next tier and drives
    its dispatch.

    `handoff` shapes the seed message history (the cheap, sync transform);
    `escalate` runs the full next-tier dispatch via the supplied
    `TierRunner`. Subclasses that only customise the history transform can
    rely on the default `escalate`, which re-prefills and re-runs."""

    @abstractmethod
    def handoff(self, prev_context: list) -> list:
        """Return the seed message history for the next tier, given the
        previous tier's message history."""

    async def escalate(
        self, ctx: EscalationContext, run_tier: TierRunner
    ) -> EscalationOutcome:
        """Dispatch the next tier. Default: hand the prior history forward
        (via `handoff`) and re-run the tier. A wall-hit entry resumes from
        the handed-off trajectory; a fresh entry seeds from the initial
        prompt. Override to add infra-specific behaviour (e.g. KV flush)."""
        prior = self.handoff(ctx.prev_context)
        seed_prompt = ctx.initial_user_prompt if ctx.entry == "fresh" else None
        body, reason, tools = await run_tier(ctx.next_tier, prior, seed_prompt)
        return EscalationOutcome(
            body=body,
            terminated_reason=reason,
            tools=tools,
            tier_ran=ctx.next_tier.model,
        )


class ReprefillConnector(EscalationConnector):
    """Generic default — pass the prior tier's messages forward; the next
    (stronger) model re-prefills them. No infra required. Deep mode's
    trajectory-resume variant is a separate connector
    (`cora.core.kv_continuation.KvContinuationConnector`) with its own
    entry framing + terminated_reason vocabulary."""

    def handoff(self, prev_context: list) -> list:
        return list(prev_context)


@dataclass
class EscalationPolicy:
    """The tier ladder + escalation rule. Single-tier (the default via
    `EscalationPolicy.single`) never escalates."""

    tiers: list[Tier]
    escalate_on: frozenset[str] = frozenset()
    connector: EscalationConnector = field(default_factory=ReprefillConnector)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("EscalationPolicy needs at least one tier")
        unknown = self.escalate_on - ESCALATE_TRIGGERS
        if unknown:
            raise ValueError(f"unknown escalation triggers: {sorted(unknown)}")

    @classmethod
    def single(cls, model: str, *, max_iterations: int = 0) -> EscalationPolicy:
        """No escalation — one tier. The adopter default."""
        return cls(tiers=[Tier(model=model, max_iterations=max_iterations)])

    def next_tier(self, from_index: int) -> Tier | None:
        nxt = from_index + 1
        return self.tiers[nxt] if 0 <= nxt < len(self.tiers) else None

    def should_escalate(
        self,
        result: ReviewResult,
        from_index: int,
        *,
        blocker_word: str = "needs changes",
    ) -> bool:
        """True if `result` from tier `from_index` trips a configured trigger
        and a higher tier exists to escalate to. `blocker_word` threads a
        custom verdict vocabulary into the `blocker` trigger."""
        if self.next_tier(from_index) is None:
            return False
        return bool(
            escalation_triggers(result, blocker_word=blocker_word)
            & self.escalate_on
        )


async def run_escalation(
    policy: EscalationPolicy,
    ctx: EscalationContext,
    run_tier: TierRunner,
    *,
    from_index: int = 0,
) -> EscalationOutcome:
    """Drive one rung of the ladder via the policy's connector.

    The caller (`run_review`) has already run tier `from_index` and decided
    a higher tier should run (either the policy's `should_escalate` tripped,
    or a forced entry like classifier-large / per-call fresh-start). This
    hands the connector the working context and returns the next tier's
    outcome.

    Thin by design: the policy decides *whether* to escalate; the connector
    owns *how* (re-prefill vs. trajectory resume). Forced entries set
    `ctx.entry="fresh"` and bypass `should_escalate`."""
    return await policy.connector.escalate(ctx, run_tier)
