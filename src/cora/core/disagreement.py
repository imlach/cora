"""T0 / T1 / T2 verdict-disagreement resolver.

Pure-logic module — no I/O, no LLM, no MCP. Takes the per-tier
verdict + blocker signals as input, returns the resolution
(`adopt`, `escalate_t3`) plus the verdict that should reach the
PR comment. The dispatcher composes this with the per-tier call
functions.

Policy summary:

  gap = abs(verdict_rank(T0) - verdict_rank(T2)) where
  rank: looks_good < minor < needs_changes
  (T1 is same-family as T0 — when present, it overrides T0's
  verdict for the disagreement comparison because T1's bigger-
  context continuation is by definition the more authoritative
  same-family view. T2 vs T1 is the only comparison that matters
  if T1 ran.)

  - gap == 0  → adopt the leading verdict, append other tier's
                extra findings as Notes (no banner)
  - gap == 1  → adopt the conservative verdict + collapsed dissent
                block; banner: "Tier disagreed — adopted <tier>
                (conservative)"
  - gap == 2  → categorical disagreement.
                  - if T2 has a Blocker → escalate to T3 cloud
                  - if T2 has only concerns → adopt T2 conservative
                    (the default; a later iteration may switch
                    Blocker-less gap=2 to T3 too once there is
                    eval-corpus data)

T3 escalation is gated by `t3_enabled` (callers default it off).
When disabled, gap=2+Blocker collapses to `adopt T2 conservative`
rather than escalating — same conservative posture but no cloud
cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cora.core.config import VERDICT_WORDS


# Verdict order. Higher = more concerning — rank is the word's index in
# the (ascending-concern) verdict vocabulary, `config.VERDICT_WORDS` by
# default: looks good=0, minor=1, needs changes=2. The mapper in
# `leak.py` keys off the same tuple, so the two can't drift; a custom
# vocabulary passed to `resolve_disagreement(words=…)` must match the
# one given to `parse_verdict_from_body`.


# `tier` is the alpha-numeric label the loop-logging events use; the
# resolver doesn't care about the underlying model alias.
Tier = Literal["T0", "T1", "T2", "T3"]


@dataclass(frozen=True)
class TierVerdict:
    """Per-tier verdict snapshot — what the dispatcher hands the
    resolver. Mirrors the fields `log_tier_verdict` emits to Loki
    so an offline analysis (eval-pipeline) can reconstruct the
    same input."""

    tier: Tier
    verdict: str | None  # lowercase verdict word; None when no body
    body: str  # full review body for the dissent block
    has_blocker: bool  # parsed from the `🚨 **Blocker:**` marker


@dataclass(frozen=True)
class Resolution:
    """The dispatcher's marching orders after the resolver runs.

    `path` discriminates the trace + banner shape; `adopted_tier`
    points the dispatcher at which body to use as the PR comment's
    primary content. `gap` is the verdict-rank distance between
    the two compared tiers (0 / 1 / 2); None when only one tier
    ran (no disagreement to compute).
    """

    path: Literal["agree", "adopt_conservative", "escalate_t3", "single_tier"]
    adopted_tier: Tier
    adopted_verdict: str | None
    dissent_tier: Tier | None  # the tier whose body goes in <details>
    gap: int | None
    banner: str | None  # markdown banner; None for the agree path


def resolve_disagreement(
    *,
    t0: TierVerdict,
    t1: TierVerdict | None = None,
    t2: TierVerdict | None = None,
    t3_enabled: bool = False,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> Resolution:
    """Resolve the post-pipeline verdict given the per-tier inputs.

    `t1` and `t2` are optional — pipelines without wall-hit
    continuation pass `t1=None`, pipelines without an alternate-family
    T2 pass `t2=None`. With only `t0`, this returns the `single_tier`
    path: no disagreement to resolve, dispatcher emits the T0 body
    as-is and no banner.

    `t3_enabled=False` (the default) collapses the
    gap=2-with-Blocker case to `adopt_conservative` instead of
    `escalate_t3`. A later iteration can flip this once the
    `event=disagreement gap=2` rate on a dashboard confirms the
    cloud escalation budget is acceptable.

    `words` is the verdict vocabulary in ascending-concern order
    (`config.VERDICT_WORDS` by default) — pass the same tuple the
    tier verdicts were parsed with.
    """
    # No T2 → nothing to disagree with. T1 may have run, but T1 is
    # same-family with T0 and the policy only treats it as
    # authoritative for T0-vs-T2 comparison (i.e., when there's a
    # T2 to compare against). With no T2, the leading body is
    # whichever same-family tier ran last (T1 if it did, else T0).
    if t2 is None:
        leading = t1 or t0
        return Resolution(
            path="single_tier",
            adopted_tier=leading.tier,
            adopted_verdict=leading.verdict,
            dissent_tier=None,
            gap=None,
            banner=None,
        )

    # T1 (same-family continuation) overrides T0 as the comparison
    # baseline when present — T1's bigger-context window is by
    # definition more authoritative on the same-family axis.
    same_family = t1 or t0
    gap = _verdict_gap(same_family.verdict, t2.verdict, words)

    if gap == 0:
        # Agree. Adopt the same-family body; T2's extra findings
        # surface as Notes via the dispatcher's body merge (not
        # this resolver's responsibility — return a clear `agree`
        # path so the caller knows to do the merge).
        return Resolution(
            path="agree",
            adopted_tier=same_family.tier,
            adopted_verdict=same_family.verdict,
            dissent_tier=None,
            gap=0,
            banner=None,
        )

    if gap == 1:
        conservative = _pick_conservative(same_family, t2, words)
        dissent = t2 if conservative is same_family else same_family
        return Resolution(
            path="adopt_conservative",
            adopted_tier=conservative.tier,
            adopted_verdict=conservative.verdict,
            dissent_tier=dissent.tier,
            gap=1,
            banner=(
                f"🔄 {same_family.tier}/{t2.tier} disagreed — adopted "
                f"{conservative.tier} (conservative)"
            ),
        )

    # gap == 2 — categorical disagreement.
    if t2.has_blocker and t3_enabled:
        return Resolution(
            path="escalate_t3",
            adopted_tier="T3",
            adopted_verdict=None,  # filled in by the dispatcher after T3 runs
            dissent_tier=same_family.tier,
            gap=2,
            banner=(
                f"🔄 {same_family.tier}/{t2.tier} disagreed categorically — "
                f"escalated to T3 (cloud); T3 verdict adopted"
            ),
        )

    # gap == 2 with T2 Blocker but T3 disabled (the default)
    # OR gap == 2 without T2 Blocker. Both collapse to the conservative
    # side — which is whichever tier has the higher verdict rank, not
    # always T2.
    #
    # When T2 has a Blocker AND T3 is disabled, the conservative-adopt
    # collapse is the safety call: the categorical disagreement
    # combined with T2's concrete evidence justifies the more-concerned
    # verdict even if T0 was the higher-ranked side.
    if t2.has_blocker:
        # T2 has concrete blocker — its `needs_changes` always wins
        # the conservative pick regardless of which side is
        # numerically higher (covers the edge case where T2's verdict
        # somehow ranks lower than T0's but T2 found a blocker).
        conservative = t2
        dissent = same_family
    else:
        conservative = _pick_conservative(same_family, t2, words)
        dissent = t2 if conservative is same_family else same_family
    return Resolution(
        path="adopt_conservative",
        adopted_tier=conservative.tier,
        adopted_verdict=conservative.verdict,
        dissent_tier=dissent.tier,
        gap=2,
        banner=(
            f"🔄 {same_family.tier}/{t2.tier} disagreed categorically — "
            f"adopted {conservative.tier} (conservative)"
        ),
    )


def _verdict_gap(a: str | None, b: str | None, words: tuple[str, str, str]) -> int:
    """Absolute distance between two verdicts on the
    looks_good < minor < needs_changes axis. Missing / unparseable
    verdicts rank as `minor` (the middle entry, `words[1]`) so the
    resolver doesn't collapse into an artificial agree path when either
    tier produced no parseable verdict — `minor` is the neutral default
    in `leak.py` too."""
    return abs(_rank(a, words) - _rank(b, words))


def _rank(verdict: str | None, words: tuple[str, str, str]) -> int:
    if verdict is None:
        return 1  # neutral middle rank — words[1] ("minor" by default)
    try:
        return words.index(verdict)
    except ValueError:
        return 1


def _pick_conservative(
    a: TierVerdict, b: TierVerdict, words: tuple[str, str, str]
) -> TierVerdict:
    """Higher verdict rank wins (more concerning = more conservative
    posture). Ties impossible by caller's gap=1 precondition."""
    return a if _rank(a.verdict, words) > _rank(b.verdict, words) else b
