"""Tests for the T0/T1/T2 verdict-disagreement resolver.

Pure-logic module — no I/O, no LLM. Verifies the documented policy
table holds for every combination the dispatcher will hand it.
"""
from __future__ import annotations


from cora.core.disagreement import (
    Resolution,
    TierVerdict,
    resolve_disagreement,
)


def _tv(tier, verdict, has_blocker=False, body=""):
    """Tiny ctor — defaults the body to empty since the resolver
    doesn't read body content, only the verdict + blocker fields."""
    return TierVerdict(tier=tier, verdict=verdict, body=body, has_blocker=has_blocker)


# --- single-tier path (no T2; T1 may or may not have run) ---


def test_single_tier_returns_t0_when_only_t0_ran():
    """No T1, no T2 — most reviews. Dispatcher should adopt T0's
    body verbatim with no banner."""
    r = resolve_disagreement(t0=_tv("T0", "looks good"))
    assert r == Resolution(
        path="single_tier",
        adopted_tier="T0",
        adopted_verdict="looks good",
        dissent_tier=None,
        gap=None,
        banner=None,
    )


def test_single_tier_prefers_t1_when_continuation_fired():
    """T1 ran (wall-hit continuation) but no T2 to compare with.
    T1's body wins — it's the more authoritative same-family view."""
    r = resolve_disagreement(
        t0=_tv("T0", "needs changes"),
        t1=_tv("T1", "looks good"),
    )
    assert r.path == "single_tier"
    assert r.adopted_tier == "T1"
    assert r.adopted_verdict == "looks good"


# --- gap == 0 (agree) ---


def test_agree_adopts_t0_with_no_banner():
    """T0 and T2 agree on `looks good`. No disagreement, no banner —
    the dispatcher merges T2's extra Notes into the T0 body
    elsewhere; this resolver just signals the agree path."""
    r = resolve_disagreement(
        t0=_tv("T0", "looks good"),
        t2=_tv("T2", "looks good"),
    )
    assert r.path == "agree"
    assert r.adopted_tier == "T0"
    assert r.adopted_verdict == "looks good"
    assert r.gap == 0
    assert r.banner is None


def test_agree_when_t1_ran_uses_t1_as_baseline():
    """T1 ran; T1 and T2 both say `needs changes`. Resolver adopts
    T1 (the same-family baseline when present)."""
    r = resolve_disagreement(
        t0=_tv("T0", "looks good"),
        t1=_tv("T1", "needs changes"),
        t2=_tv("T2", "needs changes"),
    )
    assert r.path == "agree"
    assert r.adopted_tier == "T1"
    assert r.gap == 0


# --- gap == 1 (one tier apart, adopt conservative) ---


def test_gap_one_adopts_conservative_t2():
    """T0 says `minor`, T2 says `needs changes` — adopt T2 (more
    conservative), banner reflects the disagreement."""
    r = resolve_disagreement(
        t0=_tv("T0", "minor"),
        t2=_tv("T2", "needs changes"),
    )
    assert r.path == "adopt_conservative"
    assert r.adopted_tier == "T2"
    assert r.adopted_verdict == "needs changes"
    assert r.dissent_tier == "T0"
    assert r.gap == 1
    assert r.banner is not None
    assert "T0/T2 disagreed" in r.banner
    assert "T2 (conservative)" in r.banner


def test_gap_one_adopts_conservative_t0_when_t0_is_higher():
    """Reversed: T0 says `minor`, T2 says `looks good` — T0 is the
    more conservative tier, adopt T0, dissent goes to T2."""
    r = resolve_disagreement(
        t0=_tv("T0", "minor"),
        t2=_tv("T2", "looks good"),
    )
    assert r.path == "adopt_conservative"
    assert r.adopted_tier == "T0"
    assert r.adopted_verdict == "minor"
    assert r.dissent_tier == "T2"
    assert r.gap == 1


# --- gap == 2 (categorical disagreement) ---


def test_gap_two_with_t2_blocker_and_t3_enabled_escalates():
    """T0 `looks good` vs T2 `needs changes` WITH `🚨 Blocker:` —
    escalate to T3 cloud when t3_enabled=True."""
    r = resolve_disagreement(
        t0=_tv("T0", "looks good"),
        t2=_tv("T2", "needs changes", has_blocker=True),
        t3_enabled=True,
    )
    assert r.path == "escalate_t3"
    assert r.adopted_tier == "T3"
    assert r.adopted_verdict is None  # dispatcher fills in after T3 runs
    assert r.dissent_tier == "T0"
    assert r.gap == 2
    assert "escalated to T3" in r.banner


def test_gap_two_with_t2_blocker_but_t3_disabled_collapses_to_conservative():
    """Same gap=2+Blocker case but t3_enabled=False (the default) —
    collapses to adopting T2 conservative, no T3 cost."""
    r = resolve_disagreement(
        t0=_tv("T0", "looks good"),
        t2=_tv("T2", "needs changes", has_blocker=True),
        t3_enabled=False,
    )
    assert r.path == "adopt_conservative"
    assert r.adopted_tier == "T2"
    assert r.adopted_verdict == "needs changes"
    assert r.dissent_tier == "T0"
    assert r.gap == 2
    assert "categorically" in r.banner


def test_gap_two_no_blocker_adopts_t2_conservative_regardless_of_t3_flag():
    """gap=2 but T2 found no concrete Blocker (only concerns).
    Doesn't justify cloud escalation cost; adopts T2 conservative
    either way."""
    for t3_enabled in (False, True):
        r = resolve_disagreement(
            t0=_tv("T0", "needs changes"),
            t2=_tv("T2", "looks good"),
            t3_enabled=t3_enabled,
        )
        assert r.path == "adopt_conservative", (
            f"t3_enabled={t3_enabled} unexpectedly took escalate path"
        )
        # T0's `needs changes` is the conservative side here.
        assert r.adopted_tier == "T0"
        assert r.gap == 2


# --- robustness: missing / unparseable verdicts ---


def test_none_verdict_treated_as_minor_for_ranking():
    """If a tier ran but emitted no parseable verdict, the resolver
    ranks it as `minor` (the same neutral default `leak.py` uses).
    Prevents a None from accidentally agreeing with a `looks good`
    or `needs changes` purely because of a missing field."""
    r = resolve_disagreement(
        t0=_tv("T0", "looks good"),
        t2=_tv("T2", None),
    )
    # `looks good` (0) vs `minor` (1) — gap=1, adopt conservative.
    assert r.path == "adopt_conservative"
    assert r.gap == 1
