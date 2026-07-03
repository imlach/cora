"""Escalation policy: tier ladder, trigger detection grounded in the
engine's wall-hit vocab, the generic re-prefill connector, and the
escalation driver (`run_escalation`) that wires the connector seam."""

from __future__ import annotations

import asyncio

import pytest

from cora.core.budget import Budget
from cora.escalation import (
    WALL_HIT_REASONS,
    EscalationContext,
    EscalationOutcome,
    EscalationPolicy,
    ReprefillConnector,
    Tier,
    escalation_triggers,
    run_escalation,
)
from cora.result import ReviewResult


def _result(*, verdict="looks good", terminated_reason=None) -> ReviewResult:
    return ReviewResult(
        verdict=verdict,
        verdict_line=None,
        conclusion="success",
        body="b",
        mode="deep",
        budget=Budget(max_input=0, max_output=0, max_iterations=0),
        wall_time_s=1.0,
        terminated_reason=terminated_reason,
    )


def test_reprefill_connector_copies_context():
    msgs = [{"role": "user", "content": "x"}]
    out = ReprefillConnector().handoff(msgs)
    assert out == msgs and out is not msgs  # copy, not alias


def test_single_tier_never_escalates():
    pol = EscalationPolicy.single("sonnet")
    assert len(pol.tiers) == 1
    assert pol.next_tier(0) is None
    assert pol.should_escalate(_result(verdict="needs changes"), 0) is False


def test_policy_validates_tiers_and_triggers():
    with pytest.raises(ValueError, match="at least one tier"):
        EscalationPolicy(tiers=[])
    with pytest.raises(ValueError, match="unknown escalation triggers"):
        EscalationPolicy(tiers=[Tier("m")], escalate_on=frozenset({"bogus"}))


def test_next_tier_walks_then_stops():
    pol = EscalationPolicy(tiers=[Tier("a"), Tier("b")])
    assert pol.next_tier(0) == Tier("b")
    assert pol.next_tier(1) is None


@pytest.mark.parametrize("reason", sorted(WALL_HIT_REASONS))
def test_wall_hit_reasons_trip_wall_hit(reason):
    assert "wall_hit" in escalation_triggers(_result(terminated_reason=reason))


def test_triggers_map_verdict_states():
    assert escalation_triggers(_result(verdict="needs changes")) == frozenset({"blocker"})
    assert escalation_triggers(_result(verdict=None)) == frozenset({"low_confidence"})
    assert escalation_triggers(_result(verdict="looks good")) == frozenset()
    # infra-ish terminated_reason that isn't a wall-hit doesn't trip wall_hit
    assert escalation_triggers(_result(terminated_reason="mcp-connect-failed")) == frozenset()


def test_should_escalate_requires_trigger_in_policy_and_a_higher_tier():
    two = [Tier("t0"), Tier("t1")]
    # reference-deployment shape: escalate on wall-hit
    wall = EscalationPolicy(tiers=two, escalate_on=frozenset({"wall_hit"}))
    assert wall.should_escalate(_result(terminated_reason="max_iterations"), 0) is True
    # trigger present but not opted into → no escalation
    assert wall.should_escalate(_result(verdict="needs changes"), 0) is False
    # opted-in trigger but no higher tier → no escalation
    assert wall.should_escalate(_result(terminated_reason="max_iterations"), 1) is False
    # generic adopter escalating on a blocker verdict
    blk = EscalationPolicy(tiers=two, escalate_on=frozenset({"blocker"}))
    assert blk.should_escalate(_result(verdict="needs changes"), 0) is True


# ── The escalation driver + generic connector seam ───────────────────


def _ctx(entry="wall_hit", **kw) -> EscalationContext:
    return EscalationContext(
        next_tier=Tier("opus", max_iterations=6),
        prev_context=[{"role": "assistant", "content": "t0 work"}],
        initial_user_prompt="review this PR",
        entry=entry,
        **kw,
    )


def test_reprefill_escalate_resumes_on_wall_hit():
    """Generic default: a wall-hit entry hands the prior trajectory forward
    (no fresh prompt) and re-runs the next tier — Sonnet→Opus out of the
    box, no bespoke infra."""
    seen: dict = {}

    async def run_tier(tier, prior, seed):
        seen.update(tier=tier, prior=prior, seed=seed)
        return "🟢 looks good\n\nFine.", None, ["git_show"]

    pol = EscalationPolicy(
        tiers=[Tier("sonnet"), Tier("opus", max_iterations=6)],
        escalate_on=frozenset({"wall_hit"}),
    )
    out = asyncio.run(run_escalation(pol, _ctx(entry="wall_hit"), run_tier))

    assert isinstance(out, EscalationOutcome)
    assert out.body.startswith("🟢")
    assert out.terminated_reason is None
    assert out.tools == ["git_show"]
    assert out.tier_ran == "opus"
    # Wall-hit resumes the trajectory: prior history forwarded, no fresh
    # prompt seed. The copy is a distinct object (re-prefill, not KV resume).
    assert seen["seed"] is None
    assert seen["prior"] == [{"role": "assistant", "content": "t0 work"}]
    assert seen["prior"] is not _ctx().prev_context
    assert seen["tier"].model == "opus"


def test_reprefill_escalate_fresh_seeds_initial_prompt():
    """A `fresh` entry seeds the next tier with the initial prompt instead
    of resuming — the single-model adopter's 'start the bigger model from
    scratch' path."""
    seen: dict = {}

    async def run_tier(tier, prior, seed):
        seen.update(seed=seed)
        return "verdict", None, []

    pol = EscalationPolicy(tiers=[Tier("sonnet"), Tier("opus")])
    asyncio.run(run_escalation(pol, _ctx(entry="fresh"), run_tier))
    assert seen["seed"] == "review this PR"


def test_run_escalation_delegates_to_policy_connector():
    """The driver is thin: it calls `policy.connector.escalate`. A custom
    connector proves the seam routes through the policy."""

    class StubConnector(ReprefillConnector):
        async def escalate(self, ctx, run_tier):
            return EscalationOutcome(
                body="from-stub", terminated_reason="stub", tools=[], tier_ran="z"
            )

    pol = EscalationPolicy(
        tiers=[Tier("a"), Tier("b")], connector=StubConnector()
    )

    async def never(*a):  # pragma: no cover — connector ignores the runner
        raise AssertionError("driver must not call run_tier itself")

    out = asyncio.run(run_escalation(pol, _ctx(), never))
    assert out.body == "from-stub" and out.tier_ran == "z"


def test_blocker_trigger_respects_custom_vocabulary():
    """`blocker_word` threads a custom verdict vocabulary into the
    `blocker` trigger — the default word must not fire for a deployment
    whose block-severity word differs."""
    res = _result(verdict="blocked")
    assert "blocker" not in escalation_triggers(res)
    assert "blocker" in escalation_triggers(res, blocker_word="blocked")

    pol = EscalationPolicy(
        tiers=[Tier("a"), Tier("b")], escalate_on=frozenset({"blocker"})
    )
    assert pol.should_escalate(res, 0) is False
    assert pol.should_escalate(res, 0, blocker_word="blocked") is True
