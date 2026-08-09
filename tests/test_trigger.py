"""TriggerPolicy / evaluate() — the trigger-security gate.

Pure-policy tests (no engine): the trust rules, the untrusted and fork
actions, the rate caps' best-effort semantics, the fail-loud action
validation, and `from_env` parsing of the REVIEW_TRIGGER_* knobs.
"""

from __future__ import annotations

import pytest

from cora.config import ReviewerConfig
from cora.trigger import TriggerContext, TriggerPolicy, evaluate


def _ctx(**overrides) -> TriggerContext:
    base = {
        "author": "mallory",
        "author_association": "NONE",
        "labels": frozenset(),
        "is_fork": False,
    }
    base.update(overrides)
    return TriggerContext(**base)


# ── Enforce switch ───────────────────────────────────────────────────


def test_unenforced_allows_everything_full_capability():
    d = evaluate(
        TriggerPolicy(enforce=False),
        _ctx(author_association="NONE", is_fork=True),
    )
    assert d.allowed and not d.degraded


# ── Trust rules (ANY-of) ─────────────────────────────────────────────


def test_association_trusts():
    p = TriggerPolicy(enforce=True)
    d = evaluate(p, _ctx(author="alice", author_association="MEMBER"))
    assert d.allowed and not d.degraded and d.trusted


def test_association_is_case_insensitive():
    p = TriggerPolicy(enforce=True)
    assert evaluate(p, _ctx(author_association="owner")).allowed


def test_author_allowlist_trusts_case_insensitively():
    p = TriggerPolicy(enforce=True, allowed_authors=frozenset({"Renovate[bot]"}))
    d = evaluate(p, _ctx(author="renovate[BOT]", author_association="NONE"))
    assert d.allowed and not d.degraded and d.trusted


def test_approve_label_trusts_case_insensitively():
    p = TriggerPolicy(enforce=True)
    d = evaluate(
        p, _ctx(author_association="NONE", labels=frozenset({"Cora:Approved"}))
    )
    assert d.allowed and d.trusted


def test_empty_association_is_untrusted():
    # The engine soft-fails the association fetch to "" — the gate must
    # fail CLOSED on that, not open.
    d = evaluate(TriggerPolicy(enforce=True), _ctx(author_association=""))
    assert not d.allowed


# ── Untrusted action ─────────────────────────────────────────────────


def test_untrusted_skip_denies_with_optin_hint():
    d = evaluate(TriggerPolicy(enforce=True), _ctx())
    assert not d.allowed and not d.trusted
    assert "cora:approved" in d.reason


def test_untrusted_comment_only_degrades():
    p = TriggerPolicy(enforce=True, untrusted_action="comment-only")
    d = evaluate(p, _ctx())
    assert d.allowed and d.degraded and not d.trusted


# ── Fork ceiling ─────────────────────────────────────────────────────


def test_fork_comment_only_caps_even_trusted_authors():
    p = TriggerPolicy(enforce=True)  # fork_action default: comment-only
    d = evaluate(p, _ctx(author_association="OWNER", is_fork=True))
    assert d.allowed and d.degraded and d.trusted


def test_fork_full_defers_to_trust_rules():
    p = TriggerPolicy(enforce=True, fork_action="full")
    assert not evaluate(p, _ctx(is_fork=True)).allowed
    d = evaluate(p, _ctx(author_association="MEMBER", is_fork=True))
    assert d.allowed and not d.degraded


def test_fork_skip_refuses_even_trusted_authors():
    p = TriggerPolicy(enforce=True, fork_action="skip")
    d = evaluate(p, _ctx(author_association="OWNER", is_fork=True))
    assert not d.allowed


# ── Rate caps ────────────────────────────────────────────────────────


def test_global_cap_denies_at_threshold():
    p = TriggerPolicy(enforce=True, max_runs_per_hour=10)
    d = evaluate(
        p, _ctx(author_association="OWNER", recent_runs_total=10)
    )
    assert not d.allowed and "global rate cap" in d.reason


def test_author_cap_denies_at_threshold():
    p = TriggerPolicy(enforce=True, max_runs_per_author_per_hour=3)
    d = evaluate(
        p, _ctx(author_association="OWNER", recent_runs_by_author=3)
    )
    assert not d.allowed and "per-author rate cap" in d.reason


def test_caps_below_threshold_allow():
    p = TriggerPolicy(
        enforce=True, max_runs_per_hour=10, max_runs_per_author_per_hour=3
    )
    d = evaluate(
        p,
        _ctx(
            author_association="OWNER",
            recent_runs_total=9,
            recent_runs_by_author=2,
        ),
    )
    assert d.allowed


def test_caps_skip_when_counts_unknown():
    # Probe failed → None counts → cost guard skipped (availability),
    # while the trust gate still applies.
    p = TriggerPolicy(enforce=True, max_runs_per_hour=1)
    assert evaluate(p, _ctx(author_association="OWNER")).allowed
    assert not evaluate(p, _ctx(author_association="NONE")).allowed


# ── Fail-loud validation ─────────────────────────────────────────────


def test_invalid_untrusted_action_raises():
    with pytest.raises(ValueError, match="untrusted_action"):
        TriggerPolicy(untrusted_action="skp")


def test_invalid_fork_action_raises():
    with pytest.raises(ValueError, match="fork_action"):
        TriggerPolicy(fork_action="block")


# ── from_env parsing ─────────────────────────────────────────────────


def test_from_env_defaults_to_enforced():
    cfg = ReviewerConfig.from_env({})
    assert cfg.trigger == TriggerPolicy()
    assert cfg.trigger.enforce is True


def test_from_env_can_disable_enforcement():
    cfg = ReviewerConfig.from_env({"REVIEW_TRIGGER_ENFORCE": "false"})
    assert cfg.trigger.enforce is False


def test_from_env_parses_all_knobs():
    cfg = ReviewerConfig.from_env(
        {
            "REVIEW_TRIGGER_ENFORCE": "true",
            "REVIEW_TRIGGER_ALLOWED_ASSOCIATIONS": "owner, member",
            "REVIEW_TRIGGER_ALLOWED_AUTHORS": "alice,renovate[bot]",
            "REVIEW_TRIGGER_APPROVE_LABEL": "review:approved",
            "REVIEW_TRIGGER_UNTRUSTED_ACTION": "Comment-Only",
            "REVIEW_TRIGGER_FORK_ACTION": "skip",
            "REVIEW_TRIGGER_MAX_RUNS_PER_HOUR": "30",
            "REVIEW_TRIGGER_MAX_RUNS_PER_AUTHOR_PER_HOUR": "6",
        }
    )
    t = cfg.trigger
    assert t.enforce is True
    assert t.allowed_associations == frozenset({"OWNER", "MEMBER"})
    assert t.allowed_authors == frozenset({"alice", "renovate[bot]"})
    assert t.approve_label == "review:approved"
    assert t.untrusted_action == "comment-only"
    assert t.fork_action == "skip"
    assert t.max_runs_per_hour == 30
    assert t.max_runs_per_author_per_hour == 6


def test_from_env_invalid_action_raises():
    # A typo'd security knob must fail loud even when enforcement is
    # explicitly disabled.
    with pytest.raises(ValueError):
        ReviewerConfig.from_env({
            "REVIEW_TRIGGER_ENFORCE": "false",
            "REVIEW_TRIGGER_UNTRUSTED_ACTION": "skp",
        })
