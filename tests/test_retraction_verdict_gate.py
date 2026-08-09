"""Verdict derivation from the surviving finding set (cora #38).

`detect_blocker` already discounts a `🚨 Blocker` bullet the model
retracts in its own text (#32), so automerge stopped pausing over
nothing — but the verdict line still said 🔴 `needs changes`, so the
required check went red over the same nothing. These tests pin the
gate that closes the disagreement, and the whole-review rule that keeps
it from downgrading a review with a real blocker in it.
"""

from __future__ import annotations

import pytest

from cora.config import ReviewerConfig
from cora.core.leak import count_blocker_retractions
from cora.review._retraction_gate import apply_retraction_verdict_gate


class _Run:
    """Minimal ReviewRun stand-in — the gate touches only these."""

    def __init__(self, body, verdict, cfg=None):
        self.cfg = cfg or ReviewerConfig()
        self.body_to_post = body
        self.verdict = verdict
        self.pr_number = "42"
        self.mode = "deep"
        self.loki_lines: list[str] = []

    def loki(self, line, labels=None):
        self.loki_lines.append(line)


_RETRACTED = (
    "🔴 needs changes\n\nSummary.\n\n**Findings:**\n\n"
    "- 🚨 **Blocker:** `foo()` may not exist — actually, checking the "
    "source, this is a false alarm and the code is fine.\n"
)
_LIVE = (
    "🔴 needs changes\n\nSummary.\n\n**Findings:**\n\n"
    "- 🚨 **Blocker:** `bar()` raises on empty input (bar.py:12).\n"
)


def test_downgrades_when_every_blocker_retracts():
    run = _Run(_RETRACTED, "needs changes")
    apply_retraction_verdict_gate(run)
    assert run.verdict == "minor"
    assert run.body_to_post.startswith("🟡 minor")
    assert "harness note: verdict downgraded" in run.body_to_post


def test_keeps_the_findings_visible():
    """Annotate and downgrade — never delete. A human still reads what
    the model actually wrote."""
    run = _Run(_RETRACTED, "needs changes")
    apply_retraction_verdict_gate(run)
    assert "🚨 **Blocker:**" in run.body_to_post
    assert "false alarm" in run.body_to_post


def test_one_live_blocker_blocks_the_downgrade():
    """Whole-review retraction required — a review with a real blocker
    and a withdrawn one is still a blocked review."""
    run = _Run(_RETRACTED + _LIVE.partition("\n\n")[2], "needs changes")
    apply_retraction_verdict_gate(run)
    assert run.verdict == "needs changes"
    assert run.body_to_post.startswith("🔴")


def test_no_blockers_at_all_is_a_noop():
    """A 🔴 with no Blocker bullets is the model's own stated
    conclusion — not this gate's business."""
    body = "🔴 needs changes\n\nSummary only, no findings list.\n"
    run = _Run(body, "needs changes")
    apply_retraction_verdict_gate(run)
    assert run.verdict == "needs changes"
    assert run.body_to_post == body


@pytest.mark.parametrize("verdict", ["looks good", "minor"])
def test_non_blocking_verdicts_are_untouched(verdict):
    run = _Run(_RETRACTED.replace("🔴 needs changes", f"🟡 {verdict}"), verdict)
    apply_retraction_verdict_gate(run)
    assert run.verdict == verdict


def test_killswitch_disables_it():
    run = _Run(_RETRACTED, "needs changes", cfg=ReviewerConfig(retraction_verdict_gate=False))
    apply_retraction_verdict_gate(run)
    assert run.verdict == "needs changes"


def test_env_killswitch(monkeypatch):
    assert ReviewerConfig().retraction_verdict_gate is True
    monkeypatch.setenv("AGENT_REVIEW_RETRACTION_VERDICT_GATE", "false")
    monkeypatch.setenv("GH_REPO", "o/r")
    monkeypatch.setenv("PR_NUMBER", "1")
    assert ReviewerConfig.from_env().retraction_verdict_gate is False


def test_empty_body_or_verdict_is_a_noop():
    for body, verdict in (("", "needs changes"), (_RETRACTED, "")):
        run = _Run(body, verdict)
        apply_retraction_verdict_gate(run)  # must not raise
        assert run.verdict == verdict


def test_contrastive_clause_still_counts_as_live():
    """The narrowing case the #32 pattern guards: a retraction phrase
    walked back by a contrastive clause is a live finding."""
    body = (
        "🔴 needs changes\n\nS.\n\n**Findings:**\n\n"
        "- 🚨 **Blocker:** the surrounding code is fine, but this path "
        "crashes on None (x.py:9).\n"
    )
    run = _Run(body, "needs changes")
    apply_retraction_verdict_gate(run)
    assert run.verdict == "needs changes"


def test_count_helper_reports_totals():
    assert count_blocker_retractions(_RETRACTED) == (1, 1)
    assert count_blocker_retractions(_LIVE) == (1, 0)
    assert count_blocker_retractions("no findings here") == (0, 0)


def test_the_38_phrasings_are_not_matched_documented_limit():
    """#38's own examples do not match `_BLOCKER_RETRACTION_RE`, and
    that is deliberate: widening the phrase list to catch them trades
    the safe failure direction (a retracted bullet still posts) for the
    dangerous one (a live blocker silently dropped). Pinned so the
    limitation stays visible — the real fix is upstream, in what the
    model emits."""
    body = (
        "🔴 needs changes\n\nS.\n\n**Findings:**\n\n"
        "- 🚨 **Blocker:** this could double-free — let me re-examine. "
        "This logic appears sound. So this case is unreachable. Good.\n"
    )
    total, retracted = count_blocker_retractions(body)
    assert (total, retracted) == (1, 0)
    run = _Run(body, "needs changes")
    apply_retraction_verdict_gate(run)
    assert run.verdict == "needs changes"  # still red — see #38
