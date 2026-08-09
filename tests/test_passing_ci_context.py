"""Passing CI reaches the prompt, so world-knowledge claims have
counter-evidence (cora #44).

The observed failure, on a downstream Go PR: deep mode, **zero tool
calls**, three findings — "`go 1.26.5` is not a valid Go version … this
will cause `go build` to fail", an unread-function claim, and "if these
options don't exist, compilation will fail. Recommend confirming". The
build check for that SHA was green, so every claim was already refuted.

The cause was two individually-correct behaviours composing into a
blind spot: `gather_ci_context` returned None whenever nothing was
failing, and `deep.md` forbids inferring success from silence. The
greener the PR, the less the reviewer knew.
"""

from __future__ import annotations

import json

import pytest

from cora.config import ReviewerConfig
from cora.core import pr_context as _prc
from cora.core.prompt import load_system_prompt

SHA = "abc1234def5678"


def _cr(name, conclusion, started="2026-08-09T10:00:00Z"):
    return {"name": name, "conclusion": conclusion, "started_at": started}


@pytest.fixture
def checks(monkeypatch):
    box = {"runs": []}
    monkeypatch.setattr(
        _prc, "run", lambda cmd, **k: json.dumps({"check_runs": box["runs"]})
    )
    return box


def test_all_green_now_yields_context_instead_of_none(checks):
    """The regression. Previously None — the reviewer saw no CI at all."""
    checks["runs"] = [_cr("api", "success"), _cr("web", "success")]
    out = _prc.gather_ci_context("o/r", SHA)
    assert out is not None
    assert "all reported checks passing" in out
    assert "`api` — success" in out and "`web` — success" in out


def test_green_block_states_what_it_settles(checks):
    checks["runs"] = [_cr("api", "success")]
    out = _prc.gather_ci_context("o/r", SHA)
    # The three claim classes from the observed failure.
    assert "fails to build" in out
    assert "does not exist" in out
    assert "at ANY severity" in out
    # And the honest bound — still-running checks aren't listed.
    assert "not proof the whole suite is green" in out


def test_no_checks_at_all_still_returns_none(checks):
    """Absence of CI information must stay absent, not become a green
    claim — the prompt's "never infer it passed" rule depends on it."""
    checks["runs"] = []
    assert _prc.gather_ci_context("o/r", SHA) is None


def test_only_excluded_checks_returns_none(checks):
    """cora's own verdict check and the `required` aggregator are not
    evidence about the code — a review reading its own green check is a
    feedback loop."""
    checks["runs"] = [
        _cr("cora", "success"),
        _cr("required", "success"),
        _cr("agentic-pr-review-legacy", "success"),
    ]
    assert _prc.gather_ci_context("o/r", SHA) is None


def test_mixed_state_lists_green_alongside_red(checks):
    checks["runs"] = [_cr("api", "success"), _cr("lint", "failure")]
    out = _prc.gather_ci_context("o/r", SHA)
    assert "failing checks" in out
    assert "### Passing on this commit" in out
    assert "`api`" in out


def test_pending_checks_are_not_reported_green(checks):
    checks["runs"] = [_cr("api", "success"), _cr("slow", None)]
    out = _prc.gather_ci_context("o/r", SHA)
    assert "`api` — success" in out
    assert "slow" not in out


def test_rerun_fold_keeps_the_latest_conclusion(checks):
    """A check that failed then passed on re-run is green."""
    checks["runs"] = [
        _cr("api", "failure", "2026-08-09T09:00:00Z"),
        _cr("api", "success", "2026-08-09T10:00:00Z"),
    ]
    out = _prc.gather_ci_context("o/r", SHA)
    assert "all reported checks passing" in out


def test_killswitch_restores_the_previous_behaviour(checks):
    cfg = ReviewerConfig(ci_context_include_passing=False)
    checks["runs"] = [_cr("api", "success")]
    assert _prc.gather_ci_context("o/r", SHA, cfg=cfg) is None


def test_env_killswitch(monkeypatch):
    assert ReviewerConfig().ci_context_include_passing is True
    monkeypatch.setenv("AGENT_REVIEW_CI_CONTEXT_PASSING", "false")
    monkeypatch.setenv("GH_REPO", "o/r")
    monkeypatch.setenv("PR_NUMBER", "1")
    assert ReviewerConfig.from_env().ci_context_include_passing is False


def test_fetch_failure_still_soft_fails(monkeypatch, capsys):
    def boom(cmd, **k):
        raise RuntimeError("gh down")

    monkeypatch.setattr(_prc, "run", boom)
    assert _prc.gather_ci_context("o/r", SHA) is None
    assert "::warning::" in capsys.readouterr().out


# ── prompt-side: lookup-first, and no severity-laundering ────────────


@pytest.mark.parametrize("mode", ["deep", "quick"])
def test_prompts_make_green_ci_settle_claims_at_any_severity(mode):
    """Downgrading an unverifiable claim to ⚠️ was treated as compliance
    with the old rule, which only capped severity. The observed review
    did exactly that and was still wrong three times."""
    text = load_system_prompt(None, mode=mode)
    assert "every severity" in text


def test_deep_prompt_orders_evidence_above_recall():
    text = load_system_prompt(None, mode="deep")
    assert "Look it up, don't remember it" in text
    # The hedge that shipped the bad findings, named and banned.
    assert "recommend\n  confirming" in text or "recommend confirming" in text
    assert "wearing a smaller badge" in text


def test_deep_prompt_keeps_the_silence_asymmetry():
    """Feeding green checks must not erode the other half: no CI section
    still means unknown, never "it passed"."""
    text = load_system_prompt(None, mode="deep")
    assert 'never infer "it passed" from silence' in text
