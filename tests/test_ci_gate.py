"""Tests for the finalize-time CI-verdict gate (`cora.review._ci_gate`).

Two layers:
  - Pure-function tests on the helpers (claim regex, check-run
    filtering/green-ness, annotation, downgrade) — no ReviewRun needed.
  - End-to-end tests through `run_review`, monkeypatching
    `_ci_gate._fetch_check_runs` so no subprocess/network runs, mirroring
    `test_run_review.py`'s pattern for the rest of the pipeline.
"""

from __future__ import annotations

import cora.core.pr_context as prc_mod
import cora.core.pretrigger as pretrigger_mod
import cora.review as review_mod
import cora.review._ci_gate as ci_gate_mod
from cora.config import ReviewerConfig
from cora.providers.reporter import Reporter
from cora.providers.retrieval import NullRetrievalProvider
from cora.review import run_review

# ---- Pure-function tests -------------------------------------------


def test_claim_regex_matches_documented_phrasings():
    for phrase in (
        "this won't compile",
        "raises a compile error",
        "fails in CI",
        "the tests will fail",
        "undefined symbol at link time",
    ):
        assert ci_gate_mod._CI_CONTRADICTION_CLAIM_RE.search(phrase), phrase


def test_claim_regex_does_not_match_generic_severity_words():
    """Narrow on purpose — generic "this is wrong/broken" phrasing must
    NOT match, or a legitimate blocker could get annotated as
    CI-contradicted."""
    for phrase in (
        "this is a security issue",
        "this leaks a secret",
        "this could be cleaner",
        "the logic here is wrong",
    ):
        assert not ci_gate_mod._CI_CONTRADICTION_CLAIM_RE.search(phrase), phrase


def test_relevant_check_runs_excludes_own_check_and_required():
    check_runs = [
        {"name": "cora", "status": "completed", "conclusion": "success"},
        {"name": "required", "status": "completed", "conclusion": "success"},
        {"name": "agentic-pr-review-legacy", "status": "completed", "conclusion": "success"},
        {"name": "build", "status": "completed", "conclusion": "success"},
    ]
    relevant = ci_gate_mod._relevant_check_runs(check_runs, own_check="cora")
    assert [cr["name"] for cr in relevant] == ["build"]


def test_relevant_check_runs_keeps_latest_per_name():
    check_runs = [
        {"name": "build", "status": "completed", "conclusion": "failure", "started_at": "2026-01-01T00:00:00Z"},
        {"name": "build", "status": "completed", "conclusion": "success", "started_at": "2026-01-01T01:00:00Z"},
    ]
    relevant = ci_gate_mod._relevant_check_runs(check_runs, own_check="cora")
    assert len(relevant) == 1
    assert relevant[0]["conclusion"] == "success"


def test_all_green_false_when_empty():
    """No relevant checks is NOT evidence of green — the gate must not
    annotate on silence."""
    assert ci_gate_mod._all_green([]) is False


def test_all_green_false_when_one_still_running():
    checks = [
        {"status": "completed", "conclusion": "success"},
        {"status": "in_progress", "conclusion": None},
    ]
    assert ci_gate_mod._all_green(checks) is False


def test_all_green_false_when_one_failing():
    checks = [
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "failure"},
    ]
    assert ci_gate_mod._all_green(checks) is False


def test_all_green_true_only_when_every_check_actually_succeeded():
    checks = [
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "success"},
    ]
    assert ci_gate_mod._all_green(checks) is True


def test_skipped_or_neutral_is_not_evidence_the_build_passed():
    """A job that never ran can't contradict a compile claim. `skipped`
    is the common case — a path filter or `if:` gate on the build job —
    and treating it as green would let the gate suppress a true blocker
    on the strength of a build that was never attempted."""
    for conclusion in ("skipped", "neutral", "stale", None):
        checks = [
            {"status": "completed", "conclusion": "success"},
            {"status": "completed", "conclusion": conclusion},
        ]
        assert ci_gate_mod._all_green(checks) is False, conclusion


def test_annotate_contradicted_blockers_appends_note_only_to_matches():
    body = (
        "🔴 needs changes\n\n"
        "Summary.\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** foo.py:10 this won't compile due to a typo\n"
        "- 🚨 **Blocker:** bar.py:4 SQL injection via unsanitised input\n"
    )
    new_body, total, contradicted = ci_gate_mod._annotate_contradicted_blockers(body)
    assert total == 2
    assert contradicted == 1
    assert "harness note" in new_body
    assert new_body.count("harness note") == 1
    # The unrelated (SQL injection) blocker is untouched.
    lines = new_body.splitlines()
    sql_line = next(ln for ln in lines if "SQL injection" in ln)
    assert "harness note" not in sql_line


def test_annotate_contradicted_blockers_no_blockers_is_noop():
    body = "🔴 needs changes\n\nSummary only, no findings section.\n"
    new_body, total, contradicted = ci_gate_mod._annotate_contradicted_blockers(body)
    assert new_body == body
    assert total == 0
    assert contradicted == 0


def test_downgrade_verdict_swaps_glyph_and_word_and_explains():
    body = "🔴 needs changes\n\nSummary.\n\n**Findings:**\n- 🚨 **Blocker:** x\n"
    new_body = ci_gate_mod._downgrade_verdict(
        body,
        glyphs=("🟢", "🟡", "🔴"),
        words=("looks good", "minor", "needs changes"),
    )
    assert new_body.startswith("🟡 minor")
    assert "downgraded from 🔴 needs changes" in new_body
    assert "🚨 **Blocker:** x" in new_body  # finding preserved, not deleted


# ---- End-to-end via run_review --------------------------------------


def _metadata() -> dict:
    return {
        "title": "Fix the thing",
        "body": "A change.",
        "labels": [],
        "author": {"login": "alice", "is_bot": False},
        "baseRefName": "main",
        "headRefName": "fix/x",
        "isCrossRepository": False,
        "additions": 3,
        "deletions": 1,
        "changedFiles": 1,
    }


def _cfg(**overrides) -> ReviewerConfig:
    base = {
        "repo": "owner/repo",
        "pr_number": "42",
        "llm_api_key": "test-key",
        "model": "test-model",
        "max_tool_iterations": 0,  # quick mode — simplest path to a body
    }
    base.update(overrides)
    return ReviewerConfig(**base)


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        self.complete_calls: list[dict] = []
        self.reviews: list = []
        self.skips: list[str] = []
        self._open = False

    def open_progress(self, head_sha: str) -> None:
        self._open = True

    @property
    def check_open(self) -> bool:
        return self._open

    def post_in_progress(self) -> None:
        pass

    def complete_check(self, *, verdict_line, conclusion, budget=None, wall_time_s=0.0, terminated_reason=None) -> None:
        if not self._open:
            return
        self._open = False
        self.complete_calls.append({"verdict_line": verdict_line, "conclusion": conclusion})

    def write_summary(self, **kwargs) -> None:
        pass

    def post_review(self, result) -> None:
        self.reviews.append(result)

    def post_skip(self, reason: str) -> None:
        self.skips.append(reason)

    def pause_automerge(self) -> bool:
        return True

    def dispatch_patch(self, **kwargs) -> dict:
        raise AssertionError("no propose_patch directive in these fixtures")

    def apply_label(self, label, *, pr_number=None):
        return True, None


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr(prc_mod, "fetch_pr_metadata", lambda pr: _metadata())
    monkeypatch.setattr(prc_mod, "fetch_pr_diff", lambda pr: "diff --git a/f b/f\n+x\n")
    monkeypatch.setattr(prc_mod, "gather_ci_context", lambda *a, **k: None)
    monkeypatch.setattr(prc_mod, "fetch_classifier_rationale", lambda *a, **k: None)
    monkeypatch.setattr(prc_mod, "_pr_head_sha", lambda: "headsha123")
    monkeypatch.setattr(prc_mod, "fetch_author_association", lambda *a, **k: "MEMBER")
    monkeypatch.setattr(prc_mod, "latest_commit_author_login", lambda *a, **k: None)

    async def _no_pretrigger(*a, **k):
        return None

    monkeypatch.setattr(pretrigger_mod, "fire_pretrigger", _no_pretrigger)

    for var in (
        "AGENT_REVIEW_T1_CONTINUATION",
        "AGENT_REVIEW_SKIP_T0",
        "AGENT_REVIEW_T2_DISAGREEMENT",
        "CLASSIFIER_LABEL",
        "WALL_TIME_S",
        "T0_WALL_TIME_S",
        "T1_WALL_TIME_S",
        "T1_MODEL",
        "GITHUB_EVENT_PATH",
        "GITHUB_STEP_SUMMARY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)

    review_mod._TIMEOUT_GUARD.update(
        reporter=None, budget=None, start=None, terminated_reason=None
    )


_BLOCKER_BODY = (
    "🔴 needs changes\n\n"
    "This change breaks the build.\n\n"
    "**Findings:**\n"
    "- 🚨 **Blocker:** main.py:12 this won't compile — undefined symbol `foo`\n"
)

_MIXED_BLOCKER_BODY = (
    "🔴 needs changes\n\n"
    "Two issues here.\n\n"
    "**Findings:**\n"
    "- 🚨 **Blocker:** main.py:12 this won't compile\n"
    "- 🚨 **Blocker:** auth.py:5 missing authorization check on delete\n"
)

_ALL_GREEN_CHECK_RUNS = [
    {"name": "build", "status": "completed", "conclusion": "success"},
]


def test_gate_downgrades_when_all_blockers_contradicted_and_ci_green(monkeypatch):
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(
        ci_gate_mod, "_fetch_check_runs", lambda repo, sha: _ALL_GREEN_CHECK_RUNS
    )

    rep = RecordingReporter()
    result = run_review(_cfg(), reporter=rep, retrieval=NullRetrievalProvider())

    assert result.verdict == "minor"
    assert result.conclusion == "neutral"
    assert result.body.startswith("🟡 minor")
    assert "downgraded from 🔴 needs changes" in result.body
    assert "harness note" in result.body
    assert "🚨 **Blocker:**" in result.body  # finding preserved
    assert rep.reviews == [result]


def test_gate_annotates_but_does_not_downgrade_mixed_blockers(monkeypatch):
    """A CI-contradicted claim alongside an unrelated real finding gets
    annotated, but the verdict stays `needs changes` — one contradicted
    claim doesn't clear an independent blocker."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _MIXED_BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(
        ci_gate_mod, "_fetch_check_runs", lambda repo, sha: _ALL_GREEN_CHECK_RUNS
    )

    rep = RecordingReporter()
    result = run_review(_cfg(), reporter=rep, retrieval=NullRetrievalProvider())

    assert result.verdict == "needs changes"
    assert result.conclusion == "failure"
    assert result.body.startswith("🔴 needs changes")
    assert "harness note" in result.body
    lines = result.body.splitlines()
    compile_line = next(ln for ln in lines if "won't compile" in ln)
    auth_line = next(ln for ln in lines if "authorization" in ln)
    assert "harness note" in compile_line
    assert "harness note" not in auth_line


def test_gate_noop_when_ci_not_all_green(monkeypatch):
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(
        ci_gate_mod,
        "_fetch_check_runs",
        lambda repo, sha: [{"name": "build", "status": "in_progress", "conclusion": None}],
    )

    result = run_review(_cfg(), reporter=RecordingReporter(), retrieval=NullRetrievalProvider())

    assert result.verdict == "needs changes"
    assert result.body == _BLOCKER_BODY.strip()


def test_gate_soft_fails_on_api_error(monkeypatch):
    """`_fetch_check_runs` returning None (any `gh api` failure) leaves
    the review exactly as the model produced it."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(ci_gate_mod, "_fetch_check_runs", lambda repo, sha: None)

    result = run_review(_cfg(), reporter=RecordingReporter(), retrieval=NullRetrievalProvider())

    assert result.verdict == "needs changes"
    assert result.body == _BLOCKER_BODY.strip()


def test_gate_noop_when_verdict_is_not_needs_changes(monkeypatch):
    """The gate only ever fires on the block-severity verdict — a
    `minor`/`looks good` verdict is left alone even with a fetch stub
    that would otherwise report all-green."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    body = "🟡 minor\n\nA nit.\n\n**Findings:**\n- ⚠️ **Concern:** x\n"

    async def fake_quick(**kwargs):
        return body, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    def _boom(repo, sha):
        raise AssertionError("must not re-poll CI when verdict isn't needs-changes")

    monkeypatch.setattr(ci_gate_mod, "_fetch_check_runs", _boom)

    result = run_review(_cfg(), reporter=RecordingReporter(), retrieval=NullRetrievalProvider())

    assert result.verdict == "minor"
    assert result.body == body.strip()


def test_gate_killswitch_disables_gate_entirely(monkeypatch):
    """`ci_verdict_gate=False` (env `AGENT_REVIEW_CI_VERDICT_GATE=false`)
    skips the gate wholesale — no re-poll at all."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    def _boom(repo, sha):
        raise AssertionError("must not re-poll CI when the gate is killswitched off")

    monkeypatch.setattr(ci_gate_mod, "_fetch_check_runs", _boom)

    cfg = _cfg(ci_verdict_gate=False)
    result = run_review(cfg, reporter=RecordingReporter(), retrieval=NullRetrievalProvider())

    assert result.verdict == "needs changes"
    assert result.body == _BLOCKER_BODY.strip()


def test_ci_verdict_gate_env_killswitch(monkeypatch):
    monkeypatch.setenv("AGENT_REVIEW_CI_VERDICT_GATE", "false")
    cfg = ReviewerConfig.from_env({"AGENT_REVIEW_CI_VERDICT_GATE": "false"})
    assert cfg.ci_verdict_gate is False
    cfg_default = ReviewerConfig.from_env({})
    assert cfg_default.ci_verdict_gate is True


def test_gate_no_relevant_checks_is_noop(monkeypatch):
    """Only the reviewer's own check + `required` present (no actual
    build/test signal) — empty after filtering, so `_all_green` is False
    and the gate leaves the review untouched."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return _BLOCKER_BODY, None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(
        ci_gate_mod,
        "_fetch_check_runs",
        lambda repo, sha: [
            {"name": "cora", "status": "completed", "conclusion": "success"},
            {"name": "required", "status": "completed", "conclusion": "success"},
        ],
    )

    result = run_review(_cfg(), reporter=RecordingReporter(), retrieval=NullRetrievalProvider())

    assert result.verdict == "needs changes"
    assert result.body == _BLOCKER_BODY.strip()


# ── the downgrade must see every blocker the automerge gate sees ──────
# `detect_blocker` (core/leak.py) matches `🚨 **Blocker:**` anywhere;
# the gate annotates only the prompt's bullet form. Counting with the
# narrow pattern let a differently-formatted REAL blocker go uncounted,
# so one contradicted bullet equalled "every blocker" and downgraded a
# review whose actual blocker was never examined.


def test_non_bullet_blocker_is_counted_so_no_downgrade():
    body = (
        "🔴 needs changes\n\n"
        "- 🚨 **Blocker:** this won't compile\n"
        "🚨 **Blocker:** auth.go:44 — token comparison is not constant-time\n"
    )
    _new, total, contradicted = ci_gate_mod._annotate_contradicted_blockers(body)
    assert total == 2, "the non-bullet blocker must still count"
    assert contradicted == 1
    assert total != contradicted, "a real blocker must block the downgrade"


def test_star_bullet_blocker_is_counted_too():
    body = (
        "🔴 needs changes\n\n"
        "- 🚨 **Blocker:** tests will fail\n"
        "* 🚨 **Blocker:** unrelated real problem\n"
    )
    _new, total, contradicted = ci_gate_mod._annotate_contradicted_blockers(body)
    assert (total, contradicted) == (2, 1)


def test_all_bullet_blockers_contradicted_still_downgrades():
    """The guard must not disarm the feature it protects."""
    body = (
        "🔴 needs changes\n\n"
        "- 🚨 **Blocker:** this won't compile\n"
        "- 🚨 **Blocker:** the tests will fail without a database\n"
    )
    _new, total, contradicted = ci_gate_mod._annotate_contradicted_blockers(body)
    assert total == contradicted == 2


# ── the reviewer's own workflow job must not block its own gate ───────


def test_own_workflow_job_is_excluded_by_run_id(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    checks = [
        {
            "name": "review",
            "status": "in_progress",
            "conclusion": None,
            "html_url": "https://github.com/o/r/actions/runs/12345/job/9",
        },
        {"name": "build", "status": "completed", "conclusion": "success"},
    ]
    relevant = ci_gate_mod._relevant_check_runs(checks, own_check="cora")
    assert [c["name"] for c in relevant] == ["build"]
    assert ci_gate_mod._all_green(relevant) is True


def test_another_runs_check_is_not_excluded(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    checks = [
        {
            "name": "build",
            "status": "in_progress",
            "conclusion": None,
            "html_url": "https://github.com/o/r/actions/runs/99999/job/1",
        },
    ]
    relevant = ci_gate_mod._relevant_check_runs(checks, own_check="cora")
    assert [c["name"] for c in relevant] == ["build"]


def test_no_run_id_env_excludes_nothing_extra(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    checks = [
        {
            "name": "build",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/o/r/actions/runs/12345/job/9",
        },
    ]
    assert len(ci_gate_mod._relevant_check_runs(checks, own_check="cora")) == 1
