"""ReviewResult + Reporter seam: Null is side-effect-free, GitHubReporter
wraps the engine's check_run / summary / comment helpers, and from_config
picks GitHub vs Null on review identity."""

from __future__ import annotations

import os

import pytest

from cora.config import ReviewerConfig
from cora.core.budget import Budget
from cora.providers import GitHubReporter, NullReporter, Reporter
from cora.result import ReviewResult


def _budget() -> Budget:
    return Budget(max_input=0, max_output=0, max_iterations=0)


def _result() -> ReviewResult:
    return ReviewResult(
        verdict="looks good",
        verdict_line="🟢 looks good",
        conclusion="success",
        body="the review body",
        mode="deep",
        budget=_budget(),
        wall_time_s=1.0,
    )


def test_review_result_minimal_construction():
    r = _result()
    assert r.verdict == "looks good"
    assert r.tiers_run == []
    assert r.pause_automerge is False
    assert r.retrieval_source == "none"


def test_from_config_github_when_identity_present():
    rep = Reporter.from_config(ReviewerConfig(repo="o/r", pr_number="7", model="m"))
    assert isinstance(rep, GitHubReporter)
    assert (rep.repo, rep.pr_number, rep.model) == ("o/r", "7", "m")


def test_from_config_null_without_identity():
    assert isinstance(Reporter.from_config(ReviewerConfig()), NullReporter)


def test_null_reporter_is_side_effect_free():
    n = NullReporter()
    assert n.open_progress("sha") is None
    assert (
        n.complete_check(
            verdict_line="x",
            conclusion="success",
            budget=_budget(),
            wall_time_s=0.0,
            terminated_reason=None,
        )
        is None
    )
    assert n.post_review(_result()) is None
    assert n.post_skip("nope") is None
    assert n.pause_automerge() is False
    assert n.check_open is False
    assert n.post_in_progress() is None
    assert (
        n.write_summary(
            body="b",
            budget=_budget(),
            wall_time_s=0.0,
            terminated_reason=None,
            is_leak=False,
            tools_available=[],
        )
        is None
    )
    assert n.apply_label("escalation") == (True, None)


def test_null_dispatch_patch_returns_zero_outcome():
    out = NullReporter().dispatch_patch(
        directive={"edits": []},
        diff_text="",
        base_ref="main",
        head_sha="sha",
    )
    assert out == {
        "inline_url": None,
        "inline_count": 0,
        "draft_url": None,
        "draft_count": 0,
        "source_branch_commit_sha": None,
        "source_branch_count": 0,
        "rejected": [],
        "error": None,
    }
    # Fresh dict per call — a caller mutating one outcome must not
    # poison the next dispatch.
    assert out is not NullReporter().dispatch_patch(
        directive={}, diff_text="", base_ref="main", head_sha="s"
    )


def test_github_open_progress_creates_and_stores_check(monkeypatch):
    seen = {}

    def fake_create(repo, pr, sha, *, check_run_name):
        seen["args"] = (repo, pr, sha)
        seen["check_run_name"] = check_run_name
        return ("CID", "url")

    monkeypatch.setattr("cora.core.check_run.create_check_run", fake_create)
    rep = GitHubReporter("o/r", "7")
    rep.open_progress("abc123")
    assert seen["args"] == ("o/r", "7", "abc123")
    # No explicit name → the engine's default gate name.
    assert seen["check_run_name"] == "cora"
    assert rep._check_id == "CID"


def test_github_open_progress_skips_without_head_sha(monkeypatch):
    monkeypatch.setattr(
        "cora.core.check_run.create_check_run",
        lambda *a, **k: pytest.fail("create_check_run should not be called"),
    )
    rep = GitHubReporter("o/r", "7")
    rep.open_progress("")
    assert rep._check_id is None


def test_github_complete_check_noop_without_open(monkeypatch):
    monkeypatch.setattr(
        "cora.core.check_run.update_check_run_completed",
        lambda *a, **k: pytest.fail("update should not be called with no check"),
    )
    GitHubReporter("o/r", "7").complete_check(
        verdict_line="🟢 looks good",
        conclusion="success",
        budget=_budget(),
        wall_time_s=1.0,
        terminated_reason=None,
    )


def test_github_complete_check_calls_engine(monkeypatch):
    seen = {}
    monkeypatch.setattr("cora.core.check_run.create_check_run", lambda *a, **k: ("CID", "u"))
    monkeypatch.setattr(
        "cora.core.check_run.update_check_run_completed",
        lambda repo, cid, pr, **kw: seen.update({"cid": cid, **kw}),
    )
    rep = GitHubReporter("o/r", "7")
    rep.open_progress("sha")
    rep.complete_check(
        verdict_line="🟢 looks good",
        conclusion="success",
        budget=_budget(),
        wall_time_s=2.0,
        terminated_reason=None,
    )
    assert seen["cid"] == "CID"
    assert seen["conclusion"] == "success"


def test_github_post_review_renders_then_posts(monkeypatch):
    posted = {}
    minimized = []
    monkeypatch.setattr("cora.core.summary.make_review_comment", lambda *a, **k: "RENDERED")
    monkeypatch.setattr(
        "cora.core.comment.update_run_comment",
        lambda repo, pr, body, *, final: posted.update(
            {"repo": repo, "pr": pr, "body": body, "final": final}
        ),
    )
    monkeypatch.setattr(
        "cora.core.comment.minimize_superseded_comments",
        lambda repo, pr: minimized.append((repo, pr)),
    )
    GitHubReporter("o/r", "7", model="m").post_review(_result())
    assert posted == {"repo": "o/r", "pr": "7", "body": "RENDERED", "final": True}
    # Collapse of older comments runs AFTER this run's own is finalised.
    assert minimized == [("o/r", "7")]


def test_github_post_review_default_off_uses_comment_not_review(monkeypatch):
    # Default-OFF: the comment path is taken and the Review
    # API is never touched. This is the zero-behaviour-change guard.
    posted = {}
    monkeypatch.setattr("cora.core.summary.make_review_comment", lambda *a, **k: "RENDERED")
    monkeypatch.setattr(
        "cora.core.comment.update_run_comment",
        lambda repo, pr, body, *, final: posted.update({"repo": repo, "pr": pr, "body": body}),
    )
    monkeypatch.setattr("cora.core.comment.minimize_superseded_comments", lambda repo, pr: None)
    monkeypatch.setattr(
        "cora.core.comment.create_pr_review",
        lambda *a, **k: pytest.fail("create_pr_review must not run when default-off"),
    )
    GitHubReporter("o/r", "7", model="m").post_review(_result())
    assert posted == {"repo": "o/r", "pr": "7", "body": "RENDERED"}


def test_github_post_review_files_first_class_review_when_enabled(monkeypatch):
    # Opt-in: post a first-class PR Review instead of a comment.
    reviewed = {}
    monkeypatch.setattr("cora.core.summary.make_review_comment", lambda *a, **k: "RENDERED")
    monkeypatch.setattr(
        "cora.core.comment.update_run_comment",
        lambda *a, **k: pytest.fail("update_run_comment must not run when review-on"),
    )
    monkeypatch.setattr(
        "cora.core.comment.minimize_superseded_comments",
        lambda *a, **k: pytest.fail("minimize_superseded_comments must not run when review-on"),
    )
    monkeypatch.setattr(
        "cora.core.comment.create_pr_review",
        lambda repo, pr, body, event: reviewed.update(
            {"repo": repo, "pr": pr, "body": body, "event": event}
        ),
    )
    rep = GitHubReporter("o/r", "7", model="m", use_github_review=True)
    rep.post_review(_result())  # _result() verdict is "looks good"
    # looks good → COMMENT (never a bot APPROVE).
    assert reviewed == {
        "repo": "o/r", "pr": "7", "body": "RENDERED", "event": "COMMENT",
    }


def test_github_post_review_maps_block_verdict_to_request_changes(monkeypatch):
    reviewed = {}
    monkeypatch.setattr("cora.core.summary.make_review_comment", lambda *a, **k: "BODY")
    monkeypatch.setattr(
        "cora.core.comment.create_pr_review",
        lambda repo, pr, body, event: reviewed.update({"event": event}),
    )
    r = ReviewResult(
        verdict="needs changes",
        verdict_line="🔴 needs changes",
        conclusion="failure",
        body="b",
        mode="deep",
        budget=_budget(),
        wall_time_s=1.0,
    )
    GitHubReporter("o/r", "7", use_github_review=True).post_review(r)
    assert reviewed["event"] == "REQUEST_CHANGES"


def test_github_post_review_exposes_app_token_to_review_helper(monkeypatch):
    seen = {}
    monkeypatch.setattr("cora.core.summary.make_review_comment", lambda *a, **k: "BODY")

    def fake_review(repo, pr, body, event):
        seen["app_token"] = os.environ.get("CORA_GH_TOKEN")

    monkeypatch.delenv("CORA_GH_TOKEN", raising=False)
    monkeypatch.setattr("cora.core.comment.create_pr_review", fake_review)
    GitHubReporter(
        "o/r", "7", use_github_review=True, github_app_token="TOK"
    ).post_review(_result())
    assert seen["app_token"] == "TOK"
    # Restored after the call (the contextmanager unwinds).
    assert os.environ.get("CORA_GH_TOKEN") is None


def test_from_config_threads_use_github_review_and_words():
    cfg = ReviewerConfig(repo="o/r", pr_number="7", use_github_review=True)
    rep = Reporter.from_config(cfg)
    assert isinstance(rep, GitHubReporter)
    assert rep.use_github_review is True
    assert rep.verdict_words is cfg.verdict_words
    # Default-off config leaves the reporter on the comment path.
    off = Reporter.from_config(ReviewerConfig(repo="o/r", pr_number="7"))
    assert off.use_github_review is False


def test_github_post_skip_renders_then_posts(monkeypatch):
    posted = {}
    minimized = []
    monkeypatch.setattr("cora.core.comment.make_skip_comment", lambda reason: f"SKIP:{reason}")
    monkeypatch.setattr(
        "cora.core.comment.update_run_comment",
        lambda repo, pr, body, *, final: posted.update({"body": body, "final": final}),
    )
    monkeypatch.setattr(
        "cora.core.comment.minimize_superseded_comments",
        lambda repo, pr: minimized.append((repo, pr)),
    )
    GitHubReporter("o/r", "7").post_skip("no key")
    assert posted == {"body": "SKIP:no key", "final": True}
    # A skip is a completed run — same collapse-the-rest treatment as a verdict.
    assert minimized == [("o/r", "7")]


def test_github_pause_automerge_delegates(monkeypatch):
    monkeypatch.setattr("cora.core.comment.remove_automerge_label", lambda repo, pr: True)
    assert GitHubReporter("o/r", "7").pause_automerge() is True


def test_github_check_open_lifecycle(monkeypatch):
    monkeypatch.setattr("cora.core.check_run.create_check_run", lambda *a, **k: ("CID", "u"))
    monkeypatch.setattr(
        "cora.core.check_run.update_check_run_completed", lambda *a, **k: None
    )
    rep = GitHubReporter("o/r", "7")
    assert rep.check_open is False
    rep.open_progress("sha")
    assert rep.check_open is True
    rep.complete_check(verdict_line="x", conclusion="success")
    assert rep.check_open is False


def test_github_complete_check_first_terminal_wins(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("cora.core.check_run.create_check_run", lambda *a, **k: ("CID", "u"))
    monkeypatch.setattr(
        "cora.core.check_run.update_check_run_completed",
        lambda repo, cid, pr, **kw: calls.append(kw["conclusion"]),
    )
    rep = GitHubReporter("o/r", "7")
    rep.open_progress("sha")
    rep.complete_check(verdict_line="verdict: minor", conclusion="neutral")
    # Late SIGTERM-style finalize must no-op, not clobber the verdict.
    rep.complete_check(verdict_line="timed out", conclusion="timed_out")
    assert calls == ["neutral"]


def test_github_complete_check_defaults_zero_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr("cora.core.check_run.create_check_run", lambda *a, **k: ("CID", "u"))
    monkeypatch.setattr(
        "cora.core.check_run.update_check_run_completed",
        lambda repo, cid, pr, **kw: seen.update(kw),
    )
    rep = GitHubReporter("o/r", "7")
    rep.open_progress("sha")
    rep.complete_check(verdict_line="skipped (preflight)", conclusion="cancelled")
    b = seen["budget"]
    assert (b.input_used, b.output_used, b.iterations) == (0, 0, 0)
    assert seen["wall_time_s"] == 0.0
    assert seen["terminated_reason"] is None


def test_github_post_in_progress_renders_then_posts(monkeypatch):
    from datetime import datetime, timezone

    posted = {}
    seen = {}

    def fake_initial(pr, started_at):
        seen["args"] = (pr, started_at)
        return "INITIAL"

    monkeypatch.setattr("cora.core.comment.make_initial_comment", fake_initial)
    monkeypatch.setattr(
        "cora.core.comment.create_progress_comment",
        lambda repo, pr, body: posted.update({"repo": repo, "pr": pr, "body": body}),
    )
    started = datetime(2026, 6, 1, tzinfo=timezone.utc)
    GitHubReporter("o/r", "7", started_at=started).post_in_progress()
    assert seen["args"] == ("7", started)
    assert posted == {"repo": "o/r", "pr": "7", "body": "INITIAL"}


def test_github_write_summary_delegates_with_context(monkeypatch):
    seen = {}

    def fake_summary(pr, model, budget, wall, reason, body, leak, tools):
        seen.update(
            pr=pr, model=model, wall=wall, reason=reason,
            body=body, leak=leak, tools=tools,
        )

    monkeypatch.setattr("cora.core.summary.write_step_summary", fake_summary)
    GitHubReporter("o/r", "7", model="m").write_summary(
        body="BODY",
        budget=_budget(),
        wall_time_s=3.0,
        terminated_reason="wall_time",
        is_leak=True,
        tools_available=["grep_repo"],
    )
    assert seen == {
        "pr": "7", "model": "m", "wall": 3.0, "reason": "wall_time",
        "body": "BODY", "leak": True, "tools": ["grep_repo"],
    }


def test_github_post_skip_exposes_configured_app_token_to_comment_helper(monkeypatch):
    seen = {}

    def fake_post(repo, pr, body, *, final):
        seen.update(
            repo=repo,
            pr=pr,
            body=body,
            final=final,
            app_token=os.environ.get("CORA_GH_TOKEN"),
        )

    monkeypatch.delenv("CORA_GH_TOKEN", raising=False)
    monkeypatch.setattr("cora.core.comment.update_run_comment", fake_post)
    monkeypatch.setattr("cora.core.comment.minimize_superseded_comments", lambda repo, pr: None)
    GitHubReporter("o/r", "7", github_app_token="TOK").post_skip("no verdict")

    assert seen["repo"] == "o/r"
    assert seen["pr"] == "7"
    assert "no verdict" in seen["body"]
    assert seen["app_token"] == "TOK"
    assert os.environ.get("CORA_GH_TOKEN") is None


def test_github_dispatch_patch_supplies_identity_and_token(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "cora.core.patch_dispatch.apply_propose_patch_dispatch",
        lambda **kw: (seen.update(kw), {"inline_count": 1})[1],
    )
    rep = GitHubReporter("o/r", "7", github_app_token="TOK")
    out = rep.dispatch_patch(
        directive={"edits": [1]},
        diff_text="DIFF",
        base_ref="main",
        head_sha="HEAD",
        head_ref="feat/x",
        is_bot_author_pr=True,
        is_fork_pr=False,
        suppress_other_file_edits=True,
    )
    assert out == {"inline_count": 1}
    assert seen["repo"] == "o/r" and seen["pr_number"] == "7"
    assert seen["gh_token"] == "TOK"
    assert seen["base_ref"] == "main" and seen["head_sha"] == "HEAD"
    assert seen["head_ref"] == "feat/x"
    assert seen["is_bot_author_pr"] is True and seen["is_fork_pr"] is False
    assert seen["suppress_other_file_edits"] is True


def test_github_dispatch_patch_token_falls_back_to_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "cora.core.patch_dispatch.apply_propose_patch_dispatch",
        lambda **kw: (seen.update(kw), {})[1],
    )
    monkeypatch.setenv("CORA_GH_TOKEN", "ENVTOK")
    GitHubReporter("o/r", "7").dispatch_patch(
        directive={}, diff_text="", base_ref="main", head_sha="s"
    )
    assert seen["gh_token"] == "ENVTOK"


def test_github_apply_label_defaults_to_reviewed_pr(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "cora.core.patch_escalation.apply_label_to_pr",
        lambda **kw: (seen.update(kw), (True, None))[1],
    )
    rep = GitHubReporter("o/r", "7", github_app_token="TOK")
    assert rep.apply_label("escalation") == (True, None)
    assert seen == {
        "repo": "o/r", "pr_number": "7", "label": "escalation", "gh_token": "TOK",
    }
    # The escalation path also labels the draft PR it opened.
    rep.apply_label("escalation", pr_number="99")
    assert seen["pr_number"] == "99"


def test_from_config_threads_github_app_token():
    cfg = ReviewerConfig(repo="o/r", pr_number="7", github_app_token="TOK")
    rep = Reporter.from_config(cfg)
    assert isinstance(rep, GitHubReporter)
    assert rep.github_app_token == "TOK"


# ── create_pr_review helper ───────────────────────────────────────────


class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_create_pr_review_posts_to_reviews_endpoint(monkeypatch):
    import json

    from cora.core import comment as comment_mod

    seen = {}

    def fake_retry(cmd, env, *, input_=None, **kwargs):
        seen["cmd"] = cmd
        seen["payload"] = json.loads(input_)
        seen["gh_token"] = env.get("GH_TOKEN")
        return _FakeProc(returncode=0)

    monkeypatch.setenv("CORA_GH_TOKEN", "APPTOK")
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", fake_retry)
    comment_mod.create_pr_review("o/r", "7", "the body", "REQUEST_CHANGES")

    assert seen["cmd"] == [
        "gh", "api", "-X", "POST", "repos/o/r/pulls/7/reviews", "--input", "-",
    ]
    assert seen["payload"] == {"body": "the body", "event": "REQUEST_CHANGES"}
    # Prefers the cora App token so the audit actor is cora[bot].
    assert seen["gh_token"] == "APPTOK"


def test_create_pr_review_raises_on_failure(monkeypatch):
    from cora.core import comment as comment_mod

    monkeypatch.setattr(
        comment_mod, "_gh_with_one_retry",
        lambda *a, **k: _FakeProc(returncode=1, stderr="HTTP 403 forbidden"),
    )
    with pytest.raises(RuntimeError, match="create review failed"):
        comment_mod.create_pr_review("o/r", "7", "b", "COMMENT")
