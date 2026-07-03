"""Unit tests for `apply_propose_patch_dispatch` routing.

The dispatcher has two output paths:
  - in-hunk edit → inline suggestion comment on the PR review
  - out-of-hunk edit → propose-patch PR (non-draft, classifier-skip,
    base = source PR's head for same-repo, base_ref for forks)

The propose-patch PR path covers ALL out-of-hunk edits regardless of
whether the file was touched by the source PR (same-file) or not.
Minor verdict (suppress_other_file_edits=True) skips the PR and adds
to rejected[].

The human-authored same-repo case was a real observed bug — basing on
`main` made the patch PR's CI fail because the kustomization entry it
added referenced a file that only existed on the source PR's branch.
"""
from __future__ import annotations


from cora.core import patch_dispatch as pd


_EDIT = {
    "path": "k8s/apps/monitoring/extras/kustomization.yaml",
    "old_string": "- existing.yml\n",
    "new_string": "- existing.yml\n- new.yml\n",
}

_DIRECTIVE = {
    "title": "Add new.yml to extras kustomization resources",
    "body": "The file is missing from the kustomization.",
    "edits": [_EDIT],
}


def _stub_dispatch(monkeypatch):
    """Make the dispatcher's collaborators predictable: classify
    returns one out-of-hunk edit, the propose-patch path captures its
    call args, and the back-link comment soft-fails."""
    captured = {"patch_kwargs": None}

    def fake_classify(repo, head_sha, diff_text, edits):
        # One out-of-hunk edit, no inline suggestions, no rejections.
        return [], list(edits), []

    def fake_apply_propose_patch(**kwargs):
        captured["patch_kwargs"] = kwargs
        return "https://github.com/o/r/pull/999", None

    def silent_api(*args, **kwargs):
        return {}

    monkeypatch.setattr(pd, "classify_edits_for_dispatch", fake_classify)
    monkeypatch.setattr(pd, "apply_propose_patch", fake_apply_propose_patch)
    monkeypatch.setattr(pd, "_gh_api_with_token", silent_api)
    return captured


def test_human_same_repo_pr_targets_source_head_ref(monkeypatch):
    """Out-of-hunk edit, human same-repo → patch PR based on source PR's
    head branch (not main). Pins the base-ref regression fix."""
    captured = _stub_dispatch(monkeypatch)

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1567",
        base_ref="main",
        head_sha="deadbeef",
        diff_text="",
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref="feat/source-pr-branch",
        is_bot_author_pr=False,
        is_fork_pr=False,
    )

    assert outcome["error"] is None
    assert outcome["draft_url"] == "https://github.com/o/r/pull/999"
    assert captured["patch_kwargs"] is not None
    assert captured["patch_kwargs"]["base_ref"] == (
        "feat/source-pr-branch"
    ), "same-repo PR should base the patch on the source PR's head"


def test_fork_pr_falls_back_to_base_ref(monkeypatch):
    """Out-of-hunk edit, fork PR → patch PR based on base_ref (main).
    The App can't push to a fork branch."""
    captured = _stub_dispatch(monkeypatch)

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1567",
        base_ref="main",
        head_sha="deadbeef",
        diff_text="",
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref="contributor/feature",
        is_bot_author_pr=False,
        is_fork_pr=True,
    )

    assert outcome["error"] is None
    assert captured["patch_kwargs"] is not None
    assert captured["patch_kwargs"]["base_ref"] == "main", (
        "fork PR can't push to head branch — fall back to source base"
    )


def test_bot_same_repo_pr_still_opens_patch_pr(monkeypatch):
    """Out-of-hunk edit, bot same-repo → patch PR (not a direct push).
    Consistent with human PR path; classifier-skip prevents re-review."""
    captured = _stub_dispatch(monkeypatch)

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1500",
        base_ref="main",
        head_sha="deadbeef",
        diff_text="",
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref="renovate/foo-1.2.3",
        is_bot_author_pr=True,
        is_fork_pr=False,
    )

    assert outcome["error"] is None
    assert captured["patch_kwargs"] is not None, (
        "bot-authored same-repo PR should open a patch PR"
    )
    assert captured["patch_kwargs"]["base_ref"] == "renovate/foo-1.2.3", (
        "bot same-repo: patch PR should target the bot's head branch"
    )
    assert outcome["draft_count"] == 1


def test_no_head_ref_falls_back_to_base_ref(monkeypatch):
    """Defensive: if head_ref is somehow None (older call site, soft-
    failed metadata fetch), the patch PR should still open against
    `base_ref` rather than crashing."""
    captured = _stub_dispatch(monkeypatch)

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1567",
        base_ref="main",
        head_sha="deadbeef",
        diff_text="",
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref=None,
        is_bot_author_pr=False,
        is_fork_pr=False,
    )

    assert outcome["error"] is None
    assert captured["patch_kwargs"]["base_ref"] == "main"


def test_minor_verdict_suppresses_patch_pr(monkeypatch):
    """suppress_other_file_edits=True (minor verdict) skips the patch PR
    entirely — minor findings don't warrant a separate PR artifact."""
    captured = _stub_dispatch(monkeypatch)

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1900",
        base_ref="main",
        head_sha="deadbeef",
        diff_text="",
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref="feat/branch",
        is_bot_author_pr=False,
        is_fork_pr=False,
        suppress_other_file_edits=True,
    )

    assert captured["patch_kwargs"] is None, (
        "minor verdict must not open a patch PR"
    )
    assert outcome["draft_count"] == 0
    assert any("minor" in r for r in outcome["rejected"]), (
        "suppressed minor edits should appear in rejected[] with explanation"
    )


def test_all_out_of_hunk_routes_to_patch_pr(monkeypatch):
    """All out-of-hunk edits (same-file or other-file) go to one path:
    apply_propose_patch. No comment path, no push-to-source-branch."""
    captured = _stub_dispatch(monkeypatch)

    # Use a diff that includes the edit's file — previously this triggered
    # the same-file plain-comment path; now it should still go to patch PR.
    diff_with_same_file = """\
diff --git a/k8s/apps/monitoring/extras/kustomization.yaml b/k8s/apps/monitoring/extras/kustomization.yaml
--- a/k8s/apps/monitoring/extras/kustomization.yaml
+++ b/k8s/apps/monitoring/extras/kustomization.yaml
@@ -1,3 +1,4 @@
 resources:
+- other.yml
 - existing.yml
"""

    outcome = pd.apply_propose_patch_dispatch(
        repo="o/r",
        pr_number="1900",
        base_ref="main",
        head_sha="deadbeef",
        diff_text=diff_with_same_file,
        directive=_DIRECTIVE,
        gh_token="ghs_token",
        head_ref="feat/power-dash",
        is_bot_author_pr=False,
        is_fork_pr=False,
    )

    assert outcome["error"] is None
    assert captured["patch_kwargs"] is not None, (
        "same-file out-of-hunk edit must still go to patch PR"
    )
    assert outcome["draft_count"] == 1
