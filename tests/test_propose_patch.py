"""Tests for the propose_patch path in agent_review.py.

Pure-function tests against the parser + validator + branch-name slugifier.
`apply_propose_patch` does network I/O and is not covered here — it's the
integration surface and lives behind the soft-fail boundary; lean on the
PR-side smoke test for it.

Run via:
    pytest tests/test_propose_patch.py
"""
from __future__ import annotations

import pytest

import cora.core as a   # diff parser + hunk helpers + suggestion-body formatter
import cora.core.propose_patch as p  # shared directive shape (parser / validator / slugifier)
from cora.core.pr_context import is_bot_author, is_fork_pr


# ---------------------------------------------------------------------------
# parse_propose_patch_directive
# ---------------------------------------------------------------------------

VALID_DIRECTIVE = """{
  "title": "Fix typo",
  "body": "Mechanical typo fix.",
  "edits": [
    {"path": "k8s/apps/foo/values.yml", "old_string": "a", "new_string": "b"}
  ]
}"""


def test_parse_no_block_returns_body_unchanged():
    body = "🟢 looks good\n\nSummary."
    out, directive = p.parse_propose_patch_directive(body)
    assert directive is None
    assert out == body


def test_parse_happy_path_strips_block_and_returns_directive():
    body = (
        "🟢 looks good\n\nSummary.\n\n"
        f"```json propose_patch\n{VALID_DIRECTIVE}\n```\n"
    )
    out, directive = p.parse_propose_patch_directive(body)
    assert directive is not None
    assert directive["title"] == "Fix typo"
    assert "propose_patch" not in out
    assert "🟢 looks good" in out


def test_parse_malformed_json_leaves_block_in_body():
    body = (
        "🟢 looks good\n\n"
        "```json propose_patch\n{not valid json}\n```\n"
    )
    out, directive = p.parse_propose_patch_directive(body)
    assert directive is None
    assert "{not valid json}" in out


def test_parse_non_object_json_returns_none():
    # Valid JSON but not an object — list, string, number all rejected.
    body = "```json propose_patch\n[1, 2, 3]\n```"
    out, directive = p.parse_propose_patch_directive(body)
    assert directive is None
    assert "[1, 2, 3]" in out


# ---------------------------------------------------------------------------
# validate_propose_patch — happy path + each rejection branch
# ---------------------------------------------------------------------------

def _valid_directive() -> dict:
    return {
        "title": "Fix typo",
        "body": "Mechanical typo fix.",
        "edits": [
            {
                "path": "k8s/apps/foo/values.yml",
                "old_string": "replicaCunt: 3",
                "new_string": "replicaCount: 3",
            }
        ],
    }


def test_validate_happy_path():
    assert p.validate_propose_patch(_valid_directive()) is None


def test_validate_missing_title():
    d = _valid_directive()
    d["title"] = ""
    assert "title" in p.validate_propose_patch(d).lower()


def test_validate_title_too_long():
    d = _valid_directive()
    d["title"] = "x" * (p.PROPOSE_PATCH_MAX_TITLE_CHARS + 1)
    err = p.validate_propose_patch(d)
    assert err and "title" in err


def test_validate_missing_body():
    d = _valid_directive()
    d["body"] = "   "
    assert "body" in p.validate_propose_patch(d).lower()


def test_validate_body_too_long():
    d = _valid_directive()
    d["body"] = "x" * (p.PROPOSE_PATCH_MAX_BODY_CHARS + 1)
    err = p.validate_propose_patch(d)
    assert err and "body" in err


def test_validate_empty_edits():
    d = _valid_directive()
    d["edits"] = []
    err = p.validate_propose_patch(d)
    assert err and "edits" in err


def test_validate_edits_not_list():
    d = _valid_directive()
    d["edits"] = {"path": "foo"}
    err = p.validate_propose_patch(d)
    assert err and "edits" in err


def test_validate_too_many_edits():
    d = _valid_directive()
    d["edits"] = [
        {"path": "k8s/apps/foo/v.yml", "old_string": f"a{i}", "new_string": f"b{i}"}
        for i in range(p.PROPOSE_PATCH_MAX_EDITS + 1)
    ]
    err = p.validate_propose_patch(d)
    assert err and "too many edits" in err


def test_validate_too_many_distinct_files():
    d = _valid_directive()
    d["edits"] = [
        {
            "path": f"k8s/apps/app{i}/values.yml",
            "old_string": "a",
            "new_string": "b",
        }
        for i in range(p.PROPOSE_PATCH_MAX_FILES + 1)
    ]
    err = p.validate_propose_patch(d)
    assert err and "too many distinct files" in err


def test_validate_path_outside_allowlist_rejected():
    d = _valid_directive()
    d["edits"][0]["path"] = "ansible.cfg"
    err = p.validate_propose_patch(d)
    assert err and "allowlist" in err


def test_validate_path_on_denylist_rejected():
    d = _valid_directive()
    d["edits"][0]["path"] = ".github/workflows/foo.yml"
    err = p.validate_propose_patch(d)
    assert err and "denylist" in err


def test_validate_path_traversal_rejected():
    d = _valid_directive()
    d["edits"][0]["path"] = "k8s/apps/../etc/passwd"
    err = p.validate_propose_patch(d)
    assert err and "escapes repo root" in err


def test_validate_absolute_path_rejected():
    d = _valid_directive()
    d["edits"][0]["path"] = "/etc/passwd"
    err = p.validate_propose_patch(d)
    assert err and "escapes repo root" in err


def test_validate_string_too_long():
    d = _valid_directive()
    d["edits"][0]["new_string"] = "x" * (p.PROPOSE_PATCH_MAX_STRING_CHARS + 1)
    err = p.validate_propose_patch(d)
    assert err and "new_string" in err


def test_validate_noop_edit_rejected():
    d = _valid_directive()
    d["edits"][0]["new_string"] = d["edits"][0]["old_string"]
    err = p.validate_propose_patch(d)
    assert err and "no-op" in err


def test_validate_old_string_not_a_string():
    d = _valid_directive()
    d["edits"][0]["old_string"] = 42
    err = p.validate_propose_patch(d)
    assert err and "old_string" in err


def test_validate_notes_path_accepted():
    d = _valid_directive()
    d["edits"][0]["path"] = "notes/some-notebook.md"
    assert p.validate_propose_patch(d) is None


# ---------------------------------------------------------------------------
# Reviewer policy — wider gate, ONLY `.github/` denied.
# Exercises the keyword-only policy params on validate_propose_patch.
# ---------------------------------------------------------------------------

REVIEWER_POLICY = {
    "allowed_prefixes": p.PROPOSE_PATCH_REVIEWER_ALLOWED_PREFIXES,
    "denied_prefixes": p.PROPOSE_PATCH_REVIEWER_DENIED_PREFIXES,
}


@pytest.mark.parametrize(
    "path",
    [
        "scripts/ci/agent_review.py",            # reviewer's own source
        "containers/canary/Dockerfile",          # container build context
        "inventory/group_vars/all.yml",          # Ansible inventory
        "roles/baseline/tasks/main.yml",         # Ansible role
        "playbooks/site.yml",                    # top-level playbook
        "ansible.cfg",                           # repo-root config
        "ARCHITECTURE.md",                       # repo-root doc
        "k8s/bootstrap/canary.yml",              # bootstrap Argo Application
        "k8s/apps/foo/values.yml",               # still accepted
        "notes/spike.md",                        # still accepted
    ],
)
def test_reviewer_policy_accepts_widened_paths(path):
    d = _valid_directive()
    d["edits"][0]["path"] = path
    assert p.validate_propose_patch(d, **REVIEWER_POLICY) is None


def test_reviewer_policy_still_rejects_dotgithub():
    """Workflow files run from the PR head ref → self-rewrite path stays denied."""
    d = _valid_directive()
    d["edits"][0]["path"] = ".github/workflows/cora-review.yml"
    err = p.validate_propose_patch(d, **REVIEWER_POLICY)
    assert err and "denylist" in err


def test_reviewer_policy_still_rejects_traversal():
    d = _valid_directive()
    d["edits"][0]["path"] = "scripts/../../etc/passwd"
    err = p.validate_propose_patch(d, **REVIEWER_POLICY)
    assert err and "escapes repo root" in err


def test_reviewer_policy_still_rejects_absolute():
    d = _valid_directive()
    d["edits"][0]["path"] = "/etc/passwd"
    err = p.validate_propose_patch(d, **REVIEWER_POLICY)
    assert err and "escapes repo root" in err


def test_reviewer_policy_size_caps_still_apply():
    """Policy parameters only swap the path gate; size caps are unchanged."""
    d = _valid_directive()
    d["edits"][0]["path"] = "scripts/ci/agent_review.py"
    d["edits"][0]["new_string"] = "x" * (p.PROPOSE_PATCH_MAX_STRING_CHARS + 1)
    err = p.validate_propose_patch(d, **REVIEWER_POLICY)
    assert err and "new_string" in err


def test_default_policy_unchanged_for_triage_callers():
    """Calling validate_propose_patch() with no policy args must keep the
    narrow (triage) gate intact — `scripts/` rejected, `k8s/apps/` accepted.
    This is the regression test for backward compat of the triage path.
    """
    d = _valid_directive()
    d["edits"][0]["path"] = "scripts/ci/agent_review.py"
    err = p.validate_propose_patch(d)
    assert err and "denylist" in err


# ---------------------------------------------------------------------------
# propose_patch_branch_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title, suffix, expected",
    [
        ("Fix typo", "pr-1385", "cora/pr-1385-fix-typo"),
        ("UPPERCASE Stuff!!!", "pr-42", "cora/pr-42-uppercase-stuff"),
        ("   leading and trailing   ", "pr-1", "cora/pr-1-leading-and-trailing"),
        # Empty / fully-punctuation slugs fall back to a generic suffix.
        ("!!!", "pr-100", "cora/pr-100-patch"),
        ("", "pr-100", "cora/pr-100-patch"),
        # Triage suffix shape — exercises the suffix-is-arbitrary contract.
        ("Add missing NetworkPolicy", "triage-kube-pod-not-ready",
         "cora/triage-kube-pod-not-ready-add-missing-networkpolicy"),
    ],
)
def test_branch_name_slugifies(title, suffix, expected):
    assert p.propose_patch_branch_name(title, suffix) == expected


def test_branch_name_caps_at_40_chars_of_slug():
    long_title = "this is a very long title that goes on and on and on and on"
    out = p.propose_patch_branch_name(long_title, "pr-9")
    # cora/pr-9- = 12 chars, slug ≤ 40
    slug = out.removeprefix("cora/pr-9-")
    assert 0 < len(slug) <= 40


# ---------------------------------------------------------------------------
# parse_pr_diff_hunks — unified diff → per-path new-side hunk ranges
# ---------------------------------------------------------------------------

SIMPLE_DIFF = """diff --git a/k8s/apps/foo/values.yml b/k8s/apps/foo/values.yml
index abc..def 100644
--- a/k8s/apps/foo/values.yml
+++ b/k8s/apps/foo/values.yml
@@ -10,7 +10,7 @@ spec:
       containers:
         - name: foo
-          image: foo:v1
+          image: foo:v2
           ports:
             - 8080
diff --git a/notes/foo.md b/notes/foo.md
index xyz..uvw 100644
--- a/notes/foo.md
+++ b/notes/foo.md
@@ -1,3 +1,4 @@
 # Foo
+New section.
 Body.
@@ -100,5 +101,5 @@ More
 content
-old
+new
 here
"""


def test_parse_diff_hunks_basic():
    hunks = a.parse_pr_diff_hunks(SIMPLE_DIFF)
    assert hunks["k8s/apps/foo/values.yml"] == [(10, 17)]
    assert hunks["notes/foo.md"] == [(1, 5), (101, 106)]


def test_parse_diff_hunks_handles_omitted_length():
    # `@@ -X +A @@` (no commas) is valid unified-diff for single-line hunks.
    diff = (
        "diff --git a/f.txt b/f.txt\n"
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -5 +5 @@\n"
        "-old\n"
        "+new\n"
    )
    hunks = a.parse_pr_diff_hunks(diff)
    assert hunks["f.txt"] == [(5, 6)]


def test_parse_diff_hunks_skips_deletions():
    # File deleted in PR → no new-side content → no entry.
    diff = (
        "diff --git a/gone.txt b/gone.txt\n"
        "deleted file mode 100644\n"
        "--- a/gone.txt\n"
        "+++ /dev/null\n"
        "@@ -1,3 +0,0 @@\n"
        "-line1\n"
        "-line2\n"
        "-line3\n"
    )
    hunks = a.parse_pr_diff_hunks(diff)
    assert "gone.txt" not in hunks


def test_parse_diff_hunks_empty_input():
    assert a.parse_pr_diff_hunks("") == {}


# ---------------------------------------------------------------------------
# find_line_range — 1-indexed line range of unique snippet
# ---------------------------------------------------------------------------

CONTENT_3_LINES = "line one\nline two\nline three\n"


def test_find_line_range_single_line():
    # "line two" is on line 2 (1-indexed). Range (2, 2).
    assert a.find_line_range(CONTENT_3_LINES, "line two") == (2, 2)


def test_find_line_range_multi_line_snippet():
    # Snippet spans lines 2-3. Range (2, 3).
    assert a.find_line_range(CONTENT_3_LINES, "line two\nline three") == (2, 3)


def test_find_line_range_not_found():
    assert a.find_line_range(CONTENT_3_LINES, "missing") is None


def test_find_line_range_ambiguous_returns_none():
    # Snippet appears twice → ambiguous → None.
    content = "foo\nbar\nfoo\nbaz"
    assert a.find_line_range(content, "foo") is None


def test_find_line_range_first_line():
    assert a.find_line_range(CONTENT_3_LINES, "line one") == (1, 1)


def test_find_line_range_single_line_trailing_newline():
    # Snippet "line two\n" — the trailing \n is the line terminator, so the
    # snippet still occupies only line 2. Without the trailing-newline fix
    # end_line would be 3 (2 + 1), causing a last-line-of-hunk edit to be
    # misclassified as out-of-hunk.
    assert a.find_line_range(CONTENT_3_LINES, "line two\n") == (2, 2)


def test_find_line_range_multi_line_trailing_newline():
    # Multi-line snippet with trailing newline spans lines 2-3, not 2-4.
    assert a.find_line_range(CONTENT_3_LINES, "line two\nline three\n") == (2, 3)


def test_find_line_range_trailing_newline_stays_in_hunk():
    # Regression: edit on last line of hunk [10, 20) where old_string ends
    # with \n. Before the fix end_line = 20 which fails ``< 20``, routing
    # the edit to the draft-PR path instead of inline suggestion.
    content = "".join(f"line {i}\n" for i in range(1, 25))
    snippet = "line 19\n"
    start, end = a.find_line_range(content, snippet)
    assert (start, end) == (19, 19)
    assert a._is_in_hunk(start, end, [(10, 20)]) is True


# ---------------------------------------------------------------------------
# _is_in_hunk — range fully within a single hunk
# ---------------------------------------------------------------------------

def test_is_in_hunk_inside_single_hunk():
    # Hunk [10, 20), range (12, 14) is fully inside.
    assert a._is_in_hunk(12, 14, [(10, 20)]) is True


def test_is_in_hunk_at_hunk_start():
    # range (10, 10) is at the hunk's first line.
    assert a._is_in_hunk(10, 10, [(10, 20)]) is True


def test_is_in_hunk_at_hunk_last_line():
    # Hunk [10, 20) means lines 10..19 inclusive. (19, 19) is in.
    assert a._is_in_hunk(19, 19, [(10, 20)]) is True


def test_is_in_hunk_just_past_hunk_end():
    # Line 20 is outside (h_end is exclusive).
    assert a._is_in_hunk(20, 20, [(10, 20)]) is False


def test_is_in_hunk_before_hunk():
    assert a._is_in_hunk(5, 7, [(10, 20)]) is False


def test_is_in_hunk_crossing_hunk_boundary():
    # Range crosses hunk end → False (even though most lines are in).
    assert a._is_in_hunk(18, 22, [(10, 20)]) is False


def test_is_in_hunk_multiple_hunks_first():
    assert a._is_in_hunk(3, 4, [(1, 5), (10, 20)]) is True


def test_is_in_hunk_multiple_hunks_second():
    assert a._is_in_hunk(15, 16, [(1, 5), (10, 20)]) is True


def test_is_in_hunk_multiple_hunks_between():
    # Range falls in the gap between hunks → False.
    assert a._is_in_hunk(7, 8, [(1, 5), (10, 20)]) is False


def test_is_in_hunk_no_hunks():
    assert a._is_in_hunk(5, 7, []) is False


# ---------------------------------------------------------------------------
# _format_suggestion_body — wraps content in suggestion fence
# ---------------------------------------------------------------------------

def test_format_suggestion_body_single_line():
    out = a._format_suggestion_body("foo: bar")
    assert out.startswith("```suggestion\n")
    assert "foo: bar" in out
    assert "Click **Apply suggestion**" in out


def test_format_suggestion_body_strips_one_trailing_newline():
    # The closing fence adds its own boundary; one trailing newline in
    # new_string would otherwise produce a blank line at the end of the
    # suggested block.
    out = a._format_suggestion_body("foo\n")
    assert "foo\n```" in out
    assert "foo\n\n```" not in out


def test_format_suggestion_body_preserves_internal_newlines():
    out = a._format_suggestion_body("line1\nline2\nline3")
    assert "line1\nline2\nline3\n```" in out


# ---------------------------------------------------------------------------
# is_bot_author / is_fork_pr — classification predicates driving the
# out-of-hunk routing decision (push-to-source-branch vs draft PR).
# Doesn't exercise apply_push_to_source_branch itself (network I/O).
# ---------------------------------------------------------------------------


def _bot_authored_metadata():
    """Mirror the shape `fetch_pr_metadata` returns for a bot-authored
    same-repo PR. Modelled on a Renovate PR. NOTE: `gh pr view`
    returns `headRepository.owner: null` for same-repo PRs (the owner is
    implied to be the base repo's), and `isCrossRepository: false`."""
    return {
        "title": "chore(deps): bump foo",
        "author": {"login": "renovate", "is_bot": True},
        "baseRefName": "main",
        "headRefName": "renovate/foo",
        "headRepository": {"name": "repo", "owner": None},
        "isCrossRepository": False,
    }


def _human_authored_metadata():
    return {
        "title": "feat: bar",
        "author": {"login": "someuser", "is_bot": False},
        "baseRefName": "main",
        "headRefName": "feat/bar",
        "headRepository": {"name": "repo", "owner": None},
        "isCrossRepository": False,
    }


def _fork_metadata():
    """A PR from a fork — author may be human or bot, but the head
    repository differs from the base. `gh` populates the fork owner and
    sets `isCrossRepository: true`."""
    return {
        "title": "feat: from a fork",
        "author": {"login": "contributor", "is_bot": False},
        "baseRefName": "main",
        "headRefName": "feat/from-a-fork",
        "headRepository": {"name": "repo", "owner": {"login": "contributor"}},
        "isCrossRepository": True,
    }


def test_is_bot_author_recognises_renovate(monkeypatch):
    monkeypatch.setenv("GH_REPO", "owner/repo")
    md = _bot_authored_metadata()
    assert is_bot_author(md) is True


def test_is_bot_author_human_negative(monkeypatch):
    monkeypatch.setenv("GH_REPO", "owner/repo")
    md = _human_authored_metadata()
    assert is_bot_author(md) is False


def test_is_fork_pr_same_repo_negative(monkeypatch):
    monkeypatch.setenv("GH_REPO", "owner/repo")
    md = _bot_authored_metadata()
    assert is_fork_pr(md) is False


def test_is_fork_pr_fork_positive(monkeypatch):
    monkeypatch.setenv("GH_REPO", "owner/repo")
    md = _fork_metadata()
    assert is_fork_pr(md) is True


def test_is_fork_pr_same_repo_null_owner_regression(monkeypatch):
    """Regression: `gh pr view` returns
    `headRepository.owner: null` for same-repo PRs, which made the old
    owner-equality heuristic classify them as forks and base fix-PRs off
    main. The canonical `isCrossRepository: false` must win."""
    monkeypatch.setenv("GH_REPO", "owner/repo")
    md = {
        "headRefName": "feat/some-branch",
        "headRepository": {"name": "repo", "owner": None},
        "isCrossRepository": False,
    }
    assert is_fork_pr(md) is False


def test_is_fork_pr_falls_back_to_heuristic_without_flag(monkeypatch):
    """When `isCrossRepository` is absent (older metadata), fall back to
    the head-repo identity heuristic."""
    monkeypatch.setenv("GH_REPO", "owner/repo")
    same_repo = {"headRepository": {"name": "repo", "owner": {"login": "owner"}}}
    fork = {"headRepository": {"name": "repo", "owner": {"login": "contributor"}}}
    assert is_fork_pr(same_repo) is False
    assert is_fork_pr(fork) is True
