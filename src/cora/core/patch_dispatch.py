"""propose_patch hybrid dispatch.

Parse a `propose_patch` JSON directive emitted by the reviewer at the
end of its response; validate it against the path allowlist + size
caps (validator lives in `propose_patch.py`); route each
edit individually:

  - If `old_string` resolves to a line inside one of the PR's diff hunks
    → post as an inline `suggestion` comment on a PR review. The
      operator clicks "Apply suggestion" in the diff view; GitHub
      creates the commit under their identity. No App token needed
      (workflow GITHUB_TOKEN already has `pull-requests: write`).

  - Otherwise (line outside any hunk, file untouched by the PR, file
    doesn't exist at HEAD) → bundle into a draft PR via the cora
    App-minted token. Branch + edits + draft PR opened under
    `cora[bot]`. The draft PR's base is the source PR's head
    branch for same-repo PRs (so the fix can see the source PR's
    diff — e.g. a kustomization entry referencing a file the source
    PR is adding; the draft merges into the source PR's branch),
    falling back to the source PR's base for fork PRs where the App
    can't push to the head. Operator reviews/merges separately.

Both paths can fire on the same directive — some edits as inline
suggestions, some as a draft PR. The outcome footer in the review
comment links to whichever artifact(s) the dispatcher produced.

Soft-fail throughout — invalid / missing-token / API-error all
result in a one-line outcome note appended to the review comment,
never block the comment itself.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess

from cora.core.propose_patch import propose_patch_branch_name

def parse_pr_diff_hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff (from `gh pr diff`) into per-path hunk ranges.

    Returns `{path: [(new_start_line, new_end_line_exclusive), ...]}` —
    line numbers in the NEW (post-PR) version of each file. Used by the
    propose_patch dispatcher to decide whether an edit's target line is
    inside a hunk the PR is already touching (→ inline suggestion) or
    outside (→ draft PR).

    `@@ -X,Y +A,B @@` header semantics: the new-file slice runs from
    line A (inclusive, 1-indexed) for B lines. When `B` is omitted it
    defaults to 1 per the unified-diff spec.
    """
    hunks_by_path: dict[str, list[tuple[int, int]]] = {}
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_path = line[len("+++ b/"):].strip()
            hunks_by_path.setdefault(current_path, [])
        elif line.startswith("+++ ") and "/dev/null" in line:
            # File deletion: no NEW-side content. Skip.
            current_path = None
        elif line.startswith("@@") and current_path:
            # `@@ -X,Y +A,B @@` — capture A and B (omit Y; we only need
            # the new-file range to anchor suggestions).
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or "1")
                hunks_by_path[current_path].append((start, start + length))
    return hunks_by_path


def find_line_range(content: str, snippet: str) -> tuple[int, int] | None:
    """1-indexed `(start_line, end_line)` of `snippet`'s first occurrence
    in `content`, or None if not found / appears more than once.

    Uniqueness is enforced — matches `validate_propose_patch`'s contract
    that `old_string` must unambiguously identify the target. Multiple
    occurrences → caller must reject the edit (the directive should
    have used a more distinctive snippet).

    Trailing-newline handling: a snippet ending with `\\n` terminates on
    that newline, so the last content line is `start_line + count("\\n") - 1`
    (the `\\n` is the line terminator, not the start of a new line).
    Without this correction `end_line` would be one too large, causing
    edits that land on the last line of a diff hunk to be misclassified
    as out-of-hunk (``end_line == h_end`` fails ``< h_end``).
    """
    first_idx = content.find(snippet)
    if first_idx == -1:
        return None
    if content.find(snippet, first_idx + 1) != -1:
        return None  # ambiguous
    start_line = content[:first_idx].count("\n") + 1
    # A trailing \n is the *terminator* of the last line, not the start of
    # a new one.  Subtract 1 when present so end_line = last content line.
    snippet_newlines = snippet.count("\n") - (1 if snippet.endswith("\n") else 0)
    end_line = start_line + snippet_newlines
    return start_line, end_line


def _is_in_hunk(
    start_line: int, end_line: int, hunks: list[tuple[int, int]]
) -> bool:
    """True if `[start_line, end_line]` (inclusive) is fully within a
    single hunk's `[h_start, h_end)` range. Edits that cross hunk
    boundaries fall back to the draft-PR path."""
    for h_start, h_end in hunks:
        if start_line >= h_start and end_line < h_end:
            return True
    return False


def _gh_api_with_token(
    args: list[str], gh_token: str, input_data: dict | None = None
) -> dict:
    """Run `gh api <args>` with GH_TOKEN overridden to `gh_token` so the
    call carries the cora App identity instead of the workflow's
    default GITHUB_TOKEN. Raises RuntimeError on non-zero exit.

    The surrounding pipeline already uses subprocess+gh for its GitHub
    writes, so this stays consistent with that surface rather than
    introducing a new HTTP library.
    """
    env = os.environ.copy()
    env["GH_TOKEN"] = gh_token
    cmd = ["gh", "api"] + args
    if input_data is not None:
        cmd += ["--input", "-"]
    proc = subprocess.run(
        cmd,
        env=env,
        input=json.dumps(input_data) if input_data is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Truncate stderr to keep error messages readable in PR comments.
        err = proc.stderr.strip()
        if len(err) > 300:
            err = err[:300] + "…"
        raise RuntimeError(err or "gh api call failed with no stderr")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def apply_propose_patch(
    repo: str,
    pr_number: str,
    base_ref: str,
    directive: dict,
    gh_token: str,
    *,
    body_prefix: str | None = None,
) -> tuple[str | None, str | None]:
    """Apply a validated directive — resolve base SHA, create branch,
    apply edits via the Contents API (one PUT per file), open a draft PR.

    Returns `(draft_pr_url, error_message)` — exactly one non-None.

    `body_prefix`, when set, is prepended to the draft PR body above the
    directive's own body — used by the patch-escalation flow to surface
    a prominent warning ("⚠️ Escalation disagreement …") at the top of
    the PR conversation so reviewers see it before the patch summary.

    Soft-fails on any API error: never raises out. If a partial state
    is left behind (branch created, some files committed, draft PR
    didn't open), the branch URL is in the error message so the
    operator can inspect / clean up.
    """
    title = directive["title"].strip()
    body = directive["body"].strip()
    edits = directive["edits"]
    branch_name = propose_patch_branch_name(title, f"pr-{pr_number}")

    # 1. Resolve base ref → SHA. `gh pr view` doesn't return the SHA in
    #    fetch_pr_metadata above, so we look it up directly.
    try:
        resp = _gh_api_with_token(
            ["-X", "GET", f"repos/{repo}/git/refs/heads/{base_ref}"],
            gh_token,
        )
        base_sha = ((resp or {}).get("object") or {}).get("sha", "")
        if not base_sha:
            return None, f"could not resolve base ref `{base_ref}`"
    except Exception as exc:  # noqa: BLE001 — soft-fail
        return None, f"base ref lookup failed: {exc}"

    # 2. Create the branch. If it already exists from a prior run on the
    #    same PR, re-use it — re-running the same directive should land
    #    a clean new draft PR (or no-op if the patch is already there).
    try:
        _gh_api_with_token(
            ["-X", "POST", f"repos/{repo}/git/refs"],
            gh_token,
            input_data={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail
        if "Reference already exists" not in str(exc):
            return None, f"branch create failed: {exc}"

    # 3. Group edits by path, apply each file's edits sequentially. Fetch
    #    each file from the BRANCH (not base) so re-runs after a prior
    #    partial apply see the in-flight commits.
    edits_by_path: dict[str, list[dict]] = {}
    for edit in edits:
        edits_by_path.setdefault(edit["path"].strip(), []).append(edit)

    for path, file_edits in edits_by_path.items():
        try:
            file_resp = _gh_api_with_token(
                ["-X", "GET", f"repos/{repo}/contents/{path}?ref={branch_name}"],
                gh_token,
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"could not read `{path}` on `{branch_name}`: {exc}"
        if (file_resp or {}).get("type") != "file":
            return None, f"`{path}` is not a regular file"
        try:
            current = base64.b64decode(file_resp["content"]).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"could not decode `{path}`: {exc}"
        prev_sha = file_resp["sha"]

        new_content = current
        for i, edit in enumerate(file_edits):
            old_s = edit["old_string"]
            new_s = edit["new_string"]
            count = new_content.count(old_s)
            if count == 0:
                return None, f"edit on `{path}` #{i}: `old_string` not found"
            if count > 1:
                return None, (
                    f"edit on `{path}` #{i}: `old_string` matches {count} times "
                    f"(must be unique — narrow the snippet)"
                )
            new_content = new_content.replace(old_s, new_s, 1)

        if new_content == current:
            return None, f"`{path}` edits produced no net change"

        commit_msg = (
            f"propose_patch: {title}\n\n"
            f"Proposed by cora on PR #{pr_number}."
        )
        try:
            _gh_api_with_token(
                ["-X", "PUT", f"repos/{repo}/contents/{path}"],
                gh_token,
                input_data={
                    "message": commit_msg,
                    "content": base64.b64encode(
                        new_content.encode("utf-8")
                    ).decode("ascii"),
                    "sha": prev_sha,
                    "branch": branch_name,
                },
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"PUT `{path}` failed: {exc}"

    # 4. Open the PR (non-draft — classifier-skip suppresses review + CI until
    #    it merges into the source PR's branch, at which point the source PR's
    #    CI runs normally).
    pr_body = (
        body
        + f"\n\n---\n\n_Proposed by cora on #{pr_number}. "
        + "Merge into the source PR's branch to apply; the reviewer re-runs "
        + "on the source PR when this lands._"
    )
    if body_prefix:
        pr_body = body_prefix.rstrip() + "\n\n---\n\n" + pr_body
    try:
        pr_resp = _gh_api_with_token(
            ["-X", "POST", f"repos/{repo}/pulls"],
            gh_token,
            input_data={
                "title": title,
                "body": pr_body,
                "head": branch_name,
                "base": base_ref,
                "draft": False,
            },
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail
        branch_url = f"https://github.com/{repo}/tree/{branch_name}"
        return None, f"PR open failed: {exc} (branch: {branch_url})"

    pr_url = (pr_resp or {}).get("html_url")

    # 5. Label the new PR `classifier-skip` so the classifier doesn't run and
    #    no `review-quick/deep/large` label is applied — the agentic reviewer
    #    fires on the source PR after this merges, not on this PR.
    if pr_url:
        new_pr_number = pr_url.rsplit("/", 1)[-1]
        if new_pr_number.isdigit():
            try:
                _gh_api_with_token(
                    ["-X", "POST",
                     f"repos/{repo}/issues/{new_pr_number}/labels"],
                    gh_token,
                    input_data={"labels": ["classifier-skip"]},
                )
            except Exception:  # noqa: BLE001 — soft-fail; PR creation succeeded
                pass

    return pr_url, None


def apply_push_to_source_branch(
    repo: str,
    pr_number: str,
    head_ref: str,
    head_sha: str,
    directive: dict,
    gh_token: str,
) -> tuple[str | None, str | None]:
    """Apply a validated directive by pushing a commit onto the SOURCE
    PR's head branch — used for bot-authored same-repo PRs where the
    lockstep edit should ride with the source PR rather than forking a
    sibling draft PR (a dependency-bump PR needing a lockstep manifest
    edit is the motivating example).

    Mirrors the file-edit machinery of `apply_propose_patch` but skips
    the create-branch and open-draft-PR steps. Returns `(commit_sha,
    error_message)` — exactly one non-None. `commit_sha` is the SHA
    of the last PUT response (each PUT creates one commit on the
    Contents API; multi-file directives produce multiple commits in
    sequence on the same branch).

    Soft-fails on any API error; never raises out. Partial state
    (some files committed, some not) leaves the branch in that
    intermediate shape — surface the error so the operator can inspect.
    """
    title = directive["title"].strip()
    edits = directive["edits"]

    # Group edits by path — same shape as apply_propose_patch.
    edits_by_path: dict[str, list[dict]] = {}
    for edit in edits:
        edits_by_path.setdefault(edit["path"].strip(), []).append(edit)

    last_commit_sha: str | None = None

    for path, file_edits in edits_by_path.items():
        # Read the file at the branch HEAD (not base) — re-runs after a
        # prior partial apply see the in-flight commits.
        try:
            file_resp = _gh_api_with_token(
                ["-X", "GET", f"repos/{repo}/contents/{path}?ref={head_ref}"],
                gh_token,
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"could not read `{path}` on `{head_ref}`: {exc}"
        if (file_resp or {}).get("type") != "file":
            return None, f"`{path}` is not a regular file"
        try:
            current = base64.b64decode(file_resp["content"]).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"could not decode `{path}`: {exc}"
        prev_sha = file_resp["sha"]

        new_content = current
        for i, edit in enumerate(file_edits):
            old_s = edit["old_string"]
            new_s = edit["new_string"]
            count = new_content.count(old_s)
            if count == 0:
                return None, f"edit on `{path}` #{i}: `old_string` not found"
            if count > 1:
                return None, (
                    f"edit on `{path}` #{i}: `old_string` matches {count} times "
                    f"(must be unique — narrow the snippet)"
                )
            new_content = new_content.replace(old_s, new_s, 1)

        if new_content == current:
            return None, f"`{path}` edits produced no net change"

        commit_msg = (
            f"propose_patch: {title}\n\n"
            f"Proposed by cora on PR #{pr_number} as a "
            f"lockstep commit. Routed to source branch (bot-authored "
            f"same-repo PR) instead of a forked draft PR."
        )
        try:
            put_resp = _gh_api_with_token(
                ["-X", "PUT", f"repos/{repo}/contents/{path}"],
                gh_token,
                input_data={
                    "message": commit_msg,
                    "content": base64.b64encode(
                        new_content.encode("utf-8")
                    ).decode("ascii"),
                    "sha": prev_sha,
                    "branch": head_ref,
                },
            )
            commit_sha = ((put_resp or {}).get("commit") or {}).get("sha")
            if commit_sha:
                last_commit_sha = commit_sha
        except Exception as exc:  # noqa: BLE001 — soft-fail
            return None, f"PUT `{path}` failed: {exc}"

    return last_commit_sha, None


def _fetch_file_at_ref(repo: str, path: str, ref: str) -> str | None:
    """Read a file from `ref` via `gh api`. Uses workflow GITHUB_TOKEN
    (already has `contents: read`). Returns the file content as text,
    or None if the file doesn't exist / isn't a regular file / the
    API call errors. Used by the dispatcher for in-hunk classification
    — no App token needed since this is a read."""
    proc = subprocess.run(
        ["gh", "api", "-X", "GET", f"repos/{repo}/contents/{path}?ref={ref}"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        resp = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        return None
    if resp.get("type") != "file":
        return None
    try:
        return base64.b64decode(resp["content"]).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def classify_edits_for_dispatch(
    repo: str,
    head_sha: str,
    diff_text: str,
    edits: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    """Partition `edits` into (inline_comments, out_of_hunk_edits, rejected_reasons).

    Each edit goes to one of three buckets:

      - **inline_comments**: the edit's `old_string` resolves to a unique
        line range fully inside a hunk of `diff_text`. Returned as a
        list of `comments` dicts ready for POST /pulls/{n}/reviews —
        `path`, `line` (and `start_line` for multi-line), `side` set,
        body wrapping the `new_string` in a ```suggestion``` fence.
      - **out_of_hunk_edits**: file is unchanged-by-PR, old_string is on
        a line outside any hunk, edit spans hunk boundaries, or file
        couldn't be fetched at HEAD. Re-bundled into a draft PR via
        the existing apply_propose_patch path.
      - **rejected_reasons**: edit's `old_string` not found anywhere in
        the file, ambiguous (matches multiple times), or some other
        unrecoverable shape. Surfaced in the outcome footer; not applied.

    The dispatcher reads each unique path once (via `_fetch_file_at_ref`)
    using GITHUB_TOKEN — no App token needed for reads. Hunks come from
    the same `gh pr diff` fetch the reviewer already did.
    """
    hunks_by_path = parse_pr_diff_hunks(diff_text)

    # Cache file content per path to avoid re-fetching for multi-edit batches.
    content_cache: dict[str, str | None] = {}

    inline_comments: list[dict] = []
    out_of_hunk: list[dict] = []
    rejected: list[str] = []

    for edit in edits:
        path = edit["path"].strip()
        old_s = edit["old_string"]
        new_s = edit["new_string"]

        if path not in content_cache:
            content_cache[path] = _fetch_file_at_ref(repo, path, head_sha)
        content = content_cache[path]
        if content is None:
            # File doesn't exist at HEAD (creates aren't supported by
            # validate_propose_patch v1 anyway) — route to draft PR
            # where apply_propose_patch will fail cleanly with a "could
            # not read" error. v2 may add creates; today this is dead
            # path but cheap to keep.
            out_of_hunk.append(edit)
            continue

        line_range = find_line_range(content, old_s)
        if line_range is None:
            # Not found OR not unique. validate_propose_patch already
            # capped string length and per-edit-shape; this is purely
            # a "doesn't resolve in the actual file" failure.
            rejected.append(
                f"`{path}`: `old_string` not found or matches more than once"
            )
            continue

        start_line, end_line = line_range
        hunks = hunks_by_path.get(path, [])

        if hunks and _is_in_hunk(start_line, end_line, hunks):
            # Build a PR review comment with a ```suggestion``` fence.
            # GitHub anchors single-line vs multi-line differently —
            # multi-line wants start_line + line + start_side + side.
            comment: dict = {
                "path": path,
                "line": end_line,
                "side": "RIGHT",
                "body": _format_suggestion_body(new_s),
            }
            if start_line != end_line:
                comment["start_line"] = start_line
                comment["start_side"] = "RIGHT"
            inline_comments.append(comment)
        else:
            out_of_hunk.append(edit)

    return inline_comments, out_of_hunk, rejected


def _format_suggestion_body(new_string: str) -> str:
    """Wrap `new_string` in a GitHub `suggestion` fence. The trailing
    newline before the closing fence is intentional — without it, the
    last line of the suggestion gets lost on some clients."""
    # Strip a single trailing newline from new_string (the suggestion
    # block adds its own boundary). Multiple trailing newlines preserved
    # — they're significant for YAML/manifest edits.
    body = new_string
    if body.endswith("\n"):
        body = body[:-1]
    return (
        "```suggestion\n"
        f"{body}\n"
        "```\n\n"
        "_Suggested by cora. "
        "Click **Apply suggestion** to commit._"
    )


def _format_diff_block(old_string: str, new_string: str) -> str:
    """Format an old→new change as a fenced ```diff block.

    Each line of `old_string` is prefixed with `- ` and each line of
    `new_string` with `+ `.  A trailing `\\n` on either string is
    stripped before splitting — it's the file-line terminator, not an
    extra blank line.
    """
    old_lines = old_string.rstrip("\n").splitlines()
    new_lines = new_string.rstrip("\n").splitlines()
    diff_body = "\n".join(f"- {ln}" for ln in old_lines)
    if new_lines:
        diff_body += "\n" + "\n".join(f"+ {ln}" for ln in new_lines)
    return f"```diff\n{diff_body}\n```"


def apply_inline_suggestions(
    repo: str,
    pr_number: str,
    head_sha: str,
    summary: str,
    comments: list[dict],
    gh_token: str | None,
    *,
    summary_prefix: str | None = None,
) -> tuple[str | None, str | None]:
    """Post a single PR review with all inline suggestion comments.

    Uses the cora App token when `gh_token` is set (audit actor =
    `cora[bot]`); otherwise the workflow's default GITHUB_TOKEN
    (audit actor = `github-actions[bot]`). The suggestion-apply commit,
    when the operator clicks "Apply suggestion", is authored by the
    operator regardless of who posted the review.

    Returns `(review_html_url, error_message)` — exactly one non-None.
    `event: "COMMENT"` (not APPROVE / REQUEST_CHANGES) — the reviewer
    bot is advisory, not gating.
    """
    if not comments:
        return None, "no inline comments to post"
    review_body = summary
    if summary_prefix:
        review_body = summary_prefix.rstrip() + "\n\n" + summary
    payload = {
        "commit_id": head_sha,
        "body": review_body,
        "event": "COMMENT",
        "comments": comments,
    }
    try:
        if gh_token:
            resp = _gh_api_with_token(
                ["-X", "POST", f"repos/{repo}/pulls/{pr_number}/reviews"],
                gh_token,
                input_data=payload,
            )
        else:
            # No App token — use the inherited workflow GITHUB_TOKEN.
            # subprocess inherits env from the parent, so plain `gh api`
            # picks up GH_TOKEN naturally.
            proc = subprocess.run(
                [
                    "gh", "api", "-X", "POST",
                    f"repos/{repo}/pulls/{pr_number}/reviews",
                    "--input", "-",
                ],
                input=json.dumps(payload),
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                err = proc.stderr.strip()[:300]
                return None, f"review POST failed: {err}"
            resp = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except Exception as exc:  # noqa: BLE001 — soft-fail
        return None, f"review POST failed: {exc}"
    return (resp or {}).get("html_url"), None


def apply_propose_patch_dispatch(
    repo: str,
    pr_number: str,
    base_ref: str,
    head_sha: str,
    diff_text: str,
    directive: dict,
    gh_token: str,
    *,
    head_ref: str | None = None,
    is_bot_author_pr: bool = False,
    is_fork_pr: bool = True,
    escalation_warning_body: str | None = None,
    escalation_warning_summary: str | None = None,
    suppress_other_file_edits: bool = False,
) -> dict:
    """Hybrid dispatcher — classify each edit, route to inline
    suggestions or a propose-patch PR, compose a unified outcome dict.

    Returns:
        {
          "inline_url": str | None,               # PR review URL if posted
          "inline_count": int,
          "draft_url": str | None,                # propose-patch PR URL
          "draft_count": int,
          "source_branch_commit_sha": str | None, # unused; kept for compat
          "source_branch_count": int,
          "rejected": list[str],                  # human-readable rejection reasons
          "error": str | None,                    # top-level error if dispatch failed entirely
        }

    Out-of-hunk routing — two paths only:

      - **In-hunk edit**: inline ``suggestion`` comment on the PR review.
        GitHub anchors suggestions to diff lines; single-click apply.

      - **Out-of-hunk edit** (any file, any size, non-minor verdict):
        ``apply_propose_patch`` → new branch off ``base_ref`` (= source
        PR's head for same-repo, or source PR's base for forks), then a
        regular (non-draft) PR with the ``classifier-skip`` label so the
        classifier doesn't assign a review label. The agentic reviewer
        fires on the source PR after this merges — not on this PR.

      - **Out-of-hunk + 🟡 minor** (``suppress_other_file_edits=True``):
        skipped — minor findings don't warrant a separate PR artifact.
        Added to ``rejected[]`` so the footer notes it.

    Fork PRs (``is_fork_pr=True``) target ``base_ref`` (usually main)
    because the App can't push to a fork branch.

    `escalation_warning_body` / `escalation_warning_summary`, when set,
    are prepended to the PR body and the inline review summary —
    the "T2 disagrees" banner for the patch-escalation flow.

    Soft-fail throughout: per-mechanism failures surface in the outcome
    but never raise out. Caller renders the outcome into the review
    comment footer.
    """
    outcome: dict = {
        "inline_url": None,
        "inline_count": 0,
        "draft_url": None,
        "draft_count": 0,
        "source_branch_commit_sha": None,
        "source_branch_count": 0,
        "rejected": [],
        "error": None,
    }

    try:
        inline_comments, out_of_hunk_edits, rejected = (
            classify_edits_for_dispatch(
                repo, head_sha, diff_text, directive["edits"]
            )
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail
        outcome["error"] = f"dispatch classification failed: {exc}"
        return outcome

    outcome["rejected"] = rejected

    # Inline path — workflow GITHUB_TOKEN is sufficient (pull-requests:write).
    # Pass the App token when available for consistent audit actor.
    if inline_comments:
        summary = (
            f"_Proposed by cora on #{pr_number}._ "
            f"Click **Apply suggestion** in the diff view below to commit each fix."
        )
        review_url, review_err = apply_inline_suggestions(
            repo=repo, pr_number=pr_number, head_sha=head_sha,
            summary=summary, comments=inline_comments,
            gh_token=gh_token or None,
            summary_prefix=escalation_warning_summary,
        )
        if review_url:
            outcome["inline_url"] = review_url
            outcome["inline_count"] = len(inline_comments)
        else:
            outcome["error"] = (
                f"inline suggestions failed: {review_err}"
            )
            # Don't bail — still try draft PR for out_of_hunk edits.

    # Out-of-hunk routing — one path for all out-of-hunk edits:
    #
    #   non-minor → apply_propose_patch: new branch off head_ref (same-repo)
    #               or base_ref (fork), regular PR with classifier-skip label.
    #               The agentic reviewer fires on the source PR after merge.
    #   minor     → skip (no PR for minor findings).
    #
    # Needs the cora App token (contents: write + pull-requests: write).
    if out_of_hunk_edits:
        if suppress_other_file_edits:
            # Verdict is minor — no PR artifact for minor findings.
            outcome["rejected"].append(
                f"{len(out_of_hunk_edits)} out-of-hunk edit(s) skipped "
                "(verdict is minor — propose_patch PR suppressed for minor findings)"
            )
        elif not gh_token:
            outcome["rejected"].append(
                f"{len(out_of_hunk_edits)} out-of-hunk edit(s) skipped: "
                "cora App token not available (workflow mint step soft-failed)"
            )
        else:
            # Build a body note if inline suggestions also fired on this directive.
            patch_body = directive["body"]
            if outcome["inline_count"]:
                patch_body += (
                    f"\n\n_{outcome['inline_count']} edit(s) in this directive "
                    f"were posted as inline suggestions on #{pr_number}. "
                    f"This PR covers only the out-of-hunk edits._"
                )
            sub_directive = {
                "title": directive["title"],
                "body": patch_body,
                "edits": out_of_hunk_edits,
            }
            # Same-repo → target source PR's head so the fix sees the source
            # PR's diff (avoids the base-ref regression where fix PRs based
            # off main couldn't see the source PR's changes).  Fork → fall
            # back to base_ref (App can't push to fork branches).
            patch_base_ref = head_ref if (head_ref and not is_fork_pr) else base_ref
            patch_url, patch_err = apply_propose_patch(
                repo=repo, pr_number=pr_number,
                base_ref=patch_base_ref, directive=sub_directive,
                gh_token=gh_token,
                body_prefix=escalation_warning_body,
            )
            if patch_url:
                outcome["draft_url"] = patch_url
                outcome["draft_count"] = len(out_of_hunk_edits)
                # Back-link comment on the new PR so the operator knows its
                # purpose without reading the full body.
                new_pr_number = patch_url.rsplit("/", 1)[-1]
                if new_pr_number.isdigit():
                    if patch_base_ref == base_ref:
                        merge_note = (
                            "Merge into the source PR's branch before or "
                            "alongside that PR."
                        )
                    else:
                        merge_note = (
                            f"Targets `{patch_base_ref}` (source PR's head) — "
                            f"merging here lands the fix on #{pr_number}'s branch "
                            f"and that PR's own merge carries it through. "
                            f"Don't change the base."
                        )
                    backlink_body = (
                        f"🤖 Proposed by cora on #{pr_number}. {merge_note}"
                    )
                    try:
                        _gh_api_with_token(
                            ["-X", "POST",
                             f"repos/{repo}/issues/{new_pr_number}/comments"],
                            gh_token,
                            input_data={"body": backlink_body},
                        )
                    except Exception:  # noqa: BLE001 — soft-fail
                        pass
            else:
                if not outcome["error"]:
                    outcome["error"] = f"propose-patch PR failed: {patch_err}"

    return outcome
