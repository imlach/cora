"""GitHub PR-comment posting (find-or-edit, retry, skip + initial bodies)
and the auto-merge label-remove helper.

`post_or_edit_comment` recognises the current cora marker plus the
retired-marker shapes (pre-rename and earlier) so an in-flight PR's
existing comment is edited in place rather than getting a stale orphan
+ a fresh comment. Subprocess-based —
prefers the cora App-minted token when set so the audit actor is
`cora[bot]`; falls back to the workflow's inherited GITHUB_TOKEN.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

from cora.core.check_run import _grafana_drilldown_url, _workflow_run_url
from cora.core.config import (
    AUTOMERGE_LABEL,
    COMMENT_MARKER,
    LEGACY_COMMENT_MARKERS,
)


# 4xx/5xx in gh's stderr — App-token install-token edges occasionally
# 401-flake (a transient 401 once lost a review comment that way), and
# the comments API throws transient 5xx during GitHub incidents. One
# retry absorbs both classes; non-HTTP failures (bad args, missing
# binary) skip the retry and fail through.
_GH_HTTP_TRANSIENT_RE = re.compile(r"HTTP\s+[45]\d\d")


def _gh_with_one_retry(
    cmd: list[str],
    env: dict[str, str],
    *,
    input_: str | None = None,
    retry_sleep_s: float = 1.5,
) -> subprocess.CompletedProcess:
    """Run `gh` once, retry once on a 4xx/5xx response. Returns the
    final CompletedProcess; the caller decides how to surface a
    non-zero rc."""
    proc = subprocess.run(
        cmd, env=env, input=input_, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return proc
    if not _GH_HTTP_TRANSIENT_RE.search(proc.stderr or ""):
        return proc
    first_line = (proc.stderr or "").splitlines()[0] if proc.stderr else ""
    print(f"::warning::gh transient ({first_line}); retrying once after {retry_sleep_s}s")
    time.sleep(retry_sleep_s)
    return subprocess.run(
        cmd, env=env, input=input_, capture_output=True, text=True,
    )


def post_or_edit_comment(repo: str, pr_number: str, body: str) -> None:
    # Edit-last across migrations: recognise the new marker AND the
    # retired ones (`<!-- agentic-review:v2 -->` pre-rename, `:v1` from
    # the old single-shot path, `-loop:v1` from the old loop path) so an
    # in-flight PR's existing comment is edited in place rather than
    # getting a stale orphan + a fresh cora comment.
    #
    # Comment posting uses the cora App-minted token (CORA_GH_TOKEN
    # env, set by the workflow's create-github-app-token step) when
    # available so the comment actor is `cora[bot]`. Falls back to
    # the inherited workflow GITHUB_TOKEN when the mint step soft-
    # failed or the App isn't configured.
    # Same App token already used for the inline-suggestion PR review
    # and the draft-PR path — consolidates the audit identity.
    env = os.environ.copy()
    app_token = os.environ.get("CORA_GH_TOKEN", "").strip()
    if app_token:
        env["GH_TOKEN"] = app_token

    marker_predicates = " or ".join(
        f'(.body | startswith("{m}"))'
        for m in (COMMENT_MARKER, *LEGACY_COMMENT_MARKERS)
    )
    list_proc = _gh_with_one_retry(
        [
            "gh", "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "--jq",
            f"[.[] | select({marker_predicates})] | first | .id // empty",
        ],
        env=env,
    )
    if list_proc.returncode != 0:
        raise RuntimeError(
            f"list comments failed (rc={list_proc.returncode}): "
            f"{list_proc.stderr.strip()}"
        )
    listing = list_proc.stdout.strip()
    if listing:
        proc = _gh_with_one_retry(
            ["gh", "api", "-X", "PATCH",
             f"repos/{repo}/issues/comments/{listing}",
             "--input", "-"],
            env=env,
            input_=json.dumps({"body": body}),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"PATCH comment failed: {proc.stderr.strip()}")
    else:
        proc = _gh_with_one_retry(
            ["gh", "pr", "comment", pr_number, "--body", body],
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"create comment failed: {proc.stderr.strip()}")


def create_pr_review(repo: str, pr_number: str, body: str, event: str) -> None:
    """Post the verdict as a first-class GitHub PR Review via
    `POST /repos/{repo}/pulls/{pr}/reviews` with `body` + `event`
    (COMMENT / REQUEST_CHANGES / APPROVE), instead of an issue comment.

    Opt-in (`ReviewerConfig.use_github_review`) — the default path stays
    `post_or_edit_comment`. Unlike a comment, a Review is append-only:
    GitHub has no "edit the bot's last review in place" affordance the
    way the issue-comments API does, so each run files a fresh review
    (the check-run still carries the single authoritative verdict).

    Same token + transient-retry handling as `post_or_edit_comment` — the
    cora App token when set (audit actor `cora[bot]`), else the
    inherited GITHUB_TOKEN.
    """
    env = os.environ.copy()
    app_token = os.environ.get("CORA_GH_TOKEN", "").strip()
    if app_token:
        env["GH_TOKEN"] = app_token

    proc = _gh_with_one_retry(
        [
            "gh", "api", "-X", "POST",
            f"repos/{repo}/pulls/{pr_number}/reviews",
            "--input", "-",
        ],
        env=env,
        input_=json.dumps({"body": body, "event": event}),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"create review failed: {proc.stderr.strip()}")


def _fmt_ts(dt: datetime) -> str:
    """Footer timestamp — UTC, second precision. Unambiguous across
    readers' mixed local zones and stable when the comment is read days
    after the run."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def make_skip_comment(reason: str) -> str:
    return (
        f"{COMMENT_MARKER}\n"
        f"## cora review — skipped\n\n"
        f"_{reason}_\n\n"
        f"<sub>soft-fail by design</sub>"
    )


def make_initial_comment(pr_number: str, started_at: datetime) -> str:
    """Posted at workflow start in deep mode, before the agent loop runs.
    Quick mode skips this — the single LLM call completes in ~20-30s and
    posting an in-progress placeholder then editing it 25 s later is
    churn without benefit. Replaced by `make_review_comment` (or
    `make_skip_comment`) once the loop completes.
    """
    links = []
    run_url = _workflow_run_url()
    if run_url:
        links.append(f'<a href="{run_url}">workflow logs</a>')
    grafana_url = _grafana_drilldown_url(pr_number)
    if grafana_url:
        links.append(f'<a href="{grafana_url}">grafana</a>')
    links_line = " · ".join(links)
    return (
        f"{COMMENT_MARKER}\n"
        f"## cora review — in progress\n\n"
        f"🔄 Reviewing PR — live progress: {links_line}\n\n"
        f"<sub>Started {_fmt_ts(started_at)} · this comment will update "
        f"with the verdict when the review completes (~30-120s typical "
        f"for deep mode).</sub>"
    )


def remove_automerge_label(repo: str, pr_number: str) -> bool:
    """Remove `automerge` from the PR. Idempotent — `gh pr edit
    --remove-label` no-ops if absent and returns rc=0.

    Uses the cora App token when set so the `unlabeled` event actor
    matches the rest of the bot's writes (`cora[bot]` rather than
    `github-actions[bot]`). Falls back to GITHUB_TOKEN when the App
    isn't configured.
    """
    env = os.environ.copy()
    app_token = os.environ.get("CORA_GH_TOKEN", "").strip()
    if app_token:
        env["GH_TOKEN"] = app_token
    proc = subprocess.run(
        [
            "gh", "pr", "edit", pr_number,
            "--repo", repo,
            "--remove-label", AUTOMERGE_LABEL,
        ],
        env=env,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(
            f"::warning::could not remove `{AUTOMERGE_LABEL}` label: "
            f"{proc.stderr.strip()}"
        )
        return False
    return True
