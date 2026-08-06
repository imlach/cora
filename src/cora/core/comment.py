"""GitHub PR-comment posting (per-run create/update, retry, skip +
initial bodies) and the auto-merge label-remove helper.

Per-run comment loop (cora#29): a review run gets exactly one comment
that it owns — created fresh by `create_progress_comment`, PATCHed in
place by `update_run_comment` as the run progresses and again to post
the verdict — and, once that verdict lands, every *other* cora comment
on the PR is collapsed via `minimize_superseded_comments`. That split
replaces the old `post_or_edit_comment` (find-the-first-cora-comment,
edit forever), which destroyed review history: run N+1's PATCH
overwrote run N's verdict with no trace it had changed. Comments are
now append-then-collapse, the same shape a human reviewer's repeated
reviews produce, and the collapse runs LAST so a cancelled/failed run
never hides the last good review.

Subprocess-based throughout — prefers the cora App-minted token when
set so the audit actor is `cora[bot]`; falls back to the workflow's
inherited GITHUB_TOKEN.
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
    MARKER_SUFFIX,
    PROGRESS_MARKER_PREFIX,
    VERDICT_MARKER_PREFIX,
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


def _run_id() -> str:
    """`GITHUB_RUN_ID` identifies the current workflow run — the same
    env var `_tiers.py`'s trajectory-capture path keys per-run files
    on, and `_workflow_run_url` reads for the run-logs deeplink.
    "local" outside Actions (bare invocations, tests) so run-scoping
    degrades gracefully instead of crashing; it just isn't unique
    across local invocations, which don't share a PR to collide on
    anyway."""
    return os.environ.get("GITHUB_RUN_ID", "local")


def _progress_marker(run_id: str) -> str:
    return f"{PROGRESS_MARKER_PREFIX}{run_id}{MARKER_SUFFIX}"


def _verdict_marker(run_id: str) -> str:
    return f"{VERDICT_MARKER_PREFIX}{run_id}{MARKER_SUFFIX}"


def _startswith_any_jq(markers: tuple[str, ...]) -> str:
    """jq boolean expression: true if `.body` starts with any of
    `markers`. Works for bare prefixes too (`PROGRESS_MARKER_PREFIX`
    without its `<run_id>` suffix) — jq's `startswith` just compares
    the literal characters given, so a prefix alone matches any run id.
    `GITHUB_RUN_ID` is GitHub-minted (numeric), so no escaping is
    needed for the f-string interpolation here."""
    return " or ".join(f'(.body | startswith("{m}"))' for m in markers)


def _cora_env() -> dict[str, str]:
    """`os.environ` with the cora App token (when set) swapped in as
    `GH_TOKEN`, so the comment/mutation actor is `cora[bot]` rather
    than the workflow's inherited `GITHUB_TOKEN`."""
    env = os.environ.copy()
    app_token = os.environ.get("CORA_GH_TOKEN", "").strip()
    if app_token:
        env["GH_TOKEN"] = app_token
    return env


def create_progress_comment(repo: str, pr_number: str, body: str) -> None:
    """Create THIS run's progress placeholder as a brand-new comment —
    never finds-or-edits. A prior run's leftover placeholder (left
    behind when the workflow's `concurrency` block cancels an
    in-flight run on `synchronize`) carries a different run id, so it's
    never mistaken for this run's; `minimize_superseded_comments`
    collapses it once this run finishes.
    """
    proc = _gh_with_one_retry(
        ["gh", "pr", "comment", pr_number, "--body", f"{_progress_marker(_run_id())}\n{body}"],
        env=_cora_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"create comment failed: {proc.stderr.strip()}")


def update_run_comment(repo: str, pr_number: str, body: str, *, final: bool) -> None:
    """PATCH THIS run's own comment — matched by `GITHUB_RUN_ID`, never
    another run's — swapping in a fresh marker + body. `final=False`
    keeps the progress marker (mid-run updates); `final=True` swaps to
    the verdict marker (`post_review` / `post_skip` — a completed run
    is never edited again once a later run supersedes it).

    Creates a comment instead of PATCHing when this run has none yet:
    quick mode and every early-exit skip path never call
    `create_progress_comment` (the placeholder is deep-mode only), so
    the verdict/skip comment is often this run's first and only one.
    """
    run_id = _run_id()
    marker = _verdict_marker(run_id) if final else _progress_marker(run_id)
    full_body = f"{marker}\n{body}"
    env = _cora_env()

    own_predicate = _startswith_any_jq((_progress_marker(run_id), _verdict_marker(run_id)))
    list_proc = _gh_with_one_retry(
        [
            "gh", "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "--jq",
            f"[.[] | select({own_predicate})] | first | .id // empty",
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
            input_=json.dumps({"body": full_body}),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"PATCH comment failed: {proc.stderr.strip()}")
    else:
        proc = _gh_with_one_retry(
            ["gh", "pr", "comment", pr_number, "--body", full_body],
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"create comment failed: {proc.stderr.strip()}")


# GraphQL is the only way to minimise a comment — REST has no
# equivalent. `subjectId` is the comment's *node_id*, not its REST
# `id`; `OUTDATED` is the honest classifier for "a later review
# replaced this one" (vs. SPAM/ABUSE/OFF_TOPIC/RESOLVED/DUPLICATE).
_MINIMIZE_COMMENT_MUTATION = (
    "mutation($id: ID!) { "
    "minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { "
    "minimizedComment { isMinimized } } }"
)


def _minimize_comment(node_id: str, env: dict[str, str]) -> bool:
    """Best-effort GraphQL `minimizeComment` on one comment. Returns
    whether it worked; never raises. Two failure shapes: a non-zero rc
    (permissions, transient API error), and the sneakier one — `gh api
    graphql` exits 0 on an HTTP 200 whose JSON body still carries a
    top-level `errors` array, so a clean rc alone doesn't mean the
    mutation applied.
    """
    proc = _gh_with_one_retry(
        [
            "gh", "api", "graphql",
            "-f", f"query={_MINIMIZE_COMMENT_MUTATION}",
            "-f", f"id={node_id}",
        ],
        env=env,
    )
    if proc.returncode != 0:
        print(
            f"::warning::minimize comment {node_id} failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()[:300]}"
        )
        return False
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        print(f"::warning::minimize comment {node_id}: unparseable graphql response")
        return False
    if payload.get("errors"):
        print(f"::warning::minimize comment {node_id}: graphql errors {payload['errors']}")
        return False
    return True


def minimize_superseded_comments(repo: str, pr_number: str) -> None:
    """Collapse every OTHER cora comment on the PR as OUTDATED, once
    this run's own verdict/skip comment is live. `post_review` /
    `post_skip` call this LAST, after `update_run_comment` finalises
    this run's own comment — so a run that never gets that far
    (cancelled, or the finalize PATCH itself failing) leaves the
    previous verdict visible instead of collapsing it out from under a
    review that never replaced it.

    Purely cosmetic, so unlike its siblings above this soft-fails
    end-to-end: any failure (listing the PR's comments, an
    unparseable response, a missing `node_id`, the mutation itself)
    prints `::warning::` and returns rather than raising. A review must
    never be lost because a predecessor couldn't be collapsed.
    """
    try:
        run_id = _run_id()
        env = _cora_env()
        cora_predicate = _startswith_any_jq(
            (COMMENT_MARKER, *LEGACY_COMMENT_MARKERS, PROGRESS_MARKER_PREFIX, VERDICT_MARKER_PREFIX)
        )
        own_predicate = _startswith_any_jq((_progress_marker(run_id), _verdict_marker(run_id)))
        list_proc = _gh_with_one_retry(
            [
                "gh", "api",
                f"repos/{repo}/issues/{pr_number}/comments",
                "--jq",
                (
                    f"[.[] | select({cora_predicate}) | select(({own_predicate}) | not) "
                    f"| {{id: .id, node_id: .node_id}}]"
                ),
            ],
            env=env,
        )
        if list_proc.returncode != 0:
            print(
                f"::warning::list comments for minimize failed "
                f"(rc={list_proc.returncode}): {list_proc.stderr.strip()}"
            )
            return
        targets = json.loads(list_proc.stdout or "[]")
        for target in targets:
            node_id = target.get("node_id")
            if not node_id:
                print(f"::warning::minimize: comment id={target.get('id')} missing node_id, skipping")
                continue
            _minimize_comment(node_id, env)
    except Exception as exc:  # noqa: BLE001 — cosmetic sweep, must never fail the review
        print(f"::warning::minimize superseded comments failed: {exc}")


def create_pr_review(repo: str, pr_number: str, body: str, event: str) -> None:
    """Post the verdict as a first-class GitHub PR Review via
    `POST /repos/{repo}/pulls/{pr}/reviews` with `body` + `event`
    (COMMENT / REQUEST_CHANGES / APPROVE), instead of an issue comment.

    Opt-in (`ReviewerConfig.use_github_review`) — the default path stays
    the per-run comment loop (`update_run_comment` +
    `minimize_superseded_comments`). Unlike a comment, a Review is
    append-only: GitHub has no "edit the bot's last review in place"
    affordance the way the issue-comments API does, so each run files a
    fresh review (the check-run still carries the single authoritative
    verdict).

    Same token + transient-retry handling as the rest of this module —
    the cora App token when set (audit actor `cora[bot]`), else the
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
