"""PR + CI metadata fetch helpers.

Wraps `gh pr view`, `gh pr diff`, and the GH check-runs / actions-jobs
APIs into a small set of functions that feed the initial prompt
assembly. Subprocess-based — uses the workflow's inherited `gh` binary
+ GITHUB_TOKEN. Everything here is read-only and best-effort: a failed
lookup yields None (or raises into the caller's soft-fail wrapper).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core.config import CHECK_RUN_NAME, CI_CONTEXT_CHAR_CAP, CI_LOG_TAIL_CHARS

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


def run(cmd: list[str], **kwargs) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def read_capped(path: Path, cap: int) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw) <= cap:
        return raw, False
    return raw[:cap] + f"\n\n…[truncated at {cap} chars]…\n", True


def fetch_pr_metadata(pr_number: str) -> dict:
    raw = run(
        [
            "gh", "pr", "view", pr_number,
            "--json",
            "title,body,labels,author,baseRefName,headRefName,headRepository,isCrossRepository,additions,deletions,changedFiles",
        ]
    )
    return json.loads(raw)


def fetch_pr_diff(pr_number: str) -> str:
    return run(["gh", "pr", "diff", pr_number])


def fetch_author_association(repo: str, pr_number: str) -> str:
    """REST `author_association` for the PR (OWNER / MEMBER / … / NONE).

    `gh pr view --json` doesn't expose the field, so this is a separate
    REST read — the trigger gate only makes it when the policy is
    enforced, keeping the unenforced API-call profile unchanged.
    Soft-fails to "" (treated as untrusted): a gate that fails OPEN on a
    transient API error would be a bypass."""
    try:
        raw = run([
            "gh", "api", f"/repos/{repo}/pulls/{pr_number}",
            "--jq", ".author_association // \"\"",
        ])
        return raw.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::author_association fetch failed: {exc}")
        return ""


# Actions check_runs carry the job in their html_url as `/job/<id>` —
# the only handle for pulling that job's log via the Actions API.
_JOB_ID_RE = re.compile(r"/job/(\d+)")


def gather_ci_context(
    repo: str,
    head_sha: str | None,
    *,
    cfg: "ReviewerConfig | None" = None,
) -> str | None:
    """Fetch the PR head commit's failing CI checks and a bounded tail of
    each failing job's log, rendered as a markdown block to fold into the
    review prompt. Returns None when nothing is failing — or when the
    lookup itself fails: CI context is best-effort and never blocks the
    review.

    The reviewer's own checks (`cfg.check_run_name`, plus the legacy
    `agentic-pr-review*` prefix from before the cora rename) and the
    `required` aggregator are excluded — the aggregator only mirrors
    sibling state, and a reviewer reviewing its own red check is a
    feedback loop.

    `cfg` supplies the section / log-tail caps; None falls back to the
    engine constants the config defaults mirror (bit-identical).
    """
    ci_context_char_cap = (
        cfg.ci_context_char_cap if cfg is not None else CI_CONTEXT_CHAR_CAP
    )
    ci_log_tail_chars = (
        cfg.ci_log_tail_chars if cfg is not None else CI_LOG_TAIL_CHARS
    )
    if not head_sha:
        return None
    try:
        raw = run([
            "gh", "api", "-H", "Accept: application/vnd.github+json",
            f"/repos/{repo}/commits/{head_sha}/check-runs?per_page=100",
        ])
        check_runs = json.loads(raw).get("check_runs", [])
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::CI context fetch failed, continuing without: {exc}")
        return None

    # Latest run per check name — folds re-runs (same fold a
    # required-check aggregator applies).
    latest: dict[str, dict] = {}
    for cr in check_runs:
        name = cr.get("name", "")
        prev = latest.get(name)
        if prev is None or (cr.get("started_at") or "") > (prev.get("started_at") or ""):
            latest[name] = cr

    bad = {"failure", "timed_out", "cancelled", "action_required"}
    # A rebranded verdict check (cfg.check_run_name) is the reviewer's own
    # output too — same feedback-loop exclusion as the legacy pre-rename
    # prefix, which stays until no live PR carries old-name check runs.
    own_check = cfg.check_run_name if cfg is not None else CHECK_RUN_NAME
    failing = sorted(
        (
            cr for name, cr in latest.items()
            if cr.get("conclusion") in bad
            and not name.startswith("agentic-pr-review")
            and name != own_check
            and name != "required"
        ),
        key=lambda cr: cr.get("name", ""),
    )
    if not failing:
        return None

    parts = [
        "## CI status — failing checks",
        "",
        "These CI checks are red on the PR head commit. Weigh them in the "
        "verdict — approving a PR whose checks fail is misleading. Say "
        "whether the diff is the cause and, where the log makes it clear, "
        "what the fix is.",
    ]
    per_check = max(ci_context_char_cap // len(failing), 600)
    for cr in failing:
        name = cr.get("name", "?")
        conclusion = cr.get("conclusion", "?")
        url = cr.get("html_url") or cr.get("details_url") or ""
        parts += ["", f"### `{name}` — {conclusion}"]
        excerpt = ""
        m = _JOB_ID_RE.search(url)
        if m:
            try:
                log = run([
                    "gh", "api", f"/repos/{repo}/actions/jobs/{m.group(1)}/logs",
                ])
                excerpt = log[-ci_log_tail_chars:].strip()
            except Exception:  # noqa: BLE001
                excerpt = ""  # non-Actions check or log expired — fall back
        if not excerpt:
            out = cr.get("output") or {}
            excerpt = (out.get("summary") or out.get("title") or "").strip()
        if excerpt:
            parts += ["", "```", excerpt[:per_check], "```"]
        if url:
            parts.append(f"[check details]({url})")
    return "\n".join(parts)[:ci_context_char_cap + 2_000]


_CLASSIFIER_COMMENT_MARKER = "<!-- pr-label-classifier:v1 -->"
_CLASSIFIER_META_RE = re.compile(
    r"<!--\s*classifier-meta:v1\s+(\{.*?\})\s*-->",
    re.DOTALL,
)


def fetch_classifier_rationale(repo: str, pr_number: str) -> str | None:
    """Return the classifier's machine-readable rationale block rendered
    as a markdown section, or None if the comment is missing / unparseable.

    Best-effort: every lookup / parse failure returns None so the reviewer
    runs with the same shape as before classifier-meta existed. The
    matching comment is anchored by `<!-- pr-label-classifier:v1 -->`;
    the structured payload lives in a sibling `<!-- classifier-meta:v1
    {json} -->` line emitted by the classifier's comment writer.
    """
    try:
        raw = run([
            "gh", "api", "-H", "Accept: application/vnd.github+json",
            f"/repos/{repo}/issues/{pr_number}/comments?per_page=100",
        ])
        comments = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(comments, list):
        return None
    body = ""
    for c in comments:
        b = (c or {}).get("body") or ""
        if _CLASSIFIER_COMMENT_MARKER in b:
            body = b
            # Don't break — the classifier edits its own comment in
            # place, so the last marked comment in listing order is the
            # freshest rationale.
    if not body:
        return None
    m = _CLASSIFIER_META_RE.search(body)
    if not m:
        return None
    try:
        meta = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    label = (meta.get("label") or "").strip()
    rationale = (meta.get("rationale") or "").strip()
    if not label and not rationale:
        return None
    lines = ["## Classifier rationale"]
    lines.append(
        "The PR-label classifier (a separate, smaller model) decided the "
        "review tier for this PR. Its reason is below — treat as a hint "
        "about what to weight in the review, not a verdict."
    )
    lines.append("")
    if label:
        lines.append(f"Label: `{label}`")
    if rationale:
        lines.append(f"Reason: {rationale}")
    return "\n".join(lines)


def fetch_thread_evidence(
    repo: str,
    pr_number: str,
    *,
    cfg: "ReviewerConfig | None" = None,
) -> str | None:
    """Recent maintainer comments on the PR, rendered as an
    `<untrusted-content>`-wrapped prompt section, or None.

    cora #37: a re-review saw nothing a human had said. The only reason
    this module listed PR comments was `fetch_classifier_rationale`
    hunting its own marker — so a maintainer who rebutted a false
    finding *with log evidence* was invisible, and the re-review
    re-asserted the finding verbatim. Only a human override broke it.

    What's excluded and why:

    - **cora's own comments** (`COMMENT_MARKER` / the legacy and
      run-scoped markers). Feeding the reviewer its own prior verdict
      invites it to anchor on the finding it is supposed to re-examine
      — the opposite of the point.
    - **The classifier's comment.** Already rendered separately by
      `fetch_classifier_rationale`, and duplicating it wastes context.
    - **Bots.** CI chatter is noise, and the CI context block is where
      build signal belongs.
    - **Anyone outside `thread_evidence_associations`.** On a public
      repo anyone can comment; this text lands in the reviewer's
      context, so the standing bar matches the trigger policy's.

    The association filter narrows *whose* comments are read. It does
    NOT make them instructions: the block is wrapped in
    `<untrusted-content>` either way, because a maintainer account is
    still an account, and "the maintainer told you to approve" is
    exactly the escalation this boundary exists to stop.

    Best-effort — every failure returns None and the review runs with
    the prompt shape it had before this existed."""
    from cora.core.config import (
        COMMENT_MARKER,
        LEGACY_COMMENT_MARKERS,
        PROGRESS_MARKER_PREFIX,
        THREAD_EVIDENCE_ASSOCIATIONS,
        THREAD_EVIDENCE_BLOCK_CHAR_CAP,
        THREAD_EVIDENCE_COMMENT_CHAR_CAP,
        THREAD_EVIDENCE_MAX_COMMENTS,
        VERDICT_MARKER_PREFIX,
    )

    associations = (
        cfg.thread_evidence_associations if cfg is not None
        else THREAD_EVIDENCE_ASSOCIATIONS
    )
    max_comments = (
        cfg.thread_evidence_max_comments if cfg is not None
        else THREAD_EVIDENCE_MAX_COMMENTS
    )
    per_cap = (
        cfg.thread_evidence_comment_char_cap if cfg is not None
        else THREAD_EVIDENCE_COMMENT_CHAR_CAP
    )
    block_cap = (
        cfg.thread_evidence_block_char_cap if cfg is not None
        else THREAD_EVIDENCE_BLOCK_CHAR_CAP
    )

    cora_markers = (
        COMMENT_MARKER,
        PROGRESS_MARKER_PREFIX,
        VERDICT_MARKER_PREFIX,
        _CLASSIFIER_COMMENT_MARKER,
        *LEGACY_COMMENT_MARKERS,
    )
    try:
        raw = run([
            "gh", "api", "-H", "Accept: application/vnd.github+json",
            f"/repos/{repo}/issues/{pr_number}/comments?per_page=100",
        ])
        comments = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::thread-evidence fetch failed: {exc}")
        return None
    if not isinstance(comments, list):
        return None

    allowed = {a.upper() for a in associations}
    kept: list[tuple[str, str]] = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        body = (c.get("body") or "").strip()
        if not body or any(m in body for m in cora_markers):
            continue
        user = c.get("user") or {}
        if (user.get("type") or "") == "Bot":
            continue
        login = (user.get("login") or "").strip()
        if login.endswith("[bot]"):
            continue
        if (c.get("author_association") or "").upper() not in allowed:
            continue
        if len(body) > per_cap:
            body = body[:per_cap] + "\n…[comment truncated]"
        kept.append((login or "unknown", body))

    if not kept:
        return None
    # GitHub returns oldest-first; the newest comments are the ones a
    # re-review needs, so keep the tail and re-order oldest-first for
    # readability.
    kept = kept[-max_comments:]

    lines = ["<untrusted-content>"]
    used = 0
    dropped = 0
    for login, body in kept:
        entry = f"\n**@{login}** wrote:\n\n{body}\n"
        if used + len(entry) > block_cap and lines[-1] != "<untrusted-content>":
            dropped += 1
            continue
        lines.append(entry)
        used += len(entry)
    if dropped:
        lines.append(f"\n_[{dropped} older comment(s) omitted for budget]_\n")
    lines.append("</untrusted-content>")
    return "\n".join(lines)


def is_bot_author(metadata: dict) -> bool:
    """Detect bot-authored PRs (Renovate, Dependabot, etc.). gh returns
    `author.is_bot` directly; the login-suffix check is a fallback."""
    author = metadata.get("author") or {}
    if author.get("is_bot"):
        return True
    login = author.get("login", "") or ""
    return login.endswith("[bot]")


def is_fork_pr(metadata: dict) -> bool:
    """True when the PR's head ref lives in a different repository
    from the base — i.e. the PR comes from a fork. The cora App
    cannot push to forks, so out-of-hunk edits on fork PRs must
    fall back to the draft-PR path regardless of author type."""
    # `isCrossRepository` is GitHub's canonical fork signal and is always
    # populated. Prefer it: `gh pr view --json headRepository` returns
    # `owner: null` for SAME-repo PRs, which silently flipped the
    # owner-equality heuristic below to "fork" and based fix-PRs off main
    # instead of the source branch.
    xrepo = metadata.get("isCrossRepository")
    if isinstance(xrepo, bool):
        return xrepo
    # Fallback only when the flag is absent (e.g. metadata from an older
    # fetch). Reconstruct from head-repo vs base-repo identity; the env
    # carries `GH_REPO`/`GITHUB_REPOSITORY` as `owner/repo`.
    head_repo = metadata.get("headRepository") or {}
    head_owner = ((head_repo.get("owner") or {}).get("login") or "").lower()
    head_name = (head_repo.get("name") or "").lower()
    repo_env = os.environ.get("GH_REPO") or os.environ.get("GITHUB_REPOSITORY") or ""
    if not repo_env or "/" not in repo_env:
        # Couldn't resolve; treat as fork (safer — falls back to draft PR).
        return True
    repo_owner, repo_name = repo_env.split("/", 1)
    return not (head_owner == repo_owner.lower() and head_name == repo_name.lower())


def latest_commit_author_login(repo: str, head_sha: str) -> str | None:
    """The login of whoever authored `head_sha`. Used by the
    synchronize-loop guard to skip propose_patch when the latest
    commit on the PR's head was made by the reviewer bot itself —
    without this, the next `pull_request: synchronize` event re-fires
    the reviewer against its own commit and the patch dispatch loops.

    Returns None on lookup failure — caller treats that as "unknown,
    don't skip"."""
    if not head_sha:
        return None
    try:
        raw = run([
            "gh", "api",
            f"/repos/{repo}/commits/{head_sha}",
            "--jq", ".author.login // .committer.login // empty",
        ])
        return (raw or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _pr_head_sha() -> str | None:
    """The check run needs to be anchored to the PR head commit. GHA's
    `GITHUB_SHA` is the merge commit for `pull_request` events, not the
    head — read the event payload directly to get the right SHA."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        return ((event.get("pull_request") or {}).get("head") or {}).get("sha")
    except Exception:  # noqa: BLE001
        return None
