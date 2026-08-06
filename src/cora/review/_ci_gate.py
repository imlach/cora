"""Finalize-time CI-verdict gate — the backstop half of cora issue #23.

`context_refresher.py`'s green-delta injection (see its module docstring)
is the PRIMARY fix: while the agent loop is still running, a check-run
turning green gets pushed into context so the model can re-verify or
downgrade its own compile/test-failure claim before it ever posts. That
fix has a gap — a review that finishes BEFORE the relevant check-run
does never sees an injection at all, because there are no more turns
left to inject into. The motivating case: a deep review ran concurrently
with the build, finished first, and posted two 🚨 Blocker findings
("this won't compile") that CI contradicted a few minutes later. The
`needs changes` verdict was driven entirely by those two findings.

This module runs once, from `produce_output` — after the verdict is
parsed, before the verdict check-run posts. When the settled verdict is
the block-severity word (`needs changes` by default), it does ONE bounded
`gh api` re-poll of check-runs for the REVIEWED HEAD SHA (`run.head_sha`
— the SHA the diff in the initial prompt was actually built from; a
green run for an OLDER SHA proves nothing about this review, which is
why the poll is keyed to that exact SHA rather than "the PR's current
head"). If every relevant check-run is green for that SHA, blocker
findings whose text matches a conservative compile/test-failure claim
pattern get a visible harness note appended, and — only when EVERY
blocker in the review matches — the verdict downgrades one step.

Design choices worth knowing:

- **Annotate, never delete.** A contradicted-looking finding still
  posts, with a note attached, so a human can judge it — the same
  "friction is the point" posture `_finalize.py` uses for the
  automerge pause. The regex is a narrow, documented heuristic; the
  failure mode of missing a real contradiction (false negative, finding
  posts unannotated) is preferred over annotating a legitimate blocker
  (false positive).
- **Whole-review downgrade requires whole-review contradiction.** One
  matched claim among several distinct blockers doesn't downgrade the
  verdict — a PR can have both a CI-contradicted claim AND a real,
  independent blocker in the same review.
- **Soft-fail on any API error** — an unreachable `gh api` call leaves
  the review exactly as the model produced it. This gate can only make
  a `needs changes` verdict less severe, never more; failing open in
  that direction can't newly block a merge, so soft-failing to "do
  nothing" is the safe default in both directions.
- **Runs BEFORE the check-run posts, so a downgrade reaches GitHub.**
  `Reporter.complete_check` is a first-write-wins gate (an invariant
  the SIGTERM guard relies on), so this gate cannot re-open an already
  posted conclusion — instead `produce_output` calls it between the
  verdict parse and `_finalize_observability`, making the first (and
  only) check-run write carry the post-gate verdict. A fully
  contradicted review therefore posts a non-blocking required check,
  not a red one. The automerge pause is deliberately NOT changed:
  `_finalize.py`'s `detect_blocker` scans for the literal
  `🚨 **Blocker:**` marker text, which this gate leaves in place (see
  "Annotate, never delete" above), so a downgraded review still pauses
  automerge for human judgment — friction is the point there, and
  lifting it belongs to the human who reads the annotated findings,
  not to this heuristic.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from cora.core.leak import _BLOCKER_LINE_RE
from cora.core.log import _gha_log
from cora.review._state import ReviewRun

# One bounded call, capped wall time — "bounded re-poll" means both "at
# most once" (enforced by only calling this from one place, unconditionally
# once per finalize) and "can't hang the finalize path" (enforced by this
# timeout). A stuck `gh api` call must not delay posting the review.
_GH_API_TIMEOUT_S = 15

# Check-run states that mean "this run is not evidence the code is fine" —
# same vocabulary `pr_context.gather_ci_context` uses for the failing side.
_BAD_CONCLUSIONS = frozenset({"failure", "timed_out", "cancelled", "action_required"})

# A single blocker bullet, exactly as the prompt's Output format asks for
# it (`- 🚨 **Blocker:** <text>`). Matched per-line so multiple blockers in
# one review are annotated independently.
_BLOCKER_BULLET_RE = re.compile(r"^-\s*🚨\s*\*\*Blocker:\*\*.*$", re.MULTILINE)

# Conservative compile/test-failure claim pattern — deliberately narrow.
# Only phrasings that read as a specific, falsifiable "this won't build /
# this won't pass" assertion; generic severity words ("bug", "broken",
# "wrong") are excluded on purpose. A false negative here just means the
# harness note doesn't get added and the finding posts as the model wrote
# it; a false positive would annotate a LEGITIMATE blocker as
# CI-contradicted, which is the worse failure mode — bias the pattern
# toward under-matching.
_CI_CONTRADICTION_CLAIM_RE = re.compile(
    r"""
    won.t\ compile              # "won't compile" / "wont compile"
    | compil\w*\s+error         # "compile error" / "compilation error"
    | fails?\ in\ CI            # "fails in CI" / "fail in CI"
    | tests?\ will\ fail        # "test(s) will fail"
    | undefined\ symbol         # "undefined symbol"
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONTRADICTION_NOTE = (
    " _(harness note: CI for this SHA passed — this claim appears "
    "contradicted; treat as a question.)_"
)


def _fetch_check_runs(repo: str, head_sha: str) -> list[dict] | None:
    """One bounded `gh api` GET for `head_sha`'s check-runs. Returns
    None on ANY failure (network, auth, timeout, malformed JSON) — the
    caller treats None as "no evidence either way", never as "green"."""
    try:
        proc = subprocess.run(
            [
                "gh", "api", "-H", "Accept: application/vnd.github+json",
                f"/repos/{repo}/commits/{head_sha}/check-runs?per_page=100",
            ],
            capture_output=True,
            text=True,
            timeout=_GH_API_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout).get("check_runs") or []
    except Exception:  # noqa: BLE001 — soft-fail is the whole point here
        return None


def _is_own_workflow_run(cr: dict) -> bool:
    """True when this check-run belongs to the Actions run we are
    executing inside.

    Name-based exclusion isn't enough. The reviewer publishes TWO
    check-runs: the verdict check (`cfg.check_run_name`) and the
    workflow JOB itself, whose name is whatever the workflow calls it —
    `review` in the canonical wiring, anything in an adopter's. That job
    is necessarily `in_progress` while this gate runs, so leaving it in
    the relevant set means `_all_green` is permanently False and the
    gate can never fire. Matching on `GITHUB_RUN_ID` in the check's URL
    identifies it structurally, whatever it was named."""
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if not run_id:
        return False
    urls = f"{cr.get('html_url') or ''} {cr.get('details_url') or ''}"
    return f"/runs/{run_id}" in urls or f"/runs/{run_id}/" in urls


def _relevant_check_runs(check_runs: list[dict], *, own_check: str) -> list[dict]:
    """Latest run per check name, excluding the reviewer's own verdict
    check (+ its pre-rename prefix), its own workflow job, and the
    `required` aggregator — same exclusion `gather_ci_context` applies,
    for the same reason: none is evidence about the PR's build/test
    state."""
    latest: dict[str, dict] = {}
    for cr in check_runs:
        name = cr.get("name", "")
        prev = latest.get(name)
        if prev is None or (cr.get("started_at") or "") > (prev.get("started_at") or ""):
            latest[name] = cr
    return [
        cr
        for name, cr in latest.items()
        if not name.startswith("agentic-pr-review")
        and name != own_check
        and name != "required"
        and not _is_own_workflow_run(cr)
    ]


def _all_green(check_runs: list[dict]) -> bool:
    """True only when there's at least one relevant check AND every one
    of them is `completed` with conclusion `success`. Empty is NOT green —
    no relevant checks means no evidence to contradict a claim with, so
    the gate must not annotate on silence. A still-`in_progress`/`queued`
    check is also not green — the gate only fires once CI has actually
    finished, not while it's still racing the review.

    `success` specifically, not merely "not failed": a `skipped` job
    (path filter, `if:` gate, matrix exclude) or a `neutral`/`stale`
    conclusion NEVER RAN, so it is not evidence the code compiles. This
    is the burden the gate exists to respect — it suppresses findings,
    so silence must never read as proof. Matches the green-delta
    injection's own `conclusion == "success"` test in
    `context_refresher._newly_green_checks`."""
    if not check_runs:
        return False
    return all(
        cr.get("status") == "completed" and cr.get("conclusion") == "success"
        for cr in check_runs
    )


def _annotate_contradicted_blockers(body: str) -> tuple[str, int, int]:
    """Append the harness note to every blocker bullet whose text
    matches the narrow claim pattern. Returns `(new_body, total_blockers,
    contradicted_count)` — the caller downgrades the verdict only when
    the two counts are equal and non-zero (EVERY blocker matched).

    `total_blockers` is counted with `leak._BLOCKER_LINE_RE`, the SAME
    pattern `detect_blocker` uses to decide a review blocks, NOT the
    bullet pattern used for annotation. The bullet form
    (`- 🚨 **Blocker:** …`) is what the prompt asks for, but a model
    that writes `* `, omits the dash, or puts the colon outside the bold
    still produced a blocker — and if such a finding were invisible to
    the count, one contradicted bullet could equal "every blocker" and
    downgrade a review whose real blocker was never even looked at.
    Counting wide and annotating narrow makes the mismatch fail closed:
    the counts differ, so no downgrade."""
    total = len(_BLOCKER_LINE_RE.findall(body))
    contradicted = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal contradicted
        line = m.group(0)
        if _CI_CONTRADICTION_CLAIM_RE.search(line):
            contradicted += 1
            return line + _CONTRADICTION_NOTE
        return line

    new_body = _BLOCKER_BULLET_RE.sub(_sub, body)
    return new_body, total, contradicted


def _downgrade_verdict(body: str, *, glyphs: tuple[str, str, str], words: tuple[str, str, str]) -> str:
    """Replace the leading verdict line with the one-step-down entry
    (`glyphs[1]`/`words[1]`, `🟡 minor` by default) and insert an
    explanatory line right after it. The body's first line is the
    verdict marker by construction (`detect_reasoning_leak` guarantees
    the posted body starts there) — swapping it plus one inserted line
    keeps every finding, including the now-annotated ones, untouched."""
    _, _, rest = body.partition("\n")
    new_verdict_line = f"{glyphs[1]} {words[1]}"
    explainer = (
        "_(harness note: verdict downgraded from "
        f"{glyphs[2]} {words[2]} — every 🚨 Blocker finding below matches "
        "a compile/test-failure claim contradicted by CI, which passed "
        "for this HEAD SHA. Findings are kept below for human judgment.)_"
    )
    return f"{new_verdict_line}\n\n{explainer}\n{rest.lstrip(chr(10))}"


async def apply_ci_verdict_gate(run: ReviewRun) -> None:
    """Run the gate for one review. No-op (leaves `run.body_to_post` /
    `run.verdict` untouched) unless every one of these holds:

    1. the gate isn't killswitched off (`cfg.ci_verdict_gate`)
    2. this isn't an eval-mode run (no live re-poll against historical
       PRs — same posture `dispatch_patches` uses for its own network
       writes in eval mode)
    3. the settled verdict is the block-severity word (`needs changes`)
    4. the review has a known head SHA to poll against
    5. the `gh api` re-poll succeeds
    6. every relevant check-run for that SHA is green

    Only step 6 onward touches the body; steps 1-5 are cheap early-outs.
    """
    cfg = run.cfg
    if not cfg.ci_verdict_gate:
        return
    if run.eval_mode:
        return
    words = cfg.verdict_words
    if (run.verdict or "").lower() != words[2].lower():
        return
    head_sha = run.head_sha
    if not head_sha:
        return

    check_runs = _fetch_check_runs(run.repo, head_sha)
    if check_runs is None:
        _gha_log(
            f"ci_verdict_gate pr_number={run.pr_number} outcome=soft-fail "
            "reason=check-runs-fetch-failed"
        )
        return

    relevant = _relevant_check_runs(check_runs, own_check=cfg.check_run_name)
    if not _all_green(relevant):
        _gha_log(
            f"ci_verdict_gate pr_number={run.pr_number} outcome=skip "
            f"reason=not-all-green relevant_checks={len(relevant)}"
        )
        return

    new_body, total, contradicted = _annotate_contradicted_blockers(run.body_to_post)
    if contradicted == 0:
        _gha_log(
            f"ci_verdict_gate pr_number={run.pr_number} outcome=skip "
            f"reason=no-matching-claims total_blockers={total}"
        )
        return

    downgraded = contradicted == total
    if downgraded:
        new_body = _downgrade_verdict(new_body, glyphs=cfg.verdict_glyphs, words=words)
        run.verdict = words[1].lower()

    run.body_to_post = new_body
    _gha_log(
        f"ci_verdict_gate pr_number={run.pr_number} outcome=annotated "
        f"total_blockers={total} contradicted={contradicted} "
        f"downgraded={'true' if downgraded else 'false'}"
    )
    run.loki(
        f"agent_review ci_verdict_gate pr_number={run.pr_number} "
        f"total_blockers={total} contradicted={contradicted} "
        f"downgraded={'true' if downgraded else 'false'}",
        labels={"consumer": "pr-review", "kind": run.mode},
    )
