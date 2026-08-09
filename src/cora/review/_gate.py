"""Trigger-security gate — who may fire a review, at what cost.

The policy itself lives dependency-free in `cora.trigger`; this module is
the orchestrator-side glue: the GHA rate-cap probe and (in the run
pipeline) the gate evaluation that can deny a review or force the
comment-only ceiling before any content fetch, prewarm, or model spend.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta

from cora.core import pr_context as _prc
from cora.core.log import _gha_log
from cora.result import ReviewResult
from cora.review._state import ReviewRun
from cora.trigger import TriggerContext
from cora.trigger import evaluate as evaluate_trigger


def _recent_run_counts(repo: str, author: str) -> tuple[int | None, int | None]:
    """(total, by-`author`) runs of THIS workflow in the last hour — the
    counts `TriggerPolicy`'s rate caps compare against. Identified via
    `GITHUB_WORKFLOW` (the workflow's `name:`); off GHA, or on any API
    error, returns (None, None) and the caps are skipped — the caps are
    a best-effort cost guard, not a security boundary (SECURITY.md)."""
    workflow = os.environ.get("GITHUB_WORKFLOW", "").strip()
    if not workflow or not repo:
        return None, None
    since = (
        datetime.now(UTC) - timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        raw = subprocess.run(
            [
                "gh", "api",
                f"/repos/{repo}/actions/runs?created=%3E%3D{since}&per_page=100",
                "--jq",
                '[.workflow_runs[] | {name, actor: (.actor.login // "")}]',
            ],
            capture_output=True, text=True, check=True,
        ).stdout
        runs = [r for r in json.loads(raw) if r.get("name") == workflow]
        total = len(runs)
        by_author = sum(
            1 for r in runs if r["actor"].lower() == (author or "").lower()
        )
        return total, by_author
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::trigger rate-cap probe failed: {exc}")
        return None, None


def trigger_gate(run: ReviewRun) -> ReviewResult | None:
    """Trigger-security gate — evaluated before any content
    fetch, prewarm, or model spend. Enforcement is config-driven and
    default-on (`TriggerPolicy.enforce`); the association lookup and
    rate-cap probe only fire when enforced, so explicit opt-out preserves
    the legacy API-call profile. A degraded (comment-only) decision
    forces quick mode here; propose_patch dispatch is suppressed at its
    call site in `_patches`."""
    cfg = run.cfg
    if not cfg.trigger.enforce:
        return None

    metadata = run.metadata
    trigger_author = ((metadata.get("author") or {}).get("login") or "")
    runs_total = runs_by_author = None
    if (
        cfg.trigger.max_runs_per_hour is not None
        or cfg.trigger.max_runs_per_author_per_hour is not None
    ):
        runs_total, runs_by_author = _recent_run_counts(run.repo, trigger_author)
    trigger_decision = evaluate_trigger(
        cfg.trigger,
        TriggerContext(
            author=trigger_author,
            author_association=_prc.fetch_author_association(
                run.repo, run.pr_number
            ),
            labels=frozenset(
                ((lbl.get("name") or "") if isinstance(lbl, dict) else str(lbl))
                for lbl in (metadata.get("labels") or [])
            ),
            is_fork=_prc.is_fork_pr(metadata),
            recent_runs_total=runs_total,
            recent_runs_by_author=runs_by_author,
        ),
    )
    run.iter_log(
        f"trigger gate: allowed={str(trigger_decision.allowed).lower()} "
        f"degraded={str(trigger_decision.degraded).lower()} "
        f"({trigger_decision.reason})"
    )
    if not trigger_decision.allowed:
        # `cancelled` (not `neutral`) so a verdict-gating required-check
        # aggregator stays red: a policy-denied PR must not pass the
        # merge gate unreviewed.
        try:
            run.reporter.post_skip(f"Trigger policy: {trigger_decision.reason}.")
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::could not post skip comment: {exc}")
        run.reporter.complete_check(
            verdict_line="skipped (trigger policy)",
            conclusion="cancelled",
            budget=None,
            wall_time_s=0.0,
            terminated_reason="trigger-policy",
        )
        return run.skip_result("skipped (trigger policy)", "trigger-policy")
    if trigger_decision.degraded and not run.is_quick:
        # Comment-only ceiling: quick mode is the no-tools path (no
        # MCP connect, no local repo tools); propose_patch dispatch
        # is suppressed at its call site in `_patches`.
        _gha_log("trigger gate: comment-only ceiling — forcing quick mode")
        run.max_iterations = 0
        run.is_quick = True
        run.mode = "quick"
    run.trigger_degraded = trigger_decision.degraded
    return None
