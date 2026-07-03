"""GitHub Check-Run lifecycle + Grafana drilldown URL helpers.

Surfaces live review status on the PR's status indicator without waiting
for the comment edit. `create_check_run` posts an in-progress run on
workflow start; `update_check_run_completed` PATCHes it to a terminal
conclusion mapped from the parsed verdict.

`_workflow_run_url` / `_grafana_drilldown_url` are reused by the comment
module for the same set of "live progress" deeplinks.
"""

from __future__ import annotations

import json
import os
import subprocess

from cora.core.budget import Budget

from cora.core.config import CHECK_RUN_NAME, GRAFANA_BASE, DASHBOARD_PATH


def _gh_check_api(args: list[str], payload: dict) -> subprocess.CompletedProcess:
    """Run a `gh api` check-runs call, preferring the cora App-minted
    token (CORA_GH_TOKEN) so the check lands in the App's *own* check
    suite. Check runs created via the REST API with the ambient
    GITHUB_TOKEN attach to an arbitrary github-actions check suite for
    the head SHA — GitHub Actions makes one suite per workflow run, so
    the progress check shows up grouped under a random sibling workflow
    (`secrets-hygiene`, `required`, …) in the PR rollup. A GitHub App
    gets a single dedicated suite, so the App token groups it
    consistently under the App's name.

    Falls back to the inherited GITHUB_TOKEN when the App token is absent
    OR when the App attempt fails (e.g. the App lacks `checks: write` —
    an optional permission). The fallback preserves the pre-App
    behaviour exactly, so this can never silently drop the progress check.
    Permission state is stable within a run, so create + update + the
    SIGTERM finalize all resolve to the same token → same owning App →
    the PATCH always targets a check the resolver is allowed to update.
    """
    app_token = os.environ.get("CORA_GH_TOKEN", "").strip()
    envs = []
    if app_token:
        app_env = os.environ.copy()
        app_env["GH_TOKEN"] = app_token
        envs.append(("cora App", app_env))
    envs.append(("GITHUB_TOKEN", os.environ.copy()))

    proc = None
    for i, (label, env) in enumerate(envs):
        proc = subprocess.run(
            args, input=json.dumps(payload),
            capture_output=True, text=True, env=env,
        )
        if proc.returncode == 0:
            return proc
        # More attempts left → log the fall-through and retry; the last
        # attempt's failure is returned for the caller to surface.
        if i < len(envs) - 1:
            print(
                f"::warning::check-runs call via {label} failed "
                f"(rc={proc.returncode}); falling back: {proc.stderr.strip()}"
            )
    return proc


def _workflow_run_url() -> str | None:
    """Reconstruct the GHA workflow-run URL from env. Returns None when
    not running under Actions (e.g., local invocation)."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not (server and repo and run_id):
        return None
    return f"{server}/{repo}/actions/runs/{run_id}"


def _grafana_drilldown_url(pr_number: str) -> str:
    """Deeplink to a per-PR review-detail dashboard with the pr_number
    variable pre-set.

    Forces `refresh=5s` so the page auto-updates as the run progresses
    even if the user's account default disables auto-refresh, and
    widens the time range to `now-30m` so the run's iter events are
    visible immediately instead of needing a manual time-picker tweak.

    Returns "" when no Grafana host or dashboard path is configured
    (`GRAFANA_BASE` / `DASHBOARD_PATH` empty) so callers omit the
    drilldown link entirely."""
    if not GRAFANA_BASE or not DASHBOARD_PATH:
        return ""
    return (
        f"{GRAFANA_BASE}{DASHBOARD_PATH}"
        f"?var-pr_number={pr_number}&refresh=5s&from=now-30m&to=now"
    )


def create_check_run(
    repo: str,
    pr_number: str,
    head_sha: str,
    *,
    check_run_name: str = CHECK_RUN_NAME,
) -> tuple[str | None, str | None]:
    """POST a check run in `in_progress` state. Returns (check_id, html_url)
    on success, (None, None) on failure (soft-fail — the comment + step
    summary are still useful even if this fails)."""
    grafana_url = _grafana_drilldown_url(pr_number)
    run_url = _workflow_run_url()
    summary_parts = []
    if run_url:
        summary_parts.append(f"**[workflow logs]({run_url})** — GHA run page")
    if grafana_url:
        summary_parts.append(
            f"**[grafana]({grafana_url})** — live agent-loop logs scoped to PR #{pr_number}"
        )
    summary = "\n\n".join(summary_parts)
    # This check IS the authoritative verdict a deployment's required-check
    # aggregator can gate merges on — a non-success/neutral
    # conclusion blocks the merge fail-closed. The engine-posted name must
    # stay distinct from the workflow job's own check, which such a gate
    # should deliberately EXCLUDE (its skipped no-op runs, spawned by every
    # non-review `labeled` event, would shadow the real verdict under
    # `max_by(started_at)`). Deployments rebrand via
    # `ReviewerConfig.check_run_name`.
    payload = {
        "name": check_run_name,
        "head_sha": head_sha,
        "status": "in_progress",
        "output": {
            "title": "Reviewing PR…",
            "summary": summary,
        },
    }
    try:
        proc = _gh_check_api(
            ["gh", "api", "-X", "POST", f"repos/{repo}/check-runs", "--input", "-"],
            payload,
        )
        if proc.returncode != 0:
            print(f"::warning::check run create failed: {proc.stderr.strip()}")
            return None, None
        data = json.loads(proc.stdout)
        return str(data.get("id") or ""), data.get("html_url")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::check run create raised: {exc}")
        return None, None


def update_check_run_completed(
    repo: str,
    check_id: str,
    pr_number: str,
    verdict_line: str | None,
    conclusion: str,
    budget: Budget,
    wall_time_s: float,
    terminated_reason: str | None,
) -> None:
    """PATCH the check run into `completed` state with conclusion +
    verdict-bearing title. Title shows up on the PR status dot's
    hover/click; summary shows in the Checks tab."""
    grafana_url = _grafana_drilldown_url(pr_number)
    run_url = _workflow_run_url()
    title_bits = ["cora review"]
    if verdict_line:
        title_bits.append(verdict_line)
    title = " — ".join(title_bits)
    summary_parts = [
        f"**{budget.iterations}** tool calls · "
        f"**{budget.input_used:,}** in / **{budget.output_used:,}** out tokens · "
        f"**{wall_time_s:.1f}s** wall",
    ]
    if budget.resolved_model:
        summary_parts.append(f"Backend: `{budget.resolved_model}`")
    if terminated_reason:
        summary_parts.append(f"Terminated: {terminated_reason}")
    link_bits = []
    if run_url:
        link_bits.append(f"[logs]({run_url})")
    if grafana_url:
        link_bits.append(f"[grafana]({grafana_url})")
    if link_bits:
        summary_parts.append(" · ".join(link_bits))
    summary = "\n\n".join(summary_parts)
    payload = {
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": title, "summary": summary},
    }
    try:
        proc = _gh_check_api(
            ["gh", "api", "-X", "PATCH",
             f"repos/{repo}/check-runs/{check_id}",
             "--input", "-"],
            payload,
        )
        if proc.returncode != 0:
            print(f"::warning::check run update failed: {proc.stderr.strip()}")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::check run update raised: {exc}")
