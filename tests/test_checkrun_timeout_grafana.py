"""Pins the contract that the cora verdict check-run keeps
its Grafana drilldown link on every terminal path — including failure and
a hard `timeout-minutes` SIGTERM.

Background: the progress check is created via the REST API, so unlike the
workflow job's own check GitHub never auto-completes it when the job dies.
A hard kill used to leave it stuck `in_progress` — spinning dot, no
clickable Grafana link, and a `required` aggregator that waits forever for
a terminal state. The reviewer entrypoint now installs a SIGTERM
finalizer (`_on_sigterm` → `_finalize_check_on_signal`) that PATCHes the
check to `timed_out` with the link intact.

Two layers:
  - behavioural: `update_check_run_completed` (the package helper every
    finalize path funnels through) always emits the Grafana link, for any
    conclusion. This is the actual guarantor.
  - source guard: the entrypoint wires SIGTERM → finalize with a tolerated
    `timed_out` conclusion and an idempotence flag. Cheap regex check —
    importing the entrypoint pulls openai/pydantic-ai, too heavy here.
"""
from __future__ import annotations

import json

import pytest


def _capture_patch_payload(monkeypatch, fail_first=False):
    """Monkeypatch `subprocess.run` inside agent_review.check_run to
    capture each `gh api` invocation (payload + the GH_TOKEN it ran
    under) and return a success proc. With `fail_first=True` the first
    attempt returns rc=1 so the App→GITHUB_TOKEN fallback path is
    exercised. Returns the capture list of {payload, gh_token} dicts."""
    from cora.core import check_run

    captured: list[dict] = []

    class _Proc:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = '{"id": 1, "html_url": "https://x/checks/1"}'
            self.stderr = "boom" if rc else ""

    def _fake_run(cmd, input=None, capture_output=None, text=None, env=None):  # noqa: A002
        captured.append({
            "payload": json.loads(input),
            "gh_token": (env or {}).get("GH_TOKEN"),
        })
        rc = 1 if (fail_first and len(captured) == 1) else 0
        return _Proc(rc)

    monkeypatch.setattr(check_run.subprocess, "run", _fake_run)
    return captured


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "cancelled", "success"])
def test_grafana_link_present_on_every_conclusion(monkeypatch, conclusion):
    """`update_check_run_completed` must include the per-PR Grafana
    drilldown URL in the summary regardless of conclusion — a failed or
    timed-out review is exactly when the operator wants to click through
    to the agent-loop logs."""
    from cora.core.budget import Budget
    from cora.core.check_run import update_check_run_completed

    # A configured Grafana host (deployments set GRAFANA_BASE_URL; the
    # helper reads check_run's module-level constant).
    monkeypatch.setattr("cora.core.check_run.GRAFANA_BASE", "https://grafana.example.com")
    monkeypatch.setattr("cora.core.check_run.DASHBOARD_PATH", "/d/pr-review-detail")
    captured = _capture_patch_payload(monkeypatch)
    update_check_run_completed(
        repo="owner/repo",
        check_id="123",
        pr_number="2213",
        verdict_line="timed out before producing a verdict",
        conclusion=conclusion,
        budget=Budget(max_input=0, max_output=0, max_iterations=0),
        wall_time_s=463.6,
        terminated_reason="gha_timeout",
    )

    assert captured, "update_check_run_completed never called gh api"
    summary = captured[-1]["payload"]["output"]["summary"]
    assert "https://grafana.example.com" in summary, f"Grafana link dropped on {conclusion}"
    assert "var-pr_number=2213" in summary, "PR-scoped drilldown var missing"
    assert captured[-1]["payload"]["conclusion"] == conclusion


def test_grafana_link_present_on_initial_in_progress(monkeypatch):
    """The in-progress create also carries the link, so even a check left
    mid-flight (e.g. SIGKILL before the SIGTERM grace finishes) still has
    a clickable Grafana URL from creation time."""
    from cora.core.check_run import create_check_run

    monkeypatch.setattr("cora.core.check_run.GRAFANA_BASE", "https://grafana.example.com")
    monkeypatch.setattr("cora.core.check_run.DASHBOARD_PATH", "/d/pr-review-detail")
    captured = _capture_patch_payload(monkeypatch)
    create_check_run("owner/repo", "2213", "deadbeef")

    assert captured, "create_check_run never called gh api"
    summary = captured[-1]["payload"]["output"]["summary"]
    assert "https://grafana.example.com" in summary
    assert "var-pr_number=2213" in summary


def test_grafana_link_omitted_when_unconfigured(monkeypatch):
    """Public default has no Grafana host (`GRAFANA_BASE` empty): the
    reporter must omit the drilldown link, not emit a broken empty-href
    `[grafana]()` / `var-pr_number` with no base."""
    from cora.core.budget import Budget
    from cora.core.check_run import create_check_run, update_check_run_completed

    monkeypatch.setattr("cora.core.check_run.GRAFANA_BASE", "")
    captured = _capture_patch_payload(monkeypatch)
    create_check_run("owner/repo", "2213", "deadbeef")
    update_check_run_completed(
        repo="owner/repo",
        check_id="123",
        pr_number="2213",
        verdict_line="looks good",
        conclusion="success",
        budget=Budget(max_input=0, max_output=0, max_iterations=0),
        wall_time_s=1.0,
        terminated_reason=None,
    )

    assert captured
    for c in captured:
        summary = c["payload"]["output"]["summary"]
        assert "grafana" not in summary.lower(), "grafana link not omitted when unconfigured"
        assert "var-pr_number" not in summary


def test_checkrun_prefers_cora_app_token(monkeypatch):
    """Bucketing fix: when CORA_GH_TOKEN is set, the check-runs POST
    must run under it (not the ambient GITHUB_TOKEN) so the check lands
    in the App's own check suite and groups consistently in the PR rollup
    instead of a random sibling github-actions workflow's suite."""
    from cora.core.check_run import create_check_run

    monkeypatch.setenv("CORA_GH_TOKEN", "app-tok-xyz")
    monkeypatch.setenv("GH_TOKEN", "ghs-default")
    captured = _capture_patch_payload(monkeypatch)
    create_check_run("owner/repo", "2213", "deadbeef")

    assert len(captured) == 1, "should succeed on the first (App-token) attempt"
    assert captured[0]["gh_token"] == "app-tok-xyz", (
        "check-runs call did not use the cora App token — it'll land in "
        "a random github-actions suite again"
    )


def test_checkrun_falls_back_to_github_token_on_app_failure(monkeypatch):
    """Regression guard: if the App token attempt fails (e.g. the App
    lacks `checks: write`), the call retries under GITHUB_TOKEN so the
    progress check is never silently dropped."""
    from cora.core.check_run import create_check_run

    monkeypatch.setenv("CORA_GH_TOKEN", "app-tok-xyz")
    monkeypatch.setenv("GH_TOKEN", "ghs-default")
    captured = _capture_patch_payload(monkeypatch, fail_first=True)
    check_id, _ = create_check_run("owner/repo", "2213", "deadbeef")

    assert [c["gh_token"] for c in captured] == ["app-tok-xyz", "ghs-default"], (
        "App-token failure must fall back to GITHUB_TOKEN"
    )
    assert check_id == "1", "fallback attempt's success must still yield the check id"


def test_checkrun_uses_github_token_when_no_app_token(monkeypatch):
    """With no App token minted (mint soft-failed / App unconfigured),
    the single attempt runs under the inherited GITHUB_TOKEN — pre-App
    behaviour, no extra call."""
    from cora.core.check_run import create_check_run

    monkeypatch.delenv("CORA_GH_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghs-default")
    captured = _capture_patch_payload(monkeypatch)
    create_check_run("owner/repo", "2213", "deadbeef")

    assert len(captured) == 1
    assert captured[0]["gh_token"] == "ghs-default"
