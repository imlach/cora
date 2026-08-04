"""SIGTERM safety net for the progress check-run.

GHA `timeout-minutes` (and concurrency cancel-in-progress) deliver
SIGTERM, then SIGKILL after a short grace window. The progress check is
created via the REST API, so GitHub never auto-completes it when the job
dies. The guard PATCHes it to a tolerated terminal conclusion
(`timed_out`) via the Reporter; idempotency lives in the Reporter
(`complete_check`: first terminal conclusion wins).

Module state by design: a signal handler has no call context, so the
live reporter/budget/start are parked in `_TIMEOUT_GUARD` (one review
per process). Tests reach the dict through the `cora.review` re-export —
it is the same object.
"""

from __future__ import annotations

import os
import signal
import time

from cora.core.budget import Budget
from cora.providers.reporter import Reporter

_TIMEOUT_GUARD: dict[str, object] = {
    "reporter": None,
    "budget": None,
    "start": None,
    "terminated_reason": None,
    # `() -> None` closure that writes the `agent_review finish` line.
    # Parked here because this is the one exit that never returns
    # through the pipeline wrapper: the handler re-raises SIGTERM with
    # default disposition and the process is gone.
    "finish": None,
}


def _finalize_check_on_signal() -> None:
    """Finalize the in-progress check to `timed_out` through the Reporter.
    Idempotent via `Reporter.complete_check` (first terminal wins), so a
    normal terminal path that already concluded the check isn't clobbered
    by a late signal. Split out from `_on_sigterm` so the finalize logic
    is unit-testable without the process-killing re-raise."""
    g = _TIMEOUT_GUARD
    # Close the log stream first — cheapest of the two, and the one whose
    # absence makes a hard kill look like a run that never ended.
    finish = g.get("finish")
    if callable(finish):
        try:
            finish()
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::SIGTERM finish line failed: {exc}")
    reporter = g.get("reporter")
    if reporter is None or not reporter.check_open:
        return
    start = g.get("start")
    wall_s = (time.monotonic() - start) if isinstance(start, float) else 0.0
    budget = g.get("budget")
    try:
        reporter.complete_check(
            verdict_line="timed out before producing a verdict",
            # In the merge gate's tolerated set — a hard wall-hit is a
            # soft-fail, same as the in-band terminated_reason=wall_time
            # path, so it doesn't gate auto-merge.
            conclusion="timed_out",
            budget=budget if isinstance(budget, Budget) else None,
            wall_time_s=wall_s,
            terminated_reason=g.get("terminated_reason") or "gha_timeout",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::SIGTERM check finalize failed: {exc}")


def _on_sigterm(signum, frame):  # noqa: ARG001
    """Finalize the in-progress check-run on a hard kill, then re-raise
    SIGTERM with default disposition so the process exits. Must stay
    cheap (one PATCH) to fit GHA's SIGTERM→SIGKILL grace window."""
    _finalize_check_on_signal()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def _arm_timeout_guard(reporter: Reporter) -> None:
    """Register the reporter with the SIGTERM finalizer once its progress
    check exists. No-op off the main thread (signal handlers can't
    install there) — soft-fails like every other live-progress surface."""
    _TIMEOUT_GUARD.update(
        reporter=reporter,
        budget=None,
        start=None,
        terminated_reason=None,
        finish=None,
    )
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError) as exc:  # not main thread / unsupported
        print(f"::warning::could not install SIGTERM check guard: {exc}")
