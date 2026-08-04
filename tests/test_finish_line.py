"""`agent_review finish` must be written on EVERY exit path.

The gap this closes: a review that logged `turn 1` and then nothing at
all. No wall-hit line, no finish line — indistinguishable in the logs
from a run still in flight, which makes "how many reviews died, and how"
unanswerable. Skips, cancellation and the SIGTERM guard all have to
close the stream, and the field set has to be identical across them
because downstream dashboards parse it positionally by name.
"""

from __future__ import annotations

import asyncio

import pytest

from cora.config import ReviewerConfig
from cora.providers.reporter import NullReporter
from cora.review import _output, run_review
from cora.review._preflight import build_run
from cora.review._state import ReviewRun


def _build_run() -> ReviewRun:
    return build_run(
        ReviewerConfig(repo="o/r", pr_number="42"),
        reporter=NullReporter(),
        retrieval=None,
        git=None,
        second_opinion=None,
    )


def _capture(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(_output, "_gha_log", lines.append)
    return lines


def _finish_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("agent_review finish ")]


_EXPECTED_FIELDS = (
    "pr_number=",
    "mode=",
    "turns=",
    "in_tokens=",
    "out_tokens=",
    "wall_s=",
    "terminated=",
    "resolved_model=",
    "leak=",
    "leak_retry=",
    "preamble_stripped=",
    "reasoning_stripped=",
    "tools=",
)


def test_emit_finish_is_idempotent_and_carries_the_full_field_set(monkeypatch):
    lines = _capture(monkeypatch)
    run = _build_run()

    _output.emit_finish(run, terminated_reason="wall_time")
    _output.emit_finish(run, terminated_reason="something_else")

    assert len(_finish_lines(lines)) == 1
    line = _finish_lines(lines)[0]
    for field in _EXPECTED_FIELDS:
        assert field in line, f"{field} missing — dashboards parse by name"
    assert "terminated=wall_time" in line


def test_early_skip_still_closes_the_stream(monkeypatch):
    """The `missing-pr-identity` skip returns from preflight, long
    before the output pipeline that used to own the finish line."""
    lines = _capture(monkeypatch)

    result = run_review(ReviewerConfig())

    assert result.terminated_reason == "missing-pr-identity"
    finish = _finish_lines(lines)
    assert len(finish) == 1
    assert "terminated=missing-pr-identity" in finish[0]


def test_cancellation_closes_the_stream_and_re_raises(monkeypatch):
    """The observed silent run. `CancelledError` is a `BaseException`,
    so an `except Exception` wrapper would not have caught it."""
    import cora.review as review

    lines = _capture(monkeypatch)
    monkeypatch.setattr(review, "preflight", lambda _run: None)

    def _cancelled(_run):
        raise asyncio.CancelledError()

    monkeypatch.setattr(review, "trigger_gate", _cancelled)

    with pytest.raises(asyncio.CancelledError):
        run_review(ReviewerConfig(repo="o/r", pr_number="42"))

    finish = _finish_lines(lines)
    assert len(finish) == 1
    assert "terminated=cancelled:CancelledError" in finish[0]


def test_hard_kill_closes_the_stream_before_the_check_run(monkeypatch):
    """SIGTERM re-raises with default disposition, so this path never
    returns through the pipeline wrapper and needs its own emit."""
    from cora.review._signals import _TIMEOUT_GUARD, _finalize_check_on_signal

    lines = _capture(monkeypatch)
    run = _build_run()
    monkeypatch.setitem(
        _TIMEOUT_GUARD,
        "finish",
        lambda: _output.emit_finish(run, terminated_reason="gha_timeout"),
    )
    # No reporter — the check-run half is a no-op, the log half is not.
    monkeypatch.setitem(_TIMEOUT_GUARD, "reporter", None)

    _finalize_check_on_signal()

    finish = _finish_lines(lines)
    assert len(finish) == 1
    assert "terminated=gha_timeout" in finish[0]


def test_hard_kill_survives_a_broken_finish_closure(monkeypatch):
    """The guard runs inside a signal handler with a SIGKILL close
    behind it — a failure here must not cost the check-run finalize."""
    from cora.review._signals import _TIMEOUT_GUARD, _finalize_check_on_signal

    def _boom():
        raise RuntimeError("nope")

    monkeypatch.setitem(_TIMEOUT_GUARD, "finish", _boom)
    monkeypatch.setitem(_TIMEOUT_GUARD, "reporter", None)

    _finalize_check_on_signal()  # must not raise
