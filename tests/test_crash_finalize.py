"""Crash finalize: an exception escaping the pipeline must not leave the
PR comment reading "in progress" (cora #14).

Every handled terminal path already finalizes the placeholder. An
exception did not — it unwound to `__main__`'s hard-failure boundary,
which prints and exits 1, leaving a comment that reads exactly like a
run still in flight. The progress check-run had the same hole; the
SIGTERM guard covers a kill, not a crash.
"""

from __future__ import annotations

import pytest

from cora.providers.reporter import Reporter
from cora.review import _finalize_surfaces_on_crash


class _SpyReporter(Reporter):
    def __init__(self, *, progress: bool, check: bool) -> None:
        self._progress = progress
        self._check = check
        self.skips: list[str] = []
        self.completions: list[dict] = []

    # ── surfaces under test ──
    @property
    def progress_open(self) -> bool:
        return self._progress

    @property
    def check_open(self) -> bool:
        return self._check

    def post_skip(self, reason: str) -> None:
        self.skips.append(reason)
        self._progress = False

    def complete_check(self, **kwargs) -> None:
        self.completions.append(kwargs)
        self._check = False

    # ── unused abstract surface ──
    def open_progress(self, head_sha: str) -> None: ...
    def post_in_progress(self) -> None: ...
    def write_summary(self, **kwargs) -> None: ...
    def post_review(self, result) -> None: ...
    def pause_automerge(self) -> bool:
        return False
    def dispatch_patch(self, **kwargs) -> dict:
        return {}
    def apply_label(self, *a, **k) -> None: ...


def test_finalizes_both_surfaces():
    r = _SpyReporter(progress=True, check=True)
    _finalize_surfaces_on_crash(r, RuntimeError("boom"))

    assert len(r.skips) == 1
    assert "errored (RuntimeError)" in r.skips[0]
    assert "re-push to retry" in r.skips[0]
    # Never leaves the reader thinking a verdict is still coming.
    assert "in progress" not in r.skips[0].lower()

    assert len(r.completions) == 1
    assert r.completions[0]["conclusion"] == "cancelled"
    assert r.completions[0]["terminated_reason"] == "crashed:RuntimeError"


def test_does_not_invent_a_comment_when_none_was_posted():
    """`update_run_comment` CREATES a comment when the run has none, so
    an unguarded post_skip would put a crash comment on quick-mode and
    early-exit runs that never had a placeholder. Guarded on
    `progress_open`, which is False in exactly those cases."""
    r = _SpyReporter(progress=False, check=True)
    _finalize_surfaces_on_crash(r, RuntimeError("boom"))
    assert r.skips == []
    assert len(r.completions) == 1


def test_skips_an_already_concluded_check():
    r = _SpyReporter(progress=True, check=False)
    _finalize_surfaces_on_crash(r, RuntimeError("boom"))
    assert len(r.skips) == 1
    assert r.completions == []


def test_none_reporter_is_a_noop():
    _finalize_surfaces_on_crash(None, RuntimeError("boom"))


@pytest.mark.parametrize("failing", ["post_skip", "complete_check"])
def test_a_failing_write_never_replaces_the_original_exception(failing, capsys):
    """This runs while an exception is already propagating — the original
    traceback is what matters, so neither write may raise."""

    class _Broken(_SpyReporter):
        pass

    r = _Broken(progress=True, check=True)
    setattr(
        r,
        failing,
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("github down")),
    )
    _finalize_surfaces_on_crash(r, ValueError("original"))
    assert "::warning::" in capsys.readouterr().out


def test_the_other_surface_still_finalizes_when_one_fails(capsys):
    r = _SpyReporter(progress=True, check=True)
    r.post_skip = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    _finalize_surfaces_on_crash(r, ValueError("original"))
    assert len(r.completions) == 1  # check still finalized
    assert "::warning::" in capsys.readouterr().out


def test_wired_into_the_pipelines_exception_path(monkeypatch):
    """The regression guard: an exception out of `_pipeline` must reach
    `_finalize_surfaces_on_crash` and still propagate. Unit-testing the
    helper alone would pass even if the call site were dropped."""
    import cora.review as review_mod
    from cora.config import ReviewerConfig

    spy = _SpyReporter(progress=True, check=True)

    async def _boom(run):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(review_mod, "_pipeline", _boom)

    with pytest.raises(RuntimeError, match="engine exploded"):
        review_mod.run_review(
            ReviewerConfig(repo="o/r", pr_number="42"), reporter=spy
        )

    assert len(spy.skips) == 1
    assert "errored (RuntimeError)" in spy.skips[0]
    assert spy.completions[0]["terminated_reason"] == "crashed:RuntimeError"


def test_base_reporter_progress_open_defaults_false():
    """`progress_open` is concrete, not abstract — a third-party Reporter
    predating #14 keeps subclassing, and False means the crash handler
    leaves its comments alone."""

    class _Old(Reporter):
        def open_progress(self, head_sha: str) -> None: ...
        @property
        def check_open(self) -> bool:
            return False
        def post_in_progress(self) -> None: ...
        def complete_check(self, **kwargs) -> None: ...
        def write_summary(self, **kwargs) -> None: ...
        def post_review(self, result) -> None: ...
        def post_skip(self, reason: str) -> None: ...
        def pause_automerge(self) -> bool:
            return False
        def dispatch_patch(self, **kwargs) -> dict:
            return {}
        def apply_label(self, *a, **k) -> None: ...

    assert _Old().progress_open is False
