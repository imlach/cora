"""Unit tests for the `SecondOpinion` seam.

Covers the public ABC's defaults — `NullSecondOpinion` is inert and
`SecondOpinionResult` carries the empty "nothing dispatched" shape — and
the config selection that keeps the built-in T2 impl as the
byte-identical default. The T2 impl's resolve/compose behaviour is exercised
end-to-end in `test_run_review.py` (which monkeypatches the alt-reviewer
call) and unit-tested in `test_disagreement.py` / `test_phase_d_t2_dispatch.py`.
"""

from __future__ import annotations

import asyncio

from cora.config import ReviewerConfig
from cora.second_opinion import (
    NullSecondOpinion,
    SecondOpinionProvider,
    SecondOpinionResult,
)


def _cfg(**overrides) -> ReviewerConfig:
    base = {"repo": "o/r", "pr_number": "1", "llm_api_key": "k", "model": "m"}
    base.update(overrides)
    return ReviewerConfig(**base)


def test_result_defaults_are_the_not_dispatched_shape():
    r = SecondOpinionResult()
    assert r.dispatched is False
    assert r.body is None
    assert r.terminated_reason is None
    assert r.tools_available == []
    assert r.model_alias is None
    assert r.verdict is None
    assert r.has_blocker is False
    assert r.resolution is None


def test_null_should_dispatch_always_false():
    null = NullSecondOpinion()
    assert null.should_dispatch(
        cfg=_cfg(t2_disagreement=True), is_quick=False, primary_body="🟢 ok"
    ) is False


def test_null_compose_returns_primary_unchanged():
    null = NullSecondOpinion()
    body = null.compose(
        result=SecondOpinionResult(),
        cfg=_cfg(),
        is_quick=False,
        primary_body_to_post="🟢 looks good\n\nbody",
        primary_terminated_reason=None,
        pr_number="1",
        log=lambda _l: None,
        iter_log=lambda _l: None,
    )
    assert body == "🟢 looks good\n\nbody"


def test_null_dispatch_is_inert():
    null = NullSecondOpinion()
    r = asyncio.run(null.dispatch())
    assert r.dispatched is False


def test_null_emit_events_is_noop():
    null = NullSecondOpinion()
    # Should not raise and should emit nothing through the log callback.
    emitted: list[str] = []
    null.emit_events(
        result=SecondOpinionResult(),
        cfg=_cfg(),
        is_quick=False,
        pr_number="1",
        iter_log=emitted.append,
    )
    assert emitted == []


def test_from_config_returns_t2_impl():
    from cora.core.t2_second_opinion import T2SecondOpinion

    provider = SecondOpinionProvider.from_config(_cfg())
    assert isinstance(provider, T2SecondOpinion)


def test_t2_should_dispatch_skips_when_primary_is_blocker():
    """A primary body that already carries a hard blocker doesn't need the
    diversity vote — the gate stays closed even with the flag on."""
    from cora.core.t2_second_opinion import T2SecondOpinion

    provider = T2SecondOpinion()
    blocker_body = "🔴 needs changes\n\n🚨 **Blocker:** boom"
    assert provider.should_dispatch(
        cfg=_cfg(t2_disagreement=True), is_quick=False, primary_body=blocker_body
    ) is False
