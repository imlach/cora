"""The escalation must say so when T1 landed on the same engine as T0.

Every escalation entry assumes a second endpoint fails differently. A
gateway alias can hide that two aliases share one backend, which turns
the escalation into a silent re-draw on the same pods.
"""

from __future__ import annotations

from cora.core.litellm_capture import served_model_from_messages
from cora.review._tiers import _warn_no_op_escalation


class _Resp:
    def __init__(self, model_name):
        self.model_name = model_name


class _Req:
    """A `ModelRequest` has no `model_name` — skipped, never raises."""


def test_identical_alias_warns(capsys):
    _warn_no_op_escalation(
        t0_alias="review", t1_alias="review", t0_served=None, t1_served=None
    )
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "same as T0" in out


def test_distinct_aliases_same_served_model_warns(capsys):
    _warn_no_op_escalation(
        t0_alias="review", t1_alias="cora", t0_served="piano", t1_served="piano"
    )
    out = capsys.readouterr().out
    assert "no-op" in out
    assert "`piano`" in out


def test_distinct_backends_stay_quiet(capsys):
    _warn_no_op_escalation(
        t0_alias="review", t1_alias="forte", t0_served="piano", t1_served="forte"
    )
    assert capsys.readouterr().out == ""


def test_unresolved_served_names_stay_quiet(capsys):
    """Attribution is best-effort — an unknown name must never read as a
    match, or every unattributed run gets a spurious warning."""
    _warn_no_op_escalation(
        t0_alias="review", t1_alias="forte", t0_served=None, t1_served=None
    )
    assert capsys.readouterr().out == ""


def test_served_model_from_messages_is_last_write_wins():
    assert (
        served_model_from_messages([_Resp("piano"), _Req(), _Resp("forte")]) == "forte"
    )
    assert served_model_from_messages([_Req()]) is None
    assert served_model_from_messages([]) is None
    assert served_model_from_messages(None) is None


def test_reason_with_no_detail_keeps_the_plain_verdict_line():
    """`partition` returns an empty tail when there is no colon, so a
    bare marker must not render as `errored: `."""
    from cora.review._tiers import _agent_loop_verdict_line

    assert (
        _agent_loop_verdict_line("agent-loop-errored")
        == "skipped (agent loop errored)"
    )
    assert (
        _agent_loop_verdict_line("agent-loop-errored: spiral-redraw-exhausted")
        == "skipped (agent loop errored: spiral-redraw-exhausted)"
    )
    assert (
        _agent_loop_verdict_line("agent-loop-errored:   ")
        == "skipped (agent loop errored)"
    )
