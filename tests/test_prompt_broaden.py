"""The tools-available task framing (validate-any-claim grounding).

Grounding is the default now — the old "≤2 tool calls" nudge trained
production reviews down to 82% zero-tool-call verdicts. The former
REVIEWER_BROADEN_TOOLS teacher-trajectory variant IS the default
framing, and the flag is accepted as a no-op for env compat.
"""

from __future__ import annotations

from cora.core.prompt import assemble_initial_user_prompt

_META = {
    "number": 1,
    "title": "t",
    "author": {"login": "x"},
    "headRefName": "a",
    "baseRefName": "main",
    "additions": 1,
    "deletions": 0,
    "changedFiles": 1,
}


def _prompt(**kw):
    return assemble_initial_user_prompt(
        _META, "diff", False, "CONVENTIONS", False, False, **kw
    )


def test_default_grounds_every_claim():
    p = _prompt()
    assert "validate ANY claim" in p
    assert "Aim for ≤2 tool calls" not in p
    # encourages the underused knowledge / web tools by name
    for tool in ("read_decision", "search_cluster_docs", "web_fetch_doc"):
        assert tool in p
    # still nudges parallel batching to bound wall-time
    assert "parallel tool calls" in p


def test_broaden_flag_is_a_noop():
    assert _prompt(broaden_tools=True) == _prompt()


def test_no_tools_framing_untouched():
    # quick mode (tools_available=False) gets no grounding framing
    p = _prompt(tools_available=False, broaden_tools=True)
    assert "validate ANY claim" not in p
    assert "Produce the markdown review" in p
