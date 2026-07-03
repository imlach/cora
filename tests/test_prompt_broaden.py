"""The broaden_tools prompt variant (teacher-trajectory mode).

Asserts the opt-in variant swaps the production "≤2 tool calls" nudge
for full-palette encouragement, and that the default is unchanged.
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


def test_default_keeps_two_call_cap():
    p = _prompt()
    assert "Aim for ≤2 tool calls" in p
    assert "GROUND every finding with a tool" not in p


def test_broaden_swaps_to_full_palette():
    p = _prompt(broaden_tools=True)
    assert "Aim for ≤2 tool calls" not in p
    assert "GROUND every finding with a tool" in p
    # encourages the underused knowledge / web tools by name
    for tool in ("read_decision", "search_cluster_docs", "web_fetch_doc"):
        assert tool in p
    # still nudges parallel batching to bound wall-time
    assert "parallel tool calls" in p


def test_broaden_ignored_without_tools():
    # quick mode (tools_available=False) never gets the broaden framing
    p = _prompt(tools_available=False, broaden_tools=True)
    assert "GROUND every finding with a tool" not in p
    assert "Produce the markdown review" in p
