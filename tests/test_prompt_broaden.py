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
    # encourages the underused knowledge tools by name — always
    # available (the read-tools MCP session, when configured).
    for tool in ("read_decision", "search_cluster_docs"):
        assert tool in p
    # still nudges parallel batching to bound wall-time
    assert "parallel tool calls" in p


def test_fetch_tool_mentioned_only_when_configured():
    """`web_fetch_doc` used to be named unconditionally, even for a
    deployment with no fetch session at all — claiming a tool that isn't
    there. `fetch_tool_configured` gates the (now tool-name-agnostic)
    mention on whether a fetch session is actually configured this run."""
    configured = _prompt(fetch_tool_configured=True)
    assert "fetch tool" in configured
    assert "dependency bump" in configured
    # Generic wording, not the literal tool name — a deployment could
    # rename or replace it; the static system prompt already avoids
    # naming optional tools, this framing stays consistent with that.
    assert "web_fetch_doc" not in configured

    unconfigured = _prompt()
    assert "fetch tool" not in unconfigured
    assert "web_fetch_doc" not in unconfigured


def test_grounding_carries_the_context_budget_counterweight():
    # validate-everything without a context budget saturated small
    # context windows (repeated / bulk tool results) — the framing
    # pairs the grounding norm with targeted-lookup discipline.
    p = _prompt()
    assert "context window is the budget" in p
    assert "never re-issue a call" in p
    assert "stop calling tools" in p


def test_broaden_flag_is_a_noop():
    assert _prompt(broaden_tools=True) == _prompt()


def test_no_tools_framing_untouched():
    # quick mode (tools_available=False) gets no grounding framing
    p = _prompt(tools_available=False, broaden_tools=True)
    assert "validate ANY claim" not in p
    assert "Produce the markdown review" in p
