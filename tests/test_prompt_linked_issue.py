"""`assemble_initial_user_prompt`'s `linked_issue_context` param — the
pre-fetched, trust-wrapped linked-issue block (see
`cora.core.issue_context`) lands in the initial prompt the same way
`prefetched_release_notes` does.
"""

from __future__ import annotations

from cora.core.prompt import assemble_initial_user_prompt

_META = {
    "number": 1,
    "title": "t",
    "body": "A change.",
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


def test_absent_by_default():
    p = _prompt()
    assert "Linked issue" not in p


def test_injected_when_present():
    block = (
        '<untrusted-content from="https://github.com/o/r/issues" refs="#12">\n'
        "### #12 — Bug (open)\nacceptance criteria\n"
        "</untrusted-content>"
    )
    p = _prompt(linked_issue_context=block)
    assert "## Linked issue(s) (pre-fetched)" in p
    assert block in p
    assert "acceptance criteria" in p


def test_wrap_tag_preserved_verbatim():
    """The `<untrusted-content>` wrapper is the trust signal — it must
    round-trip intact, same load-bearing requirement as
    `prefetch.format_release_notes_block`'s `<external-content>`."""
    block = '<untrusted-content from="x" refs="#1">\nDATA\n</untrusted-content>'
    p = _prompt(linked_issue_context=block)
    assert "<untrusted-content" in p
    assert "</untrusted-content>" in p


def test_placed_after_pr_description():
    block = "ISSUE-BLOCK-MARKER"
    p = _prompt(linked_issue_context=block)
    assert p.index("## PR description") < p.index("## Linked issue")
    assert p.index("## Linked issue") < p.index(block)


def test_empty_string_is_treated_as_absent():
    p = _prompt(linked_issue_context="")
    assert "Linked issue" not in p
