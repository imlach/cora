from __future__ import annotations


def test_loaded_tool_names_reflects_successful_server_topology():
    """The footer's unused denominator should describe loaded tools,
    not just whichever tools fired in this run."""
    from cora.core import config as c
    from cora.core.deep_review import _loaded_tool_names

    allowed = set(c.ALLOWED_TOOLS)

    base = _loaded_tool_names(allowed)
    # Local tools (repo-checkout + read_issue) are always registered,
    # independent of which optional MCP servers connected.
    assert set(base) == set(c.READ_TOOLS) | set(c.LOCAL_REPO_TOOLS) | set(c.LOCAL_ISSUE_TOOLS)
    assert not (set(c.ACTION_TOOLS) & set(base))
    assert not (set(c.WEB_TOOLS) & set(base))

    full = _loaded_tool_names(allowed, actions_enabled=True, web_enabled=True)
    assert set(full) == set(c.ALLOWED_TOOLS)


def test_loaded_tool_names_honours_local_issue_tools_override():
    """An adopter that drops `read_issue` from
    `ReviewerConfig.local_issue_tools` (an empty set) sees it disappear
    from the loaded palette, same override shape as `local_repo_tools`."""
    from cora.core import config as c
    from cora.core.deep_review import _loaded_tool_names

    allowed = set(c.ALLOWED_TOOLS)
    reduced = _loaded_tool_names(allowed, local_issue_tools=frozenset())
    assert "read_issue" not in reduced
    assert set(c.READ_TOOLS) <= set(reduced)
