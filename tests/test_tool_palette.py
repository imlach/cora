from __future__ import annotations


def test_loaded_tool_names_reflects_successful_server_topology():
    """The footer's unused denominator should describe loaded tools,
    not just whichever tools fired in this run."""
    from cora.core import config as c
    from cora.core.deep_review import _loaded_tool_names

    allowed = set(c.ALLOWED_TOOLS)

    base = _loaded_tool_names(allowed)
    assert set(base) == set(c.READ_TOOLS) | set(c.LOCAL_REPO_TOOLS)
    assert not (set(c.ACTION_TOOLS) & set(base))
    assert not (set(c.WEB_TOOLS) & set(base))

    full = _loaded_tool_names(allowed, actions_enabled=True, web_enabled=True)
    assert set(full) == set(c.ALLOWED_TOOLS)
