"""`read_issue` as a registered Pydantic-AI local tool
(`deep_review._make_pydantic_ai_local_tools`) — registration gating,
the duplicate-call guard shared with grep_repo/git_show, and that the
tool call reaches `issue_context.local_read_issue` with this review's
repo.
"""

from __future__ import annotations

import asyncio

from cora.core import config as _c
from cora.core.deep_review import _make_pydantic_ai_local_tools


def _get(tools, name):
    return next((t for t in tools if t.name == name), None)


def test_registered_when_repo_given():
    tools = _make_pydantic_ai_local_tools(None, repo="o/r")
    assert {t.name for t in tools} == {
        "grep_repo", "git_show", "list_files", "read_issue",
    }


def test_not_registered_without_a_repo():
    """Local tools always need a repo scope; no `repo` (shouldn't
    happen for a real deep-mode run, but defensively) means no
    `read_issue`, same as grep_repo/git_show still work off the
    checkout alone."""
    tools = _make_pydantic_ai_local_tools(None, repo=None)
    assert {t.name for t in tools} == {"grep_repo", "git_show", "list_files"}


def test_local_issue_tools_override_disables_it():
    class _FakeCfg:
        local_issue_tools = frozenset()
        local_repo_tools = frozenset(_c.LOCAL_REPO_TOOLS)

    tools = _make_pydantic_ai_local_tools(None, repo="o/r", cfg=_FakeCfg())
    assert {t.name for t in tools} == {"grep_repo", "git_show", "list_files"}


def test_calls_local_read_issue_with_review_repo(monkeypatch):
    import cora.core.issue_context as ic_mod

    seen = {}

    def fake_local_read_issue(args, *, repo, cfg=None):
        seen["args"] = args
        seen["repo"] = repo
        return "WRAPPED-RESULT"

    monkeypatch.setattr(ic_mod, "local_read_issue", fake_local_read_issue)

    tools = _make_pydantic_ai_local_tools(None, repo="o/r")
    read_issue = _get(tools, "read_issue").function

    out = asyncio.run(read_issue(number=7))
    assert out == "WRAPPED-RESULT"
    assert seen["repo"] == "o/r"
    assert seen["args"] == {"number": 7, "repo": None}


def test_duplicate_call_returns_stub(monkeypatch):
    import cora.core.issue_context as ic_mod

    monkeypatch.setattr(ic_mod, "local_read_issue", lambda *a, **k: "RESULT")

    tools = _make_pydantic_ai_local_tools(None, repo="o/r")
    read_issue = _get(tools, "read_issue").function

    first = asyncio.run(read_issue(number=7))
    second = asyncio.run(read_issue(number=7))
    assert first == "RESULT"
    assert second.startswith("duplicate call")

    # Different args are not a duplicate.
    third = asyncio.run(read_issue(number=9))
    assert third == "RESULT"


def test_duplicate_guard_is_shared_across_local_tools(monkeypatch):
    """The dedupe set is per-factory-call (one per review), shared
    across grep_repo/git_show/read_issue — a distinct-args call to a
    different tool is never mistaken for a repeat."""
    import cora.core.issue_context as ic_mod

    monkeypatch.setattr(ic_mod, "local_read_issue", lambda *a, **k: "R")

    class _NullProvider:
        def grep_repo(self, args):
            return "{}"

        def git_show(self, args):
            return "{}"

    tools = _make_pydantic_ai_local_tools(
        None, repo="o/r", git_provider=_NullProvider()
    )
    read_issue = _get(tools, "read_issue").function
    grep_repo = _get(tools, "grep_repo").function

    r1 = asyncio.run(read_issue(number=1))
    r2 = asyncio.run(grep_repo(pattern="x"))
    assert not r1.startswith("duplicate call")
    assert not r2.startswith("duplicate call")
