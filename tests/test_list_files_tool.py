"""`list_files` — the path-existence tool (cora #36).

The bug it closes: `grep_repo` matches file CONTENT, so a zero-match
result never proved a path absent, but the model had no path-level tool
to reach for and read the empty grep as "the file is missing" — a false
🚨 Blocker. These tests pin the distinction (grep is content, list_files
is paths), the bounded output, and the fact that a provider without the
capability reports the gap rather than implying absence.
"""

from __future__ import annotations

import json
from pathlib import Path

from cora.core import config as _c
from cora.core.repo_tools import _LIST_FILES_MAX, local_list_files
from cora.providers import GitProvider, LocalGitProvider


def _fixture_tree(tmp_path: Path) -> Path:
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "foo.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (tmp_path / "tests" / "test_uses_it.py").write_text(
        "PATH = 'tests/fixtures/foo.jsonl'\n", encoding="utf-8"
    )
    (tmp_path / "readme.md").write_text("docs\n", encoding="utf-8")
    return tmp_path


def test_finds_an_exact_path(tmp_path: Path):
    root = _fixture_tree(tmp_path)
    data = json.loads(
        LocalGitProvider(repo_root=root).list_files(
            {"glob": "tests/fixtures/foo.jsonl"}
        )
    )
    assert data["paths"] == ["tests/fixtures/foo.jsonl"]
    assert data["count"] == 1


def test_basename_glob_finds_a_file_of_unknown_location(tmp_path: Path):
    root = _fixture_tree(tmp_path)
    data = json.loads(
        LocalGitProvider(repo_root=root).list_files({"glob": "*/foo.jsonl"})
    )
    assert data["paths"] == ["tests/fixtures/foo.jsonl"]


def test_directory_glob_lists_the_subtree(tmp_path: Path):
    """Same `_normalize_glob` semantics as grep_repo — a bare directory
    means its subtree, not a full-path fnmatch that selects nothing."""
    root = _fixture_tree(tmp_path)
    for glob in ("tests/fixtures", "tests/fixtures/"):
        data = json.loads(
            LocalGitProvider(repo_root=root).list_files({"glob": glob})
        )
        assert data["paths"] == ["tests/fixtures/foo.jsonl"], glob


def test_the_36_scenario_grep_misses_but_list_files_finds(tmp_path: Path):
    """The regression this tool exists for: a content grep for the
    fixture's own path finds only the references, and a grep scoped to
    the fixture matches nothing in it — neither says the file exists.
    `list_files` does."""
    root = _fixture_tree(tmp_path)
    provider = LocalGitProvider(repo_root=root)

    grep = json.loads(
        provider.grep_repo(
            {"pattern": "nonexistent-symbol", "glob": "tests/fixtures/foo.jsonl"}
        )
    )
    assert grep["match_count"] == 0  # would have been read as "file missing"

    listing = json.loads(provider.list_files({"glob": "tests/fixtures/foo.jsonl"}))
    assert listing["count"] == 1


def test_no_match_note_says_what_it_does_not_prove(tmp_path: Path):
    root = _fixture_tree(tmp_path)
    data = json.loads(
        LocalGitProvider(repo_root=root).list_files({"glob": "tests/nope.jsonl"})
    )
    assert data["paths"] == []
    assert "before concluding the file is absent" in data["note"]


def test_skips_the_same_dirs_and_extensions_as_grep(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "keep.py").write_text("x\n", encoding="utf-8")
    data = json.loads(LocalGitProvider(repo_root=tmp_path).list_files({}))
    assert data["paths"] == ["keep.py"]


def test_unglobbed_listing_is_capped_and_says_so(tmp_path: Path):
    for i in range(_LIST_FILES_MAX + 25):
        (tmp_path / f"f{i:05d}.py").write_text("x\n", encoding="utf-8")
    data = json.loads(local_list_files({}, root=tmp_path))
    assert data["truncated"] is True
    assert data["count"] == _LIST_FILES_MAX
    assert "NOT the complete set" in data["note"]


def test_rejects_escaping_globs(tmp_path: Path):
    for bad in ("/etc/passwd", "../outside/*"):
        out = LocalGitProvider(repo_root=tmp_path).list_files({"glob": bad})
        assert out.startswith("ERROR: list_files:"), bad


def test_paths_are_sorted(tmp_path: Path):
    (tmp_path / "b.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    sub = tmp_path / "z"
    sub.mkdir()
    (sub / "c.py").write_text("x\n", encoding="utf-8")
    data = json.loads(local_list_files({}, root=tmp_path))
    assert data["paths"] == sorted(data["paths"])


def test_base_provider_reports_the_gap_instead_of_implying_absence():
    """A third-party GitProvider predating #36 must keep importing and
    subclassing — `list_files` is concrete, not abstract — and its
    fallback must not read as evidence a file is missing."""

    class OldProvider(GitProvider):
        def grep_repo(self, args: dict) -> str:
            return "{}"

        def git_show(self, args: dict) -> str:
            return "{}"

    out = OldProvider().list_files({"glob": "anything"})
    assert "not supported" in out
    assert "Do not treat that as evidence a file is missing" in out


def test_list_files_is_in_the_default_allowed_tool_set():
    assert "list_files" in _c.LOCAL_REPO_TOOLS
    assert "list_files" in _c.ALLOWED_TOOLS


def test_local_repo_tools_override_disables_registration():
    """`LOCAL_REPO_TOOLS` is the opt-out for a deployment that wants the
    pre-#36 palette — dropping the name must unregister the tool without
    disturbing grep_repo/git_show."""
    from cora.core.deep_review import _make_pydantic_ai_local_tools

    class _FakeCfg:
        local_issue_tools = frozenset()
        local_repo_tools = frozenset({"grep_repo", "git_show"})

    tools = _make_pydantic_ai_local_tools(None, repo="o/r", cfg=_FakeCfg())
    assert {t.name for t in tools} == {"grep_repo", "git_show"}
