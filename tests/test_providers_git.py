"""GitProvider seam: LocalGitProvider runs grep/git-show against a
configurable checkout root (not just the module-global REPO_ROOT)."""

from __future__ import annotations

import json
from pathlib import Path

from cora.config import ReviewerConfig
from cora.core import config as _c
from cora.providers import GitProvider, LocalGitProvider


def test_from_config_returns_local_git_provider():
    assert isinstance(GitProvider.from_config(ReviewerConfig()), LocalGitProvider)


def test_default_repo_root_is_engine_repo_root():
    assert LocalGitProvider().repo_root == _c.REPO_ROOT


def test_grep_repo_runs_against_the_provider_root(tmp_path: Path):
    (tmp_path / "a.py").write_text("alpha = 1\nNEEDLE here\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("nothing to see\n", encoding="utf-8")
    out = LocalGitProvider(repo_root=tmp_path).grep_repo({"pattern": "NEEDLE"})
    data = json.loads(out)
    assert data["match_count"] == 1
    assert data["matches"][0]["path"] == "a.py"
    assert data["matches"][0]["line"] == 2


def _tree_with_subdir(tmp_path: Path) -> Path:
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("NEEDLE in subtree\n", encoding="utf-8")
    (tmp_path / "top.py").write_text("NEEDLE at top\n", encoding="utf-8")
    return tmp_path


def test_grep_repo_directory_glob_searches_the_subtree(tmp_path: Path):
    """A bare directory path — with or without trailing "/" — must search
    the directory's subtree, not full-match against file paths (which
    silently scanned zero files and read as "code doesn't exist")."""
    root = _tree_with_subdir(tmp_path)
    for glob in ("pkg/sub/", "pkg/sub", "pkg/"):
        data = json.loads(
            LocalGitProvider(repo_root=root).grep_repo(
                {"pattern": "NEEDLE", "glob": glob}
            )
        )
        assert data["match_count"] == 1, glob
        assert data["matches"][0]["path"] == "pkg/sub/mod.py"
        assert "note" not in data


def test_grep_repo_wildcard_free_file_glob_still_exact(tmp_path: Path):
    root = _tree_with_subdir(tmp_path)
    data = json.loads(
        LocalGitProvider(repo_root=root).grep_repo(
            {"pattern": "NEEDLE", "glob": "top.py"}
        )
    )
    assert data["match_count"] == 1
    assert data["matches"][0]["path"] == "top.py"


def test_grep_repo_zero_file_glob_carries_a_note(tmp_path: Path):
    """A glob that selects no files is a mis-aimed glob, not evidence of
    absence — the envelope says so, so the model can re-aim."""
    root = _tree_with_subdir(tmp_path)
    data = json.loads(
        LocalGitProvider(repo_root=root).grep_repo(
            {"pattern": "NEEDLE", "glob": "no/such/dir/*"}
        )
    )
    assert data["match_count"] == 0
    assert data["files_scanned"] == 0
    assert "zero files" in data["note"]


def test_git_show_delegates_with_provider_root(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake_git_show(args, *, root):
        captured["args"] = args
        captured["root"] = root
        return "{}"

    monkeypatch.setattr("cora.providers.git.local_git_show", fake_git_show)
    LocalGitProvider(repo_root=tmp_path).git_show({"ref": "HEAD", "path": "x"})
    assert captured["root"] == tmp_path
    assert captured["args"]["path"] == "x"


def test_deep_review_local_tools_route_through_injected_provider():
    """The engine's grep_repo / git_show agent tools go through the
    GitProvider seam — an injected provider is what they call."""
    import asyncio

    from cora.core.deep_review import _make_pydantic_ai_local_tools

    class _SpyGit(GitProvider):
        def __init__(self) -> None:
            self.calls: list = []

        def grep_repo(self, args: dict) -> str:
            self.calls.append(("grep", args))
            return "GREP_OUT"

        def git_show(self, args: dict) -> str:
            self.calls.append(("show", args))
            return "SHOW_OUT"

    spy = _SpyGit()
    tools = _make_pydantic_ai_local_tools(None, git_provider=spy)
    by = {t.name: t for t in tools}
    assert set(by) == {"grep_repo", "git_show"}
    # Tool.function is the wrapped async callable; invoke it directly.
    assert asyncio.run(by["grep_repo"].function(pattern="x")) == "GREP_OUT"
    assert asyncio.run(by["git_show"].function(ref="HEAD", path="f.py")) == "SHOW_OUT"
    assert spy.calls[0][0] == "grep" and spy.calls[0][1]["pattern"] == "x"
    assert ("show", {"ref": "HEAD", "path": "f.py"}) in spy.calls
