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


# ── grep_repo(corpus="deps") dispatch on LocalGitProvider ─────────────


def test_local_git_provider_grep_repo_defaults_to_repo_corpus(tmp_path: Path):
    (tmp_path / "a.py").write_text("NEEDLE\n", encoding="utf-8")
    out = json.loads(LocalGitProvider(repo_root=tmp_path).grep_repo({"pattern": "NEEDLE"}))
    assert out["match_count"] == 1
    assert "corpus" not in out  # unchanged repo-corpus envelope


def test_local_git_provider_grep_repo_corpus_deps_dispatches(tmp_path: Path):
    dep_root = tmp_path / "vendor"
    dep_root.mkdir()
    (dep_root / "lib.go").write_text("func Public() { NEEDLE }\n", encoding="utf-8")

    provider = LocalGitProvider(repo_root=tmp_path, dep_source_roots=[dep_root])
    out = json.loads(provider.grep_repo({"pattern": "NEEDLE", "corpus": "deps"}))
    assert out["corpus"] == "deps"
    assert out["match_count"] == 1
    assert out["matches"][0]["path"] == "vendor/lib.go"


def test_local_git_provider_grep_repo_deps_corpus_absent_by_default(tmp_path: Path):
    """A `LocalGitProvider` built without `dep_source_roots` (the common
    case in direct construction / tests) has no deps corpus — it does
    NOT silently fall back to scanning `repo_root`."""
    out = LocalGitProvider(repo_root=tmp_path).grep_repo(
        {"pattern": "NEEDLE", "corpus": "deps"}
    )
    assert "no dependency-source corpus" in out


def test_local_git_provider_grep_repo_unknown_corpus_errors(tmp_path: Path):
    out = LocalGitProvider(repo_root=tmp_path).grep_repo(
        {"pattern": "NEEDLE", "corpus": "bogus"}
    )
    assert out == "ERROR: grep_repo: corpus must be 'repo' or 'deps' (got 'bogus')"


# ── _resolve_dep_source_roots — config → provider seam ─────────────────


def test_resolve_dep_source_roots_keeps_existing_explicit_paths(tmp_path: Path):
    from cora.providers.git import _resolve_dep_source_roots

    root_a = tmp_path / "a"
    root_a.mkdir()
    cfg = ReviewerConfig(dep_source_roots=(str(root_a),))
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == [root_a]


def test_resolve_dep_source_roots_drops_missing_with_warning(tmp_path: Path, capsys):
    from cora.providers.git import _resolve_dep_source_roots

    missing = tmp_path / "does-not-exist"
    cfg = ReviewerConfig(dep_source_roots=(str(missing),))
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == []
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert str(missing) in out


def test_resolve_dep_source_roots_partial_drop_keeps_the_rest(tmp_path: Path):
    from cora.providers.git import _resolve_dep_source_roots

    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing"
    cfg = ReviewerConfig(dep_source_roots=(str(present), str(missing)))
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == [present]


def test_resolve_dep_source_roots_auto_detects_in_repo_vendor_and_node_modules(
    tmp_path: Path,
):
    from cora.providers.git import _resolve_dep_source_roots

    (tmp_path / "vendor").mkdir()
    (tmp_path / "node_modules").mkdir()
    cfg = ReviewerConfig()  # dep_source_roots unset
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert set(resolved) == {tmp_path / "vendor", tmp_path / "node_modules"}


def test_resolve_dep_source_roots_auto_detect_only_what_exists(tmp_path: Path):
    from cora.providers.git import _resolve_dep_source_roots

    (tmp_path / "vendor").mkdir()
    cfg = ReviewerConfig()
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == [tmp_path / "vendor"]


def test_resolve_dep_source_roots_no_autodetect_and_no_config_is_empty(tmp_path: Path):
    from cora.providers.git import _resolve_dep_source_roots

    cfg = ReviewerConfig()
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == []


def test_resolve_dep_source_roots_explicit_wins_over_autodetect(tmp_path: Path):
    """An explicit DEP_SOURCE_ROOTS list is never second-guessed by
    auto-detection — even when the repo also happens to vendor deps."""
    from cora.providers.git import _resolve_dep_source_roots

    (tmp_path / "vendor").mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    cfg = ReviewerConfig(dep_source_roots=(str(other),))
    resolved = _resolve_dep_source_roots(cfg, repo_root=tmp_path)
    assert resolved == [other]


def test_from_config_wires_resolved_dep_source_roots_into_provider(tmp_path: Path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.go").write_text("NEEDLE\n", encoding="utf-8")
    cfg = ReviewerConfig()

    import cora.core.config as _c

    old_root = _c.REPO_ROOT
    try:
        _c.REPO_ROOT = tmp_path
        provider = GitProvider.from_config(cfg)
        assert isinstance(provider, LocalGitProvider)
        assert provider.dep_source_roots == [tmp_path / "vendor"]
        out = json.loads(provider.grep_repo({"pattern": "NEEDLE", "corpus": "deps"}))
        assert out["match_count"] == 1
    finally:
        _c.REPO_ROOT = old_root


def test_from_config_wires_dep_source_max_files_scanned(tmp_path: Path):
    cfg = ReviewerConfig(dep_source_max_files_scanned=7)
    provider = GitProvider.from_config(cfg)
    assert provider.dep_source_max_files_scanned == 7
