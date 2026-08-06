"""Per-result char caps on the in-process repo tools + the duplicate-call
guard. Before these, a whole-file `git_show` on a large repo doc injected
the entire file into one agent turn (observed +45K tokens from a single
155 KB read), and a looping model re-paid for identical results every
turn."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from cora.core import config as _c
from cora.core.deep_review import _make_pydantic_ai_local_tools
from cora.core.repo_tools import (
    _cap_content,
    local_git_show,
    local_grep_deps,
    local_grep_repo,
)

# ── _cap_content ────────────────────────────────────────────────────


def test_cap_content_passthrough_under_cap():
    text, truncated = _cap_content("short", cap=100)
    assert text == "short"
    assert truncated is False


def test_cap_content_truncates_head_and_tail():
    src = "H" * 900 + "M" * 900 + "T" * 900
    text, truncated = _cap_content(src, cap=300)
    assert truncated is True
    assert text.startswith("H" * 200)
    assert text.endswith("T" * 100)
    assert "omitted" in text
    # Bounded: original chars kept is exactly the cap.
    assert len(text) <= 300 + 200  # cap + marker slack


# ── git_show content cap ────────────────────────────────────────────


def _git_repo_with_big_file(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    big = "x" * (3 * _c.TOOL_RESULT_CHAR_CAP)
    (tmp_path / "big.md").write_text(big, encoding="utf-8")
    (tmp_path / "small.md").write_text("tiny\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


def test_git_show_caps_large_file(tmp_path: Path):
    root = _git_repo_with_big_file(tmp_path)
    data = json.loads(local_git_show({"path": "big.md"}, root=root))
    assert data["truncated"] is True
    assert data["total_chars"] == 3 * _c.TOOL_RESULT_CHAR_CAP
    assert len(data["content"]) < 2 * _c.TOOL_RESULT_CHAR_CAP
    assert "omitted" in data["content"]


def test_git_show_small_file_untouched(tmp_path: Path):
    root = _git_repo_with_big_file(tmp_path)
    data = json.loads(local_git_show({"path": "small.md"}, root=root))
    assert data["content"] == "tiny\n"
    assert "truncated" not in data


# ── grep_repo char budget ───────────────────────────────────────────


def test_grep_repo_char_budget_bounds_result(tmp_path: Path):
    line = "NEEDLE " + "y" * 400
    (tmp_path / "many.txt").write_text((line + "\n") * 400, encoding="utf-8")
    out = local_grep_repo({"pattern": "NEEDLE", "max_count": 500}, root=tmp_path)
    assert len(out) < 2 * _c.TOOL_RESULT_CHAR_CAP
    data = json.loads(out)
    assert data["truncated"] is True
    assert "char budget" in data["note"]
    assert data["match_count"] >= 1


def test_grep_repo_small_result_has_no_budget_note(tmp_path: Path):
    (tmp_path / "a.txt").write_text("NEEDLE\n", encoding="utf-8")
    data = json.loads(local_grep_repo({"pattern": "NEEDLE"}, root=tmp_path))
    assert data["truncated"] is False
    assert "note" not in data


# ── duplicate-call guard ────────────────────────────────────────────


def test_duplicate_call_returns_stub(tmp_path: Path):
    (tmp_path / "a.txt").write_text("NEEDLE\n", encoding="utf-8")
    from cora.providers.git import LocalGitProvider

    tools = _make_pydantic_ai_local_tools(
        None, git_provider=LocalGitProvider(repo_root=tmp_path)
    )
    grep = next(t for t in tools if t.name == "grep_repo").function

    first = asyncio.run(grep(pattern="NEEDLE"))
    second = asyncio.run(grep(pattern="NEEDLE"))
    assert json.loads(first)["match_count"] == 1
    assert second.startswith("duplicate call")
    # Different args are not a duplicate.
    third = asyncio.run(grep(pattern="NEEDLE", max_count=10))
    assert json.loads(third)["match_count"] == 1


def test_duplicate_guard_is_per_factory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("NEEDLE\n", encoding="utf-8")
    from cora.providers.git import LocalGitProvider

    provider = LocalGitProvider(repo_root=tmp_path)
    grep_a = next(
        t for t in _make_pydantic_ai_local_tools(None, git_provider=provider)
        if t.name == "grep_repo"
    ).function
    grep_b = next(
        t for t in _make_pydantic_ai_local_tools(None, git_provider=provider)
        if t.name == "grep_repo"
    ).function
    asyncio.run(grep_a(pattern="NEEDLE"))
    # A fresh factory (fresh review) starts with a clean guard.
    fresh = asyncio.run(grep_b(pattern="NEEDLE"))
    assert json.loads(fresh)["match_count"] == 1


# ── grep_repo(corpus="deps") — dependency-source corpus (cora #23) ────


def test_grep_deps_no_roots_returns_plain_message():
    """No configured corpus is a plain one-line message, not `ERROR:` —
    the model should learn "absent" and not retry the call."""
    out = local_grep_deps({"pattern": "NEEDLE"}, roots=[])
    assert not out.startswith("ERROR")
    assert "no dependency-source corpus" in out
    # Deliberately not JSON — a distinct shape from every other result.
    assert not out.strip().startswith("{")


def test_grep_deps_finds_matches_repo_corpus_skips(tmp_path: Path):
    """The whole point: node_modules is exactly what the repo corpus
    excludes as build noise, and exactly what the deps corpus exists to
    search."""
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = NEEDLE_DEP\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("no match here\n", encoding="utf-8")

    repo_out = json.loads(local_grep_repo({"pattern": "NEEDLE_DEP"}, root=tmp_path))
    assert repo_out["match_count"] == 0

    deps_out = json.loads(local_grep_deps({"pattern": "NEEDLE_DEP"}, roots=[tmp_path]))
    assert deps_out["match_count"] == 1
    assert deps_out["corpus"] == "deps"
    assert deps_out["roots"] == [str(tmp_path)]
    assert deps_out["matches"][0]["path"] == f"{tmp_path.name}/node_modules/pkg/index.js"


def test_grep_deps_multi_root_labels_provenance(tmp_path: Path):
    root_a = tmp_path / "gomodcache"
    root_b = tmp_path / "vendor"
    (root_a / "pkg").mkdir(parents=True)
    (root_b / "pkg").mkdir(parents=True)
    (root_a / "pkg" / "a.go").write_text("func Foo() { NEEDLE }\n", encoding="utf-8")
    (root_b / "pkg" / "b.go").write_text("func Bar() { NEEDLE }\n", encoding="utf-8")

    out = json.loads(local_grep_deps({"pattern": "NEEDLE"}, roots=[root_a, root_b]))
    paths = {m["path"] for m in out["matches"]}
    assert paths == {"gomodcache/pkg/a.go", "vendor/pkg/b.go"}


def test_grep_deps_directory_glob_searches_subtree_per_root(tmp_path: Path):
    root = tmp_path / "sitepkgs"
    sub = root / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("NEEDLE\n", encoding="utf-8")
    (root / "top.py").write_text("NEEDLE\n", encoding="utf-8")

    out = json.loads(
        local_grep_deps({"pattern": "NEEDLE", "glob": "pkg/sub/"}, roots=[root])
    )
    assert out["match_count"] == 1
    assert out["matches"][0]["path"] == "sitepkgs/pkg/sub/mod.py"


def test_grep_deps_scan_cap_truncates_with_note(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"file{i}.txt").write_text(f"NEEDLE {i}\n", encoding="utf-8")

    out = json.loads(
        local_grep_deps({"pattern": "NEEDLE"}, roots=[tmp_path], max_files_scanned=3)
    )
    assert out["truncated"] is True
    assert "scan cap" in out["note"]
    assert out["files_scanned"] <= 3


def test_grep_deps_invalid_regex_errors(tmp_path: Path):
    out = local_grep_deps({"pattern": "("}, roots=[tmp_path])
    assert out.startswith("ERROR: grep_repo: invalid regex")


def test_grep_deps_no_pattern_errors(tmp_path: Path):
    out = local_grep_deps({"pattern": ""}, roots=[tmp_path])
    assert out == "ERROR: grep_repo: pattern is required"


def test_grep_repo_corpus_envelope_unaffected(tmp_path: Path):
    """`corpus="repo"` behaviour (including the envelope shape) is
    byte-identical to before the deps corpus existed — no `corpus` or
    `roots` key leaks in, and `local_grep_repo` doesn't even look at an
    args["corpus"] key."""
    (tmp_path / "a.txt").write_text("NEEDLE\n", encoding="utf-8")
    out = json.loads(local_grep_repo({"pattern": "NEEDLE", "corpus": "deps"}, root=tmp_path))
    assert "corpus" not in out
    assert "roots" not in out
    assert out["ref"] == "PR branch (merge ref) — the code under review"
    assert out["match_count"] == 1
