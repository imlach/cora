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
from cora.core.repo_tools import _cap_content, local_git_show, local_grep_repo

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
