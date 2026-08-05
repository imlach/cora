"""_decisions_body — local ADR resolution across both repo layouts.

Adopter repos keep decisions either as a monolithic DECISIONS.md or as
one file per decision under decisions/ (DEC-NNN-<slug>.md). The
directory wins when both exist; the monolith is the fallback.
"""

from __future__ import annotations

from pathlib import Path

from cora.core.retrieval import _decisions_body

_MONOLITH = """\
# Decisions

## DEC-010: First choice (2026-01-01)

Body of ten.

## DEC-002: Older choice (2025-12-01)

Body of two.
"""


def _write_dir_layout(root: Path) -> None:
    dec = root / "decisions"
    dec.mkdir()
    (dec / "DEC-010-first-choice.md").write_text(
        "## DEC-010: First choice (2026-01-01)\n\nBody of ten (per-file).\n"
    )
    (dec / "DEC-002.md").write_text(
        "## DEC-002: Older choice (2025-12-01)\n\nBody of two (per-file).\n"
    )


def test_directory_layout_slugged_file(tmp_path: Path) -> None:
    _write_dir_layout(tmp_path)
    body = _decisions_body(tmp_path, "DEC-010")
    assert body is not None
    assert "Body of ten (per-file)." in body
    assert body.startswith("## DEC-010")


def test_directory_layout_bare_file(tmp_path: Path) -> None:
    _write_dir_layout(tmp_path)
    body = _decisions_body(tmp_path, "DEC-002")
    assert body is not None
    assert "Body of two (per-file)." in body


def test_directory_wins_over_monolith(tmp_path: Path) -> None:
    _write_dir_layout(tmp_path)
    (tmp_path / "DECISIONS.md").write_text(_MONOLITH)
    body = _decisions_body(tmp_path, "DEC-010")
    assert body is not None
    assert "(per-file)" in body


def test_id_prefix_does_not_cross_match(tmp_path: Path) -> None:
    # DEC-01 must not resolve to DEC-010's file.
    _write_dir_layout(tmp_path)
    assert _decisions_body(tmp_path, "DEC-01") is None


def test_monolith_fallback(tmp_path: Path) -> None:
    (tmp_path / "DECISIONS.md").write_text(_MONOLITH)
    body = _decisions_body(tmp_path, "DEC-002")
    assert body is not None
    assert "Body of two." in body


def test_miss_returns_none(tmp_path: Path) -> None:
    _write_dir_layout(tmp_path)
    (tmp_path / "DECISIONS.md").write_text(_MONOLITH)
    assert _decisions_body(tmp_path, "DEC-999") is None
