"""Local grep_repo + git_show handlers served in-process from the
CI runner's PR checkout (`REPO_ROOT`).

The MCP server's grep_repo / git_show read a `main`-branch mirror —
they go PR-blind on files the PR adds. Serving these locally from
the PR merge ref means existence checks reflect the code actually
under review.

Both handlers take a single `args: dict` for parity with the
MCP-server call shape; `deep_review._make_pydantic_ai_local_tools`
wraps them as typed `pydantic_ai.Tool` instances (schema inferred
from the wrapper's type hints).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

from cora.core.config import REPO_ROOT


_GREP_SKIP_DIRS = frozenset(
    {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".agent-venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    }
)
_GREP_SKIP_EXT = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
        ".zip", ".gz", ".tar", ".woff", ".woff2", ".ttf", ".bin",
    }
)
_GREP_PATTERN_MAX = 256
_GREP_HARD_MAX_COUNT = 500
_GREP_CONTEXT_MAX = 5
_GREP_LINE_MAX = 512
_GREP_FILE_SIZE_MAX = 2 * 1024 * 1024  # 2 MiB — bounds worst-case scan cost


def _is_binary(path: Path) -> bool:
    """NUL-byte sniff on the first 1 KiB — same heuristic the MCP server uses."""
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True


def _normalize_glob(glob: str, root: Path) -> str:
    """Treat a directory glob as its whole subtree.

    The glob is fnmatch'd against the FULL repo-relative path, so a bare
    directory path ("pkg/sub/" or "pkg/sub") matches no file at all —
    the search silently scans zero files and the model reads the empty
    result as "this code doesn't exist" (observed against PR-added
    directories). `*` in fnmatch crosses `/`, so `dir/*` covers the
    subtree."""
    if glob.endswith("/"):
        return glob + "*"
    if not any(c in glob for c in "*?[") and (root / glob).is_dir():
        return glob + "/*"
    return glob


def local_grep_repo(args: dict, *, root: Path | None = None) -> str:
    """grep_repo over the checkout root (the PR merge tree). Python `re`,
    same output envelope as the MCP server's grep_repo so the agent's
    learned usage carries over — only the corpus differs (PR branch,
    not the `main` mirror). `root` defaults to the module-global
    `REPO_ROOT`; `GitProvider` passes an explicit root."""
    root = root if root is not None else REPO_ROOT
    pattern = (args.get("pattern") or "").strip()
    if not pattern:
        return "ERROR: grep_repo: pattern is required"
    if len(pattern) > _GREP_PATTERN_MAX:
        return (
            f"ERROR: grep_repo: pattern too long "
            f"({len(pattern)} > {_GREP_PATTERN_MAX})"
        )
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: grep_repo: invalid regex: {exc}"

    glob = args.get("glob") or None
    if glob is not None:
        glob = glob.strip() or None
        if glob and (os.path.isabs(glob) or ".." in glob.split("/")):
            return (
                f"ERROR: grep_repo: glob {glob!r} must be repo-relative "
                "and may not contain '..'"
            )
        if glob:
            glob = _normalize_glob(glob, root)
    max_count = max(1, min(int(args.get("max_count") or 50), _GREP_HARD_MAX_COUNT))
    context_lines = max(
        0, min(int(args.get("context_lines") or 0), _GREP_CONTEXT_MAX)
    )

    matches: list[dict] = []
    files_scanned = 0
    files_matched: set[str] = set()
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _GREP_SKIP_DIRS]
        if truncated:
            break
        for fn in sorted(filenames):
            if truncated:
                break
            if os.path.splitext(fn)[1].lower() in _GREP_SKIP_EXT:
                continue
            abs_path = Path(dirpath) / fn
            rel_path = os.path.relpath(abs_path, root)
            if glob and not fnmatch.fnmatch(rel_path, glob):
                continue
            try:
                if abs_path.stat().st_size > _GREP_FILE_SIZE_MAX:
                    continue
            except OSError:
                continue
            if _is_binary(abs_path):
                continue
            files_scanned += 1
            try:
                lines = abs_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue
                entry: dict = {
                    "path": rel_path,
                    "line": i + 1,
                    "content": (
                        line
                        if len(line) <= _GREP_LINE_MAX
                        else line[:_GREP_LINE_MAX] + " …(truncated)"
                    ),
                }
                if context_lines:
                    entry["before"] = lines[max(0, i - context_lines):i]
                    entry["after"] = lines[i + 1:i + 1 + context_lines]
                matches.append(entry)
                files_matched.add(rel_path)
                if len(matches) >= max_count:
                    truncated = True
                    break

    out = {
        "pattern": pattern,
        "glob": glob,
        "ref": "PR branch (merge ref) — the code under review",
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "files_scanned": files_scanned,
        "files_matched": len(files_matched),
    }
    if glob and files_scanned == 0:
        # Distinguish "no matches" from "glob selected no files" — the
        # former is evidence, the latter is a mis-aimed glob.
        out["note"] = (
            "glob selected zero files — it is fnmatch'd against the full "
            "repo-relative path; use 'dir/*' for a subtree or check the path"
        )
    return json.dumps(out, indent=2)


def local_git_show(args: dict, *, root: Path | None = None) -> str:
    """git_show against the checkout root. path-mode returns a file's
    content at the PR's state; commit-mode (no path) returns commit
    metadata. `repo` is accepted for MCP-server call-shape parity but
    ignored — the local checkout is always the one repo under review.
    `root` defaults to the module-global `REPO_ROOT`."""
    root = root if root is not None else REPO_ROOT
    ref = (args.get("ref") or "HEAD").strip() or "HEAD"
    path = args.get("path")
    if path is not None:
        path = path.strip()

    if path:
        proc = subprocess.run(
            ["git", "-C", str(root), "show", f"{ref}:{path}"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return (
                f"ERROR: git_show: could not read {path!r} at ref {ref!r}: "
                f"{proc.stderr.strip()}"
            )
        return json.dumps(
            {"ref": ref, "path": path, "content": proc.stdout}, indent=2
        )

    # Commit-mode — metadata only. The checkout is shallow, so refs
    # older than the PR head may not resolve; the error says so.
    proc = subprocess.run(
        [
            "git", "-C", str(root), "show", "--no-patch",
            "--format=%H%n%s%n%an <%ae>%n%aI%n%P%n%n%B", ref,
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return (
            f"ERROR: git_show: could not resolve ref {ref!r} "
            f"(shallow checkout?): {proc.stderr.strip()}"
        )
    parts = proc.stdout.split("\n", 5)
    if len(parts) < 6:
        return f"ERROR: git_show: unexpected git output for ref {ref!r}"
    sha, subject, author, date, parents, message = parts
    return json.dumps(
        {
            "sha": sha,
            "subject": subject,
            "author": author,
            "date": date,
            "parents": parents.split(),
            "message": message.lstrip("\n"),
        },
        indent=2,
    )
