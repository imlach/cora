"""Local grep_repo + list_files + git_show handlers served in-process
from the CI runner's PR checkout (`REPO_ROOT`).

The MCP server's grep_repo / git_show read a `main`-branch mirror —
they go PR-blind on files the PR adds. Serving these locally from
the PR merge ref means existence checks reflect the code actually
under review.

`grep_repo` searches two corpora: `local_grep_repo` (the PR checkout,
`corpus="repo"`, the default) and `local_grep_deps` (the deployment's
resolved dependency source — a Go module cache, vendor dir,
node_modules, site-packages, ..., `corpus="deps"`) — added so a
library-API claim can be checked against the pinned dependency's
actual source instead of asserted from memory (cora issue #23). They
share validation (`_parse_grep_query`) and per-entry match building,
but walk separately: the repo corpus has exactly one root and skips
`node_modules`/`.venv`/`venv` as build noise, while the deps corpus has
one-or-many roots where those same directories ARE the corpus.

`list_files` (`local_list_files`) answers the question grep cannot:
grep matches file CONTENT, so an empty result is not evidence a path is
absent, and reading it that way produced false "file missing" blockers
(cora #36). Path-level existence needs a path-level tool.

All handlers take a single `args: dict` for parity with the MCP-server
call shape; `deep_review._make_pydantic_ai_local_tools` wraps them as
typed `pydantic_ai.Tool` instances (schema inferred from the wrapper's
type hints).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

from cora.core.config import (
    DEP_SOURCE_MAX_FILES_SCANNED,
    REPO_ROOT,
    TOOL_RESULT_CHAR_CAP,
)

_GREP_SKIP_DIRS = frozenset(
    {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".agent-venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    }
)
# Deps-corpus skip dirs — deliberately NOT the same set as
# `_GREP_SKIP_DIRS`: `node_modules` / `.venv` / `venv` are exactly what
# a deps-corpus root points at, so excluding them would grep nothing.
# Only true noise (VCS metadata, tool caches) is skipped.
_DEP_GREP_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
# Traversal ceiling as a multiple of the candidate-file cap. Bounds a glob
# that matches almost nothing against a huge corpus; see the two-bounds
# comment in `local_grep_deps`.
_DEP_TRAVERSAL_CAP_MULTIPLE = 20
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

# `list_files` result ceiling. Paths are short, so this is generous
# relative to `_GREP_HARD_MAX_COUNT` — the point of the tool is to
# answer "does this path exist" and "what's in this directory", and a
# truncated listing can't answer the second one honestly.
_LIST_FILES_MAX = 1000


def _cap_content(text: str, cap: int = 0) -> tuple[str, bool]:
    """Head+tail truncation to `cap` chars (default `TOOL_RESULT_CHAR_CAP`).

    A whole-file read must never inject an unbounded result into the
    agent's context — one uncapped read of a large doc can saturate a
    small T0 context window in a single turn. Head 2/3 + tail 1/3 keeps
    both the opening (imports, headers) and the end (recent appends)
    visible; the marker tells the model how much is missing and how to
    narrow the request."""
    cap = cap or TOOL_RESULT_CHAR_CAP
    if len(text) <= cap:
        return text, False
    head = (cap * 2) // 3
    tail = cap - head
    marker = (
        f"\n…[{len(text) - cap:,} of {len(text):,} chars omitted — "
        "narrow the request (a specific section, grep_repo with a "
        "pattern) instead of re-reading the whole file]…\n"
    )
    return text[:head] + marker + text[-tail:], True


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


def _glob_static_segments(glob: str) -> list[str]:
    """Leading wildcard-free path segments of a glob.

    `"github.com/pkg/errors/*"` → `["github.com", "pkg", "errors"]`;
    `"*.go"` → `[]`. Lets the deps walk skip subtrees that cannot contain
    a match instead of paying for them out of the scan budget — on a
    module cache the difference is the whole feature working or not.
    Stops at the first segment carrying a wildcard, so pruning never
    excludes a directory the glob could still match."""
    segments: list[str] = []
    for seg in glob.split("/"):
        if not seg or any(c in seg for c in "*?["):
            break
        segments.append(seg)
    return segments


def _parse_grep_query(args: dict) -> dict | str:
    """Validate + normalize the args `local_grep_repo` and
    `local_grep_deps` share (pattern, glob shape, max_count,
    context_lines). Returns a dict on success or an `ERROR: ...` string
    to return verbatim from the caller. Glob directory-subtree expansion
    (`_normalize_glob`) is NOT done here — it needs a specific root, and
    the deps corpus may have several, so each caller normalizes
    per-root."""
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
    max_count = max(1, min(int(args.get("max_count") or 50), _GREP_HARD_MAX_COUNT))
    context_lines = max(
        0, min(int(args.get("context_lines") or 0), _GREP_CONTEXT_MAX)
    )
    return {
        "pattern": pattern,
        "rx": rx,
        "glob": glob,
        "max_count": max_count,
        "context_lines": context_lines,
    }


def local_grep_repo(args: dict, *, root: Path | None = None) -> str:
    """grep_repo over the checkout root (the PR merge tree, `corpus=
    "repo"`). Python `re`, same output envelope as the MCP server's
    grep_repo so the agent's learned usage carries over — only the
    corpus differs (PR branch, not the `main` mirror). `root` defaults
    to the module-global `REPO_ROOT`; `GitProvider` passes an explicit
    root. See `local_grep_deps` for the dependency-source corpus."""
    root = root if root is not None else REPO_ROOT
    parsed = _parse_grep_query(args)
    if isinstance(parsed, str):
        return parsed
    pattern, rx, glob = parsed["pattern"], parsed["rx"], parsed["glob"]
    max_count, context_lines = parsed["max_count"], parsed["context_lines"]
    if glob:
        glob = _normalize_glob(glob, root)

    matches: list[dict] = []
    files_scanned = 0
    files_matched: set[str] = set()
    truncated = False
    budget_hit = False
    # Char budget across the whole result — max_count alone doesn't
    # bound size (500 matches × 512-char lines × context is ~1.5 MB).
    chars_used = 0
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
                chars_used += len(entry["content"]) + len(rel_path) + 40
                if context_lines:
                    chars_used += sum(
                        len(s) for s in entry["before"] + entry["after"]
                    )
                if matches and chars_used > TOOL_RESULT_CHAR_CAP:
                    # Keep at least one match; report the budget stop
                    # distinctly from the max_count stop below.
                    truncated = True
                    budget_hit = True
                    break
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
    if budget_hit:
        out["note"] = (
            f"result char budget ({TOOL_RESULT_CHAR_CAP}) reached at "
            f"{len(matches)} matches — narrow with a tighter pattern, a "
            "glob, or context_lines=0"
        )
    elif glob and files_scanned == 0:
        # Distinguish "no matches" from "glob selected no files" — the
        # former is evidence, the latter is a mis-aimed glob.
        out["note"] = (
            "glob selected zero files — it is fnmatch'd against the full "
            "repo-relative path; use 'dir/*' for a subtree or check the path"
        )
    return json.dumps(out, indent=2)


def local_list_files(args: dict, *, root: Path | None = None) -> str:
    """List repo-relative PATHS at this PR's state (`corpus="repo"` only).

    The existence tool. `grep_repo` matches file CONTENT — its `glob`
    only filters which files get searched — so an empty grep result says
    nothing about whether a path exists, and reading it as proof of
    absence produced false "file missing" blockers (cora #36). There was
    no correct tool to reach for; this is it.

    Same skip-dirs / skip-extensions / binary rules as `local_grep_repo`
    so the two agree on what "in the repo" means. Unglobbed listings are
    capped at `_LIST_FILES_MAX` with `truncated` set — never silently."""
    root = root if root is not None else REPO_ROOT
    glob = (args.get("glob") or "").strip() or None
    if glob and (os.path.isabs(glob) or ".." in glob.split("/")):
        return (
            f"ERROR: list_files: glob {glob!r} must be repo-relative "
            "and may not contain '..'"
        )
    if glob:
        glob = _normalize_glob(glob, root)

    paths: list[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _GREP_SKIP_DIRS]
        if truncated:
            break
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in _GREP_SKIP_EXT:
                continue
            rel_path = os.path.relpath(Path(dirpath) / fn, root)
            if glob and not fnmatch.fnmatch(rel_path, glob):
                continue
            paths.append(rel_path)
            if len(paths) >= _LIST_FILES_MAX:
                truncated = True
                break

    paths.sort()
    out = {
        "glob": glob,
        "ref": "PR branch (merge ref) — the code under review",
        "paths": paths,
        "count": len(paths),
        "truncated": truncated,
    }
    if truncated:
        out["note"] = (
            f"listing capped at {_LIST_FILES_MAX} paths — narrow with a "
            "glob; this is NOT the complete set of files"
        )
    elif glob and not paths:
        # The one result that gets misread. Say plainly what it does and
        # does not prove, because this is the tool a model reaches for
        # when it is about to write "this file is missing".
        out["note"] = (
            "no path matches this glob at the PR's state — the glob is "
            "fnmatch'd against the full repo-relative path, so check the "
            "path shape ('dir/*' for a subtree, '*/name.py' for a "
            "basename anywhere) before concluding the file is absent"
        )
    return json.dumps(out, indent=2)


def local_grep_deps(
    args: dict,
    *,
    roots: list[Path],
    max_files_scanned: int | None = None,
) -> str:
    """grep_repo over the deployment's resolved dependency-source trees
    (`corpus="deps"`) — a Go module cache, a vendor dir, node_modules, a
    site-packages tree, whatever a deployment's CI runner has already
    materialized. Answers "what is this library's API at the pinned
    version" from the actual source instead of the model's training-data
    memory (cora issue #23).

    Differs from `local_grep_repo` in exactly the ways the second corpus
    needs: potentially several `roots` (each match's `path` is prefixed
    with which root matched, since roots can share relative paths);
    `_DEP_GREP_SKIP_DIRS` instead of `_GREP_SKIP_DIRS` (`node_modules` /
    `.venv` / `venv` are the corpus here, not build noise to skip); and a
    `max_files_scanned` walk cap, because a dependency tree can run into
    the hundreds of MB where a repo checkout does not — hitting it
    truncates with an explicit note rather than a silent partial scan.
    Same `_GREP_FILE_SIZE_MAX` / `_GREP_HARD_MAX_COUNT` / `_GREP_LINE_MAX`
    as the repo corpus: a single file worth quoting, or a result worth
    returning, isn't bigger just because the corpus is.

    `roots` is pre-resolved by the caller (`GitProvider.from_config` via
    `_resolve_dep_source_roots`: validated to exist, auto-detected from
    in-repo `vendor/`/`node_modules/` when `DEP_SOURCE_ROOTS` is unset).
    An empty list means this deployment has no dependency corpus at all
    — reported as a plain one-line message, not `ERROR:`, so the model
    learns the corpus is absent instead of retrying the call."""
    if not roots:
        return (
            "no dependency-source corpus configured for this deployment "
            "(DEP_SOURCE_ROOTS unset, and no vendor/ or node_modules/ "
            'found under the checkout) — grep_repo(corpus="repo") is the '
            "only corpus available here"
        )
    parsed = _parse_grep_query(args)
    if isinstance(parsed, str):
        return parsed
    pattern, rx, glob = parsed["pattern"], parsed["rx"], parsed["glob"]
    max_count, context_lines = parsed["max_count"], parsed["context_lines"]
    cap = (
        max_files_scanned
        if max_files_scanned is not None
        else DEP_SOURCE_MAX_FILES_SCANNED
    )

    # Two bounds, because they answer different questions. `cap` bounds
    # how many CANDIDATE files (post-glob) get opened — the useful-work
    # budget. `traversal_cap` bounds how many directory entries get
    # looked at at all, so a glob that matches nothing still terminates
    # on a hundreds-of-thousands-of-files module cache instead of walking
    # it whole. Generous multiple: the traversal itself is cheap (stat,
    # no read) next to opening and scanning a file.
    traversal_cap = cap * _DEP_TRAVERSAL_CAP_MULTIPLE
    entries_traversed = 0
    traversal_cap_hit = False

    matches: list[dict] = []
    files_scanned = 0
    files_walked = 0
    files_matched: set[str] = set()
    truncated = False
    budget_hit = False
    scan_cap_hit = False
    chars_used = 0
    for root in roots:
        if truncated:
            break
        root_glob = _normalize_glob(glob, root) if glob else None
        static = _glob_static_segments(root_glob) if root_glob else []
        for dirpath, dirnames, filenames in os.walk(root):
            # Sorted so a cap-truncated result is reproducible: os.walk
            # yields directories in filesystem order, which would make two
            # reviews of one PR disagree about whether a symbol exists.
            # Symlinked dirs are dropped — same containment reasoning as
            # the auto-detected roots (providers/git.py).
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _DEP_GREP_SKIP_DIRS
                and not os.path.islink(os.path.join(dirpath, d))
            )
            rel_dir = os.path.relpath(dirpath, root)
            depth = 0 if rel_dir == "." else len(rel_dir.split(os.sep))
            if depth < len(static):
                # Still inside the glob's wildcard-free prefix — only the
                # one named directory can lead to a match.
                dirnames[:] = [d for d in dirnames if d == static[depth]]
            if truncated:
                break
            for fn in sorted(filenames):
                if truncated:
                    break
                entries_traversed += 1
                if entries_traversed > traversal_cap:
                    truncated = True
                    traversal_cap_hit = True
                    break
                if os.path.splitext(fn)[1].lower() in _GREP_SKIP_EXT:
                    continue
                abs_path = Path(dirpath) / fn
                if abs_path.is_symlink():
                    continue
                rel_to_root = os.path.relpath(abs_path, root)
                if root_glob and not fnmatch.fnmatch(rel_to_root, root_glob):
                    continue
                # Counted AFTER the glob, not before: a dep tree runs to
                # hundreds of thousands of files, so a pre-glob cap let
                # unrelated files eat the whole budget and a precisely
                # targeted lookup returned zero matches — while the note
                # advised narrowing the glob, which could not help.
                files_walked += 1
                if files_walked > cap:
                    truncated = True
                    scan_cap_hit = True
                    break
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
                # Label with the matched root's directory name — the
                # deps corpus can have several roots, so a bare
                # root-relative path is ambiguous about provenance.
                rel_path = f"{root.name}/{rel_to_root}"
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
                    chars_used += len(entry["content"]) + len(rel_path) + 40
                    if context_lines:
                        chars_used += sum(
                            len(s) for s in entry["before"] + entry["after"]
                        )
                    if matches and chars_used > TOOL_RESULT_CHAR_CAP:
                        truncated = True
                        budget_hit = True
                        break
                    matches.append(entry)
                    files_matched.add(rel_path)
                    if len(matches) >= max_count:
                        truncated = True
                        break

    out = {
        "pattern": pattern,
        "glob": glob,
        "corpus": "deps",
        "roots": [str(r) for r in roots],
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "files_scanned": files_scanned,
        "files_matched": len(files_matched),
    }
    if scan_cap_hit:
        out["note"] = (
            f"dependency-corpus scan cap ({cap} candidate files) reached "
            f"— {files_scanned} files searched, {len(matches)} matches; "
            "narrow with a tighter glob or pattern, or scope "
            "DEP_SOURCE_ROOTS to search more of the corpus"
        )
    elif traversal_cap_hit:
        out["note"] = (
            f"dependency-corpus traversal ceiling ({traversal_cap} "
            f"entries) reached before the glob selected {cap} files — "
            "the corpus is large and this glob matches little of it; "
            "anchor the glob at the package subtree you mean "
            "(e.g. 'pkgname/*'), or scope DEP_SOURCE_ROOTS"
        )
    elif budget_hit:
        out["note"] = (
            f"result char budget ({TOOL_RESULT_CHAR_CAP}) reached at "
            f"{len(matches)} matches — narrow with a tighter pattern, a "
            "glob, or context_lines=0"
        )
    elif glob and files_scanned == 0:
        out["note"] = (
            "glob selected zero files across the configured roots — it "
            "is fnmatch'd against each root-relative path; use 'dir/*' "
            "for a subtree or check the path"
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
        content, truncated = _cap_content(proc.stdout)
        payload = {"ref": ref, "path": path, "content": content}
        if truncated:
            payload["truncated"] = True
            payload["total_chars"] = len(proc.stdout)
        return json.dumps(payload, indent=2)

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
