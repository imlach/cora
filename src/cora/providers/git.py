"""Git/SCM providers — how cora introspects the repo under review.

`grep_repo` and `git_show` are the two repo-introspection tools the agent
calls. `LocalGitProvider` serves them from a local checkout (the PR merge
tree), so existence checks reflect the code actually under review rather
than a `main`-branch mirror — and it's all an adopter running cora in CI
needs. The seam lets a future provider back these with an SCM API instead.

`LocalGitProvider` also serves `grep_repo`'s second corpus
(`corpus="deps"`, cora issue #23) over the deployment's resolved
dependency source. `_resolve_dep_source_roots` is where `DEP_SOURCE_ROOTS`
config turns into an existence-checked, possibly auto-detected root list
— run once at startup in `from_config`, not per call, since it does
filesystem checks and belongs at the same "resolve config into a running
provider" point the checkout root itself is resolved at.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core import config as _c
from cora.core.repo_tools import local_git_show, local_grep_deps, local_grep_repo

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


def _resolve_dep_source_roots(cfg: "ReviewerConfig", repo_root: Path) -> list[Path]:
    """`DEP_SOURCE_ROOTS`, existence-validated; auto-detects in-repo
    vendored trees when unset.

    An explicit `cfg.dep_source_roots` list wins outright and is never
    second-guessed by auto-detection — a deployment that names its own
    roots knows what it wants searched. Each configured path is checked
    with `Path.is_dir()`; a missing one is dropped with a `::warning::`
    rather than failing the review, matching the engine's soft-fail
    posture elsewhere (a stale corpus path should cost the corpus, not
    the whole run). Bare `print`, not the `_gha_log`/`log=` callback
    threaded through the agent loop — this runs once at startup, outside
    any loop that injects a log sink, same as the rest of the engine's
    `::warning::` call sites (`review/_preflight.py` et al.)."""
    configured = [Path(p) for p in cfg.dep_source_roots]
    if configured:
        resolved: list[Path] = []
        for root in configured:
            if root.is_dir():
                resolved.append(root)
            else:
                print(
                    f"::warning::DEP_SOURCE_ROOTS entry not found, "
                    f"dropping: {root}"
                )
        return resolved
    return [
        repo_root / d
        for d in _c.DEP_SOURCE_AUTO_DIRS
        if (repo_root / d).is_dir()
    ]


class GitProvider(ABC):
    """Repo-introspection backend for the reviewer's `grep_repo` /
    `git_show` tools. Both take the MCP server's `args: dict` call shape
    and return the same JSON-string envelope."""

    @abstractmethod
    def grep_repo(self, args: dict) -> str: ...

    @abstractmethod
    def git_show(self, args: dict) -> str: ...

    @classmethod
    def from_config(cls, cfg: "ReviewerConfig") -> "GitProvider":
        """`LocalGitProvider` is the only implementation today (serves the
        in-CI checkout). An SCM-API-backed provider would branch here."""
        return LocalGitProvider(
            dep_source_roots=_resolve_dep_source_roots(cfg, _c.REPO_ROOT),
            dep_source_max_files_scanned=cfg.dep_source_max_files_scanned,
        )


class LocalGitProvider(GitProvider):
    """grep + git show against a local checkout — the PR merge tree.
    `repo_root` defaults to the engine's resolved `CORA_REPO_ROOT`.

    `dep_source_roots` is the (already existence-checked) dependency-
    source corpus for `grep_repo(corpus="deps")`; empty (the default)
    means this instance has no deps corpus — direct construction (e.g.
    in tests) opts out of `DEP_SOURCE_ROOTS` resolution and auto-detect,
    both of which only run through `GitProvider.from_config`."""

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        dep_source_roots: list[Path] | None = None,
        dep_source_max_files_scanned: int = _c.DEP_SOURCE_MAX_FILES_SCANNED,
    ) -> None:
        self.repo_root = repo_root if repo_root is not None else _c.REPO_ROOT
        self.dep_source_roots = dep_source_roots or []
        self.dep_source_max_files_scanned = dep_source_max_files_scanned

    def grep_repo(self, args: dict) -> str:
        corpus = (args.get("corpus") or "repo").strip().lower() or "repo"
        if corpus == "deps":
            return local_grep_deps(
                args,
                roots=self.dep_source_roots,
                max_files_scanned=self.dep_source_max_files_scanned,
            )
        if corpus != "repo":
            return (
                f"ERROR: grep_repo: corpus must be 'repo' or 'deps' "
                f"(got {corpus!r})"
            )
        return local_grep_repo(args, root=self.repo_root)

    def git_show(self, args: dict) -> str:
        return local_git_show(args, root=self.repo_root)
