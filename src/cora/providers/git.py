"""Git/SCM providers — how cora introspects the repo under review.

`grep_repo` and `git_show` are the two repo-introspection tools the agent
calls. `LocalGitProvider` serves them from a local checkout (the PR merge
tree), so existence checks reflect the code actually under review rather
than a `main`-branch mirror — and it's all an adopter running cora in CI
needs. The seam lets a future provider back these with an SCM API instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core import config as _c
from cora.core.repo_tools import local_grep_repo, local_git_show

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


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
        return LocalGitProvider()


class LocalGitProvider(GitProvider):
    """grep + git show against a local checkout — the PR merge tree.
    `repo_root` defaults to the engine's resolved `CORA_REPO_ROOT`."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root if repo_root is not None else _c.REPO_ROOT

    def grep_repo(self, args: dict) -> str:
        return local_grep_repo(args, root=self.repo_root)

    def git_show(self, args: dict) -> str:
        return local_git_show(args, root=self.repo_root)
