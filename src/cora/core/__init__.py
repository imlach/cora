"""Agentic PR-review engine internals.

Contents fall into a few groups:

- Agent + LLM plumbing — `agent.py` (Pydantic-AI factory),
  `budget.py` (token + tool-call accounting), `quick_review.py` +
  `deep_review.py` (quick / deep call helpers).
- PR context + retrieval — `pr_context.py`, `retrieval.py`,
  `prefetch.py`, `prompt.py`, `repo_tools.py`.
- GitHub side effects — `comment.py`, `check_run.py`,
  `patch_dispatch.py`, `summary.py`.
- Output validation — `leak.py` (reasoning-leak guard + verdict
  parser).

Re-exports below provide backward compatibility for downstream test
suites that predate the `cora` package name and still import these
helpers from the package root. Keep the list in sync with what those
tests actually use.
"""

# Re-exports for backward compat with pre-rename downstream tests.
from cora.core.leak import (  # noqa: F401
    detect_reasoning_leak,
    parse_verdict_from_body,
)
from cora.core.patch_dispatch import (  # noqa: F401
    _format_suggestion_body,
    _is_in_hunk,
    find_line_range,
    parse_pr_diff_hunks,
)
