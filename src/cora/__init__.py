"""cora — a convention-oriented review agent.

Self-hostable agentic PR reviewer. The engine lives in `cora.core`.
`run_review` is the entry function (`run_review(ReviewerConfig.from_env())`
is the whole workflow entrypoint); `ReviewerConfig` is the public config
surface and `RetrievalProvider` / `GitProvider` / `Reporter` the provider
seams it accepts.
"""

from cora.config import ReviewerConfig
from cora.escalation import (
    EscalationConnector,
    EscalationPolicy,
    ReprefillConnector,
    Tier,
)
from cora.providers import GitProvider, Reporter, RetrievalProvider
from cora.result import ReviewResult
from cora.review import run_review
from cora.trigger import TriggerContext, TriggerDecision, TriggerPolicy

__all__ = [
    "run_review",
    "ReviewerConfig",
    "ReviewResult",
    "RetrievalProvider",
    "GitProvider",
    "Reporter",
    "EscalationPolicy",
    "EscalationConnector",
    "ReprefillConnector",
    "Tier",
    "TriggerPolicy",
    "TriggerContext",
    "TriggerDecision",
]
try:
    # Written by the hatch-vcs build hook (per-commit versions so a
    # rebuilt wheel never no-ops at install). Absent on a bare source
    # tree that was never built/installed.
    from cora._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
