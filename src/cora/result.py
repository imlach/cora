"""`ReviewResult` — the structured outcome of a review.

`run_review` returns one of these; side-effects (posting the comment,
finalizing the check-run, gating automerge) happen through a `Reporter`,
not inside the result. That keeps the engine output inspectable — eval /
dry-run runs a review with a `NullReporter` and reads the result directly,
no GitHub mocking required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cora.core.budget import Budget


@dataclass
class ReviewResult:
    """What a review produced. Outcome only — no review *context* (repo,
    pr_number, model, started_at live on the caller / the Reporter)."""

    verdict: str | None          # parsed verdict word, or None → suppressed
    verdict_line: str | None     # the check-title verdict line (glyph + word)
    conclusion: str              # GitHub check conclusion: success/neutral/failure/cancelled
    body: str                    # review comment body (post leak-strip), pre-footer
    mode: str                    # "quick" | "deep"
    budget: Budget
    wall_time_s: float
    terminated_reason: str | None = None
    tools_available: list[str] | None = None
    bot_author: bool = False
    retrieval_source: str = "none"
    retrieval_trace: dict = field(default_factory=dict)
    pause_automerge: bool = False
    reasoning_stripped_chars: int = 0
    # Escalation path — the tier model aliases that actually ran, in
    # order (e.g. ["sonnet", "opus"]). Single-tier
    # reviews carry one entry.
    tiers_run: list[str] = field(default_factory=list)
