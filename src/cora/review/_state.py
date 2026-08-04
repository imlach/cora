"""`ReviewRun` — the single state object threaded through the pipeline.

The orchestrator used to be one 1,600-line function whose phases shared
~40 locals; this dataclass is those locals, named, grouped, and
documented. Each phase module mutates the fields it owns and the
orchestrator in `cora/review/__init__.py` stays a linear list of phase
calls. Field names deliberately match the log lines and events they
feed — renames would blur that audit trail.

Two late-binding contracts preserved from the closure version:

- `iter_log`/`loki` read `self.mode` **at call time** — after the
  trigger gate forces a degraded run down quick mode, subsequent log
  labels say `kind=quick`, exactly like the old late-bound closures.
- `loki_push` resolves the package-level `_loki_push` hook at call time,
  so a deployment's graft (`cora.review._loki_push = push_line`, applied
  after import) is honoured by every phase module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core.budget import Budget
from cora.result import ReviewResult
from cora.second_opinion import SecondOpinionResult

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.providers.git import GitProvider
    from cora.providers.reporter import Reporter
    from cora.providers.retrieval import RetrievalProvider
    from cora.second_opinion import SecondOpinionProvider


def loki_push(line: str, labels: dict | None = None) -> None:
    """Call the package-level `_loki_push` hook, late-bound so a pusher
    grafted onto `cora.review` after import is honoured."""
    from cora import review as _pkg

    _pkg._loki_push(line, labels)


def _eval_dump(cfg: "ReviewerConfig", filename: str, content: str) -> None:
    """Write `content` under the eval output dir. Soft-fail — eval-mode
    runs aren't load-bearing for a real PR."""
    try:
        out_dir = Path(str(cfg.eval_output_dir).strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::eval dump {filename!r} failed: {exc}")


@dataclass
class ReviewRun:
    """Everything one review accumulates, phase by phase."""

    # ── construction (build_run) ────────────────────────────────────
    cfg: "ReviewerConfig"
    reporter: "Reporter"
    retrieval: "RetrievalProvider"
    git: "GitProvider"
    second_opinion: "SecondOpinionProvider"
    eval_mode: bool
    started_at: datetime
    pr_number: str
    repo: str
    api_key: str
    base_url: str
    model: str
    mcp_url: str
    max_iterations: int
    is_quick: bool
    mode: str
    per_call_timeout_s: int

    # ── preflight ───────────────────────────────────────────────────
    head_sha: str | None = None
    system_prompt: str = ""
    metadata: dict = field(default_factory=dict)

    # ── trigger gate ────────────────────────────────────────────────
    trigger_degraded: bool = False

    # ── context assembly ────────────────────────────────────────────
    bot_author: bool = False
    deps_labelled: bool = False
    diff_raw: str = ""
    diff_text: str = ""
    diff_truncated: bool = False
    retrieved_docs: list[dict] = field(default_factory=list)
    retrieval_source: str = "none"
    retrieval_trace: dict = field(default_factory=dict)
    prefetched_release_notes: str | None = None
    prefetch_status: str | None = None
    prefetch_url: str | None = None
    initial_user_prompt: str = ""

    # ── tier dispatch ───────────────────────────────────────────────
    budget: Budget | None = None
    start: float = 0.0
    t0_wall_time_s: int = 0
    t1_wall_time_s: int = 0
    wall_time_total_s: int = 0
    final_body: str = ""
    terminated_reason: str | None = None
    tools_available: list[str] = field(default_factory=list)
    tiers_run: list[str] = field(default_factory=list)
    per_call_fresh_start: bool = False
    # Deep-mode dispatch surface, reused by the second-opinion seam and
    # the propose_patch T2 verifier. `None` in quick mode (which never
    # dispatches either consumer through a tool-bearing call).
    mcp_headers: dict | None = None
    mcp_actions_url: str | None = None
    mcp_actions_headers: dict | None = None
    web_fetch_url: str | None = None

    # ── second opinion ──────────────────────────────────────────────
    second_opinion_result: SecondOpinionResult = field(
        default_factory=SecondOpinionResult
    )

    # ── output pipeline ─────────────────────────────────────────────
    wall_time_s: float = 0.0
    body_to_post: str = ""
    verdict: str | None = None
    reasoning_stripped_chars: int = 0
    # Exactly one `agent_review finish` line per review, on whichever
    # path exits first. Set by `_output.emit_finish`; the pipeline
    # wrapper and the SIGTERM guard both call it, and this flag is what
    # stops the normal path emitting twice.
    finish_emitted: bool = False

    # ── propose_patch ───────────────────────────────────────────────
    patch_directive: object | None = None

    # ── shared plumbing ─────────────────────────────────────────────

    def loki(self, line: str, labels: dict | None = None) -> None:
        loki_push(line, labels)

    def iter_log(self, line: str) -> None:
        """Per-event log fan-out: GHA workflow UI + (hook-only here)
        Loki. `kind` reads the *current* mode, as the closure did."""
        from cora.core.log import _gha_log

        _gha_log(line)
        loki_push(line, labels={"consumer": "pr-review", "kind": self.mode})

    def skip_result(
        self,
        verdict_line: str,
        terminated_reason: str,
        *,
        conclusion: str = "cancelled",
        budget: Budget | None = None,
        wall_time_s: float = 0.0,
        tiers_run: list[str] | None = None,
    ) -> ReviewResult:
        return ReviewResult(
            verdict=None,
            verdict_line=verdict_line,
            conclusion=conclusion,
            body="",
            mode=self.mode,
            budget=budget
            if budget is not None
            else Budget(max_input=0, max_output=0, max_iterations=0),
            wall_time_s=wall_time_s,
            terminated_reason=terminated_reason,
            tiers_run=tiers_run or [],
        )

    def eval_dump(self, filename: str, content: str) -> None:
        _eval_dump(self.cfg, filename, content)
