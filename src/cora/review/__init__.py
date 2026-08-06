"""`run_review` — the review orchestrator.

Every GitHub side-effect is routed through the
`Reporter` seam and every tunable read from `ReviewerConfig` where a
field exists. The sequencing, skip reasons, verdict→conclusion mapping,
wall-hit→T1 continuation, leak handling, automerge-pause condition and
propose_patch dispatch below are the review pipeline; with a
default-constructed config the defaults drive every decision.

Two modes, one function. `cfg.max_tool_iterations == 0` → quick (single
LLM call, no MCP); `> 0` → deep (full agent loop with MCP tools).

Package map — one module per pipeline phase, `ReviewRun` (in `_state`)
is the shared state object each phase mutates; `_arun_review_inner`
below is the whole control flow:

    _state      ReviewRun + the loki/iter-log/skip plumbing
    _signals    SIGTERM safety net for the progress check-run
    _preflight  provider resolution, identity/secret/prompt/metadata skips
    _gate       TriggerPolicy gate + GHA rate-cap probe
    _context    concurrent prefetches, retrieval pre-pack, prompt assembly
    _tiers      quick call / deep loop, escalation ladder, second opinion
    _output     leak pipeline, observability trail, CI-verdict gate
                (`_ci_gate`, issue #23 — pre-check-post), verdict parse
    _patches    propose_patch dispatch + T2 patch escalation
    _finalize   ReviewResult, eval dumps, comment post, automerge pause

Side-effect discipline:
  - All externalised writes (check-run lifecycle, comments, step
    summary, automerge label, propose_patch dispatch, escalation
    labels) go through the `Reporter`.
  - Observability (`_gha_log` lines, eval dumps) stays here — it is a
    property of the run, not of the SCM.
  - Structured-log streaming is deployment-specific;
    `_loki_push` below is a package-level no-op hook for it, so
    a deployment can graft its own structured log sink onto it
    (`cora.review._loki_push = push_line`) — phase modules resolve the
    attribute at call time, so a graft applied after import is honoured.

Eval mode (`cfg.eval_output_dir` set) forces a `NullReporter` — one
side-effect-free reporter instead of per-write
short-circuits. The `.md` / `.trace.json`
dumps are written by the phases.

Every workflow env knob has a `ReviewerConfig` field and is read
from cfg here; `ReviewerConfig.from_env()` folds the workflow env in
(tier escalation `T1_MODEL` / `AGENT_REVIEW_T1_*` / `AGENT_REVIEW_T2_*`
/ `AGENT_REVIEW_SKIP_T0` / `AGENT_REVIEW_PATCH_ESCALATION`, the
`T0_WALL_TIME_S` / `T1_WALL_TIME_S` / `WALL_TIME_S` ops overrides,
`CLASSIFIER_LABEL`, `AGENT_REVIEW_CONTEXT_INJECTION*`,
`AGENT_REVIEW_CI_VERDICT_GATE`, `REVIEWER_TRANSCRIPT_SOURCE`). Deliberately
still env-read (runtime
facts or env-folded at their own layer, NOT reviewer config):
  - `AGENT_REVIEW_CACHE_DIR` / `AGENT_REVIEW_CACHE_TTL_S` — call-time
    ops overrides inside `core.retrieval` (config-mirrored defaults).
  - `AGENT_REVIEW_ENABLE_THINKING` — legacy fallback inside
    `deep_review._thinking_extra_body` for direct engine calls with
    cfg=None; this orchestrator always threads `cfg.enable_thinking`.
    `AGENT_REVIEW_PATCH_ESCALATION` and the context-injection toggles
    keep the same cfg=None env fallback in their home modules.
  - `AGENT_REVIEW_PER_CALL_TIMEOUT_S` /
    `AGENT_REVIEW_T0_COLD_START_ALLOWANCE_S` — import-time constants
    in `core.budget` (per-call timeout is config-mirrored).
  - `CORA_GH_TOKEN` — App-token fallback in the reporter / comment /
    check_run subprocess env (config-mirrored as `github_app_token`).
  - `AGENT_REVIEW_LOOP_GUARD_BOT_LOGIN` — ctor fallback in
    `ContextRefresher` when no bot_login is passed (threaded from
    `cfg.loop_guard_bot_login` here).
  - GitHub-Actions runtime identity (`GITHUB_RUN_ID`, `GITHUB_SHA`,
    `GITHUB_STEP_SUMMARY`, `GITHUB_EVENT_PATH`, `GH_REPO`, …) — facts
    of the host run, not configuration.

Deliberate design choices worth knowing:
  - Structured-log pushes are a no-op hook (`_loki_push`) rather than
    a live stream — the structured lines still land in the GHA log.
  - In eval mode the NullReporter swallows `write_summary` too, so
    even `$GITHUB_STEP_SUMMARY` stays untouched.
  - Eval mode logs a single "eval mode" line at reporter selection
    rather than per-write `eval mode: skipped <action>` lines.
"""

from __future__ import annotations

import asyncio

from cora.config import ReviewerConfig
from cora.providers.git import GitProvider
from cora.providers.reporter import Reporter
from cora.providers.retrieval import RetrievalProvider
from cora.result import ReviewResult
from cora.review._context import assemble_context
from cora.review._finalize import finalize

# Underscore-name re-exports: the back-compat surface tests and
# embedders reach via `cora.review.<name>`. `_TIMEOUT_GUARD` is the
# same dict object `_signals` mutates.
from cora.review._gate import _recent_run_counts, trigger_gate  # noqa: F401
from cora.review._output import (  # noqa: F401
    emit_finish,
    produce_output,
    quick_review_retry_for_format,
)
from cora.review._patches import dispatch_patches
from cora.review._preflight import build_run, preflight
from cora.review._signals import (  # noqa: F401
    _TIMEOUT_GUARD,
    _arm_timeout_guard,
    _finalize_check_on_signal,
    _on_sigterm,
)
from cora.review._state import ReviewRun, _eval_dump  # noqa: F401
from cora.review._tiers import (  # noqa: F401
    _default_policy,
    _continuation_tier_runner,
    _wall_budgets,
    dispatch_tiers,
    second_opinion_dispatch,
)
from cora.second_opinion import SecondOpinionProvider

__all__ = ["run_review"]


def _loki_push(line: str, labels: dict | None = None) -> None:
    """No-op structured-log hook. Important lines fan out to
    the GHA log and to this hook; a deployment
    can monkeypatch/replace it with a live pusher for its
    structured log sink (e.g. a Loki pusher)."""


def run_review(
    cfg: ReviewerConfig,
    *,
    reporter: Reporter | None = None,
    retrieval: RetrievalProvider | None = None,
    git: GitProvider | None = None,
    second_opinion: SecondOpinionProvider | None = None,
) -> ReviewResult:
    """Run one PR review and return its `ReviewResult`.

    Providers default via each seam's `from_config(cfg)`:
    `Reporter.from_config` (GitHubReporter when repo+PR are set, else
    NullReporter; eval mode always forces NullReporter),
    `RetrievalProvider.from_config` (TEI+Qdrant when fully configured,
    else retrieval-free) and `GitProvider.from_config` (local checkout).

    Soft-fail discipline: infra failures and skip
    paths return a result (conclusion `cancelled`/`failure`) rather than
    raising — the check-run conclusion carries the signal."""
    return asyncio.run(
        _arun_review(
            cfg,
            reporter=reporter,
            retrieval=retrieval,
            git=git,
            second_opinion=second_opinion,
        )
    )


async def _arun_review(
    cfg: ReviewerConfig,
    *,
    reporter: Reporter | None,
    retrieval: RetrievalProvider | None,
    git: GitProvider | None,
    second_opinion: SecondOpinionProvider | None,
) -> ReviewResult:
    # OTel init — exports spans when the env var is set, no-op otherwise.
    # Flushed before return; BatchSpanProcessor would otherwise drop
    # pending spans on a short-lived run.
    from cora.core.otel import init_tracing

    _otel_provider = init_tracing()
    try:
        return await _arun_review_inner(
            cfg,
            reporter=reporter,
            retrieval=retrieval,
            git=git,
            second_opinion=second_opinion,
        )
    finally:
        if _otel_provider is not None:
            try:
                _otel_provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"::warning::otel shutdown failed: {exc}")


async def _arun_review_inner(
    cfg: ReviewerConfig,
    *,
    reporter: Reporter | None,
    retrieval: RetrievalProvider | None,
    git: GitProvider | None,
    second_opinion: SecondOpinionProvider | None,
) -> ReviewResult:
    """The pipeline, wrapped so every exit closes the log stream.

    `_pipeline` is where the phases live; this wrapper's only job is the
    `agent_review finish` line. A review that logs its turns and then
    goes silent is indistinguishable from one still running, so the
    finish line has to survive the skip returns AND cancellation —
    `BaseException` deliberately, since `CancelledError` is the case
    that produced the silent runs."""
    run = build_run(
        cfg,
        reporter=reporter,
        retrieval=retrieval,
        git=git,
        second_opinion=second_opinion,
    )
    try:
        result = await _pipeline(run)
    except BaseException as exc:
        emit_finish(run, terminated_reason=f"cancelled:{type(exc).__name__}")
        raise
    # No-op on the normal path — `produce_output` already emitted with
    # the leak/preamble detail. This catches the skip returns, which
    # never reach it.
    emit_finish(run, terminated_reason=result.terminated_reason)
    return result


async def _pipeline(run: ReviewRun) -> ReviewResult:
    """Each phase mutates `run` and may return a terminal `ReviewResult`
    (a skip — conclusion `cancelled` for transient infra and policy
    denials, `failure` for unusable model output); the first one wins."""
    if (skip := preflight(run)) is not None:
        return skip
    if (skip := trigger_gate(run)) is not None:
        return skip
    if (skip := await assemble_context(run)) is not None:
        return skip
    if (skip := await dispatch_tiers(run)) is not None:
        return skip
    await second_opinion_dispatch(run)
    if (skip := await produce_output(run)) is not None:
        return skip
    await dispatch_patches(run)
    return finalize(run)
