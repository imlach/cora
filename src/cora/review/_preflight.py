"""Run construction + preflight — everything before any model spend.

`build_run` resolves the providers (eval mode forces a `NullReporter`)
and derives the identity/mode fields; `preflight` opens the verdict
check-run (created up-front in EVERY mode: it is the signal a
required-check aggregator gates on), arms the SIGTERM guard, and walks
the preflight skip ladder — missing PR identity, missing LLM
secret, missing system prompt, failed metadata fetch. Every skip
completes the check with a real conclusion (`cancelled` is tolerated by
the gate: transient, re-push to retry).
"""

from __future__ import annotations

from datetime import datetime, timezone

from cora.core import pr_context as _prc
from cora.core.budget import PER_CALL_TIMEOUT_S
from cora.core.check_run import _workflow_run_url
from cora.core.log import _gha_log
from cora.core.prompt import load_system_prompt
from cora.providers.git import GitProvider
from cora.providers.reporter import NullReporter, Reporter
from cora.providers.retrieval import RetrievalProvider
from cora.result import ReviewResult
from cora.review._signals import _TIMEOUT_GUARD, _arm_timeout_guard
from cora.review._state import ReviewRun
from cora.config import ReviewerConfig
from cora.second_opinion import SecondOpinionProvider


def build_run(
    cfg: ReviewerConfig,
    *,
    reporter: Reporter | None,
    retrieval: RetrievalProvider | None,
    git: GitProvider | None,
    second_opinion: SecondOpinionProvider | None,
) -> ReviewRun:
    """Resolve providers and derive the run's identity/mode fields."""
    # In-log mirror of the run URL (the PR-comment footer carries it too).
    run_url = _workflow_run_url()
    if run_url:
        _gha_log(f"workflow run: {run_url}")

    # Wall-clock start — surfaced in the in-progress and final comment
    # footers so "started vs completed" is readable in place.
    started_at = datetime.now(timezone.utc)

    eval_mode = bool((cfg.eval_output_dir or "").strip())
    if reporter is None or eval_mode:
        if eval_mode:
            # Eval mode is write-free by contract: every externalised GH
            # side-effect the entrypoint short-circuited per call site is
            # suppressed wholesale by the NullReporter.
            if reporter is not None:
                _gha_log("eval mode: supplied reporter replaced by NullReporter")
            else:
                _gha_log("eval mode: side-effects suppressed (NullReporter)")
            reporter = NullReporter()
        else:
            reporter = Reporter.from_config(cfg, started_at=started_at)
    if retrieval is None:
        retrieval = RetrievalProvider.from_config(cfg)
    if git is None:
        git = GitProvider.from_config(cfg)
    if second_opinion is None:
        second_opinion = SecondOpinionProvider.from_config(cfg)

    # Mode is derived from the iteration budget. 0 → quick (no MCP, no
    # tool loop, single LLM call); >0 → deep (full agent loop).
    max_iterations = cfg.max_tool_iterations
    is_quick = max_iterations <= 0

    return ReviewRun(
        cfg=cfg,
        reporter=reporter,
        retrieval=retrieval,
        git=git,
        second_opinion=second_opinion,
        eval_mode=eval_mode,
        started_at=started_at,
        pr_number=(cfg.pr_number or "").strip(),
        repo=(cfg.repo or "").strip(),
        api_key=(cfg.llm_api_key or "").strip(),
        base_url=cfg.llm_base_url.strip(),
        model=cfg.model.strip(),
        mcp_url=cfg.mcp_url.strip(),
        max_iterations=max_iterations,
        is_quick=is_quick,
        mode="quick" if is_quick else "deep",
        per_call_timeout_s=(
            int(cfg.per_call_timeout_s)
            if cfg.per_call_timeout_s
            else PER_CALL_TIMEOUT_S
        ),
    )


def preflight(run: ReviewRun) -> ReviewResult | None:
    """The preflight skip ladder. Returns a skip result, or
    None to proceed. Side-effects: opens the verdict check-run, arms the
    SIGTERM guard, loads the system prompt, fetches PR metadata."""
    cfg = run.cfg
    if not run.pr_number or not run.repo:
        _gha_log("::warning::repo and pr_number must be set — skipping review")
        return run.skip_result(
            "skipped (repo / pr_number not set)", "missing-pr-identity"
        )

    # Verdict check-run — created up-front in EVERY mode: it is
    # the authoritative signal a required-check aggregator gates on. Every
    # exit path completes it with a real conclusion; the SIGTERM guard
    # finalizes it to `timed_out` on a hard kill.
    run.head_sha = _prc._pr_head_sha()
    if run.head_sha:
        run.reporter.open_progress(run.head_sha)
        if run.reporter.check_open:
            _arm_timeout_guard(run.reporter)
            # A hard kill never returns through the pipeline wrapper, so
            # hand the guard a closure that can still close the log
            # stream. Idempotent — a normal exit that already emitted
            # makes this a no-op.
            from cora.review._output import emit_finish

            _TIMEOUT_GUARD["finish"] = lambda: emit_finish(
                run, terminated_reason="gha_timeout"
            )
    else:
        print("::warning::could not derive PR head SHA; verdict check skipped")

    if not run.api_key:
        print("::warning::LLM gateway key not set — skipping review")
        try:
            run.reporter.post_skip(
                "LLM_GATEWAY_KEY secret not configured. The reviewer "
                "will run once the secret is set."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::could not post skip comment: {exc}")
        run.reporter.complete_check(
            verdict_line="skipped (LLM_GATEWAY_KEY not configured)",
            conclusion="cancelled",
            budget=None,
            wall_time_s=0.0,
            terminated_reason="secret-missing",
        )
        return run.skip_result(
            "skipped (LLM_GATEWAY_KEY not configured)", "secret-missing"
        )

    # System prompt. An explicitly configured path that doesn't exist is
    # the entrypoint's "prompt-missing" preflight skip; a None path falls
    # back to the prompt cora packages for the mode.
    prompt_path = cfg.quick_prompt_path if run.is_quick else cfg.deep_prompt_path
    if prompt_path is not None and not prompt_path.exists():
        print(f"::warning::system prompt not found at {prompt_path}")
        run.reporter.complete_check(
            verdict_line="skipped (system prompt missing)",
            conclusion="cancelled",
            budget=None,
            wall_time_s=0.0,
            terminated_reason="prompt-missing",
        )
        return run.skip_result("skipped (system prompt missing)", "prompt-missing")
    run.system_prompt = load_system_prompt(prompt_path, mode=run.mode, cfg=cfg)

    try:
        metadata = _prc.fetch_pr_metadata(run.pr_number)
        metadata["number"] = run.pr_number
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::PR metadata fetch failed: {exc}")
        run.reporter.complete_check(
            verdict_line="skipped (PR metadata fetch failed)",
            conclusion="cancelled",
            budget=None,
            wall_time_s=0.0,
            terminated_reason="metadata-fetch-failed",
        )
        return run.skip_result(
            "skipped (PR metadata fetch failed)", "metadata-fetch-failed"
        )
    run.metadata = metadata
    return None
