"""Tier dispatch — quick call, deep agent loop, and the escalation ladder.

Holds the default `EscalationPolicy` builder (T0 → T1 continuation
expressed on the generalised ladder) and the generic `TierRunner` that
gives a `ReprefillConnector` adopter a working two-tier handoff with no
extra infrastructure. A custom `ReviewerConfig.escalation_policy`
replaces the built ladder wholesale.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cora.config import ReviewerConfig
from cora.core import config as _c
from cora.core import pr_context as _prc
from cora.core.budget import Budget
from cora.core.log import _gha_log
from cora.escalation import (
    WALL_HIT_REASONS,
    EscalationContext,
    EscalationPolicy,
    Tier,
    run_escalation,
)
from cora.result import ReviewResult
from cora.review._signals import _TIMEOUT_GUARD
from cora.review._state import ReviewRun


def _default_policy(
    cfg: ReviewerConfig,
    *,
    is_quick: bool,
    t1_enabled: bool,
    t1_model: str,
    t1_max_iterations: int,
) -> EscalationPolicy:
    """The default ladder: T0 = the configured review model; a T1
    continuation tier exists only in deep mode with
    `cfg.t1_continuation` on, escalating on `cfg.escalation_triggers`
    (wall-hit by default). The forced T1 entries (`cfg.skip_t0`
    classifier-large, and per_call_timeout with no committed messages)
    bypass the policy by design — they escalate even when the ladder is
    single-tier.

    Deep mode always carries the trajectory-resume
    `KvContinuationConnector` as the seam's default implementation — the
    `t1_continuation` flag only gates the *triggered* escalation (the T1
    rung + `escalate_on`), not the forced entries, which drive through the
    same connector even on a single-tier ladder. A generic adopter's policy
    keeps the dependency-free `ReprefillConnector`."""
    tiers = [Tier(model=cfg.model, max_iterations=cfg.max_tool_iterations)]
    if is_quick:
        return EscalationPolicy(tiers=tiers)

    from cora.core.kv_continuation import KvContinuationConnector

    connector = KvContinuationConnector()
    if t1_enabled:
        tiers.append(Tier(model=t1_model, max_iterations=t1_max_iterations))
        return EscalationPolicy(
            tiers=tiers,
            escalate_on=frozenset(cfg.escalation_triggers),
            connector=connector,
        )
    # Single-tier ladder: no triggered escalation, but the forced entries
    # (skip_t0 / per-call fresh start) still use the KV connector.
    return EscalationPolicy(tiers=tiers, connector=connector)


def _continuation_tier_runner(extra: dict):
    """Build the generic `TierRunner` the escalation driver hands the
    connector — dispatches `continue_on_t1` with the shared per-call kwargs
    from `EscalationContext.extra`.

    The deep-mode `KvContinuationConnector` ignores this and calls
    `continue_on_t1` itself (so it can thread the entry framing); it exists
    for the generic `ReprefillConnector` default, so a two-tier adopter
    (Sonnet→Opus) gets a working re-prefill handoff with no extra infra."""

    async def _run_tier(tier, prior_messages, seed_prompt):
        from cora.core.continuation import continue_on_t1

        # Same reserved-window deadline as `KvContinuationConnector`: T1
        # gets its window counted from now, absorbing T0's slack if T0
        # finished early — without it the generic tier would run with no
        # overall wall guard (only the per-call timeout).
        t1_deadline_monotonic = max(
            extra["now"]() + extra["t1_wall_time_s"],
            extra["start"] + extra["t0_wall_time_s"] + extra["t1_wall_time_s"],
        )
        return await continue_on_t1(
            endpoint_base_url=extra["endpoint_base_url"],
            llm_gateway_key=extra["llm_gateway_key"],
            t1_model_alias=tier.model,
            system_prompt=extra["system_prompt"],
            prior_messages=prior_messages,
            initial_user_prompt=seed_prompt,
            budget=extra["budget"],
            timeout_s=extra["timeout_s"],
            pr_number=extra["pr_number"],
            repo=extra["repo"],
            mcp_url=extra["mcp_url"],
            mcp_headers=extra["mcp_headers"],
            mcp_actions_url=extra["mcp_actions_url"],
            mcp_actions_headers=extra["mcp_actions_headers"],
            web_fetch_url=extra["web_fetch_url"],
            web_fetch_headers=extra["web_fetch_headers"],
            allowed_tools=extra["allowed_tools"],
            tool_arg_defaults=extra["tool_arg_defaults"],
            max_iterations=tier.max_iterations,
            loop_deadline_monotonic=t1_deadline_monotonic,
            context_refresher=extra["context_refresher"],
            gha_log=extra["iter_log"],
            cfg=extra["cfg"],
            git_provider=extra["git_provider"],
        )

    return _run_tier


def _wall_budgets(cfg: ReviewerConfig) -> tuple[int, int, int]:
    """Per-tier wall budgets `(t0_wall_s, t1_wall_s, total_wall_s)`,
    straight from config. The `T0_WALL_TIME_S` /
    `T1_WALL_TIME_S` / `WALL_TIME_S` ops env overrides are folded into
    the cfg fields by `ReviewerConfig.from_env` (a bare `WALL_TIME_S`
    pins `wall_time_override_s` so the reported total stays exact while
    the tiers take the 40/60 split of total − headroom)."""
    return cfg.t0_wall_time_s, cfg.t1_wall_time_s, cfg.wall_time_s


async def dispatch_tiers(run: ReviewRun) -> ReviewResult | None:
    """Run the review itself: budget + wall clocks, then the quick call
    or the deep agent loop, the wall-hit/forced T1 escalation through
    the connector seam, and the deep-mode skip-class failure ladder
    (transient infra → `cancelled`, re-push to retry)."""
    cfg = run.cfg
    pr_number = run.pr_number
    model = run.model

    budget = Budget.from_config(cfg)
    run.budget = budget
    # Hand the live budget to the SIGTERM finalizer so a hard-kill summary
    # carries real token / tool-call counts, not the zero placeholder.
    _TIMEOUT_GUARD["budget"] = budget
    # Wall-time guards — split per tier so a slow T0 can't starve T1.
    # T1 computes its own deadline from `now + T1` at dispatch so its
    # reserved budget is always available.
    t0_wall_time_s, t1_wall_time_s, wall_time_s = _wall_budgets(cfg)
    run.t0_wall_time_s = t0_wall_time_s
    run.t1_wall_time_s = t1_wall_time_s
    run.wall_time_total_s = wall_time_s
    start = time.monotonic()
    run.start = start
    _TIMEOUT_GUARD["start"] = start
    t0_deadline_monotonic = start + t0_wall_time_s

    start_line = (
        f"agent_review start pr_number={pr_number} repo={run.repo} model={model} "
        f"mode={run.mode} bot_author={run.bot_author} "
        f"retrieval={run.retrieval_source} "
        f"wall_time_s={wall_time_s} t0_wall_s={t0_wall_time_s} "
        f"t1_wall_s={t1_wall_time_s}"
    )
    _gha_log(start_line)
    run.loki(start_line, labels={"consumer": "pr-review", "kind": run.mode})

    # Tier-escalation knobs — threaded from config (from_env folds the
    # workflow's AGENT_REVIEW_T1_* / T1_MODEL env vars into these). The
    # alias keeps an empty→default guard for callers that
    # blank the field on a hand-built config.
    t1_model = (cfg.t1_model or "").strip() or _c.DEFAULT_T1_MODEL
    t1_max_iterations = cfg.t1_max_iterations
    # A caller-supplied policy (tiers + triggers + connector) replaces the
    # built default ladder wholesale.
    policy = cfg.escalation_policy or _default_policy(
        cfg,
        is_quick=run.is_quick,
        t1_enabled=cfg.t1_continuation,
        t1_model=t1_model,
        t1_max_iterations=t1_max_iterations,
    )

    if run.is_quick:
        # Quick mode: no MCP, no agent loop. Single LLM call with the
        # diff + retrieval already pre-packed in the initial prompt.
        from cora.core.quick_review import quick_review_call

        run.tiers_run.append(model)
        run.final_body, run.terminated_reason = await quick_review_call(
            endpoint_base_url=f"{run.base_url.rstrip('/')}/v1",
            llm_gateway_key=run.api_key,
            model_alias=model,
            system_prompt=run.system_prompt,
            initial_user_prompt=run.initial_user_prompt,
            budget=budget,
            timeout_s=run.per_call_timeout_s,
            pr_number=pr_number,
            cfg=cfg,
        )
        return None

    # Deep mode — Pydantic-AI factory + MCP toolsets.
    from cora.core.context_refresher import ContextRefresher
    from cora.core.deep_review import deep_review_call

    # Push-based context refresher. Constructed ONCE per review so
    # dedupe state carries across the T0 → T1 handoff.
    context_refresher = ContextRefresher(
        repo=run.repo,
        pr_number=pr_number,
        head_sha=_prc._pr_head_sha(),
        start_timestamp_iso=run.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        bot_login=cfg.loop_guard_bot_login or "cora[bot]",
        enabled=cfg.context_injection,
        ci_enabled=cfg.context_injection_ci,
        head_enabled=cfg.context_injection_head,
        comments_enabled=cfg.context_injection_comments,
    )

    mcp_token = (cfg.mcp_token or "").strip()
    # No token → no auth headers; a deployment fronting its MCP server
    # with an auth proxy sets `mcp_token` (or terminates auth upstream).
    run.mcp_headers = (
        {"Authorization": f"Bearer {mcp_token}"} if mcp_token else {}
    )
    run.mcp_actions_url = (cfg.mcp_actions_url or "").strip() or None
    mcp_actions_token = (cfg.mcp_actions_token or "").strip()
    run.mcp_actions_headers = (
        {"Authorization": f"Bearer {mcp_actions_token}"}
        if mcp_actions_token
        else None
    )
    run.web_fetch_url = (cfg.web_fetch_gate_url or "").strip() or None
    # Shared T0/T1 dispatch surface — the same allow-set + tool defaults
    # feed `deep_review_call` and the escalation connector's
    # `continue_on_t1` so the two tiers see an identical tool topology.
    allowed_tools = (
        set(cfg.read_tools)
        | set(cfg.action_tools)
        | set(cfg.web_tools)
        | set(cfg.local_repo_tools)
    )
    tool_arg_defaults = {"web_fetch_doc": {"caller": "cora"}}

    # Classifier-large-diff: `cfg.skip_t0` (the workflow's
    # AGENT_REVIEW_SKIP_T0 mode-step output) bypasses T0 and jumps
    # straight to T1 (bigger context); the T1 dispatch below
    # recognises the seeded terminated_reason.
    if cfg.skip_t0:
        _gha_log(
            "deep mode: AGENT_REVIEW_SKIP_T0=true — skipping T0, "
            "starting on T1 (classifier-large-diff)"
        )
        run.final_body = ""
        run.terminated_reason = "classifier_large_start"
        run.tools_available = []
        t0_messages: list = []
    else:
        _gha_log("deep mode: dispatching via deep_review_call")
        run.tiers_run.append(model)
        (
            run.final_body,
            run.terminated_reason,
            run.tools_available,
            t0_messages,
        ) = await deep_review_call(
            endpoint_base_url=f"{run.base_url.rstrip('/')}/v1",
            llm_gateway_key=run.api_key,
            model_alias=model,
            system_prompt=run.system_prompt,
            initial_user_prompt=run.initial_user_prompt,
            budget=budget,
            timeout_s=run.per_call_timeout_s,
            pr_number=pr_number,
            repo=run.repo,
            mcp_url=run.mcp_url,
            mcp_headers=run.mcp_headers,
            mcp_actions_url=run.mcp_actions_url,
            mcp_actions_headers=run.mcp_actions_headers,
            web_fetch_url=run.web_fetch_url,
            web_fetch_headers=None,
            allowed_tools=allowed_tools,
            tool_arg_defaults=tool_arg_defaults,
            max_iterations=run.max_iterations,
            loop_deadline_monotonic=t0_deadline_monotonic,
            context_refresher=context_refresher,
            gha_log=run.iter_log,
            cfg=cfg,
            git_provider=run.git,
        )

    # Tool-use trajectory capture (opt-in; no-op unless
    # `cfg.transcript_dir` is set). Only on clean completion. Never
    # raises: a capture bug must not break a live review.
    if (
        (cfg.transcript_dir or "").strip()
        and run.terminated_reason is None
        and t0_messages
    ):
        try:
            from cora.core.transcript import serialize_trajectory

            _src = (
                (cfg.transcript_source or "").strip()
                or _c.DEFAULT_TRANSCRIPT_SOURCE
            )
            _row = serialize_trajectory(
                t0_messages,
                meta={
                    "source": _src,
                    "pr_number": int(pr_number)
                    if str(pr_number).isdigit()
                    else pr_number,
                    "model_alias": model,
                },
            )
            if _row is not None:
                _d = Path(str(cfg.transcript_dir).strip())
                _d.mkdir(parents=True, exist_ok=True)
                _run = os.environ.get("GITHUB_RUN_ID", "local")
                # Per-review file — avoids interleaving when several
                # PRs review concurrently.
                _out = _d / f"trajectory-pr{pr_number}-{_run}.jsonl"
                _out.write_text(
                    json.dumps(_row, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                _gha_log(
                    f"captured tool-use trajectory → {_out.name} "
                    f"({len(_row['messages'])} turns, "
                    f"{sum(1 for m in _row['messages'] if m.get('tool_calls'))} "
                    f"tool-call turns)"
                )
        except Exception as _exc:  # noqa: BLE001
            _gha_log(f"trajectory capture skipped (non-fatal): {_exc!r}")

    # Triggered T0 → T1 escalation, driven by the escalation policy
    # (the ladder has a T1 rung only when `t1_continuation` is on;
    # `should_escalate` intersects the tripped triggers with the
    # policy's `escalate_on` — wall-hit by default, optionally
    # blocker / low_confidence via `cfg.escalation_triggers`).
    # Two forced entries bypass the policy by design:
    #   - `classifier_large_start` (skip-T0 path) — T1 starts fresh;
    #   - `per_call_timeout` with NO committed messages — T0's first
    #     turn hung before any history existed; T1 runs on a
    #     different endpoint so it likely won't stall the same way.
    skip_t0_start = run.terminated_reason == "classifier_large_start"
    run.per_call_fresh_start = (
        run.terminated_reason == "per_call_timeout" and not t0_messages
    )
    # Parse the T0 verdict up front so the blocker / low_confidence
    # triggers see it; the final parse for the posted comment happens
    # later on whichever tier's body wins.
    from cora.core.leak import parse_verdict_from_body

    _t0_interim = ReviewResult(
        verdict=parse_verdict_from_body(
            run.final_body,
            glyphs=cfg.verdict_glyphs,
            words=cfg.verdict_words,
        ),
        verdict_line=None,
        conclusion="",
        body=run.final_body,
        mode=run.mode,
        budget=budget,
        wall_time_s=time.monotonic() - start,
        terminated_reason=run.terminated_reason,
    )
    triggered_escalation = bool(t0_messages) and policy.should_escalate(
        _t0_interim, 0, blocker_word=cfg.verdict_words[2]
    )
    if triggered_escalation or skip_t0_start or run.per_call_fresh_start:
        # Hand off to the escalation connector. The default deep-mode
        # policy selects the trajectory-resume `KvContinuationConnector`;
        # the per-call dispatch surface rides in `EscalationContext.extra`
        # so the seam itself stays engine-free. `entry`/`tag` carry the
        # entry paths the connector maps onto the finish-line
        # `terminated_reason`.
        if skip_t0_start:
            entry, tag = "fresh", "classifier_large"
        elif run.per_call_fresh_start:
            entry, tag = "fresh", "per_call_fresh"
        elif run.terminated_reason in WALL_HIT_REASONS:
            entry, tag = "wall_hit", "wall_hit"
        else:
            # blocker / low_confidence: T0 completed — resume its
            # trajectory so the stronger tier re-examines the findings.
            entry, tag = "wall_hit", "verdict_trigger"
        t1_tier = policy.next_tier(0) or Tier(
            model=t1_model, max_iterations=t1_max_iterations
        )
        esc_ctx = EscalationContext(
            next_tier=t1_tier,
            prev_context=t0_messages,
            initial_user_prompt=run.initial_user_prompt,
            entry=entry,
            tag=tag,
            terminated_reason=run.terminated_reason,
            extra={
                "endpoint_base_url": f"{run.base_url.rstrip('/')}/v1",
                "llm_gateway_key": run.api_key,
                "system_prompt": run.system_prompt,
                "budget": budget,
                "timeout_s": run.per_call_timeout_s,
                "pr_number": pr_number,
                "repo": run.repo,
                "mcp_url": run.mcp_url,
                "mcp_headers": run.mcp_headers,
                "mcp_actions_url": run.mcp_actions_url,
                "mcp_actions_headers": run.mcp_actions_headers,
                "web_fetch_url": run.web_fetch_url,
                "web_fetch_headers": None,
                "allowed_tools": allowed_tools,
                "tool_arg_defaults": tool_arg_defaults,
                "context_refresher": context_refresher,
                "iter_log": run.iter_log,
                "gha_log": _gha_log,
                "cfg": cfg,
                "git_provider": run.git,
                "now": time.monotonic,
                "start": start,
                "t0_wall_time_s": t0_wall_time_s,
                "t1_wall_time_s": t1_wall_time_s,
            },
        )

        # `tiers_run.append` stays here (before dispatch) so the trail
        # records the attempt even when T1 produces no body.
        run.tiers_run.append(t1_tier.model)
        outcome = await run_escalation(
            policy, esc_ctx, _continuation_tier_runner(esc_ctx.extra)
        )
        if outcome.body:
            # T1 produced a verdict — adopt it. The connector mapped the
            # entry path to the finish-line `terminated_reason`.
            run.final_body = outcome.body
            run.terminated_reason = outcome.terminated_reason
            if outcome.tools:
                run.tools_available = sorted(
                    set(run.tools_available) | set(outcome.tools)
                )
        else:
            # T1 also failed; the connector returned the T0
            # terminated_reason so the original wall-hit isn't masked.
            run.terminated_reason = outcome.terminated_reason

    # Skip-class failures (transient infra) → conclusion=cancelled
    # so the merge gate stays open for a re-push retry.
    if run.terminated_reason == "mcp-connect-failed":
        try:
            run.reporter.post_skip("MCP server unreachable")
        except Exception as post_exc:  # noqa: BLE001
            print(f"::warning::could not post skip comment: {post_exc}")
        run.reporter.complete_check(
            verdict_line="skipped (MCP server unreachable)",
            conclusion="cancelled",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            terminated_reason="mcp-connect-failed",
        )
        return run.skip_result(
            "skipped (MCP server unreachable)",
            "mcp-connect-failed",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            tiers_run=run.tiers_run,
        )
    if run.terminated_reason and run.terminated_reason.startswith(
        "agent-loop-errored"
    ):
        print(f"::warning::agent loop failed: {run.terminated_reason}")
        try:
            run.reporter.post_skip(
                f"Agent loop errored: `{run.terminated_reason}`. "
                "Review will retry on the next push."
            )
        except Exception as post_exc:  # noqa: BLE001
            print(f"::warning::could not post skip comment: {post_exc}")
        run.reporter.complete_check(
            verdict_line="skipped (agent loop errored)",
            conclusion="cancelled",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            terminated_reason="agent-loop-errored",
        )
        return run.skip_result(
            "skipped (agent loop errored)",
            "agent-loop-errored",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            tiers_run=run.tiers_run,
        )
    # `per_call_timeout` + no body: T1 was skipped or also failed.
    # The backend stalled — treat as transient infra, same as
    # mcp-connect-failed; re-push to retry.
    if run.terminated_reason == "per_call_timeout" and not run.final_body:
        print(
            f"::warning::per_call_timeout with no review body "
            f"(T1 {('not attempted' if not run.per_call_fresh_start else 'also failed')})"
            f" — posting cancelled check-run"
        )
        try:
            run.reporter.post_skip(
                "Inference backend stalled (per-call timeout, 0 tool calls). "
                "Review will retry on the next push."
            )
        except Exception as post_exc:  # noqa: BLE001
            print(f"::warning::could not post skip comment: {post_exc}")
        run.reporter.complete_check(
            verdict_line="skipped (inference backend stalled)",
            conclusion="cancelled",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            terminated_reason="per_call_timeout",
        )
        return run.skip_result(
            "skipped (inference backend stalled)",
            "per_call_timeout",
            budget=budget,
            wall_time_s=time.monotonic() - start,
            tiers_run=run.tiers_run,
        )
    # Otherwise (success, or max_iterations / wall_time /
    # budget_exhausted) fall through to the shared finalize path.
    return None


async def second_opinion_dispatch(run: ReviewRun) -> None:
    """Second-opinion dispatch — runs an independent
    review on a different model and folds its verdict in, behind the
    `SecondOpinionProvider` seam. The T2 alt-reviewer +
    disagreement resolution is the default impl; `NullSecondOpinion`
    never dispatches. The generic path here doesn't know what a T2 /
    alt-reviewer is — it just carries the result object forward."""
    if not run.second_opinion.should_dispatch(
        cfg=run.cfg, is_quick=run.is_quick, primary_body=run.final_body
    ):
        return
    run.second_opinion_result = await run.second_opinion.dispatch(
        cfg=run.cfg,
        endpoint_base_url=f"{run.base_url.rstrip('/')}/v1",
        api_key=run.api_key,
        system_prompt=run.system_prompt,
        initial_user_prompt=run.initial_user_prompt,
        budget=run.budget,
        timeout_s=run.per_call_timeout_s,
        pr_number=run.pr_number,
        repo=run.repo,
        mcp_url=run.mcp_url,
        mcp_headers=run.mcp_headers,
        mcp_actions_url=run.mcp_actions_url,
        mcp_actions_headers=run.mcp_actions_headers,
        web_fetch_url=run.web_fetch_url,
        git=run.git,
        log=_gha_log,
        iter_log=run.iter_log,
        primary_terminated_reason=run.terminated_reason,
    )
    if run.second_opinion_result.model_alias:
        run.tiers_run.append(run.second_opinion_result.model_alias)
    if run.second_opinion_result.body and run.second_opinion_result.tools_available:
        run.tools_available = sorted(
            set(run.tools_available) | set(run.second_opinion_result.tools_available)
        )
