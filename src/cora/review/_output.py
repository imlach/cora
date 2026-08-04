"""Output pipeline — reasoning-leak handling and the leak-retry turn.

The verdict must survive contact with self-hosted reasoning models:
`cora.core.leak` does the detection/stripping; this module owns the
orchestrator-side retry (one bounded reformat turn, never a re-review).
"""

from __future__ import annotations

import json
import time

from cora.core.budget import Budget
from cora.core import config as _c
from cora.core.leak import (
    build_leak_retry_messages,
    detect_blocker,
    detect_reasoning_leak,
    parse_verdict_from_body,
    strip_reasoning,
    verdict_to_conclusion,
)
from cora.core.log import _gha_log
from cora.result import ReviewResult
from cora.review._state import ReviewRun


def emit_finish(
    run: ReviewRun,
    *,
    terminated_reason: str | None,
    wall_time_s: float | None = None,
    is_leak: bool = False,
    leak_retry_state: str = "none",
    preamble_chars: int = 0,
) -> None:
    """Write the single `agent_review finish` line for this review.

    One line per review, on whichever path exits first — the normal
    output pipeline, an early skip, a cancellation, or the SIGTERM
    guard. Idempotent via `run.finish_emitted`, so the wrapper can call
    it unconditionally on the way out without doubling up on the
    happy path.

    Every review MUST close its stream: a run that logged `turn 1` and
    then nothing at all is indistinguishable from a run still in flight,
    which makes "how many reviews died and how" unanswerable from the
    logs. The skip/cancel paths carry zeros for the leak/preamble
    fields rather than dropping them — the field set is a parsing
    contract for downstream dashboards and must not vary by exit path.
    """
    if run.finish_emitted:
        return
    run.finish_emitted = True
    budget = run.budget or Budget(max_input=0, max_output=0, max_iterations=0)
    if wall_time_s is None:
        wall_time_s = (time.monotonic() - run.start) if run.start else 0.0
    finish_line = (
        f"agent_review finish pr_number={run.pr_number} mode={run.mode} "
        f"turns={budget.iterations} "
        f"in_tokens={budget.input_used} out_tokens={budget.output_used} "
        f"wall_s={wall_time_s:.1f} terminated={terminated_reason} "
        f"resolved_model={budget.resolved_model or 'unknown'} "
        f"leak={'true' if is_leak else 'false'} "
        f"leak_retry={leak_retry_state} "
        f"preamble_stripped={preamble_chars} "
        f"reasoning_stripped={run.reasoning_stripped_chars} "
        f"tools={dict(budget.tool_calls)}"
    )
    _gha_log(finish_line)
    run.loki(finish_line, labels={"consumer": "pr-review", "kind": run.mode})


async def quick_review_retry_for_format(
    *,
    llm_client,
    model: str,
    leaked_body: str,
    budget: Budget,
    timeout_s: int,
    pr_number: str,
    max_tokens: int = _c.MAX_OUTPUT_TOKENS,
) -> tuple[str, str | None]:
    """One follow-up LLM turn after a leak — the model produced analysis
    but missed the verdict marker. Pure REFORMAT of the already-produced
    body (no tool re-work); fired at most once by the caller. Used by
    BOTH modes — quick leaks regularly on bot-batch runs; deep mode
    usually self-corrects but isn't immune."""
    messages = build_leak_retry_messages(leaked_body)
    try:
        raw = await llm_client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            # Same ceiling as the first call — the model re-emits the
            # same body shape with the verdict line prepended.
            max_tokens=max_tokens,
            # Tighter than the review call's 0.2 — formatting task.
            temperature=0.1,
            timeout=timeout_s,
            extra_body={
                "metadata": {
                    "tags": [
                        f"pr-{pr_number}",
                        "consumer:pr-review",
                        "kind:leak-retry",
                    ],
                },
            },
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"leak-retry LLM call failed: {exc}"
    response = raw.parse()
    usage = getattr(response, "usage", None)
    if usage:
        # Track retry tokens against the same Budget so the comment
        # footer reflects the FULL spend — first call AND retry.
        budget.add_usage(usage)
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return "", "empty retry response"
    return content, None


def _finalize_observability(
    run: ReviewRun,
    *,
    verdict_line: str | None,
    conclusion: str,
    body_for_summary: str,
    leak_flag: bool,
) -> None:
    """Update check run + step summary at the end of any terminal
    path. Soft-fail; the comment post is the primary user-facing
    artifact and happens on its own. `complete_check`'s idempotency
    also disarms the SIGTERM finalizer."""
    run.reporter.complete_check(
        verdict_line=verdict_line,
        conclusion=conclusion,
        budget=run.budget,
        wall_time_s=run.wall_time_s,
        terminated_reason=run.terminated_reason,
    )
    run.reporter.write_summary(
        body=body_for_summary,
        budget=run.budget,
        wall_time_s=run.wall_time_s,
        terminated_reason=run.terminated_reason,
        is_leak=leak_flag,
        tools_available=run.tools_available,
    )


async def produce_output(run: ReviewRun) -> ReviewResult | None:  # noqa: PLR0915
    """The output pipeline: reasoning-leak handling (strip → detect →
    one retry turn), the observability trail (finish line, tool-counter
    trace, leak preview), second-opinion composition, and the verdict
    parse. Returns the no-body / leak terminal results (`failure` —
    the LLM was reachable but nothing postable came out), or None with
    `run.body_to_post` + `run.verdict` set."""
    cfg = run.cfg
    budget = run.budget
    pr_number = run.pr_number
    mode = run.mode
    run.wall_time_s = time.monotonic() - run.start
    wall_time_s = run.wall_time_s
    final_body = run.final_body
    terminated_reason = run.terminated_reason

    if not final_body:
        _finalize_observability(
            run,
            verdict_line="no review produced",
            # Review ran to completion but emitted no body. Distinct from
            # the preflight infra failures — the LLM was reachable, the
            # loop didn't error, but nothing usable came out. `failure`
            # blocks auto-merge instead of treating it as a transient.
            conclusion="failure",
            body_for_summary="",
            leak_flag=False,
        )
        if run.eval_mode:
            # Dump empty body + trace so the harness sees the entry ran
            # end-to-end (with a reason) instead of disappearing.
            run.eval_dump(f"{pr_number}.md", "")
            run.eval_dump(
                f"{pr_number}.trace.json",
                json.dumps(
                    {
                        "pr_number": pr_number,
                        "mode": mode,
                        "retrieval_trace": run.retrieval_trace,
                        "tool_counts": dict(budget.tool_calls),
                        "total_tool_calls": int(sum(budget.tool_calls.values())),
                        "verdict_line": None,
                        "elapsed_s": round(wall_time_s, 3),
                        "terminated_reason": terminated_reason,
                        "no_final_body": True,
                        "propose_patch_directive": None,
                    },
                    indent=2,
                    default=str,
                ),
            )
        else:
            try:
                run.reporter.post_skip(
                    f"Agent produced no final review "
                    f"(reason: {terminated_reason or 'unknown'})."
                )
            except Exception as exc:  # noqa: BLE001
                print(f"::warning::could not post skip comment: {exc}")
        return run.skip_result(
            "no review produced",
            terminated_reason or "unknown",
            conclusion="failure",
            budget=budget,
            wall_time_s=wall_time_s,
            tiers_run=run.tiers_run,
        )

    # Strip inline `<think>` blocks before the leak detector runs —
    # backends without a working reasoning parser leak the trace inline.
    cleaned_body, reasoning_stripped_chars = strip_reasoning(final_body)

    # Leak guard: reject reasoning-only output, strip reasoning preamble
    # when the model produced a valid review after thinking.
    body_to_post, preamble_chars, is_leak = detect_reasoning_leak(cleaned_body)

    # Retry-on-leak — quick AND deep mode. Single bounded reformat call,
    # NOT a re-review: it re-emits the already-produced body leading with
    # the verdict marker.
    leak_retry_state = "not_attempted"
    if is_leak and final_body:
        leak_retry_state = "attempted"
        _gha_log(
            f"agent_review leak-retry start pr_number={pr_number} "
            f"mode={mode} first_out_tokens={budget.output_used}"
        )
        # AsyncOpenAI directly (not the agent factory) — a one-shot
        # reformat call on a leaked body. Lazy import keeps the client
        # construction off the no-leak path.
        from openai import AsyncOpenAI

        llm_client = AsyncOpenAI(
            api_key=run.api_key, base_url=f"{run.base_url.rstrip('/')}/v1"
        )
        retry_body, retry_err = await quick_review_retry_for_format(
            llm_client=llm_client,
            model=run.model,
            leaked_body=final_body,
            budget=budget,
            timeout_s=run.per_call_timeout_s,
            pr_number=pr_number,
            # Match the first call's ceiling: quick's single-shot cap (a
            # reasoning model re-thinks even on a reformat turn), deep's
            # per-call cap otherwise.
            max_tokens=(
                cfg.quick_max_output_tokens
                if mode == "quick"
                else cfg.max_output_tokens
            ),
        )
        if retry_err:
            _gha_log(f"::warning::leak-retry errored: {retry_err}")
            leak_retry_state = "errored"
        elif retry_body:
            # Replay the same post-processing pipeline on the retry
            # output so a rescued body lands with identical shape.
            retry_cleaned, retry_reasoning_stripped = strip_reasoning(retry_body)
            retry_to_post, retry_preamble, retry_is_leak = detect_reasoning_leak(
                retry_cleaned
            )
            if not retry_is_leak:
                _gha_log(f"agent_review leak-retry succeeded pr_number={pr_number}")
                final_body = retry_body
                run.final_body = retry_body
                cleaned_body = retry_cleaned
                reasoning_stripped_chars = retry_reasoning_stripped
                body_to_post = retry_to_post
                preamble_chars = retry_preamble
                is_leak = False
                leak_retry_state = "succeeded"
            else:
                _gha_log(
                    f"agent_review leak-retry also leaked pr_number={pr_number}"
                )
                leak_retry_state = "still_leaked"

    run.reasoning_stripped_chars = reasoning_stripped_chars

    emit_finish(
        run,
        terminated_reason=terminated_reason,
        wall_time_s=wall_time_s,
        is_leak=is_leak,
        leak_retry_state=leak_retry_state,
        preamble_chars=preamble_chars,
    )

    # Context-tool counter trace (`agent-review-tools` consumer).
    tracked_tool_names = (
        "search_knowledge",
        "read_decision",
        "read_note",
        "read_agents_section",
        "grep_repo",
        "git_show",
        "search_cluster_docs",
        "web_fetch_doc",
    )
    tracked_tool_counts = {
        name: int(budget.tool_calls.get(name, 0)) for name in tracked_tool_names
    }
    try:
        tools_payload = {
            "event": "agent_review_tools",
            "pr_number": pr_number,
            "mode": mode,
            "deps_labelled": run.deps_labelled,
            "web_fetch_doc_calls": tracked_tool_counts["web_fetch_doc"],
            "search_cluster_docs_calls": tracked_tool_counts["search_cluster_docs"],
            "tool_counts": tracked_tool_counts,
            "total_tool_calls": int(sum(budget.tool_calls.values())),
            "retrieval_ran": bool(run.retrieval_trace.get("retrieval_ran")),
            "skip_reason": run.retrieval_trace.get("skip_reason"),
            "release_notes_prefetched": run.prefetched_release_notes is not None,
            "release_notes_prefetch_status": run.prefetch_status,
            "release_notes_prefetch_url": run.prefetch_url,
        }
        tools_line = json.dumps(tools_payload, default=str)
        run.loki(tools_line, labels={"consumer": "agent-review-tools", "kind": mode})
        _gha_log(tools_line)
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::agent_review_tools trace push failed: {exc}")

    # Diagnostic-only line: which x-litellm-* headers LiteLLM exposed.
    if budget.litellm_headers:
        diag_line = (
            f"agent_review litellm_headers pr_number={pr_number} "
            + " ".join(f"{k}={v}" for k, v in budget.litellm_headers.items())
        )
        _gha_log(diag_line)
        run.loki(diag_line, labels={"consumer": "pr-review", "kind": mode})

    if is_leak:
        # Body-preview trace — ground truth on the opener phrasings that
        # trip the guard, without posting the leak to the PR.
        try:
            preview = (final_body or "")[:500].replace("\n", " ").strip()
            leak_payload = {
                "event": "agent_review_leak",
                "pr_number": pr_number,
                "mode": mode,
                "out_tokens": int(getattr(budget, "output_used", 0) or 0),
                "wall_s": round(wall_time_s, 1),
                "resolved_model": budget.resolved_model or "unknown",
                "body_first_500": preview,
            }
            run.loki(
                json.dumps(leak_payload, default=str),
                labels={"consumer": "pr-review-leak", "kind": mode},
            )
            _gha_log(
                f"agent_review_leak pr_number={pr_number} mode={mode} "
                f"preview={preview[:120]!r}..."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::leak preview push failed: {exc}")

        _finalize_observability(
            run,
            verdict_line="⚠️ reasoning leak (body suppressed)",
            # Reviewer produced output but it was unusable even after the
            # leak-retry turn. Block auto-merge — re-push to retry.
            conclusion="failure",
            body_for_summary=final_body,
            leak_flag=True,
        )
        if run.eval_mode:
            # Dump the raw body + a leak-tagged trace so the harness can
            # inspect what the model produced.
            run.eval_dump(f"{pr_number}.md", final_body)
            run.eval_dump(
                f"{pr_number}.trace.json",
                json.dumps(
                    {
                        "pr_number": pr_number,
                        "mode": mode,
                        "retrieval_trace": run.retrieval_trace,
                        "tool_counts": dict(budget.tool_calls),
                        "total_tool_calls": int(sum(budget.tool_calls.values())),
                        "verdict_line": None,
                        "elapsed_s": round(wall_time_s, 3),
                        "terminated_reason": terminated_reason,
                        "reasoning_leak": True,
                        "propose_patch_directive": None,
                    },
                    indent=2,
                    default=str,
                ),
            )
        else:
            try:
                run.reporter.post_skip(
                    "Model returned reasoning-only output (no `Verdict:` "
                    "marker found). Posting the raw body would leak "
                    "stream-of-consciousness analysis to the PR — skipping "
                    "instead. Re-push or re-label to retry."
                )
            except Exception as exc:  # noqa: BLE001
                print(f"::warning::could not post leak-skip comment: {exc}")
        return run.skip_result(
            "⚠️ reasoning leak (body suppressed)",
            terminated_reason or "reasoning-leak",
            conclusion="failure",
            budget=budget,
            wall_time_s=wall_time_s,
            tiers_run=run.tiers_run,
        )

    # Fold the second opinion into the post-leak `body_to_post`.
    # Composition happens HERE (post-leak) so the disagreement
    # banner lands at position 0 of the final comment. The seam runs the
    # second body through leak detection, resolves the verdict gap, and
    # rewrites the body; `second_opinion_result` carries the post-leak
    # verdict/body forward for the events + propose-patch escalation reuse.
    body_to_post = run.second_opinion.compose(
        result=run.second_opinion_result,
        cfg=cfg,
        is_quick=run.is_quick,
        primary_body_to_post=body_to_post,
        primary_terminated_reason=terminated_reason,
        pr_number=pr_number,
        log=_gha_log,
        iter_log=run.iter_log,
    )
    run.body_to_post = body_to_post

    run.verdict = parse_verdict_from_body(body_to_post)
    verdict = run.verdict
    _finalize_observability(
        run,
        verdict_line=f"verdict: {verdict}" if verdict else "no verdict parsed",
        conclusion=verdict_to_conclusion(verdict),
        body_for_summary=body_to_post,
        leak_flag=False,
    )

    # Tier-verdict plumbing — emit a `tier_verdict` event per tier that
    # ran. T0 always; T1 when continuation fired; T2 when it produced.
    if not run.is_quick:
        from cora.core.kv_continuation import T1_TERMINATED_REASONS
        from cora.core.loop_logging import log_tier_verdict

        primary_tier = (
            "T1" if terminated_reason in T1_TERMINATED_REASONS else "T0"
        )
        log_tier_verdict(
            pr_number=pr_number,
            tier=primary_tier,
            verdict=verdict,
            has_blocker=detect_blocker(body_to_post),
            body_chars=len(body_to_post or ""),
            log=run.iter_log,
        )

        # Second-opinion events (T2 tier_verdict + disagreement) — emitted
        # after the primary tier_verdict so the stream order is unchanged.
        run.second_opinion.emit_events(
            result=run.second_opinion_result,
            cfg=cfg,
            is_quick=run.is_quick,
            pr_number=pr_number,
            iter_log=run.iter_log,
        )
    return None
