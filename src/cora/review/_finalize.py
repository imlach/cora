"""Finalize — build the `ReviewResult`, dump eval artifacts, post.

The result is pure outcome (no side-effects) so eval mode returns it
directly after the dumps; a live run posts the comment through the
Reporter and, on a blocker verdict with the automerge label present,
strips the label so GitHub cancels its queued merge (the operator must
re-apply consciously; the friction is the point).
"""

from __future__ import annotations

import json

from cora.core.leak import detect_blocker, verdict_to_conclusion
from cora.result import ReviewResult
from cora.review._state import ReviewRun


def finalize(run: ReviewRun) -> ReviewResult:
    """Assemble the result, write eval dumps or post the comment, and
    handle the automerge pause."""
    cfg = run.cfg
    pr_number = run.pr_number
    verdict = run.verdict
    body_to_post = run.body_to_post

    # Auto-merge pause: blocker verdict while the automerge label is on
    # the PR → strip the label so GitHub cancels its queued merge. The
    # operator must re-apply consciously; the friction is the point.
    blocker = detect_blocker(body_to_post)
    current_labels = {
        (lbl.get("name") or "") for lbl in (run.metadata.get("labels") or [])
    }
    pause_automerge = blocker and cfg.automerge_label in current_labels

    result = ReviewResult(
        verdict=verdict,
        verdict_line=f"verdict: {verdict}" if verdict else "no verdict parsed",
        conclusion=verdict_to_conclusion(verdict),
        body=body_to_post,
        mode=run.mode,
        budget=run.budget,
        wall_time_s=run.wall_time_s,
        terminated_reason=run.terminated_reason,
        tools_available=run.tools_available,
        bot_author=run.bot_author,
        retrieval_source=run.retrieval_source,
        retrieval_trace=run.retrieval_trace,
        pause_automerge=pause_automerge,
        reasoning_stripped_chars=run.reasoning_stripped_chars,
        tiers_run=run.tiers_run,
    )

    if run.eval_mode:
        # Final review path — dump the FOOTER-LESS body (the comment
        # wrapper adds footer markup that's noise for rubric scoring)
        # plus the full trace.
        run.eval_dump(f"{pr_number}.md", body_to_post)
        run.eval_dump(
            f"{pr_number}.trace.json",
            json.dumps(
                {
                    "pr_number": pr_number,
                    "mode": run.mode,
                    "retrieval_trace": run.retrieval_trace,
                    "tool_counts": dict(run.budget.tool_calls),
                    "total_tool_calls": int(sum(run.budget.tool_calls.values())),
                    "verdict_line": (
                        f"verdict: {verdict}" if verdict else "no verdict parsed"
                    ),
                    "verdict": verdict,
                    "elapsed_s": round(run.wall_time_s, 3),
                    "terminated_reason": run.terminated_reason,
                    "resolved_model": run.budget.resolved_model,
                    "retrieval_source": run.retrieval_source,
                    "automerge_paused": pause_automerge,
                    "propose_patch_directive": run.patch_directive,
                    "tools_available": run.tools_available,
                    "input_tokens": run.budget.input_used,
                    "output_tokens": run.budget.output_used,
                },
                indent=2,
                default=str,
            ),
        )
        return result

    try:
        run.reporter.post_review(result)
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::posting review comment failed: {exc}")
        return result

    if pause_automerge and run.reporter.pause_automerge():
        run.loki(
            f"agent_review automerge_paused pr_number={pr_number} "
            f"mode={run.mode} reason=blocker",
            labels={"consumer": "pr-review", "kind": run.mode},
        )

    return result
