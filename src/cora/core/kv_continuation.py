"""KV-cache continuation connector — resume the trajectory on the next tier.

The default deep-mode implementation of the `EscalationConnector` seam
(`cora.escalation`): when a tier hits a wall, the next tier *resumes* the
prior tier's message trajectory as-is instead of re-prefilling a fresh
prompt. On self-hosted backends that share prefix/KV cache across model
endpoints this makes the handoff cheap; against a stateless API it still
works — the next tier simply re-reads the carried-forward history.

It lives under `cora.core` (not `cora.escalation`) because it imports the
engine — `continue_on_t1` (`core/continuation.py`). The escalation module
itself stays dependency-free; `ReprefillConnector` there is the generic
re-prefill default a single- or two-model adopter uses with zero extra
machinery.

`run_review` selects this connector for the default deep-mode policy and
feeds it the per-call dispatch parameters via `EscalationContext.extra`
(`continue_on_t1`'s ~25-arg surface mirrors the T0 dispatch, so threading
the shared bag once is cleaner than a parallel parameter list). The `tag`
on the context distinguishes the three entry paths so the finish-line
`terminated_reason` names how T1 was entered:

  - wall-hit resume         → ``t1-continuation``
  - large-diff direct start → ``t1-classifier-large``
  - per-call fresh start    → ``t1-per-call-retry``
  - verdict-trigger resume  → ``t1-verdict-trigger``
  - exhausted-spiral resume → ``t1-spiral-escalation``
"""

from __future__ import annotations

from cora.escalation import (
    EscalationConnector,
    EscalationContext,
    EscalationOutcome,
    TierRunner,
)

# Maps the entry `tag` to the finish-line `terminated_reason` T1 adopts on a
# successful body.
_T1_SUCCESS_REASON = {
    "classifier_large": "t1-classifier-large",
    "per_call_fresh": "t1-per-call-retry",
    "wall_hit": "t1-continuation",
    "verdict_trigger": "t1-verdict-trigger",
    "spiral_exhausted": "t1-spiral-escalation",
    "no_tool_use": "t1-no-tool-use-retry",
}

# Every `terminated_reason` that means "the posted body came from T1" —
# consumers deciding tier attribution (e.g. the `tier_verdict` event)
# must use this set, not a hand-picked subset that drifts when a new
# entry path is added.
T1_TERMINATED_REASONS = frozenset(_T1_SUCCESS_REASON.values())


class KvContinuationConnector(EscalationConnector):
    """Deep-mode default: continue T0's trajectory on T1.

    `handoff` is the identity message pass-forward (T1 resumes T0's
    trajectory rather than re-prefilling, so the *same* history object is
    threaded as-is into `continue_on_t1`'s `prior_messages`). `escalate`
    owns the entry framing; the `run_tier` callback is ignored — this
    connector dispatches `continue_on_t1` directly so it can thread the
    resume-vs-fresh prompt framing the generic re-prefill path lacks."""

    def handoff(self, prev_context: list) -> list:
        # Identity (not a copy): T1 resumes T0's trajectory, so the same
        # history object is threaded unchanged.
        return prev_context

    async def escalate(
        self, ctx: EscalationContext, run_tier: TierRunner
    ) -> EscalationOutcome:
        # Imported here (module-attribute access) so test monkeypatches on
        # `cora.core.continuation.continue_on_t1` are honoured.
        from cora.core import continuation as _continuation

        x = ctx.extra
        # Two log sinks, kept distinct: the entry / no-body lines go to
        # `gha_log` (stderr only); the `continue_on_t1` internals go to
        # `iter_log` (stderr + the structured log sink).
        gha_log = x["gha_log"]
        iter_log = x["iter_log"]
        tag = ctx.tag or "wall_hit"
        is_fresh = ctx.entry == "fresh"

        # T1 always gets its reserved window counted from NOW; if T0 finished
        # early, T1 absorbs the slack (whichever is later). `now()` returns
        # `time.monotonic()`.
        t1_deadline_monotonic = max(
            x["now"]() + x["t1_wall_time_s"],
            x["start"] + x["t0_wall_time_s"] + x["t1_wall_time_s"],
        )
        t1_budget_s = t1_deadline_monotonic - x["now"]()
        t1_model = ctx.next_tier.model
        t1_max_iterations = ctx.next_tier.max_iterations

        if tag == "classifier_large":
            gha_log(
                f"classifier-large-diff: starting on T1 directly "
                f"on `{t1_model}` (max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f}); T0 skipped"
            )
        elif tag == "per_call_fresh":
            gha_log(
                f"T0 per_call_timeout with no committed messages; "
                f"starting fresh on T1 `{t1_model}` "
                f"(max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f})"
            )
        elif tag == "verdict_trigger":
            gha_log(
                f"T0 verdict tripped an escalation trigger; "
                f"double-checking on T1 `{t1_model}` "
                f"(max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f})"
            )
        elif tag == "spiral_exhausted":
            gha_log(
                f"T0 exhausted its spiral re-draw; resuming the committed "
                f"trajectory on T1 `{t1_model}` "
                f"(max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f})"
            )
        elif tag == "no_tool_use":
            gha_log(
                f"T0 verdicted with zero tool calls (unverified by "
                f"construction); re-running fresh on T1 `{t1_model}` "
                f"(max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f})"
            )
        else:
            gha_log(
                f"T0 wall-hit ({ctx.terminated_reason}); escalating to T1 "
                f"on `{t1_model}` (max_iterations={t1_max_iterations}, "
                f"budget_s={t1_budget_s:.0f})"
            )

        t1_body, t1_terminated, t1_tools = await _continuation.continue_on_t1(
            endpoint_base_url=x["endpoint_base_url"],
            llm_gateway_key=x["llm_gateway_key"],
            t1_model_alias=t1_model,
            system_prompt=x["system_prompt"],
            prior_messages=self.handoff(ctx.prev_context),
            # Fresh-start paths get the initial prompt as the first turn; the
            # wall-hit path passes None and keeps the resume framing.
            initial_user_prompt=ctx.initial_user_prompt if is_fresh else None,
            budget=x["budget"],
            timeout_s=x["timeout_s"],
            pr_number=x["pr_number"],
            repo=x["repo"],
            mcp_url=x["mcp_url"],
            mcp_headers=x["mcp_headers"],
            mcp_actions_url=x["mcp_actions_url"],
            mcp_actions_headers=x["mcp_actions_headers"],
            web_fetch_url=x["web_fetch_url"],
            web_fetch_headers=x["web_fetch_headers"],
            # `.get(..., ())` — defensive default so an `extra` bag built
            # before this field existed (e.g. an older test fixture)
            # still resumes T1 with zero extra sessions instead of a
            # KeyError.
            extra_sessions=x.get("extra_sessions", ()),
            allowed_tools=x["allowed_tools"],
            tool_arg_defaults=x["tool_arg_defaults"],
            max_iterations=t1_max_iterations,
            loop_deadline_monotonic=t1_deadline_monotonic,
            # Same refresher instance T0 used — dedupe state persists across
            # the handoff.
            context_refresher=x["context_refresher"],
            gha_log=iter_log,
            # Verdict-triggered entries resume a *completed* trajectory —
            # frame the handoff as a second look, not a budget top-up.
            # Exhausted-spiral entries resume a *stalled* one — frame it
            # as picking up where the prior tier's reasoning stalled.
            resume_prompt=(
                _continuation.VERDICT_ESCALATION_PROMPT
                if tag == "verdict_trigger"
                else _continuation.SPIRAL_ESCALATION_PROMPT
                if tag == "spiral_exhausted"
                else None
            ),
            cfg=x["cfg"],
            git_provider=x["git_provider"],
        )

        if t1_body:
            # T1 produced a verdict — adopt it. The finish-line
            # `terminated_reason` distinguishes the T1 entry paths.
            return EscalationOutcome(
                body=t1_body,
                terminated_reason=_T1_SUCCESS_REASON[tag],
                tools=t1_tools,
                tier_ran=t1_model,
            )
        # T1 also failed; keep the T0 result + terminated_reason for the
        # finalize path so the original wall-hit isn't masked.
        gha_log(
            f"T1 continuation produced no body "
            f"(reason: {t1_terminated or 'unknown'}); keeping "
            f"T0 terminated_reason={ctx.terminated_reason}"
        )
        return EscalationOutcome(
            body="",
            terminated_reason=ctx.terminated_reason,
            tools=[],
            tier_ran=t1_model,
        )
