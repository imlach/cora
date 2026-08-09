"""propose_patch dispatch — the hybrid patch path.

Soft-fail throughout; never blocks the comment post. Owns the loop
guard (skip when the App's own bot identity authored the head
commit), the trigger-policy comment-only suppression, the minor-verdict
gate, the mandatory T2 escalation on patch verdicts (reusing the
second-opinion run when one fired), the dispatch through
`Reporter.dispatch_patch`, and the outcome footer appended to the
comment body.
"""

from __future__ import annotations

from cora.core import config as _c
from cora.core import pr_context as _prc
from cora.core.leak import detect_reasoning_leak, strip_reasoning
from cora.core.log import _gha_log
from cora.core.propose_patch import (
    PROPOSE_PATCH_REVIEWER_ALLOWED_PREFIXES,
    PROPOSE_PATCH_REVIEWER_DENIED_PREFIXES,
    parse_propose_patch_directive,
    validate_propose_patch,
)
from cora.review._state import ReviewRun


def _is_self_authored(login: str | None, expected: str) -> bool:
    """Match `<slug>[bot]` and `<slug>[bot]@users.noreply.github.com`
    shapes so the guard fires consistently across commit paths."""
    if not login or not expected:
        return False
    normalised = login.strip().lower()
    return normalised == expected or normalised.startswith(expected + "@")


async def dispatch_patches(run: ReviewRun) -> None:
    """Parse + act on the model's propose_patch directive, then append
    the dispatch outcome to `run.body_to_post`. Never returns a skip —
    every failure degrades to a footer line."""
    cfg = run.cfg
    pr_number = run.pr_number
    metadata = run.metadata
    body_to_post = run.body_to_post
    verdict = run.verdict
    mode = run.mode
    budget = run.budget

    # Synchronize loop guard: if the latest commit on the PR's
    # head was made by the App's own bot identity (a prior propose_patch
    # push), skip dispatch this iteration to avoid self-retriggering.
    loop_guard_bot_login = (
        cfg.loop_guard_bot_login or "cora[bot]"
    ).strip().lower()
    latest_author = (
        _prc.latest_commit_author_login(run.repo, run.head_sha)
        if run.head_sha
        else None
    )

    if run.trigger_degraded:
        # Comment-only ceiling (trigger policy): strip any directive the
        # model emitted from the comment body and drop it — the verdict
        # comment + check-run are the only writes a degraded run makes.
        body_to_post, patch_directive = parse_propose_patch_directive(body_to_post)
        if patch_directive is not None:
            _gha_log(
                "propose_patch suppressed — trigger-policy comment-only ceiling"
            )
            patch_directive = None
    elif _is_self_authored(latest_author, loop_guard_bot_login):
        _gha_log(
            f"propose_patch skipped — latest commit on head authored by "
            f"`{latest_author}` (matches loop-guard `{loop_guard_bot_login}`)"
        )
        patch_directive = None  # forced into the no-directive branch below
    else:
        body_to_post, patch_directive = parse_propose_patch_directive(body_to_post)

    patch_outcome_line: str | None = None
    if patch_directive is not None and not cfg.propose_patch_dispatch:
        _gha_log(
            "propose_patch suppressed — REVIEW_PROPOSE_PATCH_DISPATCH is not enabled"
        )
        run.loki(
            f"agent_review propose_patch pr_number={pr_number} "
            f"outcome=suppressed reason=dispatch-disabled",
            labels={"consumer": "pr-review", "kind": mode},
        )
        patch_directive = None
        patch_outcome_line = (
            "⚠️ _Proposed patch ignored: patch dispatch is disabled for this "
            "reviewer. Set `REVIEW_PROPOSE_PATCH_DISPATCH=true` to enable "
            "inline suggestions and draft fix PRs._"
        )

    # Minor-verdict gate: suppress the propose-patch PR for minor
    # findings; inline suggestions still fire (in-diff, low friction).
    suppress_other_file_edits = patch_directive is not None and verdict == "minor"
    if suppress_other_file_edits:
        _gha_log(
            f"propose_patch PR suppressed pr_number={pr_number} "
            f"reason=verdict-is-minor (inline suggestions still dispatched)"
        )
        run.loki(
            f"agent_review propose_patch pr_number={pr_number} "
            f"outcome=suppressed reason=verdict-is-minor",
            labels={"consumer": "pr-review", "kind": mode},
        )

    if patch_directive is not None:
        # Reviewer path policy — any path except
        # `.github/` may be edited; draft PRs land under
        # the App identity for human merge.
        validation_err = validate_propose_patch(
            patch_directive,
            allowed_prefixes=PROPOSE_PATCH_REVIEWER_ALLOWED_PREFIXES,
            denied_prefixes=PROPOSE_PATCH_REVIEWER_DENIED_PREFIXES,
        )
        if validation_err:
            patch_outcome_line = f"⚠️ _Proposed patch rejected: {validation_err}._"
            _gha_log(
                f"propose_patch rejected pr_number={pr_number} "
                f"reason={validation_err}"
            )
            run.loki(
                f"agent_review propose_patch pr_number={pr_number} "
                f"outcome=rejected reason={validation_err}",
                labels={"consumer": "pr-review", "kind": mode},
            )
        elif run.eval_mode:
            # Directive already parsed + validated; the trace records what
            # would have happened. No dispatch against historical PRs.
            _gha_log(
                "eval mode: skipped propose_patch dispatch "
                "(directive parsed + validated only)"
            )
        else:
            base_ref = metadata.get("baseRefName", "main")
            dispatch_head = run.head_sha or _prc._pr_head_sha() or ""

            # Mandatory T2 escalation on propose_patch verdicts (writes
            # demand a higher bar than comment-only). Kill switch:
            # `cfg.patch_escalation` (AGENT_REVIEW_PATCH_ESCALATION).
            # Reuses the review run's T2 result when one exists.
            from cora.core.patch_escalation import (
                ESCALATION_ENV_VAR,
                build_patch_verification_prompt,
                classify_t2_verdict,
                compose_escalation_outcome,
                escalation_enabled,
                log_escalation_outcome,
                patch_kind_from_dispatch_outcome,
                pr_number_from_url,
            )

            escalation_warning_body: str | None = None
            escalation_warning_summary: str | None = None
            escalation_outcome_obj = None
            so_result = run.second_opinion_result
            # Patch-verification reuses the second-opinion run when one
            # fired (same alias), else fires a dedicated verifier on the
            # configured alt-reviewer alias. Outside the seam, but reads
            # the seam's result to avoid a redundant T2 call.
            t2_model_alias = so_result.model_alias or (
                (cfg.t2_model or "").strip() or _c.DEFAULT_T2_MODEL
            )
            if not escalation_enabled(cfg.patch_escalation):
                _gha_log(
                    f"propose_patch escalation bypassed pr_number={pr_number} "
                    f"reason={ESCALATION_ENV_VAR}=false"
                )
            elif run.is_quick:
                # Quick mode doesn't initialise the deep-mode MCP vars the
                # T2 verifier needs; accept the gap (rare path), logged.
                _gha_log(
                    f"propose_patch escalation bypassed pr_number={pr_number} "
                    f"reason=quick_mode (T2 only fires from deep mode)"
                )
            elif (
                so_result.body is not None
                or so_result.terminated_reason is not None
            ):
                # T2 already ran in the disagreement opt-in path — reuse.
                t2_v, t2_reason = classify_t2_verdict(
                    t2_body=so_result.body,
                    t2_terminated_reason=so_result.terminated_reason,
                )
                escalation_outcome_obj = compose_escalation_outcome(
                    t2_verdict=t2_v,
                    t2_reason=t2_reason,
                    t2_body=so_result.body,
                    t2_model_alias=t2_model_alias,
                )
                escalation_warning_body = escalation_outcome_obj.warning_body
                escalation_warning_summary = escalation_outcome_obj.warning_summary
                _gha_log(
                    f"propose_patch escalation reused-prior-T2 "
                    f"pr_number={pr_number} verdict={t2_v} "
                    f"flagged_for_human={escalation_outcome_obj.flagged_for_human}"
                )
            else:
                # No prior T2 — fire a dedicated patch-verification call.
                # Soft-fail to "skipped" so the patch still applies.
                t2_max_iters_esc = cfg.t2_max_iterations
                verifier_prompt = build_patch_verification_prompt(
                    base_initial_user_prompt=run.initial_user_prompt,
                    t0_verdict_body=body_to_post,
                )
                from cora.core.t2_dispatch import call_t2_alt_reviewer

                try:
                    (
                        esc_t2_body,
                        esc_t2_term,
                        _esc_t2_tools,
                    ) = await call_t2_alt_reviewer(
                        endpoint_base_url=f"{run.base_url.rstrip('/')}/v1",
                        llm_gateway_key=run.api_key,
                        t2_model_alias=t2_model_alias,
                        system_prompt=run.system_prompt,
                        initial_user_prompt=verifier_prompt,
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
                        extra_sessions=cfg.mcp_servers,
                        allowed_tools=set(cfg.read_tools)
                        | set(cfg.action_tools)
                        | set(cfg.web_tools)
                        | set(cfg.local_repo_tools)
                        | set(cfg.extra_tools),
                        tool_arg_defaults={
                            "web_fetch_doc": {"caller": "cora"},
                        },
                        max_iterations=t2_max_iters_esc,
                        gha_log=run.iter_log,
                        cfg=cfg,
                        git_provider=run.git,
                    )
                except Exception as exc:  # noqa: BLE001 — soft-fail
                    esc_t2_body, esc_t2_term, _esc_t2_tools = (
                        None,
                        f"agent-loop-errored: {type(exc).__name__}",
                        [],
                    )
                # Leak detection on the T2 body — a leaked body collapses
                # to skipped, not implicit-disagree.
                if esc_t2_body:
                    _t2c, _ = strip_reasoning(esc_t2_body)
                    _t2b, _, _t2leak = detect_reasoning_leak(_t2c)
                    if _t2leak:
                        esc_t2_body = None
                        if not esc_t2_term:
                            esc_t2_term = "reasoning_leak"
                    else:
                        esc_t2_body = _t2b
                t2_v, t2_reason = classify_t2_verdict(
                    t2_body=esc_t2_body, t2_terminated_reason=esc_t2_term
                )
                escalation_outcome_obj = compose_escalation_outcome(
                    t2_verdict=t2_v,
                    t2_reason=t2_reason,
                    t2_body=esc_t2_body,
                    t2_model_alias=t2_model_alias,
                )
                escalation_warning_body = escalation_outcome_obj.warning_body
                escalation_warning_summary = escalation_outcome_obj.warning_summary
                _gha_log(
                    f"propose_patch escalation t2-fired pr_number={pr_number} "
                    f"verdict={t2_v} reason={t2_reason or 'none'} "
                    f"flagged_for_human={escalation_outcome_obj.flagged_for_human}"
                )

            try:
                dispatch_outcome = run.reporter.dispatch_patch(
                    directive=patch_directive,
                    diff_text=run.diff_raw,
                    base_ref=base_ref,
                    head_sha=dispatch_head,
                    head_ref=metadata.get("headRefName") or None,
                    is_bot_author_pr=_prc.is_bot_author(metadata),
                    is_fork_pr=_prc.is_fork_pr(metadata),
                    escalation_warning_body=escalation_warning_body,
                    escalation_warning_summary=escalation_warning_summary,
                    suppress_other_file_edits=suppress_other_file_edits,
                )
            except Exception as exc:  # noqa: BLE001 — soft-fail
                dispatch_outcome = {
                    "inline_url": None,
                    "inline_count": 0,
                    "draft_url": None,
                    "draft_count": 0,
                    "source_branch_commit_sha": None,
                    "source_branch_count": 0,
                    "rejected": [],
                    "error": f"unexpected error: {type(exc).__name__}: {exc}",
                }

            # Apply the escalation label to whichever PR(s) the
            # dispatcher actually wrote to; fall back to the source PR.
            if escalation_outcome_obj is not None and escalation_outcome_obj.label:
                label_targets: list[str] = []
                draft_pr_num = pr_number_from_url(dispatch_outcome.get("draft_url"))
                if draft_pr_num:
                    label_targets.append(draft_pr_num)
                if dispatch_outcome.get("inline_count") or dispatch_outcome.get(
                    "source_branch_count"
                ):
                    label_targets.append(pr_number)
                if not label_targets:
                    label_targets.append(pr_number)
                for target in label_targets:
                    ok, err = run.reporter.apply_label(
                        escalation_outcome_obj.label, pr_number=target
                    )
                    if not ok:
                        _gha_log(
                            f"::warning::propose_patch escalation label apply "
                            f"failed pr={target} label="
                            f"{escalation_outcome_obj.label}: {err}"
                        )

            # Structured log line — dashboards parse on this.
            if escalation_outcome_obj is not None:
                log_escalation_outcome(
                    pr_number=pr_number,
                    t2_verdict=escalation_outcome_obj.verdict,
                    patch_kind=patch_kind_from_dispatch_outcome(dispatch_outcome),
                    flagged_for_human=escalation_outcome_obj.flagged_for_human,
                    log=run.iter_log,
                )
                run.loki(
                    f"agent_review escalation_outcome pr_number={pr_number} "
                    f"t2_verdict={escalation_outcome_obj.verdict} "
                    f"patch_kind={patch_kind_from_dispatch_outcome(dispatch_outcome)} "
                    f"flagged_for_human={str(escalation_outcome_obj.flagged_for_human).lower()}",
                    labels={"consumer": "pr-review", "kind": mode},
                )

            # Render the dispatch outcome into the footer fragment.
            footer_parts: list[str] = []
            if dispatch_outcome["inline_count"]:
                footer_parts.append(
                    f"💡 **{dispatch_outcome['inline_count']} inline "
                    f"suggestion(s):** see the [Files changed]"
                    f"({dispatch_outcome['inline_url']}) tab — click "
                    f"**Apply suggestion** on each."
                )
            if dispatch_outcome["draft_count"]:
                footer_parts.append(
                    f"📦 **{dispatch_outcome['draft_count']} out-of-hunk "
                    f"edit(s)** proposed in [PR]({dispatch_outcome['draft_url']}) "
                    f"targeting this branch — merge to apply."
                )
            for rejection in dispatch_outcome["rejected"]:
                footer_parts.append(f"⚠️ _Edit rejected: {rejection}._")
            if dispatch_outcome["error"]:
                footer_parts.append(
                    f"⚠️ _Dispatch error: {dispatch_outcome['error']}._"
                )
            if (
                escalation_outcome_obj is not None
                and escalation_outcome_obj.verdict_footer
            ):
                footer_parts.append(escalation_outcome_obj.verdict_footer)
            if footer_parts:
                patch_outcome_line = "\n\n".join(footer_parts)

            _gha_log(
                f"propose_patch dispatch pr_number={pr_number} "
                f"inline={dispatch_outcome['inline_count']} "
                f"patch_pr={dispatch_outcome['draft_count']} "
                f"rejected={len(dispatch_outcome['rejected'])} "
                f"error={dispatch_outcome['error'] or 'none'}"
            )
            run.loki(
                f"agent_review propose_patch pr_number={pr_number} "
                f"outcome=dispatch "
                f"inline={dispatch_outcome['inline_count']} "
                f"patch_pr={dispatch_outcome['draft_count']} "
                f"rejected={len(dispatch_outcome['rejected'])}",
                labels={"consumer": "pr-review", "kind": mode},
            )

    if patch_outcome_line:
        body_to_post = body_to_post.rstrip() + "\n\n---\n\n" + patch_outcome_line

    run.body_to_post = body_to_post
    run.patch_directive = patch_directive
