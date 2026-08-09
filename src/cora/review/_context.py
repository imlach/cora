"""Context assembly — everything the model sees in its first turn.

Launches the three independent pre-flight fetches (diff, CI context,
classifier rationale) concurrently plus the fire-and-forget cold-start
pretrigger, then folds the results — capped diff, capped PR body,
CLAUDE.md, retrieval pre-pack (skipped for bot authors and
classifier-trivial PRs), failing-CI detail, dep-bump release notes,
linked-issue context (`core/issue_context.py`) — into the initial user
prompt via `assemble_initial_user_prompt`.

A failed diff fetch is the one hard skip here (`cancelled`, transient —
re-push to retry); retrieval and every enrichment soft-fail to a leaner
prompt instead.
"""

from __future__ import annotations

import asyncio
import json

from cora.core import config as _c
from cora.core import pr_context as _prc
from cora.core import pretrigger as _pt
from cora.core.log import _gha_log
from cora.core.prompt import assemble_initial_user_prompt
from cora.result import ReviewResult
from cora.review._state import ReviewRun


async def assemble_context(run: ReviewRun) -> ReviewResult | None:
    """Populate the run's context fields and the initial user prompt.
    Returns a skip result only for the diff-fetch failure."""
    cfg = run.cfg
    metadata = run.metadata
    pr_number = run.pr_number

    # Launch the three independent pre-flight fetches concurrently; by
    # the time retrieval completes their awaits are instant.
    _head_sha_for_ci = _prc._pr_head_sha()
    _diff_task = asyncio.create_task(
        asyncio.to_thread(_prc.fetch_pr_diff, pr_number)
    )
    _ci_task = asyncio.create_task(
        asyncio.to_thread(_prc.gather_ci_context, run.repo, _head_sha_for_ci, cfg=cfg)
    )
    _classifier_task = asyncio.create_task(
        asyncio.to_thread(_prc.fetch_classifier_rationale, run.repo, pr_number)
    )
    # Recent maintainer comments (cora #37) — a re-review that can't see
    # a rebuttal re-asserts the finding it refutes. Same concurrent
    # pre-flight shape as the classifier lookup; disarmed by config.
    _thread_task = (
        asyncio.create_task(
            asyncio.to_thread(
                _prc.fetch_thread_evidence, run.repo, pr_number, cfg=cfg
            )
        )
        if cfg.thread_evidence
        else None
    )
    # Cold-start pretrigger — warm a scale-from-zero backend now so its
    # restore overlaps retrieval instead of the first call. The warmup
    # alias set is config-supplied (empty default = disarmed).
    _prewarm_task = asyncio.create_task(  # noqa: RUF006 — fire-and-forget
        _pt.fire_pretrigger(
            run.base_url,
            run.api_key,
            run.model,
            pr_number,
            cfg.pretrigger_warmup_models,
        )
    )

    run.bot_author = _prc.is_bot_author(metadata)
    pr_label_set = {
        ((lbl.get("name") or "") if isinstance(lbl, dict) else str(lbl)).lower()
        for lbl in (metadata.get("labels") or [])
    }
    run.deps_labelled = "deps" in pr_label_set

    # In-progress *comment* is deep-mode only: quick finishes in ~20-30s,
    # so a placeholder + seconds-later edit is churn without benefit.
    if not run.is_quick:
        try:
            run.reporter.post_in_progress()
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::initial comment post failed: {exc}")

    try:
        run.diff_raw = await _diff_task
    except Exception as exc:  # noqa: BLE001
        _ci_task.cancel()
        _classifier_task.cancel()
        if _thread_task is not None:
            _thread_task.cancel()
        print(f"::warning::PR diff fetch failed: {exc}")
        # Preflight infra failure (GH diff API unreachable). `cancelled`
        # is tolerated by the merge gate so a transient GH blip doesn't
        # gate auto-merge — re-push to retry.
        run.reporter.complete_check(
            verdict_line="skipped (diff fetch failed)",
            conclusion="cancelled",
            budget=None,
            wall_time_s=0.0,
            terminated_reason="diff-fetch-failed",
        )
        return run.skip_result("skipped (diff fetch failed)", "diff-fetch-failed")

    run.diff_truncated = len(run.diff_raw) > cfg.diff_char_cap
    run.diff_text = (
        run.diff_raw[: cfg.diff_char_cap] if run.diff_truncated else run.diff_raw
    )

    body = metadata.get("body") or ""
    body_truncated = len(body) > cfg.pr_body_char_cap
    if body_truncated:
        metadata["body"] = (
            body[: cfg.pr_body_char_cap]
            + f"\n\n…[truncated at {cfg.pr_body_char_cap} chars]…\n"
        )

    # Bot PRs skip CLAUDE.md + retrieval (description IS the changelog
    # for version bumps; conventions don't help judge a digest change).
    claude_md, claude_md_truncated = ("", False)
    run.retrieved_docs = []
    run.retrieval_source = "none"
    run.retrieval_trace = {"retrieval_ran": False, "skip_reason": "bot_author"}

    # Skip-trivial. The classifier verdict (Renovate / docs-only
    # PR; `CLASSIFIER_LABEL` env via from_env) or any of the PR's own
    # labels in the configured skip set bypasses the retrieval pipeline
    # entirely.
    classifier_label = (cfg.classifier_label or "").strip().lower()
    pr_label_names = {
        ((lbl.get("name") or "") if isinstance(lbl, dict) else str(lbl)).lower()
        for lbl in (metadata.get("labels") or [])
    }
    skip_match = None
    if classifier_label and classifier_label in cfg.retrieval_skip_labels:
        skip_match = f"classifier:{classifier_label}"
    else:
        overlap = pr_label_names & cfg.retrieval_skip_labels
        if overlap:
            skip_match = "label:" + sorted(overlap)[0]

    if not run.bot_author and skip_match is None:
        claude_md, claude_md_truncated = _prc.read_capped(
            _c.REPO_ROOT / "CLAUDE.md", cfg.claude_md_char_cap
        )
        # Pre-pack top-K relevant context through the retrieval seam.
        # Both modes benefit: quick gets its only retrieval here; deep
        # gets it as the initial-prompt bundle so most reviews finish
        # in 1-2 turns.
        try:
            run.retrieved_docs, run.retrieval_trace = run.retrieval.retrieve(
                repo=run.repo,
                pr_number=pr_number,
                head_sha=_prc._pr_head_sha() or "",
                metadata=metadata,
                diff_text=run.diff_text,
            )
            run.retrieval_trace["retrieval_ran"] = True
            run.retrieval_source = (
                f"retrieval ({len(run.retrieved_docs)} docs)"
                if run.retrieved_docs
                else "retrieval (empty)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::retrieval failed, continuing without: {exc}")
            run.retrieval_source = f"fallback (error: {type(exc).__name__})"
            run.retrieval_trace = {
                "retrieval_ran": False,
                "skip_reason": f"error:{type(exc).__name__}",
                "error_detail": str(exc)[:300],
            }
    elif skip_match is not None:
        run.retrieval_source = f"skipped ({skip_match})"
        run.retrieval_trace = {
            "retrieval_ran": False,
            "skip_reason": f"trivial_label_{skip_match}",
        }

    # Retrieval trace — pure JSON line; `event` is the grep handle.
    try:
        run.retrieval_trace["event"] = "agent_review_rag"
        run.retrieval_trace["pr_number"] = pr_number
        run.retrieval_trace["mode"] = run.mode
        retrieval_trace_line = json.dumps(run.retrieval_trace, default=str)
        run.loki(
            retrieval_trace_line,
            labels={"consumer": "agent-review-rag", "kind": run.mode},
        )
        _gha_log(retrieval_trace_line)
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::retrieval trace push failed: {exc}")

    # CI awareness — fold red sibling checks (+ failed-log tails) into
    # the prompt. Best-effort; a failed lookup yields None.
    ci_context = await _ci_task
    if ci_context:
        _gha_log(f"CI context: {len(ci_context)} chars of failing-check detail")

    # Classifier rationale — best-effort. Discarded for bot authors (the
    # classifier's LLM call is skipped for them).
    _classifier_raw = await _classifier_task
    classifier_rationale = _classifier_raw if not run.bot_author else None
    if classifier_rationale:
        _gha_log(f"classifier rationale: {len(classifier_rationale)} chars")

    # Thread evidence soft-fails to None like every other enrichment —
    # a leaner prompt, never a failed review.
    thread_evidence = None
    if _thread_task is not None:
        try:
            thread_evidence = await _thread_task
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::thread-evidence fetch failed: {exc}")
    if thread_evidence:
        _gha_log(f"thread evidence: {len(thread_evidence)} chars")

    # Release-notes pre-fetch for dep-bump PRs (`deps` label): extract the
    # upstream release/compare URL from the body and fetch it server-side
    # through the web-fetch gate. Soft-fail across the board.
    # Endpoint resolution shared by the prefetch below and the initial
    # prompt's "a fetch tool is available" advertisement: explicit
    # `WEB_FETCH_GATE_URL` wins, else the first `MCP_SERVERS` entry
    # declared `name: "web-fetch"`. Config-time, not probe-time — the
    # prefetch needs the URL before any MCP session opens (it makes its
    # own short-lived session), so there's nothing to introspect yet.
    from cora.core.mcp_sessions import resolve_web_fetch_url

    web_fetch_url = resolve_web_fetch_url(cfg)

    if run.deps_labelled:
        from cora.core.prefetch import (
            extract_release_url,
            fetch_release_notes,
            format_release_notes_block,
        )

        candidate_url = extract_release_url(metadata.get("body") or "")
        if web_fetch_url and candidate_url:
            try:
                result = await fetch_release_notes(web_fetch_url, candidate_url)
            except Exception as exc:  # noqa: BLE001
                print(f"::warning::release-notes pre-fetch errored: {exc}")
                result = None
            if result:
                run.prefetched_release_notes = format_release_notes_block(result)
                run.prefetch_status = result.get("status")
                run.prefetch_url = result.get("url")
                _gha_log(
                    f"release-notes prefetch: url={run.prefetch_url} "
                    f"status={run.prefetch_status} "
                    f"chars={len(run.prefetched_release_notes or '')}"
                )
            else:
                _gha_log(
                    f"release-notes prefetch: url={candidate_url} "
                    f"status=transport-error"
                )
        elif candidate_url and not web_fetch_url:
            _gha_log(
                "release-notes prefetch skipped: no web-fetch endpoint "
                "configured (WEB_FETCH_GATE_URL / an MCP_SERVERS "
                "'web-fetch' entry)"
            )
        elif web_fetch_url and not candidate_url:
            _gha_log(
                "release-notes prefetch skipped: no GitHub release URL in PR body"
            )

    # Linked-issue pre-fetch: the PR title/body may reference issue(s)
    # in this repo via a closing keyword ("fixes #12") or a bare
    # mention ("see #34"). Fetch title/state/body/earliest-comments for
    # up to `cfg.issue_prefetch_max_issues` distinct same-repo issues
    # server-side (same precedent as the release-notes pre-fetch above)
    # and inject as one bounded, trust-wrapped block. Bot PRs skip this
    # — same reasoning as CLAUDE.md/retrieval above: a Renovate/
    # Dependabot body doesn't carry human-authored issue links.
    if run.bot_author:
        _gha_log("linked-issue prefetch skipped: bot-authored PR")
    elif not cfg.issue_context_prefetch:
        _gha_log("linked-issue prefetch skipped: AGENT_REVIEW_ISSUE_PREFETCH=false")
    else:
        from cora.core.issue_context import (
            fetch_issue,
            format_issue_context_block,
            parse_linked_issues,
            wrap_issue_context_block,
        )

        linked_numbers = parse_linked_issues(
            metadata.get("title") or "", metadata.get("body") or "",
            repo=run.repo, cfg=cfg,
        )
        if not linked_numbers:
            _gha_log("linked-issue prefetch: no same-repo issue references in title/body")
        else:
            try:
                fetched = await asyncio.gather(
                    *(
                        asyncio.to_thread(fetch_issue, run.repo, n, cfg=cfg)
                        for n in linked_numbers
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"::warning::linked-issue prefetch errored: {exc}")
                fetched = []
            issues = [i for i in fetched if i]
            run.linked_issue_numbers = [i["number"] for i in issues]
            if not issues:
                _gha_log(
                    f"linked-issue prefetch: refs={linked_numbers} — all "
                    "fetches failed (gh error, not found, or no access)"
                )
            else:
                block = format_issue_context_block(issues, cfg=cfg)
                run.linked_issue_context = wrap_issue_context_block(
                    block, repo=run.repo, numbers=run.linked_issue_numbers
                )
                _gha_log(
                    f"linked-issue prefetch: refs={linked_numbers} "
                    f"fetched={run.linked_issue_numbers} "
                    f"chars={len(run.linked_issue_context)}"
                )

    run.initial_user_prompt = assemble_initial_user_prompt(
        metadata,
        run.diff_text,
        run.diff_truncated,
        claude_md,
        claude_md_truncated,
        body_truncated,
        bot_author=run.bot_author,
        retrieved_docs=run.retrieved_docs,
        prefetched_release_notes=run.prefetched_release_notes,
        linked_issue_context=run.linked_issue_context,
        tools_available=not run.is_quick,
        ci_context=ci_context,
        classifier_rationale=classifier_rationale,
        thread_evidence=thread_evidence,
        # Teacher-trajectory mode — opt-in, never the live default.
        broaden_tools=cfg.broaden_tools,
        fetch_tool_configured=bool(web_fetch_url),
    )
    _gha_log(
        f"initial user prompt: {len(run.initial_user_prompt)} chars "
        f"(~{len(run.initial_user_prompt) // 3} tokens)"
    )
    return None
