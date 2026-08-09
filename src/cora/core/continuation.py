"""T1 wall-hit continuation — escalate a cut-off T0 deep review to a
bigger-context T1 endpoint with the full working state
carried across.

When `deep_review_call` returns with
`terminated_reason ∈ {max_iterations, wall_time, budget_exhausted}`,
the model was actively working and got cut off mid-thought. Re-running
from scratch would waste the trajectory and re-derive the same partial
conclusions before hitting the same wall.

`continue_on_t1` hands the captured message history forward to a fresh
Agent run pointed at the T1 model alias. The framework's
`message_history=` parameter lets us resume rather than restart — the
model sees the prior conversation (system prompt, user prompt, every
tool call + result, the partial findings) and only needs to finish what
it started.

Same MCP toolsets + local tools as the T0 run so the tool surface is
identical. Different model alias (`main` vs `review`); the
gateway routes the T1 alias to a backend with roughly 2× the
KV-cache headroom of the T0 backend.

Opt-in via `AGENT_REVIEW_T1_CONTINUATION=true` while the path proves
itself; the default off keeps existing review behaviour unchanged.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.core.mcp_sessions import McpServerSpec
    from cora.providers.git import GitProvider

from cora.core import config as _c
from cora.core.budget import resolve_run_usage, usage_tokens
from cora.core.deep_review import (
    _WALL_TIME_WAIT_FOR_GRACE_S,
    _loaded_tool_names,
    _make_pydantic_ai_local_tools,
    _make_verdict_probe,
)
from cora.core.mcp_probe import probe_mcp_server as _probe_mcp_server

# Continuation directive prepended as the new user turn after the
# T0 messages. Frames the resumption explicitly so the model
# understands the prior turns are its own working state, not a
# user-authored conversation it needs to summarize back.
_CONTINUATION_PROMPT = (
    "You hit the iteration cap before producing a terminal verdict. "
    "The conversation above is your own working state — tool results, "
    "partial findings, the diff you've been analysing. **Resume from "
    "where you left off.** Do not re-summarize the prior turns or "
    "restart the analysis. Use the additional context budget on this "
    "endpoint to deepen the partial findings into a complete review, "
    "then emit the standard `Verdict:` line and review body."
)

# Resume framing for a verdict-triggered escalation (`blocker` /
# `low_confidence` in `cfg.escalation_triggers`): T0 completed — the next
# tier is a second look, not a budget continuation.
VERDICT_ESCALATION_PROMPT = (
    "A stronger reviewer tier is taking over to double-check this review. "
    "The conversation above is the prior tier's working state — tool "
    "results, findings, and its verdict (if any). Re-examine the diff and "
    "each flagged finding, verifying or refuting it with the available "
    "tools, then emit the standard `Verdict:` line and review body "
    "reflecting your own judgement."
)

# Resume framing for an exhausted-spiral escalation: T0's reasoning
# stalled twice on the same draw, so the next tier picks up the committed
# trajectory (the caller drops the spiralled draw itself before the
# handoff) and finishes the review on a different endpoint.
SPIRAL_ESCALATION_PROMPT = (
    "A stronger reviewer tier is taking over: the prior tier's reasoning "
    "stalled before it could produce a verdict. The conversation above is "
    "its working state — tool results and partial findings that remain "
    "valid. Resume from where it left off, complete the analysis, then "
    "emit the standard `Verdict:` line and review body."
)

_UNPROCESSED_TOOL_STUB = (
    "[no result — the prior tier hit its time/iteration budget before "
    "this tool call returned. Treat it as unavailable and continue "
    "without it; re-issue the call only if the finding depends on it.]"
)


def _part_kind(part) -> str:
    """The part's `part_kind` discriminator literal, with a class-name
    fallback for stand-ins. Mirrors `transcript.py` — lets us inspect
    message parts without a hard `pydantic_ai` import at module scope."""
    kind = getattr(part, "part_kind", None)
    if kind:
        return str(kind)
    return {
        "ToolCallPart": "tool-call",
        "ToolReturnPart": "tool-return",
    }.get(type(part).__name__, type(part).__name__)


def _reconcile_unprocessed_tool_calls(
    prior_messages: list,
    *,
    log: Callable[[str], None] = print,
) -> list:
    """Make a carried-forward T0 history safe to resume on T1.

    T0 can wall-hit (or per-call-timeout) right after a `ModelResponse`
    that requested tool calls but *before* the matching tool results
    were appended as the following `ModelRequest`. Pydantic-ai rejects a
    new user prompt on top of such a history with
    `UserError: Cannot provide a new user prompt when the message
    history contains unprocessed tool calls.` — which silently kills the
    T1 escalation safety net on exactly the slow, exploration-heavy PRs
    that need it (observed in the wild).

    For every `ToolCallPart` with no matching `ToolReturnPart` anywhere
    in the history, append a synthetic return so the history is
    well-formed before the continuation prompt is injected. Preserves
    the full trajectory (including any partial-findings text in the
    trailing response) — we only fill the dangling results.

    Pure / non-mutating: returns the input unchanged when there's
    nothing to reconcile, otherwise a new list with one extra
    `ModelRequest` carrying the stub returns appended.
    """
    if not prior_messages:
        return prior_messages

    # Every tool_call_id that already has a return is "processed".
    returned_ids: set[str] = set()
    for msg in prior_messages:
        for part in getattr(msg, "parts", []) or []:
            if _part_kind(part) == "tool-return":
                tcid = getattr(part, "tool_call_id", None)
                if tcid:
                    returned_ids.add(tcid)

    dangling = [
        part
        for msg in prior_messages
        for part in (getattr(msg, "parts", []) or [])
        if _part_kind(part) == "tool-call"
        and getattr(part, "tool_call_id", None) not in returned_ids
    ]
    if not dangling:
        return prior_messages

    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    stub_returns = [
        ToolReturnPart(
            tool_name=getattr(p, "tool_name", "") or "",
            content=_UNPROCESSED_TOOL_STUB,
            tool_call_id=getattr(p, "tool_call_id", "") or "",
            outcome="failed",
        )
        for p in dangling
    ]
    log(
        f"T1 continuation: reconciled {len(stub_returns)} unprocessed "
        f"tool call(s) with synthetic returns before resume "
        f"(tools: {sorted({getattr(p, 'tool_name', '?') for p in dangling})})"
    )
    return [*prior_messages, ModelRequest(parts=stub_returns)]


async def continue_on_t1(
    *,
    endpoint_base_url: str,
    llm_gateway_key: str,
    t1_model_alias: str,
    system_prompt: str,
    prior_messages: list,
    # When `prior_messages` is empty, the call is a "start fresh on T1"
    # (classifier-large-diff entry) rather than a wall-hit resumption
    # — there's nothing to resume from, so the agent gets the full
    # initial user prompt instead of the `_CONTINUATION_PROMPT` lead-in.
    # Ignored when `prior_messages` is non-empty (normal T1 wall-hit
    # path).
    initial_user_prompt: str | None = None,
    budget,  # Budget — loose typing avoids cross-module dep
    timeout_s: int,
    pr_number: str,
    repo: str,
    # MCP topology — same shape as deep_review_call.
    mcp_url: str,
    mcp_headers: dict[str, str],
    mcp_actions_url: str | None = None,
    mcp_actions_headers: dict[str, str] | None = None,
    web_fetch_url: str | None = None,
    web_fetch_headers: dict[str, str] | None = None,
    # Generic extra MCP sessions (from `MCP_SERVERS`) — same shape as
    # `deep_review_call`; see `cora.core.mcp_sessions`.
    extra_sessions: Sequence[McpServerSpec] = (),
    allowed_tools: set[str],
    tool_arg_defaults: dict[str, dict[str, Any]] | None = None,
    # T1 gets a tighter iteration cap than T0 because the trajectory
    # is already half-spent — we're deepening, not re-exploring.
    max_iterations: int = 6,
    # Same shape as deep_review_call — shared deadline; None disables.
    # The caller (agent_review.amain) computes ONE deadline at start
    # and passes the same value to both T0 and T1, so any time T0
    # consumed before tripping wall-hit is automatically subtracted
    # from T1's remaining budget.
    loop_deadline_monotonic: float | None = None,
    # Same shape as deep_review_call — the SAME ContextRefresher
    # instance T0 used. Dedupe state on it (last_check_hash,
    # last_head_sha, last_seen_comment_id, extensions_consumed)
    # persists across the handoff so T1 doesn't re-inject what T0
    # already saw and the +90s/injection cap is shared across the
    # whole review.
    context_refresher=None,
    gha_log: Callable[[str], None] = print,
    # Overrides the default wall-hit resume framing when T1 resumes a
    # non-empty trajectory (e.g. `VERDICT_ESCALATION_PROMPT` for a
    # verdict-triggered escalation). None keeps the wall-hit lead-in.
    resume_prompt: str | None = None,
    # The run's ReviewerConfig — threaded into Deps so T1's tools/hooks
    # read the same config object T0 used. None keeps the legacy
    # behaviour (Deps default-constructs a mirror config).
    cfg: ReviewerConfig | None = None,
    # Repo-introspection backend for the local grep_repo / git_show
    # tools — the SAME provider T0 used. None → LocalGitProvider.
    git_provider: GitProvider | None = None,
) -> tuple[str, str | None, list[str]]:
    """Resume a wall-hit T0 run on the T1 endpoint with the prior
    message history carried forward.

    Returns `(final_body, terminated_reason, tools_available)` — same
    leading-three shape as `deep_review_call`'s tuple so the caller can
    swap the return into the same finalize / leak / comment-post path
    without branching.

    The MCP topology is rebuilt fresh (the T0 toolsets closed when
    that agent's context exited), so this re-probes optional servers
    and re-constructs the Agent. A required-MCP-server failure here
    is unlikely if T0 succeeded, but still returns the distinct
    `mcp-connect-failed` reason for finalize-path consistency.
    """
    from pydantic_ai import ModelSettings, UsageLimits
    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    from cora.core.agent import (
        AgentConfig,
        Deps,
        first_successful_tool_name,
        initial_tool_contract_satisfied,
        make_review_agent,
    )
    from cora.core.loop_logging import (
        PerCallTimeoutExceeded,
        ReasoningSpiralDetected,
        StreamStallDetected,
        WallTimeExceeded,
        iter_with_turn_logging,
        log_continuation_start,
        log_spiral_redraw,
        log_wall_hit,
    )

    # Same optionality as T0: an empty `mcp_url` self-disarms, a
    # configured one is re-probed. T0 already passed this check, so a
    # failure here usually means a transient network blip mid-run.
    mcp_url = (mcp_url or "").strip()

    from cora.core.mcp_sessions import compose_mcp_sessions, open_mcp_sessions

    configured_sessions = compose_mcp_sessions(
        mcp_url=mcp_url,
        mcp_headers=mcp_headers,
        mcp_actions_url=mcp_actions_url,
        mcp_actions_headers=mcp_actions_headers,
        web_fetch_url=web_fetch_url,
        web_fetch_headers=web_fetch_headers,
        extra_sessions=extra_sessions,
        log=gha_log,
    )
    opened = await open_mcp_sessions(
        configured_sessions, probe=_probe_mcp_server, log=gha_log, label_suffix=" (T1)"
    )
    if opened is None:
        return "", "mcp-connect-failed", []
    mcp_servers, sessions_opened = opened
    read_enabled = "mcp" in sessions_opened
    actions_enabled = "actions" in sessions_opened
    web_enabled = "web-fetch" in sessions_opened
    extra_enabled = bool(set(sessions_opened) - {"mcp", "actions", "web-fetch"})

    tools_available = _loaded_tool_names(
        allowed_tools,
        read_enabled=read_enabled,
        actions_enabled=actions_enabled,
        web_enabled=web_enabled,
        extra_enabled=extra_enabled,
        read_tools=cfg.read_tools if cfg is not None else None,
        local_repo_tools=cfg.local_repo_tools if cfg is not None else None,
        action_tools=cfg.action_tools if cfg is not None else None,
        web_tools=cfg.web_tools if cfg is not None else None,
        extra_tools=cfg.extra_tools if cfg is not None else None,
    )

    mcp_allowed_for_filter = allowed_tools - {"grep_repo", "git_show"}

    contract_armed = bool(
        cfg is not None
        and cfg.require_initial_tool_call
        and first_successful_tool_name(prior_messages) is None
    )
    config = AgentConfig(
        endpoint_base_url=endpoint_base_url,
        api_key=llm_gateway_key,
        model_alias=t1_model_alias,
        system_prompt=system_prompt,
        mcp_servers=mcp_servers,
        mcp_allowed_tools=mcp_allowed_for_filter,
        local_tools=_make_pydantic_ai_local_tools(
            tool_arg_defaults, git_provider=git_provider
        ),
        retries=1,
        session_id=cfg.session_header if cfg is not None else None,
        require_initial_tool_call=contract_armed,
    )
    agent = make_review_agent(config)

    deps_kwargs = {"cfg": cfg} if cfg is not None else {}
    deps = Deps(
        repo=repo,
        pr_number=pr_number,
        mode="deep",
        gha_log=gha_log,
        **deps_kwargs,
    )

    # Quick stats on the T0 carry-over so operators can see how
    # much trajectory we're handing forward. Useful for tuning
    # T1's `max_iterations` cap and spotting cases where the T0
    # messages are unexpectedly small/large.
    t0_text_chars = 0
    t0_tool_call_count = 0
    try:
        # `prior_messages` is a list of ModelMessage (Request /
        # Response) — count the response-side TextPart + ToolCallPart
        # weight as a rough handoff size.
        from pydantic_ai.messages import TextPart, ToolCallPart
        for m in prior_messages:
            for part in getattr(m, "parts", []) or []:
                if isinstance(part, TextPart):
                    t0_text_chars += len(getattr(part, "content", "") or "")
                elif isinstance(part, ToolCallPart):
                    t0_tool_call_count += 1
    except Exception:  # noqa: BLE001
        pass

    log_continuation_start(
        pr_number=pr_number,
        t1_model_alias=t1_model_alias,
        prior_messages=len(prior_messages),
        prior_text_chars=t0_text_chars,
        prior_tool_calls=t0_tool_call_count,
        request_limit=max_iterations,
        log=gha_log,
    )

    # Same structured per-turn logging as T0 but under the `T1`
    # phase prefix so dashboards can split escalated work from
    # original-tier work.
    tool_call_counter: dict[str, int] = {}
    turn_counter: list[int] = [0]

    # Hoisted to outer scope so the inner wall-hit / iteration-cap
    # handlers can record the legitimate reason BEFORE the framework's
    # cleanup of agent.iter() + agent context runs — see the parallel
    # comment in deep_review.py for the background.
    early_terminated_reason: str | None = None

    # Uncommitted-draw re-draw — same shape as T0 (deep_review.py).
    # Supplying `verdict_probe` is what arms detection.
    redraw_on = cfg is None or cfg.spiral_redraw
    verdict_probe = _make_verdict_probe(cfg) if redraw_on else None
    redraw_from_turn: int | None = None
    redraw_prior_text: str = ""
    stream_on = cfg is not None and cfg.stream_detection
    result = None
    messages: list = []
    if contract_armed and not tools_available:
        gha_log(
            f"agent_review iter pr_number={pr_number} phase=T1 "
            "event=initial_tool_contract outcome=no_tools first_tool=none"
        )
        return "", "required-tool-unavailable", tools_available

    # Two entry shapes:
    #   - resume: prior_messages non-empty → _CONTINUATION_PROMPT framing
    #   - fresh-start: prior_messages empty + initial_user_prompt given
    #     → use the initial prompt verbatim, no message_history
    # If neither holds we fall back to _CONTINUATION_PROMPT against
    # an empty history (the legacy soft-fail; effectively a smoke check).
    if not prior_messages and initial_user_prompt is not None:
        leadin_prompt = initial_user_prompt
        message_history_arg: list | None = None
    else:
        leadin_prompt = resume_prompt or _CONTINUATION_PROMPT
        # Reconcile any tool call T0 left dangling — a wall-hit between
        # the `model_response` requesting tools and the tool-return turn
        # leaves the history ending on unprocessed tool calls, which
        # pydantic-ai refuses to extend with a new user prompt. Stub the
        # missing returns so the resume framing injects cleanly.
        message_history_arg = _reconcile_unprocessed_tool_calls(
            prior_messages, log=gha_log
        )

    try:
        async with agent:
            async with agent.iter(
                leadin_prompt,
                message_history=message_history_arg,
                deps=deps,
                model_settings=ModelSettings(
                    # Same per-call cap as T0 (deep_review.py): fits the
                    # reasoning trace plus the turn's output, and stays
                    # reachable inside `per_call_timeout_s`.
                    # See `_c.DEEP_MAX_OUTPUT_TOKENS`.
                    max_tokens=(
                        cfg.deep_max_output_tokens
                        if cfg is not None
                        else _c.DEEP_MAX_OUTPUT_TOKENS
                    ),
                    temperature=0.2,
                    timeout=timeout_s,
                ),
                usage_limits=UsageLimits(request_limit=max_iterations),
            ) as agent_run:
                if loop_deadline_monotonic is not None:
                    remaining = loop_deadline_monotonic - time.monotonic()
                    # Pre-pad for push-injection extensions — see the
                    # matching block in deep_review.py for the rationale.
                    from cora.core.context_refresher import (
                        INJECTION_DEADLINE_EXTENSION_S,
                        MAX_INJECTION_EXTENSIONS,
                    )
                    injection_headroom = (
                        INJECTION_DEADLINE_EXTENSION_S * MAX_INJECTION_EXTENSIONS
                        if context_refresher is not None
                        else 0.0
                    )
                    wait_for_timeout = max(
                        1.0,
                        remaining + _WALL_TIME_WAIT_FOR_GRACE_S + injection_headroom,
                    )
                else:
                    wait_for_timeout = None

                # No-op extend callback (presence signals to the iter
                # helper that extensions are enabled; the outer
                # wait_for was pre-padded above and there's no
                # separate caller-side deadline state to mutate here).
                def _extend_t1_deadline(_seconds: float) -> None:
                    return None

                try:
                    inner_coro = iter_with_turn_logging(
                        agent_run,
                        phase="T1",
                        pr_number=pr_number,
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                        # Same hook as T0 — `budget.iterations` is the
                        # combined T0+T1 dispatch count after this run.
                        # T1's per-run cap is `UsageLimits(request_limit)`
                        # above, not `budget.max_iterations` (which
                        # stays at T0's construction value).
                        on_tool_call=budget.add_tool_call,
                        loop_deadline_monotonic=loop_deadline_monotonic,
                        # Per-turn hard cap — same reasoning as T0
                        # (`ModelSettings(timeout=…)` is httpx
                        # read_timeout, useless against steady-stream
                        # slow generation).
                        per_call_timeout_s=float(timeout_s),
                        context_refresher=context_refresher,
                        extend_deadline_fn=(
                            _extend_t1_deadline if context_refresher is not None else None
                        ),
                        verdict_probe=verdict_probe,
                        # Streaming detection (opt-in) — same knobs T0
                        # uses; off means the helper awaits each call
                        # whole, exactly as before.
                        stream_detect=stream_on,
                        stall_timeout_s=(
                            cfg.stall_timeout_s if cfg is not None else 30.0
                        ),
                        thinking_budget_tokens=(
                            cfg.thinking_budget_tokens if cfg is not None else 0
                        ),
                    )
                    if wait_for_timeout is not None:
                        await asyncio.wait_for(inner_coro, timeout=wait_for_timeout)
                    else:
                        await inner_coro
                    result = agent_run.result
                except UsageLimitExceeded as exc:
                    # If T1 also hits the cap, we're out of escalation
                    # tiers for this iteration. Return the same
                    # `max_iterations` reason — a T2 second opinion
                    # is a future escalation.
                    log_wall_hit(
                        phase="T1",
                        pr_number=pr_number,
                        terminated_reason="max_iterations",
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                    )
                    print(f"::warning::T1 continuation hit iteration cap: {exc}")
                    # Even on cap-trip, surface the tools T1 actually
                    # fired so the merged finish-line + comment footer
                    # reflect them. Caller union-merges with T0's set.
                    early_terminated_reason = "max_iterations"
                except PerCallTimeoutExceeded as exc:
                    # T1's bigger endpoint also hung on a single turn —
                    # this is unusual (the T1 backend has more
                    # headroom than T0's) but
                    # surfaces the same way for dashboard splitting.
                    log_wall_hit(
                        phase="T1",
                        pr_number=pr_number,
                        terminated_reason="per_call_timeout",
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                    )
                    print(
                        f"::warning::T1 continuation hit per-call timeout: "
                        f"turn {exc.turn} took {exc.elapsed_s:.1f}s "
                        f"(cap {exc.cap_s:.0f}s)"
                    )
                    early_terminated_reason = "per_call_timeout"
                except (ReasoningSpiralDetected, StreamStallDetected) as exc:
                    if isinstance(exc, StreamStallDetected):
                        print(
                            f"::warning::T1 turn {exc.turn} stalled "
                            f"({exc.idle_s:.0f}s with no delta after "
                            f"{exc.streamed_tokens} tokens) — re-drawing once"
                        )
                    else:
                        print(
                            f"::warning::T1 turn {exc.turn} hit the completion "
                            f"ceiling without committing ({exc.out_tokens} out "
                            f"tokens) — re-drawing once"
                        )
                    try:
                        messages = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        messages = []
                    redraw_from_turn = exc.turn
                    redraw_prior_text = getattr(exc, "partial_text", "") or ""
                except UnexpectedModelBehavior as exc:
                    # Chiefly a tool call truncated mid-arguments, which
                    # the boundary check above doesn't claim. Same root
                    # cause, same re-draw; anything else keeps today's
                    # `agent-loop-errored` mapping via the outer handler.
                    from cora.core import spiral as _spiral

                    if not (
                        redraw_on and _spiral.is_completion_ceiling_exception(exc)
                    ):
                        raise
                    print(
                        f"::warning::T1 hit the completion ceiling mid-commit "
                        f"({exc!s:.120}) — re-drawing once"
                    )
                    try:
                        messages = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        messages = []
                    redraw_from_turn = turn_counter[0]
                except (TimeoutError, WallTimeExceeded) as exc:
                    overshoot = (
                        getattr(exc, "overshoot_s", None)
                        if isinstance(exc, WallTimeExceeded)
                        else None
                    )
                    log_wall_hit(
                        phase="T1",
                        pr_number=pr_number,
                        terminated_reason="wall_time",
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                    )
                    detail = (
                        f"overshoot={overshoot:.1f}s"
                        if overshoot is not None
                        else f"asyncio.wait_for fired ({exc!s})"
                    )
                    print(f"::warning::T1 continuation hit wall-time guard ({detail})")
                    early_terminated_reason = "wall_time"

            # Uncommitted-draw re-draw — inside `async with agent`, so
            # the MCP sessions stay warm. Re-sends the exact payload
            # that spiralled (no new user prompt), re-rolling the
            # sampler rather than rephrasing the request. See
            # `deep_review.deep_review_call` for the full rationale.
            if redraw_from_turn is not None and result is None:
                from cora.core import spiral as _spiral

                prefix = _spiral.committed_prefix(messages)
                if not prefix:
                    log_spiral_redraw(
                        phase="T1",
                        pr_number=pr_number,
                        turn=redraw_from_turn,
                        outcome="skipped",
                        log=gha_log,
                        reason="no_committed_prefix",
                    )
                else:
                    try:
                        redraw = await agent.run(
                            # Pure re-send unless an aborted stream left
                            # visible text — then resume from it.
                            (
                                _spiral.build_resume_leadin(redraw_prior_text)
                                if redraw_prior_text.strip()
                                else None
                            ),
                            message_history=prefix,
                            deps=deps,
                            model_settings=ModelSettings(
                                max_tokens=(
                                    cfg.deep_max_output_tokens
                                    if cfg is not None
                                    else _c.DEEP_MAX_OUTPUT_TOKENS
                                ),
                                temperature=0.2,
                                timeout=timeout_s,
                            ),
                            usage_limits=UsageLimits(
                                request_limit=max(
                                    1, max_iterations - redraw_from_turn
                                )
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log_spiral_redraw(
                            phase="T1",
                            pr_number=pr_number,
                            turn=redraw_from_turn,
                            outcome=(
                                "spiralled_again"
                                if _spiral.is_completion_ceiling_exception(exc)
                                else "errored"
                            ),
                            log=gha_log,
                            detail=f'"{exc!s:.120}"',
                        )
                    else:
                        redraw_messages = list(redraw.all_messages())
                        redraw_tools = _spiral.tool_call_names(
                            redraw_messages, start=len(prefix)
                        )
                        for name in redraw_tools:
                            tool_call_counter[name] = (
                                tool_call_counter.get(name, 0) + 1
                            )
                            try:
                                budget.add_tool_call(name)
                            except Exception:  # noqa: BLE001
                                pass
                        result = redraw
                        messages = redraw_messages
                        log_spiral_redraw(
                            phase="T1",
                            pr_number=pr_number,
                            turn=redraw_from_turn,
                            outcome="recovered",
                            log=gha_log,
                            redraw_tool_calls=len(redraw_tools),
                        )
    except Exception as exc:  # noqa: BLE001
        # See deep_review.py for the rationale — preserve an inner
        # wall-hit reason if one was already set; only treat outer
        # exceptions as `agent-loop-errored:` when nothing else
        # terminated cleanly. `!r` surfaces the class for genuine
        # errors that stringify to nothing (CancelledError etc.).
        if early_terminated_reason is None:
            return "", f"agent-loop-errored: {exc!r}", tools_available
        gha_log(
            f"T1 agent context cleanup raised after {early_terminated_reason} "
            f"trip — suppressing: {exc!r}"
        )

    # Re-draw exhausted: keep whatever the spiralled turn said rather
    # than dropping T1's contribution. Same salvage rule as T0.
    if redraw_from_turn is not None and result is None and early_terminated_reason is None:
        from cora.core import spiral as _spiral

        salvage = _spiral.extract_final_text(messages)
        if salvage.strip():
            if contract_armed and not initial_tool_contract_satisfied(
                messages, gha_log=gha_log, pr_number=pr_number, phase="T1"
            ):
                return "", "required-tool-unhonored", tools_available
            gha_log(
                f"T1 re-draw did not recover (PR #{pr_number}) — posting "
                f"the truncated turn ({len(salvage)} chars)"
            )
            return salvage, None, tools_available
        return "", "agent-loop-errored: spiral-redraw-exhausted", tools_available

    if early_terminated_reason is not None:
        return "", early_terminated_reason, tools_available

    # Best-effort usage accumulation (additive — T0 already counted).
    try:
        budget.add_usage(_PydanticAIUsageAdapter(resolve_run_usage(result)))
    except Exception as exc:  # noqa: BLE001
        gha_log(f"pydantic-ai usage adapter failed (T1): {exc}")

    from cora.core.litellm_capture import (
        drain_captured_headers,
        resolve_backend_attribution,
    )
    captured = drain_captured_headers()
    budget.record_litellm_headers(captured)
    budget.set_resolved_model(
        resolve_backend_attribution(
            captured,
            result,
            fallback=f"unknown (T1 on {t1_model_alias}, no header or body model)",
        )
    )

    body = result.output if isinstance(result.output, str) else str(result.output)
    if contract_armed and not initial_tool_contract_satisfied(
        messages, gha_log=gha_log, pr_number=pr_number, phase="T1"
    ):
        return "", "required-tool-unhonored", tools_available
    return body, None, tools_available


class _PydanticAIUsageAdapter:
    """Adapt Pydantic-AI's `RunUsage` shape (`input_tokens` /
    `output_tokens` / `total_tokens`, with the legacy `request_tokens` /
    `response_tokens` aliases as fallback) to the OpenAI-shaped object
    `Budget.add_usage` expects."""

    def __init__(self, pa_usage):
        self.prompt_tokens = usage_tokens(pa_usage, "input_tokens", "request_tokens")
        self.completion_tokens = usage_tokens(pa_usage, "output_tokens", "response_tokens")
        self.total_tokens = usage_tokens(pa_usage, "total_tokens")
