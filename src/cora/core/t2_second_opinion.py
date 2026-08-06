"""Default second-opinion implementation — the T2 alt-reviewer dispatch +
disagreement resolution, behind the `SecondOpinionProvider` seam.

This is the seam's default implementation —
`SecondOpinionProvider.from_config` returns it, and it self-disarms when
`cfg.t2_disagreement` is off (the default), so a default config produces
the same behaviour as running with no second opinion at all.

Owns the engine-coupled work the public seam (`cora/second_opinion.py`)
deliberately keeps out: the `call_t2_alt_reviewer` dispatch
(`core/t2_dispatch.py`), the `resolve_disagreement` policy
(`core/disagreement.py`), the `compose_disagreement_body` banner/dissent
layout, and the `tier_verdict` / `disagreement` Loki events
(`core/loop_logging.py`). All imported lazily (or inside methods) so the
generic Null path never pulls in pydantic_ai / openai transitively, and so
existing tests that monkeypatch `cora.core.t2_dispatch.call_t2_alt_reviewer`
at module level keep hitting the live symbol.

T2 is about *epistemic diversity*, not "bigger model": T0/T1 share a
model family with correlated blind spots; the alt-reviewer is a
different family. See `t2_dispatch.py` / `disagreement.py` for the
policy scoping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from cora.core import config as _c
from cora.core.leak import (
    detect_blocker,
    detect_reasoning_leak,
    parse_verdict_from_body,
    strip_reasoning,
)
from cora.second_opinion import SecondOpinionProvider, SecondOpinionResult

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.core.mcp_sessions import McpServerSpec
    from cora.providers.git import GitProvider


class T2SecondOpinion(SecondOpinionProvider):
    """T2 alt-reviewer + verdict-disagreement resolution.

    Gated by `cfg.t2_disagreement` (off by default);
    when off, `should_dispatch` returns False and the run carries no second
    opinion."""

    def should_dispatch(
        self, *, cfg: "ReviewerConfig", is_quick: bool, primary_body: str | None
    ) -> bool:
        # The dispatch gate: opt-in via `cfg.t2_disagreement` (default
        # off), deep mode, a primary body was produced,
        # and the primary isn't already a hard blocker (no diversity vote
        # needed when the same-family verdict is already maximally
        # conservative).
        return bool(
            (not is_quick)
            and primary_body
            and cfg.t2_disagreement
            and not detect_blocker(primary_body)
        )

    async def dispatch(
        self,
        *,
        cfg: "ReviewerConfig",
        endpoint_base_url: str,
        api_key: str,
        system_prompt: str,
        initial_user_prompt: str,
        budget: Any,
        timeout_s: int,
        pr_number: str,
        repo: str,
        mcp_url: str,
        mcp_headers: dict[str, str],
        mcp_actions_url: str | None,
        mcp_actions_headers: dict[str, str] | None,
        web_fetch_url: str | None,
        git: "GitProvider | None",
        log: Callable[[str], None],
        iter_log: Callable[[str], None],
        primary_terminated_reason: str | None = None,
        extra_sessions: "Sequence[McpServerSpec]" = (),
    ) -> SecondOpinionResult:
        # Resolve the alias once — the disagreement dispatch, its composition
        # banner, and the patch-escalation verifier all route to the same
        # alias (the caller reads `result.model_alias` for the latter two).
        t2_model = (cfg.t2_model or "").strip() or _c.DEFAULT_T2_MODEL
        t2_max_iters = cfg.t2_max_iterations

        log(
            f"T2 second-opinion dispatch on `{t2_model}` "
            f"(primary terminated={primary_terminated_reason or 'clean'})"
        )

        from cora.core.t2_dispatch import call_t2_alt_reviewer

        body, terminated_reason, tools = await call_t2_alt_reviewer(
            endpoint_base_url=endpoint_base_url,
            llm_gateway_key=api_key,
            t2_model_alias=t2_model,
            system_prompt=system_prompt,
            initial_user_prompt=initial_user_prompt,
            budget=budget,
            timeout_s=timeout_s,
            pr_number=pr_number,
            repo=repo,
            mcp_url=mcp_url,
            mcp_headers=mcp_headers,
            mcp_actions_url=mcp_actions_url or None,
            mcp_actions_headers=mcp_actions_headers,
            web_fetch_url=web_fetch_url or None,
            web_fetch_headers=None,
            extra_sessions=extra_sessions,
            allowed_tools=set(cfg.read_tools)
            | set(cfg.action_tools)
            | set(cfg.web_tools)
            | set(cfg.local_repo_tools)
            | set(cfg.extra_tools),
            tool_arg_defaults={
                "web_fetch_doc": {"caller": "cora"},
            },
            max_iterations=t2_max_iters,
            gha_log=iter_log,
            cfg=cfg,
            git_provider=git,
        )

        if body:
            log(
                f"T2 review produced ({len(body)} chars); "
                f"composition deferred until after primary leak detection"
            )
        else:
            log(
                f"T2 produced no body (reason: {terminated_reason or 'unknown'}); "
                f"keeping primary body unchanged"
            )

        return SecondOpinionResult(
            dispatched=True,
            body=body or None,
            terminated_reason=terminated_reason,
            tools_available=list(tools or []),
            model_alias=t2_model,
        )

    def compose(
        self,
        *,
        result: SecondOpinionResult,
        cfg: "ReviewerConfig",
        is_quick: bool,
        primary_body_to_post: str,
        primary_terminated_reason: str | None,
        pr_number: str,
        log: Callable[[str], None],
        iter_log: Callable[[str], None],
    ) -> str:
        body_to_post = primary_body_to_post

        # Fold the T2 second-opinion verdict into the post-leak body.
        # Composition happens HERE (post-leak) so the resolver banner lands
        # at position 0 of the final comment.
        if not is_quick and result.body:
            t2_cleaned, _t2_reasoning_stripped = strip_reasoning(result.body)
            t2_body_to_post, _t2_preamble, t2_is_leak = detect_reasoning_leak(t2_cleaned)
            if t2_is_leak:
                log(
                    "T2 body failed leak detection (no Verdict marker); "
                    "keeping primary body, skipping disagreement composition"
                )
                result.body = None
            else:
                from cora.core.disagreement import TierVerdict, resolve_disagreement
                from cora.core.kv_continuation import T1_TERMINATED_REASONS
                from cora.core.t2_dispatch import compose_disagreement_body

                t2_model = result.model_alias or (
                    (cfg.t2_model or "").strip() or _c.DEFAULT_T2_MODEL
                )
                primary_tier_label = (
                    "T1"
                    if primary_terminated_reason in T1_TERMINATED_REASONS
                    else "T0"
                )
                primary_verdict = parse_verdict_from_body(body_to_post)
                primary_has_blocker = detect_blocker(body_to_post)
                t2_verdict = parse_verdict_from_body(t2_body_to_post)
                t2_has_blocker = detect_blocker(t2_body_to_post)

                primary_tv = TierVerdict(
                    tier=primary_tier_label,
                    verdict=primary_verdict,
                    body=body_to_post,
                    has_blocker=primary_has_blocker,
                )
                t2_tv = TierVerdict(
                    tier="T2",
                    verdict=t2_verdict,
                    body=t2_body_to_post,
                    has_blocker=t2_has_blocker,
                )

                if primary_tier_label == "T1":
                    t2_resolution = resolve_disagreement(
                        t0=primary_tv,
                        t1=primary_tv,
                        t2=t2_tv,
                        t3_enabled=False,
                    )
                else:
                    t2_resolution = resolve_disagreement(
                        t0=primary_tv,
                        t2=t2_tv,
                        t3_enabled=False,
                    )

                body_to_post = compose_disagreement_body(
                    resolution=t2_resolution,
                    primary_body=body_to_post,
                    primary_tier_label=primary_tier_label,
                    t2_body=t2_body_to_post,
                    t2_model_alias=t2_model,
                    primary_model_alias=cfg.model,
                )
                # Keep the post-leak body for the T2 tier_verdict emission and
                # the propose-patch escalation reuse.
                result.body = t2_body_to_post
                result.verdict = t2_verdict
                result.has_blocker = t2_has_blocker
                result.resolution = t2_resolution
                log(
                    f"T2 resolution path={t2_resolution.path} "
                    f"gap={t2_resolution.gap} "
                    f"adopted={t2_resolution.adopted_tier}"
                )

        return body_to_post

    def emit_events(
        self,
        *,
        result: SecondOpinionResult,
        cfg: "ReviewerConfig",
        is_quick: bool,
        pr_number: str,
        iter_log: Callable[[str], None],
    ) -> None:
        # Emit the T2 `tier_verdict` event +
        # the `disagreement` event when T2 produced and resolved. Called
        # after the primary tier's `tier_verdict` so ordering is unchanged.
        if not (not is_quick and result.body and result.resolution is not None):
            return

        from cora.core.loop_logging import log_tier_verdict

        escalation_recommended = result.resolution.gap == 2 and result.has_blocker
        # `mode` lets the dashboard split the disagreement stream by
        # review tier; large == the SKIP_T0 path.
        review_mode = "large" if cfg.skip_t0 else "deep"
        log_tier_verdict(
            pr_number=pr_number,
            tier="T2",
            verdict=result.verdict,
            has_blocker=result.has_blocker,
            body_chars=len(result.body),
            log=iter_log,
        )
        iter_log(
            f"agent_review iter pr_number={pr_number} phase=T2 "
            f"event=disagreement gap={result.resolution.gap} "
            f"path={result.resolution.path} "
            f"adopted_tier={result.resolution.adopted_tier} "
            f"mode={review_mode} "
            f"escalation_recommended={str(escalation_recommended).lower()}"
        )
