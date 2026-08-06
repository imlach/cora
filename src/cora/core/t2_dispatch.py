"""T2 second-opinion dispatch on the alt-reviewer endpoint (an
alternate-family model) + body composition for the
disagreement-resolved comment.

The point of T2 is *epistemic diversity*, not "bigger model": T0/T1
share a model family and correlated blind spots; the alt-reviewer
is a different model family with a different tokenizer / RLHF /
code priors. Running it against the same diff gives the disagreement
resolver in `disagreement.py` a meaningfully independent vote.

This module owns two things:
  - `call_t2_alt_reviewer` — a fresh, independent deep review (no
    message-history carry-over from T0; diversity is the point).
  - `compose_disagreement_body` — once T0 + T2 verdicts are in,
    take the resolver's `Resolution` and assemble the final PR
    comment body: banner above, leading body in the middle,
    collapsed `<details>` dissent block at the bottom.

T3 escalation stays disabled here (future work). When the
resolver returns `escalate_t3`, we surface both bodies in the
dissent block and emit `escalation_recommended=true` on the T2
`tier_verdict` event so a dashboard can rank PRs the user should
look at manually — but no cloud call fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from cora.core.disagreement import Resolution

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.core.mcp_sessions import McpServerSpec
    from cora.providers.git import GitProvider


async def call_t2_alt_reviewer(
    *,
    endpoint_base_url: str,
    llm_gateway_key: str,
    t2_model_alias: str,
    system_prompt: str,
    initial_user_prompt: str,
    budget,  # Budget — loose typing avoids cross-module dep
    timeout_s: int,
    pr_number: str,
    repo: str,
    mcp_url: str,
    mcp_headers: dict[str, str],
    mcp_actions_url: str | None = None,
    mcp_actions_headers: dict[str, str] | None = None,
    web_fetch_url: str | None = None,
    web_fetch_headers: dict[str, str] | None = None,
    # Generic extra MCP sessions (from `MCP_SERVERS`) — forwarded
    # straight through to `deep_review_call`; see `cora.core.mcp_sessions`.
    extra_sessions: "Sequence[McpServerSpec]" = (),
    allowed_tools: set[str],
    tool_arg_defaults: dict[str, dict[str, Any]] | None = None,
    max_iterations: int = 8,
    gha_log: Callable[[str], None] = print,
    # The run's ReviewerConfig — forwarded to `deep_review_call` so the
    # T2 leg reads the same config object the primary tiers used. None
    # keeps the legacy behaviour exactly.
    cfg: "ReviewerConfig | None" = None,
    # Repo-introspection backend, forwarded to `deep_review_call`'s
    # local grep_repo / git_show tools. None → LocalGitProvider.
    git_provider: "GitProvider | None" = None,
) -> tuple[str, str | None, list[str]]:
    """Fire a T2 second-opinion review on the alt-reviewer endpoint.

    Returns `(final_body, terminated_reason, tools_available)` —
    same leading-three shape as `deep_review_call`'s tuple so the
    caller's downstream leak / verdict / observability code can
    consume the result uniformly.

    Unlike `continue_on_t1`, T2 does NOT carry T0's message history
    forward. The whole point is a fresh, independent run — same
    diff + retrieval pre-pack + system prompt, different model.
    Correlated bias on the prior turns would defeat the diversity
    signal we're trying to introduce.

    `max_iterations=8` is tighter than T0's 12 because the
    alt-reviewer's role is to form an independent verdict on the
    pre-packed initial prompt, not to do deep tool-use exploration —
    the retrieval pre-pack already covers the static-corpus work.
    Override via `AGENT_REVIEW_T2_MAX_ITERATIONS` env if a soak
    surfaces a need for more headroom.
    """
    from cora.core.deep_review import deep_review_call

    # Same dispatch shape as T0 — the `phase="T0"` logging label
    # inside `deep_review_call` is set inside that function and we
    # let it stay; the T2-vs-T0 split is observable via the
    # surrounding `agent_review t2 …` markers + the `tier_verdict`
    # event the caller emits with `tier=T2`.
    body, terminated_reason, tools_available, _messages = await deep_review_call(
        endpoint_base_url=endpoint_base_url,
        llm_gateway_key=llm_gateway_key,
        model_alias=t2_model_alias,
        system_prompt=system_prompt,
        initial_user_prompt=initial_user_prompt,
        budget=budget,
        timeout_s=timeout_s,
        pr_number=pr_number,
        repo=repo,
        mcp_url=mcp_url,
        mcp_headers=mcp_headers,
        mcp_actions_url=mcp_actions_url,
        mcp_actions_headers=mcp_actions_headers,
        web_fetch_url=web_fetch_url,
        web_fetch_headers=web_fetch_headers,
        extra_sessions=extra_sessions,
        allowed_tools=allowed_tools,
        tool_arg_defaults=tool_arg_defaults,
        max_iterations=max_iterations,
        gha_log=gha_log,
        cfg=cfg,
        git_provider=git_provider,
    )
    return body, terminated_reason, tools_available


def compose_disagreement_body(
    *,
    resolution: Resolution,
    primary_body: str,
    primary_tier_label: str,
    t2_body: str,
    t2_model_alias: str,
    primary_model_alias: str = "primary",
) -> str:
    """Assemble the final PR comment body from the resolver's verdict.

    Layout:

        🔄 {banner}                        <- only on disagreement paths
        {leading verdict body}
        ---
        <details><summary>{dissent tier} ({model}) said: {verdict}</summary>
        {dissent body verbatim}
        </details>

    `primary_tier_label` is the same-family tier that produced
    `primary_body` (T0 or T1). The dissent tier comes from the
    resolver; for `escalate_t3` (with T3 disabled) the dissent
    block carries BOTH local verdicts so the human reviewer sees
    the categorical split.

    Single-tier path (`path="single_tier"`, no T2 ran) is unreachable
    from this function's caller — the dispatcher only calls
    `compose_disagreement_body` when T2 actually ran. Defensive
    early-return preserves the primary body verbatim if it does
    slip through.
    """
    if resolution.path == "single_tier":
        return primary_body

    if resolution.path == "agree":
        # Agree — no banner, no dissent block. The dispatcher could
        # merge T2's extra findings as Notes later (TODO future
        # iteration), but for now keeping the primary body unchanged
        # is the safe minimum: high-confidence reviews shouldn't get
        # diluted by appended duplication.
        return primary_body

    # Banner is set by the resolver for both `adopt_conservative` and
    # `escalate_t3` paths. `escalate_t3` here means "would escalate if
    # T3 were enabled" — the resolver defaults t3_enabled=False so
    # this branch only fires when the dispatcher explicitly opts in.
    banner = resolution.banner or ""

    # Pick which body is "leading" (renders normally) and which is
    # the dissent (collapsed). The resolver's `adopted_tier` tells
    # us which side won.
    if resolution.adopted_tier == primary_tier_label:
        leading = primary_body
        dissent_body = t2_body
        dissent_tier = "T2"
        dissent_model = t2_model_alias
    else:
        # T2 won the conservative pick.
        leading = t2_body
        dissent_body = primary_body
        dissent_tier = primary_tier_label
        dissent_model = primary_model_alias  # T0/T1 share the primary alias

    dissent_verdict_label = _verdict_label_for_dissent_summary(
        resolution=resolution,
        dissent_tier=dissent_tier,
        primary_tier_label=primary_tier_label,
    )

    return (
        f"{banner}\n\n"
        f"{leading.rstrip()}\n\n"
        f"---\n\n"
        f"<details>\n"
        f"<summary>{dissent_tier} ({dissent_model}) said: "
        f"{dissent_verdict_label}</summary>\n\n"
        f"{dissent_body.rstrip()}\n\n"
        f"</details>\n"
    )


def _verdict_label_for_dissent_summary(
    *,
    resolution: Resolution,
    dissent_tier: str,
    primary_tier_label: str,
) -> str:
    """Render the dissent verdict for the `<details>` summary line.

    The resolver carries `adopted_verdict` but not the dissent's
    verdict explicitly — we know the resolver compared two tiers, so
    the dissent verdict is the one that *isn't* `adopted_verdict`.
    The Resolution doesn't track it; the caller composing this body
    knows which tier is dissenting from `resolution.dissent_tier` but
    not its raw verdict word.

    Practical workaround: the summary line is purely cosmetic for the
    human reader, so we use a stable label rather than the verdict
    word — `<dissent_tier>'s view` reads cleanly and avoids needing
    to thread the dissent verdict through this function. If a future
    iteration wants the verdict word here, plumb it through as an
    explicit parameter.
    """
    _ = resolution, dissent_tier, primary_tier_label  # reserved
    return "click to expand"
