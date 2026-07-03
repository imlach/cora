"""Quick-mode reviewer entrypoint — single LLM call, no MCP, no tools.

`quick_review_call` is the entry the reviewer's quick path
(`MAX_TOOL_ITERATIONS=0`, fired by the `review-quick` label) calls
into. The diff + retrieval pre-pack is already in
`initial_user_prompt`; this just runs one Pydantic-AI Agent invocation
with `output_type=str` and returns the body for downstream leak
detection + verdict parsing.

Returns `(body, terminated_reason)`. `terminated_reason` is `None` on
success, otherwise a short string the caller treats as a soft-fail
skip via the existing no-final-body route (check-run conclusion
`cancelled`, retry on next push).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cora.core import config as _c
from cora.core.budget import resolve_run_usage, usage_tokens

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


async def quick_review_call(
    *,
    endpoint_base_url: str,
    llm_gateway_key: str,
    model_alias: str,
    system_prompt: str,
    initial_user_prompt: str,
    budget,  # Budget — typed loosely to avoid an import cycle
    timeout_s: int,
    pr_number: str,
    # The run's ReviewerConfig, threaded into Deps so tools/hooks read
    # tunables from the config object. None → Deps default-constructs
    # one whose fields mirror the engine constants (same behaviour).
    cfg: "ReviewerConfig | None" = None,
) -> tuple[str, str | None]:
    """Single-turn quick review. See module docstring."""
    # Local import keeps the framework dep off the module-load path.
    # ImportError here is fatal (no fallback path); the outer wrapper
    # still soft-fails the workflow via the no-final-body skip route
    # — `("", "pydantic-ai unavailable: ...")` maps to a `cancelled`
    # check-run.
    try:
        from pydantic_ai import ModelSettings
    except ImportError as exc:
        return "", f"pydantic-ai unavailable: {exc}"

    from cora.core.agent import AgentConfig, Deps, make_review_agent
    from cora.core.rate_limit import run_with_rate_limit_backoff

    agent_config = AgentConfig(
        endpoint_base_url=endpoint_base_url,
        api_key=llm_gateway_key,
        model_alias=model_alias,
        system_prompt=system_prompt,
        # `retries=1` matches the existing leak-retry budget — no
        # tool-call retries because quick mode doesn't expose tools.
        retries=1,
    )
    agent = make_review_agent(agent_config)
    # Quick mode's own (larger) output cap — single-shot, so reasoning +
    # verdict must fit one call. See `_c.QUICK_MAX_OUTPUT_TOKENS`.
    max_tokens = (
        cfg.quick_max_output_tokens if cfg is not None
        else _c.QUICK_MAX_OUTPUT_TOKENS
    )

    # Minimal Deps for quick mode — observability callables stay at
    # no-op defaults because quick mode doesn't fire tools/hooks that
    # would call them. `repo` is unused in this path.
    deps_kwargs = {"cfg": cfg} if cfg is not None else {}
    deps = Deps(
        repo="",
        pr_number=pr_number,
        mode="quick",
        **deps_kwargs,
    )

    # Spiral-recovery flag (default-OFF). When off, the call path below is
    # unchanged from before this feature: no capture wrapper, no retry.
    spiral_on = cfg is not None and cfg.spiral_recovery

    # Wrap the single LLM call in 429-backoff so that LiteLLM's
    # `max_parallel_requests` throttle on the review backend
    # appears as "still queued" in the PR check rather
    # than a skipped review. See `cora.core.rate_limit` for the
    # full rationale and the service-ification follow-up note.
    async def _call():
        return await agent.run(
            initial_user_prompt,
            deps=deps,
            model_settings=ModelSettings(
                max_tokens=max_tokens,
                temperature=0.2,
                timeout=timeout_s,
            ),
        )

    if not spiral_on:
        try:
            result = await run_with_rate_limit_backoff(
                _call,
                description=f"quick review PR #{pr_number}",
            )
        except Exception as exc:  # noqa: BLE001
            return "", f"agent run failed: {exc}"
    else:
        # Spiral-recovery path. `capture_run_messages()` exposes the
        # partial message history even when `agent.run` raises mid-flight,
        # so on a reasoning-spiral raise we can re-seed a bounded recovery
        # turn with the model's own partial reasoning (see `cora.core.spiral`).
        from pydantic_ai import capture_run_messages
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        from cora.core import spiral as _spiral

        try:
            with capture_run_messages() as msgs:
                try:
                    result = await run_with_rate_limit_backoff(
                        _call,
                        description=f"quick review PR #{pr_number}",
                    )
                except UnexpectedModelBehavior as exc:
                    # Only intervene on the thinking-only spiral; any other
                    # UnexpectedModelBehavior soft-fails exactly as today.
                    if not _spiral.is_reasoning_spiral(msgs):
                        raise
                    print(
                        "::warning::quick mode hit reasoning spiral "
                        f"(PR #{pr_number}) — attempting bounded recovery: {exc}"
                    )
                    result = await _recover_quick(
                        agent=agent,
                        msgs=msgs,
                        deps=deps,
                        cfg=cfg,
                        timeout_s=timeout_s,
                        pr_number=pr_number,
                    )
                    if result is None:
                        # Recovery itself spiralled/failed — fall back to
                        # today's soft-fail (no regression).
                        return "", f"agent run failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            return "", f"agent run failed: {exc}"

    # Forward usage to Budget. Pydantic-AI's `result.usage` is a
    # `RunUsage` (now a property, not a method — see `resolve_run_usage`);
    # map its `input_tokens` / `output_tokens` onto OpenAI's
    # `prompt_tokens` / `completion_tokens` for `Budget.add_usage`.
    # Soft-fail — token tracking isn't critical path.
    try:
        budget.add_usage(_PydanticAIUsageAdapter(resolve_run_usage(result)))
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::pydantic-ai usage adapter failed: {exc}")

    # Drain the `x-litellm-*` headers the httpx hook captured into
    # the ContextVar during the call (see `litellm_capture.py`).
    # `resolved_model` is the most-useful single header; full dict
    # lands in `litellm_headers` for diagnostics.
    from cora.core.litellm_capture import (
        drain_captured_headers,
        resolved_model_from,
    )
    captured = drain_captured_headers()
    budget.record_litellm_headers(captured)
    budget.set_resolved_model(
        resolved_model_from(captured) or "unknown (no x-litellm-* headers)"
    )

    # `result.output` is a string when `output_type=str` (the factory
    # default). Defensive str() in case a future framework version
    # ever returns something else under the same configuration.
    body = result.output if isinstance(result.output, str) else str(result.output)
    return body, None


async def _recover_quick(
    *,
    agent,
    msgs: list,
    deps,
    cfg: "ReviewerConfig",
    timeout_s: int,
    pr_number: str,
):
    """ONE bounded recovery turn for a quick-mode reasoning spiral.

    Re-runs the SAME agent with the captured `msgs` as `message_history`
    plus a "you've already analyzed this, conclude now" lead-in seeded
    with the partial-reasoning tail (see `cora.core.spiral`). Reasoning
    stays ENABLED but the output cap is the tight
    `spiral_recovery_max_output_tokens`. Returns the pydantic-ai result
    on success, or None if recovery raises (caller soft-fails).
    """
    from pydantic_ai import ModelSettings

    from cora.core import spiral as _spiral

    leadin = _spiral.build_recovery_leadin(
        _spiral.extract_partial_reasoning(
            msgs, char_cap=cfg.spiral_recovery_reasoning_char_cap
        )
    )
    try:
        return await agent.run(
            leadin,
            message_history=msgs,
            deps=deps,
            model_settings=ModelSettings(
                max_tokens=cfg.spiral_recovery_max_output_tokens,
                temperature=0.2,
                timeout=timeout_s,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — recovery is best-effort
        print(
            "::warning::quick mode spiral recovery failed "
            f"(PR #{pr_number}) — falling back to soft-fail: {exc!r}"
        )
        return None


class _PydanticAIUsageAdapter:
    """Adapt Pydantic-AI's `RunUsage` shape (`input_tokens` /
    `output_tokens` / `total_tokens`, with the legacy `request_tokens` /
    `response_tokens` aliases as fallback) to the OpenAI-shaped object
    `Budget.add_usage` expects (`prompt_tokens` / `completion_tokens` /
    `total_tokens`)."""

    def __init__(self, pa_usage):
        self.prompt_tokens = usage_tokens(pa_usage, "input_tokens", "request_tokens")
        self.completion_tokens = usage_tokens(pa_usage, "output_tokens", "response_tokens")
        self.total_tokens = usage_tokens(pa_usage, "total_tokens")
