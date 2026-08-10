"""Pydantic-AI Agent factory + typed `Deps` for the agentic reviewer
and triage agent.

Both consumers — the review orchestrator (deep mode) and
the triage agent — build their per-call Agent through
`make_review_agent` so MCP wire-up, tool-allowlist filtering, and
local-tool registration stay in one place. The factory keeps the
output free-form (`output_type=str`) so the existing `leak.py`
regex-parse pipeline downstream is unchanged; structured output is
opt-in per call site if it ever needs to harden a specific tier.

Provider-agnostic by design: the reviewer talks to whatever
OpenAI-compatible endpoint sits behind `endpoint_base_url`. The
reference deployment fronts its models with LiteLLM; the application
code doesn't know or care.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


# Request header carrying the per-review session identifier. Fixed
# name, caller-supplied value — a gateway configured to hash on it can
# pin one review's turns to one replica; anything else ignores it.
SESSION_HEADER = "x-review-session"


# How an MCP server's tool errors come back to the model.
#
# Pydantic-AI's per-tool retry budget is cumulative across a whole run
# and only clears when that tool succeeds. Under the default `'retry'`
# behaviour a recoverable MCP error (wrong identifier type, nonexistent
# id) spends it — so a first mistake the model routed around by picking
# a *different* tool leaves the budget at zero, and the next mistake on
# the same tool raises `UnexpectedModelBehavior` and kills the review
# mid-flight. Two unrelated bad arguments, one dead review.
#
# `'failed'` hands the server's error text back as a
# `ToolReturnPart(outcome='failed')` and spends no budget: the model
# reads the error and chooses its next call, the way it already does
# with a tool that returns an `ERROR: …` string. The request, iteration
# and wall limits stay the only hard bound on the loop.
#
# A protocol-level `McpError` — the transport itself failing rather
# than a tool rejecting its arguments — is forced back to `'retry'`
# inside pydantic-ai, so connection failure keeps its own bounded path.
MCP_TOOL_ERROR_BEHAVIOR = "failed"


def mcp_tool_error_behavior() -> str:
    """`MCP_TOOL_ERROR_BEHAVIOR`, degraded to `'retry'` when the
    installed pydantic-ai can't express it.

    `'failed'` arrived with `ToolFailed` in pydantic-ai 2.16. An older
    build accepts the string and then falls through its `'retry'` /
    `'error'` branches, re-raising the raw `fastmcp.ToolError` out of
    the toolset — a hard crash on the *first* tool error, which is
    worse than the defect this replaces. The check isn't redundant with
    the floor pin in `pyproject.toml`: a deployment can resolve
    pydantic-ai from a pre-baked runner image rather than from cora's
    own dependency set.
    """
    try:
        from pydantic_ai.exceptions import ToolFailed  # noqa: F401
    except ImportError:
        return "retry"
    return MCP_TOOL_ERROR_BEHAVIOR


def _default_reviewer_config() -> ReviewerConfig:
    """Default-construct a `ReviewerConfig` lazily — the import happens at
    first `Deps()` construction, not at module load, so `cora.core.agent`
    never participates in an import cycle with the `cora` package root."""
    from cora.config import ReviewerConfig

    return ReviewerConfig()


# Per-PR runtime dependencies the Agent + its tools/hooks reach
# back through. Replaces the current module-globals + threaded-args
# pattern for the GitHub client, Loki pusher, eval-mode flag, etc.
# Accessed inside tool/hook bodies via `RunContext[Deps].deps`.
@dataclass
class Deps:
    """Typed dependency bundle injected into the Pydantic-AI Agent."""

    # PR identity — used for Loki labelling, check-run posting,
    # comment marker resolution.
    repo: str
    pr_number: str

    # Observability surfaces. Both default to no-op callables so a
    # test or local-dev invocation can construct Deps without
    # wiring real Loki / GHA outputs.
    loki_push: Callable[[str, dict | None], None] = field(
        default=lambda _line, _labels=None: None,
    )
    gha_log: Callable[[str], None] = field(
        default=lambda _msg: None,
    )

    # GitHub auth. `github_token` is GITHUB_TOKEN (always present
    # in the workflow context); `gh_app_token` is the optional
    # cora App installation token for `propose_patch` dispatch
    # (empty when the mint step soft-failed; agent then runs in
    # comment-only mode).
    github_token: str = ""
    gh_app_token: str | None = None

    # Mode flags.
    mode: Literal["quick", "deep"] = "quick"
    eval_mode: bool = False

    # The run's full `ReviewerConfig` — the wiring object threaded
    # through the engine call paths so tools/hooks read tunables from
    # here instead of `cora.core.config` module constants. Defaults to
    # a fresh `ReviewerConfig()`, whose field defaults mirror those
    # constants by reference — so a Deps built without an explicit
    # config behaves identically to one built with the defaults.
    cfg: ReviewerConfig = field(default_factory=_default_reviewer_config)


# Construction-time Agent parameters. Kept separate from `Deps`
# because these are framework-build inputs (endpoint, model, MCP
# servers, retry budget) rather than runtime dependencies tools
# need at call time.
@dataclass
class AgentConfig:
    """Parameters fed to `make_review_agent` at construction time."""

    # OpenAI-compatible endpoint (a LiteLLM gateway, vLLM-direct,
    # OpenRouter — anything OpenAI-compatible, per deployment).
    # Reviewer code stays generic via the generic `OpenAIProvider`.
    endpoint_base_url: str
    api_key: str

    # Gateway model alias (e.g. `"review"` for the T0 endpoint,
    # `"main"` for the T1 endpoint). Routed by the gateway, opaque to
    # the reviewer.
    model_alias: str

    # System prompt text. Loaded from the configured quick/deep
    # prompt path by the
    # caller. Free-form-output-shaped so the existing downstream
    # leak detector + verdict parser consume the response unchanged.
    system_prompt: str

    # MCP servers — list of `(url, auth_headers)` tuples. Empty in
    # quick mode. Populated in deep mode with the three sessions
    # (the read-tools MCP server, the actions MCP server, the
    # web-fetch gate). The factory
    # wires each as a `MCPToolset(url, headers=...)` and passes them
    # via `toolsets=[...]`; Pydantic-AI owns the session lifecycle,
    # tool discovery, and dispatch from there.
    #
    # Callers MUST pre-probe optional servers (the actions server,
    # the web-fetch gate) before adding them here. An unreachable entry
    # in `toolsets` fails the whole run; the best-effort probe lives
    # one level up (in `deep_review.py` / the triage entrypoint).
    mcp_servers: list[tuple[str, dict[str, str]]] = field(
        default_factory=list,
    )

    # Local in-process tools (PR-aware grep_repo / git_show served
    # from the merge checkout — shadow the MCP server's main-mirror
    # copies). Each entry is a Pydantic-AI `Tool` instance OR a
    # callable the framework can introspect. Empty for quick mode and
    # for triage (no PR checkout). Populated in deep mode from
    # `deep_review._make_pydantic_ai_local_tools()`.
    local_tools: list[Any] = field(default_factory=list)

    # Optional allowlist applied to MCP-served tools. Each
    # `MCPToolset` is wrapped via `.filtered(...)` so only tool names
    # in this set reach the agent — matches the `ALLOWED_TOOLS`
    # tool-surface narrowing both consumers do. Empty/None means no
    # filter (every tool the server lists is exposed).
    mcp_allowed_tools: set[str] | None = None

    # Framework retry budget — Pydantic-AI retries failed tool calls
    # up to `retries` times before surfacing the error to the agent.
    # Output-validation retries don't apply here (`output_type=str`).
    #
    # It governs the local typed tools and argument-schema failures.
    # MCP-served tool errors are routed around it entirely — see
    # `MCP_TOOL_ERROR_BEHAVIOR` — because the budget is cumulative per
    # tool across the run, which made one early recoverable mistake arm
    # a later unrelated one to abort the loop.
    retries: int = 1

    # Opaque per-review session identifier. When set, every model call
    # this agent makes carries `x-review-session: <value>`.
    #
    # The client half of gateway session affinity: a gateway that can
    # hash on the header keeps one review's turns on one replica, so
    # the prefix cache the earlier turns warmed is still the one serving
    # the later ones. cora neither knows nor cares whether anything
    # downstream reads it — an unconfigured deployment sends no header
    # and nothing changes.
    session_id: str | None = None

    # Deep-review grounding contract. While no successful tool return exists
    # in the current trajectory, each model request is constrained to one
    # required tool call. Off by default for backwards compatibility.
    require_initial_tool_call: bool = False


def first_successful_tool_name(messages: list) -> str | None:
    """Return the first successfully completed tool in a trajectory."""
    for message in messages:
        for part in getattr(message, "parts", ()) or ():
            if getattr(part, "part_kind", None) != "tool-return":
                continue
            if getattr(part, "outcome", None) != "success":
                continue
            return str(getattr(part, "tool_name", "unknown") or "unknown")
    return None


def required_initial_tool_model_settings(ctx):
    """Require one serial tool call until the trajectory has a success.

    Pydantic AI invokes agent-level model-settings callables before every
    request, so the constraint automatically relaxes after the first tool
    return without coupling the policy to any particular tool name.
    """
    from pydantic_ai import ModelSettings

    if first_successful_tool_name(list(ctx.messages)) is not None:
        return ModelSettings()
    return ModelSettings(tool_choice="required", parallel_tool_calls=False)


def initial_tool_contract_satisfied(
    messages: list, *, gha_log: Callable[[str], None], pr_number: str, phase: str
) -> bool:
    """Log the contract outcome and return whether a tool succeeded."""
    first_tool = first_successful_tool_name(messages)
    outcome = "satisfied" if first_tool is not None else "required_ignored"
    gha_log(
        f"agent_review iter pr_number={pr_number} phase={phase} "
        f"event=initial_tool_contract outcome={outcome} "
        f"first_tool={first_tool or 'none'}"
    )
    return first_tool is not None


def make_review_agent(config: AgentConfig, deps_type: type = Deps):
    """Build a Pydantic-AI Agent from the supplied config.

    Returns an `Agent` instance ready to be invoked with
    `await agent.run(prompt, deps=Deps(...))`. The Agent itself is
    reusable across runs; only the per-PR `Deps` changes.

    `output_type=str` by design. Free-form output keeps the existing
    `leak.py` regex-parse pipeline unchanged downstream. Per-call
    structured output stays available for future call sites where the
    cost is justified (likely T3 cloud escalation against frontier
    models with reliable first-try schema convergence).

    MCP servers in `config.mcp_servers` are wired as
    `MCPToolset(url, headers=...)` toolsets — the framework owns the
    session lifecycle (open on `agent.run`, close on exit). Each
    entry's `auth_headers` flows directly into the toolset's
    `headers=` kwarg, and each carries
    `tool_error_behavior=mcp_tool_error_behavior()` so a recoverable
    tool error returns control to the model instead of spending the
    cumulative per-tool retry budget.

    Local tools in `config.local_tools` register as Pydantic-AI
    `tools=[...]` — declarations the framework introspects for
    schema. Local tools are listed before MCP toolsets, so on a
    name collision local wins (deep mode uses this so the PR-aware
    `grep_repo` / `git_show` shadow the MCP server's main-mirror copies).
    """
    # Local imports keep the heavy framework dep off the module-load
    # path for environments where `pydantic-ai` isn't installed yet
    # (the smoke test handles that case with `pytest.importorskip`).
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from cora.core.litellm_capture import build_capture_client

    # Custom httpx client carries a response event hook that drains
    # `x-litellm-*` headers into a ContextVar so the call site can
    # populate `Budget.resolved_model` / `litellm_headers` after
    # `agent.run()` — the openai SDK otherwise hides them behind
    # pydantic-ai's Agent wrapper. The client is owned by the
    # underlying AsyncOpenAI; not explicitly closed because the
    # per-PR process exits after the review lands.
    session_id = (config.session_id or "").strip()
    provider = OpenAIProvider(
        base_url=config.endpoint_base_url,
        api_key=config.api_key,
        http_client=build_capture_client(
            {SESSION_HEADER: session_id} if session_id else None
        ),
    )
    model = OpenAIChatModel(config.model_alias, provider=provider)

    toolsets: list[Any] = []
    if config.mcp_servers:
        # `MCPToolset` defaults to Streamable HTTP for HTTP URLs;
        # backed by fastmcp's `Client` internally. The legacy
        # `MCPServerStreamableHTTP` class is deprecated upstream.
        from pydantic_ai.mcp import MCPToolset

        allow = config.mcp_allowed_tools
        tool_error_behavior = mcp_tool_error_behavior()
        for url, headers in config.mcp_servers:
            toolset: Any = MCPToolset(
                url,
                headers=headers or None,
                tool_error_behavior=tool_error_behavior,
            )
            if allow:
                # `.filtered(...)` returns a wrapping toolset that drops
                # any tool whose name isn't in the allowlist. Defined as
                # a closure over `allow` so the filter is evaluated at
                # tool-discovery time, after the MCP server has listed
                # what it offers.
                toolset = toolset.filtered(
                    lambda _ctx, tdef, _allow=allow: tdef.name in _allow,
                )
            toolsets.append(toolset)

    return Agent(
        model,
        deps_type=deps_type,
        system_prompt=config.system_prompt,
        output_type=str,
        retries=config.retries,
        model_settings=(
            required_initial_tool_model_settings
            if config.require_initial_tool_call
            else None
        ),
        tools=tuple(config.local_tools),
        toolsets=toolsets or None,
    )
