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
code doesn't know or care. `AgentConfig.provider` widens that seam to
two gateway-less direct-SDK dialects (`"anthropic"`, `"bedrock"`) for
an adopter with only a cloud API key/credentials — see
`_build_anthropic_model` / `_build_bedrock_model` and
`core.config.SUPPORTED_LLM_PROVIDERS`. The default `"openai-compatible"`
path is untouched byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from cora.core import config as _c

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from pydantic_ai.settings import ModelSettings


def _default_reviewer_config() -> "ReviewerConfig":
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
    cfg: "ReviewerConfig" = field(default_factory=_default_reviewer_config)


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
    # Ignored on the `"anthropic"` / `"bedrock"` provider paths unless
    # explicitly pointed away from the localhost placeholder — see
    # `_build_anthropic_model`.
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

    # Which dialect to speak — mirrors `ReviewerConfig.llm_provider` /
    # `core.config.SUPPORTED_LLM_PROVIDERS`. Default keeps the
    # OpenAI-compatible path unchanged for every existing deployment.
    provider: str = _c.DEFAULT_LLM_PROVIDER

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
    retries: int = 1


def _build_openai_compatible_model(config: AgentConfig):
    """The stock, default path — byte-identical to cora's pre-seam
    behaviour. Talks OpenAI Chat Completions to whatever
    `config.endpoint_base_url` points at (a LiteLLM gateway,
    vLLM-direct, OpenRouter, ...)."""
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
    openai_provider = OpenAIProvider(
        base_url=config.endpoint_base_url,
        api_key=config.api_key,
        http_client=build_capture_client(),
    )
    return OpenAIChatModel(config.model_alias, provider=openai_provider)


def _provider_base_url_override(config: AgentConfig) -> str | None:
    """Optional base-url override for the direct-SDK provider paths.

    `config.endpoint_base_url` always carries a value (the OpenAI-
    compatible path needs one structurally), so a bare gateway-less
    `LLM_PROVIDER=anthropic` run — no `LITELLM_BASE_URL` set — still
    arrives here holding the unmodified `DEFAULT_LITELLM_BASE`
    placeholder. Forwarding that would point the direct SDK at a local
    gateway that doesn't exist for this path, so it's treated as "no
    override" — the SDK falls back to its own default endpoint. An
    adopter who explicitly points `LITELLM_BASE_URL` somewhere else
    (an Anthropic-compatible proxy, a regional endpoint, ...) gets it
    passed through untouched."""
    base_url = (config.endpoint_base_url or "").strip()
    if not base_url or base_url == _c.DEFAULT_LITELLM_BASE:
        return None
    return base_url


def _build_anthropic_model(config: AgentConfig):
    """Direct Anthropic SDK path — no gateway in between.

    Requires the optional `cora[anthropic]` extra
    (`pydantic-ai-slim[anthropic]`, which pulls in the `anthropic`
    SDK); the import is lazy so the default openai-compatible path
    never needs it installed."""
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError as exc:
        raise ImportError(
            "LLM_PROVIDER=anthropic requires the 'anthropic' extra: "
            "pip install 'cora[anthropic]'"
        ) from exc

    anthropic_provider = AnthropicProvider(
        api_key=config.api_key or None,
        base_url=_provider_base_url_override(config),
    )
    return AnthropicModel(config.model_alias, provider=anthropic_provider)


def _build_bedrock_model(config: AgentConfig):
    """Direct Amazon Bedrock SDK path — AWS credential chain, no
    gateway in between.

    Requires the optional `cora[bedrock]` extra
    (`pydantic-ai-slim[bedrock]`, which pulls in `boto3`); the import
    is lazy so the default openai-compatible path never needs it
    installed. Bedrock model IDs carry an `anthropic.` prefix — a bare
    alias like `claude-opus-4-8` (the shape every other provider
    path/dashboard uses) is prefixed automatically; an already-prefixed
    `model_alias` is left alone."""
    try:
        from pydantic_ai.models.bedrock import BedrockConverseModel
    except ImportError as exc:
        raise ImportError(
            "LLM_PROVIDER=bedrock requires the 'bedrock' extra: "
            "pip install 'cora[bedrock]'"
        ) from exc

    model_name = config.model_alias
    if not model_name.startswith("anthropic."):
        model_name = f"anthropic.{model_name}"
    return BedrockConverseModel(model_name)


def build_model_settings(
    cfg: "ReviewerConfig | None",
    *,
    max_tokens: int,
    timeout_s: float,
    extra: dict[str, Any] | None = None,
) -> "ModelSettings":
    """Build the per-call `ModelSettings` for the configured provider.

    On the default `"openai-compatible"` path this is byte-identical
    to what every call site constructed before the provider seam:
    `max_tokens` + a fixed `temperature=0.2` + `timeout`, plus whatever
    `extra` the caller threads (deep mode's `_thinking_extra_body`
    vLLM `extra_body` toggle).

    On the direct `"anthropic"` / `"bedrock"` SDK paths, `temperature`
    and `extra` are both dropped: current Claude models reject
    non-default sampling parameters with a 400, the vLLM
    `chat_template_kwargs.enable_thinking` shape is a no-op there
    (cloud Claude's adaptive-thinking defaults are already correct),
    and explicit thinking configuration isn't sent either — same
    reasoning as the dropped sampling params.
    """
    from pydantic_ai import ModelSettings

    provider = cfg.llm_provider if cfg is not None else _c.DEFAULT_LLM_PROVIDER
    if provider == "openai-compatible":
        return ModelSettings(
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=timeout_s,
            **(extra or {}),
        )
    return ModelSettings(max_tokens=max_tokens, timeout=timeout_s)


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
    `headers=` kwarg.

    Local tools in `config.local_tools` register as Pydantic-AI
    `tools=[...]` — declarations the framework introspects for
    schema. Local tools are listed before MCP toolsets, so on a
    name collision local wins (deep mode uses this so the PR-aware
    `grep_repo` / `git_show` shadow the MCP server's main-mirror copies).

    `config.provider` selects the dialect. Default
    `"openai-compatible"` is this exact path, unchanged. `"anthropic"` /
    `"bedrock"` build the model through pydantic-ai's direct SDK
    integrations instead — see `_build_anthropic_model` /
    `_build_bedrock_model`. Both require their own optional extra
    (`cora[anthropic]` / `cora[bedrock]`); the import is lazy so the
    default path never needs them installed.
    """
    # Local imports keep the heavy framework dep off the module-load
    # path for environments where `pydantic-ai` isn't installed yet
    # (the smoke test handles that case with `pytest.importorskip`).
    from pydantic_ai import Agent

    provider = config.provider or _c.DEFAULT_LLM_PROVIDER
    if provider == "openai-compatible":
        model = _build_openai_compatible_model(config)
    elif provider == "anthropic":
        model = _build_anthropic_model(config)
    elif provider == "bedrock":
        model = _build_bedrock_model(config)
    else:
        raise ValueError(
            f"unknown AgentConfig.provider: {provider!r} "
            f"(supported: {sorted(_c.SUPPORTED_LLM_PROVIDERS)})"
        )

    toolsets: list[Any] = []
    if config.mcp_servers:
        # `MCPToolset` defaults to Streamable HTTP for HTTP URLs;
        # backed by fastmcp's `Client` internally. The legacy
        # `MCPServerStreamableHTTP` class is deprecated upstream.
        from pydantic_ai.mcp import MCPToolset

        allow = config.mcp_allowed_tools
        for url, headers in config.mcp_servers:
            toolset: Any = MCPToolset(
                url,
                headers=headers or None,
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
        tools=tuple(config.local_tools),
        toolsets=toolsets or None,
    )
