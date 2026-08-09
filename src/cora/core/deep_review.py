"""Deep-mode reviewer entrypoint — multi-turn agent loop with MCP
tools (the read-tools MCP server + the actions MCP server + the
web-fetch gate) plus the
PR-aware local `grep_repo` / `git_show` handlers.

`deep_review_call` is the entry the reviewer's deep path
(`MAX_TOOL_ITERATIONS>0`, fired by the `review-deep` label) calls
into. Builds a Pydantic-AI Agent through the shared `make_review_agent`
factory, registers MCP toolsets + typed local tools, drives one
`agent.run(...)` to completion, and returns the body for downstream
leak detection + verdict parsing.

What lives here (vs the factory):
- Eager MCP connectivity probes — Pydantic-AI's `MCPToolset` opens
  the transport lazily on `agent.run`, so an unreachable optional
  server would fail the whole run mid-flight. The probes drop
  unreachable optional servers cleanly and surface a distinct
  `mcp-connect-failed` `terminated_reason` for the required
  MCP server so the check-run conclusion stays stable.
- Local-tool wrappers — `grep_repo` / `git_show` as typed async
  functions; schema is inferred from signatures rather than the
  hand-built JSON envelope the MCP server uses.
- A `FunctionToolCallEvent` stream handler that mirrors framework
  tool dispatches into `budget.add_tool_call(name)` so the
  finish-line summary + Loki dashboards see real per-tool counts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from cora.core.budget import (
    T0_COLD_START_ALLOWANCE_S,
    resolve_run_usage,
    usage_tokens,
)

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.core.mcp_sessions import McpServerSpec
    from cora.providers.git import GitProvider


# Extra grace given to the outer `asyncio.wait_for` ceiling over the
# in-loop `loop_deadline_monotonic` check. The in-loop check fires at
# node boundaries (cheap); the wait_for ceiling is the backstop for a
# single LLM/tool call that hangs through the deadline. Keeping the
# gap small (a few seconds) means a hang is force-cancelled promptly
# while a one-node overshoot still gets the clean WallTimeExceeded
# path, not the asyncio.TimeoutError path.
_WALL_TIME_WAIT_FOR_GRACE_S = 5


# Default iteration cap mirrors `config.DEFAULT_MAX_TOOL_ITERATIONS`
# (12). Overridable per call.
_DEFAULT_REQUEST_LIMIT = 12


def _thinking_extra_body(enable_thinking: bool | None = None) -> dict[str, Any]:
    """Opt-in reasoning toggle for thinking-capable local models.

    Some models (e.g. Gemma 4) gate their reasoning channel behind a
    chat-template kwarg (`enable_thinking`) that's off by default. When
    enabled, pass it through as OpenAI `extra_body` so the served model
    emits its `<think>` block (captured server-side by the reasoning
    parser, kept out of the verdict body).

    `enable_thinking` is the config-threaded flag (`cfg.enable_thinking`);
    `None` preserves the legacy call-time `AGENT_REVIEW_ENABLE_THINKING`
    env read for callers that don't pass a config yet. Inert when off —
    existing reviewer behaviour is unchanged."""
    if enable_thinking is None:
        import os

        enable_thinking = os.environ.get(
            "AGENT_REVIEW_ENABLE_THINKING", ""
        ).strip().lower() in ("1", "true", "yes")
    if enable_thinking:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    return {}


def _no_thinking_extra_body() -> dict[str, Any]:
    """The explicit reasoning-OFF chat-template kwarg.

    Distinct from `_thinking_extra_body`, which only ever sends the ON
    direction and stays silent otherwise. A template that gates on
    `enable_thinking is false` needs the key present and false — an
    absent key means "default", which for a reasoning model means ON.

    Only used by the last-resort degraded write-up turn; see
    `core.config.SPIRAL_DEGRADE_THINKING` for why that is off by
    default and bounded to a single turn."""
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _make_verdict_probe(cfg: ReviewerConfig | None) -> Callable[[str], bool]:
    """`(text) -> bool`: does this response body already carry a
    parseable verdict?

    Bound to the deployment's glyph/word vocabulary so a rebranded
    verdict line is judged by its own markers. Used to tell a turn that
    ran out of budget mid-answer (re-draw it) from one whose answer is
    complete and merely clipped in the tail (keep it)."""
    from cora.core import config as _c
    from cora.core.leak import parse_verdict_from_body

    glyphs = cfg.verdict_glyphs if cfg is not None else _c.VERDICT_GLYPHS
    words = cfg.verdict_words if cfg is not None else _c.VERDICT_WORDS

    def _probe(text: str) -> bool:
        return parse_verdict_from_body(text, glyphs=glyphs, words=words) is not None

    return _probe


def _make_pydantic_ai_local_tools(
    tool_arg_defaults: dict[str, dict[str, Any]] | None,
    *,
    git_provider: GitProvider | None = None,
    repo: str | None = None,
    cfg: ReviewerConfig | None = None,
):
    """Build typed Pydantic-AI tool wrappers around the repo-introspection
    grep_repo + git_show handlers (served via a `GitProvider`) plus, when
    `repo` is given, `read_issue` (served via `core/issue_context.py`).

    Returns a list of `Tool` instances ready to pass into the agent
    factory's `local_tools` config field. grep_repo / git_show names +
    signatures mirror the MCP server's copies so the model's learned
    tool use carries over (only the corpus differs — PR branch vs main
    mirror).

    `git_provider` defaults to `LocalGitProvider` (the in-CI checkout); a
    caller — eventually `run_review` from config — can inject another
    backend. `tool_arg_defaults` flows through here so a future caller can
    stamp e.g. `caller="cora"` onto a tool's args without the
    model needing to set it. Today `web_fetch_doc` is the only consumer of
    that pattern and it lives on the MCP side — so the hook is currently a
    no-op, reserved for when a local tool needs the same treatment.

    `read_issue` is registered only when `repo` is set (deep mode always
    has one) AND `"read_issue"` is in `cfg.local_issue_tools` (default:
    `core.config.LOCAL_ISSUE_TOOLS`, i.e. on) — an adopter can drop it
    from that set to disable the tool without touching this factory.
    """
    from pydantic_ai import Tool

    from cora.core import config as _c
    from cora.providers.git import LocalGitProvider

    provider: GitProvider = git_provider if git_provider is not None else LocalGitProvider()
    _ = tool_arg_defaults  # reserved; see docstring

    # Duplicate-call guard, scoped to this review (the factory is called
    # once per run). A model stuck re-issuing the byte-identical call —
    # observed as the same (tool, args) pair on alternating turns — gets
    # a short stub instead of the full result re-injected, so a repeat
    # loop costs tokens once, not every turn.
    seen_calls: set[tuple] = set()

    def _dedup(key: tuple) -> str | None:
        if key in seen_calls:
            return (
                f"duplicate call: {key[0]} already ran with these exact "
                "arguments in this review — its result is earlier in the "
                "conversation. Use that result, or change the arguments."
            )
        seen_calls.add(key)
        return None

    async def grep_repo(
        pattern: str,
        glob: str | None = None,
        max_count: int = 50,
        context_lines: int = 0,
        corpus: str = "repo",
    ) -> str:
        """Regex-search a corpus AT THIS PR's STATE. Two corpora:
        `corpus="repo"` (default) is the PR checkout — the PR branch
        merged onto its base — so results reflect files this PR adds,
        modifies, or deletes. `corpus="deps"` is the deployment's
        resolved dependency source (a Go module cache, vendor dir,
        node_modules, site-packages, ...), when configured — use it to
        check a third-party library's actual API at the pinned version
        instead of asserting it from memory. If no dependency corpus is
        configured, `corpus="deps"` returns a one-line message saying so
        (not an error) — don't retry it.

        Args:
            pattern: Python regular expression.
            glob: Optional fnmatch glob to restrict which paths are
                searched (repo-relative, or root-relative for
                `corpus="deps"`). A directory path (with or without
                trailing "/") searches its whole subtree.
            max_count: Max matches (default 50, cap 500).
            context_lines: Lines of context each side (0-5, default 0).
            corpus: "repo" (default, the PR checkout) or "deps" (the
                deployment's dependency-source corpus, if configured).
        """
        stub = _dedup(
            ("grep_repo", pattern, glob, max_count, context_lines, corpus)
        )
        if stub is not None:
            return stub
        return provider.grep_repo(
            {
                "pattern": pattern,
                "glob": glob,
                "max_count": max_count,
                "context_lines": context_lines,
                "corpus": corpus,
            }
        )

    async def git_show(
        ref: str = "HEAD",
        path: str | None = None,
    ) -> str:
        """Show a file's full content AT THIS PR's STATE. Use it when
        the truncated diff doesn't show enough of a changed file, or to
        read an unchanged file the PR depends on. Pass `path` (repo-
        relative) for file content; omit it for commit metadata. `ref`
        defaults to the PR head.

        Args:
            ref: Git ref (default HEAD = the PR state).
            path: Repo-relative file path. Omit for commit metadata.
        """
        stub = _dedup(("git_show", ref, path))
        if stub is not None:
            return stub
        return provider.git_show({"ref": ref, "path": path})

    async def list_files(glob: str | None = None) -> str:
        """List file PATHS at THIS PR's state. This is the tool for
        "does `<path>` exist?" — `grep_repo` searches file CONTENT, so
        an empty grep result is NOT evidence that a path is absent.
        Before writing any finding that says a file is missing, confirm
        it here.

        Args:
            glob: Optional fnmatch glob against the full repo-relative
                path — `"tests/fixtures/*"` for a subtree, `"*/conf.py"`
                for a basename anywhere, `"tests/fixtures/foo.jsonl"`
                for one exact path. Omit to list the whole repo
                (capped; the result says so when it truncates).
        """
        stub = _dedup(("list_files", glob))
        if stub is not None:
            return stub
        return provider.list_files({"glob": glob})

    tools = [
        Tool(grep_repo, name="grep_repo"),
        Tool(git_show, name="git_show"),
    ]

    repo_tool_names = (
        cfg.local_repo_tools if cfg is not None else _c.LOCAL_REPO_TOOLS
    )
    if "list_files" in repo_tool_names:
        tools.append(Tool(list_files, name="list_files"))

    issue_tools = cfg.local_issue_tools if cfg is not None else _c.LOCAL_ISSUE_TOOLS
    if repo and "read_issue" in issue_tools:
        from cora.core.issue_context import local_read_issue

        async def read_issue(number: int, repo_hint: str | None = None) -> str:
            """Fetch a GitHub issue from THIS repository — title, state,
            body, and its earliest comments (bounded, same trust-wrapped
            shape as the pre-fetched linked-issue block, if you saw one
            above). Use it when the PR references an issue the
            pre-fetch's 2-issue cap or reference parsing didn't catch —
            a third linked issue, or one mentioned without a closing
            keyword.

            Args:
                number: Issue number, e.g. 42 for `#42`.
                repo_hint: Optional `owner/repo` sanity check. Only this
                    review's own repo is supported — anything else is
                    rejected, never fetched. Omit it; it defaults to
                    this repo.
            """
            stub = _dedup(("read_issue", number, repo_hint))
            if stub is not None:
                return stub
            return local_read_issue(
                {"number": number, "repo": repo_hint}, repo=repo, cfg=cfg
            )

        tools.append(Tool(read_issue, name="read_issue"))

    return tools


def _loaded_tool_names(
    allowed_tools: set[str],
    *,
    # The read-tools MCP server is optional now, so its names are only
    # "loaded" when one was actually configured and probed. Defaults
    # True so existing callers keep counting them.
    read_enabled: bool = True,
    actions_enabled: bool = False,
    web_enabled: bool = False,
    # True when at least one non-legacy `MCP_SERVERS` session opened.
    # Same session-level granularity as the three flags above — we
    # don't introspect which tool came from which extra session
    # (`probe_mcp_server` only returns pass/fail), so any opened extra
    # session toggles the whole `extra_tools` allow-set on, mirroring
    # how one `mcp-actions` session toggles the whole `ACTION_TOOLS` set.
    extra_enabled: bool = False,
    read_tools: set[str] | frozenset[str] | None = None,
    local_repo_tools: set[str] | frozenset[str] | None = None,
    local_issue_tools: set[str] | frozenset[str] | None = None,
    action_tools: set[str] | frozenset[str] | None = None,
    web_tools: set[str] | frozenset[str] | None = None,
    extra_tools: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return the tool names actually exposed for this review topology.

    `allowed_tools` is the broad policy allowlist, not the loaded
    palette: optional MCP servers may be absent, while local repo tools
    (and `read_issue`) are always registered in-process. Keep the
    comment footer's "unused" denominator tied to the successfully
    opened server classes so it remains useful as a tool-loading signal.
    """
    from cora.core import config as _c

    read_set = set(_c.READ_TOOLS if read_tools is None else read_tools)
    local_set = set(
        _c.LOCAL_REPO_TOOLS if local_repo_tools is None else local_repo_tools
    )
    issue_set = set(
        _c.LOCAL_ISSUE_TOOLS if local_issue_tools is None else local_issue_tools
    )
    action_set = set(_c.ACTION_TOOLS if action_tools is None else action_tools)
    web_set = set(_c.WEB_TOOLS if web_tools is None else web_tools)
    extra_set = set(extra_tools or ())

    loaded = set(allowed_tools) & (local_set | issue_set)
    if read_enabled:
        loaded |= set(allowed_tools) & read_set
    if actions_enabled:
        loaded |= set(allowed_tools) & action_set
    if web_enabled:
        loaded |= set(allowed_tools) & web_set
    if extra_enabled:
        loaded |= set(allowed_tools) & extra_set
    return sorted(loaded)


# `_probe_mcp_server` moved to `cora.core.mcp_probe` as the
# public `probe_mcp_server` — both reviewer (this file + continuation)
# and triage now import the single copy. Keep the
# legacy name as a thin alias so existing intra-module references
# don't need a sweeping rename in the same change.
from cora.core.mcp_probe import (  # noqa: E402 — deliberate: the
    # comment above explains why this alias sits here rather than at
    # the top of the module.
    probe_mcp_server as _probe_mcp_server,
)


async def deep_review_call(
    *,
    endpoint_base_url: str,
    llm_gateway_key: str,
    model_alias: str,
    system_prompt: str,
    initial_user_prompt: str,
    budget,  # Budget — loose typing avoids cross-module dep
    timeout_s: int,
    pr_number: str,
    repo: str,
    # Required read-tools MCP server.
    mcp_url: str,
    mcp_headers: dict[str, str],
    # Optional actions MCP server (observe-only write-intent).
    mcp_actions_url: str | None = None,
    mcp_actions_headers: dict[str, str] | None = None,
    # Optional web-fetch gate (live web fetch).
    web_fetch_url: str | None = None,
    web_fetch_headers: dict[str, str] | None = None,
    # Generic extra MCP sessions (from `MCP_SERVERS`) — appended after
    # the three named slots above; see `cora.core.mcp_sessions`.
    extra_sessions: Sequence[McpServerSpec] = (),
    # Allowlist filter applied to MCP toolsets.
    allowed_tools: set[str],
    # Tool-arg defaults (e.g. {"web_fetch_doc": {"caller": "cora"}}).
    tool_arg_defaults: dict[str, dict[str, Any]] | None = None,
    # Iteration cap. Mirrors `MAX_TOOL_ITERATIONS` env var.
    max_iterations: int = _DEFAULT_REQUEST_LIMIT,
    # Monotonic-clock deadline (from `time.monotonic()`). When set,
    # the inner loop checks it at every node boundary and raises
    # `WallTimeExceeded` once crossed; an outer `asyncio.wait_for`
    # caps the whole run at `deadline + grace` so a single hung call
    # can't sit past the deadline indefinitely. None disables both
    # checks (back-compat for callers that don't enforce wall time
    # yet — triage, tests).
    loop_deadline_monotonic: float | None = None,
    # Optional push-context refresher (constructed ONCE per review in
    # `agent_review.amain` so dedupe state carries across T0 → T1).
    # When set, `iter_with_turn_logging` polls it at node boundaries
    # and grafts new context into the next ModelRequest. None disables.
    context_refresher=None,
    # `gha_log` writes one logfmt-parseable line per event. The
    # caller's wrapper fans the same line into both
    # `print("::notice::" + line)` (GHA workflow UI) and
    # `loki_push(line, labels={...})` (Loki stream) so dashboards +
    # workflow logs see the same data without two formats to keep in
    # sync. See the orchestrator entrypoint's `_iter_log` wrapper for
    # the reference wiring.
    gha_log: Callable[[str], None] = print,
    # The run's ReviewerConfig — threaded into Deps (tools/hooks) and
    # the thinking toggle. None keeps the legacy behaviour exactly
    # (Deps default-constructs a mirror config; the thinking toggle
    # falls back to its call-time env read).
    cfg: ReviewerConfig | None = None,
    # Repo-introspection backend for the local grep_repo / git_show
    # tools. None → LocalGitProvider (the in-CI checkout), the
    # default behaviour.
    git_provider: GitProvider | None = None,
) -> tuple[str, str | None, list[str], list]:
    """Drive one deep-mode review run. Returns
    `(final_body, terminated_reason, tools_available, messages)` so
    the caller's finalize / check-run / Loki-summary path can read
    the first three unchanged, and the optional T1-continuation
    dispatcher can pick up `messages` when `terminated_reason`
    indicates a wall-hit.

    `messages` is the full message history captured via
    `agent_run.all_messages()` — populated regardless of completion
    state so a caller can dump it for trace debugging. On a wall-hit
    (`max_iterations` / future `wall_time` / `budget_exhausted`)
    this is the working state the T1 continuation hands forward to
    the T1 endpoint. On clean completion `messages` reflects the final
    conversation; on a probe-failure short-circuit it's empty.

    Behaviour summary:
      - Free-form text response (`output_type=str`) — `leak.py`
        parses it
      - MCP tool surface filtered to `allowed_tools`
      - Local grep_repo + git_show shadow MCP-served copies
      - Optional MCP servers (actions + web-fetch-gate) best-effort
        connect; failure drops their tools but doesn't fail the run
      - Iteration cap via `UsageLimits(request_limit=max_iterations)` —
        framework returns a `UsageLimitExceeded` exception which we
        map to `terminated_reason="max_iterations"`
      - MCP is optional. An empty `mcp_url` self-disarms: no probe, no
        toolset, and the loop runs on the in-process repo tools alone.
        A *configured* server is probed up front and failure short-
        circuits to `terminated_reason="mcp-connect-failed"` so the
        caller emits the existing "skipped (MCP server unreachable)"
        comment + `cancelled` check-run
      - Any other framework error returns
        `terminated_reason="agent-loop-errored: <exc>"` and the caller
        routes through the no-final-body skip path
      - Uncommitted-draw re-draw (`cfg.spiral_redraw`, default-ON): a
        turn that hits the completion ceiling with no tool call and no
        parseable verdict is re-sent ONCE, identical payload, inside the
        same agent context. Covers both the thinking-only shape (which
        pydantic-ai raises on) and the truncated-prose shape (which it
        doesn't). Logs `event=spiral_redraw outcome=…`.
      - Reasoning-spiral recovery (opt-in via `cfg.spiral_recovery`,
        default-OFF): rung two. When the re-draw is skipped or comes
        back empty, ONE bounded recovery `agent.run` re-seeded with the
        captured partial reasoning + a "conclude now" directive (see
        `cora.core.spiral`).
      - Salvage: with both rungs spent, a turn that produced any visible
        text returns that text — a truncated review, which is what this
        path produced before either rung existed. Only a thinking-only
        turn falls through to the `agent-loop-errored:
        spiral-redraw-exhausted` outcome — which the tier dispatcher
        escalates to T1 (`cfg.spiral_escalation`) before soft-failing.
    """
    from pydantic_ai import ModelSettings, UsageLimits
    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    from cora.core.agent import (
        AgentConfig,
        Deps,
        initial_tool_contract_satisfied,
        make_review_agent,
    )

    # No MCP URL configured → don't probe, don't register a toolset.
    # Deep mode runs on the in-process repo tools alone: a smaller tool
    # surface, but a working agent loop, which is what an adopter with
    # no MCP server should get instead of a review that always skips.
    #
    # Configured-but-unreachable is still fatal. Failure returns the
    # distinct `mcp-connect-failed` terminated_reason so the caller takes
    # the existing "skipped (MCP server unreachable)" path — a review
    # that quietly drops the tools you asked for is worse than one that
    # asks to be retried, and it keeps the Loki finish-line buckets +
    # verdict-line text stable.
    mcp_url = (mcp_url or "").strip()
    if not mcp_url:
        gha_log(
            "no MCP server configured (MCP_URL unset) — deep mode running "
            "on local repo tools only"
        )

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
        configured_sessions, probe=_probe_mcp_server, log=gha_log
    )
    if opened is None:
        return "", "mcp-connect-failed", [], []
    mcp_servers, sessions_opened = opened
    read_enabled = "mcp" in sessions_opened
    actions_enabled = "actions" in sessions_opened
    web_enabled = "web-fetch" in sessions_opened
    extra_enabled = bool(set(sessions_opened) - {"mcp", "actions", "web-fetch"})

    # Pydantic-AI's `Agent` enforces unique tool names across local +
    # MCP. Strip the locally-served names from the MCP allowlist so
    # the MCP server's `grep_repo` / `git_show` get filtered out at
    # toolset registration time, before they collide with the typed
    # local `Tool` instances we register via `tools=`. The model sees
    # one of each name; the local (PR-aware) version wins.
    mcp_allowed_for_filter = allowed_tools - {"grep_repo", "git_show"}

    config = AgentConfig(
        endpoint_base_url=endpoint_base_url,
        api_key=llm_gateway_key,
        model_alias=model_alias,
        system_prompt=system_prompt,
        mcp_servers=mcp_servers,
        mcp_allowed_tools=mcp_allowed_for_filter,
        local_tools=_make_pydantic_ai_local_tools(
            tool_arg_defaults, git_provider=git_provider, repo=repo, cfg=cfg
        ),
        # Tool-call retries on transient MCP failures.
        retries=1,
        session_id=cfg.session_header if cfg is not None else None,
        require_initial_tool_call=(
            cfg.require_initial_tool_call if cfg is not None else False
        ),
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

    # Structured per-turn logging shared with the T1 continuation.
    # `iter_with_turn_logging` drives node iteration, emits the
    # `T0 turn N …` lines, bumps the tool-call counter, and invokes
    # `on_tool_call` (Budget hook). `agent.iter()` doesn't accept an
    # `event_stream_handler` — only `agent.run()` does — so all
    # event extraction lives in the node loop.
    from cora.core.loop_logging import (
        PerCallTimeoutExceeded,
        ReasoningSpiralDetected,
        StreamStallDetected,
        WallTimeExceeded,
        iter_with_turn_logging,
        log_spiral_redraw,
        log_wall_hit,
    )

    tool_call_counter: dict[str, int] = {}
    turn_counter: list[int] = [0]

    # Loaded tool palette for the comment/footer "unused" denominator.
    # Per-tool usage still comes from `budget.tool_calls`; do not
    # collapse this to only the tools that fired, or the footer stops
    # being a useful signal that MCP tools were available.
    tools_available = _loaded_tool_names(
        allowed_tools,
        read_enabled=read_enabled,
        actions_enabled=actions_enabled,
        web_enabled=web_enabled,
        extra_enabled=extra_enabled,
        read_tools=cfg.read_tools if cfg is not None else None,
        local_repo_tools=cfg.local_repo_tools if cfg is not None else None,
        local_issue_tools=cfg.local_issue_tools if cfg is not None else None,
        action_tools=cfg.action_tools if cfg is not None else None,
        web_tools=cfg.web_tools if cfg is not None else None,
        extra_tools=cfg.extra_tools if cfg is not None else None,
    )
    # Conversation history accumulator. Populated from `agent_run`
    # whether the run completes cleanly or trips `UsageLimitExceeded`
    # — the T1 continuation dispatcher reads this on a wall-hit.
    messages: list = []
    contract_armed = config.require_initial_tool_call
    if contract_armed and not tools_available:
        gha_log(
            f"agent_review iter pr_number={pr_number} phase=T0 "
            "event=initial_tool_contract outcome=no_tools first_tool=none"
        )
        return "", "required-tool-unavailable", tools_available, messages

    # Terminal reason hoisted to outer scope so the inner wall-hit /
    # iteration-cap handlers can set it BEFORE the `async with`
    # cleanup runs. When wait_for cancels the inner coroutine, the
    # framework's cleanup of agent.iter() + agent context can raise
    # secondary exceptions (e.g. CancelledError from MCP transport
    # teardown, str() = "") that would otherwise be caught by the
    # outer `except Exception` and overwrite our legitimate
    # `wall_time` reason with `agent-loop-errored:`. Observed in
    # early runs — fixed alongside.
    early_terminated_reason: str | None = None

    # Per-call output budget — must fit the model's `<think>` trace AND the
    # turn's text/tool-call. See `_c.DEEP_MAX_OUTPUT_TOKENS`.
    from cora.core import config as _c

    deep_max_tokens = (
        cfg.deep_max_output_tokens if cfg is not None else _c.DEEP_MAX_OUTPUT_TOKENS
    )

    # Spiral-recovery flag (default-OFF). When off, nothing below this
    # changes behaviour: the inner UnexpectedModelBehavior handler only
    # intervenes when the flag is on AND the captured messages show a
    # thinking-only spiral; otherwise it re-raises so the outer handler
    # maps to `agent-loop-errored` exactly as today.
    spiral_on = cfg is not None and cfg.spiral_recovery
    # Messages snapshotted at the spiral trip — the working state a
    # bounded recovery turn re-seeds from. None = no spiral detected.
    spiral_messages: list | None = None

    # Uncommitted-draw re-draw (default-ON killswitch). Arming detection
    # is the presence of `verdict_probe`: a turn that hits the completion
    # ceiling with no tool call and no parseable verdict raises
    # `ReasoningSpiralDetected` at the response boundary instead of being
    # carried forward as a dead turn.
    redraw_on = cfg is None or cfg.spiral_redraw
    verdict_probe = _make_verdict_probe(cfg) if redraw_on else None
    # Set by the detection handler to the turn number that spiralled;
    # the re-draw below runs inside the still-open agent context.
    redraw_from_turn: int | None = None
    # Visible text an aborted stream had already produced. Seeds the
    # re-draw so it resumes instead of restarting blind; "" keeps the
    # re-draw a pure re-send.
    redraw_prior_text: str = ""
    stream_on = cfg is not None and cfg.stream_detection

    result = None
    try:
        # Open the agent context once so all MCP toolset sessions stay
        # alive for the whole run. `async with agent` is the framework's
        # supported way to keep MCP transports up across multiple
        # `agent.run` calls — even though we only run once here, this
        # gives us a single connect/disconnect cycle per review.
        async with agent:
            # `agent.iter()` exposes the run as an async-iterable of
            # graph nodes so we can (a) capture `all_messages()` even
            # when `UsageLimitExceeded` fires inside the loop and
            # (b) emit per-turn structured log lines via
            # `iter_with_turn_logging`.
            async with agent.iter(
                initial_user_prompt,
                deps=deps,
                model_settings=ModelSettings(
                    # Per-call output budget — absorbs the model's
                    # `<think>...</think>` block AND the turn's text /
                    # tool-call, bounded so one draw always finishes inside
                    # `per_call_timeout_s` (see `_c.DEEP_MAX_OUTPUT_TOKENS`).
                    # An over-long draw therefore ends as `finish_reason=
                    # 'length'` — a signal the loop re-draws on — instead of
                    # a call the timeout discards mid-generation.
                    # Config-threaded; T1 (continuation.py) uses the same cap.
                    max_tokens=deep_max_tokens,
                    temperature=0.2,
                    timeout=timeout_s,
                    **_thinking_extra_body(
                        cfg.enable_thinking if cfg is not None else None
                    ),
                ),
                usage_limits=UsageLimits(request_limit=max_iterations),
            ) as agent_run:
                # Outer ceiling — caps the entire iter() at deadline +
                # small grace. Catches the pathology where a single
                # LLM/tool call hangs past the in-loop deadline check
                # (which only fires between nodes). The in-loop check
                # still produces the cleaner WallTimeExceeded path
                # most of the time; this is just the backstop.
                if loop_deadline_monotonic is not None:
                    remaining = loop_deadline_monotonic - time.monotonic()
                    # Pre-pad the outer wait_for cap by the worst-case
                    # injection extension budget so a mid-run deadline
                    # bump (push context injection) doesn't get
                    # back-stopped by the wait_for firing early. The
                    # in-loop check is the primary deadline trip; the
                    # outer wait_for stays generous so it only kicks
                    # in for truly hung calls.
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

                # When the refresher pushes new context, the in-loop
                # `iter_with_turn_logging` mirror is bumped directly
                # via its internal `_current_deadline`. We also expose
                # an extend callback here that's a no-op on the
                # caller side — the outer `wait_for` was pre-padded
                # above for the cap, and there's no separate caller-
                # side deadline state to mutate in T0's shape. The
                # callback existing (rather than being None) is what
                # tells the iter helper that extensions are wired.
                def _extend_t0_deadline(_seconds: float) -> None:
                    return None

                try:
                    inner_coro = iter_with_turn_logging(
                        agent_run,
                        phase="T0",
                        pr_number=pr_number,
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                        # Budget.add_tool_call also bumps `iterations`,
                        # which `reason_if_over` uses for the
                        # max-iterations cap.
                        on_tool_call=budget.add_tool_call,
                        loop_deadline_monotonic=loop_deadline_monotonic,
                        # Per-turn hard cap. `ModelSettings(timeout=…)`
                        # below becomes httpx's read_timeout for
                        # streaming responses and only fires on
                        # inactivity — useless when vLLM is slowly
                        # but steadily emitting `<think>` tokens
                        # (observed: 771 tokens in 324s without a
                        # gap big enough to trip read_timeout=180).
                        per_call_timeout_s=float(timeout_s),
                        # T0's first call may hit a cold scale-to-zero backend
                        # mid-warmup; give it a one-time allowance so the
                        # cold-start doesn't eat the inference window (later
                        # calls hit the warm backend).
                        first_call_extra_timeout_s=float(T0_COLD_START_ALLOWANCE_S),
                        context_refresher=context_refresher,
                        extend_deadline_fn=(
                            _extend_t0_deadline if context_refresher is not None else None
                        ),
                        verdict_probe=verdict_probe,
                        # Streaming detection (opt-in). Off = the helper
                        # awaits each call whole, exactly as before.
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
                    messages = list(agent_run.all_messages())
                except UsageLimitExceeded as exc:
                    # Iteration cap tripped before a terminal assistant
                    # message. Snapshot the working state so the T1
                    # continuation dispatcher can hand it forward.
                    log_wall_hit(
                        phase="T0",
                        pr_number=pr_number,
                        terminated_reason="max_iterations",
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                    )
                    # Bare `print` so it lands as a `::warning::` in the
                    # GHA UI (elevated to top-of-summary annotation),
                    # not a parseable `agent_review iter …` event.
                    print(f"::warning::deep mode hit iteration cap: {exc}")
                    messages = list(agent_run.all_messages())
                    early_terminated_reason = "max_iterations"
                except PerCallTimeoutExceeded as exc:
                    # A single turn (model call) blew past the per-call
                    # cap. Operationally distinct from wall_time — this
                    # means backend contention or a slow inference path,
                    # not "we used up the overall budget". T1 dispatch
                    # still kicks in (T1 runs on a larger endpoint with more
                    # KV-cache headroom; the same prompt may finish there).
                    log_wall_hit(
                        phase="T0",
                        pr_number=pr_number,
                        terminated_reason="per_call_timeout",
                        turn_counter=turn_counter,
                        tool_call_counter=tool_call_counter,
                        log=gha_log,
                    )
                    print(
                        f"::warning::deep mode hit per-call timeout: "
                        f"turn {exc.turn} took {exc.elapsed_s:.1f}s "
                        f"(cap {exc.cap_s:.0f}s)"
                    )
                    try:
                        messages = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        messages = []
                    early_terminated_reason = "per_call_timeout"
                except (ReasoningSpiralDetected, StreamStallDetected) as exc:
                    # Either the turn came back empty of anything usable,
                    # or (streaming) it was aborted mid-flight for
                    # reasoning past its budget or for going silent.
                    # Both leave the history at the payload we'd re-send,
                    # so both take the re-draw below — the agent context
                    # is still open there, so MCP isn't rebuilt.
                    if isinstance(exc, StreamStallDetected):
                        print(
                            f"::warning::deep mode turn {exc.turn} stalled "
                            f"({exc.idle_s:.0f}s with no delta after "
                            f"{exc.streamed_tokens} tokens) — re-drawing once"
                        )
                    else:
                        print(
                            f"::warning::deep mode turn {exc.turn} hit the "
                            f"completion ceiling without committing "
                            f"({exc.out_tokens} out tokens) — re-drawing once"
                        )
                    try:
                        messages = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        messages = []
                    redraw_from_turn = exc.turn
                    redraw_prior_text = getattr(exc, "partial_text", "") or ""
                except (TimeoutError, WallTimeExceeded) as exc:
                    # Wall-time guard tripped. Same downstream shape as
                    # max_iterations — snapshot messages so T1
                    # continuation can pick up the trajectory, then
                    # return with terminated_reason="wall_time" for the
                    # finalize / dashboard buckets to split clean
                    # iteration caps from wall-time caps.
                    overshoot = (
                        getattr(exc, "overshoot_s", None)
                        if isinstance(exc, WallTimeExceeded)
                        else None
                    )
                    log_wall_hit(
                        phase="T0",
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
                    print(f"::warning::deep mode hit wall-time guard ({detail})")
                    try:
                        messages = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        messages = []
                    early_terminated_reason = "wall_time"
                except UnexpectedModelBehavior as exc:
                    # pydantic-ai's own completion-ceiling raise. The
                    # boundary check above already catches the common
                    # thinking-only shape; this handler is what's left —
                    # chiefly a tool call truncated mid-arguments
                    # (`IncompleteToolCall`), which the boundary check
                    # deliberately doesn't claim (it has a tool call).
                    # Same root cause, so it takes the same re-draw.
                    try:
                        snapshot = list(agent_run.all_messages())
                    except Exception:  # noqa: BLE001
                        snapshot = []
                    from cora.core import spiral as _spiral
                    if redraw_on and _spiral.is_completion_ceiling_exception(exc):
                        print(
                            f"::warning::deep mode hit the completion ceiling "
                            f"mid-commit ({exc!s:.120}) — re-drawing once"
                        )
                        messages = snapshot
                        redraw_from_turn = turn_counter[0]
                    elif spiral_on and _spiral.is_reasoning_spiral(snapshot):
                        # Bounded conclude-now recovery (opt-in) — the
                        # second rung, reached when the re-draw is off.
                        spiral_messages = snapshot
                        messages = snapshot
                    else:
                        raise

            # ── Uncommitted-draw re-draw ─────────────────────────────
            # Still inside `async with agent`, so the MCP sessions the
            # main loop opened are reused rather than rebuilt. `agent.run`
            # with the committed prefix and NO new user prompt re-sends
            # the exact payload that spiralled — the point is to re-roll
            # the sampler, not to ask a different question — and drives
            # its own tool loop, so a re-draw that decides to call a tool
            # still finishes the review.
            if redraw_from_turn is not None and result is None:
                from cora.core import spiral as _spiral

                prefix = _spiral.committed_prefix(messages)
                if not prefix:
                    log_spiral_redraw(
                        phase="T0",
                        pr_number=pr_number,
                        turn=redraw_from_turn,
                        outcome="skipped",
                        log=gha_log,
                        reason="no_committed_prefix",
                    )
                else:
                    try:
                        redraw = await agent.run(
                            # A pure re-send unless an aborted stream
                            # left visible text — then resume from it
                            # rather than throwing the work away.
                            (
                                _spiral.build_resume_leadin(redraw_prior_text)
                                if redraw_prior_text.strip()
                                else None
                            ),
                            message_history=prefix,
                            deps=deps,
                            model_settings=ModelSettings(
                                max_tokens=deep_max_tokens,
                                temperature=0.2,
                                timeout=timeout_s,
                                **_thinking_extra_body(
                                    cfg.enable_thinking if cfg is not None else None
                                ),
                            ),
                            # Whatever iteration budget the main loop
                            # left; the re-draw is a continuation of the
                            # same review, not a fresh allowance.
                            usage_limits=UsageLimits(
                                request_limit=max(1, max_iterations - redraw_from_turn)
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        redraw_outcome = (
                            "spiralled_again"
                            if _spiral.is_completion_ceiling_exception(exc)
                            else "errored"
                        )
                        log_spiral_redraw(
                            phase="T0",
                            pr_number=pr_number,
                            turn=redraw_from_turn,
                            outcome=redraw_outcome,
                            log=gha_log,
                            detail=f'"{exc!s:.120}"',
                        )
                        # Last resort. The payload has now spiralled
                        # twice, so the model has shown that on THIS
                        # draw its reasoning doesn't terminate. One
                        # bounded write-up turn with reasoning
                        # mechanically off — `request_limit=1` so it
                        # cannot make an un-reasoned tool decision, and
                        # seeded with the conclude-now directive so it
                        # commits what it already worked out. Opt-in.
                        if (
                            redraw_outcome == "spiralled_again"
                            and cfg is not None
                            and cfg.spiral_degrade_thinking
                        ):
                            try:
                                degraded = await agent.run(
                                    _spiral.build_recovery_leadin(
                                        _spiral.extract_partial_reasoning(
                                            messages,
                                            char_cap=(
                                                cfg.spiral_recovery_reasoning_char_cap
                                            ),
                                        )
                                    ),
                                    message_history=prefix,
                                    deps=deps,
                                    model_settings=ModelSettings(
                                        max_tokens=(
                                            cfg.spiral_recovery_max_output_tokens
                                        ),
                                        temperature=0.2,
                                        timeout=timeout_s,
                                        **_no_thinking_extra_body(),
                                    ),
                                    usage_limits=UsageLimits(request_limit=1),
                                )
                            except Exception as degrade_exc:  # noqa: BLE001
                                log_spiral_redraw(
                                    phase="T0",
                                    pr_number=pr_number,
                                    turn=redraw_from_turn,
                                    outcome="degraded_failed",
                                    log=gha_log,
                                    detail=f'"{degrade_exc!s:.120}"',
                                )
                            else:
                                result = degraded
                                try:
                                    messages = list(degraded.all_messages())
                                except Exception:  # noqa: BLE001
                                    pass
                                log_spiral_redraw(
                                    phase="T0",
                                    pr_number=pr_number,
                                    turn=redraw_from_turn,
                                    outcome="recovered_degraded",
                                    log=gha_log,
                                )
                    else:
                        redraw_messages = list(redraw.all_messages())
                        # Fold the re-draw's own dispatches into the
                        # shared accounting — `agent.run` drives its loop
                        # outside `iter_with_turn_logging`, so nothing
                        # else counts them.
                        for name in _spiral.tool_call_names(
                            redraw_messages, start=len(prefix)
                        ):
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
                            phase="T0",
                            pr_number=pr_number,
                            turn=redraw_from_turn,
                            outcome="recovered",
                            log=gha_log,
                            redraw_tool_calls=len(
                                _spiral.tool_call_names(
                                    redraw_messages, start=len(prefix)
                                )
                            ),
                        )
    except Exception as exc:  # noqa: BLE001
        # If a wall_time / max_iterations trip already set the reason
        # inside the inner handler, preserve it — the framework's
        # cleanup path (cancelling agent.iter() + closing MCP
        # transports after wait_for fires) routinely raises secondary
        # exceptions like CancelledError that have `str(exc) == ""`
        # and would otherwise overwrite our legitimate reason with
        # `agent-loop-errored:` (an empty-string reason trail seen
        # in early runs). `{exc!r}` also surfaces the class so
        # genuine errors don't render as bare `agent-loop-errored:`.
        if early_terminated_reason is None:
            return "", f"agent-loop-errored: {exc!r}", tools_available, messages
        gha_log(
            f"agent context cleanup raised after {early_terminated_reason} "
            f"trip — suppressing: {exc!r}"
        )

    # Rung two. A re-draw that was skipped (nothing committed to re-send)
    # or came back empty hands the working state to the bounded
    # conclude-now recovery, when the deployment has opted into it.
    if (
        redraw_from_turn is not None
        and result is None
        and spiral_on
        and spiral_messages is None
    ):
        from cora.core import spiral as _spiral

        if _spiral.is_reasoning_spiral(messages):
            spiral_messages = messages

    # Bounded reasoning-spiral recovery (deep). When the inner handler
    # detected a thinking-only spiral, re-run the SAME agent ONCE with the
    # captured working state carried forward + a "conclude now" lead-in
    # seeded with the partial-reasoning tail (reasoning stays ON, output
    # cap is the tight `spiral_recovery_max_output_tokens`). On success the
    # recovered body feeds the normal finalize / verdict path below. On any
    # failure (another spiral, an MCP re-connect blip, …) we fall through
    # to the existing `agent-loop-errored` soft-fail — no regression. This
    # runs only when the flag is on AND a spiral was detected, so the
    # flag-off / no-spiral paths are untouched.
    if spiral_messages is not None and result is None:
        from cora.core import spiral as _spiral

        gha_log(
            f"deep mode hit reasoning spiral (PR #{pr_number}) — "
            "attempting one bounded recovery turn"
        )
        leadin = _spiral.build_recovery_leadin(
            _spiral.extract_partial_reasoning(
                spiral_messages,
                char_cap=cfg.spiral_recovery_reasoning_char_cap,
            )
        )
        try:
            async with agent:
                recovery = await agent.run(
                    leadin,
                    message_history=spiral_messages,
                    deps=deps,
                    model_settings=ModelSettings(
                        max_tokens=cfg.spiral_recovery_max_output_tokens,
                        temperature=0.2,
                        timeout=timeout_s,
                        **_thinking_extra_body(
                            cfg.enable_thinking if cfg is not None else None
                        ),
                    ),
                )
            result = recovery
            try:
                messages = list(recovery.all_messages())
            except Exception:  # noqa: BLE001
                messages = spiral_messages
        except Exception as exc:  # noqa: BLE001 — recovery is best-effort
            gha_log(
                f"deep mode spiral recovery failed (PR #{pr_number}) — "
                f"falling back to soft-fail: {exc!r}"
            )
            return (
                "",
                f"agent-loop-errored: spiral-recovery-failed {exc!r}",
                tools_available,
                messages,
            )

    # Re-draw exhausted (and any bounded recovery declined or failed).
    # Fall back to whatever the spiralled turn managed to say: a review
    # truncated mid-sentence is what this path produced before the
    # re-draw existed, and it beats dropping the review entirely. Only a
    # thinking-only turn has nothing to salvage.
    if redraw_from_turn is not None and result is None and early_terminated_reason is None:
        from cora.core import spiral as _spiral

        salvage = _spiral.extract_final_text(messages)
        if salvage.strip():
            if contract_armed and not initial_tool_contract_satisfied(
                messages, gha_log=gha_log, pr_number=pr_number, phase="T0"
            ):
                return "", "required-tool-unhonored", tools_available, messages
            gha_log(
                f"deep mode re-draw did not recover (PR #{pr_number}) — "
                f"posting the truncated turn ({len(salvage)} chars)"
            )
            return salvage, None, tools_available, messages
        return (
            "",
            "agent-loop-errored: spiral-redraw-exhausted",
            tools_available,
            messages,
        )

    if early_terminated_reason is not None:
        return "", early_terminated_reason, tools_available, messages

    run_usage = resolve_run_usage(result)
    try:
        budget.add_usage(_PydanticAIUsageAdapter(run_usage))
    except Exception as exc:  # noqa: BLE001
        gha_log(f"pydantic-ai usage adapter failed: {exc}")

    # Cross-check the event-stream counter against `RunUsage.tool_calls`
    # — if they diverge it points to events we're missing (output-tool
    # vs function-tool kinds, or the framework batching internally).
    try:
        framework_tool_total = int(getattr(run_usage, "tool_calls", 0) or 0)
        observed_tool_total = sum(tool_call_counter.values())
        gha_log(
            f"deep tool-call observability: framework={framework_tool_total} "
            f"event_stream={observed_tool_total} "
            f"per_tool={tool_call_counter}"
        )
    except Exception:  # noqa: BLE001
        pass

    from cora.core.litellm_capture import (
        drain_captured_headers,
        resolve_backend_attribution,
    )
    captured = drain_captured_headers()
    budget.record_litellm_headers(captured)
    budget.set_resolved_model(
        resolve_backend_attribution(
            captured, result, fallback="unknown (no header or body model)"
        )
    )

    body = result.output if isinstance(result.output, str) else str(result.output)
    if contract_armed and not initial_tool_contract_satisfied(
        messages, gha_log=gha_log, pr_number=pr_number, phase="T0"
    ):
        return "", "required-tool-unhonored", tools_available, messages
    return body, None, tools_available, messages


class _PydanticAIUsageAdapter:
    """Adapt Pydantic-AI's `RunUsage` shape (`input_tokens` /
    `output_tokens` / `total_tokens`, with the legacy `request_tokens` /
    `response_tokens` aliases as fallback) to the OpenAI-shaped object
    `Budget.add_usage` expects (`prompt_tokens` / `completion_tokens` /
    `total_tokens`).

    Duplicated from `quick_review.py` to keep this module self-
    contained; the field-name reconciliation now lives in
    `budget.usage_tokens` so the three copies cannot drift again."""

    def __init__(self, pa_usage):
        self.prompt_tokens = usage_tokens(pa_usage, "input_tokens", "request_tokens")
        self.completion_tokens = usage_tokens(pa_usage, "output_tokens", "response_tokens")
        self.total_tokens = usage_tokens(pa_usage, "total_tokens")
