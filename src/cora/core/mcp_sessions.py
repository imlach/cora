"""Generic multi-session MCP wiring.

Historically deep mode hand-rolled three named slots — the read-tools
MCP server (`MCP_URL`), the actions server (`MCP_ACTIONS_URL`), and the
web-fetch gate (`WEB_FETCH_GATE_URL`) — as three near-identical probe/open
blocks duplicated across `deep_review.py`, `continuation.py`, and
`t2_dispatch.py`. `McpServerSpec` is the common shape all three (plus any
`MCP_SERVERS`-configured extra) normalize into; `compose_mcp_sessions` +
`open_mcp_sessions` are the shared normalize/probe pipeline every dispatch
site now iterates over instead of hand-rolling its own.

`MCP_SERVERS` (env, JSON array) appends arbitrary extra sessions onto the
three named slots — a deployment wiring a fourth MCP server (a second
knowledge base, a different action surface) doesn't need a new named env
var + a new hand-rolled branch anymore. Each entry:

    {"name": "docs2", "url": "https://mcp.example/docs", "token_env": "DOCS2_TOKEN", "required": false}

`token_env` is env-var INDIRECTION, never a literal token — the JSON
value is the NAME of an environment variable, resolved once here. This
keeps `MCP_SERVERS` itself safe to log or paste into a workflow file: it
never carries a secret, only a pointer to one. A `token_env` that names a
var which isn't actually set is treated as tokenless (with a warning) —
the session still opens, just without an `Authorization` header, matching
how an unauthenticated server is configured today (no token slot at all).
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora.config import ReviewerConfig

# `(url, headers, name, log) -> bool` — the shape of `mcp_probe.probe_mcp_server`.
# Threaded as a parameter (not imported directly) so callers keep using
# their own module-level alias — `deep_review.py` / `continuation.py` both
# expose `_probe_mcp_server` specifically so tests can monkeypatch it;
# importing the real function here would bypass that seam.
ProbeFn = Callable[[str, Mapping[str, str], str, Callable[[str], None]], Awaitable[bool]]


@dataclass(frozen=True)
class McpServerSpec:
    """One MCP session cora may open in deep mode.

    `headers` is the already-resolved per-request header dict (Bearer-
    shaped from a token when one applies) — downstream code (the probe,
    the agent factory) never touches a raw token or an env-var name
    again once this is built.

    `required=True` mirrors the legacy `MCP_URL` contract: an
    unreachable required session fails the whole review
    (`mcp-connect-failed`). `required=False` — the default, and every
    slot except `mcp` — means a failed probe just drops that session and
    logs; the review continues on whatever else loaded.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    required: bool = False


def parse_mcp_servers_env(
    raw: str | None, *, environ: Mapping[str, str] | None = None
) -> tuple[McpServerSpec, ...]:
    """Parse the `MCP_SERVERS` env value (a JSON array) into specs.

    Unset / blank → `()`, same "not configured" shape as the named
    slots. Malformed JSON or a missing required field raises `ValueError`
    with the offending index/field named — a misconfigured deployment
    should fail loud at startup, not silently run with fewer tools than
    it asked for (the same posture `CORA_ESCALATION_TRIGGERS` takes for
    a misspelt trigger).

    `token_env` unset entirely means "this server needs no auth" (no
    warning — that's a deliberate, tokenless server, same as
    `WEB_FETCH_GATE_URL` today). `token_env` SET but pointing at an
    unset/empty env var is different — the operator declared an
    intent to authenticate that didn't resolve — so that case warns and
    falls back to tokenless rather than silently degrading.
    """
    if raw is None or not raw.strip():
        return ()
    env = environ if environ is not None else os.environ
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MCP_SERVERS is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("MCP_SERVERS must be a JSON array of server objects")

    specs: list[McpServerSpec] = []
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"MCP_SERVERS[{idx}] must be a JSON object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"MCP_SERVERS[{idx}] is missing a non-empty 'name'")
        url = str(entry.get("url") or "").strip()
        if not url:
            raise ValueError(f"MCP_SERVERS[{idx}] ({name}) is missing a non-empty 'url'")
        required = entry.get("required", False)
        if not isinstance(required, bool):
            raise ValueError(
                f"MCP_SERVERS[{idx}] ({name}): 'required' must be a JSON boolean"
            )
        token_env = entry.get("token_env")
        headers: dict[str, str] = {}
        if token_env:
            token_env = str(token_env).strip()
            token = (env.get(token_env) or "").strip()
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            else:
                print(
                    f"::warning::MCP_SERVERS[{idx}] ({name}): token_env "
                    f"'{token_env}' is unset — connecting without auth"
                )
        specs.append(McpServerSpec(name=name, url=url, headers=headers, required=required))
    return tuple(specs)


def compose_mcp_sessions(
    *,
    mcp_url: str,
    mcp_headers: Mapping[str, str] | None,
    mcp_actions_url: str | None,
    mcp_actions_headers: Mapping[str, str] | None,
    web_fetch_url: str | None,
    web_fetch_headers: Mapping[str, str] | None,
    extra_sessions: Sequence[McpServerSpec] = (),
    log: Callable[[str], None] = lambda _line: None,
) -> list[McpServerSpec]:
    """Normalize the three legacy MCP slots + any `MCP_SERVERS` extras
    into ONE ordered, probe-ready list.

    Order: `mcp` (the required slot) → `actions` → `web-fetch` → extras,
    in `MCP_SERVERS` order. Every dispatch site that opens MCP sessions
    (`deep_review_call`, `continue_on_t1`, `call_t2_alt_reviewer` via
    `deep_review_call`) builds the list this same way, so the probe/open
    loop (`open_mcp_sessions`) is the only place session-open logic lives.

    Pure — does not probe. A configured-but-token-less `actions` slot is
    dropped HERE (with the legacy "skipping mcp-actions" log line)
    because that slot's contract has always been "no token, no session,
    not even an attempt" — unlike a generic extra session, which attempts
    tokenless when no `token_env` was given.
    """
    sessions: list[McpServerSpec] = []

    mcp_url = (mcp_url or "").strip()
    if mcp_url:
        sessions.append(
            McpServerSpec(
                name="mcp", url=mcp_url, headers=dict(mcp_headers or {}), required=True
            )
        )

    mcp_actions_url = (mcp_actions_url or "").strip() or None
    if mcp_actions_url and mcp_actions_headers:
        sessions.append(
            McpServerSpec(
                name="actions",
                url=mcp_actions_url,
                headers=dict(mcp_actions_headers),
                required=False,
            )
        )
    elif mcp_actions_url:
        log("MCP_ACTIONS_TOKEN unset — skipping mcp-actions")

    web_fetch_url = (web_fetch_url or "").strip() or None
    if web_fetch_url:
        sessions.append(
            McpServerSpec(
                name="web-fetch",
                url=web_fetch_url,
                headers=dict(web_fetch_headers or {}),
                required=False,
            )
        )

    sessions.extend(extra_sessions)
    return sessions


async def open_mcp_sessions(
    sessions: Sequence[McpServerSpec],
    *,
    probe: ProbeFn,
    log: Callable[[str], None],
    label_suffix: str = "",
) -> tuple[list[tuple[str, dict[str, str]]], dict[str, bool]] | None:
    """Probe each session in list order; returns `(toolset_args, opened)`.

    `toolset_args` is ready to hand to `AgentConfig.mcp_servers` (a list
    of `(url, headers)`, same shape as before). `opened` maps session
    name → True for every session that actually loaded — the
    generalisation of the old `read_enabled` / `actions_enabled` /
    `web_enabled` booleans; a name absent from `opened` was either not
    configured or dropped after a failed optional probe.

    Returns `None` (instead of a partial result) the moment a REQUIRED
    session's probe fails — same short-circuit as the old inline check:
    the caller maps that to `terminated_reason="mcp-connect-failed"` and
    the whole run fails rather than continuing on a reduced tool surface
    the operator didn't ask for.
    """
    toolsets: list[tuple[str, dict[str, str]]] = []
    opened: dict[str, bool] = {}
    for spec in sessions:
        probe_name = f"{spec.name}{label_suffix}"
        ok = await probe(spec.url, spec.headers, probe_name, log)
        if not ok:
            if spec.required:
                return None
            continue
        toolsets.append((spec.url, spec.headers))
        opened[spec.name] = True
        if not spec.required:
            suffix = " (observe-only)" if spec.name == "actions" else ""
            log(f"{spec.name} session opened{suffix}")
    return toolsets, opened


def resolve_web_fetch_url(cfg: "ReviewerConfig") -> str:
    """Resolve the web-fetch-gate endpoint for the release-notes
    pre-fetch (`review/_context.py`) and the initial-prompt's "a fetch
    tool is available" advertisement.

    Explicit `WEB_FETCH_GATE_URL` wins; otherwise the first `MCP_SERVERS`
    entry declared `"name": "web-fetch"` — a NAME CONVENTION, not a
    probe: the release-notes pre-fetch runs before any MCP session opens
    (it needs the URL to make its own short-lived session), so there is
    nothing to introspect yet. Returns `""` when neither resolves.
    """
    explicit = (cfg.web_fetch_gate_url or "").strip()
    if explicit:
        return explicit
    for spec in cfg.mcp_servers:
        if spec.name == "web-fetch":
            return spec.url
    return ""
