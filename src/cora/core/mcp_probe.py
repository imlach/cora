"""Shared MCP-server connectivity probe.

Originally inlined into `deep_review.py` and duplicated into a sibling
triage pipeline. The reviewer's `continuation.py` then imported the
deep_review copy as a private function, and the triage continuation
did the same against its own copy. Both patterns surfaced as a
coupling concern in review
("private function imported from a sibling file is brittle"), so
the probe was promoted to a real shared utility.

Single public symbol: `probe_mcp_server`. Both reviewer + triage call
it identically — same fail-soft contract, same log shape.
"""

from __future__ import annotations

from typing import Callable

# Probe-failure log lines cap the exception text — a malformed multi-line
# credential once ballooned the message, and the tail adds no signal.
_EXC_LOG_CHAR_CAP = 400


def _redact(exc_text: str, headers: dict[str, str]) -> str:
    """Strip header values (credentials) from an exception message.

    The HTTP client embeds the offending value verbatim in errors like
    ``Illegal header value b'…'`` — both the raw string and its escaped
    bytes-repr form — so a probe failure must never log the exception
    unsanitized: outside GHA's secret masking that prints the token.
    """
    for value in headers.values():
        if not value:
            continue
        escaped = repr(value.encode())[2:-1]  # bytes-repr inner text
        for needle in (value, escaped):
            if needle and needle in exc_text:
                exc_text = exc_text.replace(needle, "[redacted]")
    if len(exc_text) > _EXC_LOG_CHAR_CAP:
        exc_text = exc_text[:_EXC_LOG_CHAR_CAP] + "…[truncated]"
    return exc_text


async def probe_mcp_server(
    url: str,
    headers: dict[str, str],
    name: str,
    log: Callable[[str], None],
) -> bool:
    """Eager connectivity probe for an MCP server. Returns True if the
    server answered a `list_tools()` call, False if anything broke
    (connect refused, auth rejected, timeout, malformed response).

    Necessary because Pydantic-AI's `MCPToolset` opens its transport
    lazily on first tool-list during `agent.run`; an unreachable entry
    in `toolsets` would fail the whole run mid-flight. We probe at
    construction time so optional servers can fail silently and the
    required MCP server returns a distinct `mcp-connect-failed`
    `terminated_reason` (check-run conclusion → `cancelled`, retry on
    re-push) instead of a generic agent-loop crash.

    The duplicate handshake (probe session vs. the agent's later
    session) is the cost of resilience against soft-fail MCP servers.
    """
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset(url, headers=headers or None)
    try:
        async with toolset:
            await toolset.list_tools()
        return True
    except Exception as exc:  # noqa: BLE001
        reason = _redact(str(exc), headers or {})
        log(f"{name} probe failed ({reason}) — continuing without it")
        return False
