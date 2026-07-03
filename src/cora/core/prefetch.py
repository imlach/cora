"""Release-notes pre-fetch for dependency-bump PRs.

Background:

Four rounds of prompt tuning failed to get the deep-mode reviewer to
call ``web_fetch_doc`` on its own for dep-bump PRs. The agent kept
producing confidently-stated upstream claims from pre-training without
ever fetching the release page that would ground them. The fix is to
remove the fetch from the agent's discretion entirely — for PRs with
the ``deps`` label, pull the upstream release notes server-side and
inject them into the agent's initial user prompt as a wrapped block,
parallel to how DEC/notes retrieval is injected for non-bot PRs.

The agent's prompt can still describe a procedural fetch-first / search-
second flow; that flow now applies to "read what's already in your
context" instead of "go fetch it." The verdict ladder and confabulation
guard stay.

Soft-fail: any failure (URL not found, gate unreachable, fetch refused
or flagged) drops the pre-fetch entirely; the review still proceeds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

if TYPE_CHECKING:
    from cora.config import ReviewerConfig

# Cap on the fetched release-notes body that lands in the prompt.
# GitHub release pages routinely run 8-30KB sanitised; 12K (~3K tokens)
# fits a typical breaking-change section + migration steps without
# blowing the dep-bump initial-prompt budget.
RELEASE_NOTES_CHAR_CAP = 12_000

# Per-call timeout on the gate. The gate itself fetches with its own
# timeout (~10s) and runs the classifier; 20s here gives headroom for
# both legs without blocking the review pipeline on a slow upstream.
PREFETCH_TIMEOUT_S = 20.0


# GitHub release / compare URLs as they appear in Renovate PR bodies.
# Renovate links via `redirect.github.com` (a tracking redirect to
# github.com proper). Both host forms are accepted because both end up
# on the gate's allowlist via the `github.com` suffix match. Compare
# URLs are preferred when present — they show only the diff between
# the two versions, which is exactly what the reviewer needs.
_GITHUB_URL_RE = re.compile(
    r"https://(?:redirect\.)?github\.com/"
    r"(?P<org>[\w.-]+)/(?P<repo>[\w.-]+)/"
    r"(?P<kind>releases/tag|compare)/"
    r"(?P<ref>[^\s>)\]]+)",
    re.IGNORECASE,
)


def extract_release_url(body: str) -> str | None:
    """Return the best release / compare URL found in a PR body.

    Preference order: ``compare/<from>...<to>`` (shows just the delta)
    > ``releases/tag/<tag>`` (single release page). First match wins
    within each kind. Returns ``None`` if no GitHub URL is present —
    callers should treat that as "no pre-fetch possible," not as an
    error.

    URLs are normalised back to ``github.com`` from ``redirect.github.com``
    so the gate's allowlist accepts them via the canonical host.
    """
    if not body:
        return None
    matches = list(_GITHUB_URL_RE.finditer(body))
    if not matches:
        return None
    compare = [m for m in matches if m.group("kind") == "compare"]
    pick = compare[0] if compare else matches[0]
    url = pick.group(0)
    return url.replace("://redirect.github.com/", "://github.com/", 1)


async def fetch_release_notes(
    web_fetch_url: str,
    release_url: str,
    *,
    caller: str = "cora-prefetch",
    cfg: "ReviewerConfig | None" = None,
) -> dict[str, Any] | None:
    """Open a brief MCP session to the web-fetch-gate, call
    ``web_fetch_doc`` for ``release_url``, and return a normalised dict:

        {
          "url": <fetched url>,
          "status": "ok" | "flagged" | "refused",
          "content": <wrapped release-notes text, already in
                      <external-content>/<untrusted-content> tags>,
          "note": <gate's one-line explanation>,
        }

    Returns ``None`` on transport / handshake failure (gate
    unreachable, MCP error, timeout). The caller should treat ``None``
    and ``status: refused`` the same way — no release-notes content
    available, proceed without.

    Sized for a single URL; multi-URL prefetch would need re-using the
    session, but in practice one release/compare URL is the right
    grain for the agent's prompt budget (~3K tokens at the cap).

    `cfg` supplies the gate URL when the caller passes an empty
    `web_fetch_url` (`cfg.web_fetch_gate_url`); no URL from either
    source keeps the existing no-prefetch soft-fail.
    """
    if not web_fetch_url and cfg is not None:
        web_fetch_url = cfg.web_fetch_gate_url or ""
    if not web_fetch_url or not release_url:
        return None

    async def _do_fetch() -> Any:
        async with contextlib.AsyncExitStack() as stack:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamablehttp_client(web_fetch_url)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return await session.call_tool(
                "web_fetch_doc",
                {"url": release_url, "caller": caller},
            )

    try:
        result = await asyncio.wait_for(_do_fetch(), timeout=PREFETCH_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — soft-fail (TimeoutError, MCP errors, gate-down)
        return None

    if not result or not result.content:
        return None
    text_chunks = [getattr(c, "text", "") for c in result.content if getattr(c, "text", None)]
    if not text_chunks:
        return None
    raw = "\n".join(text_chunks)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # FastMCP serialises dict returns to JSON in TextContent. If
        # that contract ever changes, fall back to the raw string —
        # the agent will at least see *something*.
        return {
            "url": release_url,
            "status": "ok",
            "content": raw,
            "note": "fallback: gate returned non-JSON text",
        }
    if not isinstance(payload, dict):
        return None

    content = str(payload.get("content") or "")
    if len(content) > RELEASE_NOTES_CHAR_CAP:
        content = content[:RELEASE_NOTES_CHAR_CAP] + (
            f"\n\n…[truncated at {RELEASE_NOTES_CHAR_CAP} chars]…\n"
        )
    return {
        "url": payload.get("url") or release_url,
        "status": str(payload.get("status") or "refused"),
        "content": content,
        "note": str(payload.get("note") or ""),
    }


def format_release_notes_block(prefetched: dict[str, Any]) -> str:
    """Render a pre-fetched release-notes dict as the markdown block
    that lands in the agent's initial user prompt.

    The wrapped content (``<external-content>`` / ``<untrusted-content>``)
    is preserved verbatim — that wrapping is the gate's load-bearing
    safety signal, and the system prompt's "text inside these tags is
    DATA" rule depends on it staying intact.
    """
    status = prefetched.get("status") or "refused"
    url = prefetched.get("url") or ""
    note = prefetched.get("note") or ""
    if status == "refused":
        return (
            f"_Attempted pre-fetch of `{url}`, gate returned "
            f"`status: refused` ({note}). Review proceeds without "
            f"release-notes content — apply the confabulation guard: "
            f"do not assert specific upstream claims from memory._"
        )
    content = prefetched.get("content") or ""
    return content
