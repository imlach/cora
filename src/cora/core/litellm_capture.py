"""LiteLLM response-header capture for the Pydantic-AI agent path.

Pydantic-AI's `Agent` wraps an `openai.AsyncOpenAI` client and doesn't
surface raw response headers, so the `Budget.resolved_model` chip
(which concrete backend LiteLLM picked)
lost its source when the reviewer cut over from the bespoke agent
loop to pydantic-ai.

This module restores it by attaching an httpx response event hook to
a custom `httpx.AsyncClient` that the call sites then thread into
`OpenAIProvider(http_client=...)`. The hook reads every `x-litellm-*`
header off the response and stashes them in a module-level bucket;
the call site drains the bucket after `agent.run()` and feeds it
into Budget.

Module-level (not ContextVar) is deliberate. Pydantic-AI dispatches
the model request via `asyncio.create_task` (see
`pydantic_ai/_agent_graph.py` around line 665), and ContextVar values
do not propagate from a child task back to its parent — only the
other direction. A ContextVar set inside the response hook was
landing in the child task's context and the parent's `drain_*` saw
nothing. The module-level bucket sidesteps the task-boundary issue
entirely. Single-PR-per-process workflow means there's no concurrency
concern; last-write-wins matches `Budget.resolved_model`'s existing
semantics (final-call backend is the one reported).
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import httpx


# Most recently captured `x-litellm-*` headers. Drained + cleared by
# `drain_captured_headers()` after each `agent.run()`. Empty dict
# means no LiteLLM response landed (either the endpoint isn't
# LiteLLM-fronted, or the call failed before headers came back).
_captured: dict[str, str] = {}


# LiteLLM doesn't emit a `x-litellm-model-name` header — only
# `x-litellm-model-id` (the deployment's `model_id`, which auto-hashes
# when unset) and `x-litellm-model-api-base` (the upstream URL).
# When a configmap doesn't set explicit model_ids, model-id arrives as
# an opaque hex string. Hostname parsing of the api_base gives us the
# deployment identity in a form humans can read:
# `backend.namespace.svc.cluster.local:8000/v1` → `backend`.
_HEX_HASH = re.compile(r"^[0-9a-f]{32,}$")


# `LITELLM_CAPTURE_DEBUG=1` flips on a one-line dump of every response's
# headers to stderr (visible in GHA logs). Useful the first time after
# a LiteLLM upgrade or routing change to confirm the expected
# `x-litellm-*` keys are actually present; off by default to keep the
# normal reviewer logs quiet.
_DEBUG = os.environ.get("LITELLM_CAPTURE_DEBUG", "") == "1"


async def _capture_response(response: httpx.Response) -> None:
    """httpx response event hook — stash `x-litellm-*` headers."""
    global _captured  # noqa: PLW0603 — last-write-wins bucket
    if _DEBUG:
        # Lowercased keys, sorted for stable log diffing.
        all_keys = sorted(k.lower() for k in response.headers.keys())
        print(
            f"::debug::litellm_capture saw response headers: {all_keys}",
            flush=True,
        )
    captured = {
        k.lower(): v
        for k, v in response.headers.items()
        if k.lower().startswith("x-litellm-")
    }
    if captured:
        _captured = captured


def build_capture_client() -> httpx.AsyncClient:
    """Build the `httpx.AsyncClient` to thread into `OpenAIProvider`.

    Returns a fresh client with `_capture_response` registered as the
    sole response event hook. Caller owns the lifecycle (close on
    teardown); OpenAIProvider hands it to the underlying
    `openai.AsyncOpenAI` which does not close it on exit either, so
    the call site or its enclosing context manager is responsible.
    """
    return httpx.AsyncClient(event_hooks={"response": [_capture_response]})


def drain_captured_headers() -> dict[str, str]:
    """Return the current captured headers and reset to empty.

    Call after `agent.run()` completes (success or failure) to read
    whatever LiteLLM set on the final turn. Returns `{}` if nothing
    was captured (non-LiteLLM endpoint, or the call errored before a
    response landed)."""
    global _captured  # noqa: PLW0603
    headers = _captured
    _captured = {}
    return headers


def resolved_model_from(headers: dict[str, str]) -> str | None:
    """Derive the human-readable backend identity from captured
    `x-litellm-*` headers, or `None` if neither header is present.

    Preference order:
    1. `x-litellm-model-id` IF the operator set an explicit value
       (LiteLLM auto-hashes when unset, hence the hex-hash filter).
    2. First hostname segment of `x-litellm-model-api-base` for
       in-cluster Kubernetes services (the deployment name).
    3. Full hostname of `x-litellm-model-api-base` for anything else
       (external / cloud endpoints).
    4. The raw `x-litellm-model-id` hash as last resort — better than
       nothing if the api_base header is missing for some reason.
    """
    model_id = headers.get("x-litellm-model-id", "").strip()
    if model_id and not _HEX_HASH.match(model_id):
        return model_id

    api_base = headers.get("x-litellm-model-api-base", "").strip()
    if api_base:
        host = urlparse(api_base).hostname or api_base
        # In-cluster Kubernetes services have multi-segment FQDNs
        # ending in `.svc.cluster.local`; the first segment is the
        # deployment name we want. External hosts stay
        # whole — collapsing them to `api` would lose all signal.
        if host.endswith(".svc.cluster.local"):
            return host.split(".", 1)[0]
        return host

    return model_id or None
