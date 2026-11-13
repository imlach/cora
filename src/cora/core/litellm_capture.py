"""Backend attribution for the Pydantic-AI agent path.

Populates the `Budget.resolved_model` chip — the concrete engine that
actually served the review, so a fallback-served review is
distinguishable from a primary-served one. Two independent sources,
preference-ordered by `resolve_backend_attribution`:

1. `x-litellm-*` response headers (the LiteLLM path). Pydantic-AI's
   `Agent` wraps an `openai.AsyncOpenAI` client and doesn't surface raw
   response headers, so this module restores them via an httpx response
   event hook threaded into `OpenAIProvider(http_client=...)`; the call
   site drains the bucket after `agent.run()`. Harmless once the gateway
   stops emitting `x-litellm-*` (drain returns `{}` → this source is
   skipped).
2. The completion response body's `model` field, read back off the
   pydantic-ai result (`body_model_from_result`). Every OpenAI-shaped
   completion carries the served-model name, so this survives a gateway
   (Envoy AI Gateway v2) that emits no `x-litellm-*` headers at all.

The header hook reads every `x-litellm-*` header off the response and
stashes them in a module-level bucket; the call site drains the bucket
after `agent.run()` and feeds it into Budget.

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

import json
import math
import os
import re
from urllib.parse import urlparse

import httpx

# Most recently captured `x-litellm-*` headers. Drained + cleared by
# `drain_captured_headers()` after each `agent.run()`. Empty dict
# means no LiteLLM response landed (either the endpoint isn't
# LiteLLM-fronted, or the call failed before headers came back).
_captured: dict[str, str] = {}
_billing: dict[str, dict] = {}


def drain_billing(response_id: str | None) -> dict:
    """Return allowlisted provider-reported metadata for one response."""
    return _billing.pop(response_id, {})


def merge_provider_preferences(defaults: dict, explicit: dict) -> dict:
    """Keep explicit provider constraints while adding missing defaults."""
    routing_keys = {"order", "sort", "only", "ignore", "quantizations", "max_price"}
    if routing_keys.intersection(explicit):
        return dict(explicit)
    return {**defaults, **explicit}


async def _capture_billing(response: httpx.Response) -> None:
    if not response.headers.get("content-type", "").startswith("application/json"):
        return
    try:
        await response.aread()
        body = response.json()
        response_id = body.get("id")
        if not isinstance(response_id, str):
            return
        usage = body.get("usage") or {}
        fields = {"generation_id": json.dumps(response_id[:256])}
        cost = usage.get("cost")
        if (isinstance(cost, (int, float)) and not isinstance(cost, bool)
                and math.isfinite(cost) and cost >= 0):
            fields["cost_usd"] = cost
        provider = body.get("provider")
        if isinstance(provider, str):
            fields["billing_provider"] = json.dumps(provider[:100])
        _billing[response_id] = fields
        while len(_billing) > 256:
            _billing.pop(next(iter(_billing)))
    except (ValueError, TypeError, AttributeError):
        pass



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
    global _captured
    if _DEBUG:
        # Lowercased keys, sorted for stable log diffing.
        all_keys = sorted(k.lower() for k in response.headers)
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


def build_capture_client(
    default_headers: dict[str, str] | None = None,
    provider_preferences: dict | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the `httpx.AsyncClient` to thread into `OpenAIProvider`.

    Returns a fresh client with `_capture_response` registered as the
    sole response event hook. Caller owns the lifecycle (close on
    teardown); OpenAIProvider hands it to the underlying
    `openai.AsyncOpenAI` which does not close it on exit either, so
    the call site or its enclosing context manager is responsible.

    `default_headers` are sent on every request the client makes — the
    client is the only seam that covers *all* model calls, including the
    ones the framework issues internally. Empty/None sends nothing, so
    an unconfigured deployment's requests are byte-identical to before.
    """
    class ProviderDefaultsClient(httpx.AsyncClient):
        async def send(self, request: httpx.Request, *args, **kwargs):
            if (provider_preferences and request.method == "POST"
                    and request.url.path.endswith("/chat/completions")):
                body = json.loads(await request.aread())
                explicit = body.get("provider", {})
                if isinstance(explicit, dict):
                    body["provider"] = merge_provider_preferences(provider_preferences, explicit)
                    headers = dict(request.headers)
                    headers.pop("content-length", None)
                    request = httpx.Request(
                        request.method, request.url, headers=headers,
                        json=body, extensions=request.extensions,
                    )
            return await super().send(request, *args, **kwargs)

    return ProviderDefaultsClient(
        event_hooks={"response": [_capture_response, _capture_billing]},
        headers=default_headers or None,
        transport=transport,
    )


def drain_captured_headers() -> dict[str, str]:
    """Return the current captured headers and reset to empty.

    Call after `agent.run()` completes (success or failure) to read
    whatever LiteLLM set on the final turn. Returns `{}` if nothing
    was captured (non-LiteLLM endpoint, or the call errored before a
    response landed)."""
    global _captured
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


def body_model_from_result(result) -> str | None:
    """Served-model name from the completion response body, or `None`.

    Pydantic-AI stamps each `ModelResponse.model_name` from the OpenAI
    response body's `model` field (`OpenAIChatModel` sets
    `model_name=response.model`). End-to-end through the gateway that's
    the serving engine's `--served-model-name` (e.g. `piano` / `forte`),
    so it attributes the review even when the gateway emits no
    `x-litellm-*` headers — the case after the Envoy AI Gateway cutover.

    Returns the last non-empty `model_name` across the run's messages
    (last-write-wins, matching `Budget.resolved_model`'s final-turn
    semantics), or `None` if the framework surfaced none. Duck-typed —
    reads `.model_name` off each message, so a `ModelRequest` (no such
    attr) is skipped and no pydantic-ai import is needed. Best-effort:
    any framework-shape surprise degrades to `None`, never raises."""
    try:
        messages = result.all_messages()
    except Exception:  # noqa: BLE001 — best-effort attribution only
        return None
    return served_model_from_messages(messages)


def served_model_from_messages(messages) -> str | None:
    """Last non-empty `model_name` across a message list, or `None`.

    Same last-write-wins rule as `body_model_from_result`, but reads a
    raw history rather than a run result — the tier code holds T0's
    messages even on paths where the run object never came back (an
    errored loop), and comparing the served names is the only way to
    see that two different aliases hit the same pods."""
    model_name: str | None = None
    try:
        for msg in messages or ():
            name = getattr(msg, "model_name", None)
            if isinstance(name, str) and name.strip():
                model_name = name.strip()
    except Exception:  # noqa: BLE001 — best-effort attribution only
        return None
    return model_name


def resolve_backend_attribution(
    captured: dict[str, str],
    result,
    *,
    fallback: str,
) -> str:
    """Preference-ordered backend identity for the `resolved_model` chip.

    1. `x-litellm-*` response headers (`resolved_model_from`) — LiteLLM
       path; back-compat, harmless once the gateway stops emitting them.
    2. The response body's `model` field (`body_model_from_result`) — the
       served-model name, works through Envoy AI Gateway v2.
    3. `fallback` — the caller's `unknown (...)` string when neither
       source resolved (attribution genuinely unavailable)."""
    return (
        resolved_model_from(captured)
        or body_model_from_result(result)
        or fallback
    )
