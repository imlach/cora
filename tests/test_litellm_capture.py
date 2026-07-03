"""Unit tests for `agent_review.litellm_capture` — the httpx response
event hook that surfaces `x-litellm-*` headers across pydantic-ai's
Agent wrapper into `Budget.resolved_model`.

Uses `httpx.MockTransport` so no network is required — drives the
exact code path the production `OpenAIProvider(http_client=...)`
wiring exercises."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from cora.core.litellm_capture import (
    build_capture_client,
    drain_captured_headers,
    resolved_model_from,
)


def _mock_transport(headers: dict[str, str]) -> httpx.MockTransport:
    """Build a transport that always replies 200 with the given
    headers — `x-litellm-*` is what the capture hook keys on."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, json={"ok": True})

    return httpx.MockTransport(handler)


def test_hook_captures_x_litellm_headers():
    """A response carrying `x-litellm-*` headers should land in the
    drain — non-litellm headers are filtered out."""
    drain_captured_headers()  # clear any prior-test residue

    async def go() -> dict[str, str]:
        client = build_capture_client()
        client._transport = _mock_transport(  # noqa: SLF001 — test seam
            {
                "x-litellm-model-id": "backend-b",
                "x-litellm-call-id": "abc123",
                "content-type": "application/json",
            }
        )
        await client.get("http://stub.test/anything")
        await client.aclose()
        return drain_captured_headers()

    captured = asyncio.run(go())
    assert captured == {
        "x-litellm-model-id": "backend-b",
        "x-litellm-call-id": "abc123",
    }


def test_drain_resets_to_empty():
    """Drain should clear the ContextVar so a subsequent drain on the
    same context returns `{}` — last-write-wins semantics depend on
    this so a failed turn doesn't carry stale data into the next."""
    drain_captured_headers()

    async def go() -> tuple[dict[str, str], dict[str, str]]:
        client = build_capture_client()
        client._transport = _mock_transport(  # noqa: SLF001
            {"x-litellm-model-id": "backend-a"}
        )
        await client.get("http://stub.test/anything")
        await client.aclose()
        first = drain_captured_headers()
        second = drain_captured_headers()
        return first, second

    first, second = asyncio.run(go())
    assert first == {"x-litellm-model-id": "backend-a"}
    assert second == {}


def test_no_capture_when_headers_absent():
    """A response without any `x-litellm-*` headers shouldn't seed
    the ContextVar — drain stays empty so the call site falls through
    to its placeholder string."""
    drain_captured_headers()

    async def go() -> dict[str, str]:
        client = build_capture_client()
        client._transport = _mock_transport(  # noqa: SLF001
            {"content-type": "application/json"}
        )
        await client.get("http://stub.test/anything")
        await client.aclose()
        return drain_captured_headers()

    assert asyncio.run(go()) == {}


def test_capture_visible_across_asyncio_task_boundary():
    """Regression guard for the pydantic-ai task-spawn case: the
    model request is dispatched via `asyncio.create_task` inside
    pydantic-ai's agent graph, so the hook runs in a child task. The
    parent's drain must still see the captured headers. (ContextVars
    don't propagate child→parent — this test would fail with the
    earlier ContextVar implementation.)"""
    drain_captured_headers()

    async def go() -> dict[str, str]:
        client = build_capture_client()
        client._transport = _mock_transport(  # noqa: SLF001
            {"x-litellm-model-id": "backend-b"}
        )

        async def child() -> None:
            # Hook runs inside this child task.
            await client.get("http://stub.test/anything")

        await asyncio.create_task(child())
        await client.aclose()
        # Drained from the parent — must still see the child's capture.
        return drain_captured_headers()

    assert asyncio.run(go()) == {"x-litellm-model-id": "backend-b"}


@pytest.mark.parametrize(
    "headers,expected",
    [
        # Explicit model_id wins — operator-set value beats parsing.
        (
            {
                "x-litellm-model-id": "backend-b-explicit",
                "x-litellm-model-api-base": "http://backend-b.serving.svc.cluster.local:8000/v1",
            },
            "backend-b-explicit",
        ),
        # Auto-hashed model_id (LiteLLM default when unset) is opaque
        # → parse the api_base hostname instead. Matches the shape
        # observed in the reference deployment (a 64-hex hash).
        (
            {
                "x-litellm-model-id": "205478b6d27e16f6d0dc57b7c75544816a71a6838584c7778c5342af09209e25",
                "x-litellm-model-api-base": "http://backend-a.serving.svc.cluster.local:8000/v1",
            },
            "backend-a",
        ),
        # 32-char hash also recognised as auto-generated.
        (
            {
                "x-litellm-model-id": "0123456789abcdef0123456789abcdef",
                "x-litellm-model-api-base": "http://backend-b.serving.svc.cluster.local:8000/v1",
            },
            "backend-b",
        ),
        # External cloud endpoint — full hostname, no first-segment
        # collapse (api.mistral.ai shouldn't become "api").
        (
            {
                "x-litellm-model-id": "abc123def456abc123def456abc123def456",
                "x-litellm-model-api-base": "https://api.mistral.ai/v1",
            },
            "api.mistral.ai",
        ),
        # Workstation on the local network — falls through the
        # in-cluster filter, full hostname retained.
        (
            {
                "x-litellm-model-id": "ffffeeeeddddccccbbbbaaaa99998888",
                "x-litellm-model-api-base": "http://workstation.internal:1234/v1",
            },
            "workstation.internal",
        ),
        # api_base only, no model_id at all.
        (
            {"x-litellm-model-api-base": "http://backend-b.serving.svc.cluster.local:8000/v1"},
            "backend-b",
        ),
        # Hashed model_id with NO api_base — fall back to the hash;
        # less readable than nothing in the chip.
        (
            {"x-litellm-model-id": "deadbeefdeadbeefdeadbeefdeadbeef"},
            "deadbeefdeadbeefdeadbeefdeadbeef",
        ),
        # Nothing useful at all.
        ({}, None),
        ({"x-litellm-call-id": "xyz"}, None),
    ],
)
def test_resolved_model_from_picks_preferred_header(headers, expected):
    assert resolved_model_from(headers) == expected
