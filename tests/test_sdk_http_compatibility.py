"""Exercise Cora's real SDK boundary without an external model service."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cora.core import litellm_capture
from cora.core.agent import AgentConfig, Deps, make_review_agent
from cora.core.budget import Budget, PydanticAIUsageAdapter, resolve_run_usage


def _completion():
    return {
        "id": "chat-test", "object": "chat.completion", "created": 1,
        "model": "served-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Review complete."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
    }


def _agent(monkeypatch, handler):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"response": [litellm_capture._capture_response]},
        headers={"x-imla-session": "test-session"},
    )
    monkeypatch.setattr(litellm_capture, "build_capture_client", lambda _headers=None: client)
    agent = make_review_agent(AgentConfig(
        endpoint_base_url="https://model.test/v1", api_key="test-key",
        model_alias="review", system_prompt="Review the supplied change.",
    ))
    return agent, client


def test_real_agent_completion_and_usage(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_completion(), headers={"x-litellm-model-id": "served-model"})

    async def run():
        agent, client = _agent(monkeypatch, handler)
        try:
            result = await agent.run("A test change.", deps=Deps(repo="test/repo", pr_number="1"))
            budget = Budget(max_input=100, max_output=100, max_iterations=5)
            budget.add_usage(PydanticAIUsageAdapter(resolve_run_usage(result)))
            assert result.output == "Review complete."
            assert budget.input_used == 17
            assert budget.output_used == 5
            assert litellm_capture.drain_captured_headers()["x-litellm-model-id"] == "served-model"
            assert requests[0].url.path == "/v1/chat/completions"
            assert json.loads(requests[0].content)["model"] == "review"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_real_agent_stream_and_final_usage(monkeypatch):
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        chunks = [
            {"id": "chat-test", "object": "chat.completion.chunk", "created": 1, "model": "served-model", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Review complete."}, "finish_reason": None}]},
            {"id": "chat-test", "object": "chat.completion.chunk", "created": 1, "model": "served-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def run():
        agent, client = _agent(monkeypatch, handler)
        try:
            async with agent.run_stream("A test change.", deps=Deps(repo="test/repo", pr_number="1")) as result:
                assert await result.get_output() == "Review complete."
                usage = resolve_run_usage(result)
                assert usage.input_tokens == 17
                assert usage.output_tokens == 5
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_real_agent_http_errors_preserve_status(monkeypatch, status):
    from pydantic_ai.exceptions import ModelHTTPError

    def handler(_request):
        return httpx.Response(status, json={"error": {"message": "test rejection", "type": "test_error"}})

    async def run():
        agent, client = _agent(monkeypatch, handler)
        # Disable the SDK retry delay; Cora owns its bounded rate-limit loop.
        agent.model.provider.client.max_retries = 0
        try:
            with pytest.raises(ModelHTTPError) as caught:
                await agent.run("A test change.", deps=Deps(repo="test/repo", pr_number="1"))
            assert caught.value.status_code == status
        finally:
            await client.aclose()

    asyncio.run(run())
