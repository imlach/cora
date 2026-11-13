"""The single format repair must retain hosted settings and fail closed."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from cora.core.agent import SESSION_HEADER
from cora.core.budget import Budget
from cora.core.litellm_capture import build_capture_client
from cora.review._output import quick_review_retry_for_format


@pytest.mark.parametrize("content", [None, "", "🟡 minor\nA bounded finding."])
def test_format_retry_low_effort_routing_session_and_empty_failure(content):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "gen-format", "object": "chat.completion", "created": 1,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    budget = Budget(max_input=100, max_output=100, max_iterations=1)

    async def run():
        async with AsyncOpenAI(
            api_key="test-key", base_url="https://model.test/v1",
            http_client=build_capture_client(
                {SESSION_HEADER: "test-session"},
                provider_preferences={"order": ["example"], "allow_fallbacks": True},
                transport=httpx.MockTransport(handler),
            ),
        ) as client:
            return await quick_review_retry_for_format(
                llm_client=client, model="test-model", leaked_body="A finding without a marker.",
                budget=budget, timeout_s=10, pr_number="42", max_tokens=18000,
                reasoning_effort="low",
            )

    body, error = asyncio.run(run())
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["reasoning_effort"] == "low"
    assert payload["provider"] == {"order": ["example"], "allow_fallbacks": True}
    assert payload["metadata"]["tags"] == ["pr-42", "consumer:pr-review", "kind:leak-retry"]
    assert payload["max_tokens"] == 18000
    assert requests[0].headers[SESSION_HEADER] == "test-session"
    assert budget.input_used == 10 and budget.output_used == 5
    if content:
        assert body == content and error is None
    else:
        assert body == "" and error.startswith("empty retry response")
        assert "reasoning_tokens=3" in error


def test_unconfigured_format_retry_omits_reasoning_effort():
    from types import SimpleNamespace as NS

    captured = []
    async def create(**kwargs):
        captured.append(kwargs)
        response = NS(choices=[NS(message=NS(content="🟢 looks good"))], usage=None)
        return NS(parse=lambda: response)
    client = NS(chat=NS(completions=NS(with_raw_response=NS(create=create))))
    asyncio.run(quick_review_retry_for_format(
        llm_client=client, model="test-model", leaked_body="Review.",
        budget=Budget(100, 100, 1), timeout_s=10, pr_number="42",
    ))
    assert "reasoning_effort" not in captured[0]["extra_body"]
