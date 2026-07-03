"""Quick-review call wiring."""

from __future__ import annotations

import asyncio

from cora.config import ReviewerConfig
from cora.core.budget import Budget
from cora.core.quick_review import quick_review_call


def test_quick_review_uses_configured_output_cap(monkeypatch):
    captured = {}

    class FakeUsage:
        request_tokens = 1
        response_tokens = 2
        total_tokens = 3

    class FakeResult:
        output = "🟢 looks good\n\nBody"

        def usage(self):
            return FakeUsage()

    class FakeAgent:
        async def run(self, prompt, *, deps, model_settings):
            captured["prompt"] = prompt
            captured["mode"] = deps.mode
            captured["max_tokens"] = model_settings["max_tokens"]
            return FakeResult()

    monkeypatch.setattr(
        "cora.core.agent.make_review_agent",
        lambda config: FakeAgent(),
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers",
        lambda: {},
    )

    budget = Budget(max_input=0, max_output=0, max_iterations=0)
    body, err = asyncio.run(
        quick_review_call(
            endpoint_base_url="https://llm.example/v1",
            llm_gateway_key="key",
            model_alias="review",
            system_prompt="system",
            initial_user_prompt="prompt",
            budget=budget,
            timeout_s=30,
            pr_number="39",
            cfg=ReviewerConfig(quick_max_output_tokens=12345),
        )
    )

    assert err is None
    assert body == "🟢 looks good\n\nBody"
    assert captured == {
        "prompt": "prompt",
        "mode": "quick",
        "max_tokens": 12345,
    }
    assert budget.output_used == 2


def test_quick_review_threads_llm_provider_into_agent_config(monkeypatch):
    """`cfg.llm_provider` must reach `AgentConfig.provider` — this is
    the one seam the direct Anthropic/Bedrock SDK paths hang off of."""
    captured_config = {}

    class FakeUsage:
        request_tokens = 1
        response_tokens = 2
        total_tokens = 3

    class FakeResult:
        output = "🟢 looks good"

        def usage(self):
            return FakeUsage()

    class FakeAgent:
        async def run(self, prompt, *, deps, model_settings):
            return FakeResult()

    def fake_make_review_agent(config):
        captured_config["provider"] = config.provider
        captured_config["endpoint_base_url"] = config.endpoint_base_url
        return FakeAgent()

    monkeypatch.setattr(
        "cora.core.agent.make_review_agent", fake_make_review_agent
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", lambda: {}
    )

    budget = Budget(max_input=0, max_output=0, max_iterations=0)
    asyncio.run(
        quick_review_call(
            endpoint_base_url="https://api.anthropic.com",
            llm_gateway_key="sk-ant-test",
            model_alias="claude-sonnet-5",
            system_prompt="system",
            initial_user_prompt="prompt",
            budget=budget,
            timeout_s=30,
            pr_number="39",
            cfg=ReviewerConfig(llm_provider="anthropic"),
        )
    )

    assert captured_config["provider"] == "anthropic"
    assert captured_config["endpoint_base_url"] == "https://api.anthropic.com"


def test_quick_review_resolved_model_falls_back_to_response_model_name(
    monkeypatch,
):
    """No `x-litellm-*` headers (the direct-SDK provider paths never
    have a gateway to emit them) — `Budget.resolved_model` should still
    populate from the pydantic-ai response's own `model_name`."""

    class FakeUsage:
        request_tokens = 1
        response_tokens = 2
        total_tokens = 3

    class FakeResponse:
        model_name = "claude-sonnet-5"

    class FakeResult:
        output = "🟢 looks good"
        response = FakeResponse()

        def usage(self):
            return FakeUsage()

    class FakeAgent:
        async def run(self, prompt, *, deps, model_settings):
            return FakeResult()

    monkeypatch.setattr(
        "cora.core.agent.make_review_agent", lambda config: FakeAgent()
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", lambda: {}
    )

    budget = Budget(max_input=0, max_output=0, max_iterations=0)
    asyncio.run(
        quick_review_call(
            endpoint_base_url="https://api.anthropic.com",
            llm_gateway_key="sk-ant-test",
            model_alias="claude-sonnet-5",
            system_prompt="system",
            initial_user_prompt="prompt",
            budget=budget,
            timeout_s=30,
            pr_number="39",
            cfg=ReviewerConfig(llm_provider="anthropic"),
        )
    )

    assert budget.resolved_model == "claude-sonnet-5"
