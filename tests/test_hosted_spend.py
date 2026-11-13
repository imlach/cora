"""Offline tests for spend controls and provider billing metadata."""
import asyncio
import json
from types import SimpleNamespace as NS

import httpx

from cora.config import ReviewerConfig
from cora.core.agent import (
    Deps,
    no_progress_evidence_repetitions,
    no_progress_tool_repetitions,
    review_model_settings,
)
from cora.core.litellm_capture import (
    build_capture_client,
    drain_billing,
    merge_provider_preferences,
)


def trajectory(args=None, results=None):
    args = args or [{"pattern": "needle"}] * 3
    results = results or ["same evidence"] * len(args)
    messages = []
    for n, (arguments, result) in enumerate(zip(args, results, strict=True)):
        messages.extend([
            NS(parts=[NS(part_kind="tool-call", tool_call_id=str(n),
                         tool_name="grep_repo", args=arguments)]),
            NS(parts=[NS(part_kind="tool-return", tool_call_id=str(n),
                         tool_name="grep_repo", content=result, outcome="success")]),
        ])
    return messages


def test_guard_requires_identical_arguments_and_evidence():
    assert no_progress_tool_repetitions(trajectory()) == 3
    assert no_progress_tool_repetitions(trajectory(args=[{"p": "a"}, {"p": "b"}, {"p": "c"}])) == 1
    assert no_progress_tool_repetitions(trajectory(results=["a", "b", "c"])) == 1
    messages = trajectory()
    messages[-1].parts[0].outcome = "failed"
    assert no_progress_tool_repetitions(messages) == 0
    messages[-1].parts[0].outcome = "success"
    messages[-1].parts[0].content = "ERROR: invalid glob"
    assert no_progress_tool_repetitions(messages) == 0


def test_guard_opt_in_and_grounding_contract():
    deps = Deps(repo="owner/repo", pr_number="1", cfg=ReviewerConfig())
    ctx = NS(messages=trajectory(), deps=deps)
    assert review_model_settings(ctx) == {}
    deps.cfg.no_progress_tool_limit = 3
    assert review_model_settings(ctx)["tool_choice"] == "none"
    ctx.messages = []
    assert review_model_settings(ctx, require_initial_tool_call=True)["tool_choice"] == "required"


def test_provider_defaults_preserve_explicit_constraints_and_billing():
    asyncio.run(_provider_defaults_preserve_explicit_constraints_and_billing())


async def _provider_defaults_preserve_explicit_constraints_and_billing():
    requests = []
    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "gen-test", "provider": "Example", "usage": {"cost": 0.0012},
            "irrelevant": "never emitted",
        })
    client = build_capture_client(
        provider_preferences={"order": ["cheap"], "allow_fallbacks": True},
        transport=httpx.MockTransport(handle),
    )
    async with client:
        await client.post("https://example.test/v1/chat/completions", json={
            "model": "model", "provider": {"order": ["chosen"], "zdr": True},
        })
    assert requests[0]["provider"] == {"order": ["chosen"], "zdr": True}
    assert drain_billing("gen-test") == {
        "generation_id": '"gen-test"', "billing_provider": '"Example"', "cost_usd": 0.0012,
    }
    assert drain_billing("gen-test") == {}


def test_config_from_env():
    cfg = ReviewerConfig.from_env({
        "CORA_NO_PROGRESS_TOOL_LIMIT": "3", "CORA_CONCLUSION_REASONING_EFFORT": "low",
        "CORA_OPENROUTER_PROVIDER_PREFERENCES": '{"order":["example"]}',
    })
    assert cfg.no_progress_tool_limit == 3
    assert cfg.conclusion_reasoning_effort == "low"
    assert cfg.openrouter_provider_preferences == {"order": ["example"]}


def test_explicit_routing_does_not_receive_default_order():
    defaults = {"order": ["cheap"], "allow_fallbacks": True}
    for explicit in ({"sort": "latency"}, {"only": ["chosen"]}, {"ignore": ["cheap"]}):
        assert merge_provider_preferences(defaults, explicit) == explicit
    assert merge_provider_preferences(defaults, {"zdr": True}) == {**defaults, "zdr": True}


def test_real_agent_concludes_after_three_identical_reads(monkeypatch):
    from cora.core import litellm_capture
    from cora.core.agent import AgentConfig, make_review_agent

    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        n = len(bodies)
        message = {"role": "assistant", "content": "Verdict: looks_good\nVerified."}
        finish = "stop"
        if n <= 3:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call-{n}", "type": "function", "function": {
                    "name": "grep_repo", "arguments": '{"pattern":"needle"}',
                },
            }]}
            finish = "tool_calls"
        return httpx.Response(200, json={
            "id": f"gen-{n}", "object": "chat.completion", "created": 1,
            "model": "test-model", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    def grep_repo(pattern: str) -> str:
        return "same verified evidence"

    original = litellm_capture.build_capture_client
    client = original(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(litellm_capture, "build_capture_client", lambda *a, **kw: client)
    async def run():
        agent = make_review_agent(AgentConfig(
            endpoint_base_url="https://example.test/v1", api_key="test",
            model_alias="test-model", system_prompt="Review and conclude.",
            local_tools=[grep_repo], require_initial_tool_call=True,
        ))
        try:
            return await agent.run("Review.", deps=Deps(
                repo="owner/repo", pr_number="1", cfg=ReviewerConfig(no_progress_tool_limit=3),
            ))
        finally:
            await client.aclose()
    result = asyncio.run(run())
    assert result.output.startswith("Verdict:")
    assert len(bodies) == 4
    assert bodies[0]["tool_choice"] == "required"
    assert bodies[-1]["tool_choice"] == "none"


def test_no_novel_evidence_counts_turns_not_parallel_calls():
    def returned(matches, *, truncated=False):
        return NS(part_kind="tool-return", tool_name="grep_repo", outcome="success",
                  content=json.dumps({"matches": matches, "truncated": truncated}))
    grounded = NS(parts=[returned([{"path": "a.py", "line": 1, "text": "evidence"}])])
    empty_batch = NS(parts=[returned([]) for _ in range(4)])
    messages = [grounded, empty_batch]
    assert no_progress_evidence_repetitions(messages) == 1
    assert no_progress_evidence_repetitions(messages + [empty_batch] * 2) == 3
    assert no_progress_evidence_repetitions([empty_batch] * 3) == 0
    novel = NS(parts=[returned([{"path": "b.py", "line": 2, "text": "new"}])])
    assert no_progress_evidence_repetitions([*messages, novel]) == 0
    assert no_progress_evidence_repetitions([*messages, NS(parts=[returned([], truncated=True)])]) == 0
    assert no_progress_evidence_repetitions([*messages, NS(parts=[NS(part_kind="user-prompt")])]) == 0


def test_evidence_guard_uses_real_local_grep_result_schema(tmp_path):
    from cora.core.repo_tools import local_grep_repo

    (tmp_path / "source.py").write_text("verified needle\n")
    (tmp_path / "fresh.py").write_text("novel evidence\n")
    def result(arguments):
        return NS(parts=[NS(
            part_kind="tool-return", tool_name="grep_repo", outcome="success",
            content=local_grep_repo(arguments, root=tmp_path),
        )])
    messages = [result({"pattern": "needle"})]
    messages.extend(result({"pattern": "absent", "glob": scope}) for scope in (
        "*.py", "source.*", "*",
    ))
    assert no_progress_evidence_repetitions(messages) == 3
    assert no_progress_evidence_repetitions([*messages, result({"pattern": "novel"})]) == 0
