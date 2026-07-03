"""Smoke tests for `agent_review/continuation.py`.

Live coverage of the actual T0 → T1 escalation needs a real wall-hit
PR + working LLM endpoint; a captured wall-hit trace is the canonical
fixture (added separately when the eval corpus rule lands). These
tests cover the offline-checkable bits: the continuation directive
text, the usage adapter shape, and that the module imports cleanly
without dragging extra deps.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("pydantic_ai.mcp")


def test_continuation_prompt_frames_resumption():
    """The directive should make it explicit that the prior turns are
    the model's own working state — not a user-authored conversation
    it needs to recap. Resumption-mode language is what differentiates
    the continuation prompt from a normal user turn."""
    from cora.core.continuation import _CONTINUATION_PROMPT

    text = _CONTINUATION_PROMPT.lower()
    assert "resume" in text
    assert "your own working state" in text
    # The directive must explicitly forbid restart/re-summarize so the
    # model doesn't waste its budget recapping what it already has.
    assert "do not re-summarize" in text or "do not restart" in text
    # The directive should reference the standard verdict-line format
    # so the model knows the output contract is unchanged.
    assert "verdict:" in text


def test_pydantic_ai_usage_adapter_maps_field_names():
    """`_PydanticAIUsageAdapter` should mirror the shape used by
    `deep_review` / `quick_review` — request/response → prompt/completion."""
    from cora.core.continuation import _PydanticAIUsageAdapter

    class FakeUsage:
        request_tokens = 12_345
        response_tokens = 678
        total_tokens = 13_023

    adapted = _PydanticAIUsageAdapter(FakeUsage())
    assert adapted.prompt_tokens == 12_345
    assert adapted.completion_tokens == 678
    assert adapted.total_tokens == 13_023


def test_pydantic_ai_usage_adapter_handles_missing_fields():
    """Partial usage objects (error paths) shouldn't raise — Budget
    tracking is soft-fail."""
    from cora.core.continuation import _PydanticAIUsageAdapter

    class EmptyUsage:
        pass

    adapted = _PydanticAIUsageAdapter(EmptyUsage())
    assert adapted.prompt_tokens == 0
    assert adapted.completion_tokens == 0
    assert adapted.total_tokens == 0


def test_continuation_module_imports_cleanly():
    """The continuation module shouldn't drag in unexpected deps at
    import time — `deep_review` helpers are reused but no other
    package-internal imports beyond what the factory needs.
    """
    import sys

    sys.modules.pop("cora.core.continuation", None)
    import cora.core.continuation  # noqa: F401

    # Sanity: the public surface exposes `continue_on_t1`.
    assert hasattr(cora.core.continuation, "continue_on_t1")


def test_continue_on_t1_returns_loaded_tool_palette_on_success():
    """`continue_on_t1`'s success path must surface the loaded tool
    palette in its `tools_available` return slot. The comment footer
    subtracts `budget.tool_calls` from this list to render the unused
    denominator, so returning only fired tools hides successfully loaded
    but unused tools.

    The function does I/O against MCP + the LLM in normal use, so
    monkey-patch it down to the smallest surface that exercises the
    return path: stub the agent factory + the MCP probe, return a
    fake `result` with the expected `output` / `usage()` shape.
    """
    import sys
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.modules.pop("cora.core.continuation", None)
    import cora.core.continuation as cont

    class FakeAgent:
        def __init__(self):
            self.tool_call_counter_captured = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *_args, **kwargs):
            return _FakeIterCtx(kwargs)

    class _FakeIterCtx:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return _FakeAgentRun()

        async def __aexit__(self, *_):
            return False

    class _FakeRunUsage:
        request_tokens = 100
        response_tokens = 50
        total_tokens = 150
        tool_calls = 2

    class _FakeAgentRun:
        @property
        def result(self):
            return SimpleNamespace(
                output="🟢 looks good\n\nT1 completed.",
                usage=lambda: _FakeRunUsage(),
            )

        # `async for node in agent_run:` — we want the helper to
        # populate the tool_call_counter, so feed it a CallToolsNode
        # carrying a ToolCallPart.
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            from pydantic_ai._agent_graph import CallToolsNode
            from pydantic_ai.messages import ModelResponse, ToolCallPart

            yield CallToolsNode(
                model_response=ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search_knowledge",
                            args={"query": "egress policy"},
                            tool_call_id="tc1",
                        ),
                    ],
                    model_name="cora",
                ),
            )

    class FakeBudget:
        def add_usage(self, _u):
            pass

        def set_resolved_model(self, _m):
            pass

        def record_litellm_headers(self, _h):
            pass

        def add_tool_call(self, _name):
            pass

    async def _fake_probe(_url, _hdrs, _name, _log):
        return True

    fake_agent = FakeAgent()
    # `make_review_agent` is imported lazily inside `continue_on_t1`,
    # so patch the source module's symbol; the lazy import resolves
    # to the patched value at call time.
    import cora.core.agent

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=fake_agent),
        patch.object(cont, "_probe_mcp_server", side_effect=_fake_probe),
    ):
        body, terminated, tools = asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://litellm.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="main",
                system_prompt="test prompt",
                prior_messages=[],
                budget=FakeBudget(),
                timeout_s=180,
                pr_number="1234",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={"Authorization": "Bearer x"},
                allowed_tools={"search_knowledge", "grep_repo", "git_show"},
                gha_log=lambda _m: None,
            )
        )

    assert terminated is None
    assert "T1 completed" in body
    assert tools == ["git_show", "grep_repo", "search_knowledge"]


def test_continue_on_t1_respects_max_iterations_override():
    """`max_iterations` kwarg should flow through to the
    `UsageLimits(request_limit=...)` the agent's iter loop runs
    under. Backs the `AGENT_REVIEW_T1_MAX_ITERATIONS` env path:
    the call-site in `agent_review.py` reads the env, parses to
    int, and passes it as `max_iterations=` — this test pins that
    the function actually uses the kwarg rather than ignoring it
    in favour of the default 6.
    """
    import sys
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.modules.pop("cora.core.continuation", None)
    import cora.core.continuation as cont

    captured: dict = {}

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *_args, **kwargs):
            captured["iter_kwargs"] = kwargs
            return _FakeIterCtx()

    class _FakeIterCtx:
        async def __aenter__(self):
            return _FakeAgentRun()

        async def __aexit__(self, *_):
            return False

    class _FakeRunUsage:
        request_tokens = 0
        response_tokens = 0
        total_tokens = 0

    class _FakeAgentRun:
        @property
        def result(self):
            return SimpleNamespace(
                output="🟢 done",
                usage=lambda: _FakeRunUsage(),
            )

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            # Empty turn stream — we just want to confirm the
            # iter() kwargs got the override.
            return
            yield  # pragma: no cover

    class FakeBudget:
        def add_usage(self, _u):
            pass

        def set_resolved_model(self, _m):
            pass

        def record_litellm_headers(self, _h):
            pass

        def add_tool_call(self, _name):
            pass

    async def _fake_probe(_url, _hdrs, _name, _log):
        return True

    import cora.core.agent

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=FakeAgent()),
        patch.object(cont, "_probe_mcp_server", side_effect=_fake_probe),
    ):
        asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://litellm.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="main",
                system_prompt="test prompt",
                prior_messages=[],
                budget=FakeBudget(),
                timeout_s=180,
                pr_number="1234",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={"Authorization": "Bearer x"},
                allowed_tools={"search_knowledge"},
                max_iterations=3,
                gha_log=lambda _m: None,
            )
        )

    usage_limits = captured["iter_kwargs"]["usage_limits"]
    assert usage_limits.request_limit == 3


def test_continue_on_t1_fresh_start_uses_initial_prompt():
    """Classifier-large-diff path: when `prior_messages` is
    empty AND `initial_user_prompt` is given, the function must call
    `agent.iter()` with the INITIAL prompt (not the resumption
    framing) and message_history=None. This is the skip-T0 entry
    shape — there's nothing to resume from.

    Pins the regression where the function would always feed
    `_CONTINUATION_PROMPT` regardless of inputs, which would lead T1
    to recap a non-existent prior conversation.
    """
    import sys
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.modules.pop("cora.core.continuation", None)
    import cora.core.continuation as cont

    captured: dict = {}

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *args, **kwargs):
            captured["positional"] = args
            captured["iter_kwargs"] = kwargs
            return _FakeIterCtx()

    class _FakeIterCtx:
        async def __aenter__(self):
            return _FakeAgentRun()

        async def __aexit__(self, *_):
            return False

    class _FakeRunUsage:
        request_tokens = 0
        response_tokens = 0
        total_tokens = 0

    class _FakeAgentRun:
        @property
        def result(self):
            return SimpleNamespace(
                output="🟢 done",
                usage=lambda: _FakeRunUsage(),
            )

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            return
            yield  # pragma: no cover

    class FakeBudget:
        def add_usage(self, _u):
            pass

        def set_resolved_model(self, _m):
            pass

        def record_litellm_headers(self, _h):
            pass

        def add_tool_call(self, _name):
            pass

    async def _fake_probe(_url, _hdrs, _name, _log):
        return True

    import cora.core.agent

    INITIAL = "Initial user prompt with the PR diff + retrieval bundle."

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=FakeAgent()),
        patch.object(cont, "_probe_mcp_server", side_effect=_fake_probe),
    ):
        asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://litellm.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="main",
                system_prompt="test prompt",
                prior_messages=[],
                initial_user_prompt=INITIAL,
                budget=FakeBudget(),
                timeout_s=180,
                pr_number="1234",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={"Authorization": "Bearer x"},
                allowed_tools={"search_knowledge"},
                gha_log=lambda _m: None,
            )
        )

    # First positional arg to `agent.iter(...)` is the leadin prompt.
    assert captured["positional"][0] == INITIAL
    # No message history when starting fresh — passing the empty list
    # would feed the agent a `[]` history, distinct from `None`. We
    # want `None` so the framework treats this as a fresh run.
    assert captured["iter_kwargs"]["message_history"] is None


def test_continue_on_t1_resume_still_uses_continuation_prompt():
    """Counterpart to the fresh-start test: when `prior_messages` is
    non-empty, the legacy resume framing must still fire even if an
    `initial_user_prompt` is also provided (defensive — caller
    shouldn't pass both, but if they do, resume wins).
    """
    import sys
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.modules.pop("cora.core.continuation", None)
    import cora.core.continuation as cont

    captured: dict = {}

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *args, **kwargs):
            captured["positional"] = args
            captured["iter_kwargs"] = kwargs
            return _FakeIterCtx()

    class _FakeIterCtx:
        async def __aenter__(self):
            return _FakeAgentRun()

        async def __aexit__(self, *_):
            return False

    class _FakeRunUsage:
        request_tokens = 0
        response_tokens = 0
        total_tokens = 0

    class _FakeAgentRun:
        @property
        def result(self):
            return SimpleNamespace(
                output="done",
                usage=lambda: _FakeRunUsage(),
            )

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            return
            yield  # pragma: no cover

    class FakeBudget:
        def add_usage(self, _u):
            pass

        def set_resolved_model(self, _m):
            pass

        def record_litellm_headers(self, _h):
            pass

        def add_tool_call(self, _name):
            pass

    async def _fake_probe(_url, _hdrs, _name, _log):
        return True

    import cora.core.agent

    # Sentinel — any non-empty list — exercises the resume branch.
    PRIOR = [SimpleNamespace(parts=[])]

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=FakeAgent()),
        patch.object(cont, "_probe_mcp_server", side_effect=_fake_probe),
    ):
        asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://litellm.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="main",
                system_prompt="test prompt",
                prior_messages=PRIOR,
                initial_user_prompt="should be ignored",
                budget=FakeBudget(),
                timeout_s=180,
                pr_number="1234",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={"Authorization": "Bearer x"},
                allowed_tools={"search_knowledge"},
                gha_log=lambda _m: None,
            )
        )

    assert captured["positional"][0] == cont._CONTINUATION_PROMPT
    assert captured["iter_kwargs"]["message_history"] is PRIOR


# ---------------------------------------------------------------------------
# Wall-hit-during-tool-call regression.
#
# When T0 wall-hits right after a `model_response` that requested tool
# calls but before the tool *results* are appended, the carried-forward
# history ends on unprocessed tool calls. Injecting the T1 continuation
# prompt on top of that raises pydantic-ai's
# `UserError: Cannot provide a new user prompt when the message history
# contains unprocessed tool calls.` — silently killing the T1 safety net
# on exactly the slow PRs that need it. These tests pin that the
# continuation reconciles the dangling tool calls so the resume injects
# cleanly.
# ---------------------------------------------------------------------------


def _history_ending_on_unprocessed_tool_call():
    """A T0-shaped history that ends on a `ModelResponse` whose
    `ToolCallPart` never got a matching `ToolReturnPart` — the exact
    observed wall-hit-mid-tool-call shape."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )

    return [
        ModelRequest(parts=[UserPromptPart(content="review this diff")]),
        ModelResponse(
            parts=[
                # Partial findings T0 produced before requesting the tool —
                # must survive reconciliation (we only fill the dangling
                # result, never drop the trailing response).
                TextPart(content="Partial finding: checking the egress policy…"),
                ToolCallPart(
                    tool_name="grep_repo",
                    args={"pattern": "egress"},
                    tool_call_id="tc-dangling",
                ),
            ],
            model_name="review",
        ),
    ]


def test_reconcile_unprocessed_tool_calls_makes_history_resumable():
    """The core regression: a real pydantic-ai history ending on an
    unprocessed tool call raises `UserError` when a new user prompt is
    injected — but the reconciled history does NOT. Uses a real
    `Agent(TestModel())` so the assertion rides on pydantic-ai's actual
    UserPromptNode check, not a stand-in approximation.
    """
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    from pydantic_ai.models.test import TestModel

    import cora.core.continuation as cont

    agent = Agent(TestModel())

    async def _run(history):
        async with agent.iter(
            cont._CONTINUATION_PROMPT, message_history=history
        ) as ar:
            async for _ in ar:
                pass
            return ar.result.output

    raw = _history_ending_on_unprocessed_tool_call()

    # Baseline: the raw carried history is what crashed the T1 resume.
    with pytest.raises(UserError, match="unprocessed tool calls"):
        asyncio.run(_run(raw))

    # The fix: reconciliation appends a synthetic tool-return so the
    # history is well-formed; the continuation prompt now injects cleanly.
    reconciled = cont._reconcile_unprocessed_tool_calls(raw, log=lambda _m: None)

    # A new trailing ModelRequest carrying the stub return was appended;
    # the original trailing ModelResponse (with its partial findings) is
    # left intact ahead of it.
    assert len(reconciled) == len(raw) + 1
    assert reconciled[:-1] == raw  # non-mutating: prefix unchanged
    stub_msg = reconciled[-1]
    assert isinstance(stub_msg, ModelRequest)
    returns = [p for p in stub_msg.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].tool_call_id == "tc-dangling"
    assert returns[0].tool_name == "grep_repo"

    # Must not raise — this is the safety net the regression lost.
    out = asyncio.run(_run(reconciled))
    assert out  # TestModel produces *some* terminal output

def test_continue_on_t1_reconciles_unprocessed_tool_calls_in_resume_path():
    """End-to-end through `continue_on_t1`: a `prior_messages` ending on
    unprocessed tool calls must reach `agent.iter()` reconciled — last
    message a `ModelRequest` carrying the stub return, not the dangling
    `ModelResponse`. Without the fix, pydantic-ai's UserPromptNode would
    reject the resume and the T1 escalation would produce no body.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    import cora.core.continuation as cont

    captured: dict = {}

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *args, **kwargs):
            captured["positional"] = args
            captured["iter_kwargs"] = kwargs
            return _FakeIterCtx()

    class _FakeIterCtx:
        async def __aenter__(self):
            return _FakeAgentRun()

        async def __aexit__(self, *_):
            return False

    class _FakeRunUsage:
        request_tokens = 0
        response_tokens = 0
        total_tokens = 0

    class _FakeAgentRun:
        @property
        def result(self):
            return SimpleNamespace(output="🟢 done", usage=lambda: _FakeRunUsage())

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            return
            yield  # pragma: no cover

    class FakeBudget:
        def add_usage(self, _u):
            pass

        def set_resolved_model(self, _m):
            pass

        def record_litellm_headers(self, _h):
            pass

        def add_tool_call(self, _name):
            pass

    async def _fake_probe(_url, _hdrs, _name, _log):
        return True

    import cora.core.agent

    prior = _history_ending_on_unprocessed_tool_call()

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=FakeAgent()),
        patch.object(cont, "_probe_mcp_server", side_effect=_fake_probe),
    ):
        body, terminated, _tools = asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://litellm.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="main",
                system_prompt="test prompt",
                prior_messages=prior,
                budget=FakeBudget(),
                timeout_s=180,
                pr_number="2213",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={"Authorization": "Bearer x"},
                allowed_tools={"grep_repo", "git_show"},
                gha_log=lambda _m: None,
            )
        )

    # Resume framing still fires (non-empty prior_messages).
    assert captured["positional"][0] == cont._CONTINUATION_PROMPT
    history_arg = captured["iter_kwargs"]["message_history"]
    # Reconciled: original trajectory preserved, stub return appended so
    # the last message is no longer a tool-call-bearing ModelResponse.
    assert history_arg[:-1] == prior
    assert isinstance(history_arg[-1], ModelRequest)
    assert any(
        isinstance(p, ToolReturnPart) and p.tool_call_id == "tc-dangling"
        for p in history_arg[-1].parts
    )
    # Soft-fail posture intact: a clean body, no terminated_reason.
    assert "done" in body
    assert terminated is None
