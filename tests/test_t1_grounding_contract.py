"""The grounding contract must be judged on T1's real trajectory (#65).

`continue_on_t1`'s clean path set `result` but never `messages`, while the
success tail judged `require_initial_tool_call` against that same list. So
a T1 that ran a full, well-grounded review was measured against an empty
history, `first_successful_tool_name([])` returned `None`, and the
finished body was discarded as `required-tool-unhonored`.

It only ever bit fresh T1 starts, which is what made it look like a model
refusal: the contract arms only when `prior_messages` carries no
successful tool call, so a resume entry inherits T0's calls and skips the
check entirely. One observed run reported `required-tool-unhonored`
beside `tools={'git_show': 2, 'grep_repo': 5, 'read_note': 1}` — the
model had grounded eight times.

The backstop itself is still wanted: a provider that ignores
`tool_choice` can still return a zero-call verdict, and that must not
post. These tests pin both directions.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from cora.config import ReviewerConfig
from cora.core.budget import Budget


class _Usage:
    input_tokens = 30_000
    output_tokens = 600
    total_tokens = 30_600
    tool_calls = 1


def _trajectory(*, tool_outcome: str | None):
    """A finished T1 run's history. `tool_outcome=None` → the model never
    called a tool at all (the genuine refusal the contract exists for)."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )

    if tool_outcome is None:
        return [
            ModelResponse(
                parts=[TextPart(content="🟢 looks good\n\nTrust me.")],
                model_name="cora-t1",
            )
        ]
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="grep_repo",
                    args={"pattern": "verdict"},
                    tool_call_id="tc1",
                )
            ],
            model_name="cora-t1",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="grep_repo",
                    content="src/cora/result.py:12",
                    tool_call_id="tc1",
                    outcome=tool_outcome,
                )
            ]
        ),
        ModelResponse(
            parts=[TextPart(content="🟢 looks good\n\nChecked it.")],
            model_name="cora-t1",
        ),
    ]


def _agent(*, body: str, tool_outcome: str | None):
    """A fake Agent whose iter loop completes cleanly — no exception, so
    the run leaves through the success tail."""
    messages = _trajectory(tool_outcome=tool_outcome)

    class _AgentRun:
        @property
        def result(self):
            return SimpleNamespace(
                output=body,
                usage=lambda: _Usage(),
                all_messages=lambda: list(messages),
            )

        def all_messages(self):
            return list(messages)

        def usage(self):
            return _Usage()

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            # Clean completion: the loop simply ends.
            return
            yield  # pragma: no cover — makes this an async generator

    class _IterCtx:
        async def __aenter__(self):
            return _AgentRun()

        async def __aexit__(self, *_):
            return False

    class _Agent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def iter(self, *_a, **_kw):
            return _IterCtx()

        async def run(self, *_a, **_kw):  # pragma: no cover — not reached
            raise AssertionError("a clean completion must not re-run the agent")

    return _Agent()


async def _probe_ok(_url, _hdrs, _name, _log):
    return True


def _cfg(**overrides) -> ReviewerConfig:
    base = {
        "repo": "owner/repo",
        "pr_number": "1234",
        "llm_api_key": "k",
        "model": "test-model",
        "require_initial_tool_call": True,
    }
    base.update(overrides)
    return ReviewerConfig(**base)


def _run_t1(agent, *, cfg):
    import cora.core.agent
    import cora.core.continuation as cont

    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=agent),
        patch.object(cont, "_probe_mcp_server", side_effect=_probe_ok),
    ):
        return asyncio.run(
            cont.continue_on_t1(
                endpoint_base_url="http://gateway.test/v1",
                llm_gateway_key="dummy",
                t1_model_alias="core",
                system_prompt="review prompt",
                # Fresh start — this is what arms the contract, and the
                # only shape the defect could reach.
                prior_messages=[],
                initial_user_prompt="review this PR",
                budget=Budget(
                    max_input=1_000_000, max_output=100_000, max_iterations=16
                ),
                timeout_s=180,
                pr_number="1234",
                repo="owner/repo",
                mcp_url="http://mcp.test/mcp",
                mcp_headers={},
                allowed_tools={"grep_repo", "git_show"},
                max_iterations=16,
                gha_log=lambda _m: None,
                cfg=cfg,
            )
        )


def test_grounded_fresh_t1_keeps_its_review():
    """The regression. A clean T1 completion whose trajectory contains a
    successful tool call must post — the contract is satisfied by real
    calls, and judging it against an unpopulated list threw the review
    away."""
    agent = _agent(body="🟢 looks good\n\nChecked it.", tool_outcome="success")

    body, terminated, _tools = _run_t1(agent, cfg=_cfg())

    assert terminated is None
    assert "Checked it." in body


def test_zero_call_fresh_t1_still_fails_closed():
    """The backstop still matters: a provider that ignores `tool_choice`
    can return a verdict with no tool call behind it, and that verdict is
    unverified by construction."""
    agent = _agent(body="🟢 looks good\n\nTrust me.", tool_outcome=None)

    body, terminated, _tools = _run_t1(agent, cfg=_cfg())

    assert terminated == "required-tool-unhonored"
    assert body == ""


def test_failed_tool_call_does_not_satisfy_the_contract():
    """A call that errored grounds nothing. `tool_choice` can force a
    call but not a result, so `outcome` is what the contract reads."""
    agent = _agent(body="🟢 looks good\n\nTried.", tool_outcome="failed")

    body, terminated, _tools = _run_t1(agent, cfg=_cfg())

    assert terminated == "required-tool-unhonored"
    assert body == ""


def test_contract_off_posts_regardless():
    """Deployments that never armed the contract are untouched."""
    agent = _agent(body="🟢 looks good\n\nTrust me.", tool_outcome=None)

    body, terminated, _tools = _run_t1(
        agent, cfg=_cfg(require_initial_tool_call=False)
    )

    assert terminated is None
    assert "Trust me." in body
