"""What T1 does when it runs out of iterations (#62).

Two defects met on this path and both are covered here:

- **No verdict at the cap.** `UsageLimits(request_limit=N)` raises before
  dispatching request N+1, so a model that spent response N on tool calls
  ended the run with no text. Across the tier boundary that was
  survivable — a capped T0 escalates and `_CONTINUATION_PROMPT` makes T1
  finish — but T1 has no tier after it, so its own cap-trip killed the
  review. `_elicit_final_answer` spends one tool-free turn instead.
- **Usage never accounted.** The accounting sat in the success tail, past
  the `early_terminated_reason` return, so every early exit reported
  `in_tokens=0 out_tokens=0 resolved_model=unknown` while its tool
  counters were plainly populated.

`continue_on_t1` does I/O against MCP + the LLM in normal use, so the
agent factory and the MCP probe are stubbed down to the smallest surface
that reaches the cap-trip handler — the same shape `test_continuation.py`
uses.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cora.core.budget import Budget


def _messages(*, dangling: bool = False):
    """A minimal T1 trajectory: one tool call, optionally left dangling
    (no matching return) the way a cap-trip mid-turn leaves it."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    out = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="grep_repo",
                    args={"pattern": "wall_time"},
                    tool_call_id="tc1",
                )
            ],
            model_name="cora-t1",
        )
    ]
    if not dangling:
        out.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="grep_repo",
                        content="src/cora/core/continuation.py:698",
                        tool_call_id="tc1",
                    )
                ]
            )
        )
    return out


class _CapTripUsage:
    """What the interrupted loop burned before the cap tripped."""

    input_tokens = 41_000
    output_tokens = 900
    total_tokens = 41_900
    tool_calls = 1


class _ElicitUsage:
    input_tokens = 42_000
    output_tokens = 700
    total_tokens = 42_700
    tool_calls = 0


def _build_agent(*, elicited: str, dangling: bool = False, run_raises=None):
    """A fake Agent whose iter loop always trips the iteration cap.

    Returns `(agent, calls)` — `calls` records the elicitation turn so a
    test can assert on the prompt, the history and the model settings.
    """
    from pydantic_ai.exceptions import UsageLimitExceeded

    calls: dict = {}
    messages = _messages(dangling=dangling)

    class _AgentRun:
        result = None

        def all_messages(self):
            return list(messages)

        def usage(self):
            return _CapTripUsage()

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            raise UsageLimitExceeded(
                "The next request would exceed the request_limit of 16"
            )
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

        async def run(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls.update(kwargs)
            if run_raises is not None:
                raise run_raises
            return SimpleNamespace(
                output=elicited,
                usage=lambda: _ElicitUsage(),
                all_messages=lambda: list(messages),
            )

    return _Agent(), calls


async def _probe_ok(_url, _hdrs, _name, _log):
    return True


def _run_t1(agent, *, budget, logs=None, **overrides):
    import cora.core.agent
    import cora.core.continuation as cont

    kwargs = {
        "endpoint_base_url": "http://gateway.test/v1",
        "llm_gateway_key": "dummy",
        "t1_model_alias": "core",
        "system_prompt": "review prompt",
        "prior_messages": [],
        "initial_user_prompt": "review this big PR",
        "budget": budget,
        "timeout_s": 180,
        "pr_number": "1234",
        "repo": "owner/repo",
        "mcp_url": "http://mcp.test/mcp",
        "mcp_headers": {},
        "allowed_tools": {"grep_repo", "git_show"},
        "max_iterations": 16,
        "gha_log": (logs.append if logs is not None else (lambda _m: None)),
    }
    kwargs.update(overrides)
    with (
        patch.object(cora.core.agent, "make_review_agent", return_value=agent),
        patch.object(cont, "_probe_mcp_server", side_effect=_probe_ok),
    ):
        return asyncio.run(cont.continue_on_t1(**kwargs))


def _budget() -> Budget:
    return Budget(max_input=1_000_000, max_output=100_000, max_iterations=16)


# ── the elicitation ──────────────────────────────────────────────────


def test_cap_trip_elicits_a_verdict_instead_of_dying():
    """The whole point of #62: a skip-T0 review that spends every request
    on tool calls used to end with no body at all. It now spends one more
    turn writing the review from what it verified."""
    agent, _calls = _build_agent(elicited="🟢 looks good\n\nVerdict: approve")

    body, terminated, tools = _run_t1(agent, budget=_budget())

    assert "Verdict: approve" in body
    # `None` — so the connector maps the entry tag onto the finish line
    # (`t1-classifier-large`) rather than reporting a failure.
    assert terminated is None
    assert tools == ["git_show", "grep_repo"]


def test_elicitation_forbids_tools_and_allows_exactly_one_request():
    """No calls are left, so the turn must not be spent asking for one.
    Run-level settings beat the agent-level `require_initial_tool_call`
    callable, so `tool_choice="none"` is the one that reaches the API."""
    from cora.core.continuation import FINAL_ANSWER_ELICITATION_PROMPT

    agent, calls = _build_agent(elicited="🟡 nit\n\nVerdict: approve")

    _run_t1(agent, budget=_budget())

    assert calls["prompt"] == FINAL_ANSWER_ELICITATION_PROMPT
    assert calls["model_settings"]["tool_choice"] == "none"
    assert calls["usage_limits"].request_limit == 1


def test_elicitation_resumes_the_capped_trajectory():
    """The turn is a write-up of work already done, so it must carry the
    trajectory forward — not restart from the initial prompt."""
    agent, calls = _build_agent(elicited="🟢 ok\n\nVerdict: approve")

    _run_t1(agent, budget=_budget())

    history = calls["message_history"]
    assert [type(m).__name__ for m in history][:1] == ["ModelResponse"]


def test_elicitation_reconciles_a_dangling_tool_call():
    """A cap-trip between the tool-call response and its return leaves the
    history unprocessed, which pydantic-ai refuses to extend with a new
    user prompt. Stub the missing return first."""
    agent, calls = _build_agent(
        elicited="🟢 ok\n\nVerdict: approve", dangling=True
    )

    body, _terminated, _tools = _run_t1(agent, budget=_budget())

    assert "Verdict: approve" in body
    history = calls["message_history"]
    returned = [
        p
        for m in history
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == "tool-return"
    ]
    assert [p.tool_call_id for p in returned] == ["tc1"]


def test_elicitation_failure_keeps_the_cap_trip_reason():
    """Best-effort: a backend that ignores `tool_choice="none"` (or errors)
    leaves the run exactly where it was before — never worse."""
    agent, _calls = _build_agent(
        elicited="", run_raises=RuntimeError("backend said no")
    )
    logs: list[str] = []

    body, terminated, _tools = _run_t1(agent, budget=_budget(), logs=logs)

    assert body == ""
    assert terminated == "max_iterations"
    assert any("outcome=errored" in line for line in logs)


def test_empty_elicitation_keeps_the_cap_trip_reason():
    """A turn that comes back blank is not a verdict."""
    agent, _calls = _build_agent(elicited="   ")
    logs: list[str] = []

    body, terminated, _tools = _run_t1(agent, budget=_budget(), logs=logs)

    assert body == ""
    assert terminated == "max_iterations"
    assert any("outcome=empty" in line for line in logs)


def test_elicitation_never_runs_on_a_wall_time_exit():
    """The recovery turn is scoped to `max_iterations` on purpose. A
    `wall_time` exit means the clock ran out, and one more model call is
    the thing that path cannot afford — so an exhausted wall trips the
    guard first and nothing is elicited."""
    agent, calls = _build_agent(elicited="🟢 ok\n\nVerdict: approve")

    body, terminated, _tools = _run_t1(
        agent,
        budget=_budget(),
        # Already in the past, so the loop's own deadline check fires
        # before the iteration cap can.
        loop_deadline_monotonic=0.0,
    )

    assert body == ""
    assert terminated == "wall_time"
    assert "prompt" not in calls


def test_elicitation_refused_when_grounding_never_happened():
    """`require_initial_tool_call` armed and unsatisfied: there is no
    verified trajectory to write up, so eliciting here would produce
    exactly the unverified assertion the contract exists to reject."""
    from cora.config import ReviewerConfig

    cfg = ReviewerConfig(
        repo="owner/repo",
        pr_number="1234",
        llm_api_key="k",
        model="test-model",
        require_initial_tool_call=True,
    )
    # `dangling=True` → the tool call never returned, so no tool succeeded.
    agent, calls = _build_agent(
        elicited="🟢 trust me\n\nVerdict: approve", dangling=True
    )

    body, terminated, _tools = _run_t1(agent, budget=_budget(), cfg=cfg)

    assert body == ""
    assert terminated == "required-tool-unhonored"
    assert "prompt" not in calls


# ── usage accounting ─────────────────────────────────────────────────


def test_cap_trip_accounts_the_tokens_it_burned():
    """The six reference runs reported `in_tokens=0 out_tokens=0` beside a
    populated `tools={...}`: the calls happened, so the tokens were spent
    — the accounting just sat past the early return.

    The elicitation is made to error here so the assertion is about the
    interrupted loop's own usage and nothing else."""
    agent, _calls = _build_agent(
        elicited="", run_raises=RuntimeError("backend said no")
    )
    budget = _budget()

    body, terminated, _tools = _run_t1(agent, budget=budget)

    assert (body, terminated) == ("", "max_iterations")
    assert budget.input_used == _CapTripUsage.input_tokens
    assert budget.output_used == _CapTripUsage.output_tokens


def test_elicitation_usage_is_added_on_top():
    """The recovery turn costs real tokens too, and they are additive —
    the loop and the elicitation are separate runs."""
    agent, _calls = _build_agent(elicited="🟢 ok\n\nVerdict: approve")
    budget = _budget()

    _run_t1(agent, budget=budget)

    assert budget.input_used == (
        _CapTripUsage.input_tokens + _ElicitUsage.input_tokens
    )
    assert budget.output_used == (
        _CapTripUsage.output_tokens + _ElicitUsage.output_tokens
    )


def test_cap_trip_resolves_the_backend():
    """`resolved_model=unknown` had the same cause as the zero tokens —
    same skipped block. The served name is on the messages either way."""
    agent, _calls = _build_agent(
        elicited="", run_raises=RuntimeError("backend said no")
    )
    budget = _budget()

    _run_t1(agent, budget=budget)

    assert budget.resolved_model == "cora-t1"


def test_unresolved_attribution_does_not_clobber_a_resolved_name():
    """Accounting runs more than once per review, so a later run's
    `unknown (…)` placeholder must not overwrite a name an earlier one
    resolved."""
    from cora.core.budget import account_run_usage

    budget = _budget()
    budget.set_resolved_model("piano")
    # No messages, no headers → nothing to resolve.
    account_run_usage(
        budget,
        SimpleNamespace(usage=lambda: _ElicitUsage(), all_messages=list),
        log=lambda _m: None,
        fallback="unknown (no header or body model)",
    )

    assert budget.resolved_model == "piano"


def test_accounting_a_missing_run_is_a_noop():
    """The agent context can fail to open at all; accounting must not
    care."""
    from cora.core.budget import account_run_usage

    budget = _budget()
    account_run_usage(budget, None, log=lambda _m: None, fallback="unknown")

    assert budget.input_used == 0
    assert budget.resolved_model is None


@pytest.mark.parametrize("field", ["input_used", "output_used"])
def test_accounting_survives_a_broken_usage_object(field):
    """Accounting is diagnostics — a framework shape surprise must never
    take a review down with it."""
    from cora.core.budget import account_run_usage

    budget = _budget()

    class _Exploding:
        @property
        def usage(self):
            raise RuntimeError("shape changed upstream")

        def all_messages(self):
            return []

    account_run_usage(
        budget, _Exploding(), log=lambda _m: None, fallback="unknown"
    )

    assert getattr(budget, field) == 0
