"""Reasoning-spiral detection + recovery (`cora.core.spiral`).

Unit tests use plain stand-in part/message objects (the module is
duck-typed on `part_kind` / `kind`, no `pydantic_ai` import needed) so
they stay fast + offline. The quick-mode recovery test mocks
`make_review_agent` the same way `test_quick_review.py` does.
"""

from __future__ import annotations

import asyncio

from cora.core import spiral

# Shared between the recovery tests + the patched capture_run_messages
# stand-in so the FakeAgent's first `run` can seed the captured messages.
_ACTIVE_CAPTURE: dict = {}


# ── Stand-ins ──────────────────────────────────────────────────────
class _Thinking:
    part_kind = "thinking"

    def __init__(self, content: str):
        self.content = content


class _Text:
    part_kind = "text"

    def __init__(self, content: str = "body"):
        self.content = content


class _ToolCall:
    part_kind = "tool-call"
    tool_name = "grep_repo"


class _Response:
    kind = "response"

    def __init__(self, parts):
        self.parts = parts


class _Request:
    kind = "request"

    def __init__(self, parts):
        self.parts = parts


# ── is_reasoning_spiral ────────────────────────────────────────────
def test_is_reasoning_spiral_thinking_only_is_true():
    msgs = [
        _Request([]),
        _Response([_Thinking("spiralling…")]),
    ]
    assert spiral.is_reasoning_spiral(msgs) is True


def test_is_reasoning_spiral_with_text_is_false():
    msgs = [_Response([_Thinking("thought"), _Text("verdict body")])]
    assert spiral.is_reasoning_spiral(msgs) is False


def test_is_reasoning_spiral_with_tool_call_is_false():
    msgs = [_Response([_Thinking("thought"), _ToolCall()])]
    assert spiral.is_reasoning_spiral(msgs) is False


def test_is_reasoning_spiral_no_thinking_is_false():
    msgs = [_Response([_Text("just text")])]
    assert spiral.is_reasoning_spiral(msgs) is False


def test_is_reasoning_spiral_empty_history_is_false():
    assert spiral.is_reasoning_spiral([]) is False
    assert spiral.is_reasoning_spiral(None) is False


def test_is_reasoning_spiral_uses_last_response():
    # An earlier thinking-only response then a clean final → not a spiral.
    msgs = [
        _Response([_Thinking("turn 1")]),
        _Request([]),
        _Response([_Text("done")]),
    ]
    assert spiral.is_reasoning_spiral(msgs) is False


# ── extract_partial_reasoning ──────────────────────────────────────
def test_extract_partial_reasoning_returns_tail():
    body = "HEAD" + "x" * 100 + "TAIL"
    msgs = [_Response([_Thinking(body)])]
    out = spiral.extract_partial_reasoning(msgs, char_cap=4)
    assert out == "TAIL"


def test_extract_partial_reasoning_under_cap_returned_whole():
    msgs = [_Response([_Thinking("short reasoning")])]
    assert spiral.extract_partial_reasoning(msgs, char_cap=8_000) == "short reasoning"


def test_extract_partial_reasoning_concatenates_multiple_thinking_parts():
    msgs = [_Response([_Thinking("aaa"), _Thinking("bbb")])]
    assert spiral.extract_partial_reasoning(msgs, char_cap=8_000) == "aaabbb"


def test_extract_partial_reasoning_no_response_returns_empty():
    assert spiral.extract_partial_reasoning([], char_cap=8_000) == ""


# ── build_recovery_leadin ──────────────────────────────────────────
def test_build_recovery_leadin_includes_directive_and_reasoning():
    leadin = spiral.build_recovery_leadin("the prior reasoning tail")
    # Directive: model has already analyzed, conclude now, keep brief.
    assert "already analyzed" in leadin
    assert "keep any further reasoning brief" in leadin
    # Required-format guidance preserved (verdict line + body).
    assert "Verdict:" in leadin
    assert "required output format" in leadin
    # Prior reasoning is carried in.
    assert "the prior reasoning tail" in leadin
    assert "<prior reasoning>" in leadin


# ── quick-mode recovery wiring ─────────────────────────────────────
def test_quick_recovery_returns_recovered_body(monkeypatch):
    """First `run` raises a spiral; recovery `run` returns a valid body.
    Assert the recovered body is returned AND the recovery call used the
    bounded `spiral_recovery_max_output_tokens` cap, reasoning still on."""
    import contextlib

    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from cora.config import ReviewerConfig
    from cora.core.budget import Budget
    from cora.core.quick_review import quick_review_call

    captured: dict = {}

    class FakeUsage:
        request_tokens = 1
        response_tokens = 2
        total_tokens = 3

    class FakeResult:
        output = "🟢 looks good\n\nRecovered body"

        def usage(self):
            return FakeUsage()

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        async def run(self, prompt, *, deps, model_settings, message_history=None):
            self.calls += 1
            if self.calls == 1:
                # Simulate the spiral: seed the active capture list with a
                # thinking-only response, then raise like the framework.
                _ACTIVE_CAPTURE["msgs"].extend(
                    [ModelResponse(parts=[ThinkingPart(content="HEADxxxxTAIL")])]
                )
                raise UnexpectedModelBehavior(
                    "Model token limit (32000) exceeded before any response "
                    "was generated"
                )
            captured["recovery_prompt"] = prompt
            captured["recovery_history"] = message_history
            captured["recovery_max_tokens"] = model_settings["max_tokens"]
            return FakeResult()

    monkeypatch.setattr(
        "cora.core.agent.make_review_agent",
        lambda config: FakeAgent(),
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers",
        dict,
    )

    # Patch capture_run_messages to a simple list we can seed into (the
    # real one is a framework contextvar; this keeps the test offline).
    @contextlib.contextmanager
    def _fake_capture():
        msgs: list = []
        _ACTIVE_CAPTURE["msgs"] = msgs
        try:
            yield msgs
        finally:
            _ACTIVE_CAPTURE.pop("msgs", None)

    monkeypatch.setattr("pydantic_ai.capture_run_messages", _fake_capture)

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
            cfg=ReviewerConfig(
                spiral_recovery=True,
                spiral_recovery_max_output_tokens=12345,
                spiral_recovery_reasoning_char_cap=4,
            ),
        )
    )

    assert err is None
    assert body == "🟢 looks good\n\nRecovered body"
    # Recovery used the bounded cap and carried the captured history.
    assert captured["recovery_max_tokens"] == 12345
    assert captured["recovery_history"] is not None
    # Lead-in carries the tail (char_cap=4 → "TAIL") + the directive.
    assert "TAIL" in captured["recovery_prompt"]
    assert "already analyzed" in captured["recovery_prompt"]


def test_quick_flag_off_no_recovery_attempt(monkeypatch):
    """Flag OFF: a spiral-shaped raise soft-fails exactly like today —
    no capture wrapper, no second `run`, the `agent run failed` tuple."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from cora.config import ReviewerConfig
    from cora.core.budget import Budget
    from cora.core.quick_review import quick_review_call

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        async def run(self, prompt, *, deps, model_settings, message_history=None):
            self.calls += 1
            raise UnexpectedModelBehavior(
                "Model token limit (32000) exceeded before any response"
            )

    agent_holder = {}

    def _make(config):
        agent_holder["agent"] = FakeAgent()
        return agent_holder["agent"]

    monkeypatch.setattr("cora.core.agent.make_review_agent", _make)
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", dict
    )
    # If the flag-off path wrongly entered the capture branch, this would
    # be invoked; assert it is NOT by making it explode.
    monkeypatch.setattr(
        "pydantic_ai.capture_run_messages",
        lambda: (_ for _ in ()).throw(
            AssertionError("capture_run_messages must not run when flag off")
        ),
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
            cfg=ReviewerConfig(),  # spiral_recovery defaults False
        )
    )

    assert body == ""
    assert err is not None and err.startswith("agent run failed:")
    # Exactly one run attempt — no recovery.
    assert agent_holder["agent"].calls == 1


def test_quick_recovery_also_spirals_soft_fails(monkeypatch):
    """If the bounded recovery run ALSO raises, fall back to the existing
    soft-fail tuple — no exception leaks, no regression."""
    import contextlib

    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from cora.config import ReviewerConfig
    from cora.core.budget import Budget
    from cora.core.quick_review import quick_review_call

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        async def run(self, prompt, *, deps, model_settings, message_history=None):
            self.calls += 1
            if self.calls == 1:
                _ACTIVE_CAPTURE["msgs"].extend(
                    [ModelResponse(parts=[ThinkingPart(content="tail")])]
                )
            raise UnexpectedModelBehavior("token limit exceeded before any response")

    agent_holder = {}

    def _make(config):
        agent_holder["agent"] = FakeAgent()
        return agent_holder["agent"]

    monkeypatch.setattr("cora.core.agent.make_review_agent", _make)
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", dict
    )

    @contextlib.contextmanager
    def _fake_capture():
        msgs: list = []
        _ACTIVE_CAPTURE["msgs"] = msgs
        try:
            yield msgs
        finally:
            _ACTIVE_CAPTURE.pop("msgs", None)

    monkeypatch.setattr("pydantic_ai.capture_run_messages", _fake_capture)

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
            cfg=ReviewerConfig(spiral_recovery=True),
        )
    )

    assert body == ""
    assert err is not None and err.startswith("agent run failed:")
    # Two attempts: the original + one bounded recovery.
    assert agent_holder["agent"].calls == 2


# ── deep-mode recovery wiring ──────────────────────────────────────
def _make_deep_fake_agent(*, recovery_result, recovery_capture, spiral_messages):
    """A fake pydantic-ai Agent for the deep path: supports `async with
    agent`, `agent.iter(...)` (an async-CM yielding a fake run whose
    `all_messages()` is the spiral), and `agent.run(...)` (recovery)."""
    import contextlib

    class FakeRun:
        def all_messages(self):
            return list(spiral_messages)

    class FakeAgent:
        def __init__(self):
            self.iter_calls = 0
            self.run_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @contextlib.asynccontextmanager
        async def iter(self, prompt, **kwargs):
            self.iter_calls += 1
            yield FakeRun()

        async def run(self, prompt, *, deps, model_settings, message_history=None):
            self.run_calls += 1
            recovery_capture["prompt"] = prompt
            recovery_capture["history"] = message_history
            recovery_capture["max_tokens"] = model_settings["max_tokens"]
            if recovery_result is None:
                raise _spiral_raise()
            return recovery_result

    return FakeAgent()


def _spiral_raise():
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    return UnexpectedModelBehavior(
        "Model token limit (32000) exceeded before any response was generated"
    )


def _wire_deep(monkeypatch, agent):
    """Common deep-path monkeypatching: probe OK, fake agent, and an
    `iter_with_turn_logging` that raises the spiral."""
    async def _probe_ok(*a, **k):
        return True

    async def _iter_raises(*a, **k):
        raise _spiral_raise()

    monkeypatch.setattr("cora.core.deep_review._probe_mcp_server", _probe_ok)
    monkeypatch.setattr(
        "cora.core.mcp_probe.probe_mcp_server", _probe_ok, raising=False
    )
    monkeypatch.setattr(
        "cora.core.agent.make_review_agent", lambda config: agent
    )
    monkeypatch.setattr(
        "cora.core.loop_logging.iter_with_turn_logging", _iter_raises
    )
    monkeypatch.setattr(
        "cora.core.litellm_capture.drain_captured_headers", dict
    )


def _run_deep(cfg):
    from cora.core.budget import Budget
    from cora.core.deep_review import deep_review_call

    budget = Budget(max_input=0, max_output=0, max_iterations=4)
    return asyncio.run(
        deep_review_call(
            endpoint_base_url="https://llm.example/v1",
            llm_gateway_key="key",
            model_alias="review",
            system_prompt="system",
            initial_user_prompt="prompt",
            budget=budget,
            timeout_s=30,
            pr_number="42",
            repo="o/r",
            mcp_url="http://mcp",
            mcp_headers={},
            allowed_tools=set(),
            cfg=cfg,
        )
    )


def test_deep_recovery_returns_recovered_body(monkeypatch):
    """Flag ON + thinking-only spiral → ONE bounded recovery run whose
    body is returned with no terminated_reason, bounded cap honoured."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from cora.config import ReviewerConfig

    spiral_messages = [ModelResponse(parts=[ThinkingPart(content="HEADyyyyTAIL")])]

    class FakeUsage:
        request_tokens = 1
        response_tokens = 2
        total_tokens = 3
        tool_calls = 0

    class FakeRecovery:
        output = "🟢 looks good\n\nDeep recovered body"

        def usage(self):
            return FakeUsage()

        def all_messages(self):
            return spiral_messages

    capture: dict = {}
    agent = _make_deep_fake_agent(
        recovery_result=FakeRecovery(),
        recovery_capture=capture,
        spiral_messages=spiral_messages,
    )
    _wire_deep(monkeypatch, agent)

    body, reason, _tools, _messages = _run_deep(
        ReviewerConfig(
            spiral_recovery=True,
            spiral_recovery_max_output_tokens=9999,
            spiral_recovery_reasoning_char_cap=4,
        )
    )

    assert reason is None
    assert body == "🟢 looks good\n\nDeep recovered body"
    assert capture["max_tokens"] == 9999
    # History carried forward is the captured spiral working state
    # (snapshotted via all_messages(), so equal-by-value not identity).
    assert capture["history"] == spiral_messages
    assert "TAIL" in capture["prompt"]
    assert agent.run_calls == 1


def test_deep_flag_off_no_recovery(monkeypatch):
    """Flag OFF: a spiral-shaped raise routes through the existing
    `agent-loop-errored` soft-fail — recovery `run` never fires."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from cora.config import ReviewerConfig

    spiral_messages = [ModelResponse(parts=[ThinkingPart(content="tail")])]
    capture: dict = {}
    agent = _make_deep_fake_agent(
        recovery_result=None,
        recovery_capture=capture,
        spiral_messages=spiral_messages,
    )
    _wire_deep(monkeypatch, agent)

    body, reason, _tools, _messages = _run_deep(ReviewerConfig())  # flag off

    assert body == ""
    assert reason is not None and reason.startswith("agent-loop-errored:")
    assert agent.run_calls == 0


def test_deep_recovery_also_fails_soft_fails(monkeypatch):
    """Recovery run also raises → existing soft-fail, no leak."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from cora.config import ReviewerConfig

    spiral_messages = [ModelResponse(parts=[ThinkingPart(content="tail")])]
    capture: dict = {}
    agent = _make_deep_fake_agent(
        recovery_result=None,  # recovery run raises
        recovery_capture=capture,
        spiral_messages=spiral_messages,
    )
    _wire_deep(monkeypatch, agent)

    body, reason, _tools, _messages = _run_deep(
        ReviewerConfig(spiral_recovery=True)
    )

    assert body == ""
    assert reason is not None and reason.startswith("agent-loop-errored:")
    assert "spiral-recovery-failed" in reason
    assert agent.run_calls == 1
