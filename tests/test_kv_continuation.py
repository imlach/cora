"""KV-continuation connector (the deep-mode escalation default).

The end-to-end behaviour through `run_review` is pinned in
`test_run_review.py` (`test_deep_wall_hit_continues_on_t1`,
`test_deep_skip_t0_starts_directly_on_t1`, …). These tests cover the two
pieces that wire that path behind the escalation seam in isolation:

- `_default_policy` connector selection — deep mode always carries
  the engine-importing `KvContinuationConnector`; quick mode stays on the
  dependency-free default.
- `KvContinuationConnector.escalate` — the `terminated_reason` mapping per
  entry path, and the no-body fall-back that preserves T0's reason.
"""

from __future__ import annotations

import asyncio
import importlib

from cora.config import ReviewerConfig
from cora.core.kv_continuation import KvContinuationConnector
from cora.escalation import (
    EscalationContext,
    EscalationOutcome,
    ReprefillConnector,
    Tier,
)
from cora.review import _continuation_tier_runner, _default_policy


def _cfg(**overrides) -> ReviewerConfig:
    base = {
        "repo": "owner/repo",
        "pr_number": "42",
        "llm_api_key": "test-key",
        "model": "test-model",
        "max_tool_iterations": 0,
    }
    base.update(overrides)
    return ReviewerConfig(**base)


# ── connector selection ──────────────────────────────────────────────


def test_default_policy_deep_t1_uses_kv_connector():
    """Deep mode + `t1_continuation` → a two-tier ladder escalating on
    wall-hit, driven by the KV-continuation connector."""
    pol = _default_policy(
        _cfg(max_tool_iterations=12),
        is_quick=False,
        t1_enabled=True,
        t1_model="core",
        t1_max_iterations=6,
    )
    assert isinstance(pol.connector, KvContinuationConnector)
    assert [t.model for t in pol.tiers] == ["test-model", "core"]
    assert pol.escalate_on == frozenset({"wall_hit"})


def test_default_policy_deep_single_tier_still_carries_kv_connector():
    """Forced entries (skip_t0 / per-call fresh) must drive the KV connector
    even on a single-tier ladder — so deep mode carries it regardless of the
    `t1_continuation` gate."""
    pol = _default_policy(
        _cfg(max_tool_iterations=12),
        is_quick=False,
        t1_enabled=False,
        t1_model="core",
        t1_max_iterations=6,
    )
    assert isinstance(pol.connector, KvContinuationConnector)
    assert [t.model for t in pol.tiers] == ["test-model"]
    assert pol.escalate_on == frozenset()


def test_default_policy_quick_keeps_generic_default():
    """Quick mode never escalates and never touches the engine-importing
    connector — it keeps the dependency-free re-prefill default."""
    pol = _default_policy(
        _cfg(),
        is_quick=True,
        t1_enabled=True,
        t1_model="core",
        t1_max_iterations=6,
    )
    assert isinstance(pol.connector, ReprefillConnector)
    assert not isinstance(pol.connector, KvContinuationConnector)
    assert [t.model for t in pol.tiers] == ["test-model"]


# ── the connector's dispatch ─────────────────────────────────────────


def _extra(monkeypatch, *, t1_kwargs, t1_return):
    """Wire a monkeypatched continue_on_t1 and return the shared
    `EscalationContext.extra` bag the connector reads."""

    async def fake_t1(**kwargs):
        t1_kwargs.update(kwargs)
        return t1_return

    # Resolve the *live* module object at call time: `test_continuation.py`
    # pops + reimports `cora.core.continuation`, so a module-level import here
    # could capture a stale object the connector's late binding never sees.
    cont_mod = importlib.import_module("cora.core.continuation")
    monkeypatch.setattr(cont_mod, "continue_on_t1", fake_t1)

    logs: list = []
    return {
        "endpoint_base_url": "http://gw/v1",
        "llm_gateway_key": "k",
        "system_prompt": "sys",
        "budget": object(),
        "timeout_s": 30,
        "pr_number": "42",
        "repo": "owner/repo",
        "mcp_url": None,
        "mcp_headers": None,
        "mcp_actions_url": None,
        "mcp_actions_headers": None,
        "web_fetch_url": None,
        "web_fetch_headers": None,
        "allowed_tools": {"git_show"},
        "tool_arg_defaults": {},
        "context_refresher": object(),
        "iter_log": logs.append,
        "gha_log": logs.append,
        "cfg": _cfg(),
        "git_provider": object(),
        "now": lambda: 100.0,
        "start": 0.0,
        "t0_wall_time_s": 600,
        "t1_wall_time_s": 600,
    }, logs


def _ctx(extra, *, entry, tag, terminated_reason, prev=None):
    return EscalationContext(
        next_tier=Tier("core", max_iterations=6),
        prev_context=prev if prev is not None else [{"role": "assistant", "content": "t0"}],
        initial_user_prompt="review this PR",
        entry=entry,
        tag=tag,
        terminated_reason=terminated_reason,
        extra=extra,
    )


def test_wall_hit_resumes_trajectory(monkeypatch):
    """Wall-hit entry: T0's trajectory is threaded as-is (object identity
    preserved, no fresh prompt), and a T1 body adopts the
    `t1-continuation` reason."""
    t1_kwargs: dict = {}
    prev = [{"role": "assistant", "content": "t0"}]
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nFine.", None, ["git_show"]),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="wall_hit",
               terminated_reason="wall_time", prev=prev)

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert isinstance(out, EscalationOutcome)
    assert out.body.startswith("🟢")
    assert out.terminated_reason == "t1-continuation"
    assert out.tools == ["git_show"]
    assert out.tier_ran == "core"
    # KV resume: same history object threaded, no fresh prompt.
    assert t1_kwargs["prior_messages"] is prev
    assert t1_kwargs["initial_user_prompt"] is None
    assert t1_kwargs["max_iterations"] == 6


def test_classifier_large_seeds_prompt(monkeypatch):
    """skip-T0 entry: T0 never ran, so the initial prompt seeds T1 and the
    reason marks the classifier-large path."""
    t1_kwargs: dict = {}
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nBig but fine.", None, []),
    )
    ctx = _ctx(extra, entry="fresh", tag="classifier_large",
               terminated_reason="classifier_large_start", prev=[])

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert out.terminated_reason == "t1-classifier-large"
    assert t1_kwargs["initial_user_prompt"] == "review this PR"


def test_per_call_fresh_seeds_prompt(monkeypatch):
    """per-call fresh start: T0 ran but produced nothing, so T1 starts
    fresh from the initial prompt; reason marks the retry."""
    t1_kwargs: dict = {}
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟡 nit\n\nA nit.", None, []),
    )
    ctx = _ctx(extra, entry="fresh", tag="per_call_fresh",
               terminated_reason="per_call_timeout", prev=[])

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert out.terminated_reason == "t1-per-call-retry"
    assert t1_kwargs["initial_user_prompt"] == "review this PR"


def test_no_t1_body_preserves_t0_reason(monkeypatch):
    """T1 also fails: the connector returns an empty body and hands back
    T0's `terminated_reason` so the original wall-hit isn't masked."""
    t1_kwargs: dict = {}
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("", "wall_time", []),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="wall_hit",
               terminated_reason="wall_time")

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert out.body == ""
    assert out.terminated_reason == "wall_time"   # T0's reason preserved
    assert out.tools == []
    assert out.tier_ran == "core"


def test_generic_tier_runner_threads_deadline(monkeypatch):
    """The generic runner (`_continuation_tier_runner`) enforces the same
    reserved-window deadline as the KV connector — a `ReprefillConnector`
    adopter's T1 must not run wall-unbounded (only per-call timeouts)."""
    t1_kwargs: dict = {}
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nFine.", None, []),
    )
    runner = _continuation_tier_runner(extra)
    asyncio.run(
        runner(
            Tier("core", max_iterations=6),
            [{"role": "assistant", "content": "t0"}],
            "review this PR",
        )
    )
    # now=100, start=0, t0_wall=600, t1_wall=600 → max(700, 1200) = 1200.
    assert t1_kwargs["loop_deadline_monotonic"] == 1200.0


def test_deadline_absorbs_t0_slack(monkeypatch):
    """T1's loop deadline is `max(now + t1_wall, start + t0_wall + t1_wall)`
    — if T0 finished early, T1 absorbs the slack."""
    t1_kwargs: dict = {}
    # now=100, start=0, t0_wall=600, t1_wall=600 → max(700, 1200) = 1200.
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nFine.", None, []),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="wall_hit",
               terminated_reason="wall_time")

    asyncio.run(KvContinuationConnector().escalate(ctx, _never))
    assert t1_kwargs["loop_deadline_monotonic"] == 1200.0


async def _never(*a):  # pragma: no cover — the KV connector ignores the runner
    raise AssertionError("KvContinuationConnector must not call run_tier")


def test_default_policy_reads_configured_triggers():
    """`cfg.escalation_triggers` steers the default ladder's escalate_on —
    a blocker/low_confidence deployment escalates on those without a
    custom policy object."""
    pol = _default_policy(
        _cfg(
            max_tool_iterations=12,
            escalation_triggers=frozenset({"wall_hit", "blocker"}),
        ),
        is_quick=False,
        t1_enabled=True,
        t1_model="core",
        t1_max_iterations=6,
    )
    assert pol.escalate_on == frozenset({"wall_hit", "blocker"})


def test_verdict_trigger_resumes_with_second_look_framing(monkeypatch):
    """A verdict-triggered entry resumes the completed trajectory with the
    second-look lead-in and maps to the `t1-verdict-trigger` reason."""
    import cora.core.continuation as cont

    t1_kwargs: dict = {}
    prev = [{"role": "assistant", "content": "t0"}]
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nDouble-checked.", None, []),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="verdict_trigger",
               terminated_reason=None, prev=prev)

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert out.terminated_reason == "t1-verdict-trigger"
    assert t1_kwargs["prior_messages"] is prev
    assert t1_kwargs["initial_user_prompt"] is None
    assert t1_kwargs["resume_prompt"] == cont.VERDICT_ESCALATION_PROMPT


def test_spiral_exhausted_resumes_with_stall_framing(monkeypatch):
    """An exhausted-spiral entry resumes the committed trajectory with the
    stalled-reasoning lead-in and maps to the `t1-spiral-escalation`
    reason."""
    import cora.core.continuation as cont

    t1_kwargs: dict = {}
    prev = [{"role": "assistant", "content": "t0"}]
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nRecovered on T1.", None, []),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="spiral_exhausted",
               terminated_reason="agent-loop-errored: spiral-redraw-exhausted",
               prev=prev)

    out = asyncio.run(KvContinuationConnector().escalate(ctx, _never))

    assert out.terminated_reason == "t1-spiral-escalation"
    assert t1_kwargs["prior_messages"] is prev
    assert t1_kwargs["initial_user_prompt"] is None
    assert t1_kwargs["resume_prompt"] == cont.SPIRAL_ESCALATION_PROMPT


def test_wall_hit_keeps_default_resume_framing(monkeypatch):
    """The wall-hit entry keeps the budget-continuation lead-in (no
    resume_prompt override)."""
    t1_kwargs: dict = {}
    extra, _ = _extra(
        monkeypatch,
        t1_kwargs=t1_kwargs,
        t1_return=("🟢 ok\n\nFine.", None, []),
    )
    ctx = _ctx(extra, entry="wall_hit", tag="wall_hit",
               terminated_reason="wall_time")

    asyncio.run(KvContinuationConnector().escalate(ctx, _never))
    assert t1_kwargs["resume_prompt"] is None


def test_t1_terminated_reasons_covers_every_entry_path():
    """`T1_TERMINATED_REASONS` is the tier-attribution set consumers use
    (e.g. the `tier_verdict` event's T0/T1 label). It must track
    `_T1_SUCCESS_REASON` exactly — a hand-picked subset mislabelled
    `t1-verdict-trigger` / `t1-per-call-retry` bodies as T0 in the
    structured log stream."""
    from cora.core.kv_continuation import (
        _T1_SUCCESS_REASON,
        T1_TERMINATED_REASONS,
    )

    assert frozenset(_T1_SUCCESS_REASON.values()) == T1_TERMINATED_REASONS
    assert "t1-verdict-trigger" in T1_TERMINATED_REASONS
    assert "t1-per-call-retry" in T1_TERMINATED_REASONS


def test_every_tiers_entry_tag_has_a_t1_success_reason():
    """Every `tag` the review driver can hand the connector must be a
    `_T1_SUCCESS_REASON` key. The first live `no_tool_use` escalation
    (cora 0.1.7) reached the finish line and died on exactly this:
    `_T1_SUCCESS_REASON[tag]` KeyError'd, cancelling a review whose T1
    had already produced a verdict."""
    from cora.core.kv_continuation import _T1_SUCCESS_REASON

    # The complete tag vocabulary emitted by _tiers.py's entry mapping.
    tags = {
        "classifier_large",
        "per_call_fresh",
        "wall_hit",
        "verdict_trigger",
        "spiral_exhausted",
        "no_tool_use",
    }
    assert tags <= set(_T1_SUCCESS_REASON)
