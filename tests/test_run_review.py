"""End-to-end orchestration tests for `run_review` — NullReporter /
recording reporter + a stubbed LLM layer (the quick/deep/continuation
call functions are monkeypatched), so no network, no GitHub, no model.

What's covered mirrors the entrypoint's load-bearing decision points:
quick happy path, deep wall-hit → T1 continuation (and the policy gate
that keeps it OFF by default), leak-suppressed verdict, preflight skip,
and the SIGTERM-guard idempotency that replaced the old `done` flag.
"""

from __future__ import annotations

import time

import cora.core.pr_context as prc_mod
import cora.review._output as output_mod
import cora.core.pretrigger as pretrigger_mod
import cora.review as review_mod
from cora.config import ReviewerConfig
from cora.core.budget import Budget
from cora.providers.reporter import NullReporter, Reporter
from cora.providers.retrieval import NullRetrievalProvider
from cora.review import run_review
from cora.trigger import TriggerPolicy


class RecordingReporter(Reporter):
    """Side-effect recorder with GitHubReporter's check-run semantics:
    `check_open` after `open_progress`, first terminal `complete_check`
    wins (later calls no-op)."""

    def __init__(self) -> None:
        self.progress_opened: list[str] = []
        self.in_progress_posted = 0
        self.complete_calls: list[dict] = []
        self.summaries: list[dict] = []
        self.reviews: list = []
        self.skips: list[str] = []
        self.automerge_pauses = 0
        self.dispatches: list[dict] = []
        self.labels: list[tuple[str, str | None]] = []
        self._open = False

    def open_progress(self, head_sha: str) -> None:
        self.progress_opened.append(head_sha)
        self._open = True

    @property
    def check_open(self) -> bool:
        return self._open

    def post_in_progress(self) -> None:
        self.in_progress_posted += 1

    def complete_check(
        self,
        *,
        verdict_line,
        conclusion,
        budget=None,
        wall_time_s=0.0,
        terminated_reason=None,
    ) -> None:
        if not self._open:
            return
        self._open = False
        self.complete_calls.append(
            {
                "verdict_line": verdict_line,
                "conclusion": conclusion,
                "budget": budget,
                "wall_time_s": wall_time_s,
                "terminated_reason": terminated_reason,
            }
        )

    def write_summary(self, **kwargs) -> None:
        self.summaries.append(kwargs)

    def post_review(self, result) -> None:
        self.reviews.append(result)

    def post_skip(self, reason: str) -> None:
        self.skips.append(reason)

    def pause_automerge(self) -> bool:
        self.automerge_pauses += 1
        return True

    def dispatch_patch(self, **kwargs) -> dict:
        self.dispatches.append(kwargs)
        return {
            "inline_url": None,
            "inline_count": 0,
            "draft_url": None,
            "draft_count": 0,
            "source_branch_commit_sha": None,
            "source_branch_count": 0,
            "rejected": [],
            "error": None,
        }

    def apply_label(self, label, *, pr_number=None):
        self.labels.append((label, pr_number))
        return True, None


def _metadata() -> dict:
    return {
        "title": "Bump widget to 2.0",
        "body": "A change.",
        "labels": [],
        "author": {"login": "alice", "is_bot": False},
        "baseRefName": "main",
        "headRefName": "feat/x",
        "isCrossRepository": False,
        "additions": 1,
        "deletions": 1,
        "changedFiles": 1,
    }


def _cfg(**overrides) -> ReviewerConfig:
    base = dict(
        repo="owner/repo",
        pr_number="42",
        llm_api_key="test-key",
        model="test-model",
        max_tool_iterations=0,
    )
    base.update(overrides)
    return ReviewerConfig(**base)


def _patch_common(monkeypatch) -> None:
    """Neutralise every network-touching collaborator and stray env knob,
    and reset the module-level SIGTERM guard state. Patches land on the
    *source* modules (`cora.core.pr_context` / `cora.core.pretrigger`):
    the review phase modules call them through the module attribute, so
    one patch covers every call site in the pipeline."""
    monkeypatch.setattr(prc_mod, "fetch_pr_metadata", lambda pr: _metadata())
    monkeypatch.setattr(
        prc_mod, "fetch_pr_diff", lambda pr: "diff --git a/f b/f\n+x\n"
    )
    monkeypatch.setattr(
        prc_mod, "gather_ci_context", lambda *a, **k: None
    )
    monkeypatch.setattr(
        prc_mod, "fetch_classifier_rationale", lambda *a, **k: None
    )
    monkeypatch.setattr(prc_mod, "_pr_head_sha", lambda: "headsha123")
    monkeypatch.setattr(
        prc_mod, "fetch_author_association", lambda *a, **k: "MEMBER"
    )
    monkeypatch.setattr(
        prc_mod, "latest_commit_author_login", lambda *a, **k: None
    )

    async def _no_pretrigger(*a, **k):
        return None

    monkeypatch.setattr(pretrigger_mod, "fire_pretrigger", _no_pretrigger)

    for var in (
        "AGENT_REVIEW_T1_CONTINUATION",
        "AGENT_REVIEW_SKIP_T0",
        "AGENT_REVIEW_T2_DISAGREEMENT",
        "CLASSIFIER_LABEL",
        "WALL_TIME_S",
        "T0_WALL_TIME_S",
        "T1_WALL_TIME_S",
        "T1_MODEL",
        "GITHUB_EVENT_PATH",
        "GITHUB_STEP_SUMMARY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)

    review_mod._TIMEOUT_GUARD.update(
        reporter=None, budget=None, start=None, terminated_reason=None
    )


# ── Quick mode, happy path ───────────────────────────────────────────


def test_quick_happy_path(monkeypatch):
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        assert kwargs["cfg"] is cfg  # config threading reaches the call
        return "🟢 looks good\n\nClean change.", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    cfg = _cfg()
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.verdict == "looks good"
    assert result.conclusion == "success"
    assert result.mode == "quick"
    assert result.terminated_reason is None
    assert result.body.startswith("🟢 looks good")
    assert result.tiers_run == ["test-model"]
    assert result.pause_automerge is False

    # Reporter sequence: check opened on the head SHA, finalized once
    # with the verdict, summary written, review posted; quick mode never
    # posts the in-progress placeholder and nothing was skipped/paused.
    assert rep.progress_opened == ["headsha123"]
    assert rep.in_progress_posted == 0
    assert len(rep.complete_calls) == 1
    assert rep.complete_calls[0]["conclusion"] == "success"
    assert rep.complete_calls[0]["verdict_line"] == "verdict: looks good"
    assert len(rep.summaries) == 1
    assert len(rep.reviews) == 1 and rep.reviews[0] is result
    assert rep.skips == []
    assert rep.automerge_pauses == 0


# ── Deep mode: wall-hit → T1 continuation ────────────────────────────


def test_deep_wall_hit_continues_on_t1(monkeypatch):
    """`cfg.t1_continuation` drives the escalation — no env involved
    (`_patch_common` scrubbed the legacy knobs); the cfg tier fields
    reach the T1 dispatch."""
    _patch_common(monkeypatch)
    import cora.core.continuation as cont_mod
    import cora.core.deep_review as deep_mod

    t0_messages = [{"role": "assistant", "content": "working..."}]

    async def fake_deep(**kwargs):
        return "", "wall_time", ["grep_repo"], t0_messages

    seen: dict = {}

    async def fake_t1(**kwargs):
        seen.update(kwargs)
        return "🟡 minor\n\nA nit.", None, ["git_show"]

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(cont_mod, "continue_on_t1", fake_t1)

    cfg = _cfg(max_tool_iterations=12, t1_continuation=True)
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    # T1 adopted the verdict; the finish reason marks the escalated path.
    assert result.terminated_reason == "t1-continuation"
    assert result.verdict == "minor"
    assert result.conclusion == "neutral"
    assert result.tiers_run == ["test-model", "core"]
    assert result.tools_available == ["git_show", "grep_repo"]

    # The wall-hit resume hands the T0 trajectory forward — NOT a fresh
    # start (initial_user_prompt only seeds the fresh-start entries).
    assert seen["prior_messages"] is t0_messages
    assert seen["initial_user_prompt"] is None
    assert seen["t1_model_alias"] == "core"
    assert seen["max_iterations"] == 6

    # Deep mode posts the in-progress placeholder; finalize ran once.
    assert rep.in_progress_posted == 1
    assert len(rep.complete_calls) == 1
    assert rep.complete_calls[0]["conclusion"] == "neutral"


def test_deep_cfg_tier_fields_reach_t1_dispatch(monkeypatch):
    """Non-default `t1_model` / `t1_max_iterations` cfg values steer
    the T1 dispatch and the tiers_run trail — config, not env."""
    _patch_common(monkeypatch)
    import cora.core.continuation as cont_mod
    import cora.core.deep_review as deep_mod

    async def fake_deep(**kwargs):
        return "", "wall_time", [], [{"role": "assistant", "content": "..."}]

    seen: dict = {}

    async def fake_t1(**kwargs):
        seen.update(kwargs)
        return "🟢 looks good\n\nFine.", None, []

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(cont_mod, "continue_on_t1", fake_t1)

    cfg = _cfg(
        max_tool_iterations=12,
        t1_continuation=True,
        t1_model="big-ctx",
        t1_max_iterations=9,
    )
    result = run_review(
        cfg, reporter=RecordingReporter(), retrieval=NullRetrievalProvider()
    )

    assert seen["t1_model_alias"] == "big-ctx"
    assert seen["max_iterations"] == 9
    assert result.tiers_run == ["test-model", "big-ctx"]


def test_deep_skip_t0_starts_directly_on_t1(monkeypatch):
    """`cfg.skip_t0` (classifier-large-diff) bypasses T0 entirely: the
    forced T1 entry runs fresh with the initial prompt even though the
    continuation gate is off, and the finish reason marks the path."""
    _patch_common(monkeypatch)
    import cora.core.continuation as cont_mod
    import cora.core.deep_review as deep_mod

    async def fake_deep(**kwargs):  # pragma: no cover — must not run
        raise AssertionError("T0 must not be dispatched when skip_t0 is set")

    seen: dict = {}

    async def fake_t1(**kwargs):
        seen.update(kwargs)
        return "🟢 looks good\n\nBig but fine.", None, []

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(cont_mod, "continue_on_t1", fake_t1)

    cfg = _cfg(max_tool_iterations=12, skip_t0=True)
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.terminated_reason == "t1-classifier-large"
    assert result.verdict == "looks good"
    # Fresh start: the initial prompt seeds T1; no prior trajectory.
    assert seen["initial_user_prompt"] is not None
    assert seen["prior_messages"] == []
    # Only the T1 alias ran — T0 never entered the trail.
    assert result.tiers_run == ["core"]


def test_deep_t2_disagreement_via_cfg(monkeypatch):
    """`cfg.t2_disagreement` fires the second opinion (no env), the cfg
    t2 alias/cap reach the dispatch, and the gap=1 resolution adopts
    the conservative T2 verdict."""
    _patch_common(monkeypatch)
    import cora.core.deep_review as deep_mod
    import cora.core.t2_dispatch as t2_mod

    async def fake_deep(**kwargs):
        return (
            "🟢 looks good\n\nNothing concerning.",
            None,
            [],
            [{"role": "assistant", "content": "done"}],
        )

    seen: dict = {}

    async def fake_t2(**kwargs):
        seen.update(kwargs)
        return "🟡 minor\n\nOne nit T0 missed.", None, []

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(t2_mod, "call_t2_alt_reviewer", fake_t2)

    cfg = _cfg(
        max_tool_iterations=12,
        t2_disagreement=True,
        t2_model="second-opinion",
        t2_max_iterations=4,
    )
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert seen["t2_model_alias"] == "second-opinion"
    assert seen["max_iterations"] == 4
    assert result.tiers_run == ["test-model", "second-opinion"]
    # gap=1 → adopt_conservative: T2's minor leads, banner up top,
    # dissenting T0 body folded into the details block.
    assert result.verdict == "minor"
    assert result.body.startswith("🔄 T0/T2 disagreed — adopted T2")
    assert "One nit T0 missed." in result.body


def test_deep_t2_disagreement_off_by_default(monkeypatch):
    """Default cfg keeps the soak gate closed: no T2 dispatch fires."""
    _patch_common(monkeypatch)
    import cora.core.deep_review as deep_mod
    import cora.core.t2_dispatch as t2_mod

    async def fake_deep(**kwargs):
        return "🟢 looks good\n\nFine.", None, [], [{"role": "assistant", "content": "x"}]

    async def fake_t2(**kwargs):  # pragma: no cover — must not run
        raise AssertionError("T2 must not fire when t2_disagreement is off")

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(t2_mod, "call_t2_alt_reviewer", fake_t2)

    cfg = _cfg(max_tool_iterations=12)
    result = run_review(
        cfg, reporter=RecordingReporter(), retrieval=NullRetrievalProvider()
    )
    assert result.verdict == "looks good"
    assert result.tiers_run == ["test-model"]


def test_null_second_opinion_never_dispatches(monkeypatch):
    """An explicit `NullSecondOpinion` overrides the config-default T2
    provider: even with `t2_disagreement=True`, no second review fires and
    the primary verdict reaches the comment untouched (no banner)."""
    _patch_common(monkeypatch)
    import cora.core.deep_review as deep_mod
    import cora.core.t2_dispatch as t2_mod
    from cora.second_opinion import NullSecondOpinion

    async def fake_deep(**kwargs):
        return (
            "🟢 looks good\n\nNothing concerning.",
            None,
            [],
            [{"role": "assistant", "content": "done"}],
        )

    async def fake_t2(**kwargs):  # pragma: no cover — must not run
        raise AssertionError("NullSecondOpinion must not dispatch a second review")

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(t2_mod, "call_t2_alt_reviewer", fake_t2)

    # t2_disagreement=True would have fired the default T2 provider — but the
    # explicit NullSecondOpinion takes precedence and short-circuits it.
    cfg = _cfg(max_tool_iterations=12, t2_disagreement=True, t2_model="second-opinion")
    result = run_review(
        cfg,
        reporter=RecordingReporter(),
        retrieval=NullRetrievalProvider(),
        second_opinion=NullSecondOpinion(),
    )
    assert result.verdict == "looks good"
    assert result.body.startswith("🟢 looks good")
    assert "🔄" not in result.body  # no disagreement banner
    # Only the primary tier ran — the alt-reviewer never entered the trail.
    assert result.tiers_run == ["test-model"]


def _patch_directive() -> str:
    return (
        '```json propose_patch\n'
        '{"title": "t", "body": "b", "edits": '
        '[{"path": "f.txt", "old_string": "x", "new_string": "y"}]}\n'
        '```'
    )


def test_propose_patch_dispatch_disabled_by_default(monkeypatch):
    """Fresh adopters get comment/check-run-only behavior: a model-emitted
    directive is stripped from the posted body and no write dispatcher runs."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return f"🟡 minor\n\nA finding.\n\n{_patch_directive()}\n", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    rep = RecordingReporter()
    result = run_review(_cfg(), reporter=rep, retrieval=NullRetrievalProvider())

    assert result.verdict == "minor"
    assert "propose_patch" not in result.body
    assert "patch dispatch is disabled" in result.body
    assert rep.dispatches == []


def test_propose_patch_dispatch_opt_in_reaches_reporter(monkeypatch):
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return f"🟡 minor\n\nA finding.\n\n{_patch_directive()}\n", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    cfg = _cfg(propose_patch_dispatch=True)
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.verdict == "minor"
    assert "patch dispatch is disabled" not in result.body
    assert len(rep.dispatches) == 1


def test_second_opinion_from_config_defaults_to_t2():
    """`SecondOpinionProvider.from_config` returns the built-in T2 impl by
    default; its `should_dispatch` honours the `t2_disagreement` opt-in so
    the default config never dispatches (byte-identical to the inline gate)."""
    from cora.core.t2_second_opinion import T2SecondOpinion
    from cora.second_opinion import SecondOpinionProvider

    provider = SecondOpinionProvider.from_config(_cfg())
    assert isinstance(provider, T2SecondOpinion)

    # Default config: t2_disagreement off → no dispatch even with a body.
    off = _cfg(t2_disagreement=False)
    assert not provider.should_dispatch(
        cfg=off, is_quick=False, primary_body="🟢 looks good\n\nfine"
    )
    # Opt-in on, deep mode, non-blocker body → dispatch.
    on = _cfg(t2_disagreement=True)
    assert provider.should_dispatch(
        cfg=on, is_quick=False, primary_body="🟢 looks good\n\nfine"
    )
    # Quick mode never dispatches regardless of the flag.
    assert not provider.should_dispatch(
        cfg=on, is_quick=True, primary_body="🟢 looks good\n\nfine"
    )


def test_deep_wall_hit_does_not_escalate_when_continuation_off(monkeypatch):
    """Default policy is single-tier: a wall-hit with the continuation
    flag unset falls through to the no-final-body failure path."""
    _patch_common(monkeypatch)
    import cora.core.continuation as cont_mod
    import cora.core.deep_review as deep_mod

    async def fake_deep(**kwargs):
        return "", "wall_time", [], [{"role": "assistant", "content": "..."}]

    async def fake_t1(**kwargs):  # pragma: no cover — must not run
        raise AssertionError("T1 must not be dispatched when the flag is off")

    monkeypatch.setattr(deep_mod, "deep_review_call", fake_deep)
    monkeypatch.setattr(cont_mod, "continue_on_t1", fake_t1)

    cfg = _cfg(max_tool_iterations=12)
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.conclusion == "failure"
    assert result.verdict is None
    assert result.terminated_reason == "wall_time"
    assert rep.complete_calls[0]["verdict_line"] == "no review produced"
    assert any("no final review" in s for s in rep.skips)
    assert rep.reviews == []


# ── Leak-suppressed verdict ──────────────────────────────────────────


def test_leak_suppressed_verdict(monkeypatch):
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    leaked = "Let me think about this diff. First I will check the loop."

    async def fake_quick(**kwargs):
        return leaked, None

    async def fake_retry(**kwargs):
        # The bounded reformat retry also fails to produce a marker.
        return "Hmm, still just thinking out loud here.", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(output_mod, "quick_review_retry_for_format", fake_retry)

    cfg = _cfg()
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    # Body suppressed: failure conclusion, leak-flagged summary, skip
    # comment instead of a review comment.
    assert result.verdict is None
    assert result.conclusion == "failure"
    assert result.body == ""
    assert rep.complete_calls[0]["conclusion"] == "failure"
    assert "reasoning leak" in rep.complete_calls[0]["verdict_line"]
    assert rep.summaries[0]["is_leak"] is True
    assert any("reasoning-only output" in s for s in rep.skips)
    assert rep.reviews == []


# ── Preflight skip path ──────────────────────────────────────────────


def test_skip_when_llm_key_missing(monkeypatch):
    _patch_common(monkeypatch)
    cfg = _cfg(llm_api_key=None)
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.conclusion == "cancelled"
    assert result.terminated_reason == "secret-missing"
    # Skip comment posted BEFORE the check finalize, same order as the
    # entrypoint; the check carries the zero-placeholder budget.
    assert len(rep.skips) == 1 and "LLM_GATEWAY_KEY" in rep.skips[0]
    assert len(rep.complete_calls) == 1
    call = rep.complete_calls[0]
    assert call["conclusion"] == "cancelled"
    assert call["terminated_reason"] == "secret-missing"
    assert call["budget"] is None  # zero placeholder substituted downstream
    assert rep.reviews == [] and rep.summaries == []


def test_missing_identity_soft_fails_without_side_effects(monkeypatch):
    _patch_common(monkeypatch)
    result = run_review(_cfg(repo=""), reporter=NullReporter())
    assert result.conclusion == "cancelled"
    assert result.terminated_reason == "missing-pr-identity"


# ── SIGTERM guard idempotency ────────────────────────────────────────


def test_sigterm_guard_noops_after_normal_finalize(monkeypatch):
    """A late signal after the normal terminal path must not clobber the
    real conclusion — `complete_check` idempotency replaces the old
    `done` flag."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    async def fake_quick(**kwargs):
        return "🟢 looks good\n\nFine.", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    rep = RecordingReporter()
    run_review(_cfg(), reporter=rep, retrieval=NullRetrievalProvider())
    assert len(rep.complete_calls) == 1
    assert rep.complete_calls[0]["conclusion"] == "success"

    # The guard still references this run's reporter; a late SIGTERM
    # finalize must no-op because the check is already terminal.
    assert review_mod._TIMEOUT_GUARD["reporter"] is rep
    review_mod._finalize_check_on_signal()
    assert len(rep.complete_calls) == 1
    assert rep.complete_calls[0]["conclusion"] == "success"


def test_sigterm_guard_finalizes_open_check_once(monkeypatch):
    """When the check IS still open, the guard finalizes it to timed_out
    exactly once — a second signal (or a late normal finalize) no-ops."""
    _patch_common(monkeypatch)
    rep = RecordingReporter()
    rep.open_progress("headsha123")
    budget = Budget(max_input=1, max_output=1, max_iterations=1)
    review_mod._TIMEOUT_GUARD.update(
        reporter=rep, budget=budget, start=time.monotonic(), terminated_reason=None
    )

    review_mod._finalize_check_on_signal()
    assert len(rep.complete_calls) == 1
    call = rep.complete_calls[0]
    assert call["conclusion"] == "timed_out"
    assert call["verdict_line"] == "timed out before producing a verdict"
    assert call["terminated_reason"] == "gha_timeout"
    assert call["budget"] is budget

    # Second signal: idempotent.
    review_mod._finalize_check_on_signal()
    assert len(rep.complete_calls) == 1
    # Late normal finalize: also a no-op (first terminal won).
    rep.complete_check(
        verdict_line="verdict: looks good", conclusion="success"
    )
    assert len(rep.complete_calls) == 1


# ── Trigger-security gate ─────────────────────────────────────────────


def test_trigger_unenforced_skips_association_fetch(monkeypatch):
    """Explicit legacy opt-out: the gate is inert — no REST association
    lookup, review proceeds at full capability (the pre-#2 call profile)."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    def _boom(*a, **k):
        raise AssertionError("association fetched with enforcement off")

    monkeypatch.setattr(prc_mod, "fetch_author_association", _boom)

    async def fake_quick(**kwargs):
        return "🟢 looks good\n\nClean.", None

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)

    result = run_review(
        _cfg(trigger=TriggerPolicy(enforce=False)),
        reporter=RecordingReporter(),
        retrieval=NullRetrievalProvider(),
    )
    assert result.conclusion == "success"


def test_trigger_policy_denies_untrusted_author(monkeypatch):
    """Enforced + untrusted author + default skip action → no LLM call,
    skip comment posted, check concluded `cancelled` (a verdict-gating
    aggregator must stay red on a policy-denied PR)."""
    _patch_common(monkeypatch)
    import cora.core.quick_review as quick_mod

    monkeypatch.setattr(
        prc_mod, "fetch_author_association", lambda repo, pr: "NONE"
    )

    async def fail_quick(**kwargs):
        raise AssertionError("LLM called on a policy-denied PR")

    monkeypatch.setattr(quick_mod, "quick_review_call", fail_quick)

    cfg = _cfg(trigger=TriggerPolicy(enforce=True))
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.terminated_reason == "trigger-policy"
    assert result.conclusion == "cancelled"
    assert len(rep.skips) == 1 and "cora:approved" in rep.skips[0]
    assert rep.complete_calls[0]["conclusion"] == "cancelled"
    assert rep.dispatches == []


def test_trigger_comment_only_forces_quick_and_suppresses_patch(monkeypatch):
    """Degraded (comment-only) run: a deep config still runs quick mode
    (no tools), and a propose_patch directive the model emits is stripped
    from the comment and never dispatched — the verdict comment +
    check-run are the only writes."""
    _patch_common(monkeypatch)
    import cora.core.deep_review as deep_mod
    import cora.core.quick_review as quick_mod

    monkeypatch.setattr(
        prc_mod, "fetch_author_association", lambda repo, pr: "NONE"
    )

    directive = (
        '```json propose_patch\n'
        '{"title": "t", "body": "b", "edits": '
        '[{"path": "f.txt", "old_string": "x", "new_string": "y"}]}\n'
        '```'
    )

    async def fake_quick(**kwargs):
        return f"🟡 minor\n\nA finding.\n\n{directive}\n", None

    async def fail_deep(**kwargs):
        raise AssertionError("deep loop ran under a comment-only ceiling")

    monkeypatch.setattr(quick_mod, "quick_review_call", fake_quick)
    monkeypatch.setattr(deep_mod, "deep_review_call", fail_deep)

    cfg = _cfg(
        max_tool_iterations=6,
        trigger=TriggerPolicy(enforce=True, untrusted_action="comment-only"),
    )
    rep = RecordingReporter()
    result = run_review(cfg, reporter=rep, retrieval=NullRetrievalProvider())

    assert result.mode == "quick"
    assert result.verdict == "minor"
    assert "propose_patch" not in result.body
    assert rep.dispatches == []
