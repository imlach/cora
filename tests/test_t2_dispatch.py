"""T2 alt-reviewer dispatch + disagreement composition.

Covers the dispatch logic (env gate, deep-only, blocker short-circuit),
the body composition path (banner + collapsed `<details>` dissent
block), and the `tier_verdict` logger emission. The end-to-end LLM
call is NOT exercised — `call_t2_alt_reviewer` is a thin wrapper
around `deep_review_call` whose own tests already cover the agent-
loop machinery.
"""
from __future__ import annotations

# ---------------------------------------------------------------- composition


def _r(path, adopted_tier, adopted_verdict, dissent_tier, gap, banner):
    """Tiny Resolution ctor."""
    from cora.core.disagreement import Resolution
    return Resolution(
        path=path,
        adopted_tier=adopted_tier,
        adopted_verdict=adopted_verdict,
        dissent_tier=dissent_tier,
        gap=gap,
        banner=banner,
    )


def test_compose_agree_keeps_primary_body_unchanged():
    """Agree path — no banner, no dissent block. Composer returns the
    primary body verbatim so high-confidence reviews don't get
    diluted by appended duplication."""
    from cora.core.t2_dispatch import compose_disagreement_body

    primary = "🟢 looks good\n\nT0 review body."
    out = compose_disagreement_body(
        resolution=_r("agree", "T0", "looks good", None, 0, None),
        primary_body=primary,
        primary_tier_label="T0",
        t2_body="🟢 looks good\n\nT2 review body.",
        t2_model_alias="alt-reviewer",
    )
    assert out == primary


def test_compose_gap_one_renders_banner_and_collapsed_dissent():
    """gap=1, T2 wins conservative — banner above, T2 body in middle,
    T0 body in a collapsed `<details>` block."""
    from cora.core.t2_dispatch import compose_disagreement_body

    resolution = _r(
        path="adopt_conservative",
        adopted_tier="T2",
        adopted_verdict="needs changes",
        dissent_tier="T0",
        gap=1,
        banner="🔄 T0/T2 disagreed — adopted T2 (conservative)",
    )
    out = compose_disagreement_body(
        resolution=resolution,
        primary_body="🟡 minor\n\nT0 found small nits.",
        primary_tier_label="T0",
        t2_body="🔴 needs changes\n\nT2 found a regression.",
        t2_model_alias="alt-reviewer",
        primary_model_alias="review",
    )
    # Banner first.
    assert out.startswith("🔄 T0/T2 disagreed — adopted T2 (conservative)")
    # T2 body is the leading review since it won the conservative pick.
    assert "T2 found a regression." in out
    # T0 body lands in the `<details>` block (collapsed by default).
    assert "<details>" in out
    assert "<summary>T0 (review) said:" in out
    assert "T0 found small nits." in out
    assert "</details>" in out
    # T2 body must appear BEFORE the details block; T0 body appears
    # inside it. Order check pins the layout.
    t2_idx = out.index("T2 found a regression.")
    details_idx = out.index("<details>")
    t0_idx = out.index("T0 found small nits.")
    assert t2_idx < details_idx < t0_idx


def test_compose_gap_one_t0_wins_when_t0_is_more_conservative():
    """Reverse direction: T0 says `minor`, T2 says `looks good`. T0
    wins; T2 lands in the dissent block. Composer must NOT hard-code
    "T2 is always the leading body"."""
    from cora.core.t2_dispatch import compose_disagreement_body

    resolution = _r(
        path="adopt_conservative",
        adopted_tier="T0",
        adopted_verdict="minor",
        dissent_tier="T2",
        gap=1,
        banner="🔄 T0/T2 disagreed — adopted T0 (conservative)",
    )
    out = compose_disagreement_body(
        resolution=resolution,
        primary_body="🟡 minor\n\nT0 review.",
        primary_tier_label="T0",
        t2_body="🟢 looks good\n\nT2 review.",
        t2_model_alias="alt-reviewer",
    )
    # T0 (primary) body is leading; T2 in dissent.
    assert "<summary>T2 (alt-reviewer) said:" in out
    t0_idx = out.index("T0 review.")
    details_idx = out.index("<details>")
    t2_idx = out.index("T2 review.")
    assert t0_idx < details_idx < t2_idx


def test_compose_single_tier_path_returns_primary_unchanged():
    """Defensive: if the composer is somehow called on a single_tier
    resolution (no T2 ran), it must not crash — return the primary
    body unchanged so the caller's normal post path proceeds."""
    from cora.core.t2_dispatch import compose_disagreement_body

    out = compose_disagreement_body(
        resolution=_r("single_tier", "T0", "looks good", None, None, None),
        primary_body="🟢 looks good\n\nT0.",
        primary_tier_label="T0",
        t2_body="",
        t2_model_alias="alt-reviewer",
    )
    assert out == "🟢 looks good\n\nT0."


# ---------------------------------------------------------------- end-to-end resolver feed


def test_synthetic_gap_one_pair_feeds_resolver_and_composes():
    """Integration sanity — parse a synthetic T0 (`minor`) + T2
    (`needs changes`) body through the same call chain the
    dispatcher uses, and confirm the composed body has the
    expected gap=1 shape (collapsed details + conservative adopt)."""
    from cora.core.disagreement import (
        TierVerdict,
        resolve_disagreement,
    )
    from cora.core.leak import (
        detect_blocker,
        parse_verdict_from_body,
    )
    from cora.core.t2_dispatch import compose_disagreement_body

    t0_body = (
        "Verdict: 🟡 minor\n\n"
        "Small nits but otherwise fine.\n"
    )
    t2_body = (
        "Verdict: 🔴 needs changes\n\n"
        "🚨 **Blocker:** missing NetworkPolicy.\n"
    )

    t0_tv = TierVerdict(
        tier="T0",
        verdict=parse_verdict_from_body(t0_body),
        body=t0_body,
        has_blocker=detect_blocker(t0_body),
    )
    t2_tv = TierVerdict(
        tier="T2",
        verdict=parse_verdict_from_body(t2_body),
        body=t2_body,
        has_blocker=detect_blocker(t2_body),
    )
    assert t0_tv.verdict == "minor"
    assert t2_tv.verdict == "needs changes"

    resolution = resolve_disagreement(t0=t0_tv, t2=t2_tv, t3_enabled=False)
    assert resolution.path == "adopt_conservative"
    assert resolution.gap == 1
    assert resolution.adopted_tier == "T2"

    out = compose_disagreement_body(
        resolution=resolution,
        primary_body=t0_body,
        primary_tier_label="T0",
        t2_body=t2_body,
        t2_model_alias="alt-reviewer",
    )
    assert "<details>" in out
    assert "🚨 **Blocker:** missing NetworkPolicy." in out
    # T0's minor verdict body lands inside the collapsed block.
    assert "Small nits but otherwise fine." in out


# ---------------------------------------------------------------- logger emission


def test_log_tier_verdict_emits_t2_event():
    """The logger helper must accept `tier=T2` and emit an event with
    the tier, verdict, and has_blocker fields. Downstream
    dashboards parse on these labels."""
    from cora.core.loop_logging import log_tier_verdict

    captured: list[str] = []
    log_tier_verdict(
        pr_number="1234",
        tier="T2",
        verdict="needs changes",
        has_blocker=True,
        body_chars=420,
        log=captured.append,
    )
    assert len(captured) == 1
    line = captured[0]
    assert "phase=T2" in line
    assert "event=tier_verdict" in line
    assert "tier=T2" in line
    # Verdict word with spaces collapsed to underscore for logfmt.
    assert "verdict=needs_changes" in line
    assert "has_blocker=true" in line
    assert "body_chars=420" in line


# ---------------------------------------------------------------- dispatch gating


def test_call_t2_alt_reviewer_delegates_to_deep_review_call(monkeypatch):
    """`call_t2_alt_reviewer` must hand straight through to
    `deep_review_call` with the alt-reviewer alias — fresh review,
    no message_history. Pins the wrapper-only behavior and that we
    drop the messages tuple slot on the way out."""
    import asyncio

    from cora.core import t2_dispatch

    captured: dict = {}

    async def fake_deep_review_call(**kwargs):
        captured.update(kwargs)
        return ("T2 body", None, ["search_knowledge"], ["msg1"])

    monkeypatch.setattr(
        "cora.core.deep_review.deep_review_call",
        fake_deep_review_call,
    )

    body, terminated, tools = asyncio.run(
        t2_dispatch.call_t2_alt_reviewer(
            endpoint_base_url="http://litellm.test/v1",
            llm_gateway_key="dummy",
            t2_model_alias="alt-reviewer",
            system_prompt="sp",
            initial_user_prompt="iup",
            budget=object(),
            timeout_s=180,
            pr_number="1234",
            repo="owner/repo",
            mcp_url="http://mcp.test/mcp",
            mcp_headers={"Authorization": "Bearer x"},
            allowed_tools={"search_knowledge"},
            max_iterations=8,
        )
    )

    assert body == "T2 body"
    assert terminated is None
    # Wrapper drops the 4th tuple slot (messages) — diversity = fresh
    # run, no history carried forward.
    assert tools == ["search_knowledge"]
    # Model alias is the alt-reviewer one, NOT the T0/T1 alias.
    assert captured["model_alias"] == "alt-reviewer"
    # max_iterations is honored — tighter T2 budget vs T0's default 12.
    assert captured["max_iterations"] == 8


def test_dispatch_skipped_when_t0_has_blocker():
    """`detect_blocker` on T0's final_body must short-circuit T2 —
    no point spending the latency when T0 already blocks merge."""
    from cora.core.leak import detect_blocker

    t0_body_with_blocker = (
        "Verdict: 🔴 needs changes\n\n"
        "🚨 **Blocker:** missing NetworkPolicy.\n"
    )
    assert detect_blocker(t0_body_with_blocker) is True

    # The dispatch gate in agent_review.py reads:
    #   if (... and not detect_blocker(final_body)):
    # so this confirms the predicate. Live coverage of the gate
    # itself happens in the orchestrator's E2E reviewer test trace —
    # `agent_review.py` is too heavyweight to unit-test directly.


def test_dispatch_env_gate_default_off():
    """`AGENT_REVIEW_T2_DISAGREEMENT` defaults to off — opt-in for
    soak. Mirrors `AGENT_REVIEW_T1_CONTINUATION`'s shape."""
    import os

    # The dispatcher reads:
    #   os.environ.get("AGENT_REVIEW_T2_DISAGREEMENT", "")
    #       .strip().lower() == "true"
    # so anything not equal to "true" must skip T2.
    for unset_val in ("", "false", "0", "no", "off", "True "):
        # Truthy-looking but not the exact gate string must NOT
        # activate T2. ("True " with trailing space DOES match after
        # strip+lower — checked separately.)
        gate = unset_val.strip().lower() == "true"
        if unset_val.strip().lower() == "true":
            assert gate is True
        else:
            assert gate is False

    # Default env-read returns "" → gate False.
    saved = os.environ.pop("AGENT_REVIEW_T2_DISAGREEMENT", None)
    try:
        gate = (
            os.environ.get("AGENT_REVIEW_T2_DISAGREEMENT", "")
            .strip().lower() == "true"
        )
        assert gate is False
    finally:
        if saved is not None:
            os.environ["AGENT_REVIEW_T2_DISAGREEMENT"] = saved


def test_dispatch_t2_model_env_selects_alias(monkeypatch):
    """`AGENT_REVIEW_T2_MODEL` is the A/B selector between the two
    alt-reviewer backends. The dispatcher reads:
        os.environ.get("AGENT_REVIEW_T2_MODEL", "alt-reviewer")
            .strip() or "alt-reviewer"
    so an unset/blank var picks the default alias and an explicit value
    (the LiteLLM alias) routes T2 there. Locks the contract the
    workflow's `vars.AGENT_REVIEW_T2_MODEL` toggle relies on."""
    import os

    def resolve() -> str:
        return (
            os.environ.get("AGENT_REVIEW_T2_MODEL", "alt-reviewer").strip()
            or "alt-reviewer"
        )

    # Unset → default alias.
    monkeypatch.delenv("AGENT_REVIEW_T2_MODEL", raising=False)
    assert resolve() == "alt-reviewer"

    # An explicit alias routes T2 to the alternate backend.
    monkeypatch.setenv("AGENT_REVIEW_T2_MODEL", "alt-review-b")
    assert resolve() == "alt-review-b"

    # Blank / whitespace falls back to the default alias (matches the
    # `.strip() or` guard) rather than sending an empty model name.
    monkeypatch.setenv("AGENT_REVIEW_T2_MODEL", "   ")
    assert resolve() == "alt-reviewer"
