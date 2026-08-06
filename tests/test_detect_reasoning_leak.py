"""Tests for agent_review.detect_reasoning_leak — the post-processor
guard that decides between posting a review body and suppressing it.

Covers the original `Verdict:` / `🟢 looks good` strict-marker path
plus the conservative LGTM/Approved shorthand rescue added after the
2026-05-23 renovate batch tripped the guard (the model emitted
`LGTM. …` instead of the marker).
"""
from __future__ import annotations

from cora.core import detect_reasoning_leak


# ---------------------------------------------------------------------------
# Strict marker — pre-existing behaviour, must not regress
# ---------------------------------------------------------------------------

def test_strict_marker_at_start_unchanged():
    body = "🟢 looks good\n\nNice change."
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert stripped == 0
    assert out == body


def test_strict_marker_after_preamble_sliced():
    body = "Let me think about this...\nActually, yes.\n\n🟢 looks good\n\nFine."
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert stripped > 0
    assert out.startswith("🟢 looks good")


def test_old_verdict_prefix_still_recognised():
    body = "Verdict: looks good\n\nFine."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.lower().startswith("verdict: looks good")


def test_needs_changes_marker():
    body = "🔴 needs changes\n\nfoo.py:42 is wrong."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.startswith("🔴 needs changes")


# ---------------------------------------------------------------------------
# Leak path — no marker, no shorthand → suppress
# ---------------------------------------------------------------------------

def test_no_marker_no_shorthand_is_leak():
    body = "I think this PR is reasonable. The diff looks self-consistent."
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is True
    assert stripped == 0
    assert out == body  # caller decides not to post


def test_empty_body_not_a_leak():
    out, stripped, is_leak = detect_reasoning_leak("")
    assert is_leak is False
    assert stripped == 0
    assert out == ""


# ---------------------------------------------------------------------------
# Whole-body marker scan — the buried-verdict salvage path
# ---------------------------------------------------------------------------

def test_marker_below_long_leading_prose_is_rescued():
    # Observed incident shape: a complete, correct review that opens with prose
    # analysis and only emits the verdict marker further down — beyond
    # the old ~8.6 KB probe window. The whole-body scan must find the
    # marker and slice the body to lead with it, NOT suppress.
    preamble = (
        "Let me work through this change carefully. The diff touches the "
        "leak guard and the retry path. "
    ) * 400  # ~36 KB of prose, well past the old 8.6 KB window
    body = preamble + "\n🟢 looks good\n\nThe change is correct."
    assert len(preamble) > 9_000  # guard: genuinely past the old window
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert stripped > 0
    assert out.startswith("🟢 looks good")
    assert "The change is correct." in out


def test_needs_changes_marker_below_prose_is_rescued():
    # Salvage must work for every verdict severity, not just the green
    # approve case — a `🔴 needs changes` review that opened with prose
    # is exactly the observed incident (it caught a real blocker).
    preamble = "I need to think about whether this is safe. " * 300
    body = (
        preamble
        + "\n🔴 needs changes\n\nfoo.py:42 drops the verdict marker guard."
    )
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert stripped > 0
    assert out.startswith("🔴 needs changes")


def test_old_verdict_prefix_below_prose_is_rescued():
    # The legacy `Verdict:` prefix must also be salvaged from below the
    # leading window, not just the colored-circle markers.
    preamble = "Considering the trade-offs here. " * 400
    body = preamble + "\nVerdict: minor\n\nA nit about naming."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.lower().startswith("verdict: minor")


def test_rescued_marker_parses_to_correct_verdict():
    # The salvaged body must parse to the same verdict the model meant —
    # the conclusion mapping downstream depends on it.
    from cora.core import parse_verdict_from_body
    preamble = "Analysing the change in depth. " * 400
    body = preamble + "\n🔴 needs changes\n\nReal blocker."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert parse_verdict_from_body(out) == "needs changes"


def test_genuinely_markerless_long_body_still_suppresses():
    # The anti-leak intent must hold: a long body with NO valid verdict
    # marker anywhere is still a leak, even though the scan window grew.
    body = (
        "I think this PR is reasonable. The diff looks self-consistent. "
    ) * 500  # ~30 KB of pure prose, no marker, no shorthand
    out, stripped, is_leak = detect_reasoning_leak(body)
    assert is_leak is True
    assert stripped == 0
    assert out == body  # caller decides not to post


# ---------------------------------------------------------------------------
# Shorthand-approve rescue — the new path
# ---------------------------------------------------------------------------

def test_shorthand_lgtm_synthesises_marker():
    body = "LGTM. Straightforward version sync across workflows."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.startswith("🟢 looks good")
    assert "Straightforward version sync across workflows." in out


def test_shorthand_lgtm_only_no_prose():
    body = "LGTM"
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out == "🟢 looks good"


def test_shorthand_approved_synthesises_marker():
    body = "Approved. No concerns."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.startswith("🟢 looks good")
    assert "No concerns." in out


def test_shorthand_case_insensitive():
    body = "lgtm. Fine."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.startswith("🟢 looks good")


def test_shorthand_tolerates_leading_whitespace():
    body = "\n\n  LGTM. Fine."
    out, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is False
    assert out.startswith("🟢 looks good")


# ---------------------------------------------------------------------------
# Shorthand must NOT false-match mid-body
# ---------------------------------------------------------------------------

def test_shorthand_only_matched_at_start():
    # "LGTM" appears mid-body — must not rescue this; no real verdict.
    body = "The reviewer in my last project would say LGTM here, but actually…"
    _, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is True


def test_shorthand_after_real_preamble_is_still_leak():
    # Long reasoning preamble then "LGTM" — the shorthand path is
    # anchored to the start, so this is a leak.
    body = "I considered the changes carefully and weighed the trade-offs.\nLGTM."
    _, _, is_leak = detect_reasoning_leak(body)
    assert is_leak is True


# ---------------------------------------------------------------------------
# Downstream parse_verdict_from_body recognises the synthesised marker
# ---------------------------------------------------------------------------

def test_synthesised_marker_parses_to_looks_good():
    from cora.core import parse_verdict_from_body
    body = "LGTM. Straightforward."
    out, _, _ = detect_reasoning_leak(body)
    assert parse_verdict_from_body(out) == "looks good"


# ---------------------------------------------------------------------------
# build_leak_retry_messages — pure helper used by quick-mode retry-on-leak
# ---------------------------------------------------------------------------

def test_build_leak_retry_messages_single_user_turn():
    from cora.core.leak import build_leak_retry_messages
    messages = build_leak_retry_messages("Some analysis prose.")
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)


def test_build_leak_retry_messages_lists_all_three_markers():
    from cora.core.leak import build_leak_retry_messages
    content = build_leak_retry_messages("body")[0]["content"]
    # All three markers must appear in the instruction so the model
    # has the full severity vocabulary to pick from.
    assert "🟢 looks good" in content
    assert "🟡 minor" in content
    assert "🔴 needs changes" in content


def test_build_leak_retry_messages_embeds_leaked_body():
    from cora.core.leak import build_leak_retry_messages
    leaked = "This is a straightforward bump from 1.31.0 to 1.31.1. No issues."
    content = build_leak_retry_messages(leaked)[0]["content"]
    assert leaked in content


def test_build_leak_retry_messages_caps_long_body():
    from cora.core.leak import build_leak_retry_messages
    # Body well over the 6K cap — the retry input must not include
    # the whole tail, or we balloon retry tokens on the long-tail
    # bodies (observed up to 2.7K output tokens uncapped).
    huge = "x" * 50_000
    content = build_leak_retry_messages(huge)[0]["content"]
    # 6K cap on body + ~700 chars of framing instruction
    assert len(content) < 8_000


def test_build_leak_retry_messages_forbids_shorthand_in_instruction():
    from cora.core.leak import build_leak_retry_messages
    content = build_leak_retry_messages("body")[0]["content"]
    # The instruction itself must tell the model not to use the
    # shorthand forms — otherwise we get LGTM-style retries that
    # land back on the rescue path instead of the strict marker.
    assert "LGTM" in content  # appears in the don't-use list
    assert "Approved" in content


def test_build_leak_retry_messages_empty_body_safe():
    from cora.core.leak import build_leak_retry_messages
    # Empty body is a real path — quick_review_call can return ""
    # on an empty model response, and the caller's guard above us
    # only skips retry when final_body is falsy. Defensive coverage.
    messages = build_leak_retry_messages("")
    assert len(messages) == 1
    # Instruction still well-formed even if leaked body is empty.
    assert "🟢 looks good" in messages[0]["content"]


# ---------------------------------------------------------------------------
# strip_reasoning — paired <think>…</think> + orphan </think>
# ---------------------------------------------------------------------------

def test_strip_reasoning_paired_block_removed():
    from cora.core.leak import strip_reasoning
    body = "<think>let me consider</think>\n\n🟢 looks good\n\nFine."
    out, stripped = strip_reasoning(body)
    assert out.startswith("🟢 looks good")
    assert stripped > 0
    assert "<think>" not in out
    assert "</think>" not in out


def test_strip_reasoning_no_think_tags_unchanged():
    from cora.core.leak import strip_reasoning
    body = "🟢 looks good\n\nClean review body."
    out, stripped = strip_reasoning(body)
    assert out == body
    assert stripped == 0


def test_strip_reasoning_empty_body_safe():
    from cora.core.leak import strip_reasoning
    out, stripped = strip_reasoning("")
    assert out == ""
    assert stripped == 0


def test_strip_reasoning_orphan_close_tag_strips_preamble():
    # Observed leak shape — parser stripped <think> but the close tag
    # leaked into content, and the model double-emitted the verdict
    # body (once inside reasoning, once as the real response).
    # Without orphan-close handling, detect_reasoning_leak finds the
    # first 🟢 at position 0 and posts the duplicate body verbatim.
    from cora.core.leak import strip_reasoning
    body = (
        "🟢 looks good\n\n"
        "Straightforward fallback to `message.reasoning` for the inference engine's "
        "o1-style output, with backward compat for `reasoning_content` and a "
        "`.strip()` to clean up empty parser blocks.\n\n"
        "No findings to report.\n"
        "</think>\n\n"
        "🟢 looks good\n\n"
        "Straightforward fallback to `message.reasoning` for the inference engine's "
        "o1-style output, with backward compat for `reasoning_content` and a "
        "`.strip()` to clean up empty parser blocks."
    )
    out, stripped = strip_reasoning(body)
    assert "</think>" not in out
    assert stripped > 0
    # Only one verdict marker survives — the duplicate halves collapsed
    # into the single real response after the close-tag cutpoint.
    assert out.count("🟢 looks good") == 1
    assert out.startswith("🟢 looks good")


def test_strip_reasoning_orphan_close_uses_last_occurrence():
    # When multiple orphan close tags exist (rare — would mean the
    # model emitted reasoning, then more reasoning, then the response),
    # cut at the LAST one so the final clean output survives.
    from cora.core.leak import strip_reasoning
    body = (
        "first reasoning chunk\n</think>\n"
        "second reasoning chunk\n</think>\n"
        "🟢 looks good\n\nReal review."
    )
    out, _ = strip_reasoning(body)
    assert out.startswith("🟢 looks good")
    assert "reasoning chunk" not in out


def test_strip_reasoning_paired_block_then_orphan_close():
    # Paired block at the top is stripped by _THINK_BLOCK_RE; an
    # orphan close tag later in the body still triggers the cutpoint.
    from cora.core.leak import strip_reasoning
    body = (
        "<think>opening trace</think>\n"
        "some interim output\n"
        "</think>\n"
        "🟢 looks good\n\nReal review."
    )
    out, _ = strip_reasoning(body)
    assert out.startswith("🟢 looks good")
    assert "interim output" not in out
    assert "opening trace" not in out


def test_strip_reasoning_then_leak_guard_end_to_end():
    # End-to-end pin: feed an observed leaked comment body through the same
    # post-processing pipeline the entrypoint runs, assert the single
    # clean verdict body is what posts.
    from cora.core.leak import strip_reasoning
    body = (
        "🟢 looks good\n\n"
        "Straightforward fallback to `message.reasoning` for the inference engine's "
        "o1-style output, with backward compat for `reasoning_content` and a "
        "`.strip()` to clean up empty parser blocks.\n\n"
        "No findings to report.\n"
        "</think>\n\n"
        "🟢 looks good\n\n"
        "Straightforward fallback to `message.reasoning` for the inference engine's "
        "o1-style output, with backward compat for `reasoning_content` and a "
        "`.strip()` to clean up empty parser blocks."
    )
    cleaned, _ = strip_reasoning(body)
    out, _, is_leak = detect_reasoning_leak(cleaned)
    assert is_leak is False
    assert out.count("🟢 looks good") == 1
    assert "</think>" not in out


# ---------------------------------------------------------------------------
# Custom verdict vocabulary — ReviewerConfig.verdict_glyphs/verdict_words
# wired through the keyword-only `glyphs=` / `words=` params
# ---------------------------------------------------------------------------

# An adopter's vocabulary, ascending-concern order (approve, nits, block).
_GLYPHS = ("✅", "⚠️", "❌")
_WORDS = ("approve", "nitpicks", "request changes")


def test_custom_vocab_marker_at_start_parses():
    from cora.core.leak import parse_verdict_from_body
    body = "❌ request changes\n\nfoo.py:42 is wrong."
    out, stripped, is_leak = detect_reasoning_leak(body, glyphs=_GLYPHS, words=_WORDS)
    assert is_leak is False
    assert stripped == 0
    assert out == body
    assert parse_verdict_from_body(body, glyphs=_GLYPHS, words=_WORDS) == (
        "request changes"
    )


def test_custom_vocab_marker_after_preamble_sliced():
    body = "Let me think...\n\n⚠️ nitpicks\n\nA naming nit."
    out, stripped, is_leak = detect_reasoning_leak(body, glyphs=_GLYPHS, words=_WORDS)
    assert is_leak is False
    assert stripped > 0
    assert out.startswith("⚠️ nitpicks")


def test_custom_vocab_legacy_verdict_prefix_recognised():
    # The `Verdict:` prefix path is format-stable; only the word set
    # swaps with the vocabulary.
    from cora.core.leak import parse_verdict_from_body
    body = "Verdict: approve\n\nFine."
    out, _, is_leak = detect_reasoning_leak(body, glyphs=_GLYPHS, words=_WORDS)
    assert is_leak is False
    assert out == body
    assert parse_verdict_from_body(body, glyphs=_GLYPHS, words=_WORDS) == "approve"


def test_custom_vocab_default_markers_not_recognised():
    # Symmetry of the override: with a custom vocabulary, the stock
    # `🟢 looks good` line is no longer a valid marker — the body is a
    # leak and the verdict unparseable.
    from cora.core.leak import parse_verdict_from_body
    body = "🟢 looks good\n\nFine."
    _, _, is_leak = detect_reasoning_leak(body, glyphs=_GLYPHS, words=_WORDS)
    assert is_leak is True
    assert parse_verdict_from_body(body, glyphs=_GLYPHS, words=_WORDS) is None


def test_custom_vocab_shorthand_synthesises_custom_approve_line():
    body = "LGTM. Straightforward."
    out, _, is_leak = detect_reasoning_leak(body, glyphs=_GLYPHS, words=_WORDS)
    assert is_leak is False
    assert out.startswith("✅ approve")
    assert "Straightforward." in out


def test_custom_vocab_words_with_regex_metachars_are_escaped():
    # Glyphs / words are escaped before regex assembly, so metachars are
    # literal. (Words must still END in a word character — the marker
    # regex keeps its trailing `\b`.)
    from cora.core.leak import parse_verdict_from_body
    glyphs = ("(+)", "(~)", "(-)")
    words = ("a+ approve", "b? nits", "c* changes")
    body = "(-) c* changes\n\nBroken."
    out, _, is_leak = detect_reasoning_leak(body, glyphs=glyphs, words=words)
    assert is_leak is False
    assert out == body
    assert parse_verdict_from_body(body, glyphs=glyphs, words=words) == "c* changes"


def test_custom_vocab_detect_blocker_uses_block_entry():
    from cora.core.leak import detect_blocker
    assert detect_blocker(
        "❌ request changes\n\nBroken.", glyphs=_GLYPHS, words=_WORDS
    )
    assert not detect_blocker(
        "✅ approve\n\nFine.", glyphs=_GLYPHS, words=_WORDS
    )
    # Default block marker is not the custom vocabulary's.
    assert not detect_blocker(
        "🔴 needs changes\n\nBroken.", glyphs=_GLYPHS, words=_WORDS
    )
    # The 🚨 Blocker line is vocabulary-independent.
    assert detect_blocker(
        "✅ approve\n\n🚨 **Blocker:** secret committed.",
        glyphs=_GLYPHS,
        words=_WORDS,
    )


def test_custom_vocab_verdict_to_conclusion_positional_mapping():
    from cora.core.leak import verdict_to_conclusion
    assert verdict_to_conclusion("approve", words=_WORDS) == "success"
    assert verdict_to_conclusion("nitpicks", words=_WORDS) == "neutral"
    assert verdict_to_conclusion("request changes", words=_WORDS) == "failure"
    assert verdict_to_conclusion("looks good", words=_WORDS) == "neutral"
    assert verdict_to_conclusion(None, words=_WORDS) == "neutral"


def test_verdict_to_review_event_default_vocab():
    # Only the block verdict gates (REQUEST_CHANGES); the
    # lower two stay advisory (COMMENT). `looks good` is COMMENT, never a
    # bot APPROVE.
    from cora.core.leak import verdict_to_review_event
    assert verdict_to_review_event("looks good") == "COMMENT"
    assert verdict_to_review_event("minor") == "COMMENT"
    assert verdict_to_review_event("needs changes") == "REQUEST_CHANGES"
    # Missing / unrecognised → advisory COMMENT (never gating).
    assert verdict_to_review_event(None) == "COMMENT"
    assert verdict_to_review_event("garbage") == "COMMENT"


def test_verdict_to_review_event_custom_vocab_positional_mapping():
    # A custom `words` set maps by position, like verdict_to_conclusion.
    from cora.core.leak import verdict_to_review_event
    assert verdict_to_review_event("approve", words=_WORDS) == "COMMENT"
    assert verdict_to_review_event("nitpicks", words=_WORDS) == "COMMENT"
    assert verdict_to_review_event("request changes", words=_WORDS) == "REQUEST_CHANGES"
    # A word outside the custom vocab → COMMENT.
    assert verdict_to_review_event("looks good", words=_WORDS) == "COMMENT"
    assert verdict_to_review_event(None, words=_WORDS) == "COMMENT"


def test_custom_vocab_retry_messages_list_custom_markers():
    from cora.core.leak import build_leak_retry_messages
    content = build_leak_retry_messages("body", glyphs=_GLYPHS, words=_WORDS)[0][
        "content"
    ]
    assert "✅ approve" in content
    assert "⚠️ nitpicks" in content
    assert "❌ request changes" in content
    assert "🟢 looks good" not in content


def test_custom_vocab_resolve_disagreement_ranks_custom_words():
    from cora.core.disagreement import TierVerdict, resolve_disagreement
    t0 = TierVerdict(tier="T0", verdict="approve", body="✅ approve", has_blocker=False)
    t2 = TierVerdict(
        tier="T2",
        verdict="request changes",
        body="❌ request changes",
        has_blocker=False,
    )
    res = resolve_disagreement(t0=t0, t2=t2, words=_WORDS)
    assert res.gap == 2
    assert res.path == "adopt_conservative"
    assert res.adopted_tier == "T2"
    assert res.adopted_verdict == "request changes"


def test_default_vocab_constants_referenced_not_copied():
    # Anti-drift discipline: the public config fields and the leak-module
    # defaults must all reference the SAME engine constants.
    from cora import ReviewerConfig
    from cora.core import config as c
    d = ReviewerConfig()
    assert d.verdict_glyphs is c.VERDICT_GLYPHS
    assert d.verdict_words is c.VERDICT_WORDS
    assert c.VERDICT_GLYPHS == ("🟢", "🟡", "🔴")
    assert c.VERDICT_WORDS == ("looks good", "minor", "needs changes")


def test_explicit_default_vocab_matches_implicit_default():
    # Passing the engine constants explicitly must behave identically to
    # not passing them — the params default to the constants by reference.
    from cora.core import config as c
    from cora.core.leak import parse_verdict_from_body
    for body in (
        "🟡 minor\n\nA nit.",
        "Preamble first.\n\n🔴 needs changes\n\nBroken.",
        "Verdict: looks good\n\nFine.",
        "LGTM. Fine.",
        "no marker at all, pure prose",
    ):
        assert detect_reasoning_leak(body) == detect_reasoning_leak(
            body, glyphs=c.VERDICT_GLYPHS, words=c.VERDICT_WORDS
        )
        assert parse_verdict_from_body(body) == parse_verdict_from_body(
            body, glyphs=c.VERDICT_GLYPHS, words=c.VERDICT_WORDS
        )


# ---------------------------------------------------------------------------
# detect_blocker — bullet-local retraction (cora#29's "Related" item)
#
# A model sometimes writes a `🚨 **Blocker:**` bullet and then, in the same
# bullet, walks it back ("false alarm", "the code is fine"). The old
# `detect_blocker` matched the marker text alone, so a retracted bullet
# still paused automerge and counted toward `tier_verdict` telemetry.
# These cover: a fully-retracted body (no blocker), a mixed body (still a
# blocker — retraction must never hide a real one), a plain hedge (not a
# retraction — still a blocker), the literal cora#25 phrasing, a bullet
# that wraps onto a continuation line, and the deliberate choice to ignore
# body-level reassurance text.
# ---------------------------------------------------------------------------

def test_retracted_only_body_has_no_blocker():
    from cora.core.leak import detect_blocker
    body = (
        "🟡 minor\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** this looked like a SQL injection at first "
        "glance, but it's parameterised — false alarm.\n"
    )
    assert detect_blocker(body) is False


def test_retracted_plus_genuine_still_blocker():
    from cora.core.leak import detect_blocker
    body = (
        "🔴 needs changes\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** this is a false alarm from the truncated diff "
        "display — the code is fine.\n"
        "- 🚨 **Blocker:** auth.go:44 — token comparison is not "
        "constant-time.\n"
    )
    assert detect_blocker(body) is True


def test_genuine_blocker_only_regression():
    from cora.core.leak import detect_blocker
    body = (
        "🟡 minor\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** SQL built via string concatenation from a "
        "request param — injectable.\n"
    )
    assert detect_blocker(body) is True


def test_hedge_without_retraction_still_blocker():
    from cora.core.leak import detect_blocker
    # A hedge ("might be") is not a retraction — the finding is still
    # live, and must not be discounted just because the model wasn't
    # 100% certain.
    body = (
        "🟡 minor\n\n"
        "- 🚨 **Blocker:** this might be a problem under concurrent "
        "writes — worth a second look.\n"
    )
    assert detect_blocker(body) is True


def test_cora_25_exact_phrasing_no_blocker():
    # The exact retraction phrasing observed live on imlach/cora#25,
    # paired with a non-blocking verdict word so the assertion exercises
    # the blocker-bullet path specifically (a literal `needs changes`
    # verdict word is a separate, untouched signal — see
    # `detect_blocker`'s docstring).
    from cora.core.leak import detect_blocker
    body = (
        "🟢 looks good\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** This is a false alarm from the truncated diff "
        "display — the code is fine.\n\n"
        "I have no actual blockers.\n"
    )
    assert detect_blocker(body) is False


def test_multiline_bullet_retraction_found_on_continuation_line():
    from cora.core.leak import detect_blocker
    body = (
        "🟡 minor\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** this looks concerning at first glance.\n"
        "  Actually, on further review this is a false alarm — the code "
        "is fine.\n"
        "- 🚨 **Blocker:** real issue: missing null check on line 42.\n"
    )
    # The wrapped continuation line carries the retraction for bullet 1;
    # bullet 2 is untouched by it and stays live.
    assert detect_blocker(body) is True


def test_multiline_bullet_all_retracted_no_blocker():
    from cora.core.leak import detect_blocker
    body = (
        "🟡 minor\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** this looks concerning at first glance.\n"
        "  Actually, on further review this is a false alarm — the code "
        "is fine.\n"
    )
    assert detect_blocker(body) is False


def test_body_level_reassurance_never_overrides_a_live_bullet():
    # A body-level "no actual blockers" sentence is weaker evidence than a
    # bullet-local retraction (it can't be tied to which finding it's
    # about) — detect_blocker deliberately never consults it. A live,
    # unretracted bullet still blocks even when the closing prose
    # contradicts it.
    from cora.core.leak import detect_blocker
    body = (
        "🟡 minor\n\n"
        "**Findings:**\n"
        "- 🚨 **Blocker:** auth.go:44 — token comparison is not "
        "constant-time.\n\n"
        "I have no actual blockers.\n"
    )
    assert detect_blocker(body) is True


def test_needs_changes_verdict_word_still_blocks_regardless_of_bullets():
    # The verdict-word signal is independent of bullet retraction by
    # design (see detect_blocker's docstring) — a model that states its
    # own verdict as `needs changes` is believed, even with zero blocker
    # bullets in the body.
    from cora.core.leak import detect_blocker
    assert detect_blocker("🔴 needs changes\n\nNo findings section at all.")
