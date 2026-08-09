"""Model-output post-processing for the agentic reviewer.

Detect / strip stream-of-consciousness reasoning leaks (`<think>` blocks
or pre-`Verdict:` preambles), parse the verdict word out of a well-
formed body, map verdicts to GitHub check-run conclusions, and detect
the auto-merge blocker signal.
"""

from __future__ import annotations

import re
from functools import lru_cache

from cora.core.config import VERDICT_GLYPHS, VERDICT_WORDS
from cora.core.log import _gha_log


# How much of the body to search for the structured-output Verdict marker.
# 600 chars is enough for any reasonable prose preamble between the agent
# loop's "produce the response now" nudge and the model's actual verdict.
_VERDICT_PROBE_CHARS = 600
# Upper bound on the whole-body marker scan. Observed: a complete,
# correct *deep* review opened with prose analysis and only emitted
# the verdict marker further down — beyond the old `_VERDICT_PROBE_CHARS
# + 8_000` (~8.6 KB) window — so the guard saw no marker and suppressed
# the whole review. Scan the entire body (up to this generous cap) for a
# valid marker; if one is found below the leading window, the body is
# salvageable — slice it to lead with the marker rather than suppress.
# 64 KB comfortably covers any real review (the longest leaked body we've
# observed ran ~26 KB) while still bounding a pathological
# multi-megabyte runaway. A genuinely marker-less body still suppresses.
_VERDICT_FULL_SCAN_CHARS = 64_000
# Match optional leading prefix BEFORE "verdict" so detect_reasoning_leak's
# `body[start:]` slice preserves opening markdown bolding (`**`) and emoji
# prefix. Earlier `\bverdict` pattern matched at the V of `**Verdict:**`,
# stripping the opening `**` and leaving the comment header with
# `Verdict:** 🟡 minor` (closing-but-no-opening bold) — caught when the
# agentic reviewer rendered its own verdict that way.
#
# `[^A-Za-z\n]*` consumes any leading whitespace, markdown emphasis
# markers (`*`, `_`), emoji (anything that isn't an ASCII letter), and
# punctuation. `^` + MULTILINE anchors to a line start so we don't pick
# up "verdict" embedded in prose.
#
# Built per (glyphs, words) vocabulary rather than as one module-level
# constant so `ReviewerConfig.verdict_glyphs` / `.verdict_words` can
# customise the format; the default vocabulary compiles to the exact
# pattern that used to live here. lru_cache keeps the hot default path
# at compile-once cost.
@lru_cache(maxsize=8)
def _verdict_line_re(
    glyphs: tuple[str, ...], words: tuple[str, ...]
) -> re.Pattern[str]:
    glyph_alt = "|".join(re.escape(g) for g in glyphs)
    word_alt = "|".join(re.escape(w) for w in words)
    return re.compile(
        rf"""
        ^[^A-Za-z\n]*                  # leading whitespace / markdown / emoji
        (?:
            (?:{glyph_alt})[^A-Za-z\n]*    # new format: colored-circle marker
            |
            verdict\s*[*_]*\s*:\s*[^A-Za-z\n]*  # old format: "Verdict:" prefix
        )
        ({word_alt})\b
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

# Conservative shorthand-verdict recogniser — matched only at the very
# start of the (lstripped) body, only on patterns that conventionally
# mean "approve". When the strict regex misses, this rescues bodies
# like `LGTM. Straightforward …` by synthesising a `🟢 looks good`
# verdict line and keeping the model's prose, instead of suppressing
# the whole review.
#
# Conservative on purpose: only `🟢`-mapping shorthand. There's no safe
# shorthand mapping for `🟡 minor` or `🔴 needs changes` — those are
# elaborated reviews, not single-word verdicts; misclassifying a
# Concern as `looks good` is much worse than failing to recognise one.
# The risk on `LGTM but actually…` style hedging is accepted: that
# phrasing is a self-contradiction the model effectively never writes
# (and would be caught by the prompt tightening on the producer side).
_SHORTHAND_APPROVE_RE = re.compile(
    r"""
    \A
    (?:
        LGTM
        | approved?
        | approval
    )
    \b              # don't match a prefix (e.g. `LGTMaster` mustn't qualify)
    [\s.!,;:]*      # absorb optional trailing punctuation
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_reasoning_leak(
    body: str,
    *,
    glyphs: tuple[str, str, str] = VERDICT_GLYPHS,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> tuple[str, int, bool]:
    """Guard against the model emitting stream-of-consciousness reasoning
    instead of the `Verdict / Summary / Findings` structure the prompt
    asks for. Observed in the wild: ~26 KB of "Let me check…" /
    "Actually…" planning leaked into the comment body.

    Returns `(body_to_post, stripped_chars, is_leak)`:
      - If a `Verdict:` marker sits at the top of the body, returns the
        body unchanged.
      - If `Verdict:` exists but is preceded by a reasoning preamble,
        returns the slice starting at the marker plus the stripped
        char count (the model produced a valid review eventually — just
        keep the structured part). The scan covers the WHOLE body (up to
        `_VERDICT_FULL_SCAN_CHARS`), so a review that opens with prose
        analysis and only emits the marker deep in the body is salvaged
        by leading with that marker rather than suppressed — an observed
        shape (a complete, correct deep review with a late marker).
      - If no strict marker exists but the body opens with an
        unambiguous approve-shorthand (`LGTM`, `Approved`), prepend a
        synthetic `🟢 looks good` verdict line and keep the model's
        prose. Dependency-bump batches have tripped the leak guard because
        the model emitted `LGTM. …` instead of the marker — the prompt
        tightening reduces how often this happens, this path catches
        the rest. Conservative scope: shorthand maps onto `🟢` only;
        no safe shorthand exists for `🟡` / `🔴`.
      - If no marker AND no recognisable shorthand exists in the first
        probe-window, the whole body is treated as a leak and
        `is_leak=True`. The caller replaces the comment with a skip
        note instead of posting raw reasoning to a public PR.

    `glyphs` / `words` customise the verdict vocabulary (ascending-
    concern order; see `config.VERDICT_GLYPHS` / `config.VERDICT_WORDS`).
    The synthesised shorthand verdict uses `glyphs[0]` / `words[0]`.
    """
    if not body:
        return body, 0, False
    # Scan the whole body (capped) for a valid verdict marker — not just
    # the leading probe window. A marker anywhere in the body means the
    # review is salvageable: slice to lead with it (stripping any prose
    # preamble) rather than suppress. Only a body with NO valid marker
    # anywhere falls through to the shorthand rescue / leak path below.
    m = _verdict_line_re(glyphs, words).search(body[:_VERDICT_FULL_SCAN_CHARS])
    if m:
        start = m.start()
        if start == 0:
            return body, 0, False
        return body[start:], start, False
    # Strict marker missed — try the conservative approve-shorthand
    # rescue. Anchored to the first non-whitespace char of the body.
    leading_ws = len(body) - len(body.lstrip())
    sm = _SHORTHAND_APPROVE_RE.match(body[leading_ws:])
    if sm:
        approve_line = f"{glyphs[0]} {words[0]}"
        rest = body[leading_ws + sm.end():].lstrip("\n")
        synthesised = approve_line + "\n\n" + rest if rest else approve_line
        return synthesised, 0, False
    return body, 0, True


# Cap on how much of the leaked body we feed back to the retry call.
# 6K chars (~1.5K tokens) is enough context for the model to recognise
# its own analysis and emit the verdict marker, without ballooning the
# retry input on the long-tail bodies that ran to 2.7K output tokens.
_LEAK_RETRY_BODY_CHAR_CAP = 6_000


def build_leak_retry_messages(
    leaked_body: str,
    *,
    glyphs: tuple[str, str, str] = VERDICT_GLYPHS,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> list[dict[str, str]]:
    """Build the chat message list for a quick-mode leak-retry turn.

    Quick-mode reviews are single-shot — there's no agent loop to
    self-correct when the model returns analysis prose without a
    `Verdict:` / `🟢/🟡/🔴` marker. Empirically this
    happened on ~40% of quick-mode runs with an earlier review
    model, suppressing the
    review entirely. This builds a tight follow-up turn the entrypoint
    runs through `quick_review_call`'s sibling — same LLM client,
    isolated single-turn conversation that ONLY asks the model to
    re-emit its analysis with the verdict line prepended.

    Pure function so it's testable without mocking the LLM client.
    Returns the OpenAI ChatCompletions-shaped messages list.
    """
    body_for_retry = (leaked_body or "")[:_LEAK_RETRY_BODY_CHAR_CAP]
    marker_lines = "".join(
        f"  {glyph} {word}\n" for glyph, word in zip(glyphs, words)
    )
    instruction = (
        "Your previous PR-review response analysed the change but did "
        "not begin with the required verdict-marker line. The post-"
        "processor suppressed it.\n\n"
        "Re-emit your full review with ONE of these literal strings as "
        "the FIRST LINE, on its own line, followed by your analysis:\n"
        f"{marker_lines}\n"
        "Pick the marker that matches the severity you reasoned about. "
        "Do NOT use shorthand (no `LGTM`, no `Approved`, no "
        "`Verdict:` prefix, no bolding). Output ONLY the new review "
        "body — no apology, no explanation of the change.\n\n"
        "Your previous response:\n\n"
        f"{body_for_retry}"
    )
    return [{"role": "user", "content": instruction}]


# Reasoning-model chain-of-thought blocks. Some OpenAI-compatible
# reasoning backends with the reasoning parser misconfigured
# emit `<think>...</think>` inside the standard `content`
# field. Strip before posting so the comment isn't a stream of
# consciousness.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# Orphan trailing close tag — when the parser strips `<think>` but leaks
# `</think>`, or when the model double-emits its review (once inside the
# reasoning trace, once as the real response) with only the close tag
# separating them. Observed shape: full verdict body, `</think>`, full
# verdict body again. The paired regex above can't catch this because
# there's no opening tag to match.
_ORPHAN_THINK_CLOSE_RE = re.compile(r"</think>\s*", re.IGNORECASE)


def strip_reasoning(content: str) -> tuple[str, int]:
    """Strip `<think>...</think>` chain-of-thought blocks from `content`.
    Returns (clean_content, stripped_chars). Backends with a working
    reasoning-parser flag break the trace out into `reasoning_content`
    server-side; this is the fallback for backends that leak it inline.

    After paired-block removal, treats everything up to and including
    the LAST remaining `</think>` as leaked reasoning preamble. Handles
    the observed orphan-close-tag shape where the verdict marker
    appears both inside the leaked reasoning and in the real response —
    `detect_reasoning_leak`'s "find first marker" logic would otherwise
    keep both halves and post a duplicate-body comment."""
    if not content:
        return "", 0
    cleaned = _THINK_BLOCK_RE.sub("", content)
    last_orphan = None
    for m in _ORPHAN_THINK_CLOSE_RE.finditer(cleaned):
        last_orphan = m
    if last_orphan is not None:
        cleaned = cleaned[last_orphan.end():]
    return cleaned.strip(), len(content) - len(cleaned)


# Verdict / blocker detection for the auto-merge pause. Kept here in
# the shared module so both modes apply the same brake.
_BLOCKER_LINE_RE = re.compile(r"🚨\s*\*\*Blocker:\*\*")


# Same line shape as `_verdict_line_re`, narrowed to the block-severity
# vocabulary entry (`glyphs[2]` / `words[2]`; `🔴` / `needs changes` by
# default).
@lru_cache(maxsize=8)
def _verdict_blocker_re(glyph: str, word: str) -> re.Pattern[str]:
    return re.compile(
        rf"""
        ^[^A-Za-z\n]*                  # leading whitespace / markdown / emoji
        (?:
            {re.escape(glyph)}[^A-Za-z\n]*  # new format: red-circle marker
            |
            verdict\s*[*_]*\s*:\s*[^A-Za-z\n]*  # old format: "Verdict:" prefix
        )
        {re.escape(word)}\b
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )


# Where one 🚨 Blocker bullet's "own text" ends, for retraction scanning:
# the next top-level/nested list item, or a blank-line paragraph break, or
# end of body. Deliberately generous about what still counts as the SAME
# bullet (wrapped continuation lines with no leading `-`/`*`/`1.` are kept
# in scope, so a retraction on a wrapped second line is still found) and
# conservative about what starts a NEW one — so prose the model writes
# after the findings list (e.g. a closing "I have no actual blockers."
# paragraph) never leaks into a bullet's scope, and one bullet's
# retraction never bleeds into a sibling bullet's.
_BLOCKER_BULLET_BOUNDARY_RE = re.compile(r"\n[ \t]*(?:[-*]\s|\d+[.)]\s)|\n[ \t]*\n")


def _blocker_bullet_spans(review_text: str) -> list[str]:
    """Return the local text of every `🚨 **Blocker:**` occurrence in
    `review_text` — from the marker to the next bullet/paragraph boundary
    (or end of body). Each occurrence gets its own bounded span, so a
    retraction phrase found in one bullet never discounts another."""
    spans = []
    for m in _BLOCKER_LINE_RE.finditer(review_text):
        boundary = _BLOCKER_BULLET_BOUNDARY_RE.search(review_text, m.end())
        end = boundary.start() if boundary else len(review_text)
        spans.append(review_text[m.start():end])
    return spans


# Narrow, bullet-LOCAL retraction phrases — matched only within the span
# `_blocker_bullet_spans` extracts for one 🚨 Blocker bullet, never against
# the whole body (a reassuring sentence elsewhere, e.g. "I have no actual
# blockers.", is deliberately NOT consulted — see `detect_blocker`'s
# docstring and cora#29). Each phrase reads as the model unambiguously
# withdrawing the finding it just wrote, not a hedge ("might be", "could
# be an issue") or a plain description of correct behaviour. Same posture
# as `_ci_gate._CI_CONTRADICTION_CLAIM_RE`: a false negative here just
# means a retracted bullet still counts and still posts for a human to
# read (today's behaviour, unchanged); a false positive would silently
# drop a live blocker, which is the failure mode to avoid — bias the
# pattern toward under-matching.
_BLOCKER_RETRACTION_RE = re.compile(
    r"""
    false\ alarm
    | not\ (?:actually\ |really\ )?a\ (?:real\ )?(?:bug|issue|problem)\b
    | (?:the\ )?code\ is\ (?:actually\ |in\ fact\ )?fine\b
    | \bretract(?:ing|ed)?\ this\b
    | \bdisregard\ (?:this|that)\b
    | \bignore\ (?:this|that)\ (?:finding|blocker)\b
    | \bno\ longer\ (?:a\ |an\ )?(?:concern|issue|blocker|problem)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A retraction phrase followed by a contrastive clause isn't a retraction —
# it's a narrowing. "The surrounding code is fine, BUT this path crashes"
# and "not really a problem there, HOWEVER at 10k rows it OOMs" both match
# the phrases above while stating a live defect, and discounting them would
# silently drop a real blocker (the one failure mode the phrase set is
# tuned to avoid). Scanned only in the text AFTER the matched phrase,
# inside the same bullet: a contrast before it ("this looks broken, but
# it's a false alarm") is the retraction still standing.
_RETRACTION_CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|nevertheless|still)\b", re.IGNORECASE
)


def _is_retracted(span: str) -> bool:
    """True when this bullet's own text withdraws the finding outright —
    a retraction phrase with no contrastive clause walking it back."""
    m = _BLOCKER_RETRACTION_RE.search(span)
    if not m:
        return False
    return not _RETRACTION_CONTRAST_RE.search(span, m.end())


def count_blocker_retractions(review_text: str) -> tuple[int, int]:
    """`(total_blocker_bullets, retracted_count)` for one review body.

    Exposes what `detect_blocker` computes internally so the verdict
    path can act on it too (cora #38). `detect_blocker` answers "pause
    automerge?" and already discounts retracted bullets — but the
    VERDICT line was left alone, so a review whose every Blocker
    retracts still posted 🔴 `needs changes`. Same split the CI gate
    draws, and the same whole-review rule applies there: one retracted
    bullet among several live ones changes nothing."""
    spans = _blocker_bullet_spans(review_text)
    return len(spans), sum(1 for span in spans if _is_retracted(span))


def detect_blocker(
    review_text: str,
    *,
    glyphs: tuple[str, str, str] = VERDICT_GLYPHS,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> bool:
    """True if the verdict is `needs changes` (the block-severity entry,
    `glyphs[2]` / `words[2]`) OR any `🚨 **Blocker:**` bullet in the body
    is still LIVE — i.e. wasn't retracted in its own text (see
    `_blocker_bullet_spans` / `_BLOCKER_RETRACTION_RE`). Either signal
    pauses auto-merge.

    The verdict-word check is untouched by retraction: a model that
    literally writes the `needs changes` marker is stating its own
    conclusion, not describing a bullet that can be locally withdrawn —
    second-guessing that word is a prompting/consistency problem, not
    this function's job. Only the blocker-bullet scan below is
    retraction-aware, per cora#29's "Related" item: a bullet a model
    talks itself out of in the same breath shouldn't pause automerge on
    its own."""
    if not review_text:
        return False
    if _verdict_blocker_re(glyphs[2], words[2]).search(review_text):
        return True
    spans = _blocker_bullet_spans(review_text)
    if not spans:
        return False
    live = [span for span in spans if not _is_retracted(span)]
    if live:
        return True
    # Every 🚨 Blocker bullet retracted itself locally — surface that a
    # discount happened instead of silently dropping it (the engine's
    # posture elsewhere is annotate/log, never delete quietly).
    _gha_log(
        f"detect_blocker retracted-bullets-discounted count={len(spans)}"
    )
    return False


# Verdict → check-run conclusion mapping. Maps the three Verdict values
# the system prompt asks for to GitHub's check-conclusion enum.
# `failure` on `needs changes` is load-bearing: a fail-closed
# required-check reviewer gate blocks the merge on any verdict-check
# conclusion that isn't `success`/`neutral`, so a `failure` here is what
# actually blocks auto-merge when the reviewer flags a blocker.
# The label-removal path (`pause_automerge`) still fires as a belt-and-
# suspenders gate, but the check now has teeth on its own.
# Keyed by position in the verdict vocabulary so a custom `words` set
# maps the same way: words[0] → success, words[1] → neutral,
# words[2] → failure. The default mapping is unchanged:
#   {"looks good": "success", "minor": "neutral", "needs changes": "failure"}
_CONCLUSIONS = ("success", "neutral", "failure")


@lru_cache(maxsize=8)
def _verdict_to_conclusion_map(words: tuple[str, ...]) -> dict[str, str]:
    # Lowercased keys — `parse_verdict_from_body` lowercases the
    # captured word before lookup, and the default words are already
    # lowercase.
    return {w.lower(): c for w, c in zip(words, _CONCLUSIONS)}


def parse_verdict_from_body(
    body: str,
    *,
    glyphs: tuple[str, str, str] = VERDICT_GLYPHS,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> str | None:
    """Extract the lowercase verdict word from the body's verdict line.
    Returns None if the body doesn't have a parseable verdict marker
    (colored circle or `Verdict:` prefix followed by one of the three
    known verdict words).

    The verdict word is captured directly by the verdict-line regex's
    group 1 — no further string parsing needed."""
    if not body:
        return None
    m = _verdict_line_re(glyphs, words).search(body[:_VERDICT_FULL_SCAN_CHARS])
    if not m:
        return None
    verdict = m.group(1).lower()
    return verdict if verdict in _verdict_to_conclusion_map(words) else None


def verdict_to_conclusion(
    verdict: str | None,
    *,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> str:
    """Default to `neutral` for an unrecognised / missing verdict — the
    review ran successfully (no error) but we don't have signal on
    severity. `failure` is reserved for `needs changes` verdicts and for
    reviewer-broken states emitted directly from the entrypoint (no
    review produced, reasoning leak after retries)."""
    if not verdict:
        return "neutral"
    return _verdict_to_conclusion_map(words).get(verdict, "neutral")


# Verdict → GitHub PR-Review event mapping. Used only when the
# GitHub reporter is configured to post a first-class Review object
# (`ReviewerConfig.use_github_review`) instead of an issue comment; the
# default comment+check-run path never consults this.
#
# Keyed by position in the verdict vocabulary so a custom `words` set
# maps the same way: words[0] → COMMENT, words[1] → COMMENT,
# words[2] → REQUEST_CHANGES. The block-severity verdict is the only one
# that maps onto a *gating* event; the lower two stay advisory.
#
# `looks good` deliberately maps to COMMENT, not APPROVE: a bot APPROVE
# is undesirable in most setups. GitHub branch-protection "required
# approving reviews" counts a bot's APPROVE toward the quorum, so an
# auto-APPROVE would let the reviewer satisfy a human-review gate by
# itself — exactly the safety property the gate exists to enforce. The
# advisory verdict still rides the check-run, which keeps
# its teeth on `needs changes`; the Review event is purely additive
# surface for VCS-portability.
_REVIEW_EVENTS = ("COMMENT", "COMMENT", "REQUEST_CHANGES")


@lru_cache(maxsize=8)
def _verdict_to_review_event_map(words: tuple[str, ...]) -> dict[str, str]:
    return {w.lower(): e for w, e in zip(words, _REVIEW_EVENTS)}


def verdict_to_review_event(
    verdict: str | None,
    *,
    words: tuple[str, str, str] = VERDICT_WORDS,
) -> str:
    """Map a parsed verdict word to a GitHub PR-Review `event`
    (COMMENT / REQUEST_CHANGES / APPROVE). Defaults to `COMMENT` for an
    unrecognised / missing verdict — advisory, never gating, unless the
    verdict is the block-severity entry (`words[2]`, `needs changes` by
    default), which maps to `REQUEST_CHANGES`. `looks good` maps to
    COMMENT, not APPROVE, on purpose — see `_REVIEW_EVENTS`."""
    if not verdict:
        return "COMMENT"
    return _verdict_to_review_event_map(words).get(verdict, "COMMENT")
