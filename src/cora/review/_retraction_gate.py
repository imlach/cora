"""Verdict derivation from the surviving finding set (cora #38).

`detect_blocker` (`core/leak.py`) already discounts a `🚨 Blocker`
bullet the model retracts inside its own text — that landed for #32 and
governs whether automerge pauses. The VERDICT line was left untouched,
so a review whose every Blocker self-retracts still posted 🔴
`needs changes`: the gate stopped pausing automerge over nothing while
the required check went red over the same nothing.

This closes that half — the "verdict derivation from the *filtered*
finding set" item in #38.

Design mirrors `_ci_gate` deliberately, because the situations are the
same shape (findings the review's own evidence disproves):

- **Whole-review retraction required.** One retracted bullet among live
  ones changes nothing. A review with a real blocker and a withdrawn one
  is still a blocked review.
- **Findings are kept, never deleted.** The body is annotated with a
  harness note and the verdict drops one step (`needs changes` →
  `minor`). A human still sees exactly what the model wrote.
- **Runs BEFORE the check-run posts**, since `complete_check` is
  first-write-wins — a downgrade decided after the post would leave a
  red required check standing for a verdict no longer held.

**What this does NOT fix.** The gate can only act on retractions
`_BLOCKER_RETRACTION_RE` recognises, and that pattern is deliberately
narrow — it biases toward under-matching, because a false positive
silently drops a live blocker. #38's own examples ("This logic appears
sound", "so this case is unreachable. Good.") do not match it, and
widening the phrase list to chase them trades the safe failure
direction for the dangerous one. The durable fix is upstream: have the
model emit only findings it still stands behind, rather than
regex-scrubbing exploratory reasoning after the fact. This gate is the
backstop, and `prompts/deep.md`'s "a finding is a conclusion, not an
investigation" rule is the source fix.
"""

from __future__ import annotations

from cora.core.leak import count_blocker_retractions
from cora.core.log import _gha_log
from cora.review._state import ReviewRun


def _downgrade_verdict(
    body: str,
    *,
    glyphs: tuple[str, str, str],
    words: tuple[str, str, str],
    total: int,
) -> str:
    """Swap the leading verdict line for the one-step-down entry and
    insert an explanatory line. The body's first line is the verdict
    marker by construction (`detect_reasoning_leak` guarantees the
    posted body starts there), so this leaves every finding intact."""
    _, _, rest = body.partition("\n")
    plural = "s" if total != 1 else ""
    explainer = (
        "_(harness note: verdict downgraded from "
        f"{glyphs[2]} {words[2]} — all {total} 🚨 Blocker finding{plural} "
        "below withdraw themselves in their own text, so no live blocker "
        "remains. Findings are kept below for human judgment.)_"
    )
    return f"{glyphs[1]} {words[1]}\n\n{explainer}\n{rest.lstrip(chr(10))}"


def apply_retraction_verdict_gate(run: ReviewRun) -> None:
    """Downgrade a block-severity verdict whose every `🚨 Blocker`
    bullet retracts itself. No-op unless all of these hold:

    - the gate is enabled (`cfg.retraction_verdict_gate`, default on);
    - there is a body and a parsed verdict to act on;
    - the verdict is the block-severity entry (`words[2]`) — a review
      that already landed 🟡/🟢 has nothing to downgrade;
    - the body has at least one `🚨 Blocker` bullet and EVERY one of
      them is retracted.
    """
    cfg = run.cfg
    if not getattr(cfg, "retraction_verdict_gate", True):
        return
    body = run.body_to_post
    if not body or not run.verdict:
        return

    words = cfg.verdict_words
    if run.verdict.strip().lower() != words[2].lower():
        return

    total, retracted = count_blocker_retractions(body)
    if total == 0 or retracted != total:
        if total:
            _gha_log(
                f"retraction_verdict_gate pr_number={run.pr_number} "
                f"outcome=skip reason=live-blockers total_blockers={total} "
                f"retracted={retracted}"
            )
        return

    run.body_to_post = _downgrade_verdict(
        body, glyphs=cfg.verdict_glyphs, words=words, total=total
    )
    run.verdict = words[1].lower()
    _gha_log(
        f"retraction_verdict_gate pr_number={run.pr_number} "
        f"outcome=downgraded total_blockers={total} retracted={retracted}"
    )
    run.loki(
        f"agent_review retraction_verdict_gate pr_number={run.pr_number} "
        f"total_blockers={total} retracted={retracted} downgraded=true",
        labels={"consumer": "pr-review", "kind": run.mode},
    )
