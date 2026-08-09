"""Mandatory T2 escalation when a T0 verdict includes a `propose_patch`
directive.

Propose-patch directives are *writes* — they create a branch and open a
draft PR (out-of-hunk path) or post inline `suggestion` comments
(in-hunk path) under the cora App identity. The bar should be higher
than a comment-only verdict, where T2 is opt-in via the
`AGENT_REVIEW_T2_DISAGREEMENT` soak gate.

This module mandates T2 *regardless* of the existing soak gate when the
verdict carries a `propose_patch` block. The T2 alt-reviewer is framed
as a *patch-verifier*, not an independent reviewer: same diff + T0's
verdict body (with the proposed patch) handed to it, asked to judge
whether the proposed edits correctly address an actual issue.

Behaviour per user spec ("annotate + still apply + flag for human"):

  - T2 agrees   → apply normally, no annotation.
  - T2 disagrees → apply anyway, add `escalation-disagreement` label to
                   the draft PR (or source PR for inline / push-to-source
                   paths), prepend a warning to the draft PR body / inline
                   review summary, footer-tag the verdict comment.
  - T2 fails    → apply with a softer `escalation-skipped` label + body
                   note quoting the failure reason. Intent is to escalate
                   when possible, not to gate patches on T2 availability.

Kill switch: `AGENT_REVIEW_PATCH_ESCALATION` (default "true"). When
"false", the escalation is bypassed entirely and patches apply via the
pre-existing path — emergency disable.

The module is pure-logic where it can be: the verdict comparison,
warning text composition, and outcome shape are all testable without
the network. The T2 dispatch itself reuses `call_t2_alt_reviewer` from
`t2_dispatch.py`; label application reuses `apply_label`-style
subprocess calls.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

# Env knob — default "true", set "false" to bypass entirely.
ESCALATION_ENV_VAR = "AGENT_REVIEW_PATCH_ESCALATION"

# Labels added to the affected PR (draft, source, or inline-target).
LABEL_DISAGREEMENT = "escalation-disagreement"
LABEL_SKIPPED = "escalation-skipped"


EscalationVerdict = Literal["agree", "disagree", "skipped"]


@dataclass(frozen=True)
class EscalationOutcome:
    """The escalation decision the dispatcher consumes.

    `warning_body` and `warning_summary` are pre-rendered markdown
    fragments the dispatcher injects into the draft-PR body or inline-
    review summary. `verdict_footer` is a short footer line tagged onto
    the verdict comment posted on the source PR (visible regardless of
    which dispatch path landed).

    `label` is the GitHub label name to apply to the PR; None when no
    annotation is needed (T2 agreed). `flagged_for_human` is the
    boolean that goes into the structured log line so dashboards can
    rank PRs needing manual review.
    """

    verdict: EscalationVerdict
    warning_body: str | None
    warning_summary: str | None
    verdict_footer: str | None
    label: str | None
    flagged_for_human: bool
    t2_reason: str | None  # short text — failure reason or dissent gist


def escalation_enabled(enabled: bool | None = None) -> bool:
    """Read the kill switch. Defaults to enabled — only the explicit
    string "false" (case-insensitive) disables.

    `enabled` is the config-threaded value (`cfg.patch_escalation`);
    when None (direct engine calls without a config — same pattern as
    `deep_review._thinking_extra_body`) fall back to the legacy env
    knob so pre-config callers keep their behaviour.
    """
    if enabled is not None:
        return enabled
    raw = os.environ.get(ESCALATION_ENV_VAR, "true").strip().lower()
    return raw != "false"


def build_patch_verification_prompt(
    *,
    base_initial_user_prompt: str,
    t0_verdict_body: str,
) -> str:
    """Frame the T2 dispatch as patch verification, not fresh review.

    Prepend a system-style preamble that pins T2's job: assess whether
    T0's proposed patch correctly addresses an actual issue. The base
    prompt (diff + retrieval pre-pack) is preserved so T2 has the same
    code context T0 worked from.

    T0's full verdict body (including the `propose_patch` block) is
    embedded verbatim — T2 needs to see *what* T0 wants to change to
    evaluate it. Quoted in a code fence so propose_patch JSON inside
    doesn't get re-parsed as T2's own directive.
    """
    preamble = (
        "**You are a patch-verifier, not an independent reviewer.**\n\n"
        "Another reviewer (T0) produced the verdict below for this PR. "
        "The verdict contains a `propose_patch` directive — concrete edits "
        "T0 wants to apply. Your job is to judge whether T0's proposed "
        "patch correctly addresses an actual issue in the diff.\n\n"
        "Answer with a normal review body. Use the standard `Verdict:` "
        "marker:\n"
        "  - `Verdict: 🟢 looks good` — T0's patch is a valid fix.\n"
        "  - `Verdict: 🟡 minor` — patch is mostly right but has nits.\n"
        "  - `Verdict: 🔴 needs changes` — patch is wrong, unnecessary, "
        "or misdiagnoses the issue.\n\n"
        "Do NOT emit your own `propose_patch` block. Your role is "
        "verification of T0's, not authoring a replacement.\n\n"
        "---\n\n"
        "**T0's verdict (with proposed patch):**\n\n"
        "````\n"
        f"{t0_verdict_body.rstrip()}\n"
        "````\n\n"
        "---\n\n"
        "**Original review context (same diff + retrieval pre-pack T0 used):**\n\n"
    )
    return preamble + base_initial_user_prompt


def classify_t2_verdict(
    *,
    t2_body: str | None,
    t2_terminated_reason: str | None,
) -> tuple[EscalationVerdict, str | None]:
    """Map the T2 call result to one of (agree, disagree, skipped).

    Returns `(verdict, reason)`. `reason` is a short human-readable
    string used in annotations:
      - skipped → the termination reason ("probe-failed", "wall_time", …)
      - disagree → T2's verdict word
      - agree → None

    Skip conditions are deliberately broad: any T2 call that didn't
    produce a usable body (probe-failed, agent-loop-errored,
    per_call_timeout, wall_time, no body, leaked reasoning) collapses
    to "skipped" — the intent is to escalate when possible, not to
    block patches on T2 availability.

    Verdict mapping uses `parse_verdict_from_body` (same as the rest of
    the pipeline). `looks good` → agree; everything else → disagree.
    The patch-verification prompt frames `looks good` specifically as
    "T0's patch is a valid fix", so a same-prompt T2 saying anything
    else is, by definition, dissent on the patch.
    """
    from cora.core.leak import parse_verdict_from_body

    if not t2_body:
        reason = t2_terminated_reason or "no body"
        return "skipped", reason

    verdict = parse_verdict_from_body(t2_body)
    if verdict is None:
        # T2 produced a body but no parseable verdict — treat as
        # skipped rather than implicit-disagree. Annotation will be the
        # softer label so the patch still goes through.
        return "skipped", "no verdict parsed"

    if verdict == "looks good":
        return "agree", None

    if verdict == "minor":
        # `minor` is "patch is mostly right but has nits" per the
        # patch-verifier prompt — too soft to gate behind a "human
        # review required" banner, which would produce false-flag
        # noise on otherwise-fine patches. Treat as agree; the nits
        # surface in the verdict footer for visibility but don't
        # trigger the hard `escalation-disagreement` label.
        return "agree", verdict

    # `needs changes` / blocker → real dissent on the patch. The
    # patch-verifier prompt frames `needs changes` as "patch is wrong,
    # unnecessary, or misdiagnoses the issue" — hard banner appropriate.
    return "disagree", verdict


def _extract_dissent_summary(t2_body: str, max_chars: int = 280) -> str:
    """Pull a short summary of T2's reasoning out of its body for the
    warning text. Best-effort — takes the first non-empty line after
    the `Verdict:` marker, falling back to the first non-empty line of
    the body. Truncates to `max_chars` with an ellipsis.

    The full T2 body is too long to inline into the draft PR body or
    inline review summary; a 1-2 sentence gist is what the human needs
    to decide whether to open the patch.
    """
    if not t2_body:
        return ""

    lines = [line.strip() for line in t2_body.splitlines()]
    # Find the line after `Verdict:` (any verdict word) — that's
    # typically the lead-in to T2's reasoning.
    verdict_idx: int | None = None
    for i, line in enumerate(lines):
        if line.lower().startswith("verdict:"):
            verdict_idx = i
            break

    candidates = lines[verdict_idx + 1:] if verdict_idx is not None else lines
    for line in candidates:
        if line and not line.startswith("#") and not line.startswith("---"):
            if len(line) > max_chars:
                return line[: max_chars - 1] + "…"
            return line
    return ""


def compose_escalation_outcome(
    *,
    t2_verdict: EscalationVerdict,
    t2_reason: str | None,
    t2_body: str | None,
    t2_model_alias: str,
) -> EscalationOutcome:
    """Translate the T2 verdict into the annotation shape the
    dispatcher needs (warning text, label, footer, flag).

    Four shapes:

      - agree (clean)    → no annotation, no label, no flag.
      - agree (with nits) → no banner / label / flag, but a verdict
                  footer surfaces T2's nits as informational. Used
                  when T2 returned `minor` ("patch mostly right with
                  nits") — too soft for the hard "human review
                  required" banner but worth showing.
      - skipped → softer annotation. `escalation-skipped` label + body
                  note quoting the failure reason. Not flagged for human
                  review (T2 didn't actually dissent — it just couldn't
                  participate).
      - disagree → hard annotation. `escalation-disagreement` label +
                  prominent warning prepended to body / summary +
                  verdict footer + flagged for human.
    """
    if t2_verdict == "agree":
        # `t2_reason == "minor"` is the nits-surfaced sub-case;
        # `None` is the clean-agree case (T2 said `looks good`).
        if t2_reason == "minor":
            nit_gist = _extract_dissent_summary(t2_body or "")
            footer = (
                f"_T2 alt-reviewer (`{t2_model_alias}`) noted nits on the "
                f"patch but did not block. No label / banner applied._"
            )
            if nit_gist:
                footer += f"\n\n> {nit_gist}"
            return EscalationOutcome(
                verdict="agree",
                warning_body=None,
                warning_summary=None,
                verdict_footer=footer,
                label=None,
                flagged_for_human=False,
                t2_reason="minor",
            )
        return EscalationOutcome(
            verdict="agree",
            warning_body=None,
            warning_summary=None,
            verdict_footer=None,
            label=None,
            flagged_for_human=False,
            t2_reason=None,
        )

    if t2_verdict == "skipped":
        reason = t2_reason or "unknown"
        # Terse single-line note across all surfaces — the patch is
        # still trustworthy under the same posture it had pre-
        # escalation (T0 alone), so no banner shouting. The label is
        # what the dashboard / human filter on; the note is just a
        # short trail explaining why no T2 verdict landed.
        note = (
            f"_T2 escalation skipped ({reason}) — patch applied on "
            f"T0's verdict alone._"
        )
        return EscalationOutcome(
            verdict="skipped",
            warning_body=note,
            warning_summary=note,
            verdict_footer=note,
            label=LABEL_SKIPPED,
            flagged_for_human=False,
            t2_reason=reason,
        )

    # disagree
    dissent_gist = _extract_dissent_summary(t2_body or "")
    verdict_word = t2_reason or "needs changes"

    warning_body = (
        f"⚠️ **Escalation disagreement** — T2 alt-reviewer "
        f"(`{t2_model_alias}`) disagrees with this patch. "
        f"Human review required before merge.\n\n"
        f"**T2's verdict:** {verdict_word}\n"
    )
    if dissent_gist:
        warning_body += f"\n> {dissent_gist}\n"
    warning_body += (
        f"\n_Applied per the `annotate + still apply + flag for human` "
        f"policy. Look for the `{LABEL_DISAGREEMENT}` label (soft-fail: "
        f"may be absent on label-apply errors — the banner above is the "
        f"load-bearing signal)._"
    )

    warning_summary = (
        f"⚠️ **Escalation disagreement** — T2 alt-reviewer "
        f"(`{t2_model_alias}`) disagrees with these inline suggestion(s) "
        f"({verdict_word}). Human review required before applying."
    )
    if dissent_gist:
        warning_summary += f"\n\n> {dissent_gist}"

    # Footer wording is deliberately neutral about label state — the
    # label-apply call in the dispatcher caller is soft-fail (gh api
    # may 403 on rate limits, or the label may not exist in the repo
    # yet). If the label failed, the warning_body banner is still
    # present and carries the load-bearing signal; promising "label
    # added" in the footer would be dishonest when it isn't.
    footer = (
        f"_Escalation outcome: T2 disagree ({verdict_word}). "
        f"Human review required before merge._"
    )
    return EscalationOutcome(
        verdict="disagree",
        warning_body=warning_body,
        warning_summary=warning_summary,
        verdict_footer=footer,
        label=LABEL_DISAGREEMENT,
        flagged_for_human=True,
        t2_reason=dissent_gist or verdict_word,
    )


def apply_label_to_pr(
    *,
    repo: str,
    pr_number: str,
    label: str,
    gh_token: str | None,
) -> tuple[bool, str | None]:
    """Soft-fail label application. Mirrors the label-apply subprocess
    shape used elsewhere in the pipeline, with the cora App token when
    available so the audit actor matches the rest of the patch-dispatch
    trail.

    Returns `(success, error_message)` — one True / None or False / err.
    """
    env = os.environ.copy()
    if gh_token:
        env["GH_TOKEN"] = gh_token
    proc = subprocess.run(
        [
            "gh", "api",
            "-X", "POST",
            f"repos/{repo}/issues/{pr_number}/labels",
            "--input", "-",
        ],
        input=json.dumps({"labels": [label]}),
        env=env,
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:300]
        return False, err or "gh api returned non-zero"
    return True, None


def pr_number_from_url(url: str | None) -> str | None:
    """Pluck the PR number off the tail of a GitHub PR URL. Returns
    None if the URL isn't shaped like `…/pull/<N>` or `…/pulls/<N>`."""
    if not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


def log_escalation_outcome(
    *,
    pr_number: str,
    t2_verdict: EscalationVerdict,
    patch_kind: str,
    flagged_for_human: bool,
    log: Callable[[str], None],
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single-line structured log event the Loki dashboard
    parses on. Field shape mirrors `log_tier_verdict` / `iter_log`:
    `agent_review escalation_outcome pr_number=… t2_verdict=… …`.

    `patch_kind` is one of `draft_pr`, `inline`, `push_to_source`, or
    the comma-joined union if multiple paths fired on the same
    directive (rare but possible).
    """
    parts = [
        "agent_review escalation_outcome",
        f"pr_number={pr_number}",
        f"t2_verdict={t2_verdict}",
        f"patch_kind={patch_kind}",
        f"flagged_for_human={str(flagged_for_human).lower()}",
    ]
    if extra:
        for k, v in extra.items():
            # Logfmt-friendly: collapse whitespace.
            sv = str(v).replace(" ", "_")
            parts.append(f"{k}={sv}")
    log(" ".join(parts))


def patch_kind_from_dispatch_outcome(dispatch_outcome: dict) -> str:
    """Derive a single `patch_kind` label from the dispatcher's
    multi-bucket outcome dict. When more than one mechanism fired
    (e.g. inline + draft for a multi-edit directive), join with `+`
    so the log line carries the full shape.

    Returns `"none"` if no mechanism produced a successful artifact —
    callers should still emit the log line for observability (the
    escalation ran even if dispatch flopped).
    """
    kinds: list[str] = []
    if dispatch_outcome.get("inline_count"):
        kinds.append("inline")
    if dispatch_outcome.get("draft_count"):
        kinds.append("patch_pr")
    return "+".join(kinds) if kinds else "none"
