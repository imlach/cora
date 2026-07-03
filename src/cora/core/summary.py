"""GHA step summary writer + the final review-comment builder.

`write_step_summary` appends to `$GITHUB_STEP_SUMMARY` so the workflow
run page shows a structured timeline alongside the check run + PR
comment. `make_review_comment` assembles the final body posted (or
edited) onto the PR — wraps the model's verdict-bearing markdown in the
v2 marker + run-profile footer (tokens, wall, backend, deeplinks).
"""

from __future__ import annotations

import os
from datetime import datetime

from cora.core.budget import Budget

from cora.core.check_run import _grafana_drilldown_url, _workflow_run_url
from cora.core.comment import _fmt_ts
from cora.core.config import COMMENT_MARKER
from cora.core.leak import parse_verdict_from_body

try:
    from comment_safe import escape_think_tags  # type: ignore
except ImportError:  # pragma: no cover
    def escape_think_tags(body: str) -> str:  # type: ignore
        return body


def write_step_summary(
    pr_number: str,
    model_alias: str,
    budget: Budget,
    wall_time_s: float,
    terminated_reason: str | None,
    final_body: str,
    is_leak: bool,
    tools_available: list[str],
) -> None:
    """Append the run's stats + outcome to $GITHUB_STEP_SUMMARY. Visible
    on the workflow run page; no-op outside GHA."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    grafana_url = _grafana_drilldown_url(pr_number)
    backend = budget.resolved_model or "unknown"
    verdict = parse_verdict_from_body(final_body) or "(no verdict parsed)"
    used_pairs = budget.tool_calls.most_common()
    used_line = ", ".join(f"`{n}`×{c}" for n, c in used_pairs) if used_pairs else "_(none)_"
    unused = sorted(set(tools_available) - set(budget.tool_calls.keys()))
    unused_line = ", ".join(f"`{n}`" for n in unused) if unused else "_(none)_"
    md_lines = [
        f"## cora review (loop) — PR #{pr_number}",
        "",
        f"- **Verdict**: {verdict}{' · ⚠️ reasoning leak' if is_leak else ''}",
        f"- **Backend**: `{backend}` (alias `{model_alias}`)",
        f"- **Stats**: {budget.iterations} tool calls · "
        f"{budget.input_used:,} in / {budget.output_used:,} out tokens · "
        f"{wall_time_s:.1f}s wall",
        f"- **Terminated**: {terminated_reason or 'natural finish'}",
        f"- **Tools used**: {used_line}",
        f"- **Tools unused**: {unused_line}",
    ]
    if grafana_url:
        md_lines.append(f"- **Drilldown**: [grafana]({grafana_url})")
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::step summary write failed: {exc}")


def make_review_comment(
    model: str,
    body: str,
    budget: Budget,
    wall_time_s: float,
    terminated_reason: str | None,
    tools_available: list[str] | None = None,
    pr_number: str | None = None,
    *,
    mode: str = "deep",
    bot_author: bool = False,
    retrieval_source: str = "none",
    automerge_paused: bool = False,
    reasoning_stripped_chars: int = 0,
    started_at: datetime | None = None,
) -> str:
    # Backend resolution (the concrete backend LiteLLM picked for the
    # requested alias) lives in the
    # footer summary alongside the other stats. Earlier rev had it in
    # the title as "alias `review` → `<resolved>`" which (a) made the
    # title noisy and (b) read confusingly when the resolved name was
    # the raw api_base URL (before the backend mapper knew about the
    # deployment's endpoints).
    #
    # `<code>` tags instead of markdown backticks: GitHub doesn't
    # reliably render markdown inside <details><summary> or <sub>
    # blocks, so backticks rendered as literal `chars` in the footer.
    # <code> is HTML; always renders.
    # Format: "endpoint: <alias> (<resolved>)". Alias-first reads as
    # "client asked for X, LiteLLM resolved to Y" — natural ordering
    # for the human glancing at the footer. The earlier "backend: Y
    # (alias X)" inverted the natural reading. Parens elided when the
    # alias and resolved name match (alias-named resolve case).
    if budget.resolved_model and budget.resolved_model != model:
        backend_chip = f"endpoint: <code>{model}</code> (<code>{budget.resolved_model}</code>)"
    elif budget.resolved_model:
        backend_chip = f"endpoint: <code>{model}</code>"
    else:
        backend_chip = f"endpoint: <code>{model}</code> (resolved: unknown)"

    # Single foldable details block carries the full call breakdown
    # (used + unused tools, tokens, wall, termination reason).
    used_pairs = budget.tool_calls.most_common()
    unused = (
        sorted(set(tools_available) - set(budget.tool_calls.keys()))
        if tools_available
        else []
    )
    # Tool-call summary distinguishes calls (iterations) from tool
    # types touched. Previous "10 tool calls (2 used / 7 unused)"
    # read confusingly — the parenthetical referred to distinct tool
    # *types*, not call counts, so the two numbers seemed to
    # contradict each other. Rephrased to make the units explicit.
    if used_pairs:
        n_total_tools = len(used_pairs) + len(unused) if unused else len(used_pairs)
        calls_chip = (
            f"{budget.iterations} calls across {len(used_pairs)} tool"
            + ("s" if len(used_pairs) != 1 else "")
        )
        if unused:
            calls_chip += f" ({len(unused)}/{n_total_tools} unused)"
    else:
        calls_chip = f"{budget.iterations} tool calls"
    # Two-line footer, split for scannability:
    #   line 1 — run profile: mode · backend · [calls ·] tokens · wall
    #            · [terminated / reasoning-stripped flags when present]
    #   line 2 — meta: started · context · [auto-merge paused] · logs · grafana
    # Earlier rev packed everything onto one chip strip including both
    # `started` and `completed` timestamps; the duration already says
    # how long, so start + wall is enough to know when, and the line
    # break keeps the strip readable as more chips accumulate.
    top_bits: list[str] = [f"mode: <code>{mode}</code>", backend_chip]
    if mode == "deep":
        top_bits.append(calls_chip)  # quick has no tools, no breakdown
    top_bits.append(
        f"~{budget.input_used} in / ~{budget.output_used} out tokens"
    )
    top_bits.append(f"{wall_time_s:.1f}s wall")
    if reasoning_stripped_chars:
        top_bits.append(f"reasoning stripped ({reasoning_stripped_chars} chars)")
    if terminated_reason:
        top_bits.append(f"terminated: {terminated_reason}")

    bottom_bits: list[str] = []
    if started_at is not None:
        bottom_bits.append(f"started {_fmt_ts(started_at)}")
    if bot_author:
        bottom_bits.append("conventions skipped (bot author)")
    else:
        bottom_bits.append(f"context: {retrieval_source}")
    if automerge_paused:
        bottom_bits.append("⏸ auto-merge paused (blocker)")
    # `logs` + `grafana` stay on the meta line — clickable without
    # expanding the details block. Order matches the initial-comment
    # footer (workflow logs first, grafana second).
    run_url = _workflow_run_url()
    if run_url:
        bottom_bits.append(f'<a href="{run_url}">logs</a>')
    grafana_url = _grafana_drilldown_url(pr_number) if pr_number else ""
    if grafana_url:
        bottom_bits.append(f'<a href="{grafana_url}">grafana</a>')

    top_line = " · ".join(top_bits)
    bottom_line = " · ".join(bottom_bits)
    # `<br>` (HTML) renders inside both `<sub>` and `<details><summary>`;
    # GFM markdown wouldn't (per the `<code>` note above).
    summary_line = (
        f"{top_line}<br>{bottom_line}" if bottom_line else top_line
    )

    detail_lines: list[str] = []
    if used_pairs:
        detail_lines.append(
            "**used**: " + ", ".join(f"`{n}`×{c}" for n, c in used_pairs)
        )
    if unused:
        detail_lines.append("**unused**: " + ", ".join(f"`{n}`" for n in unused))

    if detail_lines:
        footer_block = (
            f"\n\n<details><summary>{summary_line}</summary>\n\n"
            + "\n".join(detail_lines)
            + "\n\n</details>"
        )
    else:
        footer_block = f"\n\n<sub>{summary_line}</sub>"

    # Escape any unpaired `<think>` / `</think>` in the body so GitHub
    # markdown doesn't render them as HTML tags. The loop reviewer's
    # body is the model's stream output; even when the model uses
    # backtick code spans correctly, the chunk before the first
    # backtick can contain raw tag substrings that swallow lines.
    safe_body = escape_think_tags(body)
    return (
        f"{COMMENT_MARKER}\n"
        f"## cora review\n\n"
        f"{safe_body.strip()}{footer_block}"
    )
