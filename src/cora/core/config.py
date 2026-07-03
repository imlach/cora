"""Tunables, paths, and constants for the agentic PR reviewer.

Pure constants module — no behavior, no side effects beyond reading a
single env var for `GRAFANA_BASE`. Sibling modules import from here so
the rest of the package can stay stateless and unit-testable.
"""

from __future__ import annotations

import os
from pathlib import Path


# REPO_ROOT — the checked-out repo the reviewer operates on (greps,
# git-shows, reads prompts from). Resolved at runtime, NOT from this
# file's install location: installed `cora` lives in site-packages, so a
# `__file__`-relative root would point into the venv, not the repo under
# review. `CORA_REPO_ROOT` is the explicit override; otherwise the process
# CWD, since the reviewer always runs from the repo checkout root.
REPO_ROOT = Path(os.environ.get("CORA_REPO_ROOT") or os.getcwd()).resolve()
# Marker for new comments. Old markers (`<!-- agentic-review:v2 -->` from
# the pre-rename engine, `<!-- agentic-review:v1 -->` from a retired
# single-shot predecessor, `<!-- agentic-review-loop:v1 -->` from a
# retired loop-workflow predecessor) are still recognised on `find` so
# edit-last continuity survives the cora rename. Once no live PR
# carries an old-marker comment, the recognition list can drop to v1 only.
COMMENT_MARKER = "<!-- cora:v1 -->"
LEGACY_COMMENT_MARKERS = (
    "<!-- agentic-review:v2 -->",
    "<!-- agentic-review:v1 -->",
    "<!-- agentic-review-loop:v1 -->",
)

# Verdict check-run name. It IS the authoritative verdict a deployment's
# fail-closed required-check merge gate can key on. Deployments
# rebrand via `ReviewerConfig.check_run_name` / `REVIEW_CHECK_RUN_NAME`.
CHECK_RUN_NAME = "cora"

# Post the final verdict as a first-class GitHub PR Review object
# (`POST /pulls/{n}/reviews` with body + event) instead of an issue
# comment. Default-OFF: the comment + check-run path stays the
# stock behaviour. When ON,
# the GitHub reporter maps the verdict to a Review event via
# `leak.verdict_to_review_event` (block-severity → REQUEST_CHANGES,
# otherwise COMMENT — never a bot APPROVE). Deployments opt in via
# `ReviewerConfig.use_github_review` / `REVIEW_USE_GITHUB_REVIEW`.
DEFAULT_USE_GITHUB_REVIEW = False

# Generic localhost placeholders — a deployment points these at its own
# OpenAI-compatible gateway / MCP server via `LITELLM_BASE_URL` / `MCP_URL`
# (folded in by `ReviewerConfig.from_env`). Structurally required (the
# OpenAI client + MCP session need *some* URL), so they default to
# localhost rather than None: a bare run fails fast against an obvious
# unconfigured endpoint instead of silently mis-targeting. A configured
# deployment supplies its own service URLs via those env vars, so these
# defaults are never exercised there.
DEFAULT_LITELLM_BASE = "http://localhost:4000"
DEFAULT_MODEL = "review"  # LiteLLM alias

# Which dialect `cora.core.agent.make_review_agent` speaks to the LLM.
# `"openai-compatible"` (the default) is the existing behaviour —
# talk OpenAI Chat Completions to whatever `llm_base_url` points at
# (a LiteLLM gateway, vLLM-direct, OpenRouter, ...). `"anthropic"` /
# `"bedrock"` are gateway-less direct-SDK paths for an adopter with
# only an Anthropic API key / AWS credentials — no proxy in between.
# Mirrored as `ReviewerConfig.llm_provider` (env `LLM_PROVIDER`); the
# default keeps every existing deployment byte-identical.
DEFAULT_LLM_PROVIDER = "openai-compatible"
SUPPORTED_LLM_PROVIDERS: frozenset[str] = frozenset(
    {"openai-compatible", "anthropic", "bedrock"}
)
DEFAULT_MCP_URL = "http://localhost:8080/mcp"

# Cold-start pretrigger — which `model` aliases the warmup fires at (see
# `cora.core.pretrigger`). Only aliases that scale-from-zero pay a cold
# start worth hiding behind retrieval/setup; always-on endpoints don't.
# Empty (the default) disarms the pretrigger entirely. Mirrored as
# `ReviewerConfig.pretrigger_warmup_models` (CSV env
# `CORA_PRETRIGGER_WARMUP_MODELS`); a deployment lists its scale-from-zero
# aliases there.
PRETRIGGER_WARMUP_MODELS: frozenset[str] = frozenset()

# Retrieval. Top-K relevant chunks
# from the configured Qdrant collection bake straight into the
# initial prompt so the agent loop typically completes in 1-2 turns
# instead of spending 4-6 on `search_knowledge` / `read_decision`. Soft-
# falls back to no-retrieval on any HTTP failure — the agent still has
# tools in deep mode, and quick mode tolerates losing the RAG section.
# Generic localhost placeholders for the TEI+Qdrant retrieval backend —
# a deployment that uses it points these at its own services via
# `QDRANT_URL` / `TEI_URL` / `RERANKER_URL` (folded in by `from_env`). A
# bare OSS run uses the retrieval-free default (NullRetrievalProvider) and
# never touches them; a configured deployment supplies its URLs via the
# env vars, so these defaults are never exercised there.
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_TEI_URL = "http://localhost:8080"
DEFAULT_RERANKER_URL = "http://localhost:8081"
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cora-knowledge")
# Top-K final results bundled into the prompt. Bumped from
# 8 → 10 to give the agent slightly more in-pack context now that the
# two-stage rerank has higher precision — false positives in
# the top slots cost less when reranker quality is up. Conservative
# bump on purpose: the smaller T0 context window (e.g. 48 K) plus the
# other caps (diff 32 K, claude.md 17 K, PR body 3 K, CI 9 K) means
# top_k * doc_cap has to fit alongside those. Watch the
# search_knowledge calls-per-review rate in telemetry: if it doesn't
# trend down, top_k=10 isn't earning its keep and we should revert.
RETRIEVAL_TOP_K = 10
# Was 40 (4× top_k) — but the TEI reranker server has a hard
# `max_batch_size: 32` and rejects bigger batches with HTTP 422,
# which cascades through retrieve_relevant_docs as a top-level
# error (tei_rerank is unwrapped — fix that separately by adding
# batching inside tei_rerank). 32 keeps stage 1 rerank within the
# server's limit; 3.2× top_k still gives the rerankers enough
# candidates to find the right top 10.
RETRIEVAL_OVERFETCH_K = 32
# Two-stage rerank survivor count. Stage 1 reranks all
# overfetched candidates on their cheap `snippet` payloads to narrow
# down to ``RETRIEVAL_STAGE1_SURVIVORS``. Stage 2 then reranks those
# survivors on their full bodies (more expensive but more accurate),
# and we keep ``RETRIEVAL_TOP_K`` after that. The body-fetch happens
# between the two stages so we only pay it for the survivors, not
# all overfetched candidates.
RETRIEVAL_STAGE1_SURVIVORS = 20  # 2× top_k — bumped along with top_k
# Per-doc body cap. Bumped 5_000 → 6_000 — sized against the smaller
# T0 context window (e.g. 48 K); with top_k=10, the worst-case
# retrieved-docs section is 60 K chars (~15 K tokens), comfortably
# below the input budget when added to diff + claude.md + PR body + CI
# context. In practice retrieved doc bodies are well below the
# cap and total usage is much less. Revert if `query_chars`
# telemetry or LiteLLM input-token counters spike.
RETRIEVAL_DOC_CHAR_CAP = 6_000   # per-doc body cap (sized to the T0 context budget)
RETRIEVAL_QUERY_CHAR_CAP = 2_000 # how much of the diff to use as search query
RETRIEVAL_TIMEOUT_S = 15

# Domain vocabulary for query reformulation (`retrieval.extract_identifiers`).
# High-signal product/project/role names that a PR usually wants to look up;
# surfacing them as explicit identifiers gives the sparse BM25 channel concrete
# handles and seeds the dense embedder's "Identifiers:" line. Decision-record
# IDs and `#1234` PR references are matched structurally and are NOT listed here.
#
# Entries are regex *alternation fragments*, not plain literals, so a term may
# carry metacharacters (e.g. `fetch[-_]?gate`). Matching is word-bounded
# (`\b…\b`) and case-insensitive. Empty (the default) disables vocab matching
# (structural identifiers still fire) — a deployment supplies its own domain
# terms wholesale via `ReviewerConfig.retrieval_vocab`, e.g.
# ("billing", "webhook", "kustomization", "fetch[-_]?gate").
DEFAULT_RETRIEVAL_VOCAB: tuple[str, ...] = ()

# Zero-infra local retrieval (GlobRetrievalProvider).
# Globs (relative to the checkout root) whose files get BM25-ranked
# against the PR query — e.g. ("docs/**/*.md", "*.md"). Empty = the
# provider is not selected; a TEI+Qdrant deployment keeps that stack
# and a bare adopter keeps the retrieval-free default unless they opt in.
RETRIEVAL_GLOB_INCLUDE: tuple[str, ...] = ()
# Scan ceiling — bounds the per-review filesystem walk on huge repos.
RETRIEVAL_GLOB_MAX_FILES = 500
# Per-file read ceiling: larger files are skipped outright (lockfiles,
# generated bundles) rather than truncated into noise.
RETRIEVAL_GLOB_MAX_FILE_BYTES = 200_000

# Same-PR re-run cache. When a PR is synced (push, label
# flip) and the retrieval query is identical to a recent run, serve the
# top-K from disk instead of re-doing the full pipeline. TTL is small
# because the diff IS the cache key — once the diff drifts, the query
# drifts and the key changes. Override via env for ops convenience.
RETRIEVAL_CACHE_DIR = Path(
    os.environ.get("AGENT_REVIEW_CACHE_DIR", "/tmp/agent-review-cache")
)
RETRIEVAL_CACHE_TTL_S = int(os.environ.get("AGENT_REVIEW_CACHE_TTL_S", "300"))

# Trivial-PR retrieval skip. When the PR carries one of the
# labels in this set (typically applied by a label classifier or
# Renovate/Dependabot), the reviewer skips the retrieval pipeline
# entirely — the diff IS the changelog for a version bump, and repo
# conventions / decision records don't help judge it. Saves ~500ms × ~40% of PRs.
RETRIEVAL_SKIP_LABELS: frozenset[str] = frozenset(
    s.strip().lower()
    for s in os.environ.get(
        "AGENT_REVIEW_SKIP_RETRIEVAL_LABELS",
        "deps,docs,chore-renovate",
    ).split(",")
    if s.strip()
)

# Auto-merge pause. When the reviewer's verdict is `🔴 needs changes`
# AND the PR carries `automerge`, the script strips `automerge` so GH's
# queued auto-merge cancels. Operator must consciously re-apply the label
# after judging the blocker — that's the friction we want.
AUTOMERGE_LABEL = "automerge"

# Verdict vocabulary — the three severity levels the reviewer emits, in
# ascending-concern order: (approve, nits, block). Index alignment is
# load-bearing: glyph[i] pairs with word[i], and downstream maps key off
# position (0 → check-run `success`, 1 → `neutral`, 2 → `failure`;
# disagreement rank = index). `leak.py` composes and parses verdict
# lines from these; `ReviewerConfig.verdict_glyphs` / `.verdict_words`
# default to them BY REFERENCE so the public surface can't drift while
# an adopter can still pass a custom vocabulary per call.
VERDICT_GLYPHS: tuple[str, str, str] = ("🟢", "🟡", "🔴")
VERDICT_WORDS: tuple[str, str, str] = ("looks good", "minor", "needs changes")

# Budget caps (passed into the shared loop's Budget).
#
# Per-call fit (the thing that protects a small-context T0 backend
# from 4xx-ing on
# context overflow) is enforced by `DIFF_CHAR_CAP`, `CLAUDE_MD_CHAR_CAP`,
# and `result_char_cap` (LoopConfig override) — see the math block
# above those constants. These caps below are CUMULATIVE-spend
# guards, not per-call enforcement: they catch true runaway loops
# without braking normal-but-deep reviews.
#
# Cumulative input math at the per-call caps: prompt grows ~14K
# (initial) + ~2.3K per turn (assistant+result). Sum over N turns:
#    2 turns →  ~30K cumulative
#    5 turns →  ~90K
#   10 turns → ~210K
#   16 turns → ~420K  (DEFAULT_MAX_TOOL_ITERATIONS ceiling)
# Prior 100K tripped at ~10 iterations on routine reviews — bumped
# to 250K so deep investigations don't get braked; only true runaways
# past the iteration ceiling hit it. Output cap unchanged.
MAX_INPUT_TOKENS = 250_000
MAX_OUTPUT_TOKENS = 16_000
# Quick mode is single-shot (no tools, no agent loop): the one call must
# fit the model's reasoning trace AND the verdict body. The `review`
# alias typically serves a reasoning model that can think at length, so a
# per-turn-sized cap leaves no room: the model spends the whole budget in
# `<think>` and hits `finish_reason=length` with no body, which surfaces
# as "Model token limit (N) exceeded before any response was generated"
# and a failed review. Quick gets its own ceiling;
# deep gets DEEP_MAX_OUTPUT_TOKENS below. Well within the review chain's
# typical context windows (e.g. 80K–256K).
QUICK_MAX_OUTPUT_TOKENS = 32_000
# Deep mode's PER-CALL output cap (each agent-loop turn). Same failure mode
# as quick — a turn that spends its whole budget in `<think>` finishes
# `finish_reason=length` with thinking only, pydantic-ai raises, and the
# deep loop errors out. This cap has climbed 8K → 16K →
# 32K as the review model's reasoning grew: the review reasoning model blew
# the whole 16K on turn-1 thinking of a substantive PR. 32K is a safe
# ceiling on an 80K-context T0 backend across a multi-turn loop (input
# grows ~2-3K/turn); a larger-context T1 endpoint (e.g. 256K) has ample
# room. Used by BOTH the T0
# (deep_review.py) and T1 (continuation.py) legs.
DEEP_MAX_OUTPUT_TOKENS = 32_000

# ── Spiral recovery ────────────────────────────────────────────────
# Reasoning-spiral recovery for the `review` reasoning model.
# Occasionally a turn spends its ENTIRE per-call output budget
# inside `<think>` and emits no text/tool-call — `finish_reason='length'`
# with thinking-only parts — so pydantic-ai raises `UnexpectedModelBehavior`
# ("Model token limit (N) exceeded before any response was generated").
# Quick mode soft-fails; deep mode errors out (`agent-loop-errored`). The
# spiral is high-variance (the SAME PR has reasoned 58,740 chars one run,
# 8,082 the next), so bumping the per-call cap is a treadmill.
#
# Recovery (`cora.core.spiral`) detects the thinking-only response from
# the captured messages, re-issues ONE bounded call seeded with the
# partial-reasoning tail + a "produce the final review now" directive,
# and feeds the result into the normal verdict path. Recovery KEEPS
# reasoning ENABLED (no `enable_thinking=False`) but bounds it: a tight
# output cap + a "keep further reasoning brief" lead-in. One attempt; if
# it spirals/fails again, the existing soft-fail path runs unchanged.
#
# Default-OFF so a bare engine run is unchanged — no capture wrapper,
# no retry. Opt in via `AGENT_REVIEW_SPIRAL_RECOVERY=true`. Mirrored as
# the `ReviewerConfig.spiral_*`
# fields.
SPIRAL_RECOVERY_ENABLED = False
# Bounded recovery-turn output cap — tight vs the 32K main cap. The model
# has already done the analysis; recovery only needs to commit the verdict
# body, so 12K leaves room for a brief wrap-up think without inviting a
# second runaway.
SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS = 12_000
# How much of the partial-reasoning TAIL to feed back into the recovery
# prompt. The tail is where conclusions form, so we keep the last N chars
# rather than the head (which is exploratory).
SPIRAL_RECOVERY_REASONING_CHAR_CAP = 8_000
# Iteration ceiling. Matches the prompt's "≤8 tool calls" soft
# guidance with 4 of headroom for legitimately complex reviews —
# bigger truncated-diff PRs need the slack to grep_repo / git_show
# for the missing sections (observed: a large multi-file PR hit a
# 10-cap with diff-truncated and couldn't fetch the missing part via
# read_decision / search_knowledge — 6 of 9 tools unused because
# the 10 calls were already consumed on grep + git_show).
#
# Was 16 (too generous, drove 16-turn 280K-token reviews on
# routine PRs), then 10 (too tight per the data point above).
# 12 is the data-driven compromise: lets multi-file truncated
# diffs investigate adequately without the cumulative-spend cap
# firing routinely. Override via `MAX_TOOL_ITERATIONS` env on
# the workflow when a deeper investigation is wanted (spike
# phases etc.) without editing this file.
DEFAULT_MAX_TOOL_ITERATIONS = 12

# Per-run knobs (max_per_turn_output, per_call_timeout_s, result_char_cap)
# come from LoopConfig defaults — sized for the reference review backend.

# Wall-time guards — split per tier so a slow T0 can't starve T1.
#
# The workflow's `timeout-minutes` is the hard outer bound; if we let
# the loop body run all the way to it, GitHub kills the job mid-call
# and no skip comment lands on the PR. T0 (the initial deep review)
# and T1 (continuation on the larger-context endpoint) each get their
# own deadline:
#   t0_deadline = start + T0_WALL_TIME_S
#   t1_deadline = max(now + T1_WALL_TIME_S,
#                     start + T0_WALL_TIME_S + T1_WALL_TIME_S)
# T1 always gets at least T1_WALL_TIME_S of breathing room (the
# `now + T1` arm) — if T0 finishes early, the `start + T0 + T1` arm
# wins and T1 absorbs the slack; if T0 overshoots its cap, the
# `now + T1` arm guarantees a fresh window. Pre-this-change, T0 and
# T1 shared one deadline and a slow T0 deterministically starved T1
# (observed: a 324s turn 1 ate the whole envelope).
#
# When either deadline trips, the agent loop synthesises a final
# completion from whatever context was gathered.
DEFAULT_T0_WALL_TIME_S = 360
DEFAULT_T1_WALL_TIME_S = 600
POST_HEADROOM_S = 30
# Total envelope: T0 + T1 + headroom = 990s. Workflow `timeout-minutes`
# stays at 20 (1200s) so ~210s remain for checkout / venv resolve /
# MCP probes / comment post / check-run finalize after the loop ends.
# `DEFAULT_WALL_TIME_S` is the derived total — kept for the check-run
# summary + finish-line callers that report a single "wall budget".
DEFAULT_WALL_TIME_S = DEFAULT_T0_WALL_TIME_S + DEFAULT_T1_WALL_TIME_S + POST_HEADROOM_S

# ── Tier escalation (T0 → T1 continuation, T2 second opinion) ──────
# Engine defaults for the escalation ladder; a deployment's workflow
# overrides them via env (`AGENT_REVIEW_T1_CONTINUATION=true`, the
# `AGENT_REVIEW_T2_MODEL` repo variable, the Mode step's
# `AGENT_REVIEW_SKIP_T0` output, …) which `ReviewerConfig.from_env`
# folds into the mirrored fields.
#
# T0→T1 wall-hit continuation gate. Off in-code so a bare engine run is
# single-tier; a deployment's workflow turns it on.
DEFAULT_T1_CONTINUATION = False
# Which triggers escalate T0→T1 when the ladder has a T1 rung. Must be a
# subset of `cora.escalation.ESCALATE_TRIGGERS` ({"wall_hit", "blocker",
# "low_confidence"}). `wall_hit` alone is the continuation behaviour; add
# `blocker` / `low_confidence` to have a stronger model double-check a
# needs-changes or no-verdict outcome. Mirrored as
# `ReviewerConfig.escalation_triggers` (CSV env `CORA_ESCALATION_TRIGGERS`).
DEFAULT_ESCALATION_TRIGGERS: frozenset[str] = frozenset({"wall_hit"})
# LiteLLM alias the T1 continuation dispatches to (the larger-context
# T1 endpoint).
DEFAULT_T1_MODEL = "core"
# T1 per-run iteration cap — matches `continue_on_t1`'s design default.
DEFAULT_T1_MAX_ITERATIONS = 6
# T2 second-opinion (disagreement-resolver) soak gate. Off by default.
DEFAULT_T2_DISAGREEMENT = False
# LiteLLM alias for the T2 alt-reviewer endpoint.
DEFAULT_T2_MODEL = "alt-reviewer"
DEFAULT_T2_MAX_ITERATIONS = 8
# Classifier-large-diff entry: bypass T0 entirely and start
# the review on T1's bigger-context endpoint.
DEFAULT_SKIP_T0 = False
# Mandatory T2 escalation on `propose_patch` verdicts — default-on kill
# switch (`AGENT_REVIEW_PATCH_ESCALATION="false"` is the emergency
# disable; only the literal "false" turns it off, typo-safe).
DEFAULT_PATCH_ESCALATION = True
# Actual `propose_patch` writes are opt-in for adopters. When false, the
# orchestrator strips any directive the model emitted from the posted
# comment and logs that dispatch was disabled. This keeps the default
# reviewer comment/check-run-only while private deployments can enable
# inline suggestions / draft PRs explicitly.
DEFAULT_PROPOSE_PATCH_DISPATCH = False

# Skip-trivial classifier label. A label-classifier step
# exports its label via `CLASSIFIER_LABEL`; when it lands in
# `RETRIEVAL_SKIP_LABELS` the retrieval pipeline is bypassed. Empty =
# no classifier verdict for this run.
DEFAULT_CLASSIFIER_LABEL = ""

# Push-based context injection (see `context_refresher.py`) — master +
# per-source killswitches, all default-on. Like the patch-escalation
# kill switch, only the literal "false" env value disables a source.
CONTEXT_INJECTION_ENABLED = True
CONTEXT_INJECTION_CI = True
CONTEXT_INJECTION_HEAD = True
CONTEXT_INJECTION_COMMENTS = True

# Tool-use trajectory capture `source` tag (`REVIEWER_TRANSCRIPT_SOURCE`)
# — labels rows in the captured JSONL for teacher-data provenance.
DEFAULT_TRANSCRIPT_SOURCE = "trajectory-live"

# PR-bundle caps. Tightened for a small-context T0 backend
# (e.g. 48K max-model-len). At the previous sizes, big PRs accumulated
# >100K tokens by ~13 tool calls and fell back from the T0 endpoint to
# the larger T1 endpoint, defeating the decoupling for the heavy case.
#
# Worst-case initial-prompt math at these caps:
#   - system prompt (agent_review_prompt.md ~9K chars):     ~2.3K tokens
#   - diff:                                            ~8K tokens (32K chars)
#   - CLAUDE.md:                                       ~4.2K tokens (cap clears actual file size)
#   - PR body:                                         ~0.75K tokens (3K chars)
#   - framing/headers:                                 ~0.5K tokens
#   - Total initial:                                  ~15.75K tokens
#
# Combined with result_char_cap=8K override below + 12-iteration
# ceiling, last-call worst case: ~15.75K + 12×2K = ~40K per-call
# prompt with ~8K headroom for output in a 48K T0 window.
#
# Bumped 24K → 32K to cover more of medium-sized PRs without forcing
# grep_repo/git_show to fill in the missing diff (the 12-iteration
# bump already gave headroom for fetch-on-demand; this bump reduces
# the need). Multi-file PRs touching dense code (e.g. paired
# multi-hundred-line script rewrites) still get truncated, but more
# of the diff fits in the first pass.
DIFF_CHAR_CAP = 32_000
# 17K rather than 16K because a real-world CLAUDE.md ran to ~16.7K
# bytes — truncating at 16K would chop ~600 chars off the tail. 17K
# leaves small future-growth margin without changing the token-cost
# picture (~50 tokens diff vs 16K).
CLAUDE_MD_CHAR_CAP = 17_000
PR_BODY_CHAR_CAP = 3_000
# Failing-CI-checks section folded into the prompt by gather_ci_context.
# CAP is the whole section; LOG_TAIL is how much of each failing job's
# log tail to quote (errors cluster at the end of a job log).
CI_CONTEXT_CHAR_CAP = 9_000
CI_LOG_TAIL_CHARS = 2_400

# Whitelist of MCP-server tools exposed to the agent for PR review.
# Live-infrastructure tools (kubectl_*, loki_query, prometheus_query,
# pods_top, nodes_top) are excluded — reviewing a static diff doesn't
# need live state. The triage agent gets a different set.
READ_TOOLS = {
    "search_knowledge",
    "read_decision",
    "read_note",
    "read_agents_section",
    "list_decisions",
    "list_notes",
    # git_log stays on the MCP server — it answers "what's the recent
    # history of `main`", which is exactly what its mirror holds.
    "git_log",
    # Semantic search over a docs corpus
    # (upstream vendor release notes). Served by the MCP server, same
    # session as the other read tools.
    "search_cluster_docs",
}

# Repo-inspection tools served IN-PROCESS from the CI runner's local
# PR checkout instead of the MCP server, whose grep_repo / git_show
# read a `main`-branch mirror, so they go PR-blind: a file the PR adds
# doesn't exist on `main`, and the reviewer wrongly flagged it missing
# (an observed false 🔴). The CI runner already has the PR merge ref
# checked out at REPO_ROOT — grep/show that and the answers reflect the
# code actually under review. Registered as agent-loop local tools; they
# shadow the MCP server's same-named copies.
LOCAL_REPO_TOOLS = {"grep_repo", "git_show"}

# Observe-only write-intent tools served by
# the actions MCP server (a second MCP session, deep mode only). Calling
# these does NOT mutate anything; the server records intent +
# policy outcome as telemetry. The reviewer prompt explains this.
ACTION_TOOLS = {
    "deployment_restart",
    "pod_delete",
    "node_cordon",
    "node_uncordon",
    "event_annotate",
}

# Live web fetch served by a web-fetch gate (a third
# MCP session, deep mode only). Fetches an external doc page through
# the fetch → sanitize → prompt-injection-classify → wrap pipeline.
WEB_TOOLS = {
    "web_fetch_doc",
}

# Union exposed to the agent loop — the loop routes each name to its
# local handler (LOCAL_REPO_TOOLS) or whichever MCP session registered
# it (everything else).
ALLOWED_TOOLS = READ_TOOLS | ACTION_TOOLS | WEB_TOOLS | LOCAL_REPO_TOOLS

# Grafana dashboard host for the per-PR drilldown link the reporter
# embeds in comments / check-runs. Set per deployment via `GRAFANA_BASE_URL`.
# Empty by default — no host leaks into the public package, and the
# drilldown helper (`check_run._grafana_drilldown_url`) returns "" so the
# reporter simply omits the grafana link. A deployment that runs Grafana
# supplies its host via `GRAFANA_BASE_URL` in the reviewer workflow to
# get comment/check-run deeplinks.
GRAFANA_BASE = os.environ.get("GRAFANA_BASE_URL", "").rstrip("/")
# Per-PR detail dashboard path appended to `GRAFANA_BASE` for the
# drilldown deeplink. Set per deployment via `GRAFANA_DASHBOARD_PATH`
# (e.g. "/d/pr-review-detail"). Empty by default — with no path (or no
# host) the reporter simply omits the drilldown link.
DASHBOARD_PATH = os.environ.get("GRAFANA_DASHBOARD_PATH", "")
