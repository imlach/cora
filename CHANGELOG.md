# Changelog

All notable changes to **cora** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward. Pre-release builds carry `git describe`-derived
versions (e.g. `0.0.0.dev60+g257737d`).

## [Unreleased]

## [0.1.3] - 2026-08-04

Reasoning spirals, detected rather than timed out. A turn that spends
its whole completion budget without committing to a tool call or a
verdict is now a signal the loop acts on, instead of a call the per-call
timeout discards while it is still generating.

### Changed
- **The deep per-call completion ceiling drops 32K → 12K**, and is now
  env-settable as `AGENT_REVIEW_MAX_COMPLETION_TOKENS`. It is sized
  against `AGENT_REVIEW_PER_CALL_TIMEOUT_S`, not against the context
  window: at observed serving rates a 32K draw cannot finish inside a
  180s cap, so an extended-thinking turn was cancelled mid-generation —
  and a cancelled request records no usage, no TTFT and no
  `finish_reason`, which made the whole affected population invisible in
  every latency histogram. Bounded, the same episode ends as
  `finish_reason=length` data. **Deployments on a slower backend, or with
  a raised per-call timeout, should re-derive this from their own
  generation rate rather than inherit 12K.** Quick mode keeps its own
  32K ceiling (single-shot: reasoning and the full verdict must fit one
  call).

### Added
- **Uncommitted-draw re-draw** (`AGENT_REVIEW_SPIRAL_REDRAW`,
  default-ON). A turn ending `finish_reason=length` with no tool call
  and no parseable verdict is re-sent ONCE — identical payload, same
  alias, inside the still-open agent context so MCP sessions stay warm.
  Identical deliberately: a payload that spiralled usually completes on
  an immediate re-send, so the cheapest recovery is to ask again rather
  than to ask differently. Detection is on the response boundary, which
  catches both the thinking-only shape (pydantic-ai raises) and the
  truncated-prose shape (it doesn't — the run ends normally and posts a
  verdict-less body). New `event=spiral_redraw outcome=…` line in the
  `agent_review iter` stream.
- **Streaming detection** (`AGENT_REVIEW_STREAM_DETECTION`,
  default-OFF). Consumes tier model calls as delta streams so a stalled
  wire and a thinking model can be told apart while the call is in
  flight — from outside a non-streaming call they are identical, and
  both were recorded as the same `per_call_timeout`. Adds
  `event=stall_detected` (no delta for `AGENT_REVIEW_STALL_TIMEOUT_S`,
  default 30) and `event=spiral_detected` (reasoning deltas past
  `AGENT_REVIEW_THINKING_BUDGET_TOKENS`, default 10000, with nothing
  committed). Aborting cancels the in-flight request and commits
  nothing, so the history stays exactly at the payload the re-draw
  re-sends; a stall that happened mid-answer salvages its visible text
  into the re-draw rather than restarting blind. Degrades to the
  non-streaming path when the run or node doesn't expose the streaming
  surface.
- `AGENT_REVIEW_SPIRAL_DEGRADE_THINKING` (default-OFF): after a payload
  has spiralled twice, one bounded write-up turn with
  `chat_template_kwargs={"enable_thinking": false}` and
  `request_limit=1`, so it cannot make an un-reasoned tool decision. Off
  by default because disabling reasoning on a reasoning model costs real
  review quality and should never happen silently.
- `AGENT_REVIEW_SESSION_HEADER`: sends an opaque per-review value as
  `x-review-session` on every model call — the client half of gateway
  session affinity. No header when unset.

### Fixed
- **The T1 fresh-start entry never fired.** `per_call_fresh_start` gated
  on `not t0_messages`, which cannot be true: pydantic-ai appends the
  outgoing `ModelRequest` to the history *before* awaiting the model, so
  even a first-call timeout leaves a one-element history. It now tests
  what was meant — no `ModelResponse` in the history. A T0 that hangs
  before its first response escalates to T1 instead of ending as
  `skipped (inference backend stalled)`.
- **`agent_review finish` is now emitted on every exit path**, including
  the early skip returns, cancellation (`BaseException`, since
  `CancelledError` is the case that produced the observed silent runs)
  and the SIGTERM guard. A review that logged its turns and then went
  silent was indistinguishable from one still in flight. The skip and
  cancel paths carry zeros for the leak/preamble fields rather than
  omitting them — the field set is a parsing contract.

## [0.1.2] - 2026-08-03

### Changed
- The packaged deep prompt's "Verify before you flag" section now names
  two failure shapes observed in production reviews: a Blocker phrased
  as an unresolved conditional ("if X isn't guarded, this crashes" with
  no tool call to resolve X), and recommending a change the diff already
  implements. Both are pinned by `tests/test_prompts.py`.
- **Tool grounding is the default.** The deep prompt and the assembled
  task framing no longer carry the "≤2 tool calls" nudge (production
  data: 82% of clean deep verdicts were zero-tool-call under it) —
  reviewers are now told to validate ANY claim with a tool call or a
  quoted hunk, with parallel batching (not a call cap) bounding wall
  time. The former `REVIEWER_BROADEN_TOOLS` teacher-trajectory variant
  is this framing, so the flag is now an accepted no-op. Deployments
  that relied on the low-call cost profile should expect more tool
  traffic per deep review.

### Fixed
- `tier_verdict` events now attribute the posted body to T1 for **all**
  T1 entry paths: the hand-picked reason tuple missed
  `t1-verdict-trigger` (blocker / low-confidence escalation) and
  `t1-per-call-retry` (fresh T1 restart), mislabelling those verdicts as
  T0 in the structured log stream. Tier attribution now uses
  `kv_continuation.T1_TERMINATED_REASONS`, which tracks the entry-path
  map by construction.
- Backend attribution now resolves the served-model name from the
  completion response body (`ModelResponse.model_name`) when the gateway
  emits no `x-litellm-*` headers, so the review footer reads
  `endpoint: review (forte)` instead of `endpoint: review (unknown …)`.
  Header-based resolution is still preferred when present (back-compat).

## [0.1.0] - 2026-07-03

First public release. cora's development history predates this
repository going public; it starts at the v0.1.0 cut, with earlier
evolution summarised below and in the module docstrings.

### Added
- Configurable escalation: `ReviewerConfig.escalation_triggers`
  (`CORA_ESCALATION_TRIGGERS`; `wall_hit` default, `blocker` /
  `low_confidence` double-check a needs-changes or no-verdict outcome on
  the next tier under a dedicated second-look framing) and
  `ReviewerConfig.escalation_policy` for a full programmatic ladder
  override (custom tiers, triggers, connector).
- Public setup docs: getting started, configuration reference, and
  architecture overview under `docs/`.
- `QDRANT_COLLECTION` and `GRAFANA_DASHBOARD_PATH` env knobs.
- First-class GitHub Review object support behind the `Reporter` seam,
  opt-in via `use_github_review` (default off).
- Packaging metadata for publication: PyPI classifiers and `[project.urls]`.
- Project docs: this changelog, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- Release workflow (`v*` tag): builds + pushes a multi-arch GHCR container
  image and a wheel/sdist GitHub Release, with build-provenance attestation.
  Distribution is via the container and release artifacts (not PyPI).
- `ReviewerConfig.retrieval_vocab` — the query-reformulation domain
  vocabulary is now configurable (empty by default; deployments supply
  their own terms).

### Removed
- **Breaking:** the LMCache proactive-flush feature (`kv_cache_flush`,
  the `LMCACHE_FLUSH_*` env/config surface, and the `flush_outcome`
  threading). A deployment wanting a cache-warmup side-effect subclasses
  `KvContinuationConnector` and runs it before `super().escalate()`.

### Changed
- **Breaking:** deployment-neutral defaults and identity — the GitHub
  App token env is `CORA_GH_TOKEN` (was `IML_AI_GH_TOKEN`), the bot
  fallback identity and patch-branch prefix are `cora[bot]` / `cora/`,
  the retrieval vocabulary and pretrigger warmup aliases default empty
  (self-disarming), the Qdrant collection defaults to `cora-knowledge`,
  the Grafana drilldown link is omitted unless a dashboard path is
  configured, and no MCP auth header is sent when `mcp_token` is unset.
- `ClusterSecondOpinion` is now `T2SecondOpinion`
  (`cora.core.t2_second_opinion`); the disagreement dissent summary
  labels the primary tier with the configured model alias.
- Quick mode gets its own larger output budget (`QUICK_MAX_OUTPUT_TOKENS`)
  so a reasoning model fits its trace plus the verdict in a single call.
- Default endpoint constants no longer ship internal cluster hostnames;
  they fall back to neutral `localhost` placeholders and are overridden
  per deployment via config/env.
- `requires-python` raised to `>=3.11` (the tested floor); dependency
  pins relaxed to lower-bound floors for library use.

### Changed (distribution)
- Consumers pull released artifacts (GHCR container, Release wheel,
  git tag) — the previous push-model wheel vendoring into the reference
  deployment is retired.

### Fixed
- MCP probe failures no longer log the raw client exception, which could
  embed the `Authorization` header value; header values are redacted and
  the message is length-capped.

[Unreleased]: https://github.com/imlach/cora/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/imlach/cora/releases/tag/v0.1.0
