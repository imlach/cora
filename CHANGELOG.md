# Changelog

All notable changes to **cora** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward. Pre-release builds carry `git describe`-derived
versions (e.g. `0.0.0.dev60+g257737d`).

## [Unreleased]

## [0.1.5] - 2026-08-06

### Added
- **Generic extra MCP sessions** (`MCP_SERVERS`, a JSON array of
  `{"name", "url", "token_env", "required"}` objects) — a deployment
  wiring a fourth (fifth, …) MCP server no longer needs a new named env
  var. `token_env` is env-var indirection (the JSON never carries a
  literal token); a `token_env` that doesn't resolve connects tokenless
  with a warning rather than failing config parse. Internally, the three
  named slots (`MCP_URL`/`MCP_TOKEN`, `MCP_ACTIONS_URL`/`MCP_ACTIONS_TOKEN`,
  `WEB_FETCH_GATE_URL`) now normalize into the SAME session list as
  `MCP_SERVERS` entries — every dispatch site (`deep_review_call`,
  `continue_on_t1`, `call_t2_alt_reviewer`) probes/opens through one
  shared loop (`cora.core.mcp_sessions`) instead of three duplicated
  branches, preserving the required-vs-optional fail-soft distinction
  exactly. `AGENT_REVIEW_EXTRA_TOOLS` (CSV) admits an extra session's
  tool names through the MCP allow-set filter — local tool names still
  win any collision. The release-notes pre-fetch resolves its endpoint
  as explicit `WEB_FETCH_GATE_URL`, else the first `MCP_SERVERS` entry
  named `"web-fetch"` — a name convention, since the prefetch runs
  before any session opens. The initial prompt's "pull the upstream
  facts with `web_fetch_doc`" line — previously shown unconditionally,
  even with no fetch session configured — now only appears (in generic
  "a fetch tool for upstream docs" wording, not a hardcoded tool name)
  when one actually is.
- **A CI-verdict gate for false compile/test-failure blockers** (#23).
  A deep review that runs concurrently with the build can finish first
  and post 🚨 Blocker findings claiming a compile or test failure that
  CI itself contradicts a few minutes later — the `needs changes`
  verdict then rests entirely on claims CI has already disproved. Two
  cooperating mechanisms:
  - **Green-delta context injection.** The push-based context refresher
    (`context_refresher.py`) already injects fresh CI context on a new
    *failure*; it now also injects when a check-run transitions to
    *success*, especially one that was pending or missing on the start
    snapshot. The injected text names the check and tells the model to
    re-verify or downgrade any compile/test-failure claim it made —
    this is the primary fix, since the model self-corrects with the
    signal in context. Killswitch
    `AGENT_REVIEW_CONTEXT_INJECTION_CI_GREEN` (default-true, shares the
    CI source's cadence; the master `AGENT_REVIEW_CONTEXT_INJECTION_CI`
    toggle disables both).
  - **Finalize-time CI-verdict gate** (`cora.review._ci_gate`) — the
    backstop for a review that finishes before CI does. On a settled
    `needs changes` verdict, one bounded re-poll of check-runs for the
    reviewed HEAD SHA (never a different SHA — a green run for an older
    commit proves nothing here); if every relevant check is green,
    blocker findings matching a narrow, documented compile/test-failure
    claim pattern get a visible harness note appended, and — only when
    EVERY blocker in the review matches — the verdict downgrades one
    step (`needs changes` → `minor`) with an explanatory line. Findings
    are always annotated, never deleted. The gate runs before the
    verdict check-run posts, so a fully-contradicted review reports a
    non-blocking check conclusion instead of a red one; the
    automerge pause deliberately still fires (the blocker markers stay
    in the body) so a human reads the annotated findings before merge.
    Soft-fails on any API error (posts the review unchanged);
    killswitch `AGENT_REVIEW_CI_VERDICT_GATE` (default-true).
- **`grep_repo` gained a second, explicitly-labelled corpus:
  `corpus="deps"`** (#23). Library-API claims ("this function takes
  three args") were previously asserted from the model's training-data
  memory, because the only corpus `grep_repo` searched was the PR
  checkout — the actual pinned dependency source (a Go module cache, a
  vendor dir, `node_modules`, a `site-packages` tree) was structurally
  unreachable, and a stale-memory claim landed as a false 🚨 Blocker.
  `DEP_SOURCE_ROOTS` (CSV of absolute paths a deployment's CI runner has
  already materialized) configures the new corpus; empty/unset
  self-disarms unless in-repo `vendor/`/`node_modules/` are
  auto-detected under the checkout root. Results are labelled by which
  root matched and carry `"corpus": "deps"`, so provenance is never
  ambiguous with a repo-corpus result. Bounded by a new
  `DEP_SOURCE_MAX_FILES_SCANNED` walk cap (default 50 000) independent
  of the existing match-count cap, since a dependency tree can run into
  the hundreds of MB where a repo checkout does not; hitting either cap
  truncates with an explicit note. `corpus="repo"` (the default) is
  unchanged. See `docs/configuration.md` for the full knob table.
- **Linked-issue context.** A PR's title/body is parsed for same-repo
  issue references — closing keywords (`fixes #12`, `closes owner/repo#12`,
  …) and bare `#N` mentions — and up to 2 of them are fetched server-side
  (title/state/body/earliest comments, bounded and capped) and injected
  into the initial prompt as a trust-wrapped `<untrusted-content>` block,
  the same "fetch it server-side, don't leave it to the model" precedent
  `prefetch.py`'s release-notes pull uses. A `read_issue(number)` tool
  (`core/issue_context.py`, deep mode only, same-repo only) covers issues
  the pre-fetch's 2-issue cap or reference parsing misses. Both paths
  share the fetch/bound/wrap code. New env kill switch
  `AGENT_REVIEW_ISSUE_PREFETCH` (default on); the tool is toggled via
  `ReviewerConfig.local_issue_tools`. Skipped entirely for bot-authored
  PRs, same as retrieval and CLAUDE.md.
- **Per-result char cap on the in-process repo tools**
  (`TOOL_RESULT_CHAR_CAP`, env `AGENT_REVIEW_TOOL_RESULT_CHAR_CAP`,
  default 16 000 chars). `git_show` file content is head+tail truncated
  with an explicit marker; `grep_repo` stops accumulating matches at the
  same budget and says so in a `note`. Before this, a single whole-file
  `git_show` on a large repo doc injected the entire file into one turn
  (observed +45K tokens from one 155 KB read), saturating a small T0
  context window and tripping the reasoning spiral — the
  `result_char_cap` the config's budget math referenced was never
  actually implemented.
- **Duplicate-call guard on the local tools.** A byte-identical
  `(tool, args)` repeat within one review returns a short stub pointing
  at the earlier result instead of re-injecting it — a looping model
  (same call pair re-issued on alternating turns) now pays for the
  result once.
- **World-knowledge category in "Verify before you flag" (deep mode).**
  A claim about a third-party library's API shape or version-dependent
  behaviour now counts as unverified unless confirmed *this review* from
  dependency source, fetched docs, or CI for the reviewed SHA — memory
  of the library doesn't count, and such claims are capped at ⚠️,
  phrased as a question, never Blocker-eligible. Named the trap
  explicitly: recall is worst at major-version boundaries, and citing
  the pinned/lockfile version isn't verification. Quick mode gets a
  one-line parallel, since it has no tools to verify a library claim at
  all (#23).

## [0.1.4] - 2026-08-05

A review that stalls now recovers instead of dying, and the reviewer
stops mistaking its own tools' blind spots for facts about the PR:
spiralled reasoning escalates to T1, the deep prompt budgets its context
instead of saturating it, and both known PR-blindness traps (directory
globs matching nothing, doc-lookup reading the base-branch index) no
longer produce confident wrong findings.

### Added
- **ADR bodies split one-file-per-decision now resolve** (#13).
  `decision:` payloads resolved only from a monolithic `DECISIONS.md`;
  repos that split the log into a `decisions/` directory
  (`DEC-NNN-<slug>.md` or bare `DEC-NNN.md`) had every lookup silently
  resolve to nothing. Resolution is now directory-first with the
  monolith as fallback, so both shapes — including mid-transition, when
  both exist — work.

### Changed
- **An exhausted reasoning spiral now escalates to T1 instead of
  soft-failing** (#18). When a T0 draw spiralled and the bounded re-draw
  spiralled again, the review ended as a cancelled check-run
  (`agent-loop-errored: spiral-redraw-exhausted`, retry on next push).
  The spiral is a property of the T0 reasoning model, not the PR — the
  same argument the per-call-timeout fresh start already makes — so the
  outcome is now a forced T1 entry: T1 resumes the committed trajectory
  (the tool work T0 banked is kept; only the spiralled draw is dropped)
  with a stalled-reasoning resume framing. A successful T1 body finishes
  as `t1-spiral-escalation`; if T1 also fails, the original soft-fail
  posture returns unchanged, as it does for every other
  `agent-loop-errored` reason. Killswitch
  `AGENT_REVIEW_SPIRAL_ESCALATION=false`.
- **The deep prompt now frames the context window as the tool budget.**
  The validate-any-claim grounding (0.1.2) removed the tool-call cap
  entirely, and on deployments with small-context tier-0 models the
  swing overshot: reviews saturated the context window with bulk
  whole-file reads and re-issued identical calls (the same
  large-`max_chars` doc fetch re-injected on six consecutive turns was
  the observed worst case), dying in context-length errors before a
  verdict landed. The grounding norm is unchanged — unverified claims
  still get dropped — but the prompt now pairs it with lookup
  discipline: targeted globs and small size bounds first, whole-file
  reads only when the hunks aren't enough, never re-issuing a call
  whose result is already in context, and stopping exploration once
  every finding is verified.

### Fixed
- **T2 disagreement tier attribution now covers every T1 entry path.**
  The primary-tier label in the disagreement banner was matched against
  a hand-picked pair of reasons, mislabelling `t1-per-call-retry` /
  `t1-verdict-trigger` (and now `t1-spiral-escalation`) bodies as T0; it
  now uses the shared `T1_TERMINATED_REASONS` set.
- **`grep_repo` directory globs no longer silently match nothing.** The
  glob is fnmatch'd against the full repo-relative path, so a bare
  directory path (`pkg/sub/` or `pkg/sub`) selected zero files and the
  empty result read as "this code doesn't exist" — observed as a reviewer
  wrongly concluding a PR-added directory had no manifests. Directory
  globs now search the directory's subtree, and any glob that selects
  zero files carries an explicit `note` in the envelope so the model can
  tell a mis-aimed glob from a genuine no-match.
- **Doc-lookup tools no longer produce false "missing file" blockers on
  PR-added docs** (#20). `read_note`/`search_knowledge` query the
  deployed docs index, which is built from the base branch — so a PR
  referencing a doc it itself adds was blocked with a spurious 🔴
  "file does not exist". The prompts now state the tools' base-branch
  scope and direct existence checks for PR-referenced files at the PR
  checkout (`git_show`/`grep_repo`), the same PR-blindness rule
  `LOCAL_REPO_TOOLS` already established.

## [0.1.3] - 2026-08-04

Reasoning spirals, detected rather than timed out. A turn that spends
its whole completion budget without committing to a tool call or a
verdict is now a signal the loop acts on, instead of a call the per-call
timeout discards while it is still generating.

### Changed
- **Deep mode no longer requires an MCP server** (#10). `MCP_URL` unset
  (now the default — it was a `http://localhost:8080/mcp` placeholder) is
  self-disarming: no probe, no toolset, and the agent loop runs on the
  in-process `grep_repo`/`git_show` over the PR's own checkout. Deep mode
  was previously unreachable without private infrastructure — the probe
  failed against the placeholder and every deep run soft-skipped with
  "MCP server unreachable", which is why the shipped adopter example
  pins `MAX_TOOL_ITERATIONS=0`. A *configured* server that is unreachable
  still fails the review: silently dropping tools someone asked for is
  the worse failure. The comment footer's tool denominator no longer
  counts MCP-served read tools when no server was attached.
  **Deployments that relied on the localhost default must now set
  `MCP_URL` explicitly.**
- **The deep per-call completion ceiling drops 32K → 18K**, and is now
  env-settable as `AGENT_REVIEW_MAX_COMPLETION_TOKENS`. It is sized
  against `AGENT_REVIEW_PER_CALL_TIMEOUT_S`, not against the context
  window: at observed serving rates a 32K draw cannot finish inside a
  180s cap, so an extended-thinking turn was cancelled mid-generation —
  and a cancelled request records no usage, no TTFT and no
  `finish_reason`, which made the whole affected population invisible in
  every latency histogram. Bounded, the same episode ends as
  `finish_reason=length` data.

  The ceiling is bounded on **both** sides and the two must be derived
  together. Below the longest completion observed to succeed (~16K) it
  truncates real reviews and makes every long turn pay for a re-draw it
  didn't need; above `timeout × generation rate` it is unreachable and
  the original failure returns. On the reference deployment that legal
  window is roughly 16K–19.8K — under 4K wide, which is why a raised
  per-call timeout is not optional generosity. **Do not inherit 18K:
  re-derive it from your own rate and timeout.** Quick mode keeps its
  own 32K ceiling (single-shot: reasoning and the full verdict must fit
  one call).

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
  `AGENT_REVIEW_THINKING_BUDGET_TOKENS`, default 16000, with nothing
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

[Unreleased]: https://github.com/imlach/cora/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/imlach/cora/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/imlach/cora/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/imlach/cora/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/imlach/cora/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imlach/cora/releases/tag/v0.1.0
