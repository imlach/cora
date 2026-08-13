# Changelog

All notable changes to **cora** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward. Pre-release builds carry `git describe`-derived
versions (e.g. `0.0.0.dev60+g257737d`).

## [Unreleased]

### Added
- **A capped T1 now writes its review instead of dying** (#62). Nothing
  ever reserved a turn for the final answer: `UsageLimits(request_limit=N)`
  raises *before* dispatching request N+1, so a model that spent response
  N on tool calls ended the run with no text at all. Across the tier
  boundary that was survivable — a capped T0 escalates and the
  continuation prompt is the elicitation that makes T1 finish — but T1
  has no tier after it, so its own cap-trip killed the review as
  `no review produced`. T1 now spends one bounded `tool_choice="none"`
  turn turning the trajectory it already built into a verdict. Scoped to
  `max_iterations`: a `wall_time` or `per_call_timeout` exit means the
  clock ran out, and another model call is exactly what that path cannot
  afford. Skipped when `require_initial_tool_call` is armed and
  unsatisfied — with no grounding there is nothing verified to write up.
  Best-effort throughout: a backend that ignores `tool_choice` leaves the
  run where it already was, never worse.
- A machine-readable `external_id` on the verdict check run
  (`cora:no-body:<reason>`, `cora:skip:<reason>`, `cora:verdict:<reason>`).
  `conclusion` cannot separate "the reviewer found blockers" from "the
  reviewer produced nothing" — both are `failure` — which left consumers
  string-matching the output title. Purely additive: paths that don't set
  a tag omit the field, and no conclusion changed.

### Fixed
- **Token usage is recorded on every loop exit, not just the clean one**
  (#62). The accounting sat in the success tail of `deep_review_call` /
  `continue_on_t1`, past the `early_terminated_reason` return, so a
  cap-trip, wall-time, per-call-timeout or errored exit reported
  `in_tokens=0 out_tokens=0 resolved_model=unknown` beside a plainly
  populated `tools={...}` — the calls happened and the tokens were spent,
  they were just never counted. Every exit now accounts, and the spiral
  re-draw and recovery turns account themselves. Expect previously-zero
  finish lines to carry real numbers, and T1-rescued reviews to report
  higher totals than before: they were reporting T1 alone, because T0's
  usage was dropped by its own early return.
- A failed T1 on a **fresh** entry no longer reports the synthetic entry
  marker as its terminal reason (#62). `classifier_large_start` is what
  `_tiers.py` writes to *select* the skip-T0 path — it never described a
  termination — so reporting it masked the real, retryable
  `max_iterations` from any wrapper that retries on typed reasons. Resume
  entries are unchanged: they still preserve T0's genuine wall-hit, and a
  failed grounding contract still wins on either entry.

## [0.1.10] - 2026-08-11

### Added
- **Recoverable tool errors are counted and logged.** `tool_error` fires
  per failed tool result or framework retry prompt (with the tool name,
  the kind, the running count and a truncated message), and
  `tool_recovered` fires on the first success after them — the pair
  separates a model that self-corrected from one that never got a tool to
  work. Totals join the deep finish-line summary and the `wall_hit`
  break marker as `tool_errors`.
- A `::warning::` when a T1 escalation ran on the same engine as T0 — either
  the same alias, or two aliases the gateway resolves to one served model.
  Every escalation entry assumes the second endpoint fails differently; when it
  does not, the escalation is a no-op and used to fail silently.

### Fixed
- **A second bad call to the same MCP tool no longer aborts the review**
  (#58). Pydantic-AI's per-tool retry budget is cumulative across a run
  and only clears when that tool succeeds, so with `retries=1` an early
  recoverable error the model routed around by reaching for a *different*
  tool left the budget spent — and an unrelated bad argument to the first
  tool, many turns later, raised
  `UnexpectedModelBehavior("Tool 'read_note' exceeded max retries count
  of 1")` and killed the whole loop as `agent-loop-errored`. MCP toolsets
  now carry `tool_error_behavior='failed'`: the server's error text comes
  back as a tool result the model reads and corrects from, spending no
  budget. The request, iteration and wall limits are unchanged and remain
  the only hard bound; a protocol-level `McpError` still routes through
  the retry path, and an unreachable server still fails the review with
  `mcp-connect-failed`. **Requires pydantic-ai 2.16+** (the floor moved
  with it) — on an older build the factory degrades to the previous
  behaviour rather than passing a value that build mishandles.
- The `agent-loop-errored` skip path no longer flattens the reason. The
  check-run and `ReviewResult.terminated_reason` carried the bare marker while
  the specific cause (e.g. `spiral-redraw-exhausted`) survived only in the PR
  skip comment, so a soft-failed review looked causeless. Both now carry the
  full reason, matching the required-tool path. Retry markers still match —
  consumers test the prefix.

## [0.1.9] - 2026-08-09

### Added
- **Enforceable deep-review grounding contract** (#51). Opt in with
  `CORA_REQUIRE_INITIAL_TOOL_CALL=true` to constrain each model request with
  `tool_choice="required"` (and serial tool execution) until the trajectory
  contains a successful tool return. A provider that ignores the T0 constraint
  gets one fresh T1 retry under the same contract; if T1 also ignores it, cora
  discards the unverified body and posts a cancelled check-run. Structured
  `initial_tool_contract` events expose satisfied, ignored, and no-tools
  outcomes. Quick mode and default-off deployments are unchanged.

### Fixed
- Fresh T1 entries now actually discard T0 message history. The `no_tool_use`
  retry was labelled and prompted as fresh but still carried the unverified T0
  trajectory, anchoring the stronger tier on the claims it was meant to check.

## [0.1.8] - 2026-08-09

### Fixed
- **`no_tool_use` escalation died at the finish line.** The 0.1.7
  trigger's first live firing escalated correctly, T1 produced a
  verdict, and then `_T1_SUCCESS_REASON[tag]` raised `KeyError:
  'no_tool_use'` — cancelling the review. The connector now maps the
  tag (`t1-no-tool-use-retry`), logs its own entry line, and a
  regression test pins every driver entry tag to a mapped reason.

## [0.1.7] - 2026-08-09

### Added
- **`no_tool_use` escalation trigger.** A deep review can verdict without
  making a single tool call — observed on the same PR, same prompt, same
  model drawing a 0-call review that *asserted* verification and a 35-call
  review that performed it. A 0-call deep verdict is unverified by
  construction, so the new trigger escalates it to T1 with a **fresh**
  entry (resuming the trajectory would anchor the stronger tier on the
  unverified claims — same framing as the per-call fresh-start path).
  Quick mode is exempt: it runs without tools by design. Opt in via
  `CORA_ESCALATION_TRIGGERS=wall_hit,no_tool_use`; off by default.

## [0.1.6] - 2026-08-09

### Added
- **`list_files` — a path-existence tool for deep mode** (#36).
  `grep_repo` matches file *content*; its `glob` only narrows which
  files get searched. Nothing in the palette could answer "does
  `<path>` exist?", so a model checking for a file content-grepped it
  and read zero matches as proof of absence — landing false
  `🚨 Blocker` findings that a fixture "must be added" while the file
  was tracked and its tests were green on the same SHA. `list_files`
  returns repo-relative paths at the PR's state, with the same
  skip-dirs / skip-extensions / binary rules as `grep_repo` so the two
  agree on what counts as in the repo. Output is capped at 1000 paths
  and says so when it truncates; a glob matching nothing returns a note
  spelling out what that does and does not prove. `deep.md` now maps
  each question to its tool and states plainly that a "file is missing"
  finding needs path-level evidence. On by default; drop the name from
  `ReviewerConfig.local_repo_tools` for the previous palette.
  `GitProvider.list_files`
  is concrete, not abstract, so existing third-party providers keep
  working — they inherit a fallback that reports the missing capability
  rather than implying the file is absent.
- **Re-reviews can see maintainer rebuttals** (#37, part 2). A review
  had no view of anything a human had said on the PR: `pr_context`
  listed comments only to find the classifier's own marked one. So a
  maintainer who refuted a false finding *with log evidence* changed
  nothing, and the re-review re-asserted the identical claim — a
  deadlock only a human override broke. The initial prompt now carries
  a "Discussion on this PR" section with the most recent maintainer
  comments (`pr_context.fetch_thread_evidence`), instructing the model
  to weigh a rebuttal's evidence and drop or downgrade the finding
  rather than repeat it. Excluded: cora's own comments (anchoring the
  reviewer on the verdict it is meant to re-examine), the classifier's
  comment (already rendered separately), bots, and anyone outside
  `thread_evidence_associations` (`OWNER`/`MEMBER`/`COLLABORATOR` —
  the same standing bar `trigger.DEFAULT_ALLOWED_ASSOCIATIONS` uses,
  since on a public repo anyone can comment). The association filter
  narrows *whose* words are read, not whether they are instructions:
  the block is wrapped `<untrusted-content>` regardless, and the
  section text is explicit that a comment telling the reviewer what
  verdict to reach is something to report, never obey. Bounded at 6
  comments / 1.5K chars each / 6K total, all soft-fail to the previous
  prompt shape. Default on; `AGENT_REVIEW_THREAD_EVIDENCE=false`
  disables.

- **Retraction-verdict gate** (`cora.review._retraction_gate`, #38).
  #32 taught `detect_blocker` to discount a `🚨 Blocker` bullet the
  model withdraws in its own text, so automerge stopped pausing over
  nothing — but the verdict line was left alone, so the same review
  still posted 🔴 `needs changes` and the required check went red over
  that same nothing. When *every* Blocker bullet in a block-severity
  review retracts, the verdict now drops one step to 🟡 `minor` with a
  harness note. Whole-review rule, mirroring the CI-verdict gate: one
  live blocker among withdrawn ones changes nothing, and findings are
  annotated rather than deleted so a human still reads what the model
  wrote. Runs before the check-run posts (`complete_check` is
  first-write-wins) and after the CI gate, so a review that gate
  already downgraded is a no-op here rather than a second step down.
  Default on; `AGENT_REVIEW_RETRACTION_VERDICT_GATE=false` disables.
  **Known limit, deliberate**: the gate only sees retractions
  `_BLOCKER_RETRACTION_RE` recognises, and that pattern biases toward
  under-matching because a false positive silently drops a live
  blocker. #38's own examples ("This logic appears sound", "so this
  case is unreachable. Good.") do not match it and still post — a
  pinned test documents that. The durable fix is upstream, in what the
  model emits; #38 stays open for it.
### Changed
- **ruff runs in CI, pinned, against an explicit rule set.** It was in
  the dev extra unpinned and never run by CI, so 251 findings had
  accumulated against a lint nobody could reproduce — ruff's *default*
  rule set widens between releases, so two machines on different
  versions disagreed about what was even being checked. Now: `ruff`
  pinned exactly, an explicit `select` (`E F W I UP B C4 PIE SIM RUF
  TRY BLE`), and a documented `ignore` for the five rules that fight
  deliberate house style (`E501`, `TRY003`, `RUF001`-`003`) plus three
  that would change behaviour or add churn (`SIM105`, `TRY004`,
  `TRY300`). The tree is clean at zero findings; a `ruff` job runs
  alongside `pytest` on every PR. The sweep itself is mechanical —
  import sorting, PEP-604 annotations, `zip(..., strict=False)` (chosen
  over `strict=True` precisely because it preserves today's behaviour)
  — plus nine hand fixes, including a real one: a `StreamStallDetected`
  raised inside `except TimeoutError` now chains its cause, so the
  traceback says where the wait expired.
- **Prompts: look it up, don't remember it** (#44, both modes). The old
  rule capped unverifiable third-party claims at ⚠️ **Concern**, which
  the model satisfied by shipping the same wrong claim one severity
  lower with "recommend confirming" attached — handing verification back
  to the author, which "Verify before you flag" already forbids. The
  rule is now a positive default with an explicit evidence order (green
  CI for this SHA → pinned dependency source → lockfiles and manifests →
  a fetch tool, if the deployment has one), and an explicit floor: if
  none of those can settle it, **drop the finding** rather than lowering
  its severity. A ⚠️ or ℹ️ is for something you did establish and judged
  minor, never for something you didn't establish.
- **Prompts: a version's non-existence is never a Blocker** (#37, both
  modes). The reviewer blocked a PR with "`node-version: 26` is
  invalid; Node 26 does not exist (latest LTS is 22)" while
  `actions/setup-node` had already resolved and installed 26.7.0 in
  that PR's own job log — then re-asserted it verbatim on re-review.
  A release made after the training cutoff is *absent from the weights
  by construction*, so a version number ahead of expectation is the
  expected appearance of a real release, not evidence of a typo. The
  existing "library claim is not verification" and "version boundaries"
  rules cover how a pinned version *behaves*; neither covers whether it
  *exists*. Both prompts now forbid asserting non-existence outright —
  confirm from evidence in this review (lockfile, manifest, resolved CI
  output, fetched release page) or ask the author at ⚠️ **Concern** at
  most, never 🚨 **Blocker**. "The latest LTS is X" is called out as
  the same unknowable claim.
- **Prompts: a finding is a conclusion, not an investigation** (#38,
  deep mode). A review posted four `🚨 Blocker` findings whose bodies
  each reasoned their way to "actually this is fine" / "this logic
  appears sound", red-blocking the PR on zero actionable content. The
  existing hard constraint bans planning *phrases* ("Let me check…")
  but says nothing about a finding whose own analysis concludes there
  is no defect. `deep.md` now says it outright: work out whether
  something is a defect before writing the bullet, and if you reason
  your way to "no problem", delete the bullet rather than posting the
  walk-through that ends in its own refutation.
- **Per-run PR comments, collapse instead of edit-forever** (#29). Every
  review used to find-or-edit the same one comment on a PR, so review
  N+1's PATCH silently overwrote review N's verdict with no trace it
  had changed — a retracted finding and a live one were
  indistinguishable once the comment was reused. Each review run now
  owns exactly one comment (`<!-- cora:progress:<run_id> -->` while
  in-flight, PATCHed in place, then swapped to `<!-- cora:verdict:<run_id> -->`
  for the final verdict), keyed on `GITHUB_RUN_ID` so a cancelled run's
  leftover placeholder is never mistaken for a later run's. The
  run-scoped marker sits on the line *below* `COMMENT_MARKER`, which
  keeps position 0 — deployments that match cora comments with
  `startswith(COMMENT_MARKER)` (an automerge watchdog, a corpus miner)
  keep working unchanged. Once that
  run's own comment is live, every *other* cora comment on the PR
  (older verdicts, orphaned placeholders, and pre-migration
  single-comment-loop bodies still carrying only a `LEGACY_COMMENT_MARKERS`
  / `COMMENT_MARKER` marker) is collapsed via the GraphQL
  `minimizeComment` mutation (classifier `OUTDATED`) — REST has no
  equivalent. The collapse runs LAST, after the new comment is
  confirmed live, so a cancelled or failed run never hides the last
  good review; it is also purely cosmetic and fully soft-fail (a
  `::warning::` and move on for a non-zero rc, an already-minimized
  comment, or the sneakier case of a GraphQL `errors` array on an
  HTTP-200 response), so a collapse failure can never lose a review.
  A pre-migration PR's legacy comment is minimized on its next review
  rather than adopted, so the new loop starts clean instead of
  inheriting the old edit history it exists to escape.
  **Trade-off, left for a follow-up**: every review run now posts (and
  notifies watchers of) a new comment, even when a PR is pushed to
  repeatedly at the same head SHA in quick succession — noise that the
  old single-edited-comment behaviour didn't have. A head-SHA-keyed
  variant (new comment only when the SHA changes, still edit-in-place
  for reruns on the same SHA) would fix that; not implemented here.
  `post_or_edit_comment` is gone — replaced by `create_progress_comment`
  / `update_run_comment` / `minimize_superseded_comments` in
  `cora.core.comment`.
  **Adopter migration note**: `minimizeComment` bumps the collapsed
  comment's REST `updated_at`, and the collapse deliberately runs after
  the new comment is live — so the freshest `updated_at` on a reviewed
  PR is usually a *superseded* comment. Tooling that finds "the current
  cora comment" by `updated_at` recency (e.g. a
  `max_by(.updated_at)`-style watchdog) must switch to `created_at` or
  to the run-scoped verdict marker before adopting this release;
  `created_at` selection is also correct against the old
  single-edited-comment engine, so it can ship first.

### Fixed
- **The reviewer can see passing CI, so world-knowledge claims have
  counter-evidence** (#44). Observed downstream: a deep review made
  **zero tool calls** and posted three findings — "`go 1.26.5` is not a
  valid Go version … this will cause `go build` to fail", a claim about
  a function it never read, and "if these options don't exist,
  compilation will fail. Recommend confirming". The build check for that
  SHA was green, so all three were already refuted. Two
  individually-correct behaviours composed into the blind spot:
  `gather_ci_context` returned `None` whenever nothing was failing, and
  `deep.md` rightly forbids inferring success from silence — so the
  greener the PR, the less the reviewer knew. The CI block now also
  lists **passing** check names for the head SHA (names + conclusions
  only; a green job's log is noise), on both the all-green and mixed
  paths, and states that a green check settles compile / test /
  API-existence / version-existence claims at **any** severity. The
  silence asymmetry is preserved: no CI section still means unknown,
  never "it passed", and a SHA with no reported checks still yields
  `None`. Default on; `AGENT_REVIEW_CI_CONTEXT_PASSING=false` disables.
- **A crashed review no longer leaves its PR comment reading "in
  progress" forever** (#14). Every *handled* terminal path finalizes
  the "🔄 Reviewing PR … this comment will update with the verdict"
  placeholder, but an exception escaping the pipeline unwound straight
  to `python -m cora`'s hard-failure boundary, which prints and exits
  1 — leaving a comment indistinguishable from a run still in flight,
  and costing real time diagnosing whether the run was alive. The
  progress check-run had the same hole (the SIGTERM guard covers a
  kill, not a crash). Both are now finalized on the way out: the
  comment becomes "review errored (`<ExcType>`) before producing a
  verdict — re-push to retry" and the check concludes `cancelled`
  (in the merge gate's tolerated set; the crash is already loud via
  the nonzero exit). Both writes are best-effort and warn rather than
  raise — the original traceback still propagates unchanged. The
  comment write is guarded on the new `Reporter.progress_open`, so a
  run that never posted a placeholder (quick mode, early-exit skips)
  does not get a crash comment invented for it. `progress_open` is
  concrete on the ABC and defaults False, so existing third-party
  `Reporter` implementations keep working.
- **`detect_blocker` no longer counts a retracted `🚨 Blocker` bullet.**
  Observed live on imlach/cora#25: a review posted a Blocker bullet reading
  *"This is a false alarm from the truncated diff display — the code is
  fine"*, then stated *"I have no actual blockers"* — the marker text alone
  still paused automerge and counted toward `tier_verdict` telemetry (#29's
  "Related" item). `detect_blocker` now extracts each Blocker bullet's own
  text (including wrapped continuation lines) and discounts it only when
  that bullet itself contains a narrow, unambiguous retraction phrase
  ("false alarm", "not actually a bug", "the code is fine", …) that no
  contrastive clause walks back — a hedge ("might be a problem"), a
  narrowing ("the surrounding code is fine, *but* this path crashes"), or
  a reassuring sentence *elsewhere* in the body never discounts a bullet.
  A body where every Blocker bullet retracts
  still logs the discount (`::notice::detect_blocker
  retracted-bullets-discounted`) rather than silently dropping it. The
  verdict-word check (a literal `needs changes` marker) is untouched —
  that's the model's own stated conclusion, not a bullet to reason about.

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

[Unreleased]: https://github.com/imlach/cora/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/imlach/cora/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/imlach/cora/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/imlach/cora/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/imlach/cora/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/imlach/cora/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/imlach/cora/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/imlach/cora/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/imlach/cora/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imlach/cora/releases/tag/v0.1.0
