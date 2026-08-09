"""cora's public configuration surface.

`ReviewerConfig` is the single dataclass an adopter (or the workflow
entrypoint) populates to drive a review. The engine's tunables live as
module-level constants in `cora.core.config`; every field default here
mirrors those constants *by reference* so the two cannot drift.

`ReviewerConfig.from_env()` is the environment-driven wiring for a
CI-workflow deployment — including deliberate parsing quirks
(empty-string alias fallbacks, the typo-safe default-true kill
switches, the `WALL_TIME_S` 40/60 re-derivation) — so a workflow
entrypoint collapses to `run_review(ReviewerConfig.from_env())`: the
defaults *are* the engine constants, and every supported env knob
lands in a field here. The env reads that deliberately stay call-time
(retrieval cache overrides, budget.py's import-time constants, the
CORA_GH_TOKEN subprocess fallback, GHA runtime identity) are
catalogued in `cora.review`'s module docstring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core import config as _c
from cora.core.mcp_sessions import McpServerSpec
from cora.trigger import TriggerPolicy

if TYPE_CHECKING:
    from cora.escalation import EscalationPolicy


@dataclass
class ReviewerConfig:
    """Everything the reviewer needs to run, in one place.

    Required-at-runtime fields (repo, pr_number, llm_*) default to empty
    and are populated by `from_env()` or the caller; everything else
    carries the engine's default.
    """

    # ── Identity ─────────────────────────────────────────────────────
    repo: str = ""                      # "owner/repo"
    pr_number: str = ""

    # ── LLM (OpenAI-compatible) ──────────────────────────────────────
    llm_base_url: str = _c.DEFAULT_LITELLM_BASE
    llm_api_key: str | None = None
    model: str = _c.DEFAULT_MODEL

    # ── MCP attachments (optional; deep mode) ────────────────────────
    mcp_url: str = _c.DEFAULT_MCP_URL
    mcp_token: str | None = None
    mcp_actions_url: str | None = None
    mcp_actions_token: str | None = None
    web_fetch_gate_url: str | None = None
    # Generic extra MCP sessions (env `MCP_SERVERS`, a JSON array) —
    # appended onto the three named slots above; see
    # `cora.core.mcp_sessions.parse_mcp_servers_env` for the schema and
    # `compose_mcp_sessions` for how they're merged with `mcp_url` /
    # `mcp_actions_url` / `web_fetch_gate_url` into one session list every
    # deep-mode dispatch site iterates over. Empty by default.
    mcp_servers: tuple[McpServerSpec, ...] = ()
    # CSV of extra tool names admitted through the MCP allow-set filter
    # (`AGENT_REVIEW_EXTRA_TOOLS`) — extends, never replaces,
    # `read_tools | action_tools | web_tools | local_repo_tools`. An
    # `mcp_servers` session's tools would otherwise be silently dropped by
    # `AgentConfig.mcp_allowed_tools`, since that allow-set is env-only
    # (see the frozenset fields below). Local tool names still win any
    # name collision — unchanged from `agent.py`'s local-tools-first
    # registration order.
    extra_tools: frozenset[str] = frozenset()

    # ── Prompts (None → cora's packaged generic default, loaded by
    #    `cora.core.prompt.load_system_prompt`; set a path to override
    #    with a deployment-specific prompt) ─────────────────────────────
    deep_prompt_path: Path | None = None
    quick_prompt_path: Path | None = None

    # ── Verdict format ───────────────────────────────────────────────
    comment_marker: str = _c.COMMENT_MARKER
    legacy_comment_markers: tuple[str, ...] = _c.LEGACY_COMMENT_MARKERS
    automerge_label: str = _c.AUTOMERGE_LABEL
    # Name of the verdict check-run the reporter posts. A deployment's
    # merge gate can key on this name; rebrand it here (e.g. this
    # repo's own reviews use "cora").
    check_run_name: str = _c.CHECK_RUN_NAME
    # Post the verdict as a first-class GitHub PR Review instead of an
    # issue comment. Default-OFF: the comment + check-run behaviour is
    # the default. A GitHub-reporter
    # capability only — non-GitHub reporters ignore it. The verdict→event
    # mapping lives in `cora.core.leak.verdict_to_review_event` (block →
    # REQUEST_CHANGES, else COMMENT; never a bot APPROVE).
    use_github_review: bool = _c.DEFAULT_USE_GITHUB_REVIEW
    # Consumed by `cora.core.leak`'s parse/compose helpers (and
    # `disagreement.resolve_disagreement`) via their keyword-only
    # `glyphs=` / `words=` parameters.
    verdict_glyphs: tuple[str, str, str] = _c.VERDICT_GLYPHS
    verdict_words: tuple[str, str, str] = _c.VERDICT_WORDS

    # ── Budgets ──────────────────────────────────────────────────────
    max_input_tokens: int = _c.MAX_INPUT_TOKENS
    max_output_tokens: int = _c.MAX_OUTPUT_TOKENS
    # Quick mode's per-call output cap — larger than `max_output_tokens`
    # because the single-shot call must fit reasoning + verdict in one
    # turn (see `core.config.QUICK_MAX_OUTPUT_TOKENS`).
    quick_max_output_tokens: int = _c.QUICK_MAX_OUTPUT_TOKENS
    # Deep mode's per-call cap — each agent-loop turn must fit the model's
    # reasoning trace plus its text/tool-call, and must be reachable inside
    # `per_call_timeout_s` at the deployment's generation rate (see
    # `core.config.DEEP_MAX_OUTPUT_TOKENS`). Used by both the T0 and T1 legs;
    # env knob is `AGENT_REVIEW_MAX_COMPLETION_TOKENS`.
    deep_max_output_tokens: int = _c.DEEP_MAX_OUTPUT_TOKENS
    # ── Uncommitted-draw re-draw ─────────────────────────────────────
    # Re-send the identical payload once when a turn hits the completion
    # ceiling without a tool call or a verdict. Default-ON (killswitch
    # `AGENT_REVIEW_SPIRAL_REDRAW=false`) — see
    # `core.config.SPIRAL_REDRAW_ENABLED`.
    spiral_redraw: bool = _c.SPIRAL_REDRAW_ENABLED
    # ── Exhausted-spiral escalation ──────────────────────────────────
    # Escalate to T1 when the re-draw above also spirals, instead of
    # soft-failing to a cancelled check-run. Default-ON (killswitch
    # `AGENT_REVIEW_SPIRAL_ESCALATION=false`) — see
    # `core.config.SPIRAL_ESCALATION_ENABLED`.
    spiral_escalation: bool = _c.SPIRAL_ESCALATION_ENABLED
    # ── Streaming detection (default-OFF) ────────────────────────────
    # Consume tier model calls as delta streams so a stall and a spiral
    # can be told apart while they happen. See
    # `core.config.STREAM_DETECTION_ENABLED` for why this is opt-in.
    stream_detection: bool = _c.STREAM_DETECTION_ENABLED
    stall_timeout_s: float = _c.STALL_TIMEOUT_S
    thinking_budget_tokens: int = _c.THINKING_BUDGET_TOKENS
    spiral_degrade_thinking: bool = _c.SPIRAL_DEGRADE_THINKING
    # ── Spiral recovery (reasoning-spiral restart) ───────────────────
    # When a reasoning-model turn spends its whole output budget inside
    # `<think>` and emits no body, recover by re-issuing ONE bounded call
    # seeded with the partial-reasoning tail + a "conclude now" directive
    # (reasoning stays ON, just bounded). Default-OFF: the soft-fail
    # behaviour is the default until a deployment opts in via
    # `AGENT_REVIEW_SPIRAL_RECOVERY`. See `cora.core.spiral` +
    # `core.config.SPIRAL_RECOVERY_*`.
    spiral_recovery: bool = _c.SPIRAL_RECOVERY_ENABLED
    spiral_recovery_max_output_tokens: int = _c.SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS
    spiral_recovery_reasoning_char_cap: int = _c.SPIRAL_RECOVERY_REASONING_CHAR_CAP
    max_tool_iterations: int = _c.DEFAULT_MAX_TOOL_ITERATIONS
    t0_wall_time_s: int = _c.DEFAULT_T0_WALL_TIME_S
    t1_wall_time_s: int = _c.DEFAULT_T1_WALL_TIME_S
    post_headroom_s: int = _c.POST_HEADROOM_S
    # Pins `wall_time_s` to an exact total instead of the derived
    # t0 + t1 + headroom sum. Only set by `from_env` when a bare
    # `WALL_TIME_S` ops override is present — the operator's exact
    # value is reported while the tier split is re-derived from it.
    wall_time_override_s: int | None = None

    # ── Tier escalation (T0→T1 continuation, T2 second opinion) ──────
    t1_continuation: bool = _c.DEFAULT_T1_CONTINUATION
    t1_model: str = _c.DEFAULT_T1_MODEL
    t1_max_iterations: int = _c.DEFAULT_T1_MAX_ITERATIONS
    # Which result triggers escalate to the next tier when the default
    # ladder has a T1 rung (subset of `cora.escalation.ESCALATE_TRIGGERS`).
    # `wall_hit` alone is the classic continuation; add `blocker` /
    # `low_confidence` to double-check those outcomes on a stronger model.
    escalation_triggers: frozenset[str] = _c.DEFAULT_ESCALATION_TRIGGERS
    # Full programmatic override of the escalation ladder: a
    # `cora.escalation.EscalationPolicy` (tiers + escalate_on + connector)
    # used verbatim instead of the t1_* fields above. Tier 0 describes the
    # primary review tier and should agree with `model` /
    # `max_tool_iterations`. None → build the default ladder from t1_*.
    escalation_policy: "EscalationPolicy | None" = None
    t2_disagreement: bool = _c.DEFAULT_T2_DISAGREEMENT
    t2_model: str = _c.DEFAULT_T2_MODEL
    t2_max_iterations: int = _c.DEFAULT_T2_MAX_ITERATIONS
    # Classifier-large-diff entry: skip T0, start on T1 directly.
    skip_t0: bool = _c.DEFAULT_SKIP_T0
    # Whether a validated `propose_patch` directive may create inline
    # suggestions / draft PRs / source-branch edits. Default-off for
    # public adopters; private deployments opt in explicitly.
    propose_patch_dispatch: bool = _c.DEFAULT_PROPOSE_PATCH_DISPATCH
    # Mandatory T2 verification of propose_patch verdicts (kill switch).
    patch_escalation: bool = _c.DEFAULT_PATCH_ESCALATION

    # ── Context injection (push-based mid-review refresher) ──────────
    context_injection: bool = _c.CONTEXT_INJECTION_ENABLED
    context_injection_ci: bool = _c.CONTEXT_INJECTION_CI
    context_injection_head: bool = _c.CONTEXT_INJECTION_HEAD
    context_injection_comments: bool = _c.CONTEXT_INJECTION_COMMENTS
    # Green-delta sub-toggle of the CI source — a check-run turning
    # success also injects, not just a new failure. See
    # `core.config.CONTEXT_INJECTION_CI_GREEN`.
    context_injection_ci_green: bool = _c.CONTEXT_INJECTION_CI_GREEN

    # ── CI-verdict gate (finalize-time backstop; see cora.review._ci_gate) ──
    # On a settled `needs changes` verdict, one bounded re-poll of the
    # reviewed HEAD SHA's check-runs; blocker findings matching a narrow
    # compile/test-failure claim pattern get annotated (and the verdict
    # downgraded one step) when CI already passed. Default-on kill switch.
    ci_verdict_gate: bool = _c.CI_VERDICT_GATE_ENABLED
    retraction_verdict_gate: bool = _c.RETRACTION_VERDICT_GATE_ENABLED

    # ── Per-bundle caps ──────────────────────────────────────────────
    diff_char_cap: int = _c.DIFF_CHAR_CAP
    claude_md_char_cap: int = _c.CLAUDE_MD_CHAR_CAP
    pr_body_char_cap: int = _c.PR_BODY_CHAR_CAP
    ci_context_char_cap: int = _c.CI_CONTEXT_CHAR_CAP
    ci_context_include_passing: bool = _c.CI_CONTEXT_INCLUDE_PASSING
    ci_log_tail_chars: int = _c.CI_LOG_TAIL_CHARS

    # ── Linked-issue prefetch (server-side fetch of PR-referenced
    #    issues; see core/issue_context.py) — default-on kill switch,
    #    env `AGENT_REVIEW_ISSUE_PREFETCH` ──────────────────────────
    issue_context_prefetch: bool = _c.ISSUE_PREFETCH_ENABLED
    issue_prefetch_max_issues: int = _c.ISSUE_PREFETCH_MAX_ISSUES
    issue_body_char_cap: int = _c.ISSUE_BODY_CHAR_CAP
    issue_comment_char_cap: int = _c.ISSUE_COMMENT_CHAR_CAP
    issue_block_char_cap: int = _c.ISSUE_BLOCK_CHAR_CAP

    # ── Review-thread evidence (recent maintainer comments fed into the
    #    initial prompt so a re-review can converge on a rebuttal
    #    instead of re-asserting a refuted finding; cora #37) —
    #    default-on kill switch, env `AGENT_REVIEW_THREAD_EVIDENCE` ───
    thread_evidence: bool = _c.THREAD_EVIDENCE_ENABLED
    thread_evidence_associations: frozenset[str] = field(
        default_factory=lambda: frozenset(_c.THREAD_EVIDENCE_ASSOCIATIONS)
    )
    thread_evidence_max_comments: int = _c.THREAD_EVIDENCE_MAX_COMMENTS
    thread_evidence_comment_char_cap: int = _c.THREAD_EVIDENCE_COMMENT_CHAR_CAP
    thread_evidence_block_char_cap: int = _c.THREAD_EVIDENCE_BLOCK_CHAR_CAP

    # ── Retrieval (consumed via the RetrievalProvider seam) ──────────
    qdrant_url: str = _c.DEFAULT_QDRANT_URL
    tei_url: str = _c.DEFAULT_TEI_URL
    reranker_url: str = _c.DEFAULT_RERANKER_URL
    qdrant_api_key: str | None = None
    qdrant_collection: str = _c.QDRANT_COLLECTION
    retrieval_top_k: int = _c.RETRIEVAL_TOP_K
    retrieval_overfetch_k: int = _c.RETRIEVAL_OVERFETCH_K
    retrieval_stage1_survivors: int = _c.RETRIEVAL_STAGE1_SURVIVORS
    retrieval_doc_char_cap: int = _c.RETRIEVAL_DOC_CHAR_CAP
    retrieval_query_char_cap: int = _c.RETRIEVAL_QUERY_CHAR_CAP
    retrieval_timeout_s: int = _c.RETRIEVAL_TIMEOUT_S
    retrieval_cache_dir: Path = _c.RETRIEVAL_CACHE_DIR
    retrieval_cache_ttl_s: int = _c.RETRIEVAL_CACHE_TTL_S
    retrieval_skip_labels: frozenset[str] = _c.RETRIEVAL_SKIP_LABELS
    # Domain vocabulary for query reformulation — regex-fragment terms a PR
    # is likely to reference, surfaced as explicit identifiers for the sparse
    # channel + embedder. Empty default disables vocab matching
    # (decision-record / PR refs still fire); a deployment supplies its
    # own domain terms wholesale.
    # See `core.config.DEFAULT_RETRIEVAL_VOCAB`.
    retrieval_vocab: tuple[str, ...] = _c.DEFAULT_RETRIEVAL_VOCAB
    # Zero-infra local retrieval (GlobRetrievalProvider): globs relative
    # to the checkout root whose files get BM25-ranked against the PR
    # query. Empty (the default) = the provider is not selected.
    retrieval_glob_include: tuple[str, ...] = _c.RETRIEVAL_GLOB_INCLUDE
    retrieval_glob_max_files: int = _c.RETRIEVAL_GLOB_MAX_FILES
    retrieval_glob_max_file_bytes: int = _c.RETRIEVAL_GLOB_MAX_FILE_BYTES
    # pr-label-classifier verdict for this run (`CLASSIFIER_LABEL` env);
    # a value in `retrieval_skip_labels` bypasses retrieval entirely.
    classifier_label: str = _c.DEFAULT_CLASSIFIER_LABEL

    # ── Tool exposure (MCP tool allow-sets; copied, not aliased) ─────
    read_tools: frozenset[str] = field(default_factory=lambda: frozenset(_c.READ_TOOLS))
    action_tools: frozenset[str] = field(default_factory=lambda: frozenset(_c.ACTION_TOOLS))
    web_tools: frozenset[str] = field(default_factory=lambda: frozenset(_c.WEB_TOOLS))
    local_repo_tools: frozenset[str] = field(default_factory=lambda: frozenset(_c.LOCAL_REPO_TOOLS))
    local_issue_tools: frozenset[str] = field(default_factory=lambda: frozenset(_c.LOCAL_ISSUE_TOOLS))

    # ── Dependency-source corpus (optional; deep mode's
    #    grep_repo(corpus="deps")) ────────────────────────────────────
    # CSV of absolute paths to resolved dependency-source trees
    # (`DEP_SOURCE_ROOTS`), parsed in `from_env`. Empty (the default)
    # self-disarms unless in-repo `vendor/`/`node_modules/` are
    # auto-detected under the checkout root — existence + auto-detection
    # both happen at startup in the `GitProvider` seam (it needs the
    # checkout root), see `providers.git._resolve_dep_source_roots`.
    dep_source_roots: tuple[str, ...] = _c.DEFAULT_DEP_SOURCE_ROOTS
    dep_source_max_files_scanned: int = _c.DEP_SOURCE_MAX_FILES_SCANNED

    # ── Cold-start pretrigger (disarmed when empty) ──────────────────
    # The `model` aliases the warmup fires at (see `cora.core.pretrigger`).
    # Empty default = no warmup; a deployment with scale-from-zero
    # backends lists their aliases here.
    pretrigger_warmup_models: frozenset[str] = _c.PRETRIGGER_WARMUP_MODELS

    # ── Telemetry (optional) ─────────────────────────────────────────
    grafana_base: str | None = _c.GRAFANA_BASE
    dashboard_path: str = _c.DASHBOARD_PATH
    otel_endpoint: str | None = None
    otel_service_version: str | None = None

    # ── GitHub App identity (optional — for propose_patch) ───────────
    github_app_token: str | None = None

    # ── Feature flags / dev knobs ────────────────────────────────────
    enable_thinking: bool = False
    # No-op since the validate-any-claim framing became the prompt
    # default (it was this flag's teacher-trajectory variant). Accepted
    # so existing REVIEWER_BROADEN_TOOLS deployments keep working.
    broaden_tools: bool = False
    per_call_timeout_s: float | None = None
    # Opaque per-review session id sent as `x-review-session` on every
    # model call. The client half of gateway session affinity — see
    # `cora.core.agent.SESSION_HEADER`. None sends no header at all, so
    # an unconfigured deployment is byte-identical to before.
    session_header: str | None = None
    transcript_dir: str | None = None
    transcript_source: str = _c.DEFAULT_TRANSCRIPT_SOURCE
    eval_output_dir: str | None = None
    loop_guard_bot_login: str | None = None

    # ── Trigger security ─────────────────────────────────────────────
    # Who may fire a review and at what capability. Enforcement is on by
    # default for the public release posture; private deployments that
    # need the legacy call profile can explicitly set
    # REVIEW_TRIGGER_ENFORCE=false while they migrate allowlists.
    trigger: TriggerPolicy = field(default_factory=TriggerPolicy)

    @property
    def wall_time_s(self) -> int:
        """Derived total envelope, matching `config.DEFAULT_WALL_TIME_S`.
        A bare `WALL_TIME_S` ops override (folded in by `from_env`) pins
        the total exactly via `wall_time_override_s` — the 40/60 tier
        re-derivation floors each split, so the sum can drift a second
        or two below the operator's number otherwise."""
        if self.wall_time_override_s is not None:
            return self.wall_time_override_s
        return self.t0_wall_time_s + self.t1_wall_time_s + self.post_headroom_s

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "ReviewerConfig":
        """Build a config from the workflow environment.

        The env wiring for a CI-workflow deployment: every supported
        env knob is folded into the matching field, with the parsing
        quirks documented on each read below.
        """
        e = environ if environ is not None else os.environ

        def get(name: str, default: str | None = None) -> str | None:
            v = e.get(name)
            return v if v not in (None, "") else default

        def getbool(name: str, default: bool) -> bool:
            v = e.get(name)
            return default if v in (None, "") else v.strip().lower() == "true"

        def getint(name: str, default: int) -> int:
            v = e.get(name)
            return int(v) if v not in (None, "") else default

        def getfloat(name: str, default: float) -> float:
            v = e.get(name)
            return float(v) if v not in (None, "") else default

        def getflag_on(name: str) -> bool:
            """Default-true kill switch: only the literal "false"
            (case-insensitive) disables — typo-safe, matching
            `patch_escalation.escalation_enabled` and
            `context_refresher._env_on` (so e.g. "0" stays enabled)."""
            return e.get(name, "true").strip().lower() != "false"

        def getalias(name: str, default: str) -> str:
            """Model-alias chain: unset OR empty/whitespace → default
            (`env.get(name, d).strip() or d`)."""
            v = e.get(name)
            return (v if v is not None else default).strip() or default

        # T1 iteration cap forgives a malformed value (the int() is
        # wrapped in try/except ValueError → design default).
        try:
            t1_max_iterations = int(
                e.get(
                    "AGENT_REVIEW_T1_MAX_ITERATIONS",
                    str(_c.DEFAULT_T1_MAX_ITERATIONS),
                )
            )
        except ValueError:
            t1_max_iterations = _c.DEFAULT_T1_MAX_ITERATIONS

        # Malformed `MCP_SERVERS` raises loudly (ValueError, uncaught) —
        # a misconfigured deployment should fail at startup, not
        # silently run with fewer tools than it asked for. See
        # `cora.core.mcp_sessions.parse_mcp_servers_env`.
        from cora.core.mcp_sessions import parse_mcp_servers_env

        mcp_servers = parse_mcp_servers_env(e.get("MCP_SERVERS"), environ=e)
        extra_tools = frozenset(
            s.strip() for s in (e.get("AGENT_REVIEW_EXTRA_TOOLS") or "").split(",") if s.strip()
        )

        cfg = cls(
            repo=get("GH_REPO") or get("GITHUB_REPOSITORY") or "",
            pr_number=get("PR_NUMBER") or "",
            llm_base_url=get("LITELLM_BASE_URL", _c.DEFAULT_LITELLM_BASE),
            llm_api_key=get("LLM_GATEWAY_KEY"),
            model=get("REVIEW_MODEL", _c.DEFAULT_MODEL),
            mcp_url=get("MCP_URL", _c.DEFAULT_MCP_URL),
            mcp_token=get("MCP_TOKEN"),
            mcp_actions_url=get("MCP_ACTIONS_URL"),
            mcp_actions_token=get("MCP_ACTIONS_TOKEN"),
            web_fetch_gate_url=get("WEB_FETCH_GATE_URL"),
            mcp_servers=mcp_servers,
            extra_tools=extra_tools,
            check_run_name=get("REVIEW_CHECK_RUN_NAME", _c.CHECK_RUN_NAME),
            # Opt-in first-class PR Review. Default-false gate (only
            # the literal "true" enables); unset keeps the
            # comment + check-run path.
            use_github_review=getbool(
                "REVIEW_USE_GITHUB_REVIEW", _c.DEFAULT_USE_GITHUB_REVIEW
            ),
            qdrant_url=get("QDRANT_URL", _c.DEFAULT_QDRANT_URL),
            tei_url=get("TEI_URL", _c.DEFAULT_TEI_URL),
            reranker_url=get("RERANKER_URL", _c.DEFAULT_RERANKER_URL),
            qdrant_api_key=get("QDRANT_API_KEY"),
            qdrant_collection=get("QDRANT_COLLECTION", _c.QDRANT_COLLECTION),
            github_app_token=get("CORA_GH_TOKEN"),
            grafana_base=(get("GRAFANA_BASE_URL", _c.GRAFANA_BASE) or "").rstrip("/") or None,
            dashboard_path=get("GRAFANA_DASHBOARD_PATH", _c.DASHBOARD_PATH) or "",
            otel_endpoint=get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otel_service_version=get("OTEL_SERVICE_VERSION"),
            enable_thinking=getbool("AGENT_REVIEW_ENABLE_THINKING", False),
            broaden_tools=getbool("REVIEWER_BROADEN_TOOLS", False),
            transcript_dir=get("REVIEWER_TRANSCRIPT_DIR"),
            transcript_source=getalias(
                "REVIEWER_TRANSCRIPT_SOURCE", _c.DEFAULT_TRANSCRIPT_SOURCE
            ),
            eval_output_dir=get("AGENT_REVIEW_EVAL_OUTPUT_DIR"),
            # Empty is treated as unset: a deployment that exports the
            # variable but computes no value must send no header,
            # not an empty one.
            session_header=get("AGENT_REVIEW_SESSION_HEADER"),
            loop_guard_bot_login=get("AGENT_REVIEW_LOOP_GUARD_BOT_LOGIN"),
            max_tool_iterations=getint("MAX_TOOL_ITERATIONS", _c.DEFAULT_MAX_TOOL_ITERATIONS),
            # Per-call completion ceiling for the tier (deep) legs. Sized
            # against `per_call_timeout_s`, not against the context window —
            # see `_c.DEEP_MAX_OUTPUT_TOKENS`.
            deep_max_output_tokens=getint(
                "AGENT_REVIEW_MAX_COMPLETION_TOKENS", _c.DEEP_MAX_OUTPUT_TOKENS
            ),
            # Uncommitted-draw re-draw — default-true killswitch (only
            # the literal "false" disables), matching the other
            # reliability defaults.
            spiral_redraw=getflag_on("AGENT_REVIEW_SPIRAL_REDRAW"),
            # Exhausted-spiral escalation — same default-true killswitch
            # shape as the re-draw it backstops.
            spiral_escalation=getflag_on("AGENT_REVIEW_SPIRAL_ESCALATION"),
            # Streaming detection — default-false gate (only the literal
            # "true" enables); its tunables tolerate empty → engine
            # default and are inert while the gate is off.
            stream_detection=getbool(
                "AGENT_REVIEW_STREAM_DETECTION", _c.STREAM_DETECTION_ENABLED
            ),
            stall_timeout_s=getfloat("AGENT_REVIEW_STALL_TIMEOUT_S", _c.STALL_TIMEOUT_S),
            thinking_budget_tokens=getint(
                "AGENT_REVIEW_THINKING_BUDGET_TOKENS", _c.THINKING_BUDGET_TOKENS
            ),
            spiral_degrade_thinking=getbool(
                "AGENT_REVIEW_SPIRAL_DEGRADE_THINKING", _c.SPIRAL_DEGRADE_THINKING
            ),
            # Spiral recovery — default-false gate (only the literal
            # "true" enables); the bounds tolerate empty →
            # engine default. Unset keeps the default soft-fail.
            spiral_recovery=getbool(
                "AGENT_REVIEW_SPIRAL_RECOVERY", _c.SPIRAL_RECOVERY_ENABLED
            ),
            spiral_recovery_max_output_tokens=getint(
                "AGENT_REVIEW_SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS",
                _c.SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS,
            ),
            spiral_recovery_reasoning_char_cap=getint(
                "AGENT_REVIEW_SPIRAL_RECOVERY_REASONING_CHAR_CAP",
                _c.SPIRAL_RECOVERY_REASONING_CHAR_CAP,
            ),
            retrieval_cache_ttl_s=getint("AGENT_REVIEW_CACHE_TTL_S", _c.RETRIEVAL_CACHE_TTL_S),
            dep_source_max_files_scanned=getint(
                "DEP_SOURCE_MAX_FILES_SCANNED", _c.DEP_SOURCE_MAX_FILES_SCANNED
            ),
            # Tier escalation. The default-false gates use the
            # `.strip().lower() == "true"` parse (anything
            # but "true" stays off); the alias chains treat empty as
            # unset; T2's cap tolerates empty but deliberately
            # raises on a malformed non-empty value.
            t1_continuation=getbool(
                "AGENT_REVIEW_T1_CONTINUATION", _c.DEFAULT_T1_CONTINUATION
            ),
            t1_model=getalias("T1_MODEL", _c.DEFAULT_T1_MODEL),
            t1_max_iterations=t1_max_iterations,
            t2_disagreement=getbool(
                "AGENT_REVIEW_T2_DISAGREEMENT", _c.DEFAULT_T2_DISAGREEMENT
            ),
            t2_model=getalias("AGENT_REVIEW_T2_MODEL", _c.DEFAULT_T2_MODEL),
            t2_max_iterations=int(
                e.get("AGENT_REVIEW_T2_MAX_ITERATIONS")
                or _c.DEFAULT_T2_MAX_ITERATIONS
            ),
            skip_t0=getbool("AGENT_REVIEW_SKIP_T0", _c.DEFAULT_SKIP_T0),
            propose_patch_dispatch=getbool(
                "REVIEW_PROPOSE_PATCH_DISPATCH",
                _c.DEFAULT_PROPOSE_PATCH_DISPATCH,
            ),
            patch_escalation=getflag_on("AGENT_REVIEW_PATCH_ESCALATION"),
            classifier_label=e.get(
                "CLASSIFIER_LABEL", _c.DEFAULT_CLASSIFIER_LABEL
            ),
            # Context injection — default-true killswitches.
            context_injection=getflag_on("AGENT_REVIEW_CONTEXT_INJECTION"),
            context_injection_ci=getflag_on("AGENT_REVIEW_CONTEXT_INJECTION_CI"),
            context_injection_head=getflag_on("AGENT_REVIEW_CONTEXT_INJECTION_HEAD"),
            context_injection_comments=getflag_on(
                "AGENT_REVIEW_CONTEXT_INJECTION_COMMENTS"
            ),
            context_injection_ci_green=getflag_on(
                "AGENT_REVIEW_CONTEXT_INJECTION_CI_GREEN"
            ),
            # Finalize-time CI-verdict gate — default-true kill switch.
            ci_verdict_gate=getflag_on("AGENT_REVIEW_CI_VERDICT_GATE"),
            # Retraction-verdict gate — same default-true killswitch shape.
            retraction_verdict_gate=getflag_on(
                "AGENT_REVIEW_RETRACTION_VERDICT_GATE"
            ),
            # Passing-check context — same default-true killswitch shape.
            ci_context_include_passing=getflag_on(
                "AGENT_REVIEW_CI_CONTEXT_PASSING"
            ),
            # Linked-issue prefetch — default-true killswitch, same
            # typo-safe shape as the context-injection switches above.
            issue_context_prefetch=getflag_on("AGENT_REVIEW_ISSUE_PREFETCH"),
            # Review-thread evidence — same default-true killswitch shape.
            thread_evidence=getflag_on("AGENT_REVIEW_THREAD_EVIDENCE"),
        )

        # Wall-time ops overrides: an explicit
        # `T0_WALL_TIME_S` / `T1_WALL_TIME_S` always wins (membership,
        # not truthiness — an empty value deliberately raises);
        # a bare `WALL_TIME_S` pins the exact total and
        # re-derives the tiers as a 40/60 split of (total − headroom).
        if "T0_WALL_TIME_S" in e:
            cfg.t0_wall_time_s = int(e["T0_WALL_TIME_S"])
        if "T1_WALL_TIME_S" in e:
            cfg.t1_wall_time_s = int(e["T1_WALL_TIME_S"])
        if "WALL_TIME_S" in e:
            total = int(e["WALL_TIME_S"])
            cfg.wall_time_override_s = total
            if "T0_WALL_TIME_S" not in e:
                cfg.t0_wall_time_s = int((total - _c.POST_HEADROOM_S) * 0.4)
            if "T1_WALL_TIME_S" not in e:
                cfg.t1_wall_time_s = int((total - _c.POST_HEADROOM_S) * 0.6)

        # Env vars with bespoke parsing kept faithful to config.py.
        if (cache_dir := get("AGENT_REVIEW_CACHE_DIR")):
            cfg.retrieval_cache_dir = Path(cache_dir)
        if (timeout := get("AGENT_REVIEW_PER_CALL_TIMEOUT_S")):
            cfg.per_call_timeout_s = float(timeout)
        if (skip := e.get("AGENT_REVIEW_SKIP_RETRIEVAL_LABELS")) is not None:
            cfg.retrieval_skip_labels = frozenset(
                s.strip().lower() for s in skip.split(",") if s.strip()
            )
        # Zero-infra glob retrieval opt-in: csv of globs relative to the
        # checkout root, e.g. "docs/**/*.md,*.md".
        if (globs := get("AGENT_REVIEW_RETRIEVAL_GLOB")):
            cfg.retrieval_glob_include = tuple(
                s.strip() for s in globs.split(",") if s.strip()
            )
        # Dependency-source corpus roots: CSV of absolute paths a
        # deployment's CI runner has already materialized (Go module
        # cache, vendor dir, node_modules, site-packages, ...).
        # Existence + in-repo auto-detection are resolved later, by the
        # `GitProvider` seam (it needs the checkout root) — `from_env`
        # only parses the list.
        if (dep_roots := get("DEP_SOURCE_ROOTS")):
            cfg.dep_source_roots = tuple(
                s.strip() for s in dep_roots.split(",") if s.strip()
            )

        # Escalation trigger set — CSV of `ESCALATE_TRIGGERS` members.
        # Validated here so a misspelt trigger fails the run loudly
        # instead of silently never escalating.
        if (trig := get("CORA_ESCALATION_TRIGGERS")):
            from cora.escalation import ESCALATE_TRIGGERS

            triggers = frozenset(
                s.strip().lower() for s in trig.split(",") if s.strip()
            )
            unknown = triggers - ESCALATE_TRIGGERS
            if unknown:
                raise ValueError(
                    f"unknown escalation triggers: {sorted(unknown)}"
                )
            cfg.escalation_triggers = triggers

        # Cold-start pretrigger warmup aliases — csv override of the
        # default warmup set (membership, not truthiness: an
        # explicit empty value disables the warmup entirely; unset keeps
        # the default set).
        if "CORA_PRETRIGGER_WARMUP_MODELS" in e:
            cfg.pretrigger_warmup_models = frozenset(
                s.strip()
                for s in e["CORA_PRETRIGGER_WARMUP_MODELS"].split(",")
                if s.strip()
            )

        # Trigger security. REVIEW_TRIGGER_ENFORCE defaults
        # on; setting it to "false" restores the legacy inert policy.
        # Remaining knobs are still parsed either way (a misspelt action
        # value raises — a security knob must not fail silent).
        def getcsv(name: str) -> frozenset[str] | None:
            v = e.get(name)
            if v in (None, ""):
                return None
            return frozenset(s.strip() for s in v.split(",") if s.strip())

        trigger_kwargs: dict = {
            "enforce": getbool("REVIEW_TRIGGER_ENFORCE", True),
        }
        if (assoc := getcsv("REVIEW_TRIGGER_ALLOWED_ASSOCIATIONS")) is not None:
            trigger_kwargs["allowed_associations"] = frozenset(
                a.upper() for a in assoc
            )
        if (authors := getcsv("REVIEW_TRIGGER_ALLOWED_AUTHORS")) is not None:
            trigger_kwargs["allowed_authors"] = authors
        if (label := get("REVIEW_TRIGGER_APPROVE_LABEL")):
            trigger_kwargs["approve_label"] = label
        if (action := get("REVIEW_TRIGGER_UNTRUSTED_ACTION")):
            trigger_kwargs["untrusted_action"] = action.strip().lower()
        if (action := get("REVIEW_TRIGGER_FORK_ACTION")):
            trigger_kwargs["fork_action"] = action.strip().lower()
        if (cap := get("REVIEW_TRIGGER_MAX_RUNS_PER_HOUR")):
            trigger_kwargs["max_runs_per_hour"] = int(cap)
        if (cap := get("REVIEW_TRIGGER_MAX_RUNS_PER_AUTHOR_PER_HOUR")):
            trigger_kwargs["max_runs_per_author_per_hour"] = int(cap)
        cfg.trigger = TriggerPolicy(**trigger_kwargs)
        return cfg
