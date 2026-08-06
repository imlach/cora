"""ReviewerConfig surface guarantees.

Two invariants the extraction relies on:
1. Zero-drift — every mirrored default equals the engine constant, so the
   public surface can't silently diverge from `cora.core.config`.
2. from_env faithfulness — env wiring is honoured (and respects an
   injected mapping, not just the real os.environ).
"""

from cora import ReviewerConfig
from cora.core import config as c


def test_defaults_mirror_engine_constants():
    d = ReviewerConfig()
    assert d.llm_base_url == c.DEFAULT_LITELLM_BASE
    assert d.model == c.DEFAULT_MODEL
    assert d.comment_marker == c.COMMENT_MARKER
    assert d.check_run_name == c.CHECK_RUN_NAME
    # First-class PR Review is off by default (the default reporting
    # path stays comment + check-run).
    assert d.use_github_review == c.DEFAULT_USE_GITHUB_REVIEW
    assert d.use_github_review is False
    assert d.verdict_glyphs is c.VERDICT_GLYPHS
    assert d.verdict_words is c.VERDICT_WORDS
    assert d.max_input_tokens == c.MAX_INPUT_TOKENS
    assert d.max_tool_iterations == c.DEFAULT_MAX_TOOL_ITERATIONS
    assert d.diff_char_cap == c.DIFF_CHAR_CAP
    assert d.qdrant_collection == c.QDRANT_COLLECTION
    assert d.read_tools == frozenset(c.READ_TOOLS)
    assert d.wall_time_s == c.DEFAULT_WALL_TIME_S
    # Tier escalation + run-shaping knobs (env-knob graduation).
    assert d.t1_continuation == c.DEFAULT_T1_CONTINUATION
    assert d.t1_model == c.DEFAULT_T1_MODEL
    assert d.t1_max_iterations == c.DEFAULT_T1_MAX_ITERATIONS
    assert d.t2_disagreement == c.DEFAULT_T2_DISAGREEMENT
    assert d.t2_model == c.DEFAULT_T2_MODEL
    assert d.t2_max_iterations == c.DEFAULT_T2_MAX_ITERATIONS
    assert d.skip_t0 == c.DEFAULT_SKIP_T0
    assert d.propose_patch_dispatch == c.DEFAULT_PROPOSE_PATCH_DISPATCH
    assert d.propose_patch_dispatch is False
    assert d.patch_escalation == c.DEFAULT_PATCH_ESCALATION
    assert d.classifier_label == c.DEFAULT_CLASSIFIER_LABEL
    assert d.wall_time_override_s is None
    assert d.context_injection == c.CONTEXT_INJECTION_ENABLED
    assert d.context_injection_ci == c.CONTEXT_INJECTION_CI
    assert d.context_injection_head == c.CONTEXT_INJECTION_HEAD
    assert d.context_injection_comments == c.CONTEXT_INJECTION_COMMENTS
    assert d.context_injection_ci_green == c.CONTEXT_INJECTION_CI_GREEN
    assert d.ci_verdict_gate == c.CI_VERDICT_GATE_ENABLED
    # Linked-issue prefetch — killswitch + caps + tool allow-set.
    assert d.issue_context_prefetch == c.ISSUE_PREFETCH_ENABLED
    assert d.issue_context_prefetch is True
    assert d.issue_prefetch_max_issues == c.ISSUE_PREFETCH_MAX_ISSUES
    assert d.issue_body_char_cap == c.ISSUE_BODY_CHAR_CAP
    assert d.issue_comment_char_cap == c.ISSUE_COMMENT_CHAR_CAP
    assert d.issue_block_char_cap == c.ISSUE_BLOCK_CHAR_CAP
    assert d.local_issue_tools == frozenset(c.LOCAL_ISSUE_TOOLS)
    assert d.transcript_source == c.DEFAULT_TRANSCRIPT_SOURCE
    # Exhausted-spiral escalation — on by default (killswitch), like the
    # re-draw it backstops.
    assert d.spiral_escalation == c.SPIRAL_ESCALATION_ENABLED
    assert d.spiral_escalation is True
    # Spiral recovery — off by default so existing deployments keep
    # today's soft-fail bit-for-bit; bounds mirror the engine constants.
    assert d.spiral_recovery == c.SPIRAL_RECOVERY_ENABLED
    assert d.spiral_recovery is False
    assert d.spiral_recovery_max_output_tokens == c.SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS
    assert (
        d.spiral_recovery_reasoning_char_cap
        == c.SPIRAL_RECOVERY_REASONING_CHAR_CAP
    )
    # Dependency-source corpus (grep_repo corpus="deps", cora #23) —
    # empty/unset self-disarms, same convention as MCP_URL.
    assert d.dep_source_roots == c.DEFAULT_DEP_SOURCE_ROOTS
    assert d.dep_source_roots == ()
    assert d.dep_source_max_files_scanned == c.DEP_SOURCE_MAX_FILES_SCANNED


def test_from_env_respects_injected_mapping():
    env = {
        "GH_REPO": "owner/repo",
        "PR_NUMBER": "42",
        "LITELLM_BASE_URL": "http://gw:4000",
        "LLM_GATEWAY_KEY": "sk-test",
        "MAX_TOOL_ITERATIONS": "8",
        "AGENT_REVIEW_SKIP_RETRIEVAL_LABELS": "deps, docs ,wip",
        "REVIEW_CHECK_RUN_NAME": "cora",
    }
    fe = ReviewerConfig.from_env(env)
    assert fe.check_run_name == "cora"
    assert fe.repo == "owner/repo"
    assert fe.pr_number == "42"
    assert fe.llm_base_url == "http://gw:4000"
    assert fe.llm_api_key == "sk-test"
    assert fe.max_tool_iterations == 8          # int parsed from injected env
    assert fe.retrieval_skip_labels == frozenset({"deps", "docs", "wip"})
    assert fe.diff_char_cap == c.DIFF_CHAR_CAP    # unset -> engine default


def test_from_env_empty_is_safe():
    fe = ReviewerConfig.from_env({})
    assert fe.repo == ""
    assert fe.model == c.DEFAULT_MODEL
    assert fe.llm_api_key is None
    # No env → engine defaults across the graduated knobs.
    assert fe.t1_continuation is False
    assert fe.t1_model == c.DEFAULT_T1_MODEL
    assert fe.t2_model == c.DEFAULT_T2_MODEL
    assert fe.skip_t0 is False
    assert fe.propose_patch_dispatch is False
    assert fe.patch_escalation is True
    assert fe.context_injection is True
    assert fe.classifier_label == ""
    assert fe.t0_wall_time_s == c.DEFAULT_T0_WALL_TIME_S
    assert fe.t1_wall_time_s == c.DEFAULT_T1_WALL_TIME_S
    assert fe.wall_time_override_s is None
    assert fe.wall_time_s == c.DEFAULT_WALL_TIME_S
    assert fe.use_github_review is False  # PR-Review opt-in stays off
    assert fe.spiral_recovery is False  # spiral recovery opt-in stays off


def test_from_env_spiral_escalation_killswitch():
    # Default-true killswitch: only the literal "false" (case-insensitive)
    # disables — same typo-safe parse as AGENT_REVIEW_SPIRAL_REDRAW.
    assert ReviewerConfig.from_env({}).spiral_escalation is True
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_ESCALATION": "false"}
    ).spiral_escalation is False
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_ESCALATION": "FALSE"}
    ).spiral_escalation is False
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_ESCALATION": "0"}
    ).spiral_escalation is True   # typo-safe: stays enabled


def test_from_env_spiral_recovery_opt_in():
    # Default-false gate, entrypoint-style parse: only literal "true"
    # (case-insensitive) enables; anything else keeps the soft-fail path.
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_RECOVERY": "true"}
    ).spiral_recovery is True
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_RECOVERY": "True"}
    ).spiral_recovery is True
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_RECOVERY": "1"}
    ).spiral_recovery is False
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_SPIRAL_RECOVERY": ""}
    ).spiral_recovery is False
    # Bounds parse from env; unset → engine defaults.
    fe = ReviewerConfig.from_env(
        {
            "AGENT_REVIEW_SPIRAL_RECOVERY": "true",
            "AGENT_REVIEW_SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS": "5000",
            "AGENT_REVIEW_SPIRAL_RECOVERY_REASONING_CHAR_CAP": "2000",
        }
    )
    assert fe.spiral_recovery_max_output_tokens == 5000
    assert fe.spiral_recovery_reasoning_char_cap == 2000
    assert (
        ReviewerConfig.from_env({}).spiral_recovery_max_output_tokens
        == c.SPIRAL_RECOVERY_MAX_OUTPUT_TOKENS
    )


def test_from_env_use_github_review_opt_in():
    # Default-false gate, entrypoint-style parse: only literal "true"
    # enables; anything else (incl. typos) keeps the default comment
    # path.
    assert ReviewerConfig.from_env(
        {"REVIEW_USE_GITHUB_REVIEW": "true"}
    ).use_github_review is True
    assert ReviewerConfig.from_env(
        {"REVIEW_USE_GITHUB_REVIEW": "True"}
    ).use_github_review is True
    assert ReviewerConfig.from_env(
        {"REVIEW_USE_GITHUB_REVIEW": "1"}
    ).use_github_review is False
    assert ReviewerConfig.from_env(
        {"REVIEW_USE_GITHUB_REVIEW": ""}
    ).use_github_review is False


def test_from_env_propose_patch_dispatch_opt_in():
    assert ReviewerConfig.from_env(
        {"REVIEW_PROPOSE_PATCH_DISPATCH": "true"}
    ).propose_patch_dispatch is True
    assert ReviewerConfig.from_env(
        {"REVIEW_PROPOSE_PATCH_DISPATCH": "True"}
    ).propose_patch_dispatch is True
    assert ReviewerConfig.from_env(
        {"REVIEW_PROPOSE_PATCH_DISPATCH": "1"}
    ).propose_patch_dispatch is False
    assert ReviewerConfig.from_env(
        {"REVIEW_PROPOSE_PATCH_DISPATCH": ""}
    ).propose_patch_dispatch is False


# ── Tier-escalation knobs: workflow-shaped env ────────────────────────


def test_from_env_tier_escalation_knobs():
    fe = ReviewerConfig.from_env(
        {
            "AGENT_REVIEW_T1_CONTINUATION": "true",
            "T1_MODEL": " big-ctx ",
            "AGENT_REVIEW_T1_MAX_ITERATIONS": "12",
            "AGENT_REVIEW_T2_DISAGREEMENT": "TRUE",
            "AGENT_REVIEW_T2_MODEL": "alt-review-b",
            "AGENT_REVIEW_T2_MAX_ITERATIONS": "5",
            "AGENT_REVIEW_SKIP_T0": "true",
            "CLASSIFIER_LABEL": "deps",
        }
    )
    assert fe.t1_continuation is True
    assert fe.t1_model == "big-ctx"          # whitespace stripped
    assert fe.t1_max_iterations == 12
    assert fe.t2_disagreement is True        # case-insensitive "true"
    assert fe.t2_model == "alt-review-b"
    assert fe.t2_max_iterations == 5
    assert fe.skip_t0 is True
    assert fe.classifier_label == "deps"


def test_from_env_tier_knob_fallbacks_match_entrypoint():
    """The entrypoint's quirky parses, preserved: empty aliases fall to
    the default ('' or default chains), a malformed T1 cap forgives to
    the design default, an empty T2 cap falls to 8, and the default-
    false gates treat anything but 'true' as off."""
    fe = ReviewerConfig.from_env(
        {
            "T1_MODEL": "  ",
            "AGENT_REVIEW_T1_MAX_ITERATIONS": "not-a-number",
            "AGENT_REVIEW_T2_MODEL": "",
            "AGENT_REVIEW_T2_MAX_ITERATIONS": "",
            "AGENT_REVIEW_T1_CONTINUATION": "1",
            "AGENT_REVIEW_T2_DISAGREEMENT": "yes",
            "AGENT_REVIEW_SKIP_T0": "",
        }
    )
    assert fe.t1_model == c.DEFAULT_T1_MODEL
    assert fe.t1_max_iterations == c.DEFAULT_T1_MAX_ITERATIONS
    assert fe.t2_model == c.DEFAULT_T2_MODEL
    assert fe.t2_max_iterations == c.DEFAULT_T2_MAX_ITERATIONS
    assert fe.t1_continuation is False       # only literal "true" enables
    assert fe.t2_disagreement is False
    assert fe.skip_t0 is False


def test_from_env_default_true_kill_switches_are_typo_safe():
    """patch-escalation + context-injection: only the literal 'false'
    disables (entrypoint semantics) — '0' / garbage stay enabled."""
    fe = ReviewerConfig.from_env(
        {
            "AGENT_REVIEW_PATCH_ESCALATION": "0",
            "AGENT_REVIEW_CONTEXT_INJECTION": "False",
            "AGENT_REVIEW_CONTEXT_INJECTION_CI": " false ",
            "AGENT_REVIEW_CONTEXT_INJECTION_HEAD": "nope",
            "AGENT_REVIEW_CONTEXT_INJECTION_COMMENTS": "",
            "AGENT_REVIEW_CONTEXT_INJECTION_CI_GREEN": "false",
            "AGENT_REVIEW_CI_VERDICT_GATE": "0",
        }
    )
    assert fe.patch_escalation is True       # "0" is not "false"
    assert fe.context_injection is False     # case-insensitive
    assert fe.context_injection_ci is False  # whitespace tolerated
    assert fe.context_injection_head is True
    assert fe.context_injection_comments is True  # empty ≠ "false"
    assert fe.context_injection_ci_green is False
    assert fe.ci_verdict_gate is True         # "0" is not "false"


# ── Wall-time ops overrides ──────────────────────────────────────────


def test_from_env_explicit_tier_walls_win():
    fe = ReviewerConfig.from_env(
        {"T0_WALL_TIME_S": "100", "T1_WALL_TIME_S": "200"}
    )
    assert fe.t0_wall_time_s == 100
    assert fe.t1_wall_time_s == 200
    assert fe.wall_time_override_s is None
    assert fe.wall_time_s == 100 + 200 + c.POST_HEADROOM_S


def test_from_env_bare_wall_time_rederives_split_and_pins_total():
    fe = ReviewerConfig.from_env({"WALL_TIME_S": "1000"})
    # 40/60 split of (total − headroom), floored — entrypoint-verbatim.
    assert fe.t0_wall_time_s == int((1000 - c.POST_HEADROOM_S) * 0.4)
    assert fe.t1_wall_time_s == int((1000 - c.POST_HEADROOM_S) * 0.6)
    # The reported total is the operator's exact number, not the
    # (floor-drifted) re-derived sum.
    assert fe.wall_time_s == 1000


def test_from_env_explicit_tier_beats_bare_wall_time():
    fe = ReviewerConfig.from_env(
        {"WALL_TIME_S": "1000", "T0_WALL_TIME_S": "111"}
    )
    assert fe.t0_wall_time_s == 111          # explicit wins
    assert fe.t1_wall_time_s == int((1000 - c.POST_HEADROOM_S) * 0.6)
    assert fe.wall_time_s == 1000


# ── Transcript source tag ────────────────────────────────────────────


def test_from_env_transcript_source():
    assert (
        ReviewerConfig.from_env({"REVIEWER_TRANSCRIPT_SOURCE": " teacher-v2 "})
        .transcript_source
        == "teacher-v2"
    )
    assert (
        ReviewerConfig.from_env({"REVIEWER_TRANSCRIPT_SOURCE": ""}).transcript_source
        == c.DEFAULT_TRANSCRIPT_SOURCE
    )


def test_from_env_parses_escalation_triggers():
    fe = ReviewerConfig.from_env(
        {"CORA_ESCALATION_TRIGGERS": "wall_hit, blocker"}
    )
    assert fe.escalation_triggers == frozenset({"wall_hit", "blocker"})
    # Unset keeps the engine default.
    assert ReviewerConfig.from_env({}).escalation_triggers == (
        c.DEFAULT_ESCALATION_TRIGGERS
    )


def test_from_env_rejects_unknown_escalation_trigger():
    import pytest

    with pytest.raises(ValueError, match="unknown escalation triggers"):
        ReviewerConfig.from_env({"CORA_ESCALATION_TRIGGERS": "wall_hit,bogus"})


# ── Generic extra MCP sessions ─────────────────────────────────────────


def test_from_env_defaults_mcp_servers_and_extra_tools_empty():
    fe = ReviewerConfig.from_env({})
    assert fe.mcp_servers == ()
    assert fe.extra_tools == frozenset()
    assert ReviewerConfig().mcp_servers == ()
    assert ReviewerConfig().extra_tools == frozenset()


def test_from_env_parses_mcp_servers_json():
    import json

    from cora.core.mcp_sessions import McpServerSpec

    raw = json.dumps(
        [
            {
                "name": "docs2",
                "url": "https://mcp.example/docs",
                "token_env": "DOCS2_TOKEN",
                "required": False,
            }
        ]
    )
    fe = ReviewerConfig.from_env({"MCP_SERVERS": raw, "DOCS2_TOKEN": "sekret"})
    assert fe.mcp_servers == (
        McpServerSpec(
            name="docs2",
            url="https://mcp.example/docs",
            headers={"Authorization": "Bearer sekret"},
            required=False,
        ),
    )


def test_from_env_malformed_mcp_servers_raises():
    import pytest

    with pytest.raises(ValueError, match="not valid JSON"):
        ReviewerConfig.from_env({"MCP_SERVERS": "{not json"})


def test_from_env_parses_extra_tools_csv():
    fe = ReviewerConfig.from_env(
        {"AGENT_REVIEW_EXTRA_TOOLS": "custom_tool_a, custom_tool_b ,"}
    )
    assert fe.extra_tools == frozenset({"custom_tool_a", "custom_tool_b"})
    assert ReviewerConfig.from_env({}).extra_tools == frozenset()
# ── Dependency-source corpus (grep_repo corpus="deps", cora #23) ──────


def test_from_env_parses_dep_source_roots_csv():
    fe = ReviewerConfig.from_env(
        {"DEP_SOURCE_ROOTS": "/opt/gomodcache, /repo/vendor ,/repo/node_modules"}
    )
    assert fe.dep_source_roots == (
        "/opt/gomodcache",
        "/repo/vendor",
        "/repo/node_modules",
    )


def test_from_env_dep_source_roots_unset_stays_empty():
    assert ReviewerConfig.from_env({}).dep_source_roots == ()
    assert ReviewerConfig.from_env({"DEP_SOURCE_ROOTS": ""}).dep_source_roots == ()


def test_from_env_parses_dep_source_max_files_scanned():
    fe = ReviewerConfig.from_env({"DEP_SOURCE_MAX_FILES_SCANNED": "1234"})
    assert fe.dep_source_max_files_scanned == 1234
    assert (
        ReviewerConfig.from_env({}).dep_source_max_files_scanned
        == c.DEP_SOURCE_MAX_FILES_SCANNED
    )
# ── Linked-issue prefetch killswitch ─────────────────────────────────


def test_from_env_issue_prefetch_killswitch():
    # Default-true killswitch: only the literal "false" (case-insensitive)
    # disables — same typo-safe parse as the context-injection switches.
    assert ReviewerConfig.from_env({}).issue_context_prefetch is True
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_ISSUE_PREFETCH": "false"}
    ).issue_context_prefetch is False
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_ISSUE_PREFETCH": "FALSE"}
    ).issue_context_prefetch is False
    assert ReviewerConfig.from_env(
        {"AGENT_REVIEW_ISSUE_PREFETCH": "0"}
    ).issue_context_prefetch is True  # typo-safe: stays enabled
