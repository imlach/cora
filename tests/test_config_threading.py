"""ReviewerConfig threading through the engine: the config object is
the wiring, the `cora.core.config` constants remain only as the
dataclass's defaults — so a default-constructed config is bit-identical
to the pre-threading engine, and an explicit config actually steers it."""

from __future__ import annotations

import json

from cora.config import ReviewerConfig
from cora.core import config as c
from cora.core.agent import Deps
from cora.core.budget import Budget
from cora.core.deep_review import _thinking_extra_body
from cora.core.prompt import load_system_prompt
from cora.core.retrieval import _cache_get, _cache_put, build_retrieval_query


# ── Deps as the runtime carrier ──────────────────────────────────────


def test_deps_default_constructs_mirror_config():
    d = Deps(repo="o/r", pr_number="1")
    assert isinstance(d.cfg, ReviewerConfig)
    # The default-constructed config IS the engine constants.
    assert d.cfg.max_input_tokens == c.MAX_INPUT_TOKENS
    assert d.cfg.model == c.DEFAULT_MODEL


def test_deps_carries_explicit_config():
    cfg = ReviewerConfig(model="custom-alias")
    assert Deps(repo="o/r", pr_number="1", cfg=cfg).cfg is cfg


def test_deps_instances_do_not_share_a_default_config():
    a = Deps(repo="o/r", pr_number="1")
    b = Deps(repo="o/r", pr_number="2")
    assert a.cfg is not b.cfg


# ── Budget.from_config ───────────────────────────────────────────────


def test_budget_from_config_default_matches_production_caps():
    b = Budget.from_config(ReviewerConfig())
    assert (b.max_input, b.max_output, b.max_iterations) == (
        c.MAX_INPUT_TOKENS,
        c.MAX_OUTPUT_TOKENS,
        c.DEFAULT_MAX_TOOL_ITERATIONS,
    )


def test_budget_from_config_reads_overrides():
    cfg = ReviewerConfig(
        max_input_tokens=11, max_output_tokens=22, max_tool_iterations=3
    )
    b = Budget.from_config(cfg)
    assert (b.max_input, b.max_output, b.max_iterations) == (11, 22, 3)


def test_quick_output_cap_is_larger_than_deep_per_call_cap():
    # Quick mode is single-shot, so its one call must fit reasoning +
    # verdict; deep spreads reasoning across turns. The asymmetry is
    # intentional — guard against a silent regression back to parity
    # (which starved the reasoning model in the reference deployment).
    cfg = ReviewerConfig()
    assert cfg.quick_max_output_tokens == c.QUICK_MAX_OUTPUT_TOKENS
    assert cfg.quick_max_output_tokens > cfg.max_output_tokens


def test_deep_per_call_cap_is_bounded_on_both_sides():
    # The deep per-call cap (both T0 and T1 legs) is sized against the
    # per-call TIMEOUT, not the context window — but it is bounded below
    # too, and getting either side wrong is a real failure.
    from cora.core.budget import PER_CALL_TIMEOUT_S

    cfg = ReviewerConfig()
    assert cfg.deep_max_output_tokens == c.DEEP_MAX_OUTPUT_TOKENS

    # Ceiling: a draw that can't finish inside the timeout is cancelled
    # mid-generation and records nothing — no usage, no finish_reason,
    # invisible in every latency histogram. 110 tok/s is the low end of
    # the observed generation range.
    slowest_observed_tokens_per_s = 110
    assert cfg.deep_max_output_tokens / slowest_observed_tokens_per_s < PER_CALL_TIMEOUT_S

    # Floor: must clear the longest completion observed to SUCCEED, or
    # the cap truncates real reviews and every long turn pays for a
    # re-draw it didn't need.
    longest_successful_completion_tokens = 16_000
    assert cfg.deep_max_output_tokens > longest_successful_completion_tokens


def test_max_completion_tokens_env_overrides_deep_cap():
    cfg = ReviewerConfig.from_env({"AGENT_REVIEW_MAX_COMPLETION_TOKENS": "9000"})
    assert cfg.deep_max_output_tokens == 9_000
    # Unset / empty falls back to the engine constant.
    assert ReviewerConfig.from_env({}).deep_max_output_tokens == c.DEEP_MAX_OUTPUT_TOKENS
    assert (
        ReviewerConfig.from_env(
            {"AGENT_REVIEW_MAX_COMPLETION_TOKENS": ""}
        ).deep_max_output_tokens
        == c.DEEP_MAX_OUTPUT_TOKENS
    )


# ── Thinking toggle (deep mode) ──────────────────────────────────────


def test_thinking_extra_body_threads_config_flag(monkeypatch):
    # An explicit flag wins outright — env must not be consulted.
    monkeypatch.setenv("AGENT_REVIEW_ENABLE_THINKING", "true")
    assert _thinking_extra_body(False) == {}
    monkeypatch.delenv("AGENT_REVIEW_ENABLE_THINKING")
    assert _thinking_extra_body(True) == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
    }


def test_thinking_extra_body_none_keeps_legacy_env_read(monkeypatch):
    monkeypatch.delenv("AGENT_REVIEW_ENABLE_THINKING", raising=False)
    assert _thinking_extra_body(None) == {}
    monkeypatch.setenv("AGENT_REVIEW_ENABLE_THINKING", "true")
    assert _thinking_extra_body(None) != {}


# ── Prompt path resolution ───────────────────────────────────────────


def test_load_system_prompt_resolves_path_from_config(tmp_path):
    p = tmp_path / "deep.md"
    p.write_text("CFG DEEP PROMPT", encoding="utf-8")
    cfg = ReviewerConfig(deep_prompt_path=p)
    assert load_system_prompt(None, mode="deep", cfg=cfg) == "CFG DEEP PROMPT"
    # quick_prompt_path stays None → packaged quick prompt.
    assert load_system_prompt(None, mode="quick", cfg=cfg) == load_system_prompt(
        None, mode="quick"
    )


def test_load_system_prompt_default_config_is_bit_identical():
    assert load_system_prompt(None, mode="deep", cfg=ReviewerConfig()) == (
        load_system_prompt(None, mode="deep")
    )


def test_load_system_prompt_explicit_path_beats_config(tmp_path):
    explicit = tmp_path / "explicit.md"
    explicit.write_text("EXPLICIT", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("OTHER", encoding="utf-8")
    cfg = ReviewerConfig(deep_prompt_path=other)
    assert load_system_prompt(explicit, mode="deep", cfg=cfg) == "EXPLICIT"


# ── Retrieval tunables ───────────────────────────────────────────────


def test_build_retrieval_query_cap_from_config():
    cfg = ReviewerConfig(retrieval_query_char_cap=10)
    q = build_retrieval_query({"title": "t" * 100, "body": ""}, "", cfg=cfg)
    assert len(q["text"]) <= 20  # 2× the configured cap


def test_build_retrieval_query_default_config_matches_constants():
    meta = {"title": "fix cilium netpol", "body": "touches DEC-001"}
    diff = "+++ b/k8s/policy.yaml\n+spec: {}\n"
    assert build_retrieval_query(meta, diff, cfg=ReviewerConfig()) == (
        build_retrieval_query(meta, diff)
    )


def test_retrieval_cache_roundtrip_uses_config_dir(tmp_path, monkeypatch):
    # Env override must NOT leak in when a config is threaded.
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(tmp_path / "env-dir"))
    cfg = ReviewerConfig(retrieval_cache_dir=tmp_path / "cfg-dir")
    _cache_put("key1", [{"a": 1}], cfg=cfg)
    assert (tmp_path / "cfg-dir" / "key1.json").is_file()
    assert not (tmp_path / "env-dir").exists()
    docs, age = _cache_get("key1", cfg=cfg)
    assert docs == [{"a": 1}]
    assert age is not None


def test_retrieval_cache_ttl_from_config(tmp_path):
    cfg = ReviewerConfig(retrieval_cache_dir=tmp_path, retrieval_cache_ttl_s=-1)
    (tmp_path / "key2.json").write_text(json.dumps([{"a": 2}]), encoding="utf-8")
    assert _cache_get("key2", cfg=cfg) == (None, None)  # already expired

