"""Smoke tests for `agent_review/deep_review.py`.

Covers:
  - `_make_pydantic_ai_local_tools` returns Tool instances with the
    expected `grep_repo` + `git_show` names
  - `_probe_mcp_server` returns False on an unreachable URL
  - `_PydanticAIUsageAdapter` maps `request_tokens` /
    `response_tokens` onto `prompt_tokens` / `completion_tokens`
    (Budget interface parity)

`deep_review_call` itself isn't covered here — it needs a working
LLM endpoint + MCP server. Live coverage is the
`reviewer-pipeline-eval` workflow against the eval corpus.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("pydantic_ai.mcp")


def test_local_tools_have_expected_names():
    """The Pydantic-AI Tool wrappers expose the same `grep_repo` /
    `git_show` names as the MCP-server originals, so the model's
    learned tool use carries over from when those were MCP-only."""
    from cora.core.deep_review import _make_pydantic_ai_local_tools

    tools = _make_pydantic_ai_local_tools(tool_arg_defaults=None)
    names = sorted(t.name for t in tools)
    assert names == ["git_show", "grep_repo"]


def test_local_tools_callable_returns_handler_output(tmp_path, monkeypatch):
    """The Tool wrappers should dispatch to the underlying
    `local_grep_repo` / `local_git_show` handlers — adapter layer is
    intentionally thin (typed args → dict → handler)."""
    # Redirect REPO_ROOT so the test doesn't grep the real repo.
    (tmp_path / "needle.txt").write_text("haystack-marker\n")
    monkeypatch.setattr("cora.core.config.REPO_ROOT", tmp_path)
    # repo_tools captures REPO_ROOT at import time via `from ... import REPO_ROOT`,
    # so patch its module-level reference too.
    monkeypatch.setattr("cora.core.repo_tools.REPO_ROOT", tmp_path)

    from cora.core.deep_review import _make_pydantic_ai_local_tools

    tools = _make_pydantic_ai_local_tools(tool_arg_defaults=None)
    grep_tool = next(t for t in tools if t.name == "grep_repo")

    # `Tool.function` is the wrapped async callable; invoke directly.
    result = asyncio.run(grep_tool.function(pattern="haystack-marker"))
    assert "haystack-marker" in result
    assert "needle.txt" in result


def test_probe_mcp_server_returns_false_on_unreachable():
    """Probe should soft-fail to False on a connection error — the
    legacy `open_optional_mcp_session` did the same. Used to gate
    optional MCP servers before adding them to the toolsets list."""
    from cora.core.deep_review import _probe_mcp_server

    captured: list[str] = []

    def log(msg: str) -> None:
        captured.append(msg)

    # Port 1 is reserved and unreachable on any sane host; the probe
    # should hit a connection error quickly. If the loopback stack
    # disagrees the test might hang on connect timeout — pydantic-ai
    # defaults to a 5s timeout, well inside pytest's default 60s.
    result = asyncio.run(
        _probe_mcp_server(
            "http://127.0.0.1:1/mcp",
            {"Authorization": "Bearer test"},
            "test-server",
            log,
        )
    )
    assert result is False
    # Log line carries the failure reason so the GHA log shows why a
    # toolset was dropped — same UX as the legacy `::warning::` path.
    assert any("test-server probe failed" in line for line in captured)


def test_pydantic_ai_usage_adapter_maps_field_names():
    """Pydantic-AI exposes `request_tokens` / `response_tokens` /
    `total_tokens`; Budget reads OpenAI's `prompt_tokens` /
    `completion_tokens` / `total_tokens`. Adapter shim bridges the
    field-name mismatch — same shape as the quick-mode adapter."""
    from cora.core.deep_review import _PydanticAIUsageAdapter

    class FakeUsage:
        request_tokens = 100
        response_tokens = 50
        total_tokens = 150

    adapted = _PydanticAIUsageAdapter(FakeUsage())
    assert adapted.prompt_tokens == 100
    assert adapted.completion_tokens == 50
    assert adapted.total_tokens == 150


def test_pydantic_ai_usage_adapter_handles_missing_fields():
    """If the framework ever omits a field (e.g. an error path returns
    a partial usage object), the adapter should default to 0 rather
    than raising — Budget tracking is soft-fail."""
    from cora.core.deep_review import _PydanticAIUsageAdapter

    class EmptyUsage:
        pass

    adapted = _PydanticAIUsageAdapter(EmptyUsage())
    assert adapted.prompt_tokens == 0
    assert adapted.completion_tokens == 0
    assert adapted.total_tokens == 0
