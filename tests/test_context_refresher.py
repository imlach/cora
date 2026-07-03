"""Tests for the push-based ContextRefresher.

The refresher polls three side channels between agent-graph node
boundaries: PR HEAD SHA, CI check-runs delta, and new human comments.
These tests pin the contract on each source's dedupe + cadence + env
toggle, plus the wall-extension callback wiring.

All `gh api` calls are stubbed at construction (`gh_api_fn` kwarg) so
tests stay hermetic — no subprocess, no network. `gather_ci_context_fn`
+ `fetch_pr_diff_fn` are stubbed the same way.
"""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("pydantic_ai")


def _make_refresher(
    *,
    repo="owner/repo",
    pr_number="123",
    head_sha="sha-old",
    start_ts="2026-05-27T00:00:00Z",
    bot_login="cora[bot]",
    gh_api_responses: dict | None = None,
    ci_context_fn=None,
    diff_fn=None,
    env_overrides: dict | None = None,
    monkeypatch=None,
):
    """Helper — construct a ContextRefresher with stubbed network calls.

    `gh_api_responses` maps URL prefixes to either a JSON-serializable
    body (dict/list) or a callable returning one. URL matching is by
    `startswith` so callers can write e.g. `/repos/owner/repo/pulls/`
    without spelling out the full path.
    """
    if env_overrides and monkeypatch is not None:
        for k, v in env_overrides.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)

    gh_api_responses = gh_api_responses or {}

    def stub_gh(path: str) -> str | None:
        for prefix, resp in gh_api_responses.items():
            if path.startswith(prefix):
                payload = resp() if callable(resp) else resp
                if payload is None:
                    return None
                return json.dumps(payload)
        return None

    from cora.core.context_refresher import ContextRefresher

    return ContextRefresher(
        repo=repo,
        pr_number=pr_number,
        head_sha=head_sha,
        start_timestamp_iso=start_ts,
        bot_login=bot_login,
        gather_ci_context_fn=ci_context_fn or (lambda _r, _s: "## CI failing\nfake"),
        fetch_pr_diff_fn=diff_fn or (lambda _p: "diff --git a/x b/x\n@@ fake @@\n"),
        gh_api_fn=stub_gh,
    )


# ---- HEAD SHA delta ----------------------------------------------


def test_head_sha_first_call_returns_none_when_unchanged():
    """When the head SHA matches the seed value, no injection fires —
    the baseline is the initial prompt's diff snapshot, not the
    refresher's responsibility to surface."""
    refresher = _make_refresher(
        head_sha="sha-old",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha-old"}},
        },
    )
    result = asyncio.run(refresher.refresh(turn=1))
    assert result is None


def test_head_sha_delta_injects_diff_when_changed():
    """A new head SHA triggers an injection that re-fetches the diff
    and wraps it in the harness-injection envelope."""
    refresher = _make_refresher(
        head_sha="sha-old",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha-new"}},
        },
    )
    result = asyncio.run(refresher.refresh(turn=1))
    assert result is not None
    assert "[CONTEXT UPDATE" in result
    assert "PR head updated mid-review" in result
    assert "sha-old" in result
    assert "sha-new" in result
    assert "diff --git" in result
    assert refresher.last_source == "head"


def test_head_sha_dedupe_no_re_injection():
    """Same SHA on two consecutive calls — only the first fires."""
    responses = {
        "/repos/owner/repo/pulls/123": {"head": {"sha": "sha-new"}},
    }
    refresher = _make_refresher(head_sha="sha-old", gh_api_responses=responses)
    first = asyncio.run(refresher.refresh(turn=1))
    second = asyncio.run(refresher.refresh(turn=1))
    assert first is not None
    assert second is None


# ---- CI delta ----------------------------------------------------


def test_ci_delta_first_observation_is_baseline_no_injection():
    """First CI check is the baseline; only subsequent changes inject."""
    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/commits/sha/check-runs": {
                "check_runs": [
                    {"name": "test", "status": "completed", "conclusion": "failure"},
                ],
            },
        },
    )
    # Turn 3 triggers the CI cadence; first call seeds the hash.
    result = asyncio.run(refresher.refresh(turn=3))
    assert result is None


def test_ci_delta_injects_when_signature_changes():
    """A check transitioning from success → failure triggers injection."""
    state = {"runs": [
        {"name": "test", "status": "completed", "conclusion": "success"},
    ]}

    def ci_resp():
        return {"check_runs": state["runs"]}

    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/commits/sha/check-runs": ci_resp,
        },
    )
    # Turn 3 — seed baseline (success).
    asyncio.run(refresher.refresh(turn=3))
    # Flip to failure.
    state["runs"] = [
        {"name": "test", "status": "completed", "conclusion": "failure"},
    ]
    # Turn 6 — next CI cadence tick (turn % 3 == 0).
    result = asyncio.run(refresher.refresh(turn=6))
    assert result is not None
    assert "CI checks changed" in result
    assert refresher.last_source == "ci"


# ---- Comments delta ----------------------------------------------


def test_comments_delta_injects_new_human_comments():
    """A new comment from a non-bot login triggers injection."""
    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/issues/123/comments": [
                {
                    "id": 100,
                    "user": {"login": "human-author"},
                    "body": "Note: this is for a migration only.",
                },
            ],
        },
    )
    # Turn 2 — comments cadence (turn % 2 == 0).
    result = asyncio.run(refresher.refresh(turn=2))
    assert result is not None
    assert "New comments on the PR" in result
    assert "human-author" in result
    assert "migration only" in result
    assert refresher.last_source == "comments"


def test_comments_filters_out_bot_self_comments():
    """The reviewer's own bot login is filtered — comments-delta
    must not fire on the in-progress / final review comments the
    reviewer itself posts (feedback loop)."""
    refresher = _make_refresher(
        head_sha="sha",
        bot_login="cora[bot]",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/issues/123/comments": [
                {
                    "id": 200,
                    "user": {"login": "cora[bot]"},
                    "body": "Verdict: looks good",
                },
            ],
        },
    )
    result = asyncio.run(refresher.refresh(turn=2))
    assert result is None


def test_comments_dedupe_by_id():
    """Once injected, a comment id is never re-injected on later polls
    even if the API echoes it back. Tracking by max-seen-id."""
    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/issues/123/comments": [
                {"id": 300, "user": {"login": "human"}, "body": "first"},
            ],
        },
    )
    first = asyncio.run(refresher.refresh(turn=2))
    second = asyncio.run(refresher.refresh(turn=2))
    assert first is not None
    assert second is None


# ---- Polling cadence ---------------------------------------------


def test_ci_cadence_only_fires_every_third_turn():
    """CI is the most expensive call — cadence is every 3 turns. On
    turns 1, 2, 4, 5 the refresher should not have queried the
    check-runs endpoint at all."""
    call_log = []

    def ci_resp():
        call_log.append("ci")
        return {"check_runs": []}

    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/commits/sha/check-runs": ci_resp,
            # Empty comments → comments check skipped silently.
            "/repos/owner/repo/issues/123/comments": [],
        },
    )
    for turn in (1, 2, 4, 5):
        asyncio.run(refresher.refresh(turn=turn))
    assert call_log == [], f"CI fetched on non-cadence turns: {call_log}"
    # Turn 3 + 6 are the cadence ticks.
    asyncio.run(refresher.refresh(turn=3))
    asyncio.run(refresher.refresh(turn=6))
    assert call_log == ["ci", "ci"], f"CI cadence ticks missed: {call_log}"


def test_comments_cadence_every_second_turn():
    """Comments cadence: every 2 turns."""
    call_log = []

    def comments_resp():
        call_log.append("comments")
        return []

    refresher = _make_refresher(
        head_sha="sha",
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/issues/123/comments": comments_resp,
        },
    )
    for turn in (1, 3, 5):
        asyncio.run(refresher.refresh(turn=turn))
    assert call_log == [], f"comments fetched on odd turns: {call_log}"
    for turn in (2, 4, 6):
        asyncio.run(refresher.refresh(turn=turn))
    assert call_log == ["comments", "comments", "comments"]


# ---- Env toggles -------------------------------------------------


def test_master_env_toggle_disables_all(monkeypatch):
    """`AGENT_REVIEW_CONTEXT_INJECTION=false` disables every source."""
    refresher = _make_refresher(
        head_sha="sha-old",
        env_overrides={"AGENT_REVIEW_CONTEXT_INJECTION": "false"},
        monkeypatch=monkeypatch,
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha-new"}},
        },
    )
    result = asyncio.run(refresher.refresh(turn=1))
    assert result is None


def test_per_source_env_toggle_head(monkeypatch):
    """Only the HEAD source is disabled — CI + comments still poll."""
    refresher = _make_refresher(
        head_sha="sha-old",
        env_overrides={"AGENT_REVIEW_CONTEXT_INJECTION_HEAD": "false"},
        monkeypatch=monkeypatch,
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha-new"}},
        },
    )
    result = asyncio.run(refresher.refresh(turn=1))
    assert result is None


def test_per_source_env_toggle_ci(monkeypatch):
    """CI disabled — head + comments still poll. With head unchanged
    and no comments, the result is None but it's because nothing
    fired, not because CI was forced."""
    refresher = _make_refresher(
        head_sha="sha",
        env_overrides={"AGENT_REVIEW_CONTEXT_INJECTION_CI": "false"},
        monkeypatch=monkeypatch,
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            # CI would fire (signature change) but env says off.
            "/repos/owner/repo/commits/sha/check-runs": {
                "check_runs": [{"name": "x", "status": "completed", "conclusion": "failure"}],
            },
        },
    )
    # Seed and then "change" — without the toggle this would inject.
    asyncio.run(refresher.refresh(turn=3))
    result = asyncio.run(refresher.refresh(turn=6))
    assert result is None


def test_per_source_env_toggle_comments(monkeypatch):
    """Comments disabled — head + CI still poll."""
    refresher = _make_refresher(
        head_sha="sha",
        env_overrides={"AGENT_REVIEW_CONTEXT_INJECTION_COMMENTS": "false"},
        monkeypatch=monkeypatch,
        gh_api_responses={
            "/repos/owner/repo/pulls/123": {"head": {"sha": "sha"}},
            "/repos/owner/repo/issues/123/comments": [
                {"id": 1, "user": {"login": "human"}, "body": "hi"},
            ],
        },
    )
    result = asyncio.run(refresher.refresh(turn=2))
    assert result is None


def test_config_threaded_flags_win_over_env(monkeypatch):
    """Explicit ctor flags (cfg-threaded by run_review) bypass the env
    toggles entirely; None keeps the legacy env read — the cfg=None
    contract shared with `deep_review._thinking_extra_body`."""
    from cora.core.context_refresher import ContextRefresher

    # Env says ON — the explicit enabled=False must still disable.
    monkeypatch.setenv("AGENT_REVIEW_CONTEXT_INJECTION", "true")
    monkeypatch.setenv("AGENT_REVIEW_CONTEXT_INJECTION_HEAD", "true")

    def fake_gh(path: str) -> str:
        return json.dumps({"head": {"sha": "sha-new"}})

    off = ContextRefresher(
        repo="owner/repo",
        pr_number="123",
        head_sha="sha-old",
        start_timestamp_iso="2026-05-27T00:00:00Z",
        bot_login="cora[bot]",
        enabled=False,
        gh_api_fn=fake_gh,
        fetch_pr_diff_fn=lambda _p: "diff",
    )
    assert asyncio.run(off.refresh(turn=1)) is None

    # Env says OFF — the explicit per-source True must still fire.
    monkeypatch.setenv("AGENT_REVIEW_CONTEXT_INJECTION", "false")
    monkeypatch.setenv("AGENT_REVIEW_CONTEXT_INJECTION_HEAD", "false")
    on = ContextRefresher(
        repo="owner/repo",
        pr_number="123",
        head_sha="sha-old",
        start_timestamp_iso="2026-05-27T00:00:00Z",
        bot_login="cora[bot]",
        enabled=True,
        head_enabled=True,
        gh_api_fn=fake_gh,
        fetch_pr_diff_fn=lambda _p: "diff",
    )
    body = asyncio.run(on.refresh(turn=1))
    assert body is not None and "sha-new" in body


# ---- Wall-extension callback -------------------------------------


def test_extension_counter_caps_at_max():
    """`can_extend()` returns True for the first
    MAX_INJECTION_EXTENSIONS calls, then False."""
    from cora.core.context_refresher import (
        MAX_INJECTION_EXTENSIONS,
        ContextRefresher,
    )

    r = ContextRefresher(
        repo="o/r",
        pr_number="1",
        head_sha="s",
        start_timestamp_iso="2026-01-01T00:00:00Z",
    )
    for _ in range(MAX_INJECTION_EXTENSIONS):
        assert r.can_extend()
        r.record_extension()
    assert not r.can_extend()


def test_extension_default_constant_is_90s():
    """The +90s extension is a documented contract; pin it so a
    future tuning change has to update the docs too."""
    from cora.core.context_refresher import INJECTION_DEADLINE_EXTENSION_S

    assert INJECTION_DEADLINE_EXTENSION_S == 90.0


def test_max_extensions_is_three():
    """+3 × 90s = +270s ceiling per review. Pin so the wait_for
    pre-padding in deep_review + continuation stays in sync."""
    from cora.core.context_refresher import MAX_INJECTION_EXTENSIONS

    assert MAX_INJECTION_EXTENSIONS == 3


# ---- Injection envelope ------------------------------------------


def test_wrap_injection_envelope_shape():
    """The injection text must carry the 'not a user message' framing
    so the model doesn't treat the injection as a conversational
    pivot from the human author. Pin the literal text."""
    from cora.core.context_refresher import wrap_injection

    body = wrap_injection(reason="test", body="content here")
    assert body.startswith("[CONTEXT UPDATE")
    assert "not a user message" in body
    assert "Reason: test" in body
    assert "content here" in body
    assert "Adjust your in-progress verdict" in body
