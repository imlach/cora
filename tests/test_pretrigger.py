"""Tests for the cold-start pretrigger warmup (agent_review/pretrigger.py)."""
from __future__ import annotations

import asyncio
import json
from unittest import mock

import cora.core.pretrigger as pretrigger


def test_should_fire_default_is_disarmed():
    # Default (no warmup_models) = the empty engine set: nothing warms
    # until a deployment lists its scale-from-zero aliases.
    assert not pretrigger._should_fire("review", "2110")
    assert not pretrigger._should_fire("main", "2110")


def test_should_fire_uses_config_supplied_warmup_set():
    # A deployment-supplied set arms the warmup for exactly its aliases.
    models = frozenset({"sonnet"})
    assert pretrigger._should_fire("sonnet", "2110", models)
    assert not pretrigger._should_fire("review", "2110", models)
    assert not pretrigger._should_fire("sonnet", "", models)  # no PR → no routing tag
    # An empty set keeps the warmup disabled (single-model adopter).
    assert not pretrigger._should_fire("review", "2110", frozenset())


def test_post_builds_warmup_with_pr_and_prewarm_tags():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data)
        captured["auth"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        return object()

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        pretrigger._post("http://gw/v1/chat/completions", "KEY", "review", "2110", 15)

    assert captured["url"] == "http://gw/v1/chat/completions"
    assert captured["data"]["model"] == "review"
    assert captured["data"]["messages"][0]["content"] == "warmup pr-2110"
    assert captured["data"]["max_tokens"] == 1
    assert captured["data"]["cache"] == {"no-cache": True}
    assert captured["data"]["metadata"]["tags"] == ["pr-2110", "prewarm"]
    assert captured["auth"] == "Bearer KEY"
    assert captured["timeout"] == 15


def test_post_swallows_errors():
    def boom(req, timeout=None):
        raise OSError("gateway down")

    with mock.patch("urllib.request.urlopen", boom):
        pretrigger._post("http://gw/v1/chat/completions", "KEY", "review", "2110", 15)  # must not raise


def test_timeout_from_env(monkeypatch):
    monkeypatch.setenv(pretrigger.TIMEOUT_ENV, "90")
    assert pretrigger._timeout_s() == 90


def test_timeout_from_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv(pretrigger.TIMEOUT_ENV, "nope")
    assert pretrigger._timeout_s() == pretrigger.DEFAULT_TIMEOUT_S


def test_fire_pretrigger_noop_for_non_elastic():
    calls = []
    with mock.patch.object(pretrigger, "_post", lambda *a: calls.append(a)):
        asyncio.run(pretrigger.fire_pretrigger("http://gw", "KEY", "review-lite", "2110"))
    assert calls == []


def test_fire_pretrigger_fires_for_configured_alias():
    calls = []
    models = frozenset({"review"})
    with mock.patch.object(pretrigger, "_post", lambda *a: calls.append(a)):
        asyncio.run(
            pretrigger.fire_pretrigger("http://gw/", "KEY", "review", "2110", models)
        )
    assert len(calls) == 1
    assert calls[0] == ("http://gw/v1/chat/completions", "KEY", "review", "2110", 15.0)


def test_fire_pretrigger_honours_config_supplied_set():
    calls = []
    models = frozenset({"sonnet"})
    with mock.patch.object(pretrigger, "_post", lambda *a: calls.append(a)):
        # The default alias is not in the adopter's set → no warmup.
        asyncio.run(
            pretrigger.fire_pretrigger("http://gw/", "KEY", "review", "2110", models)
        )
        assert calls == []
        # The adopter's own alias warms.
        asyncio.run(
            pretrigger.fire_pretrigger("http://gw/", "KEY", "sonnet", "2110", models)
        )
    assert len(calls) == 1
    assert calls[0][2] == "sonnet"


def test_warmup_models_default_mirrors_engine_constant():
    """The `ReviewerConfig` field default IS the engine constant (by
    reference), so the two cannot drift. Empty by default — the
    pretrigger stays disarmed until a deployment lists its
    scale-from-zero aliases."""
    from cora.config import ReviewerConfig
    from cora.core import config as _c

    assert ReviewerConfig().pretrigger_warmup_models is _c.PRETRIGGER_WARMUP_MODELS
    assert pretrigger.PRETRIGGER_WARMUP_MODELS is _c.PRETRIGGER_WARMUP_MODELS
    assert frozenset() == _c.PRETRIGGER_WARMUP_MODELS


def test_from_env_overrides_warmup_models_csv():
    from cora.config import ReviewerConfig

    # Unset → the (empty) engine default.
    assert ReviewerConfig.from_env({}).pretrigger_warmup_models == frozenset()
    # CSV override replaces the set.
    cfg = ReviewerConfig.from_env(
        {"CORA_PRETRIGGER_WARMUP_MODELS": "sonnet, opus"}
    )
    assert cfg.pretrigger_warmup_models == frozenset({"sonnet", "opus"})
    # Explicit empty value disables the warmup entirely.
    assert (
        ReviewerConfig.from_env(
            {"CORA_PRETRIGGER_WARMUP_MODELS": ""}
        ).pretrigger_warmup_models
        == frozenset()
    )
