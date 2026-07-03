"""`_gh_check_api` token routing — prefer the cora App token so the
progress check lands in the App's own suite, fall back to the ambient
GITHUB_TOKEN when absent or on App failure (ported from production)."""

from __future__ import annotations

from types import SimpleNamespace

import cora.core.check_run as cr


def _ok(out="{}"):
    return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_prefers_app_token_when_set(monkeypatch):
    seen = []

    def fake_run(args, **kw):
        seen.append((kw.get("env") or {}).get("GH_TOKEN"))
        return _ok()

    monkeypatch.setenv("CORA_GH_TOKEN", "app-tok")
    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    cr._gh_check_api(["gh", "api"], {"x": 1})
    assert seen == ["app-tok"]  # one attempt, via the App token


def test_falls_back_to_github_token_when_no_app_token(monkeypatch):
    attempts = []

    def fake_run(args, **kw):
        attempts.append(kw)
        return _ok()

    monkeypatch.delenv("CORA_GH_TOKEN", raising=False)
    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    cr._gh_check_api(["gh", "api"], {})
    # no App token → single attempt with the ambient env (no GH_TOKEN override)
    assert len(attempts) == 1


def test_falls_back_on_app_attempt_failure(monkeypatch):
    seen = []

    def fake_run(args, **kw):
        tok = (kw.get("env") or {}).get("GH_TOKEN")
        seen.append(tok)
        # App attempt fails; the GITHUB_TOKEN fallback succeeds.
        return SimpleNamespace(
            returncode=1 if tok == "app-tok" else 0, stdout="{}", stderr="nope"
        )

    monkeypatch.setenv("CORA_GH_TOKEN", "app-tok")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    proc = cr._gh_check_api(["gh", "api"], {})
    assert proc.returncode == 0          # fallback succeeded
    assert seen[0] == "app-tok"          # App tried first
    assert len(seen) == 2                # then fell back


def test_create_check_run_default_name(monkeypatch):
    """Default payload name stays the engine's default gate name."""
    captured = {}

    def fake_api(args, payload):
        captured.update(payload)
        return _ok('{"id": 1, "html_url": "u"}')

    monkeypatch.setattr(cr, "_gh_check_api", fake_api)
    cr.create_check_run("o/r", "1", "sha")
    assert captured["name"] == "cora"


def test_create_check_run_custom_name(monkeypatch):
    """check_run_name= rebrands the verdict check (dogfood uses 'cora')."""
    captured = {}

    def fake_api(args, payload):
        captured.update(payload)
        return _ok('{"id": 1, "html_url": "u"}')

    monkeypatch.setattr(cr, "_gh_check_api", fake_api)
    cr.create_check_run("o/r", "1", "sha", check_run_name="cora")
    assert captured["name"] == "cora"
