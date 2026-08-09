"""`cora.core.mcp_sessions` — the generic MCP_SERVERS wiring.

Covers `parse_mcp_servers_env` (valid, malformed, token-env
indirection), `compose_mcp_sessions` (legacy-slot normalization
equivalence with the pre-refactor hand-rolled branches), `open_mcp_sessions`
(required-vs-optional probe/open semantics), and `resolve_web_fetch_url`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cora.core.mcp_sessions import (
    McpServerSpec,
    compose_mcp_sessions,
    open_mcp_sessions,
    parse_mcp_servers_env,
    resolve_web_fetch_url,
)

# ── parse_mcp_servers_env ──────────────────────────────────────────────


def test_unset_or_blank_returns_empty():
    assert parse_mcp_servers_env(None) == ()
    assert parse_mcp_servers_env("") == ()
    assert parse_mcp_servers_env("   ") == ()


def test_parses_a_single_valid_entry_with_token_indirection():
    raw = json.dumps(
        [{"name": "docs2", "url": "https://mcp.example/docs", "token_env": "DOCS2_TOKEN"}]
    )
    specs = parse_mcp_servers_env(raw, environ={"DOCS2_TOKEN": "sekret"})
    assert specs == (
        McpServerSpec(
            name="docs2",
            url="https://mcp.example/docs",
            headers={"Authorization": "Bearer sekret"},
            required=False,
        ),
    )


def test_required_flag_is_honoured():
    raw = json.dumps(
        [{"name": "docs2", "url": "https://mcp.example/docs", "required": True}]
    )
    (spec,) = parse_mcp_servers_env(raw, environ={})
    assert spec.required is True
    assert spec.headers == {}


def test_multiple_entries_preserve_order():
    raw = json.dumps(
        [
            {"name": "a", "url": "https://a.example/mcp"},
            {"name": "b", "url": "https://b.example/mcp"},
        ]
    )
    specs = parse_mcp_servers_env(raw, environ={})
    assert [s.name for s in specs] == ["a", "b"]


def test_malformed_json_raises_clear_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_mcp_servers_env("{not json", environ={})


def test_non_array_top_level_raises():
    with pytest.raises(ValueError, match="JSON array"):
        parse_mcp_servers_env(json.dumps({"name": "a", "url": "b"}), environ={})


def test_entry_missing_name_raises():
    with pytest.raises(ValueError, match="non-empty 'name'"):
        parse_mcp_servers_env(json.dumps([{"url": "https://x.example/mcp"}]), environ={})


def test_entry_missing_url_raises():
    with pytest.raises(ValueError, match="non-empty 'url'"):
        parse_mcp_servers_env(json.dumps([{"name": "a"}]), environ={})


def test_entry_not_an_object_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_mcp_servers_env(json.dumps(["not-an-object"]), environ={})


def test_required_must_be_boolean():
    raw = json.dumps([{"name": "a", "url": "https://a.example/mcp", "required": "yes"}])
    with pytest.raises(ValueError, match="'required' must be a JSON boolean"):
        parse_mcp_servers_env(raw, environ={})


def test_token_env_missing_from_environ_is_tokenless_with_a_warning(capsys):
    """A declared `token_env` that doesn't resolve is NOT the same as
    never declaring one — the operator asked for auth, so we warn —
    but the session still opens tokenless rather than the whole config
    parse failing."""
    raw = json.dumps(
        [{"name": "docs2", "url": "https://mcp.example/docs", "token_env": "MISSING_VAR"}]
    )
    (spec,) = parse_mcp_servers_env(raw, environ={})
    assert spec.headers == {}
    out = capsys.readouterr().out
    assert "MISSING_VAR" in out
    assert "docs2" in out
    assert "::warning::" in out


def test_no_token_env_at_all_is_silently_tokenless(capsys):
    """No `token_env` key means the server genuinely needs no auth —
    distinct from the warned case above, so no warning fires."""
    raw = json.dumps([{"name": "public-docs", "url": "https://mcp.example/docs"}])
    (spec,) = parse_mcp_servers_env(raw, environ={})
    assert spec.headers == {}
    assert capsys.readouterr().out == ""


def test_empty_token_env_value_also_warns(capsys):
    raw = json.dumps(
        [{"name": "docs2", "url": "https://mcp.example/docs", "token_env": "EMPTY_VAR"}]
    )
    (spec,) = parse_mcp_servers_env(raw, environ={"EMPTY_VAR": "   "})
    assert spec.headers == {}
    assert "::warning::" in capsys.readouterr().out


def test_defaults_to_real_os_environ_when_none_injected(monkeypatch):
    monkeypatch.setenv("REAL_ENV_TOKEN", "from-os-environ")
    raw = json.dumps(
        [{"name": "a", "url": "https://a.example/mcp", "token_env": "REAL_ENV_TOKEN"}]
    )
    (spec,) = parse_mcp_servers_env(raw)
    assert spec.headers == {"Authorization": "Bearer from-os-environ"}


# ── compose_mcp_sessions: legacy-slot normalization equivalence ───────


def test_no_slots_configured_yields_empty_list():
    assert (
        compose_mcp_sessions(
            mcp_url="",
            mcp_headers=None,
            mcp_actions_url=None,
            mcp_actions_headers=None,
            web_fetch_url=None,
            web_fetch_headers=None,
        )
        == []
    )


def test_mcp_slot_normalizes_to_a_required_spec():
    sessions = compose_mcp_sessions(
        mcp_url=" https://mcp.example/mcp ",
        mcp_headers={"Authorization": "Bearer t"},
        mcp_actions_url=None,
        mcp_actions_headers=None,
        web_fetch_url=None,
        web_fetch_headers=None,
    )
    assert sessions == [
        McpServerSpec(
            name="mcp",
            url="https://mcp.example/mcp",
            headers={"Authorization": "Bearer t"},
            required=True,
        )
    ]


def test_actions_slot_without_token_is_dropped_with_a_log_line():
    """Legacy quirk preserved verbatim: `MCP_ACTIONS_URL` configured but
    `MCP_ACTIONS_TOKEN` unset means "don't even attempt it" — distinct
    from a generic extra session, which attempts tokenless."""
    logs: list[str] = []
    sessions = compose_mcp_sessions(
        mcp_url="",
        mcp_headers=None,
        mcp_actions_url="https://actions.example/mcp",
        mcp_actions_headers=None,
        web_fetch_url=None,
        web_fetch_headers=None,
        log=logs.append,
    )
    assert sessions == []
    assert any("MCP_ACTIONS_TOKEN unset" in line for line in logs)


def test_actions_slot_with_token_normalizes_to_an_optional_spec():
    sessions = compose_mcp_sessions(
        mcp_url="",
        mcp_headers=None,
        mcp_actions_url="https://actions.example/mcp",
        mcp_actions_headers={"Authorization": "Bearer a"},
        web_fetch_url=None,
        web_fetch_headers=None,
    )
    assert sessions == [
        McpServerSpec(
            name="actions",
            url="https://actions.example/mcp",
            headers={"Authorization": "Bearer a"},
            required=False,
        )
    ]


def test_web_fetch_slot_normalizes_to_an_optional_tokenless_spec():
    sessions = compose_mcp_sessions(
        mcp_url="",
        mcp_headers=None,
        mcp_actions_url=None,
        mcp_actions_headers=None,
        web_fetch_url="https://fetch.example/mcp",
        web_fetch_headers=None,
    )
    assert sessions == [
        McpServerSpec(
            name="web-fetch", url="https://fetch.example/mcp", headers={}, required=False
        )
    ]


def test_all_three_legacy_slots_plus_extras_compose_in_order():
    extra = McpServerSpec(name="docs2", url="https://docs2.example/mcp", required=True)
    sessions = compose_mcp_sessions(
        mcp_url="https://mcp.example/mcp",
        mcp_headers={},
        mcp_actions_url="https://actions.example/mcp",
        mcp_actions_headers={"Authorization": "Bearer a"},
        web_fetch_url="https://fetch.example/mcp",
        web_fetch_headers=None,
        extra_sessions=[extra],
    )
    assert [s.name for s in sessions] == ["mcp", "actions", "web-fetch", "docs2"]
    assert sessions[0].required is True
    assert sessions[1].required is False
    assert sessions[2].required is False
    assert sessions[3] is extra


# ── open_mcp_sessions ───────────────────────────────────────────────────


def _probe_ok(ok_names: set[str]):
    async def _probe(url, headers, name, log):
        return name in ok_names

    return _probe


def test_required_probe_failure_short_circuits_to_none():
    sessions = [McpServerSpec(name="mcp", url="https://mcp.example/mcp", required=True)]
    result = asyncio.run(
        open_mcp_sessions(sessions, probe=_probe_ok(set()), log=lambda _l: None)
    )
    assert result is None


def test_required_failure_short_circuits_before_later_optional_sessions():
    """Matches the pre-refactor behaviour exactly: a failed REQUIRED
    probe returns before anything after it (actions / web-fetch /
    extras) is even attempted."""
    probed: list[str] = []

    async def probe(url, headers, name, log):
        probed.append(name)
        return name != "mcp"

    sessions = [
        McpServerSpec(name="mcp", url="https://mcp.example/mcp", required=True),
        McpServerSpec(name="actions", url="https://actions.example/mcp"),
    ]
    result = asyncio.run(open_mcp_sessions(sessions, probe=probe, log=lambda _l: None))
    assert result is None
    assert probed == ["mcp"]


def test_optional_probe_failure_drops_it_and_continues():
    sessions = [
        McpServerSpec(name="mcp", url="https://mcp.example/mcp", required=True),
        McpServerSpec(name="actions", url="https://actions.example/mcp"),
        McpServerSpec(name="web-fetch", url="https://fetch.example/mcp"),
    ]
    toolsets, opened = asyncio.run(
        open_mcp_sessions(sessions, probe=_probe_ok({"mcp", "web-fetch"}), log=lambda _l: None)
    )
    assert [u for u, _h in toolsets] == ["https://mcp.example/mcp", "https://fetch.example/mcp"]
    assert opened == {"mcp": True, "web-fetch": True}
    assert "actions" not in opened


def test_opened_optional_sessions_log_a_confirmation():
    logs: list[str] = []
    sessions = [
        McpServerSpec(name="actions", url="https://actions.example/mcp"),
        McpServerSpec(name="web-fetch", url="https://fetch.example/mcp"),
        McpServerSpec(name="docs2", url="https://docs2.example/mcp"),
    ]
    asyncio.run(
        open_mcp_sessions(
            sessions, probe=_probe_ok({"actions", "web-fetch", "docs2"}), log=logs.append
        )
    )
    assert any("actions session opened (observe-only)" in line for line in logs)
    assert any("web-fetch session opened" in line and "observe-only" not in line for line in logs)
    assert any("docs2 session opened" in line and "observe-only" not in line for line in logs)


def test_label_suffix_appears_in_the_probe_name():
    seen_names: list[str] = []

    async def probe(url, headers, name, log):
        seen_names.append(name)
        return True

    sessions = [McpServerSpec(name="mcp", url="https://mcp.example/mcp", required=True)]
    asyncio.run(
        open_mcp_sessions(sessions, probe=probe, log=lambda _l: None, label_suffix=" (T1)")
    )
    assert seen_names == ["mcp (T1)"]


def test_extra_required_session_failure_fails_the_whole_run_even_if_ordered_last():
    sessions = [
        McpServerSpec(name="mcp", url="https://mcp.example/mcp", required=True),
        McpServerSpec(name="docs2", url="https://docs2.example/mcp", required=True),
    ]
    result = asyncio.run(
        open_mcp_sessions(sessions, probe=_probe_ok({"mcp"}), log=lambda _l: None)
    )
    assert result is None


# ── resolve_web_fetch_url ───────────────────────────────────────────────


class _FakeCfg:
    def __init__(self, web_fetch_gate_url=None, mcp_servers=()):
        self.web_fetch_gate_url = web_fetch_gate_url
        self.mcp_servers = mcp_servers


def test_resolve_prefers_explicit_web_fetch_gate_url():
    cfg = _FakeCfg(
        web_fetch_gate_url="https://gate.example/mcp",
        mcp_servers=(McpServerSpec(name="web-fetch", url="https://extra.example/mcp"),),
    )
    assert resolve_web_fetch_url(cfg) == "https://gate.example/mcp"


def test_resolve_falls_back_to_named_extra_session():
    cfg = _FakeCfg(
        web_fetch_gate_url=None,
        mcp_servers=(
            McpServerSpec(name="docs2", url="https://docs2.example/mcp"),
            McpServerSpec(name="web-fetch", url="https://extra.example/mcp"),
        ),
    )
    assert resolve_web_fetch_url(cfg) == "https://extra.example/mcp"


def test_resolve_returns_empty_when_neither_configured():
    assert resolve_web_fetch_url(_FakeCfg()) == ""


# ── collisions between an extra and an already-claimed slot ──────────
# `opened` is name-keyed so two same-named sessions silently collapse
# there, and pydantic-ai's CombinedToolset raises UserError on the first
# duplicate TOOL name across toolsets — surfacing as `agent-loop-errored:`
# with no review posted. The documented `web-fetch` name convention hits
# this the moment WEB_FETCH_GATE_URL is also set.


def _compose(**kw):
    base = {
        "mcp_url": "https://mcp.example",
        "mcp_headers": {},
        "mcp_actions_url": None,
        "mcp_actions_headers": None,
        "web_fetch_url": None,
        "web_fetch_headers": None,
    }
    base.update(kw)
    return compose_mcp_sessions(**base)


def test_extra_named_like_a_legacy_slot_is_dropped():
    sessions = _compose(
        web_fetch_url="https://gate.example",
        web_fetch_headers={},
        extra_sessions=(
            McpServerSpec(name="web-fetch", url="https://other.example"),
        ),
    )
    assert [s.name for s in sessions] == ["mcp", "web-fetch"]
    assert [s.url for s in sessions] == ["https://mcp.example", "https://gate.example"]


def test_extra_duplicating_a_url_is_dropped():
    sessions = _compose(
        extra_sessions=(McpServerSpec(name="alias", url="https://mcp.example"),),
    )
    assert [s.name for s in sessions] == ["mcp"]


def test_two_extras_with_the_same_name_keep_only_the_first():
    sessions = _compose(
        extra_sessions=(
            McpServerSpec(name="docs2", url="https://a.example"),
            McpServerSpec(name="docs2", url="https://b.example"),
        ),
    )
    assert [s.name for s in sessions] == ["mcp", "docs2"]
    assert sessions[1].url == "https://a.example"


def test_distinct_extras_still_attach():
    """The guard must not disarm the feature it's protecting."""
    sessions = _compose(
        extra_sessions=(
            McpServerSpec(name="docs2", url="https://a.example"),
            McpServerSpec(name="docs3", url="https://b.example"),
        ),
    )
    assert [s.name for s in sessions] == ["mcp", "docs2", "docs3"]


def test_collision_is_logged():
    lines: list[str] = []
    _compose(
        extra_sessions=(McpServerSpec(name="mcp", url="https://x.example"),),
        log=lines.append,
    )
    assert any("duplicates an already-configured session name" in ln for ln in lines)


# ── URL scheme is validated at parse time, not at construction ───────


def test_non_http_url_rejected_at_parse_time():
    import pytest

    with pytest.raises(ValueError, match="must be http:// or https://"):
        parse_mcp_servers_env('[{"name": "x", "url": "not-a-url"}]')


def test_token_env_that_looks_like_a_token_is_not_echoed(capsys):
    parse_mcp_servers_env(
        '[{"name": "x", "url": "https://x.example", '
        '"token_env": "ghp_sensitiveLookingValue"}]',
        environ={},
    )
    err = capsys.readouterr().out
    assert "ghp_sensitiveLookingValue" not in err
    assert "did you paste a token" in err


def test_valid_token_env_name_is_still_echoed(capsys):
    parse_mcp_servers_env(
        '[{"name": "x", "url": "https://x.example", "token_env": "DOCS2_TOKEN"}]',
        environ={},
    )
    assert "DOCS2_TOKEN" in capsys.readouterr().out
