"""Probe-failure logging must never leak header credentials.

The HTTP client embeds the offending header value verbatim in errors
like `Illegal header value b'…'`; observed live when a malformed
multi-line token was set — only GHA's secret masking kept it out of
the public log. `_redact` strips every header value (raw and escaped
bytes-repr forms) and caps the message length.
"""

from __future__ import annotations

import asyncio

from cora.core.mcp_probe import _EXC_LOG_CHAR_CAP, _redact, probe_mcp_server


def test_redact_strips_raw_header_value():
    out = _redact("Illegal header value sk-secret-123", {"Authorization": "sk-secret-123"})
    assert "sk-secret-123" not in out
    assert "[redacted]" in out


def test_redact_strips_escaped_bytes_repr_form():
    # Multi-line value as the client renders it: b'line1\nline2' with
    # the newline escaped in the message text.
    token = "line1\nline2"
    msg = "Illegal header value b'line1\\nline2'"
    out = _redact(msg, {"Authorization": token})
    assert "line1" not in out
    assert "[redacted]" in out


def test_redact_caps_message_length():
    out = _redact("x" * 10_000, {})
    assert len(out) <= _EXC_LOG_CHAR_CAP + len("…[truncated]")
    assert out.endswith("…[truncated]")


def test_redact_ignores_empty_values():
    assert _redact("boom", {"Authorization": ""}) == "boom"


def test_probe_failure_log_is_redacted(monkeypatch):
    """End-to-end: a probe exception embedding the header value logs
    with the value redacted."""
    token = "Bearer sk-live-token"

    class _BoomToolset:
        def __init__(self, url, headers=None):
            self._headers = headers

        async def __aenter__(self):
            raise RuntimeError(f"Illegal header value {token}")

        async def __aexit__(self, *a):
            return False

    import pydantic_ai.mcp as mcp_mod

    monkeypatch.setattr(mcp_mod, "MCPToolset", _BoomToolset)

    logs: list[str] = []
    ok = asyncio.run(
        probe_mcp_server(
            "http://mcp.test/mcp",
            {"Authorization": token},
            "mcp",
            logs.append,
        )
    )
    assert ok is False
    assert logs and "sk-live-token" not in logs[0]
    assert "[redacted]" in logs[0]
