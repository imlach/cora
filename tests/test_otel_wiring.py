"""Smoke tests for `agent_review.otel.init_tracing`.

Covers the back-compat contract (no endpoint → None), the happy path
(endpoint set → real `TracerProvider`), the service-name override, and
the soft-fail guarantee on a malformed endpoint. The reviewer must
never crash because telemetry is mis-wired — that contract is what
keeps `agent_review.py` exit-0-on-failure end to end.
"""

from __future__ import annotations

import importlib

import pytest


# The OTel SDK + OTLP/gRPC exporter ship via the optional `otel`
# extra. The local test venv may or may not have them — skip the
# whole module rather than fail-skip per-test when they're absent.
pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")


# `init_tracing` reads env at call time and (on success) mutates global
# OTel state via `trace.set_tracer_provider`. Auto-cleared env keeps
# tests independent of the shell that ran pytest.
@pytest.fixture(autouse=True)
def _clear_otel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_SERVICE_VERSION",
        "GITHUB_SHA",
    ):
        monkeypatch.delenv(var, raising=False)


def _import_module():
    # Force a fresh import each test so the module's local OTel imports
    # are re-evaluated under the current env. Cheap (the SDK is already
    # loaded once after the importorskip above).
    from cora.core import otel as _otel

    return importlib.reload(_otel)


def test_returns_none_when_endpoint_unset() -> None:
    """Back-compat path — local dev / tests run without an OTLP endpoint."""
    otel = _import_module()
    assert otel.init_tracing() is None


def test_returns_real_tracer_provider_when_endpoint_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint set → real SDK `TracerProvider`, not the no-op default."""
    from opentelemetry.sdk.trace import TracerProvider

    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    otel = _import_module()
    provider = otel.init_tracing()
    try:
        assert provider is not None
        # The default global tracer provider is a no-op `ProxyTracerProvider`
        # — the SDK `TracerProvider` is what actually exports spans.
        assert isinstance(provider, TracerProvider)
    finally:
        if provider is not None:
            provider.shutdown()


def test_service_name_defaults_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default service.name = `cora` when the env override is absent."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    otel = _import_module()
    provider = otel.init_tracing()
    try:
        assert provider is not None
        attrs = provider.resource.attributes
        assert attrs.get("service.name") == "cora"
    finally:
        provider.shutdown()


def test_service_name_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OTEL_SERVICE_NAME` env override wins over the default."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "agent-review-custom")
    otel = _import_module()
    provider = otel.init_tracing()
    try:
        assert provider is not None
        assert provider.resource.attributes.get("service.name") == (
            "agent-review-custom"
        )
    finally:
        provider.shutdown()


def test_service_version_picks_up_github_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GITHUB_SHA` becomes `service.version` for CI provenance in the trace backend
    when no explicit `OTEL_SERVICE_VERSION` override is set."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    monkeypatch.setenv("GITHUB_SHA", "abc1234deadbeef")
    otel = _import_module()
    provider = otel.init_tracing()
    try:
        assert provider is not None
        assert provider.resource.attributes.get("service.version") == (
            "abc1234deadbeef"
        )
    finally:
        provider.shutdown()


def test_service_version_explicit_env_wins_over_github_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `OTEL_SERVICE_VERSION` overrides `GITHUB_SHA` — lets
    operators tag soak / eval runs without colliding with the
    workflow-set SHA."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    monkeypatch.setenv("GITHUB_SHA", "abc1234deadbeef")
    monkeypatch.setenv("OTEL_SERVICE_VERSION", "soak-2026-05-27")
    otel = _import_module()
    provider = otel.init_tracing()
    try:
        assert provider is not None
        assert provider.resource.attributes.get("service.version") == (
            "soak-2026-05-27"
        )
    finally:
        provider.shutdown()


def test_soft_fails_on_malformed_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Init must never raise — even on garbage endpoint values.

    The OTLP exporter constructor is permissive (DNS / TCP resolution
    happens lazily on the first export). We exercise the contract via
    a forced failure injected through the exporter constructor.
    """
    otel = _import_module()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )

    # Patch the exporter class on the otel module's bound import path —
    # `init_tracing()` does a local import inside the function, so we
    # need to patch where the import resolves at call time. Easiest path:
    # replace the symbol on the source module.
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as exporter_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated exporter failure")

    monkeypatch.setattr(exporter_mod, "OTLPSpanExporter", _boom)

    # Must not raise.
    result = otel.init_tracing()
    assert result is None
    captured = capsys.readouterr()
    assert "otel init failed" in captured.err
