"""OpenTelemetry tracing init for the agentic PR reviewer.

Env-driven, soft-fail. Wires the OTLP/gRPC exporter at
`OTEL_EXPORTER_OTLP_ENDPOINT` and registers the resulting
`TracerProvider` as the process global so pydantic-ai's
`Agent.instrument_all(True)` picks it up via `get_tracer_provider()`.

Standard opt-in shape: env var unset →
no-op (returns `None`), preserving back-compat for the local-dev path
and for the tests that construct `Agent` without a Tempo endpoint.
Any failure during init logs a warning and returns `None` — the
reviewer must never crash because telemetry is mis-wired.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid forcing the heavy SDK import at module load — the runtime
    # branch in `init_tracing()` is what loads it, conditional on the
    # endpoint env var being set. Keeping the type reference behind
    # TYPE_CHECKING preserves the cheap-import promise that lets the
    # back-compat (no-OTel) path stay fast.
    from opentelemetry.sdk.trace import TracerProvider


# Default service.name when the env var is unset. Matches the workflow
# name so traces in Tempo / Grafana drilldowns are findable without
# memorising a separate identifier.
DEFAULT_SERVICE_NAME = "cora"


def init_tracing() -> TracerProvider | None:
    """Initialise the global OTel tracer provider.

    Returns the configured `TracerProvider` so the caller can
    `provider.shutdown()` (or `force_flush()`) before process exit —
    short-lived runs may otherwise terminate before `BatchSpanProcessor`
    flushes pending spans.

    Returns `None` (no-op) when:
      * `OTEL_EXPORTER_OTLP_ENDPOINT` is unset — back-compat path for
        tests and local dev where Tempo isn't reachable.
      * Any exception fires during SDK setup — logged as a GHA warning,
        never raised. The reviewer keeps running with no traces.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return None

    try:
        # Local imports so the heavy SDK isn't dragged into the module-
        # load path for environments without OTel installed (tests,
        # local-dev, the ubuntu-latest fallback in the workflow).
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.environ.get(
            "OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME
        ).strip() or DEFAULT_SERVICE_NAME

        # service.version: GitHub provides GITHUB_SHA in the workflow
        # environment. Falls back to "unknown" off-CI so the attribute
        # is always set (saves a `if absent` branch in dashboards).
        # service.version: explicit `OTEL_SERVICE_VERSION` wins over the
        # CI-default `GITHUB_SHA` so operators can tag soak/eval runs
        # without colliding with the workflow-set SHA. Falls back to
        # "unknown" off-CI so the attribute is always set.
        service_version = (
            os.environ.get("OTEL_SERVICE_VERSION", "").strip()
            or os.environ.get("GITHUB_SHA", "").strip()
            or "unknown"
        )

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )

        # insecure=True: the collector's OTLP gRPC port is assumed to
        # be network-internal plain HTTP/2 (no TLS) — the common
        # in-cluster collector setup.
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Opt every Agent (this process) into instrumentation via the
        # global tracer provider just registered. Idempotent — safe to
        # call multiple times, last call wins.
        try:
            from pydantic_ai import Agent

            Agent.instrument_all(True)
        except Exception:  # noqa: BLE001
            # pydantic-ai not installed (e.g. tests for this helper
            # run in a venv without the framework). Tracer provider
            # is still useful for manual spans, so don't abort.
            pass

        return provider
    except Exception as exc:  # noqa: BLE001
        # Soft-fail: log via GHA warning channel + stderr, return None.
        # Caller treats None as "no flushing needed".
        print(
            f"::warning::otel init failed (endpoint={endpoint!r}): {exc}",
            file=sys.stderr,
        )
        return None
