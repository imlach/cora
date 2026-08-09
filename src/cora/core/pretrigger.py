"""Cold-start pretrigger — fire a best-effort warmup at a scale-from-zero
review endpoint so its restore overlaps the reviewer's own retrieval/setup
instead of landing on the critical path of the first review call.

When the review alias is served by an autoscaled (scale-to-zero) backend,
firing a tiny warmup at startup — keyed on the same `pr-<n>` tag the real
review uses — triggers the backend's activation early, so the endpoint is
warm by the time the real call lands a minute or two later. Best-effort:
any failure just means the review pays the cold-start it would have paid
anyway, so this can never make things worse.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request

# Which review aliases warm — a `ReviewerConfig` field
# (`pretrigger_warmup_models`), threaded in by the `run_review` call site.
# Re-exported here from `core.config` for direct callers that pass no
# `warmup_models`; the default set is empty, so the pretrigger stays
# disarmed until a deployment lists its scale-from-zero aliases.
from cora.core.config import PRETRIGGER_WARMUP_MODELS

DEFAULT_TIMEOUT_S = 15.0
TIMEOUT_ENV = "CORA_PRETRIGGER_TIMEOUT_S"


def _should_fire(
    model: str, pr_number: str, warmup_models: frozenset[str] | None = None
) -> bool:
    models = PRETRIGGER_WARMUP_MODELS if warmup_models is None else warmup_models
    return bool(pr_number) and model in models


def _timeout_s() -> float:
    try:
        return max(1.0, float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _post(url: str, api_key: str, model: str, pr_number: str, timeout_s: float) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"warmup pr-{pr_number}"}],
            "max_tokens": 1,
            "cache": {"no-cache": True},
            # pr-<n> drives the HRW hook to the card that will serve this review;
            # the `prewarm` tag lets the dashboard separate this activation from
            # the real review's first-token latency.
            "metadata": {"tags": [f"pr-{pr_number}", "prewarm"]},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        # Default 15s is enough to register with KEDA. The classifier workflow
        # can opt into waiting for the tiny request to actually reach the
        # restored backend, hiding first-request JIT behind classifier work.
        urllib.request.urlopen(req, timeout=timeout_s)
    except Exception:  # noqa: BLE001 — best-effort warmup
        pass  # never raise into the review path


async def fire_pretrigger(
    base_url: str,
    api_key: str,
    model: str,
    pr_number: str,
    warmup_models: frozenset[str] | None = None,
) -> None:
    """Fire-and-forget warmup. No-op unless `model` is in `warmup_models`
    (defaults to the engine constant when None); otherwise triggers the
    backend activation in a worker thread so it overlaps retrieval."""
    if not _should_fire(model, pr_number, warmup_models):
        return
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    await asyncio.to_thread(_post, url, api_key, model, pr_number, _timeout_s())
