"""Tiny logging helper shared across the agent_review entrypoint.

`_gha_log` is used as a per-event log callback (pydantic-ai
`event_stream_handler` lines, MCP session-open notices, etc.) and as
a plain stdout printer outside GHA. GHA picks up `::notice::` lines
as workflow annotations.
"""

from __future__ import annotations


def _gha_log(msg: str) -> None:
    """Loop-loop callback. GHA picks up `::notice::` lines as workflow
    annotations; outside GHA it's just a regular log line."""
    print(f"::notice::{msg}")
