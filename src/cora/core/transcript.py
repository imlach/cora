"""Serialize an agent-loop run into a trainable multi-turn trajectory.

The reviewer's `deep_review_call` already returns the full pydantic-ai
message history (`agent_run.all_messages()`) as its 4th value. This
module turns that history into the HF/OpenAI chat shape TRL's
`SFTTrainer` consumes — **with** assistant `tool_calls` and `tool`-role
result turns — so the fine-tune learns *when/which tool to call*, not
just diff→review text. That tool-trajectory signal is exactly what a
single-turn diff→review corpus cannot teach.

Two consumers, one serializer:
  - **live capture** — live reviews dump their trajectory for
    later filtering on real outcomes.
  - **teacher trajectories** — a stronger model run through the same
    loop with a palette-encouraging prompt; the broadening corpus.

Duck-typed on each part's `part_kind` discriminator (the stable literal
pydantic-ai stamps on every message part) so this module needs **no**
`pydantic_ai` import — it stays importable in the corpus-builder env
(which doesn't install pydantic-ai) and unit-testable with plain
stand-in objects. Mirrors `loop_logging.py`'s getattr-on-parts style.

Output row shape (one trajectory):

```json
{"messages": [
  {"role": "system", "content": "…"},
  {"role": "user", "content": "<diff + task>"},
  {"role": "assistant", "content": "",
   "tool_calls": [{"id": "tc1", "type": "function",
                   "function": {"name": "grep_repo",
                                "arguments": {"pattern": …}}}]},
  {"role": "tool", "tool_call_id": "tc1", "name": "grep_repo",
   "content": "{…}"},
  {"role": "assistant", "content": "<final markdown review>"}
], "_meta": {"source": "trajectory-live", "pr_number": 1234, …}}
```
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# part_kind → our handling. pydantic-ai stamps every ModelRequest /
# ModelResponse part with one of these literals. Anything not listed
# (future part kinds) is ignored rather than crashing the dump.
_REQUEST_TEXT_KINDS = {"system-prompt": "system", "user-prompt": "user"}


def _part_kind(part: Any) -> str:
    """The part's `part_kind` literal, with a class-name fallback for
    stand-ins / future renames."""
    kind = getattr(part, "part_kind", None)
    if kind:
        return str(kind)
    # Fallback: map the class name (e.g. ToolCallPart → tool-call).
    name = type(part).__name__
    mapping = {
        "SystemPromptPart": "system-prompt",
        "UserPromptPart": "user-prompt",
        "TextPart": "text",
        "ThinkingPart": "thinking",
        "ToolCallPart": "tool-call",
        "ToolReturnPart": "tool-return",
        "RetryPromptPart": "retry-prompt",
    }
    return mapping.get(name, name)


def _stringify(content: Any) -> str:
    """Render a part's content as a string. Tool results are usually
    already JSON strings (the local grep_repo/git_show handlers and most
    MCP tools), but an MCP server can return structured objects —
    JSON-encode those so the row stays a flat chat message."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _args_to_obj(args: Any) -> dict:
    """Tool-call arguments → dict (a mapping, NOT a JSON string).

    These rows feed an HF chat template via TRL's SFTTrainer, and some
    *training* templates render `function.arguments` with a `| items`
    filter — a string raises "Can only get item pairs from a mapping" and
    fails tokenization (`datasets` loads heterogeneous arg dicts fine). The
    OpenAI on-the-wire shape stores this as a string, but the trainer needs
    the object. pydantic-ai gives a dict for most models; occasionally a
    pre-encoded JSON string, which we parse back."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {"value": obj}
        except (TypeError, ValueError):
            return {"value": args}
    return {"value": args}


def trajectory_to_messages(
    pa_messages: Iterable[Any], *, include_thinking: bool = False
) -> list[dict]:
    """Convert a pydantic-ai `all_messages()` list into the HF/OpenAI
    chat shape (assistant `tool_calls` + `tool`-role results).

    `include_thinking=False` (default) drops `ThinkingPart` content: the
    served reviewer strips reasoning leaks before posting, and training
    on the verbose `<think>` trace is both heavy and off-target — we
    want the model to learn the *action* (which tool, then the review),
    not to reproduce a specific reasoning monologue.
    """
    out: list[dict] = []
    for msg in pa_messages:
        asst_text: list[str] = []
        asst_thinking: list[str] = []
        tool_calls: list[dict] = []
        for part in list(getattr(msg, "parts", []) or []):
            kind = _part_kind(part)
            if kind in _REQUEST_TEXT_KINDS:
                out.append(
                    {
                        "role": _REQUEST_TEXT_KINDS[kind],
                        "content": _stringify(getattr(part, "content", "")),
                    }
                )
            elif kind in ("tool-return", "retry-prompt"):
                # retry-prompt carries a tool validation/error retry; it
                # only belongs in the transcript as a tool result when it
                # references a specific call. A bare retry-prompt (no
                # tool_call_id) is a framework reprompt — skip it.
                tcid = getattr(part, "tool_call_id", None)
                if not tcid:
                    continue
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tcid,
                        "name": getattr(part, "tool_name", "") or "",
                        "content": _stringify(getattr(part, "content", "")),
                    }
                )
            elif kind == "text":
                asst_text.append(getattr(part, "content", "") or "")
            elif kind == "thinking":
                asst_thinking.append(getattr(part, "content", "") or "")
            elif kind == "tool-call":
                tool_calls.append(
                    {
                        "id": getattr(part, "tool_call_id", "") or "",
                        "type": "function",
                        "function": {
                            "name": getattr(part, "tool_name", "") or "",
                            "arguments": _args_to_obj(getattr(part, "args", None)),
                        },
                    }
                )
            # Unknown kinds: ignored on purpose.

        if asst_text or tool_calls or (include_thinking and asst_thinking):
            content = "".join(asst_text)
            if include_thinking and asst_thinking:
                content = "".join(asst_thinking) + content
            assistant: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
    return out


def count_tool_calls(messages: list[dict]) -> int:
    """Total assistant tool calls across a chat-format messages list."""
    return sum(len(m.get("tool_calls") or []) for m in messages)


def distinct_tools(messages: list[dict]) -> set[str]:
    """Set of distinct tool names called across the trajectory — the
    palette-diversity signal the tool-use eval keys off."""
    names: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn:
                names.add(fn)
    return names


def build_trajectory_row(
    messages: list[dict],
    *,
    meta: dict | None = None,
    require_tool_use: bool = True,
    require_final_text: bool = True,
) -> dict | None:
    """Validate + wrap a chat-format messages list into a trainable
    corpus row. Pure (no pydantic-ai) so the corpus builder can re-use
    it on ingested JSONL. Returns None when the trajectory fails the
    criteria:

    - `require_tool_use` — drop trajectories with no tool call at all
      (those are single-turn diff→review, already covered by a
      PR-history mining corpus; the point of *this* corpus is the tool turns).
    - `require_final_text` — drop trajectories whose last turn isn't a
      non-empty assistant text turn (i.e. terminated mid-loop on
      wall-time / iteration cap before producing a review — a truncated
      trajectory teaches the model to stop without concluding).
    """
    if not messages:
        return None
    if require_tool_use and count_tool_calls(messages) == 0:
        return None
    if require_final_text:
        last = messages[-1]
        if (
            last.get("role") != "assistant"
            or last.get("tool_calls")
            or not (last.get("content") or "").strip()
        ):
            return None
    row: dict = {"messages": messages}
    if meta:
        row["_meta"] = meta
    return row


def serialize_trajectory(
    pa_messages: Iterable[Any],
    *,
    meta: dict | None = None,
    include_thinking: bool = False,
    require_tool_use: bool = True,
    require_final_text: bool = True,
) -> dict | None:
    """Convenience: pydantic-ai history → validated trainable row (or
    None). Convert with `trajectory_to_messages`, then gate with
    `build_trajectory_row`."""
    messages = trajectory_to_messages(pa_messages, include_thinking=include_thinking)
    return build_trajectory_row(
        messages,
        meta=meta,
        require_tool_use=require_tool_use,
        require_final_text=require_final_text,
    )
