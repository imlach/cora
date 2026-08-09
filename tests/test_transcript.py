"""Unit tests for agent_review.transcript — the trajectory serializer.

Uses plain SimpleNamespace stand-ins for pydantic-ai message parts (the
serializer duck-types on `part_kind`), so these run without pydantic-ai
installed — same constraint the corpus-builder env has.
"""

from __future__ import annotations

from types import SimpleNamespace

from cora.core.transcript import (
    build_trajectory_row,
    count_tool_calls,
    distinct_tools,
    serialize_trajectory,
    trajectory_to_messages,
)

# --- stand-in builders (mirror pydantic-ai's part shapes) ------------------

def _req(*parts):
    return SimpleNamespace(kind="request", parts=list(parts))


def _resp(*parts):
    return SimpleNamespace(kind="response", parts=list(parts))


def _system(content):
    return SimpleNamespace(part_kind="system-prompt", content=content)


def _user(content):
    return SimpleNamespace(part_kind="user-prompt", content=content)


def _text(content):
    return SimpleNamespace(part_kind="text", content=content)


def _thinking(content):
    return SimpleNamespace(part_kind="thinking", content=content)


def _tool_call(name, args, tcid):
    return SimpleNamespace(
        part_kind="tool-call", tool_name=name, args=args, tool_call_id=tcid
    )


def _tool_return(name, content, tcid):
    return SimpleNamespace(
        part_kind="tool-return", tool_name=name, content=content, tool_call_id=tcid
    )


def _retry(content, tcid=None, name=""):
    return SimpleNamespace(
        part_kind="retry-prompt", content=content, tool_call_id=tcid, tool_name=name
    )


def _full_trajectory():
    """system+user → assistant(grep_repo) → tool → assistant(read_decision)
    → tool → assistant(final review). Two tool calls, two tools."""
    return [
        _req(_system("SYS"), _user("PR #1: diff…")),
        _resp(
            _thinking("let me check the file"),
            _tool_call("grep_repo", {"pattern": "gpumem"}, "tc1"),
        ),
        _req(_tool_return("grep_repo", '{"matches": []}', "tc1")),
        _resp(_tool_call("read_decision", {"id": "DEC-001"}, "tc2")),
        _req(_tool_return("read_decision", {"status": "accepted"}, "tc2")),
        _resp(_text("🟢 lgtm — verified against DEC-001.")),
    ]


# --- trajectory_to_messages ------------------------------------------------

def test_full_trajectory_shape():
    msgs = trajectory_to_messages(_full_trajectory())
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]

    # system + user passthrough
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "PR #1: diff…"}

    # first assistant turn: thinking dropped by default, one tool_call
    a1 = msgs[2]
    assert a1["content"] == ""
    assert a1["tool_calls"] == [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "grep_repo", "arguments": {"pattern": "gpumem"}},
        }
    ]

    # tool result carries id + name + stringified content
    assert msgs[3] == {
        "role": "tool",
        "tool_call_id": "tc1",
        "name": "grep_repo",
        "content": '{"matches": []}',
    }

    # dict tool-return content is JSON-stringified
    assert msgs[5]["content"] == '{"status": "accepted"}'

    # final assistant turn is plain text, no tool_calls
    assert msgs[-1] == {"role": "assistant", "content": "🟢 lgtm — verified against DEC-001."}


def test_args_string_parsed_to_dict():
    # OpenAI-shape string args are parsed back to a mapping — some chat
    # templates render function.arguments via a `| items` filter, which
    # raises on a string.
    msgs = trajectory_to_messages(
        [_resp(_tool_call("git_show", '{"ref": "HEAD", "path": "x"}', "t"))]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"ref": "HEAD", "path": "x"}


def test_args_dict_passthrough():
    msgs = trajectory_to_messages(
        [_resp(_tool_call("git_show", {"ref": "HEAD", "path": "x"}, "t"))]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"ref": "HEAD", "path": "x"}


def test_include_thinking_prepends():
    traj = [_resp(_thinking("reason "), _text("verdict"))]
    assert trajectory_to_messages(traj)[0]["content"] == "verdict"
    inc = trajectory_to_messages(traj, include_thinking=True)[0]["content"]
    assert inc == "reason verdict"


def test_retry_prompt_without_tcid_skipped():
    # bare retry (framework reprompt) → no tool message
    msgs = trajectory_to_messages([_req(_retry("please retry"))])
    assert msgs == []
    # retry referencing a call → tool message
    msgs = trajectory_to_messages([_req(_retry("bad args", tcid="tc9", name="grep_repo"))])
    assert msgs == [
        {"role": "tool", "tool_call_id": "tc9", "name": "grep_repo", "content": "bad args"}
    ]


def test_classname_fallback_when_no_part_kind():
    # stand-in without part_kind → falls back to class name mapping.
    class ToolCallPart(SimpleNamespace):
        pass

    part = ToolCallPart(tool_name="grep_repo", args={"pattern": "x"}, tool_call_id="t")
    msgs = trajectory_to_messages([_resp(part)])
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "grep_repo"


# --- helpers ---------------------------------------------------------------

def test_count_and_distinct_tools():
    msgs = trajectory_to_messages(_full_trajectory())
    assert count_tool_calls(msgs) == 2
    assert distinct_tools(msgs) == {"grep_repo", "read_decision"}


# --- build_trajectory_row gating ------------------------------------------

def test_row_requires_tool_use():
    single_turn = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "🟢 lgtm"},
    ]
    assert build_trajectory_row(single_turn) is None
    assert build_trajectory_row(single_turn, require_tool_use=False) == {
        "messages": single_turn
    }


def test_row_requires_final_text():
    # ends on a tool result (terminated mid-loop) → rejected
    truncated = trajectory_to_messages(
        [
            _req(_system("S"), _user("U")),
            _resp(_tool_call("grep_repo", {"pattern": "x"}, "t")),
            _req(_tool_return("grep_repo", "{}", "t")),
        ]
    )
    assert build_trajectory_row(truncated) is None

    # ends on an assistant turn that is *only* a tool call → rejected
    dangling = [*truncated, {"role": "assistant", "content": "", "tool_calls": [{"id": "z"}]}]
    assert build_trajectory_row(dangling) is None


def test_row_attaches_meta():
    row = build_trajectory_row(
        trajectory_to_messages(_full_trajectory()),
        meta={"source": "trajectory-live", "pr_number": 1},
    )
    assert row is not None
    assert row["_meta"] == {"source": "trajectory-live", "pr_number": 1}
    assert row["messages"][-1]["content"].startswith("🟢")


def test_serialize_trajectory_end_to_end():
    row = serialize_trajectory(
        _full_trajectory(), meta={"source": "trajectory-teacher"}
    )
    assert row is not None
    assert count_tool_calls(row["messages"]) == 2
    assert row["_meta"]["source"] == "trajectory-teacher"

    # a no-tool run serializes to None under the default gate
    no_tools = [_req(_system("S"), _user("U")), _resp(_text("🟢 lgtm"))]
    assert serialize_trajectory(no_tools) is None
