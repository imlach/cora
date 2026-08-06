"""Per-run comment loop (cora#29): `create_progress_comment` /
`update_run_comment` / `minimize_superseded_comments`.

`_gh_with_one_retry` is the seam every gh subprocess call goes through,
so it's the one thing these tests mock — `_FakeGhStore` stands in for
the PR's comment thread (create / list / PATCH / graphql minimize) and
lets the same in-memory state be driven across several calls, the way
a real PR accumulates comments across review runs. Fully offline: no
network, no real `gh` invocation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cora.core import comment as comment_mod
from cora.core.config import (
    COMMENT_MARKER,
    LEGACY_COMMENT_MARKERS,
    PROGRESS_MARKER_PREFIX,
    VERDICT_MARKER_PREFIX,
)


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _is_cora_body(body: str) -> bool:
    """Python-side mirror of the jq predicate `minimize_superseded_comments`
    builds — used by the fake store to decide which comments a listing
    call should return, independent of the jq text itself."""
    return body.startswith(
        (COMMENT_MARKER, *LEGACY_COMMENT_MARKERS, PROGRESS_MARKER_PREFIX, VERDICT_MARKER_PREFIX)
    )


def _is_own_body(body: str) -> bool:
    run_id = comment_mod._run_id()
    return body.startswith((comment_mod._progress_marker(run_id), comment_mod._verdict_marker(run_id)))


class _FakeGhStore:
    """In-memory stand-in for a PR's comment thread. `handle` replaces
    `comment_mod._gh_with_one_retry` and dispatches on the shape of the
    `gh` command comment.py builds, exactly like the real CLI would
    route to create/list/PATCH/graphql endpoints.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self.comments: dict[int, dict] = {}
        self.calls: list[list[str]] = []
        self.minimized: list[str] = []
        self.graphql_fail_ids: set[str] = set()
        self.graphql_error_ids: set[str] = set()

    def add(self, body: str) -> dict:
        cid = self._next_id
        self._next_id += 1
        c = {"id": cid, "node_id": f"node_{cid}", "body": body}
        self.comments[cid] = c
        return c

    def handle(self, cmd, env, *, input_=None, retry_sleep_s=1.5):
        self.calls.append(cmd)

        if cmd[:3] == ["gh", "pr", "comment"]:
            body = cmd[cmd.index("--body") + 1]
            self.add(body)
            return _proc(0)

        if cmd[:2] == ["gh", "api"] and cmd[2] == "graphql":
            id_arg = next(a for a in cmd if a.startswith("id="))
            node_id = id_arg.split("=", 1)[1]
            self.minimized.append(node_id)
            if node_id in self.graphql_fail_ids:
                return _proc(1, stderr="HTTP 403 forbidden")
            if node_id in self.graphql_error_ids:
                return _proc(
                    0,
                    stdout=json.dumps(
                        {"errors": [{"message": "already minimized or not authorized"}]}
                    ),
                )
            return _proc(
                0,
                stdout=json.dumps(
                    {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
                ),
            )

        if cmd[:2] == ["gh", "api"] and "--jq" in cmd:
            jq_expr = cmd[cmd.index("--jq") + 1]
            if "node_id: .node_id" in jq_expr:
                # minimize_superseded_comments' broad listing.
                targets = [
                    {"id": c["id"], "node_id": c["node_id"]}
                    for c in self.comments.values()
                    if _is_cora_body(c["body"]) and not _is_own_body(c["body"])
                ]
                return _proc(0, stdout=json.dumps(targets))
            # update_run_comment's "find my own run's comment" listing.
            matches = [c for c in self.comments.values() if _is_own_body(c["body"])]
            return _proc(0, stdout=str(matches[0]["id"]) if matches else "")

        if cmd[:3] == ["gh", "api", "-X"] and cmd[3] == "PATCH":
            comment_id = int(cmd[4].rsplit("/", 1)[-1])
            self.comments[comment_id]["body"] = json.loads(input_)["body"]
            return _proc(0)

        raise AssertionError(f"unexpected gh command: {cmd}")


# ── marker round-trip ───────────────────────────────────────────────


def test_marker_round_trip(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    assert comment_mod._run_id() == "42"
    progress = comment_mod._progress_marker("42")
    verdict = comment_mod._verdict_marker("42")
    assert progress == "<!-- cora:progress:42 -->"
    assert verdict == "<!-- cora:verdict:42 -->"
    assert progress != verdict
    # A different run id never collides with this one.
    assert comment_mod._progress_marker("43") not in (progress, verdict)


def test_run_id_falls_back_to_local_outside_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert comment_mod._run_id() == "local"


# ── two runs: second live, first minimized ────────────────────────────


def test_two_runs_second_live_first_minimized(monkeypatch):
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)

    monkeypatch.setenv("GITHUB_RUN_ID", "111")
    comment_mod.create_progress_comment("o/r", "7", "reviewing...")
    comment_mod.update_run_comment("o/r", "7", "verdict A", final=True)
    comment_mod.minimize_superseded_comments("o/r", "7")  # nothing to collapse yet

    monkeypatch.setenv("GITHUB_RUN_ID", "222")
    comment_mod.update_run_comment("o/r", "7", "verdict B", final=True)
    comment_mod.minimize_superseded_comments("o/r", "7")

    assert len(store.comments) == 2
    run_a = next(c for c in store.comments.values() if "verdict A" in c["body"])
    run_b = next(c for c in store.comments.values() if "verdict B" in c["body"])
    assert run_a["node_id"] in store.minimized
    assert run_b["node_id"] not in store.minimized
    assert run_b["body"].startswith(comment_mod._verdict_marker("222"))


# ── minimize soft-fails ────────────────────────────────────────────────


def test_minimize_failure_does_not_raise(monkeypatch, capsys):
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)

    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    comment_mod.update_run_comment("o/r", "7", "verdict 1 (rc failure)", final=True)
    monkeypatch.setenv("GITHUB_RUN_ID", "2")
    comment_mod.update_run_comment("o/r", "7", "verdict 2 (errors payload)", final=True)
    monkeypatch.setenv("GITHUB_RUN_ID", "3")
    comment_mod.update_run_comment("o/r", "7", "verdict 3 (current)", final=True)

    run1 = next(c for c in store.comments.values() if "verdict 1" in c["body"])
    run2 = next(c for c in store.comments.values() if "verdict 2" in c["body"])
    store.graphql_fail_ids.add(run1["node_id"])       # non-zero rc
    store.graphql_error_ids.add(run2["node_id"])      # rc=0, GraphQL `errors` array

    # Must not raise — minimisation is cosmetic and soft-fails per comment.
    comment_mod.minimize_superseded_comments("o/r", "7")

    out = capsys.readouterr().out
    assert out.count("::warning::") == 2
    assert f"minimize comment {run1['node_id']}" in out
    assert f"minimize comment {run2['node_id']}" in out
    # Both were still attempted despite failing.
    assert run1["node_id"] in store.minimized
    assert run2["node_id"] in store.minimized


def test_minimize_listing_failure_does_not_raise(monkeypatch, capsys):
    def fail_list(cmd, env, *, input_=None, retry_sleep_s=1.5):
        return _proc(1, stderr="HTTP 500 server error")

    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", fail_list)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    comment_mod.minimize_superseded_comments("o/r", "7")  # must not raise
    assert "::warning::" in capsys.readouterr().out


# ── legacy comment is collapsed, never adopted ─────────────────────────


def test_legacy_comment_minimized_not_adopted(monkeypatch):
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)
    legacy = store.add(f"{LEGACY_COMMENT_MARKERS[0]}\nold single-shot verdict")

    monkeypatch.setenv("GITHUB_RUN_ID", "9")
    comment_mod.update_run_comment("o/r", "7", "fresh per-run verdict", final=True)

    # Not adopted: a brand-new comment was created, the legacy one
    # untouched by the PATCH path.
    assert len(store.comments) == 2
    assert legacy["body"] == f"{LEGACY_COMMENT_MARKERS[0]}\nold single-shot verdict"

    comment_mod.minimize_superseded_comments("o/r", "7")
    assert legacy["node_id"] in store.minimized


# ── GraphQL uses node_id, not the REST id ───────────────────────────────


def test_minimize_uses_node_id_not_rest_id(monkeypatch):
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)

    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    comment_mod.update_run_comment("o/r", "7", "verdict 1", final=True)
    old = next(iter(store.comments.values()))

    monkeypatch.setenv("GITHUB_RUN_ID", "2")
    comment_mod.update_run_comment("o/r", "7", "verdict 2", final=True)
    comment_mod.minimize_superseded_comments("o/r", "7")

    # The mutation was called with the node_id form ("node_<n>"), never
    # the bare REST id — a `str(old["id"])` collision would be the
    # silent-failure shape the issue calls out.
    assert store.minimized == [old["node_id"]]
    assert str(old["id"]) not in store.minimized

    # The listing jq itself must select .node_id alongside .id, not
    # just .id — the field GraphQL's subjectId actually needs.
    minimize_calls = [c for c in store.calls if "--jq" in c and "graphql" not in c]
    jq_exprs = [c[c.index("--jq") + 1] for c in minimize_calls]
    assert any("node_id: .node_id" in expr for expr in jq_exprs)


# ── progress → verdict swap edits, never duplicates ─────────────────────


def test_progress_to_verdict_swap_edits_not_duplicates(monkeypatch):
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)
    monkeypatch.setenv("GITHUB_RUN_ID", "5")

    comment_mod.create_progress_comment("o/r", "7", "reviewing...")
    assert len(store.comments) == 1

    comment_mod.update_run_comment("o/r", "7", "mid-run update", final=False)
    assert len(store.comments) == 1  # PATCHed the placeholder, not duplicated

    comment_mod.update_run_comment("o/r", "7", "the verdict", final=True)
    assert len(store.comments) == 1  # still one comment for this run

    only = next(iter(store.comments.values()))
    assert only["body"].startswith(comment_mod._verdict_marker("5"))
    assert "the verdict" in only["body"]


# ── quick-mode / skip path: no placeholder, update_run_comment creates ──


def test_update_run_comment_creates_when_no_placeholder_exists(monkeypatch):
    # Quick mode / early-exit skips never call create_progress_comment.
    store = _FakeGhStore()
    monkeypatch.setattr(comment_mod, "_gh_with_one_retry", store.handle)
    monkeypatch.setenv("GITHUB_RUN_ID", "77")

    comment_mod.update_run_comment("o/r", "7", "skip: no key configured", final=True)

    assert len(store.comments) == 1
    only = next(iter(store.comments.values()))
    assert only["body"].startswith(comment_mod._verdict_marker("77"))


# ── posting-call failure contract: still raises (only minimize soft-fails) ──


def test_create_progress_comment_raises_on_failure(monkeypatch):
    monkeypatch.setattr(
        comment_mod, "_gh_with_one_retry", lambda *a, **k: _proc(1, stderr="HTTP 500")
    )
    with pytest.raises(RuntimeError, match="create comment failed"):
        comment_mod.create_progress_comment("o/r", "7", "body")


def test_update_run_comment_raises_when_listing_fails(monkeypatch):
    monkeypatch.setattr(
        comment_mod, "_gh_with_one_retry", lambda *a, **k: _proc(1, stderr="HTTP 500")
    )
    with pytest.raises(RuntimeError, match="list comments failed"):
        comment_mod.update_run_comment("o/r", "7", "body", final=True)


# ── token threading (matches the rest of the module) ────────────────────


def test_cora_env_prefers_app_token(monkeypatch):
    monkeypatch.setenv("CORA_GH_TOKEN", "app-tok")
    assert comment_mod._cora_env()["GH_TOKEN"] == "app-tok"


def test_cora_env_falls_back_without_app_token(monkeypatch):
    monkeypatch.delenv("CORA_GH_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert "GH_TOKEN" not in comment_mod._cora_env()
