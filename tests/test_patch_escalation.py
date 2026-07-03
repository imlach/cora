"""Mandatory T2 escalation on `propose_patch` verdicts.

Covers the pure-logic surface of `agent_review.patch_escalation`:
verdict classification, outcome composition (agree / disagree /
skipped), the kill switch, and the patch-verification prompt frame.
Also exercises the dispatcher's new `escalation_warning_*` kwargs by
calling `apply_inline_suggestions` / `apply_propose_patch` with
prefix args set and inspecting the rendered payload.

Test matrix mirrors the spec:

    1. Escalation NOT fired when T0 verdict has no propose_patch
       (back-compat — `escalation_enabled()` is independent of the
       parse step; absence-of-directive is exercised by the
       existing `test_triage_propose_patch.py` /
       `test_patch_dispatch_routing.py` suite).
    2. Escalation fired when propose_patch present (verdict
       classification + outcome composition exercised).
    3. T2 agree → no annotation, no label.
    4. T2 disagree → warning text + LABEL_DISAGREEMENT + flagged.
    5. T2 fail (per_call_timeout, agent-loop-errored, …) →
       LABEL_SKIPPED + softer annotation, not flagged.
    6. AGENT_REVIEW_PATCH_ESCALATION="false" → kill switch bypasses.

The end-to-end orchestrator wiring (call → label apply → dispatch
prefix → log line) is too heavyweight to unit-test directly here; the
shape of each piece is pinned individually and the orchestrator-level
trace is covered by the live reviewer + GHA `_iter_log` event stream
(logfmt-parseable).
"""
from __future__ import annotations




# ---------------------------------------------------------------- kill switch


def test_escalation_enabled_default_true(monkeypatch):
    """Default behaviour: env var unset → escalation enabled."""
    from cora.core.patch_escalation import (
        ESCALATION_ENV_VAR, escalation_enabled,
    )

    monkeypatch.delenv(ESCALATION_ENV_VAR, raising=False)
    assert escalation_enabled() is True


def test_escalation_enabled_explicit_true(monkeypatch):
    """`AGENT_REVIEW_PATCH_ESCALATION=true` → enabled."""
    from cora.core.patch_escalation import (
        ESCALATION_ENV_VAR, escalation_enabled,
    )

    monkeypatch.setenv(ESCALATION_ENV_VAR, "true")
    assert escalation_enabled() is True


def test_escalation_disabled_kill_switch(monkeypatch):
    """`AGENT_REVIEW_PATCH_ESCALATION=false` → kill switch bypasses
    escalation entirely. Patches apply on the pre-escalation path."""
    from cora.core.patch_escalation import (
        ESCALATION_ENV_VAR, escalation_enabled,
    )

    monkeypatch.setenv(ESCALATION_ENV_VAR, "false")
    assert escalation_enabled() is False

    # Case-insensitive — uppercase / mixed must also disable to avoid
    # the operator setting `False` (Python truthy) and being surprised.
    monkeypatch.setenv(ESCALATION_ENV_VAR, "FALSE")
    assert escalation_enabled() is False
    monkeypatch.setenv(ESCALATION_ENV_VAR, "False")
    assert escalation_enabled() is False


def test_escalation_other_values_default_to_enabled(monkeypatch):
    """Defensive: any value that isn't the explicit "false" string is
    treated as enabled. Avoids accidentally disabling on a typo like
    `AGENT_REVIEW_PATCH_ESCALATION=disable` — the operator gets a
    safer default."""
    from cora.core.patch_escalation import (
        ESCALATION_ENV_VAR, escalation_enabled,
    )

    for value in ("", "yes", "on", "1", "disable", "off"):
        monkeypatch.setenv(ESCALATION_ENV_VAR, value)
        assert escalation_enabled() is True, f"{value!r} should be enabled"


def test_escalation_config_threaded_value_wins_over_env(monkeypatch):
    """An explicit (cfg-threaded) argument bypasses the env entirely;
    None keeps the legacy env read — same cfg=None contract as
    `deep_review._thinking_extra_body`."""
    from cora.core.patch_escalation import (
        ESCALATION_ENV_VAR, escalation_enabled,
    )

    monkeypatch.setenv(ESCALATION_ENV_VAR, "false")
    assert escalation_enabled(True) is True
    assert escalation_enabled(None) is False

    monkeypatch.setenv(ESCALATION_ENV_VAR, "true")
    assert escalation_enabled(False) is False
    assert escalation_enabled(None) is True


# ---------------------------------------------------------------- classify_t2_verdict


def test_classify_t2_no_body_is_skipped():
    """T2 produced nothing (probe-failed, agent-loop-errored, …) →
    skipped (not implicit-disagree). Intent: escalate when possible,
    not gate patches on T2 availability."""
    from cora.core.patch_escalation import classify_t2_verdict

    verdict, reason = classify_t2_verdict(
        t2_body=None, t2_terminated_reason="probe-failed",
    )
    assert verdict == "skipped"
    assert reason == "probe-failed"


def test_classify_t2_no_body_no_reason_defaults_to_no_body():
    """When both body and termination reason are missing, the reason
    text falls back to a stable label so annotations don't render
    `unknown` or `None`."""
    from cora.core.patch_escalation import classify_t2_verdict

    verdict, reason = classify_t2_verdict(
        t2_body=None, t2_terminated_reason=None,
    )
    assert verdict == "skipped"
    assert reason == "no body"


def test_classify_t2_per_call_timeout_is_skipped():
    """`per_call_timeout` is a softer skip than agent-loop-errored —
    same softer annotation."""
    from cora.core.patch_escalation import classify_t2_verdict

    verdict, reason = classify_t2_verdict(
        t2_body=None, t2_terminated_reason="per_call_timeout",
    )
    assert verdict == "skipped"
    assert reason == "per_call_timeout"


def test_classify_t2_wall_time_is_skipped():
    """Wall-time bust → skipped. Matches the spec's `t2 itself fails`
    set: probe-failed / agent-loop-errored / per_call_timeout /
    wall_time."""
    from cora.core.patch_escalation import classify_t2_verdict

    verdict, reason = classify_t2_verdict(
        t2_body=None, t2_terminated_reason="wall_time",
    )
    assert verdict == "skipped"
    assert reason == "wall_time"


def test_classify_t2_no_verdict_parsed_is_skipped():
    """T2 produced a body but no `Verdict:` marker → skipped, not
    implicit-disagree. Avoids the failure mode where a leaked /
    malformed T2 body silently flags an otherwise-fine patch."""
    from cora.core.patch_escalation import classify_t2_verdict

    verdict, reason = classify_t2_verdict(
        t2_body="random text without a verdict marker",
        t2_terminated_reason=None,
    )
    assert verdict == "skipped"
    assert reason == "no verdict parsed"


def test_classify_t2_looks_good_is_agree():
    """T2 says `looks good` → agree (per patch-verifier framing:
    "T0's patch is a valid fix")."""
    from cora.core.patch_escalation import classify_t2_verdict

    body = "Verdict: 🟢 looks good\n\nT0's patch correctly addresses…"
    verdict, reason = classify_t2_verdict(
        t2_body=body, t2_terminated_reason=None,
    )
    assert verdict == "agree"
    assert reason is None


def test_classify_t2_needs_changes_is_disagree():
    """T2 says `needs changes` → disagree. Carries the verdict word
    forward so the annotation can quote it."""
    from cora.core.patch_escalation import classify_t2_verdict

    body = "Verdict: 🔴 needs changes\n\nT0's patch misdiagnoses the issue."
    verdict, reason = classify_t2_verdict(
        t2_body=body, t2_terminated_reason=None,
    )
    assert verdict == "disagree"
    assert reason == "needs changes"


def test_classify_t2_minor_is_agree_with_nits():
    """`minor` ("patch mostly right but has nits") is too soft to gate
    behind a hard "human review required" banner — classify as agree,
    but carry `t2_reason="minor"` forward so the composer can surface
    the nits in the verdict footer for visibility."""
    from cora.core.patch_escalation import classify_t2_verdict

    body = "Verdict: 🟡 minor\n\nT0's patch is mostly right but the new_string is off-by-one."
    verdict, reason = classify_t2_verdict(
        t2_body=body, t2_terminated_reason=None,
    )
    assert verdict == "agree"
    assert reason == "minor"


# ---------------------------------------------------------------- compose_escalation_outcome


def test_compose_agree_no_annotation_no_label():
    """T2 agree → no warning, no label, not flagged. The dispatcher's
    existing apply path proceeds unchanged."""
    from cora.core.patch_escalation import compose_escalation_outcome

    out = compose_escalation_outcome(
        t2_verdict="agree", t2_reason=None,
        t2_body="Verdict: 🟢 looks good\n", t2_model_alias="alt-reviewer",
    )
    assert out.verdict == "agree"
    assert out.warning_body is None
    assert out.warning_summary is None
    assert out.verdict_footer is None
    assert out.label is None
    assert out.flagged_for_human is False


def test_compose_disagree_hard_annotation():
    """T2 disagree → LABEL_DISAGREEMENT, prominent warning, flagged
    for human. Warning text must contain the model alias + verdict
    word + dissent summary so the human can act."""
    from cora.core.patch_escalation import (
        LABEL_DISAGREEMENT, compose_escalation_outcome,
    )

    body = (
        "Verdict: 🔴 needs changes\n\n"
        "T0's patch removes the wrong line — line 42 is the canary, "
        "not line 41.\n"
    )
    out = compose_escalation_outcome(
        t2_verdict="disagree", t2_reason="needs changes",
        t2_body=body, t2_model_alias="alt-reviewer",
    )
    assert out.verdict == "disagree"
    assert out.flagged_for_human is True
    assert out.label == LABEL_DISAGREEMENT

    # Body warning must carry the key signals.
    assert out.warning_body is not None
    assert "Escalation disagreement" in out.warning_body
    assert "alt-reviewer" in out.warning_body
    assert "needs changes" in out.warning_body
    assert "Human review required" in out.warning_body
    # Dissent gist (first non-verdict line) should be quoted.
    assert "line 42 is the canary" in out.warning_body

    # Summary warning is shorter — same load-bearing facts.
    assert out.warning_summary is not None
    assert "Escalation disagreement" in out.warning_summary
    assert "alt-reviewer" in out.warning_summary

    # Verdict footer (appended to the source-PR comment) tags the
    # outcome so the operator sees it inline. Wording is deliberately
    # neutral about label state — the label apply is soft-fail and the
    # banner above is the load-bearing signal if the label is absent.
    assert out.verdict_footer is not None
    assert "Human review required" in out.verdict_footer
    assert "needs changes" in out.verdict_footer


def test_compose_skipped_soft_annotation():
    """T2 failed (probe-failed / etc.) → LABEL_SKIPPED, terse single-
    line note across all surfaces (no "Human review required" banner),
    NOT flagged. Patch applies on T0's verdict alone."""
    from cora.core.patch_escalation import (
        LABEL_SKIPPED, compose_escalation_outcome,
    )

    out = compose_escalation_outcome(
        t2_verdict="skipped", t2_reason="probe-failed",
        t2_body=None, t2_model_alias="alt-reviewer",
    )
    assert out.verdict == "skipped"
    assert out.label == LABEL_SKIPPED
    assert out.flagged_for_human is False

    # All three surfaces share the same terse note — no separate body /
    # summary text to maintain. The label is what the dashboard /
    # human filter on; the note is just a short trail explaining why
    # no T2 verdict landed.
    assert out.warning_body == out.warning_summary == out.verdict_footer
    assert out.warning_body is not None
    assert "probe-failed" in out.warning_body
    assert "T2 escalation skipped" in out.warning_body
    assert "T0's verdict alone" in out.warning_body


def test_compose_agree_with_minor_nits_surfaces_footer_only():
    """`minor` verdict → agree (no banner / label / flag), but the
    verdict footer surfaces T2's nits as informational so the human
    can spot-check the patch even though it isn't blocking."""
    from cora.core.patch_escalation import compose_escalation_outcome

    body = (
        "Verdict: 🟡 minor\n\n"
        "Patch is mostly right but new_string is off-by-one — should "
        "increment the index by 2, not 1.\n"
    )
    out = compose_escalation_outcome(
        t2_verdict="agree", t2_reason="minor",
        t2_body=body, t2_model_alias="alt-reviewer",
    )
    # Agree shape — no label, no banner, no flag.
    assert out.verdict == "agree"
    assert out.label is None
    assert out.warning_body is None
    assert out.warning_summary is None
    assert out.flagged_for_human is False
    assert out.t2_reason == "minor"

    # But the verdict footer carries the nits gist for visibility.
    assert out.verdict_footer is not None
    assert "alt-reviewer" in out.verdict_footer
    assert "noted nits" in out.verdict_footer
    assert "did not block" in out.verdict_footer
    # The dissent summary should be quoted from T2's body.
    assert "off-by-one" in out.verdict_footer


def test_compose_skipped_handles_missing_reason():
    """Reason None falls back to 'unknown' so the rendered text is
    always a complete sentence."""
    from cora.core.patch_escalation import compose_escalation_outcome

    out = compose_escalation_outcome(
        t2_verdict="skipped", t2_reason=None,
        t2_body=None, t2_model_alias="alt-reviewer",
    )
    assert out.warning_body is not None
    assert "unknown" in out.warning_body


# ---------------------------------------------------------------- prompt frame


def test_build_patch_verification_prompt_includes_t0_verdict_verbatim():
    """The verifier prompt must embed T0's verdict body verbatim — T2
    can't judge the patch without seeing what T0 wants to change."""
    from cora.core.patch_escalation import build_patch_verification_prompt

    t0_body = (
        "Verdict: 🔴 needs changes\n\n"
        "Missing NetworkPolicy.\n\n"
        "```json propose_patch\n"
        '{"title": "add NP", "body": "…", "edits": [{"path": "k8s/apps/x/networkpolicy.yml", "old_string": "a", "new_string": "b"}]}\n'
        "```\n"
    )
    base_prompt = "Here's the diff: <diff>"
    out = build_patch_verification_prompt(
        base_initial_user_prompt=base_prompt,
        t0_verdict_body=t0_body,
    )

    # T0's body lands verbatim inside a code fence so propose_patch
    # JSON doesn't get re-parsed as T2's own directive.
    assert t0_body.rstrip() in out
    # The base prompt (diff + retrieval pre-pack) is preserved.
    assert base_prompt in out
    # Frame is verifier, not independent reviewer.
    assert "patch-verifier" in out.lower()
    assert "Do NOT emit your own `propose_patch`" in out


def test_build_patch_verification_prompt_uses_code_fence_for_t0():
    """T0's body sits in a code fence so any propose_patch JSON inside
    isn't re-parsed as T2's directive. Pins the fence so a future
    refactor can't accidentally drop it."""
    from cora.core.patch_escalation import build_patch_verification_prompt

    out = build_patch_verification_prompt(
        base_initial_user_prompt="diff",
        t0_verdict_body="Verdict: 🟡 minor\n",
    )
    # Quadruple-backtick fence to safely escape T0 bodies that
    # themselves contain triple-backtick code blocks (propose_patch
    # JSON, snippet excerpts, etc.).
    assert "````" in out


# ---------------------------------------------------------------- pr_number_from_url


def test_pr_number_from_url_extracts_tail():
    from cora.core.patch_escalation import pr_number_from_url

    assert pr_number_from_url("https://github.com/owner/repo/pull/1234") == "1234"
    assert pr_number_from_url("https://github.com/owner/repo/pulls/9999") == "9999"


def test_pr_number_from_url_handles_garbage():
    from cora.core.patch_escalation import pr_number_from_url

    assert pr_number_from_url(None) is None
    assert pr_number_from_url("") is None
    assert pr_number_from_url("not-a-url") is None
    # Tail isn't digits — return None rather than the garbage tail.
    assert pr_number_from_url("https://github.com/owner/repo/pull/abc") is None


# ---------------------------------------------------------------- log_escalation_outcome


def test_log_escalation_outcome_emits_structured_line():
    """Downstream log dashboards parse on the logfmt key=value shape.
    Pin the field names so a downstream rename gets caught in tests."""
    from cora.core.patch_escalation import log_escalation_outcome

    captured: list[str] = []
    log_escalation_outcome(
        pr_number="1234",
        t2_verdict="disagree",
        patch_kind="draft_pr",
        flagged_for_human=True,
        log=captured.append,
    )
    assert len(captured) == 1
    line = captured[0]
    assert "agent_review escalation_outcome" in line
    assert "pr_number=1234" in line
    assert "t2_verdict=disagree" in line
    assert "patch_kind=draft_pr" in line
    assert "flagged_for_human=true" in line


def test_log_escalation_outcome_handles_combined_patch_kind():
    """When the dispatcher fires inline + draft simultaneously, the
    patch_kind is joined with `+` so the log line carries the full
    shape — important for per-kind dashboard breakdowns."""
    from cora.core.patch_escalation import log_escalation_outcome

    captured: list[str] = []
    log_escalation_outcome(
        pr_number="42",
        t2_verdict="agree",
        patch_kind="inline+draft_pr",
        flagged_for_human=False,
        log=captured.append,
    )
    assert "patch_kind=inline+draft_pr" in captured[0]


# ---------------------------------------------------------------- patch_kind_from_dispatch_outcome


def test_patch_kind_from_outcome_inline_only():
    from cora.core.patch_escalation import patch_kind_from_dispatch_outcome

    assert patch_kind_from_dispatch_outcome({
        "inline_count": 2, "draft_count": 0, "source_branch_count": 0,
    }) == "inline"


def test_patch_kind_from_outcome_patch_pr_only():
    from cora.core.patch_escalation import patch_kind_from_dispatch_outcome

    assert patch_kind_from_dispatch_outcome({
        "inline_count": 0, "draft_count": 3,
    }) == "patch_pr"


def test_patch_kind_from_outcome_combined():
    """Multi-bucket directive (some inline, some out-of-hunk) — kinds
    join with `+`."""
    from cora.core.patch_escalation import patch_kind_from_dispatch_outcome

    assert patch_kind_from_dispatch_outcome({
        "inline_count": 1, "draft_count": 2,
    }) == "inline+patch_pr"


def test_patch_kind_from_outcome_none_when_no_artifact():
    """All counters zero (dispatch flopped) → `none`. Caller still
    emits the log line for observability."""
    from cora.core.patch_escalation import patch_kind_from_dispatch_outcome

    assert patch_kind_from_dispatch_outcome({
        "inline_count": 0, "draft_count": 0,
    }) == "none"


# ---------------------------------------------------------------- dispatcher prefix kwargs


def test_apply_inline_suggestions_prepends_summary_prefix(monkeypatch):
    """`summary_prefix` (carrying the T2-disagreement banner) is
    prepended above the default dispatcher summary in the rendered
    review body. Pinned so a future signature change doesn't silently
    drop the warning."""
    from cora.core import patch_dispatch

    captured_payload: dict = {}

    def fake_gh_api_with_token(args, gh_token, input_data=None):
        captured_payload["args"] = args
        captured_payload["payload"] = input_data
        return {"html_url": "https://github.com/owner/repo/pull/123#review-1"}

    monkeypatch.setattr(
        patch_dispatch, "_gh_api_with_token", fake_gh_api_with_token,
    )

    review_url, err = patch_dispatch.apply_inline_suggestions(
        repo="owner/repo", pr_number="123",
        head_sha="abc123",
        summary="_Proposed by cora._",
        comments=[{
            "path": "x.py", "line": 1, "side": "RIGHT",
            "body": "```suggestion\nfoo\n```",
        }],
        gh_token="ghs_appToken",
        summary_prefix="⚠️ **Escalation disagreement** — see above.",
    )
    assert err is None
    body = captured_payload["payload"]["body"]
    # Prefix lands first.
    assert body.startswith("⚠️ **Escalation disagreement**")
    # Original summary follows after a blank line.
    assert "_Proposed by cora._" in body
    assert body.index("Escalation disagreement") < body.index("Proposed by cora")


def test_apply_inline_suggestions_no_prefix_back_compat(monkeypatch):
    """`summary_prefix=None` (default / kill-switch path) leaves the
    body exactly as the pre-escalation behaviour. Back-compat pin."""
    from cora.core import patch_dispatch

    captured_payload: dict = {}

    def fake_gh_api_with_token(args, gh_token, input_data=None):
        captured_payload["payload"] = input_data
        return {"html_url": "https://example/review"}

    monkeypatch.setattr(
        patch_dispatch, "_gh_api_with_token", fake_gh_api_with_token,
    )

    review_url, err = patch_dispatch.apply_inline_suggestions(
        repo="owner/repo", pr_number="123",
        head_sha="abc123",
        summary="just the summary",
        comments=[{"path": "x.py", "line": 1, "side": "RIGHT", "body": "x"}],
        gh_token="ghs_appToken",
    )
    assert err is None
    assert captured_payload["payload"]["body"] == "just the summary"


def test_apply_propose_patch_prepends_body_prefix(monkeypatch):
    """`body_prefix` is prepended above the directive body in the
    draft PR body — the human reviewer sees the warning *before* the
    patch summary when they open the PR."""
    import base64
    from cora.core import patch_dispatch

    pr_payload: dict = {}

    def fake_gh_api_with_token(args, gh_token, input_data=None):
        # Sequence: GET ref → POST ref (create branch) → GET contents
        # → PUT contents → POST pulls (open PR).
        path = " ".join(args)
        if "/git/refs/heads/" in path:
            return {"object": {"sha": "basesha"}}
        if "/git/refs" in path and "POST" in path:
            return {"ref": "refs/heads/x"}
        if "/contents/" in path and "GET" in path:
            return {
                "type": "file",
                "content": base64.b64encode(b"old content").decode("ascii"),
                "sha": "filesha",
            }
        if "/contents/" in path and "PUT" in path:
            return {"commit": {"sha": "newcommitsha"}}
        if "/pulls" in path:
            pr_payload.update(input_data or {})
            return {"html_url": "https://github.com/owner/repo/pull/9999"}
        return {}

    monkeypatch.setattr(
        patch_dispatch, "_gh_api_with_token", fake_gh_api_with_token,
    )

    draft_url, err = patch_dispatch.apply_propose_patch(
        repo="owner/repo", pr_number="123", base_ref="main",
        directive={
            "title": "fix x",
            "body": "Patch body explaining the fix.",
            "edits": [{
                "path": "x.py",
                "old_string": "old content",
                "new_string": "new content",
            }],
        },
        gh_token="ghs_appToken",
        body_prefix="⚠️ **Escalation disagreement** — human review required.",
    )
    assert err is None
    assert draft_url == "https://github.com/owner/repo/pull/9999"
    body = pr_payload["body"]
    # Prefix lands above the directive body.
    assert body.startswith("⚠️ **Escalation disagreement**")
    assert "Patch body explaining the fix." in body
    assert body.index("Escalation disagreement") < body.index("Patch body")
