"""Packaged generic prompts + the path-or-packaged-default loader, so
cora loads a system prompt with no external files (Phase 3, #3)."""

from __future__ import annotations

import pytest

from cora.config import ReviewerConfig
from cora.core.prompt import load_system_prompt

# The post-processor (leak detector + verdict→conclusion mapping) parses
# these exact strings — a packaged prompt that drifts them silently breaks
# every review. Pin them here so a prompt edit that drops one trips CI.
VERDICT_MARKERS = ("🟢 looks good", "🟡 minor", "🔴 needs changes")


@pytest.mark.parametrize("mode", ["deep", "quick"])
def test_packaged_prompt_loads_and_pins_verdict_markers(mode):
    text = load_system_prompt(None, mode=mode)
    assert text.strip()
    for marker in VERDICT_MARKERS:
        assert marker in text, f"{mode}.md is missing verdict marker {marker!r}"


def test_deep_prompt_describes_its_always_available_tools():
    text = load_system_prompt(None, mode="deep")
    assert "grep_repo" in text and "git_show" in text


def test_explicit_path_overrides_packaged_default(tmp_path):
    custom = tmp_path / "my_prompt.md"
    custom.write_text("CUSTOM PROMPT BODY", encoding="utf-8")
    assert load_system_prompt(custom, mode="deep") == "CUSTOM PROMPT BODY"


def test_missing_path_falls_back_to_packaged(tmp_path):
    text = load_system_prompt(tmp_path / "does-not-exist.md", mode="quick")
    assert "🟢 looks good" in text  # the packaged quick prompt


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown prompt mode"):
        load_system_prompt(None, mode="medium")


def test_config_prompt_paths_default_to_none():
    cfg = ReviewerConfig()
    assert cfg.deep_prompt_path is None
    assert cfg.quick_prompt_path is None


def test_deep_prompt_pins_unverified_finding_rules():
    """The two production-observed escape hatches around "Verify before
    you flag" — a blocker phrased as an unresolved conditional, and a
    recommendation the diff already implements — are called out
    explicitly. Pin the headline phrases so a prompt edit that drops
    them trips CI."""
    text = load_system_prompt(None, mode="deep")
    assert "A conditional is not a finding" in text
    assert "Check it isn't already there" in text
    assert "Validate any claim" in text
    assert "≤2 tool calls" not in text


@pytest.mark.parametrize("mode", ["deep", "quick"])
def test_prompts_forbid_version_nonexistence_blockers(mode):
    """cora #37: the reviewer blocked a PR asserting "Node 26 does not
    exist (latest LTS is 22)" while the PR's own job log showed 26.7.0
    installed, then re-asserted it verbatim on re-review.

    A version absent from the weights is the *expected* look of a
    release made after the cutoff, so this class can never be settled
    from memory. The existing library-API and version-boundary rules
    cover how a pinned version *behaves*, not whether it exists — pin
    the existence rule separately so an edit collapsing them trips CI."""
    text = load_system_prompt(None, mode=mode)
    assert "does not exist" in text or "doesn't exist" in text
    assert "cutoff" in text
    # The rule is worthless if it doesn't bind the verdict.
    assert "Blocker" in text
