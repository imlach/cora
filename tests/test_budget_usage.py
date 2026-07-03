"""Regression coverage for pydantic-ai usage accounting.

Guards the bug behind an observed `~0 in / ~0 out tokens` footer: pydantic-ai
migrated `AgentRunResult.usage` from a method to a property and renamed the
token fields `request_tokens`/`response_tokens` -> `input_tokens`/`output_tokens`.
Under the new shape `result.usage()` raises `'RunUsage' object is not callable`,
and the call sites soft-fail, silently zeroing the Budget. `resolve_run_usage`
+ `usage_tokens` reconcile both shapes; these tests lock that in.
"""

from __future__ import annotations

from cora.core.budget import Budget, resolve_run_usage, usage_tokens


def test_resolve_run_usage_property_shape():
    """The production shape: `usage` is a PROPERTY returning a bare
    RunUsage-like object (NOT callable). `result.usage()` would raise
    `'RunUsage' object is not callable`; the resolver must read it directly."""

    class Usage:
        input_tokens = 20_653
        output_tokens = 412

    class Result:
        @property
        def usage(self):
            return Usage()

    resolved = resolve_run_usage(Result())
    assert resolved.input_tokens == 20_653
    assert resolved.output_tokens == 412


def test_resolve_run_usage_method_shape():
    """Back-compat: older pydantic-ai (and existing test fakes) expose
    `usage` as a method — the resolver still calls it."""

    class Usage:
        input_tokens = 7
        output_tokens = 3

    class Result:
        def usage(self):
            return Usage()

    resolved = resolve_run_usage(Result())
    assert resolved.input_tokens == 7
    assert resolved.output_tokens == 3


def test_resolve_run_usage_missing():
    """A result without a usage accessor resolves to None, not a raise —
    the caller soft-fails."""

    class Result:
        pass

    assert resolve_run_usage(Result()) is None


def test_usage_tokens_prefers_new_field_names():
    """New `input_tokens`/`output_tokens` win over the legacy aliases."""

    class Usage:
        input_tokens = 100
        request_tokens = 999  # deprecated alias — must be ignored

    assert usage_tokens(Usage(), "input_tokens", "request_tokens") == 100


def test_usage_tokens_falls_back_to_legacy_alias():
    """When the new field is absent, fall back to the deprecated alias so
    older pydantic-ai builds still account correctly."""

    class Usage:
        request_tokens = 555

    assert usage_tokens(Usage(), "input_tokens", "request_tokens") == 555


def test_usage_tokens_skips_zero_to_reach_alias():
    """A present-but-zero new field shouldn't mask a populated alias."""

    class Usage:
        input_tokens = 0
        request_tokens = 42

    assert usage_tokens(Usage(), "input_tokens", "request_tokens") == 42


def test_usage_tokens_missing_is_zero():
    assert usage_tokens(object(), "input_tokens", "request_tokens") == 0


def test_end_to_end_property_shape_populates_budget():
    """Full path: a property-style result must leave the Budget non-zero —
    the exact observed regression."""
    from cora.core.quick_review import _PydanticAIUsageAdapter

    class Usage:
        input_tokens = 1_234
        output_tokens = 56

    class Result:
        @property
        def usage(self):
            return Usage()

    budget = Budget(max_input=0, max_output=0, max_iterations=0)
    budget.add_usage(_PydanticAIUsageAdapter(resolve_run_usage(Result())))
    assert budget.input_used == 1_234
    assert budget.output_used == 56
