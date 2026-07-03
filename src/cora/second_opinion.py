"""Second-opinion seam — the T2 alt-reviewer dispatch + verdict-disagreement
resolution, kept out of the generic `run_review` hot path.

A deployment can run a *second* review on an independent model (an
alt-reviewer from a different model family, with a different tokenizer /
RLHF / code priors than the T0/T1 tiers) and fold the two verdicts
together via a disagreement resolver. That is **deployment-specific
machinery** — an alt-reviewer alias, verdict-gap scoring, a composition
banner — and the generic path should not know what an "alt-reviewer" is.
So it lives behind this ABC; the working default implementation is
`T2SecondOpinion` in `core/t2_second_opinion.py`, selected from
`ReviewerConfig`. Adopters who don't
configure a second model get `NullSecondOpinion` — no dispatch, primary
verdict untouched.

The seam is consulted at two sites in `run_review`:

  1. **dispatch** — after the primary tier produced a body, before its
     leak-processing. `dispatch()` fires the second review and returns a
     `SecondOpinionResult` the caller carries forward.
  2. **compose** — after the primary body survived leak detection.
     `compose()` runs the second body through the same leak guard, resolves
     the disagreement, composes the final comment body (banner + dissent
     block), and emits the per-tier `tier_verdict` + `disagreement` events.

The result object is also read by the propose-patch escalation path, which
reuses an already-run second opinion rather than firing a redundant verifier
call — so the seam surfaces `body` / `terminated_reason` / `verdict` /
`model_alias` on the result for that reuse.

Dependency-free at module level (no pydantic_ai / openai / engine imports),
matching `escalation.py` / `trigger.py` so it imports cheaply; the default
implementation that pulls in engine internals lives in `core/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from cora.config import ReviewerConfig
    from cora.providers.git import GitProvider


@dataclass
class SecondOpinionResult:
    """Outcome of a second-opinion dispatch, carried by `run_review` from
    the dispatch site to the compose site (and on to the propose-patch
    escalation reuse).

    `dispatched` is False for the no-op path (`NullSecondOpinion`, or the
    default impl when its opt-in flag is off / preconditions unmet) — in
    that case every other field stays at its empty default and the caller
    treats the run as "no second opinion ran".

    The post-leak fields (`verdict`, `has_blocker`, `resolution`) are
    filled in by `compose()`; before compose runs they carry the dispatch
    defaults. `body` holds the raw dispatch body until compose replaces it
    with the post-leak body (the propose-patch escalation path reads the
    post-compose value).
    """

    dispatched: bool = False
    body: str | None = None
    terminated_reason: str | None = None
    tools_available: list[str] = field(default_factory=list)
    model_alias: str | None = None
    # Filled by compose():
    verdict: str | None = None
    has_blocker: bool = False
    resolution: Any | None = None  # cora.core.disagreement.Resolution; loose to stay dep-free


class SecondOpinionProvider(ABC):
    """Strategy for running a second, independent review and folding its
    verdict into the primary one.

    Default is `NullSecondOpinion` (never dispatches). The working T2
    alt-reviewer + disagreement resolution is `T2SecondOpinion`,
    selected by `from_config` whenever a second opinion is
    configured."""

    @abstractmethod
    def should_dispatch(
        self, *, cfg: "ReviewerConfig", is_quick: bool, primary_body: str | None
    ) -> bool:
        """Whether to fire the second opinion for this run. Consulted
        before `dispatch`; the caller skips the dispatch leg entirely when
        this returns False. Encapsulates the opt-in gate (default impl:
        `cfg.t2_disagreement`) so the generic path never references it."""

    @abstractmethod
    async def dispatch(
        self,
        *,
        cfg: "ReviewerConfig",
        endpoint_base_url: str,
        api_key: str,
        system_prompt: str,
        initial_user_prompt: str,
        budget: Any,
        timeout_s: int,
        pr_number: str,
        repo: str,
        mcp_url: str,
        mcp_headers: dict[str, str],
        mcp_actions_url: str | None,
        mcp_actions_headers: dict[str, str] | None,
        web_fetch_url: str | None,
        git: "GitProvider | None",
        log: Callable[[str], None],
        iter_log: Callable[[str], None],
        primary_terminated_reason: str | None = None,
    ) -> SecondOpinionResult:
        """Fire the second review. Returns a `SecondOpinionResult` whose
        `dispatched=True` and `body` carries the raw (pre-leak) body.

        `primary_terminated_reason` is the primary tier's terminated reason,
        surfaced only in the dispatch log line."""

    @abstractmethod
    def compose(
        self,
        *,
        result: SecondOpinionResult,
        cfg: "ReviewerConfig",
        is_quick: bool,
        primary_body_to_post: str,
        primary_terminated_reason: str | None,
        pr_number: str,
        log: Callable[[str], None],
        iter_log: Callable[[str], None],
    ) -> str:
        """Fold the second opinion into the post-leak primary body.

        Runs the second body through leak detection, resolves the
        disagreement, composes the final comment body, and mutates `result`
        in place with the post-leak `verdict` / `has_blocker` / `resolution`
        / `body`. Returns the (possibly rewritten) body to post; when nothing
        ran it returns `primary_body_to_post` unchanged.

        Does NOT emit events — the caller invokes `emit_events` after the
        primary tier's `tier_verdict`, preserving the event order."""

    @abstractmethod
    def emit_events(
        self,
        *,
        result: SecondOpinionResult,
        cfg: "ReviewerConfig",
        is_quick: bool,
        pr_number: str,
        iter_log: Callable[[str], None],
    ) -> None:
        """Emit the second opinion's per-tier `tier_verdict` + `disagreement`
        events. No-op when nothing ran. Called after the primary tier's own
        `tier_verdict` so the event stream keeps primary-then-second
        ordering."""

    @classmethod
    def from_config(cls, cfg: "ReviewerConfig") -> "SecondOpinionProvider":
        """Select the provider for `cfg`. `T2SecondOpinion`
        is the default — it self-disarms when `cfg.t2_disagreement` is off
        (the default), so the generic path runs *no* second opinion unless
        the T2 knobs are set. An adopter wanting an explicit no-op
        passes `NullSecondOpinion`
        directly to `run_review`."""
        from cora.core.t2_second_opinion import T2SecondOpinion

        return T2SecondOpinion()


class NullSecondOpinion(SecondOpinionProvider):
    """No second opinion. `should_dispatch` is always False; `dispatch` and
    `compose` are inert. The generic default for any adopter without an
    alt-reviewer tier."""

    def should_dispatch(
        self, *, cfg: "ReviewerConfig", is_quick: bool, primary_body: str | None
    ) -> bool:
        return False

    async def dispatch(self, **kwargs: Any) -> SecondOpinionResult:  # pragma: no cover — never called
        return SecondOpinionResult(dispatched=False)

    def compose(
        self,
        *,
        result: SecondOpinionResult,
        primary_body_to_post: str,
        **kwargs: Any,
    ) -> str:
        return primary_body_to_post

    def emit_events(self, **kwargs: Any) -> None:
        return None
