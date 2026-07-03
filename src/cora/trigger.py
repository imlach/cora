"""Trigger security — who may fire a review, and at what capability.

A private single-operator deployment can lean on security-by-topology
(private repo + internal runner + single operator) — and all three props
vanish on public exposure. A public, tool-using, write-capable reviewer
faces three concrete threats:

- **Cost / DoS** — every PR fires an LLM call; spam fans out unbounded.
- **Prompt injection** — PR title/body/diff/filenames are untrusted
  attacker input into a tool-using agent.
- **Pwn-request** — adopters reaching for `pull_request_target` + fork
  checkout + secrets. (cora's posture is *diff-as-data*: it greps and
  reads PR content, never checks out and executes it — see SECURITY.md.)

The model is default-deny with a capability ceiling:

- `TriggerPolicy` — declarative gate: which authors are *trusted*
  (association / allowlist / maintainer-applied approval label), what
  happens to everyone else (`skip` or degraded `comment-only`), the
  ceiling for fork PRs, and rate caps.
- `TriggerContext` — one PR's facts, gathered by the caller (the engine
  wires GitHub metadata in; tests construct it directly).
- `evaluate()` — pure policy → decision. `TriggerDecision.degraded`
  means **comment-only / no-tools / no-write**: the engine forces quick
  mode (no MCP, no local repo tools) and suppresses `propose_patch`
  dispatch; posting the verdict comment + check-run is the ceiling.

`enforce` defaults to **True** for the public release posture: a public,
tool-using, write-capable reviewer should fail closed unless a deployment
explicitly opts back out for a private migration window.

Dependency-free (no pydantic_ai / openai / engine imports) so it imports
cheaply, like `cora.escalation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Author associations GitHub reports on a PR (REST `author_association`).
# OWNER/MEMBER/COLLABORATOR have write-ish standing in the repo; everything
# below (CONTRIBUTOR, FIRST_TIME_CONTRIBUTOR, FIRST_TIMER, NONE) is
# untrusted by default.
DEFAULT_ALLOWED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Maintainer-applied opt-in label: a human read the PR and vouched for it.
# Label application is itself permission-gated by GitHub (triage+), so the
# label is a trustworthy signal in a way PR content never is.
DEFAULT_APPROVE_LABEL = "cora:approved"

# What happens to PRs that clear no trust rule.
UNTRUSTED_ACTIONS = frozenset({"skip", "comment-only"})

# Capability ceiling for fork PRs. `full` defers entirely to the trust
# rules; `comment-only` caps even trusted authors when the head lives in
# a fork; `skip` refuses fork PRs outright.
FORK_ACTIONS = frozenset({"full", "comment-only", "skip"})


@dataclass(frozen=True)
class TriggerPolicy:
    """Declarative trigger gate. One instance per deployment, carried on
    `ReviewerConfig.trigger`; `evaluate()` applies it to one PR."""

    # Master switch. False → `evaluate` allows everything at full
    # capability (no gating at all). Public/default posture is on.
    enforce: bool = True

    # Trust rules — clearing ANY of the three marks the author trusted.
    allowed_associations: frozenset[str] = DEFAULT_ALLOWED_ASSOCIATIONS
    allowed_authors: frozenset[str] = frozenset()
    approve_label: str = DEFAULT_APPROVE_LABEL

    # Untrusted authors: "skip" (default-deny) or "comment-only"
    # (degraded run — no tools, no writes beyond the verdict itself).
    untrusted_action: str = "skip"

    # Fork PRs: capability ceiling applied after the trust rules.
    fork_action: str = "comment-only"

    # Rate caps (best-effort cost guard, not a security boundary — see
    # SECURITY.md). None → uncapped. Counts are supplied by the caller
    # via `TriggerContext`; when a cap is set but its count is None
    # (probe failed), the cap is skipped — availability over precision
    # for a pure cost control.
    max_runs_per_hour: int | None = None
    max_runs_per_author_per_hour: int | None = None

    def __post_init__(self) -> None:
        # Fail loud on a misspelt action: a typo silently degrading
        # "skip" into something weaker is exactly the failure mode a
        # security gate can't have.
        if self.untrusted_action not in UNTRUSTED_ACTIONS:
            raise ValueError(
                f"untrusted_action must be one of {sorted(UNTRUSTED_ACTIONS)}, "
                f"got {self.untrusted_action!r}"
            )
        if self.fork_action not in FORK_ACTIONS:
            raise ValueError(
                f"fork_action must be one of {sorted(FORK_ACTIONS)}, "
                f"got {self.fork_action!r}"
            )


@dataclass(frozen=True)
class TriggerContext:
    """One PR's trigger-relevant facts. The engine builds this from PR
    metadata; `author_association` comes from the REST PR object and is
    fetched only when the policy is enforced."""

    author: str = ""
    author_association: str = ""
    labels: frozenset[str] = field(default_factory=frozenset)
    is_fork: bool = False
    # Recent runs of the reviewer workflow (total / by this PR's author)
    # inside the cap window. None → unknown (probe failed or not run).
    recent_runs_total: int | None = None
    recent_runs_by_author: int | None = None


@dataclass(frozen=True)
class TriggerDecision:
    allowed: bool
    # comment-only / no-tools / no-write ceiling. Only meaningful when
    # `allowed` — the engine forces quick mode and suppresses
    # propose_patch dispatch.
    degraded: bool
    # Whether the author cleared a trust rule (independent of the fork
    # ceiling / caps) — surfaced for logging.
    trusted: bool
    reason: str


def is_trusted(policy: TriggerPolicy, ctx: TriggerContext) -> bool:
    """ANY-of the three trust rules. Label comparison is case-insensitive
    (GitHub label names are case-preserving but case-insensitively
    unique); author comparison likewise (logins are case-insensitive)."""
    if ctx.author_association.upper() in policy.allowed_associations:
        return True
    if ctx.author.lower() in {a.lower() for a in policy.allowed_authors}:
        return True
    approve = policy.approve_label.lower()
    return bool(approve) and approve in {lbl.lower() for lbl in ctx.labels}


def evaluate(policy: TriggerPolicy, ctx: TriggerContext) -> TriggerDecision:
    """Pure policy application — no I/O, no engine imports.

    Order: rate caps first (they bound cost regardless of trust), then
    the trust rules, then the fork ceiling. The strictest applicable
    outcome wins; `degraded` ceilings compose (trusted-but-fork and
    untrusted-but-comment-only land in the same degraded mode).
    """
    if not policy.enforce:
        return TriggerDecision(
            allowed=True, degraded=False, trusted=True,
            reason="trigger policy not enforced",
        )

    if (
        policy.max_runs_per_hour is not None
        and ctx.recent_runs_total is not None
        and ctx.recent_runs_total >= policy.max_runs_per_hour
    ):
        return TriggerDecision(
            allowed=False, degraded=False, trusted=is_trusted(policy, ctx),
            reason=(
                f"global rate cap: {ctx.recent_runs_total} runs in the "
                f"last hour (cap {policy.max_runs_per_hour})"
            ),
        )
    if (
        policy.max_runs_per_author_per_hour is not None
        and ctx.recent_runs_by_author is not None
        and ctx.recent_runs_by_author >= policy.max_runs_per_author_per_hour
    ):
        return TriggerDecision(
            allowed=False, degraded=False, trusted=is_trusted(policy, ctx),
            reason=(
                f"per-author rate cap: {ctx.recent_runs_by_author} runs by "
                f"`{ctx.author}` in the last hour "
                f"(cap {policy.max_runs_per_author_per_hour})"
            ),
        )

    trusted = is_trusted(policy, ctx)
    degraded = False
    if not trusted:
        if policy.untrusted_action == "skip":
            return TriggerDecision(
                allowed=False, degraded=False, trusted=False,
                reason=(
                    f"author `{ctx.author}` "
                    f"({ctx.author_association or 'NONE'}) cleared no trust "
                    f"rule; apply `{policy.approve_label}` to opt this PR in"
                ),
            )
        degraded = True

    if ctx.is_fork:
        if policy.fork_action == "skip":
            return TriggerDecision(
                allowed=False, degraded=False, trusted=trusted,
                reason="fork PRs are disabled by policy (fork_action=skip)",
            )
        if policy.fork_action == "comment-only":
            degraded = True

    reason = "trusted author" if trusted else (
        "untrusted author → comment-only"
    )
    if ctx.is_fork and degraded:
        reason += "; fork PR → comment-only ceiling"
    return TriggerDecision(
        allowed=True, degraded=degraded, trusted=trusted, reason=reason
    )
