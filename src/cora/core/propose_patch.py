"""Shared `propose_patch` directive primitives.

The agentic PR reviewer and an alert-triage receiver (a sibling
agent pipeline in the reference deployment) both
parse a `propose_patch` JSON directive emitted by their respective
agents at the end of the agent's response. This module holds the
*shape* of that directive — parser, validator, allowlist/denylist
constants, branch-name slugifier — so the two consumers stay in lock-
step on what the directive looks like and what's safe to apply.

What lives here:

- Constants (`PROPOSE_PATCH_*`, `_PROPOSE_PATCH_*`)
- `parse_propose_patch_directive(body)` — extract the JSON block from a
  larger markdown body, return the stripped body + parsed directive.
- `validate_propose_patch(directive)` — strict path-allowlist /
  size-cap / no-op / traversal checks. Returns a human-readable error
  or None when the directive is safe to apply.
- `_propose_patch_branch_name(title, suffix)` — deterministic slug
  used by the applier when creating the branch.

What does NOT live here (consumer-specific):

- The PR-reviewer-specific dispatch logic (in-hunk vs out-of-hunk
  classification, inline-suggestion fanout) stays in the reviewer's
  dispatch module — it depends on the PR's diff hunks,
  which triage doesn't have.
- The actual `apply_propose_patch` (branch create + edits + draft PR
  open) — the PR-review path uses `gh api` via subprocess on the CI
  runner (`gh` is baked into the runner image); the triage path will
  use httpx (no `gh` in the triage container, and adding it costs
  ~30 MB of binary). Both will call into this module for parsing +
  validating before they apply.

Stdlib only — both consumers run in different deps environments and
this module needs to be importable from either.
"""
from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Limits — strict on purpose. Catch directives that are too big / too
# wide / off the allowed paths before they make any API calls.
# ---------------------------------------------------------------------------

PROPOSE_PATCH_MAX_EDITS = 10
PROPOSE_PATCH_MAX_FILES = 5
PROPOSE_PATCH_MAX_STRING_CHARS = 4_000   # per old_string / new_string
PROPOSE_PATCH_MAX_TITLE_CHARS = 100
PROPOSE_PATCH_MAX_BODY_CHARS = 2_000

# Path policy. Two consumers, two policies:
#
#   * Triage receiver — no PR context, agent picks paths from alert
#     text alone. Stays narrow (the allowlist below) until widening is
#     justified per-call-site.
#   * PR reviewer — has the PR diff + author intent; draft PRs land
#     under cora[bot] with human review. Wide: any path except
#     `.github/` (self-rewrite of the reviewer's own workflow file
#     would land before the next reviewer run could re-evaluate it).
#
# `validate_propose_patch` takes the policy as keyword args; callers
# pass the constants below explicitly. Empty `allowed_prefixes` means
# "no allowlist gate — only denylist + traversal checks apply".
PROPOSE_PATCH_ALLOWED_PREFIXES = ("k8s/apps/", "notes/")
PROPOSE_PATCH_DENIED_PREFIXES = (
    ".github/", "scripts/", "inventory/", "roles/", "playbooks/",
    "containers/",
)

# Reviewer policy — wider by design. A canary flow catches
# regressions in the agentic stack so the previous "narrow on purpose"
# justification doesn't apply at the reviewer level any more. `.github/`
# stays denied because same-repo PR workflows run from the HEAD ref —
# a propose_patch that rewrites the reviewer's own workflow file would
# take effect on the next sync of that PR before any human reviewed it.
PROPOSE_PATCH_REVIEWER_ALLOWED_PREFIXES: tuple[str, ...] = ()
PROPOSE_PATCH_REVIEWER_DENIED_PREFIXES = (".github/",)

# Fenced code block with info-string `json propose_patch`. The
# orchestrator parses out the JSON between the fences.
_PROPOSE_PATCH_RE = re.compile(
    r"```json\s+propose_patch\s*\n(.*?)\n```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Parser — extract directive from a free-form markdown body.
# ---------------------------------------------------------------------------


def parse_propose_patch_directive(body: str) -> tuple[str, dict | None]:
    """Extract a ```json propose_patch``` block from `body`.

    Returns `(cleaned_body, directive_or_None)`:
      - If a block is found AND the JSON parses to an object, the block
        is removed from `body` (replaced with a single blank line) and
        the parsed dict is returned. The caller then validates +
        applies + appends an outcome note.
      - If a block is found but the JSON is malformed or not an object,
        returns `(body, None)` so the malformed block stays visible in
        the posted comment for the operator to see what the model tried.
      - If no block is found, returns `(body, None)` unchanged.

    At most one directive per response — the regex finds the first
    block; any subsequent ones stay in the body untouched (will read
    as code blocks to the human, which is fine).
    """
    m = _PROPOSE_PATCH_RE.search(body)
    if not m:
        return body, None
    try:
        directive = json.loads(m.group(1))
    except json.JSONDecodeError:
        return body, None
    if not isinstance(directive, dict):
        return body, None
    cleaned = body[:m.start()].rstrip() + "\n\n" + body[m.end():].lstrip()
    return cleaned.rstrip() + "\n", directive


# ---------------------------------------------------------------------------
# Validator — strict shape, size, path-safety checks.
# ---------------------------------------------------------------------------


def validate_propose_patch(
    directive: dict,
    *,
    allowed_prefixes: tuple[str, ...] | None = None,
    denied_prefixes: tuple[str, ...] | None = None,
) -> str | None:
    """Return a human-readable error message for the operator comment, or
    None when the directive is well-formed and safe to apply.

    Validation is intentionally strict — every check below has a
    blast-radius reason. The path denylist is the load-bearing one
    (keeps the App's `contents: write` permission from ever touching
    workflows / etc. via this path); the size caps prevent wholesale
    rewrites that should have been comment-suggestions.

    Policy parameters (see module-level constants for the two policies
    in use today — triage stays narrow, the PR reviewer is wide):

      * `allowed_prefixes`: if non-empty, every edit's path must start
        with one of these. Empty (`()`) means no allowlist gate.
      * `denied_prefixes`: every edit's path must NOT start with one
        of these. Always applied.

    Both default to the triage policy (`PROPOSE_PATCH_ALLOWED_PREFIXES`
    / `PROPOSE_PATCH_DENIED_PREFIXES`) when None so triage and existing
    tests keep working without explicit args.
    """
    if allowed_prefixes is None:
        allowed_prefixes = PROPOSE_PATCH_ALLOWED_PREFIXES
    if denied_prefixes is None:
        denied_prefixes = PROPOSE_PATCH_DENIED_PREFIXES
    title = directive.get("title", "")
    body = directive.get("body", "")
    edits = directive.get("edits", [])
    if not isinstance(title, str) or not title.strip():
        return "missing or empty `title`"
    if len(title) > PROPOSE_PATCH_MAX_TITLE_CHARS:
        return f"`title` longer than {PROPOSE_PATCH_MAX_TITLE_CHARS} chars"
    if not isinstance(body, str) or not body.strip():
        return "missing or empty `body`"
    if len(body) > PROPOSE_PATCH_MAX_BODY_CHARS:
        return f"`body` longer than {PROPOSE_PATCH_MAX_BODY_CHARS} chars"
    if not isinstance(edits, list) or not edits:
        return "`edits` must be a non-empty list"
    if len(edits) > PROPOSE_PATCH_MAX_EDITS:
        return f"too many edits ({len(edits)} > {PROPOSE_PATCH_MAX_EDITS})"
    paths_seen: set[str] = set()
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"edit #{i} is not an object"
        path = edit.get("path", "")
        old_s = edit.get("old_string", "")
        new_s = edit.get("new_string", "")
        if not isinstance(path, str) or not path.strip():
            return f"edit #{i} missing `path`"
        path = path.strip()
        if not isinstance(old_s, str) or not isinstance(new_s, str):
            return f"edit #{i} `old_string` / `new_string` must be strings"
        if len(old_s) > PROPOSE_PATCH_MAX_STRING_CHARS:
            return (
                f"edit #{i} `old_string` longer than "
                f"{PROPOSE_PATCH_MAX_STRING_CHARS} chars"
            )
        if len(new_s) > PROPOSE_PATCH_MAX_STRING_CHARS:
            return (
                f"edit #{i} `new_string` longer than "
                f"{PROPOSE_PATCH_MAX_STRING_CHARS} chars"
            )
        if old_s == new_s:
            return f"edit #{i} `old_string` equals `new_string` (no-op)"
        # Path safety — block traversal, absolute paths, denied prefixes.
        # Reject before any HTTP call so a bad path never reaches the GH API.
        if path.startswith("/") or ".." in path.split("/"):
            return f"edit #{i} path `{path}` escapes repo root"
        if any(path.startswith(p) for p in denied_prefixes):
            return f"edit #{i} path `{path}` is on the denylist"
        if allowed_prefixes and not any(
            path.startswith(p) for p in allowed_prefixes
        ):
            return (
                f"edit #{i} path `{path}` is outside the allowlist "
                f"({', '.join(allowed_prefixes)})"
            )
        paths_seen.add(path)
    if len(paths_seen) > PROPOSE_PATCH_MAX_FILES:
        return (
            f"too many distinct files ({len(paths_seen)} > "
            f"{PROPOSE_PATCH_MAX_FILES})"
        )
    return None


# ---------------------------------------------------------------------------
# Branch-name slugifier — used by appliers to derive the working branch.
# ---------------------------------------------------------------------------


def propose_patch_branch_name(title: str, suffix: str) -> str:
    """`cora/<suffix>-<slug>` — slug is lowercased alnum/dashes from the
    title, capped at 40 chars. Deterministic so re-runs on the same
    source hit the same branch (we re-use it if it already exists).

    `suffix` differentiates the source of the patch — `pr-<n>` for
    PR-review proposals, `triage-<alertname>` for alert-triage
    proposals, etc.
    """
    slug_chars = []
    for c in title.lower():
        slug_chars.append(c if (c.isalnum() or c == "-") else "-")
    slug = "".join(slug_chars)
    # Collapse runs of dashes, trim
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:40].rstrip("-")
    if slug:
        return f"cora/{suffix}-{slug}"
    return f"cora/{suffix}-patch"
