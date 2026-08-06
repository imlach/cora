"""Push-based context refresher for the agent reviewer's iteration loop.

The reviewer's initial prompt bakes in a single CI / PR snapshot taken
at start (via `gather_ci_context` + `fetch_pr_diff` in `pr_context.py`).
Once the agent loop starts, the model is blind to anything that lands
mid-review — a check turning red, a new commit pushing, a maintainer
posting a clarifying comment. A pull-based alternative (an MCP
`fetch_workflow_log`-style tool) lets the model PULL fresh context,
but pull has three problems:

1. The model has to *know* to ask. If it forms an early incorrect
   verdict, it never calls the tool.
2. Each pull burns one iteration of the (already tight) request budget.
3. Reactive only — misses signals the model wouldn't think to look for.

This module flips that to push: the harness checks for new signals
between graph-node boundaries and injects a synthetic `UserPromptPart`
when something changed. Three sources, all on by default, each behind
its own killswitch (`ReviewerConfig.context_injection*` when threaded
by `run_review`; the legacy env toggle otherwise):

- **CI delta** — `gh api /repos/{repo}/commits/{head_sha}/check-runs`,
  hashed and compared to the prior snapshot. New failure → inject a
  fresh `gather_ci_context` block. A check-run turning GREEN (success,
  where the prior observation had it pending/missing/red) injects too —
  the motivating case (cora issue #23): a deep review that forms a 🚨
  Blocker claiming a build/test failure while that check is still in
  flight never learns the check went green a few turns later unless
  something tells it. The green-delta body names the check and tells
  the model to re-verify or downgrade any compile/test-failure finding
  it made; this is the PRIMARY fix (the model self-corrects with the
  signal in context) — `cora.review._ci_gate` is the finalize-time
  backstop for reviews that finish before this ever polls.
- **HEAD SHA delta** — `gh api repos/{repo}/pulls/{n}` `.head.sha`.
  Changed mid-review → re-fetch the diff (`gh pr diff`) up to
  `DIFF_CHAR_CAP` and inject "diff has shifted, here it is".
- **Comments delta** — `gh api /repos/{repo}/issues/{n}/comments`
  filtered by `since=<start_ts>` with the reviewer's own bot login
  filtered out. New human comments → inject verbatim.

Per-source dedupe state lives on the `ContextRefresher` instance —
the same instance is passed to T0 and T1 so the T1 continuation
doesn't re-inject what T0 already saw. Polling cadence is per-source
so we don't pay 3 API calls per turn:

- HEAD SHA: every turn (cheap single call)
- Comments: every 2 turns
- CI: every 3 turns (slowest, most expensive)

When `refresh()` returns a non-None body, the caller (`iter_with_turn_logging`)
wraps it in the `[CONTEXT UPDATE — injected by the review harness, not a
user message]` framing and appends a `UserPromptPart` to the next
`ModelRequestNode.request.parts`. The "not a user message" framing is
load-bearing — without it the model treats the injection as a
conversational pivot from the human author.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Callable

from cora.core.config import CHECK_RUN_NAME, DIFF_CHAR_CAP
from cora.core.pr_context import fetch_pr_diff, gather_ci_context


# Per-injection wall-time extension (seconds). Each injection adds work
# the model didn't budget for — give it room to act on the new context
# without running headlong into the deadline. 90s ≈ one extra slow turn.
INJECTION_DEADLINE_EXTENSION_S = 90.0

# Cap on total extensions per review. 3 × 90s = +270s ceiling. Bounds
# the worst case where a comment-heavy PR keeps tripping the comments
# delta and the review otherwise wouldn't terminate.
MAX_INJECTION_EXTENSIONS = 3

# Polling cadence per source — turn count modulo. HEAD checks every
# turn (cheapest); comments every 2; CI every 3. The cadence is the
# *check* rate, not the *injection* rate; dedupe inside each source
# means a stale check is a no-op return.
_CADENCE_HEAD = 1
_CADENCE_COMMENTS = 2
_CADENCE_CI = 3

# Env toggles. Master + per-source. All default "true" — operationally
# we want the signal on; specific sources can be killswitched
# independently when one turns out to be noisy on a given workflow.
_ENV_MASTER = "AGENT_REVIEW_CONTEXT_INJECTION"
_ENV_CI = "AGENT_REVIEW_CONTEXT_INJECTION_CI"
_ENV_HEAD = "AGENT_REVIEW_CONTEXT_INJECTION_HEAD"
_ENV_COMMENTS = "AGENT_REVIEW_CONTEXT_INJECTION_COMMENTS"
# Sub-toggle of the CI source (same endpoint + cadence as _ENV_CI —
# turning that master CI toggle off disables this too). Separate switch
# so a deployment can keep red-delta injection while opting out of the
# green one specifically.
_ENV_CI_GREEN = "AGENT_REVIEW_CONTEXT_INJECTION_CI_GREEN"


def _env_on(name: str) -> bool:
    """Default-true env toggle. Anything other than the literal `false`
    (case-insensitive) leaves the source enabled — typo-safe."""
    return os.environ.get(name, "true").strip().lower() != "false"


def _gh_api(path: str) -> str | None:
    """Best-effort `gh api` GET. Returns stdout on success, None on
    any failure (network, auth, parsing) — context injection is
    always optional and must never break the review."""
    try:
        proc = subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except Exception:  # noqa: BLE001
        return None


def wrap_injection(*, reason: str, body: str) -> str:
    """Wrap the source body in the harness-injection framing. Public
    so callers + tests can build the same envelope without re-deriving
    the literal string. The framing tells the model the message is
    NOT a conversational pivot from the human — without that the
    model often re-greets and restarts the analysis."""
    return (
        "[CONTEXT UPDATE — injected by the review harness, not a user message]\n"
        f"Reason: {reason}\n\n"
        f"{body}\n\n"
        "Adjust your in-progress verdict if this changes your assessment."
    )


class ContextRefresher:
    """Per-review push-context state. Construct once before T0; pass
    the same instance to T0 (`deep_review_call`) and T1
    (`continue_on_t1`) so dedupe state carries across the handoff —
    a comment T0 already saw shouldn't be re-injected on T1.

    Methods:
      - `refresh(turn: int) -> str | None` — the main entry. Called
        at each node boundary in `iter_with_turn_logging`. Returns
        the wrapped injection body when a delta fires, None
        otherwise. Per-source cadence + dedupe applied inside.
      - `record_extension()` — caller invokes when it actually
        consumed an injection-driven deadline extension; the
        refresher tracks against `MAX_INJECTION_EXTENSIONS` so the
        loop can stop bumping the deadline past the cap.
      - `can_extend()` — read-only check the caller uses before
        invoking the extend callback.
      - `last_source` — `str | None` of the source that fired on the
        most recent successful `refresh()`. Used by the caller to
        tag the `context_injected` log event.

    State lives in the instance:
      - `_last_check_hash` — SHA256 of the CI check-runs payload
        normalised to (name, status, conclusion) tuples.
      - `_last_check_signature` — the normalised tuple list itself (not
        just its hash), kept so the NEXT poll can diff per-check-run
        transitions (e.g. "was this specific check green before?") —
        the hash alone only proves *something* changed, not *what*.
      - `_last_head_sha` — the head SHA observed on the most recent
        HEAD check.
      - `_last_seen_comment_id` — the highest issue-comment id we've
        already injected, so older comments don't re-fire if GitHub
        re-orders the list.
      - `_extensions_consumed` — running count vs `MAX_INJECTION_EXTENSIONS`.

    The repo / pr_number / head_sha / start_timestamp are captured at
    construction so the refresher doesn't need to know about
    `pr_context.py`'s env-fishing.
    """

    def __init__(
        self,
        *,
        repo: str,
        pr_number: str,
        head_sha: str | None,
        start_timestamp_iso: str,
        bot_login: str | None = None,
        # Config-threaded enable flags (`cfg.context_injection*`). None
        # falls back to the legacy env toggles below — same cfg=None
        # pattern as `deep_review._thinking_extra_body`, so direct
        # engine callers keep the env-driven behaviour.
        enabled: bool | None = None,
        ci_enabled: bool | None = None,
        head_enabled: bool | None = None,
        comments_enabled: bool | None = None,
        # Green-delta sub-toggle — see `_ENV_CI_GREEN`.
        ci_green_enabled: bool | None = None,
        # The reviewer's own verdict check-run name (`cfg.check_run_name`)
        # plus the always-excluded `required` aggregator: a check turning
        # green because IT'S OUR OWN check completing, or because the
        # aggregator mirrors sibling state, isn't evidence about the PR's
        # build/test — same exclusion `gather_ci_context` applies to the
        # failing side. Defaults to the engine's own check-run name.
        own_check_run_name: str = CHECK_RUN_NAME,
        # Inject callable for the gather_ci_context helper. Defaults to
        # the real implementation; tests stub it.
        gather_ci_context_fn: Callable[[str, str | None], str | None] = gather_ci_context,
        # Same — fetch_pr_diff is the real default, swappable in tests.
        fetch_pr_diff_fn: Callable[[str], str] = fetch_pr_diff,
        # Same — `_gh_api` is the real fetcher; tests stub.
        gh_api_fn: Callable[[str], str | None] = _gh_api,
    ) -> None:
        self._repo = repo
        self._pr_number = pr_number
        self._start_timestamp_iso = start_timestamp_iso
        # The reviewer's own bot login. Comments authored by us must
        # not re-trigger the comments-delta source — that's a feedback
        # loop. Defaults from env; explicit ctor arg wins.
        self._bot_login = (
            bot_login
            or os.environ.get("AGENT_REVIEW_LOOP_GUARD_BOT_LOGIN", "cora[bot]")
        ).lower()
        self._own_check_run_name = own_check_run_name
        self._gather_ci_context = gather_ci_context_fn
        self._fetch_pr_diff = fetch_pr_diff_fn
        self._gh_api = gh_api_fn

        # Master + per-source enable flags cached at construction.
        # Explicit (config-threaded) args win; None falls back to the
        # env toggles. Re-reading env per call would be more flexible
        # but risks the in-loop check seeing a half-applied env change.
        self._master_on = _env_on(_ENV_MASTER) if enabled is None else enabled
        self._ci_on = _env_on(_ENV_CI) if ci_enabled is None else ci_enabled
        self._head_on = _env_on(_ENV_HEAD) if head_enabled is None else head_enabled
        self._comments_on = (
            _env_on(_ENV_COMMENTS) if comments_enabled is None else comments_enabled
        )
        self._ci_green_on = (
            _env_on(_ENV_CI_GREEN) if ci_green_enabled is None else ci_green_enabled
        )

        # Seed the dedupe state with the initial values so the first
        # `refresh()` call doesn't fire on the pre-baked context.
        # Tests can construct with the initial head_sha they care about.
        self._last_head_sha: str | None = head_sha
        self._last_check_hash: str | None = None
        # Normalised (name, status, conclusion) signature from the most
        # recent CI poll — None until the first poll runs. See the class
        # docstring's "State lives in the instance" section.
        self._last_check_signature: list[tuple[str, str, str]] | None = None
        self._last_seen_comment_id: int = 0
        self._extensions_consumed: int = 0
        self.last_source: str | None = None

    def can_extend(self) -> bool:
        """True when an injection-driven deadline bump still fits under
        the `MAX_INJECTION_EXTENSIONS` cap."""
        return self._extensions_consumed < MAX_INJECTION_EXTENSIONS

    def record_extension(self) -> None:
        """Caller invokes after applying an extend_deadline_fn(90)
        bump. Bookkeeping only — `can_extend()` reads this; the
        actual deadline mutation lives in the caller-supplied
        callback."""
        self._extensions_consumed += 1

    async def refresh(self, *, turn: int) -> str | None:
        """Inspect each enabled source on its cadence. Returns the
        wrapped injection body if anything new fired, None otherwise.

        Ordering: HEAD → CI → comments. HEAD is cheapest and most
        operationally significant (a force-push invalidates everything
        the model has built up). At most ONE source fires per call —
        if multiple have new content, the lower-priority ones land on
        a later turn. Keeps each injection focused and the log line
        single-source.
        """
        if not self._master_on:
            return None

        # HEAD SHA — every turn. Cheapest, single API call, and the
        # most diff-invalidating event (PR head moved means the diff
        # the model has been analysing may no longer apply).
        if self._head_on and turn % _CADENCE_HEAD == 0:
            head_body = self._check_head_delta()
            if head_body is not None:
                self.last_source = "head"
                return head_body

        # CI — every 3 turns. Slowest (two-call shape: list checks,
        # potentially fetch logs for failing ones). Higher latency
        # justifies the lower cadence.
        if self._ci_on and turn % _CADENCE_CI == 0:
            ci_body = self._check_ci_delta()
            if ci_body is not None:
                self.last_source = "ci"
                return ci_body

        # Comments — every 2 turns. Cheaper than CI but still one
        # round-trip; human comments are usually rare during a 6-min
        # review so the 2-turn cadence is fine.
        if self._comments_on and turn % _CADENCE_COMMENTS == 0:
            comment_body = self._check_comments_delta()
            if comment_body is not None:
                self.last_source = "comments"
                return comment_body

        return None

    # ---- per-source check helpers ---------------------------------

    def _check_head_delta(self) -> str | None:
        """`gh api repos/{repo}/pulls/{n}` → head.sha. Changed since
        last check → re-fetch the diff (up to `DIFF_CHAR_CAP`) and
        return the injection body."""
        raw = self._gh_api(f"/repos/{self._repo}/pulls/{self._pr_number}")
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        new_sha = ((payload.get("head") or {}).get("sha") or "").strip()
        if not new_sha or new_sha == self._last_head_sha:
            return None
        prior_sha = self._last_head_sha
        self._last_head_sha = new_sha
        # Re-fetch the diff so the model has the actual new content,
        # not just "trust us, things changed". Truncate to the same
        # cap the initial prompt uses to avoid blowing the context.
        try:
            new_diff = self._fetch_pr_diff(self._pr_number)
        except Exception:  # noqa: BLE001
            new_diff = ""
        if len(new_diff) > DIFF_CHAR_CAP:
            new_diff = new_diff[:DIFF_CHAR_CAP] + (
                f"\n\n…[diff truncated at {DIFF_CHAR_CAP} chars]…\n"
            )
        body_parts = [
            "## PR head updated mid-review",
            "",
            (
                f"The PR head SHA moved from `{prior_sha or '(unknown)'}` to "
                f"`{new_sha}` while you were reviewing. The diff below is the "
                "fresh state — re-validate any verdict that depended on the "
                "prior diff."
            ),
            "",
            "```diff",
            new_diff or "(no diff content — fetch failed)",
            "```",
        ]
        return wrap_injection(
            reason=f"PR head SHA changed: {prior_sha} → {new_sha}",
            body="\n".join(body_parts),
        )

    def _check_ci_delta(self) -> str | None:
        """Hash the check-run set; emit an injection when the hash
        changes. Uses the current `_last_head_sha` (so a head bump on
        the same turn will pick the new SHA for the CI lookup too).

        Two shapes fire from here, red taking priority when both are
        present in the same poll:

        1. **Red** — `gather_ci_context` finds a failing check. Reuses
           the standard formatter so the body shape matches the initial
           prompt's CI block — the model has already learned to read it.
        2. **Green** — no failures, but a check-run that was NOT green
           on the prior observation (pending, missing, or itself
           red) is green now. See `_newly_green_checks` +
           `_green_delta_injection`; killswitch `_ci_green_on`.
        """
        head = self._last_head_sha
        if not head:
            return None
        raw = self._gh_api(f"/repos/{self._repo}/commits/{head}/check-runs?per_page=100")
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        check_runs = payload.get("check_runs") or []
        # Normalise: (name, status, conclusion) tuples sorted by name.
        # Stable across re-runs (the started_at moves but we don't
        # hash that) and across name re-ordering.
        signature = sorted(
            (
                (cr.get("name") or "", cr.get("status") or "", cr.get("conclusion") or "")
                for cr in check_runs
            )
        )
        sig_hash = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if sig_hash == self._last_check_hash:
            return None
        is_first_check = self._last_check_hash is None
        # Snapshot the PRIOR signature before overwriting it — the green
        # delta below diffs against it to tell "was already green" apart
        # from "just turned green".
        prior_signature = self._last_check_signature
        self._last_check_hash = sig_hash
        self._last_check_signature = signature
        # First-ever observation isn't a delta — it's the baseline. The
        # initial prompt already carried `gather_ci_context` at start.
        if is_first_check:
            return None
        ci_body = self._gather_ci_context(self._repo, head)
        if ci_body:
            return wrap_injection(
                reason="CI checks changed since last observation",
                body=ci_body,
            )
        if not self._ci_green_on:
            return None
        newly_green = self._newly_green_checks(prior_signature, signature)
        if not newly_green:
            return None
        return self._green_delta_injection(
            head, newly_green, all_settled=self._all_relevant_settled(signature)
        )

    def _all_relevant_settled(
        self, signature: "list[tuple[str, str, str]]"
    ) -> bool:
        """True when every relevant check has finished, whatever the
        outcome. Distinguishes "CI is done and clean" from "one check
        finished, the build is still running" — the injection wording
        depends on it, because telling the model its finding is
        "contradicted by CI" on the strength of a lint job while the
        build is still in flight is false authority."""
        return all(
            status == "completed"
            for name, status, _conclusion in signature
            if name
            and name != "required"
            and not name.startswith("agentic-pr-review")
            and name != self._own_check_run_name
        )

    def _newly_green_checks(
        self,
        prior: list[tuple[str, str, str]] | None,
        current: list[tuple[str, str, str]],
    ) -> list[str]:
        """Names of checks that are `completed`/`success` NOW but were
        not in that exact state on the prior poll (pending, missing
        entirely, or a different conclusion). Own-check + `required`
        aggregator excluded — same feedback-loop / mirrors-sibling-state
        reasoning `gather_ci_context` applies on the failing side."""
        prior_by_name = {name: (status, concl) for name, status, concl in (prior or [])}
        newly_green = []
        for name, status, conclusion in current:
            if (
                not name
                or name == "required"
                or name.startswith("agentic-pr-review")
                or name == self._own_check_run_name
            ):
                continue
            if status != "completed" or conclusion != "success":
                continue
            if prior_by_name.get(name) == (status, conclusion):
                continue  # already green last time we looked — not a delta
            newly_green.append(name)
        return newly_green

    def _green_delta_injection(
        self, head: str, newly_green: list[str], *, all_settled: bool = True
    ) -> str:
        """Build the injection body for check-run(s) that just turned
        green — the issue #23 motivating shape: a review forms a 🚨
        Blocker asserting a build/test failure while the real check is
        still in flight, then that check passes minutes later with no
        way for the model to find out. Names each check explicitly and
        tells the model what to do with a prior compile/test-failure
        claim, rather than just handing back the raw CI section again."""
        lines = [
            "## CI check(s) turned green",
            "",
            (
                f"The following check(s) completed successfully for HEAD "
                f"`{head}` — on the previous observation they were pending, "
                "missing, or not yet green:"
            ),
            "",
        ]
        for name in newly_green:
            # No "build/test" label — these are whatever checks the repo
            # happens to run, and calling a lint job a build check is the
            # false precision issue #23 is about in the first place.
            lines.append(f"- `{name}` completed successfully for HEAD `{head}`.")
        lines.append("")
        lines.append(
            # The directive is only warranted once CI has actually
            # finished. While other checks are still running, a single
            # green check contradicts nothing — saying otherwise is the
            # same unearned authority the gate exists to remove.
            "Every other check for this HEAD has also finished and "
            "passed. A finding asserting this code fails to compile or "
            "fails its tests is contradicted by CI — re-verify it, and "
            "downgrade it unless you can point at something CI does not "
            "cover."
            if all_settled
            else "Other checks for this HEAD are still running, so this "
            "does not yet settle whether the build or tests pass. Treat "
            "it as partial evidence only; do not downgrade a "
            "compile/test finding on the strength of it alone."
        )
        return wrap_injection(
            reason="CI check(s) transitioned to success since last observation",
            body="\n".join(lines),
        )

    def _check_comments_delta(self) -> str | None:
        """`gh api repos/{repo}/issues/{n}/comments?since=<start>` then
        filter out (a) the reviewer's own bot comments and (b) anything
        we've already injected (by id). The `since` param keeps the
        payload small even on long-discussion PRs."""
        # GitHub's `since` accepts ISO-8601 UTC; we captured the review
        # start in `start_timestamp_iso`. Filtering server-side first
        # means we don't pay for the full comment-list payload on every
        # poll.
        path = (
            f"/repos/{self._repo}/issues/{self._pr_number}/comments"
            f"?per_page=100&since={self._start_timestamp_iso}"
        )
        raw = self._gh_api(path)
        if raw is None:
            return None
        try:
            comments = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(comments, list):
            return None
        # Filter: drop the reviewer's own bot comments + already-seen ids.
        new_comments = []
        for c in comments:
            cid = int(c.get("id") or 0)
            if cid <= self._last_seen_comment_id:
                continue
            login = ((c.get("user") or {}).get("login") or "").lower()
            if login == self._bot_login:
                continue
            new_comments.append(c)
        if not new_comments:
            return None
        # Track the highest id we processed so subsequent polls skip them.
        self._last_seen_comment_id = max(
            int(c.get("id") or 0) for c in new_comments
        )
        # Render the new comments verbatim. Cap each body to keep the
        # injection bounded; very long pasted logs would otherwise
        # blow the context.
        parts = [
            "## New comments on the PR",
            "",
            (
                f"{len(new_comments)} new comment(s) since the review started. "
                "Read them and adjust your verdict if anything substantive lands."
            ),
        ]
        for c in new_comments:
            author = ((c.get("user") or {}).get("login") or "?")
            body = (c.get("body") or "").strip()
            if len(body) > 4_000:
                body = body[:4_000] + "\n…[comment truncated at 4000 chars]…"
            parts += ["", f"### @{author}", "", body]
        return wrap_injection(
            reason=f"{len(new_comments)} new comment(s) on the PR",
            body="\n".join(parts),
        )
