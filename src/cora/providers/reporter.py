"""Reporter — the side-effect sink for a review.

Everything a review *does* to the outside world (in-progress check-run,
in-progress comment, final review comment, terminal check conclusion,
step summary, automerge gating, propose_patch dispatch) goes through a
`Reporter`. `GitHubReporter` is the default (wraps the engine's existing
check_run / summary / comment / patch_dispatch helpers, preserving the
verdict-check lifecycle); `NullReporter` makes a review
side-effect-free for dry-run / eval / adopters with no SCM wired. A
future GitLab/Gitea/stdout reporter is just another implementation.

The reporter holds the review *context* (repo, pr_number, model,
started_at); `ReviewResult` carries the *outcome*. The orchestrator
(`run_review`) sequences the calls as follows:

  - `open_progress` at the top for EVERY mode so a started
    review always has a check to gate on;
  - `post_in_progress` right after, deep mode only (quick finishes in
    ~20-30s — a placeholder edited moments later is churn);
  - on any early exit: `post_skip` + `complete_check` (zero-placeholder
    budget, `cancelled`/`failure` conclusion carried by the caller);
  - at review end: `complete_check` + `write_summary` (the finalize
    pair), then optional `dispatch_patch` / `apply_label` for a parsed
    propose_patch directive, then `post_review` (+ `pause_automerge`
    when a blocker verdict lands on an automerge-labelled PR).

`complete_check` is idempotent — the first terminal conclusion wins.
That carries the SIGTERM timeout-guard semantics: the
normal finalize path disarms the guard by completing the check, and a
late `timed_out` finalize from a signal handler no-ops instead of
clobbering the real conclusion. `check_open` tells the orchestrator
whether there is still an in-progress check to finalize (e.g. before
arming a SIGTERM guard).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cora.core.budget import Budget

if TYPE_CHECKING:
    from cora.result import ReviewResult


def _zero_budget() -> Budget:
    """Placeholder for preflight exits where no Budget exists yet —
    keeps the check-run stats panel honest (0/0/0); the verdict line
    carries the real story."""
    return Budget(max_input=0, max_output=0, max_iterations=0)


def _empty_dispatch_outcome() -> dict:
    """The `apply_propose_patch_dispatch` outcome shape with nothing
    dispatched — what a side-effect-free dispatch reports."""
    return {
        "inline_url": None,
        "inline_count": 0,
        "draft_url": None,
        "draft_count": 0,
        "source_branch_commit_sha": None,
        "source_branch_count": 0,
        "rejected": [],
        "error": None,
    }


class Reporter(ABC):
    """Side-effect sink for a review's progress + outcome."""

    @abstractmethod
    def open_progress(self, head_sha: str) -> None:
        """Create the in-progress verdict check. Called at the top for
        EVERY mode so a started review always has a check to
        gate on."""

    @property
    @abstractmethod
    def check_open(self) -> bool:
        """True while an in-progress check exists that `complete_check`
        would finalize. The orchestrator reads this before arming its
        SIGTERM timeout guard (no check → nothing to finalize)."""

    @abstractmethod
    def post_in_progress(self) -> None:
        """Create this run's in-progress placeholder comment (deep mode
        only — quick mode's ~20-30s wall makes the placeholder churn
        without benefit). Always a new comment scoped to this run, never
        a find-or-edit onto a leftover from a superseded run."""

    @property
    def progress_open(self) -> bool:
        """True while this run has a placeholder comment still reading
        "in progress" — i.e. `post_in_progress` succeeded and no
        terminal `post_review`/`post_skip` has replaced it.

        The crash handler reads this to decide whether it has a stranded
        placeholder to finalize (cora #14). Deliberately NOT abstract and
        defaulting False: a third-party Reporter written before this must
        keep subclassing, and False means the handler leaves it alone —
        the conservative direction, since `update_run_comment` CREATES a
        comment when the run has none, and a crash should not invent a
        comment on a PR that never had one."""
        return False

    @abstractmethod
    def complete_check(
        self,
        *,
        verdict_line: str | None,
        conclusion: str,
        budget: Budget | None = None,
        wall_time_s: float = 0.0,
        terminated_reason: str | None = None,
    ) -> None:
        """Finalize the verdict check (terminal). Called on an early
        exit (before a full result exists — `budget=None` substitutes a
        zero placeholder) and at review end. Idempotent: the first
        terminal conclusion wins, so a late SIGTERM `timed_out`
        finalize cannot clobber a real conclusion."""

    @abstractmethod
    def write_summary(
        self,
        *,
        body: str,
        budget: Budget,
        wall_time_s: float,
        terminated_reason: str | None,
        is_leak: bool,
        tools_available: list[str],
    ) -> None:
        """Write the run's stats + outcome to the host's summary surface
        ($GITHUB_STEP_SUMMARY on GitHub). The finalize companion to
        `complete_check` on every terminal path that ran the review."""

    @abstractmethod
    def post_review(self, result: ReviewResult) -> None:
        """Render the full review comment and finalise it as this run's
        comment (progress → verdict marker swap, or a fresh comment for
        quick mode), then collapse every other cora comment on the PR."""

    @abstractmethod
    def post_skip(self, reason: str) -> None:
        """Finalise a skip comment (reviewer didn't produce a verdict) —
        a skip is a completed run, same treatment as `post_review`."""

    @abstractmethod
    def pause_automerge(self) -> bool:
        """Remove the automerge label (blocker verdict). True if removed."""

    @abstractmethod
    def dispatch_patch(
        self,
        *,
        directive: dict,
        diff_text: str,
        base_ref: str,
        head_sha: str,
        head_ref: str | None = None,
        is_bot_author_pr: bool = False,
        is_fork_pr: bool = True,
        escalation_warning_body: str | None = None,
        escalation_warning_summary: str | None = None,
        suppress_other_file_edits: bool = False,
    ) -> dict:
        """Dispatch a validated propose_patch directive (inline
        suggestions / draft PR / push-to-source). Returns the
        `apply_propose_patch_dispatch` outcome dict; a side-effect-free
        reporter returns the all-zero shape."""

    @abstractmethod
    def apply_label(
        self, label: str, *, pr_number: str | None = None
    ) -> tuple[bool, str | None]:
        """Apply `label` to a PR (default: the reviewed PR; the patch
        escalation path also labels the draft PR it opened). Returns
        `(ok, error_message)` like the engine helper."""

    @classmethod
    def from_config(cls, cfg, *, started_at: datetime | None = None) -> Reporter:
        """A GitHubReporter when the run has a repo + PR identity, else a
        NullReporter (dry-run / eval / no SCM)."""
        if cfg.repo and cfg.pr_number:
            return GitHubReporter(
                cfg.repo,
                cfg.pr_number,
                model=cfg.model,
                started_at=started_at,
                github_app_token=cfg.github_app_token,
                check_run_name=cfg.check_run_name,
                use_github_review=cfg.use_github_review,
                verdict_words=cfg.verdict_words,
            )
        return NullReporter()


class NullReporter(Reporter):
    """No side-effects — review runs, nothing is posted. The dry-run / eval
    / no-SCM default; inspect the returned `ReviewResult` instead."""

    def open_progress(self, head_sha: str) -> None:
        return None

    @property
    def check_open(self) -> bool:
        return False

    def post_in_progress(self) -> None:
        return None

    def complete_check(self, **_kwargs) -> None:
        return None

    def write_summary(self, **_kwargs) -> None:
        return None

    def post_review(self, result: ReviewResult) -> None:
        return None

    def post_skip(self, reason: str) -> None:
        return None

    def pause_automerge(self) -> bool:
        return False

    def dispatch_patch(self, **_kwargs) -> dict:
        return _empty_dispatch_outcome()

    def apply_label(
        self, label: str, *, pr_number: str | None = None
    ) -> tuple[bool, str | None]:
        # Reports success so the caller's "label apply failed" warning
        # path stays silent; nothing was (or needed to be) done.
        return True, None


class GitHubReporter(Reporter):
    """Posts to GitHub via the engine's check_run / summary / comment /
    patch_dispatch helpers — the default reporting behaviour, incl. the
    verdict-check lifecycle. Holds the review context; check-run
    helpers soft-fail internally, comment posts raise for the caller's
    soft-fail wrapper (a deliberate split: the comment is the primary
    artifact)."""

    def __init__(
        self,
        repo: str,
        pr_number: str,
        *,
        model: str = "",
        started_at: datetime | None = None,
        github_app_token: str | None = None,
        check_run_name: str | None = None,
        use_github_review: bool = False,
        verdict_words: tuple[str, str, str] | None = None,
    ) -> None:
        self.repo = repo
        self.pr_number = pr_number
        self.model = model
        self.started_at = started_at
        # Optional cora-App-class installation token for the patch
        # dispatch / label writes. Falls back to the CORA_GH_TOKEN env
        # at call time so env-wired deployments keep working unchanged.
        self.github_app_token = github_app_token
        # None → the engine's default CHECK_RUN_NAME (the name a merge
        # gate keys on); deployments rebrand via ReviewerConfig.check_run_name.
        self.check_run_name = check_run_name
        # Opt-in: post a first-class PR Review instead of an issue
        # comment. Default-OFF keeps the issue-comment path.
        self.use_github_review = use_github_review
        # Verdict vocabulary for the verdict→event map (None → the
        # engine default, so a custom `words` set maps the same way).
        self.verdict_words = verdict_words
        self._check_id: str | None = None
        self._check_done = False
        # Placeholder-comment lifecycle, for `progress_open` — set when
        # `post_in_progress` succeeds, cleared by the terminal comment
        # writes (`post_review` / `post_skip`).
        self._progress_posted = False

    def _app_token(self) -> str:
        if self.github_app_token is not None:
            return self.github_app_token
        return os.environ.get("CORA_GH_TOKEN", "").strip()

    @contextmanager
    def _app_token_env(self):
        """Expose a config-threaded App token to legacy env-based helpers."""
        app_token = self._app_token()
        if not app_token:
            yield
            return
        old = os.environ.get("CORA_GH_TOKEN")
        os.environ["CORA_GH_TOKEN"] = app_token
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("CORA_GH_TOKEN", None)
            else:
                os.environ["CORA_GH_TOKEN"] = old

    def open_progress(self, head_sha: str) -> None:
        from cora.core.check_run import create_check_run
        from cora.core.config import CHECK_RUN_NAME

        if not head_sha:
            return
        with self._app_token_env():
            self._check_id, _ = create_check_run(
                self.repo,
                self.pr_number,
                head_sha,
                check_run_name=self.check_run_name or CHECK_RUN_NAME,
            )

    @property
    def check_open(self) -> bool:
        return bool(self._check_id) and not self._check_done

    def post_in_progress(self) -> None:
        from cora.core.comment import create_progress_comment, make_initial_comment

        started = self.started_at or datetime.now(UTC)
        with self._app_token_env():
            # Always a NEW comment for this run — never finds-or-edits a
            # leftover placeholder from a cancelled prior run. That one
            # gets collapsed by `minimize_superseded_comments` once this
            # run's own verdict lands (post_review / post_skip).
            create_progress_comment(
                self.repo, self.pr_number, make_initial_comment(self.pr_number, started)
            )
        # Only after the create returns — a raising post leaves no
        # placeholder to strand, and the caller soft-fails it.
        self._progress_posted = True

    @property
    def progress_open(self) -> bool:
        return self._progress_posted

    def complete_check(
        self,
        *,
        verdict_line: str | None,
        conclusion: str,
        budget: Budget | None = None,
        wall_time_s: float = 0.0,
        terminated_reason: str | None = None,
    ) -> None:
        from cora.core.check_run import update_check_run_completed

        if not self.check_open:
            return
        # First terminal conclusion wins (timeout-guard semantics) —
        # marked done BEFORE the PATCH, disarming the guard early,
        # so a concurrent signal-path finalize no-ops.
        self._check_done = True
        with self._app_token_env():
            update_check_run_completed(
                self.repo,
                self._check_id,
                self.pr_number,
                verdict_line=verdict_line,
                conclusion=conclusion,
                budget=budget if budget is not None else _zero_budget(),
                wall_time_s=wall_time_s,
                terminated_reason=terminated_reason,
            )

    def write_summary(
        self,
        *,
        body: str,
        budget: Budget,
        wall_time_s: float,
        terminated_reason: str | None,
        is_leak: bool,
        tools_available: list[str],
    ) -> None:
        from cora.core.summary import write_step_summary

        write_step_summary(
            self.pr_number,
            self.model,
            budget,
            wall_time_s,
            terminated_reason,
            body,
            is_leak,
            tools_available,
        )

    def post_review(self, result: ReviewResult) -> None:
        from cora.core.summary import make_review_comment

        body = make_review_comment(
            self.model,
            result.body,
            result.budget,
            result.wall_time_s,
            result.terminated_reason,
            result.tools_available,
            pr_number=self.pr_number,
            mode=result.mode,
            bot_author=result.bot_author,
            retrieval_source=result.retrieval_source,
            automerge_paused=result.pause_automerge,
            reasoning_stripped_chars=result.reasoning_stripped_chars,
            started_at=self.started_at,
        )
        if self.use_github_review:
            # File a first-class PR Review. The verdict→event
            # map is advisory (COMMENT) except the block verdict
            # (REQUEST_CHANGES); the check-run still carries the gating
            # conclusion. Default-OFF, so this branch only runs when a
            # deployment opts in. Reviews are append-only (no comment
            # loop to speak of), so the per-run marker scheme doesn't
            # apply here.
            from cora.core.comment import create_pr_review
            from cora.core.config import VERDICT_WORDS
            from cora.core.leak import verdict_to_review_event

            words = self.verdict_words or VERDICT_WORDS
            event = verdict_to_review_event(result.verdict, words=words)
            with self._app_token_env():
                create_pr_review(self.repo, self.pr_number, body, event)
            self._progress_posted = False
            return

        from cora.core.comment import minimize_superseded_comments, update_run_comment

        with self._app_token_env():
            # Finalise THIS run's comment (progress → verdict marker
            # swap, or a fresh comment in quick mode) first; only once
            # it's live do older cora comments get collapsed. Order
            # matters: a finalize failure here must not have already
            # minimised the previous verdict — see
            # `minimize_superseded_comments`'s docstring.
            update_run_comment(self.repo, self.pr_number, body, final=True)
            minimize_superseded_comments(self.repo, self.pr_number)
        self._progress_posted = False

    def post_skip(self, reason: str) -> None:
        from cora.core.comment import (
            make_skip_comment,
            minimize_superseded_comments,
            update_run_comment,
        )

        with self._app_token_env():
            # A skip is a completed run, same as a verdict — final=True.
            update_run_comment(self.repo, self.pr_number, make_skip_comment(reason), final=True)
            minimize_superseded_comments(self.repo, self.pr_number)
        self._progress_posted = False

    def pause_automerge(self) -> bool:
        from cora.core.comment import remove_automerge_label

        with self._app_token_env():
            return remove_automerge_label(self.repo, self.pr_number)

    def dispatch_patch(
        self,
        *,
        directive: dict,
        diff_text: str,
        base_ref: str,
        head_sha: str,
        head_ref: str | None = None,
        is_bot_author_pr: bool = False,
        is_fork_pr: bool = True,
        escalation_warning_body: str | None = None,
        escalation_warning_summary: str | None = None,
        suppress_other_file_edits: bool = False,
    ) -> dict:
        from cora.core.patch_dispatch import apply_propose_patch_dispatch

        return apply_propose_patch_dispatch(
            repo=self.repo,
            pr_number=self.pr_number,
            base_ref=base_ref,
            head_sha=head_sha,
            diff_text=diff_text,
            directive=directive,
            gh_token=self._app_token(),
            head_ref=head_ref,
            is_bot_author_pr=is_bot_author_pr,
            is_fork_pr=is_fork_pr,
            escalation_warning_body=escalation_warning_body,
            escalation_warning_summary=escalation_warning_summary,
            suppress_other_file_edits=suppress_other_file_edits,
        )

    def apply_label(
        self, label: str, *, pr_number: str | None = None
    ) -> tuple[bool, str | None]:
        from cora.core.patch_escalation import apply_label_to_pr

        return apply_label_to_pr(
            repo=self.repo,
            pr_number=pr_number or self.pr_number,
            label=label,
            gh_token=self._app_token() or None,
        )
