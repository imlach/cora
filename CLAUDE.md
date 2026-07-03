# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You are an expert Python library engineer for **cora** — a convention-oriented, self-hostable agentic PR reviewer. It runs in production reviewing every PR of the private deployment it was extracted from; this repository is the standalone public package.

---

## Compatibility is the contract

cora's defaults are what existing deployments run. That shapes how you work:

- Behaviour changes ship **opt-in**: a new config field / env knob, default preserving today's behaviour. Intentional breaks are rare, explicit, and called out as `BREAKING CHANGE` in the commit + changelog.
- Deployment-specific behaviour lives behind a seam (provider ABC, config field, connector) with a working generic default — never hardcoded, never deleted. Adopters swap implementations; they don't fork.
- No secrets, internal hostnames, personal identifiers, or deployment specifics in code, comments, tests, or docs. Defaults are neutral (`localhost` placeholders, empty vocab/alias sets that self-disarm).
- CI is infrastructure-independent — GitHub-hosted runners, no private infra. Anything that needs live services (Qdrant, TEI, an LLM gateway, MCP) must degrade gracefully or be mockable; tests are fast and offline.

## Commands

```bash
pip install -e '.[dev]'         # editable install + pytest/ruff
pytest                          # full suite (testpaths = tests/)
pytest tests/test_escalation.py                 # one file
pytest tests/test_escalation.py::test_name      # one test
pytest -k "verdict"             # by keyword
ruff check .                    # lint (target py311)
```

CI runs `pytest -q` on Python 3.11 and 3.13.

## Architecture

Three layers, from public to internal:

1. **Public surface** (top of `src/cora/`) — what an adopter imports.
   - `config.py` — `ReviewerConfig`, the single dataclass driving a review. Its field defaults reference the constants in `cora.core.config` *by reference* so the two cannot drift; `from_env()` is the workflow env wiring. New tunables get a config field whose default mirrors the engine constant.
   - `result.py` — `ReviewResult`, pure outcome with no side-effects, so eval/dry-run can run a review with a `NullReporter` and inspect it directly.
   - `escalation.py` — the tier ladder (`Tier`, `EscalationPolicy`, `EscalationConnector`). A single-model adopter uses one tier; multi-tier deployments configure triggers (`escalation_triggers`) or supply a whole `escalation_policy`. Kept dependency-free (no pydantic_ai/openai imports) so it imports cheaply — preserve that.
   - `second_opinion.py` / `trigger.py` — the T2 second-opinion seam and the trigger-security policy, same dependency-free rule.

2. **Provider seams** (`src/cora/providers/`) — ABCs with a working default each, so cora runs standalone and adopters swap implementations: `RetrievalProvider` (Null / Glob / TeiQdrant), `GitProvider` (`LocalGitProvider` over a configurable checkout root — deep review's local `grep_repo`/`git_show` tools route through this), `Reporter` (the side-effect sink: GitHub comment + check-run, or `NullReporter`).

3. **The engine** (`src/cora/core/`) — the review flow internals:
   - **Two modes**: `quick_review.py` (single LLM call, no tools) and `deep_review.py` (multi-turn Pydantic-AI agent loop with MCP tools + local repo tools). `agent.py` is the shared agent factory; `budget.py` does token/tool-call accounting; `prompt.py` assembles the initial prompt from `pr_context.py` + `retrieval.py` pre-pack.
   - **Escalation machinery**: `continuation.py` (T0→T1 continuation), `kv_continuation.py` (the trajectory-resume connector), `t2_dispatch.py` + `t2_second_opinion.py` + `disagreement.py` (second-opinion dispatch and verdict-disagreement resolution).
   - **Output path**: `leak.py` strips reasoning-model leaks and parses the verdict; `patch_dispatch.py`/`propose_patch.py` handle the hybrid patch-suggestion dispatch; `comment.py`/`check_run.py`/`summary.py` are the GitHub side-effects (behind `Reporter`).
   - Default system prompts ship packaged in `src/cora/prompts/` (`deep.md`, `quick.md`); `ReviewerConfig.*_prompt_path` overrides with deployment-specific ones.

Telemetry (`core/otel.py`) is opt-in via the `otel` extra and imported lazily — never make opentelemetry a hard import.

## Git workflow — branches + PRs, never direct to main

Freely create feature branches, commit, push, and open PRs targeting `main`; the user reviews and merges. Never commit to `main` directly, never push without a PR, never force-push shared branches (`--force-with-lease` on your own PR branch is fine). Branch names: `feat/`, `fix/`, `docs/`, `chore/` + kebab-case. Commits: Conventional Commits with the scopes the log uses (`engine`, `providers`, `escalation`, `prompts`, `reporter`); pass messages via heredoc so formatting survives.

## Code style

- Module docstrings are deliberately rich: they carry each module's role, the seams it participates in, and the failure modes it guards. Maintain them when you change a module's role — they're the map of the codebase.
- Inline comments stay tight: one *what* sentence, a *why* line only when non-obvious. Trade-off essays belong in commit messages and PR descriptions.
- Engine tunables are constants in `core/config.py` mirrored as `ReviewerConfig` fields — don't introduce a second config mechanism.
