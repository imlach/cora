# cora

> a convention-oriented review agent

[![CI](https://github.com/imlach/cora/actions/workflows/ci.yml/badge.svg)](https://github.com/imlach/cora/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/imlach/cora)](https://github.com/imlach/cora/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A self-hostable agentic PR reviewer for teams running their own LLM.
cora reads a pull request the way the repo's conventions say it should
be read — DECISIONS.md records, CLAUDE.md/AGENTS.md rules, CI state,
retrieval over your own docs — and posts one verdict-bearing review
your merge gate can key on.

**Why it exists.** Most LLM review bots are stateless comment factories
against a hosted API. cora was built for the opposite constraints:
local reasoning models with tight context windows, verdicts that gate
merges (so failure modes matter more than demos), and repos where the
conventions are the review standard. That forces design you can
inspect here: per-tier wall/token budgets, wall-hit continuation onto a
larger endpoint, reasoning-leak stripping before anything is posted,
an alternate-family second opinion for verdict disagreements, and a
default-deny trigger policy because reviewers process
attacker-controlled input.

**Status: v0.1.x.** Extracted from a private GitOps deployment where it
has reviewed every PR since May 2026 — and every PR to this repo is
reviewed by cora itself (see the checks on any merged PR).

## Install

cora ships as a container image (the primary path for CI) and as a
release wheel. It is **not on PyPI** — the `cora` name there belongs to an
unrelated package — so pin to a released tag:

```bash
# Container (primary): run it in CI or self-hosted
docker pull ghcr.io/imlach/cora:0.1.0

# Library / CLI: install from the tag (import path is `cora`)
pip install "git+https://github.com/imlach/cora@v0.1.0"
```

Wheel and sdist are also attached to each
[GitHub Release](https://github.com/imlach/cora/releases).

## Layout

```
src/cora/
├── review/       # run_review orchestration phases
├── core/         # the review engine internals
├── providers/    # pluggable retrieval / git-host / reporter seams
├── config.py     # ReviewerConfig dataclass + from_env wiring
└── __main__.py   # `python -m cora`
```

## Public Surface

The library entrypoint is:

```python
from cora import ReviewerConfig, run_review

result = run_review(ReviewerConfig.from_env())
```

`python -m cora` is the thin workflow entrypoint used by dogfooding. It
builds `ReviewerConfig` from the environment and exits nonzero only for
hard entrypoint failures; skipped reviews and review verdicts are
reported through the comment/check-run path.

Provider seams let adopters replace the defaults without forking the
engine:

- `RetrievalProvider`: no-op, local BM25/glob, or TEI/Qdrant retrieval.
- `GitProvider`: read-only repository lookups for `grep_repo` /
  `list_files` / `git_show`.
- `Reporter`: side effects such as GitHub comments, check-runs, and
  patch suggestions.

## Local Development

Native Python:

```bash
make install
make check
```

Docker/Compose:

```bash
docker compose build
docker compose run --rm test
docker compose run --rm dev
```

To run a local review container, copy `.env.example` to `.env`, fill in
the GitHub and LLM settings, then run:

```bash
docker compose run --rm review
```

The runtime image includes Python, `git`, and the GitHub CLI because the
reviewer reads the checkout locally and reports through GitHub APIs.

For local runs, `GH_TOKEN` should belong to an account or GitHub App
that can read the target repository, read pull-request metadata/diffs,
post issue comments, and create/update check-runs. On fine-grained
GitHub tokens, that means:

- **Contents:** read
- **Pull requests:** read
- **Issues:** read/write
- **Checks:** read/write

Patch-writing is opt-in via `REVIEW_PROPOSE_PATCH_DISPATCH=true`. If you
enable `propose_patch` draft-fix PRs, the token used for that path also
needs **Contents:** read/write and **Pull requests:** read/write.
For GitHub Actions runs that should comment as the cora GitHub App, set
`CORA_APP_ID` and `CORA_APP_PRIVATE_KEY`; the workflow mints
`CORA_GH_TOKEN` from those secrets and falls back to `GITHUB_TOKEN`
when they are absent.
See [SECURITY.md](SECURITY.md) for the full token-scope matrix.

A minimal GitHub Actions workflow is available at
[`docs/examples/github-actions/cora-review.yml`](docs/examples/github-actions/cora-review.yml).
The checked-in dogfood workflow is intentionally gated to this
repository and its self-hosted runner.

## Security

cora reviews attacker-controlled input (PRs) with a tool-using agent.
Before pointing it at any repo where strangers can open PRs, read
[SECURITY.md](SECURITY.md) — the threat model, the default-deny
`TriggerPolicy` (author trust rules, comment-only ceiling for untrusted
and fork PRs, rate caps), and the canonical workflow wiring (and the
`pull_request_target` foot-gun to avoid).

## Dogfooding

PRs in this repo are reviewed by cora itself — label a PR `review-quick`
or `review-deep` and the dogfood workflow reviews it with the code from
that PR's own checkout, so a PR that breaks the engine fails its own
review. Labels require triage permission and the workflow only runs for
same-repo branches, so only maintainers can start a run; the
[trigger policy](SECURITY.md) is enforced on top of that.

## Design principles

- **Seams, not forks.** Deployment-specific behaviour lives behind a
  provider ABC or config field with a working default — adopters swap
  implementations, they don't patch the engine.
- **Defaults are a contract.** Behaviour changes ship opt-in;
  intentional breaks are rare and land as documented `BREAKING CHANGE`s.
- **Fail soft, gate hard.** Infrastructure hiccups degrade to a skipped
  review that keeps the merge gate closed; they never fake a verdict.
- **Tests run offline.** The full suite needs no LLM, no network, no
  cluster.

The dogfood workflow reads its endpoints from repo *variables*
(`CORA_REVIEW_RUNNER`, `CORA_LLM_BASE_URL`, `CORA_MCP_URL`,
`CORA_QDRANT_URL`, `CORA_TEI_URL`, `CORA_RERANKER_URL`,
`CORA_QDRANT_COLLECTION`) and the `CORA_MCP_TOKEN` secret, so no
infrastructure hostname lives in the workflow file. Unset variables fall
back to the engine's localhost defaults and the affected feature
soft-skips.
