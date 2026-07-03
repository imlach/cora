# Getting Started

cora is pre-release. Until a package release exists, install from a
checkout or a pinned Git ref.

## Requirements

- Python 3.11 or newer
- `git`
- GitHub CLI (`gh`) for the default GitHub reporter path
- A GitHub token for the target repository
- An OpenAI-compatible endpoint and API key

For the default GitHub reporter, the runtime image also includes Python,
`git`, and `gh` because cora reads the checkout locally and reports
through GitHub APIs.

## Install From Source

```bash
git clone https://github.com/imlach/cora.git
cd cora
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev,otel]'
```

Run the local checks:

```bash
make check
```

## Run A Local Review

Set the target PR, GitHub token, and model endpoint:

```bash
export GH_REPO=owner/repo
export PR_NUMBER=123
export GH_TOKEN=...
export LLM_GATEWAY_KEY=...
export LITELLM_BASE_URL=http://localhost:4000
export REVIEW_MODEL=review
export MAX_TOOL_ITERATIONS=0
export REVIEW_TRIGGER_ENFORCE=true

.venv/bin/python -m cora
```

`MAX_TOOL_ITERATIONS=0` runs quick mode, which avoids the deep MCP tool
loop and is the simplest path for a first local run.

`python -m cora` exits nonzero only for hard entrypoint failures, such as
missing PR identity or an unhandled exception. Skipped reviews and review
verdicts are reported through the comment/check-run path.

## Docker And Compose

Copy `.env.example` to `.env`, then fill in:

```dotenv
GH_REPO=owner/repo
PR_NUMBER=123
GH_TOKEN=...
LLM_GATEWAY_KEY=...
LITELLM_BASE_URL=http://localhost:4000
REVIEW_MODEL=review
MAX_TOOL_ITERATIONS=0
```

Build and run:

```bash
docker compose build
docker compose run --rm review
```

Development helpers:

```bash
docker compose run --rm test
docker compose run --rm dev
```

## GitHub Token Scopes

For local runs, `GH_TOKEN` should belong to an account or GitHub App that
can read the target repository, read pull-request metadata and diffs, post
issue comments, and create or update check-runs.

For fine-grained GitHub tokens, normal comment/check-run review needs:

| Permission | Access |
| --- | --- |
| Contents | read |
| Pull requests | read |
| Issues | read/write |
| Checks | read/write |

Patch-writing is opt-in through `REVIEW_PROPOSE_PATCH_DISPATCH=true`.
If you enable draft-fix PRs or inline patch suggestions, the token used
for that path also needs:

| Permission | Access |
| --- | --- |
| Contents | read/write |
| Pull requests | read/write |

For GitHub Actions runs that should comment as a cora GitHub App, set
`CORA_APP_ID` and `CORA_APP_PRIVATE_KEY`; the workflow mints
`CORA_GH_TOKEN` from those secrets and falls back to `GITHUB_TOKEN` when
they are absent.

See [../SECURITY.md](../SECURITY.md) for the full token-scope matrix and
threat model.

## GitHub Actions

Start from
[`examples/github-actions/cora-review.yml`](examples/github-actions/cora-review.yml).

Until cora is published, install it from GitHub:

```yaml
- name: Install cora
  run: pip install "cora @ git+https://github.com/imlach/cora.git@main"
```

For production use, pin that ref to a tag or commit SHA.

Keep the workflow on `pull_request`. Do not switch to
`pull_request_target` to get secrets for fork PRs; see
[../SECURITY.md](../SECURITY.md) for the approval-label flow and the
reason this matters.
