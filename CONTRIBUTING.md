# Contributing to cora

Thanks for your interest in cora — a convention-oriented, self-hostable
agentic PR reviewer. Contributions are welcome by pull request.

## Development setup

cora targets Python 3.11+ and is dependency-light. From a clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'      # editable install + pytest + ruff
```

## Before you open a PR

```bash
pytest                       # full suite — fast and offline (no network, no services)
ruff check .                 # lint (target py311)
```

CI runs the same suite on Python 3.11 and 3.13; both must pass. Tests are
deliberately offline and mockable — please keep new tests that way (no
live network, no external services).

## Workflow

- Branch off `main`; never commit to `main` directly. Branch names use a
  type prefix + kebab-case: `feat/`, `fix/`, `docs/`, `chore/`.
- Open a PR targeting `main`; a maintainer reviews and merges.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/),
  with scopes such as `engine`, `providers`, `escalation`, `prompts`,
  `reporter`.

## Conventions worth knowing

- **Engine tunables** are constants in `cora/core/config.py`, each mirrored
  as a `ReviewerConfig` field whose default references the constant. Add new
  tunables that way — don't introduce a second configuration mechanism.
- **Provider seams** (`RetrievalProvider`, `GitProvider`, `Reporter`) each
  ship a working default so cora runs standalone; adopters swap their own.
- **Telemetry** (OpenTelemetry) is opt-in via the `otel` extra and imported
  lazily — never make it a hard import.
- The packaged default prompts pin verdict markers and a `**Findings:**`
  structure that the output parser depends on. If you touch
  `cora/prompts/*.md`, keep those markers verbatim (`tests/test_prompts.py`
  enforces them).

## Reporting security issues

Please follow `SECURITY.md` rather than opening a public issue for anything
security-sensitive.
