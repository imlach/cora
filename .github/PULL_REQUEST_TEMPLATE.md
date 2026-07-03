<!--
Conventional Commit title with a repo scope, e.g.
  feat(engine): …  fix(providers): …  docs(prompts): …
Scopes: engine, providers, escalation, prompts, reporter.
-->

## What

<!-- The change, in a sentence or two. Link the issue it closes. -->

## Why

<!-- Context / trade-offs. The essay goes here, not in code comments. -->

## Behaviour change for existing deployments

<!--
Defaults are a compatibility contract. State one:
  - None — defaults reproduce current behaviour (how did you confirm?).
  - Opt-in — gated behind a new config field / env, default off.
  - Intentional — and here is why it's acceptable (call it out in the
    changelog as a BREAKING CHANGE).
-->

## Checklist

- [ ] `pytest -q` passes
- [ ] `ruff check .` passes
- [ ] New engine tunables are a `core/config.py` constant mirrored by a
      `ReviewerConfig` field (no second config mechanism)
- [ ] Docs/docstrings updated if a module's role changed
- [ ] No secrets, internal hostnames, or deployment specifics added
