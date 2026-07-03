# Changelog

All notable changes to **cora** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward. Pre-release builds carry `git describe`-derived
versions (e.g. `0.0.0.dev60+g257737d`).

## [Unreleased]

### Added
- Direct cloud-provider seam — `LLM_PROVIDER` (`ReviewerConfig.llm_provider`,
  default `openai-compatible`, byte-identical to today's behaviour) selects
  `anthropic` or `bedrock` to run cora with no LiteLLM/OpenAI-compatible
  gateway in between: an Anthropic API key (`ANTHROPIC_API_KEY`, or
  `LLM_GATEWAY_KEY` which takes precedence) or AWS credentials (Bedrock's
  standard credential chain) are all that's needed. Both paths drop
  non-default sampling parameters and the vLLM `enable_thinking` extra_body
  toggle (current Claude models reject/no-op them) and fall back to the
  provider's own response `model` field for the check-run "Backend" chip
  when no `x-litellm-*` headers are present. New optional extras
  `cora[anthropic]` / `cora[bedrock]`; both imports are lazy, so the default
  path never requires them installed. See `docs/configuration.md` →
  "Running against a cloud provider".

## [0.1.0] - 2026-07-03

First public release. cora's development history predates this
repository going public; it starts at the v0.1.0 cut, with earlier
evolution summarised below and in the module docstrings.

### Added
- Configurable escalation: `ReviewerConfig.escalation_triggers`
  (`CORA_ESCALATION_TRIGGERS`; `wall_hit` default, `blocker` /
  `low_confidence` double-check a needs-changes or no-verdict outcome on
  the next tier under a dedicated second-look framing) and
  `ReviewerConfig.escalation_policy` for a full programmatic ladder
  override (custom tiers, triggers, connector).
- Public setup docs: getting started, configuration reference, and
  architecture overview under `docs/`.
- `QDRANT_COLLECTION` and `GRAFANA_DASHBOARD_PATH` env knobs.
- First-class GitHub Review object support behind the `Reporter` seam,
  opt-in via `use_github_review` (default off).
- Packaging metadata for publication: PyPI classifiers and `[project.urls]`.
- Project docs: this changelog, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- Release workflow (`v*` tag): builds + pushes a multi-arch GHCR container
  image and a wheel/sdist GitHub Release, with build-provenance attestation.
  Distribution is via the container and release artifacts (not PyPI).
- `ReviewerConfig.retrieval_vocab` — the query-reformulation domain
  vocabulary is now configurable (empty by default; deployments supply
  their own terms).

### Removed
- **Breaking:** the LMCache proactive-flush feature (`kv_cache_flush`,
  the `LMCACHE_FLUSH_*` env/config surface, and the `flush_outcome`
  threading). A deployment wanting a cache-warmup side-effect subclasses
  `KvContinuationConnector` and runs it before `super().escalate()`.

### Changed
- **Breaking:** deployment-neutral defaults and identity — the GitHub
  App token env is `CORA_GH_TOKEN` (was `IML_AI_GH_TOKEN`), the bot
  fallback identity and patch-branch prefix are `cora[bot]` / `cora/`,
  the retrieval vocabulary and pretrigger warmup aliases default empty
  (self-disarming), the Qdrant collection defaults to `cora-knowledge`,
  the Grafana drilldown link is omitted unless a dashboard path is
  configured, and no MCP auth header is sent when `mcp_token` is unset.
- `ClusterSecondOpinion` is now `T2SecondOpinion`
  (`cora.core.t2_second_opinion`); the disagreement dissent summary
  labels the primary tier with the configured model alias.
- Quick mode gets its own larger output budget (`QUICK_MAX_OUTPUT_TOKENS`)
  so a reasoning model fits its trace plus the verdict in a single call.
- Default endpoint constants no longer ship internal cluster hostnames;
  they fall back to neutral `localhost` placeholders and are overridden
  per deployment via config/env.
- `requires-python` raised to `>=3.11` (the tested floor); dependency
  pins relaxed to lower-bound floors for library use.

### Changed (distribution)
- Consumers pull released artifacts (GHCR container, Release wheel,
  git tag) — the previous push-model wheel vendoring into the reference
  deployment is retired.

### Fixed
- MCP probe failures no longer log the raw client exception, which could
  embed the `Authorization` header value; header values are redacted and
  the message is length-capped.

[Unreleased]: https://github.com/imlach/cora/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/imlach/cora/releases/tag/v0.1.0
