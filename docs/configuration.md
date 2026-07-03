# Configuration

`ReviewerConfig.from_env()` is the default configuration path used by
`python -m cora` and the `cora` console script. The environment variables
below are the supported adopter-facing knobs.

## Required Runtime Settings

| Variable | Purpose |
| --- | --- |
| `GH_REPO` or `GITHUB_REPOSITORY` | Target repository as `owner/repo` |
| `PR_NUMBER` | Pull request number to review |
| `GH_TOKEN` | GitHub token used by the default reporter |
| `LLM_GATEWAY_KEY` | API key for the OpenAI-compatible endpoint |
| `LITELLM_BASE_URL` | OpenAI-compatible base URL |
| `REVIEW_MODEL` | Model alias sent to the endpoint |

`LITELLM_BASE_URL` is named for the LiteLLM gateway convention, but the
client expects any OpenAI-compatible API — it can point at LiteLLM, vLLM,
or another compatible gateway. See
[Running against a cloud provider](#running-against-a-cloud-provider)
for the Anthropic/Bedrock topology.

## Review Mode And Budgets

| Variable | Default behavior |
| --- | --- |
| `MAX_TOOL_ITERATIONS` | `0` gives quick mode; a positive value enables deep tool-loop review |
| `WALL_TIME_S` | Overrides the total review wall-time envelope |
| `T0_WALL_TIME_S` | Overrides primary review wall time |
| `T1_WALL_TIME_S` | Overrides continuation wall time |
| `AGENT_REVIEW_T1_CONTINUATION` | Enables T1 continuation when set to `true` |
| `T1_MODEL` | Model alias for T1 continuation |
| `AGENT_REVIEW_T1_MAX_ITERATIONS` | Iteration cap for T1 continuation |
| `AGENT_REVIEW_T2_DISAGREEMENT` | Enables second-opinion verdict resolution when set to `true` |
| `AGENT_REVIEW_T2_MODEL` | Model alias for T2 second opinion |
| `AGENT_REVIEW_T2_MAX_ITERATIONS` | Iteration cap for T2 second opinion |
| `AGENT_REVIEW_SKIP_T0` | Starts directly on T1 when set to `true` |
| `CORA_ESCALATION_TRIGGERS` | CSV of escalation triggers for the default ladder (`wall_hit`, `blocker`, `low_confidence`; default `wall_hit`) |
| `AGENT_REVIEW_PER_CALL_TIMEOUT_S` | Per-model-call timeout |

## Trigger Policy

Trigger policy is enabled by default. Public repositories should leave it
enabled.

| Variable | Purpose |
| --- | --- |
| `REVIEW_TRIGGER_ENFORCE` | Defaults to `true`; set `false` only for private migration windows |
| `REVIEW_TRIGGER_ALLOWED_ASSOCIATIONS` | CSV of trusted GitHub author associations |
| `REVIEW_TRIGGER_ALLOWED_AUTHORS` | CSV of trusted GitHub logins |
| `REVIEW_TRIGGER_APPROVE_LABEL` | Maintainer-applied opt-in label, default `cora:approved` |
| `REVIEW_TRIGGER_UNTRUSTED_ACTION` | `skip` or `comment-only` |
| `REVIEW_TRIGGER_FORK_ACTION` | `full`, `comment-only`, or `skip` |
| `REVIEW_TRIGGER_MAX_RUNS_PER_HOUR` | Best-effort hourly run cap |
| `REVIEW_TRIGGER_MAX_RUNS_PER_AUTHOR_PER_HOUR` | Best-effort per-author hourly run cap |

See [../SECURITY.md](../SECURITY.md) for the exact trust rules and fork
PR behavior.

## Retrieval And Context

| Variable | Purpose |
| --- | --- |
| `AGENT_REVIEW_RETRIEVAL_GLOB` | CSV of checkout-relative globs for local BM25 retrieval |
| `AGENT_REVIEW_SKIP_RETRIEVAL_LABELS` | CSV of classifier labels that skip retrieval |
| `AGENT_REVIEW_CACHE_DIR` | Retrieval cache directory |
| `AGENT_REVIEW_CACHE_TTL_S` | Retrieval cache TTL |
| `QDRANT_URL` | Qdrant endpoint for vector retrieval |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Qdrant collection name (default `cora-knowledge`) |
| `TEI_URL` | Text Embeddings Inference endpoint |
| `RERANKER_URL` | Reranker endpoint |
| `CLASSIFIER_LABEL` | Current classifier label for retrieval decisions |

Context injection is on by default and can be disabled with default-true
kill switches:

| Variable | Purpose |
| --- | --- |
| `AGENT_REVIEW_CONTEXT_INJECTION` | Master context-injection switch |
| `AGENT_REVIEW_CONTEXT_INJECTION_CI` | Inject CI state changes |
| `AGENT_REVIEW_CONTEXT_INJECTION_HEAD` | Inject PR head updates |
| `AGENT_REVIEW_CONTEXT_INJECTION_COMMENTS` | Inject new human comments |

Set any of those variables to the literal string `false` to disable that
piece.

## MCP And Tool Exposure

Deep review can attach MCP servers and expose tool subsets.

| Variable | Purpose |
| --- | --- |
| `MCP_URL` | Read-tool MCP server URL |
| `MCP_TOKEN` | Token for the read-tool MCP server |
| `MCP_ACTIONS_URL` | Actions MCP server URL |
| `MCP_ACTIONS_TOKEN` | Token for the actions MCP server |
| `WEB_FETCH_GATE_URL` | Fetch-gate endpoint for controlled web context |
| `REVIEWER_BROADEN_TOOLS` | Broadens the default tool exposure when set to `true` |

## Reporting And Write Paths

| Variable | Purpose |
| --- | --- |
| `REVIEW_CHECK_RUN_NAME` | Name of the verdict check-run |
| `REVIEW_USE_GITHUB_REVIEW` | Posts a first-class GitHub PR Review when set to `true` |
| `REVIEW_PROPOSE_PATCH_DISPATCH` | Enables validated patch-writing dispatch when set to `true` |
| `AGENT_REVIEW_PATCH_ESCALATION` | Default-true patch-escalation verifier kill switch |
| `CORA_GH_TOKEN` | Optional GitHub App installation token for App-attributed reporting and patch dispatch |

Patch-writing is disabled by default. Before enabling it, read
[../SECURITY.md](../SECURITY.md), especially the token scopes and workflow
wiring sections.

## Observability And Debugging

| Variable | Purpose |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enables OpenTelemetry export |
| `OTEL_SERVICE_VERSION` | Service version attached to traces |
| `GRAFANA_BASE_URL` | Base URL for trace/dashboard links |
| `GRAFANA_DASHBOARD_PATH` | Dashboard path appended to the base URL for per-PR drilldown links (empty = no link) |
| `REVIEWER_TRANSCRIPT_DIR` | Directory for saved model transcripts |
| `REVIEWER_TRANSCRIPT_SOURCE` | Transcript source label |
| `AGENT_REVIEW_EVAL_OUTPUT_DIR` | Directory for evaluation artifacts |
| `AGENT_REVIEW_LOOP_GUARD_BOT_LOGIN` | Bot login filtered from comment-loop refreshes |

Deployments with scale-from-zero inference backends can hide cold starts
behind review setup:

| Variable | Purpose |
| --- | --- |
| `CORA_PRETRIGGER_WARMUP_MODELS` | CSV of model aliases for cold-start warmup (empty default = disabled) |

## Running against a cloud provider

cora speaks one dialect — OpenAI Chat Completions — to whatever
`LITELLM_BASE_URL` points at. Cloud models work today by letting the
gateway do the translation: run a [LiteLLM proxy](https://docs.litellm.ai/)
and route cora's model aliases to hosted models. No cora configuration
changes beyond the aliases.

```yaml
# litellm proxy config: cora's aliases → Anthropic / Bedrock
model_list:
  - model_name: review          # T0 — the primary review tier
    litellm_params:
      model: anthropic/claude-sonnet-5
      api_key: os.environ/ANTHROPIC_API_KEY
      drop_params: true   # cora sends temperature; current Claude models
                          # reject non-default sampling params — let the
                          # gateway drop them
  - model_name: core            # T1 — the escalation tier
    litellm_params:
      model: anthropic/claude-opus-4-8
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: alt-reviewer    # T2 — keep this a *different* model
    litellm_params:              #      family; that diversity is the point
      model: bedrock/anthropic.claude-opus-4-8   # or another vendor entirely
      aws_region_name: us-east-1
```

```bash
export LITELLM_BASE_URL=http://localhost:4000
export LLM_GATEWAY_KEY=sk-...        # the proxy's key
export REVIEW_MODEL=review
```

Notes for cloud deployments:

- **The tier ladder maps directly** — a cheap/fast T0 escalating to a
  stronger T1 on wall-hit (or on `blocker`/`low_confidence` via
  `CORA_ESCALATION_TRIGGERS`) is the Sonnet→Opus shape the escalation
  module documents.
- **Self-hosted-only knobs self-disarm.** Reasoning-leak stripping,
  spiral recovery, `AGENT_REVIEW_ENABLE_THINKING`, and the cold-start
  pretrigger exist for self-hosted reasoning backends; against a hosted
  API they are no-ops or default-off.
- **Budgets were tuned for self-hosted throughput.** Consider lowering
  the per-call output caps and wall-time splits — hosted models are
  faster and rarely wall-hit.
- **The check-run "Backend" chip** comes from `x-litellm-*` response
  headers: it keeps working through a LiteLLM proxy and is silently
  omitted behind other gateways.
- **Cost intuition:** a typical deep review runs tens of thousands of
  input tokens and ~1–2K output tokens per tier — fractions of a dollar
  per review on current hosted pricing. Enabling the gateway's prompt
  caching helps: cora's agent loop has a frozen system prompt and an
  append-only history, so multi-turn prefixes are highly cacheable.

Direct provider SDK support (Anthropic / Bedrock without a gateway) is
planned as a provider seam in the agent factory — see the issue tracker.
