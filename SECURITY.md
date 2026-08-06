# Security

cora is a tool-using, comment-posting (and optionally patch-proposing)
LLM agent that you point at pull requests. PRs are attacker-controlled
input. This document is the threat model, the controls cora ships, and
the wiring you must (not) use. Read it before running cora anywhere a
stranger can open a PR.

## Threat model

Three threats dominate for an LLM PR reviewer:

1. **Cost / denial-of-wallet.** Every PR (and every push to one) fires
   an LLM call. On a public repo, spam PRs fan out into unbounded
   GPU/API spend.
2. **Prompt injection.** The PR title, body, diff, commit messages,
   file paths, and comments are untrusted input *into a tool-using
   agent*. An injected instruction can try to steer tool calls (exfil
   via a fetch tool), launder false verdicts, or — where write
   capability exists — weaponise the patch-proposal path.
3. **Pwn-request.** The classic GitHub Actions foot-gun: a
   `pull_request_target`-triggered workflow that checks out **fork
   code** with **secrets in the environment**. This is a workflow
   wiring mistake, not an engine behaviour — and the most important
   thing this document tells you is *don't do it*.

## cora's posture: diff-as-data

cora **never checks out and executes PR code**. It reads the diff and
repo content as *data* — `grep_repo` and `git_show` are read-only
lookups, and the review runs against the checkout your workflow already
made. There is no `npm install`, no test execution, no build of the
PR's code inside the reviewer. That single property is what makes
reviewing fork PRs tractable at all.

The packaged system prompts carry an explicit untrusted-input section:
PR content is data to review, never instructions to follow; injection
attempts are to be flagged as Blockers, not obeyed. Treat the prompt
framing as harm *reduction* — the trigger policy below is the actual
gate.

**Untrusted content that is not the diff.** Some context reaches the
model from outside the PR: a linked issue thread, fetched release
notes, MCP tool output. On a public repo that text is world-writable —
filing an issue takes no privilege at all — so it is wrapped in an
`<untrusted-content>` block, and two properties must hold together:

- **The boundary cannot be forged.** Wrapping alone is not containment.
  Text carrying a literal `</untrusted-content>` would otherwise end
  the block early, and everything after it would read as ordinary
  prompt text. Fetched text is neutralized so it holds no usable
  boundary tag, *and* the block carries a per-review random `id` on
  both tags, so a payload that survived neutralization still cannot
  guess its terminator.
- **The model is told what the tag means.** A wrapper the prompt never
  explains is decoration. `deep.md` and `quick.md` both state that
  `<untrusted-content>` marks third-party data, and that text inside it
  claiming to be a system note, a prior approval, or a closing tag is
  untrusted content lying about its own status.

Anything that injects fetched text into the prompt must do both. Adding
the wrapper without the prompt rule, or the prompt rule without
escaping, leaves the boundary asserted rather than enforced.

## TriggerPolicy — who may fire a review, at what capability

`ReviewerConfig.trigger` carries a `TriggerPolicy`
(`cora.trigger.TriggerPolicy`). Enforced, it is **default-deny** with a
capability ceiling:

- An author is **trusted** when ANY of: their `author_association` is
  in `allowed_associations` (default `OWNER`/`MEMBER`/`COLLABORATOR`);
  their login is in `allowed_authors`; or a maintainer applied the
  opt-in label (`approve_label`, default `cora:approved` — label
  application is permission-gated by GitHub, so it's a human vouch).
- Everyone else gets `untrusted_action`: **`skip`** (default — the run
  posts a skip comment, the verdict check-run concludes `cancelled`,
  and a verdict-gating merge aggregator stays red) or
  **`comment-only`** — a degraded run: quick mode (no MCP tools, no
  local repo tools), `propose_patch` dispatch suppressed. The verdict
  comment and check-run are the only writes a degraded run makes.
- **Fork PRs** get a ceiling on top (`fork_action`, default
  `comment-only`): even a trusted author's fork PR runs without tools
  or writes. `skip` refuses fork PRs outright; `full` defers to the
  trust rules.
- **Rate caps** (`max_runs_per_hour`, `max_runs_per_author_per_hour`)
  bound the spend a burst can cause, counted against recent runs of
  the same workflow. The probe is best-effort (it soft-fails open on
  API errors) — treat the caps as a cost guard, not a security
  boundary. The deny gate above is the security boundary, and *it*
  fails closed (an author whose association can't be fetched is
  untrusted).

Environment wiring (all optional; `ReviewerConfig.from_env()`):

| env var | field |
| --- | --- |
| `REVIEW_TRIGGER_ENFORCE` | `enforce` (`false` to opt out) |
| `REVIEW_TRIGGER_ALLOWED_ASSOCIATIONS` | csv → `allowed_associations` |
| `REVIEW_TRIGGER_ALLOWED_AUTHORS` | csv → `allowed_authors` |
| `REVIEW_TRIGGER_APPROVE_LABEL` | `approve_label` |
| `REVIEW_TRIGGER_UNTRUSTED_ACTION` | `untrusted_action` (`skip` / `comment-only`) |
| `REVIEW_TRIGGER_FORK_ACTION` | `fork_action` (`full` / `comment-only` / `skip`) |
| `REVIEW_TRIGGER_MAX_RUNS_PER_HOUR` | `max_runs_per_hour` |
| `REVIEW_TRIGGER_MAX_RUNS_PER_AUTHOR_PER_HOUR` | `max_runs_per_author_per_hour` |
| `REVIEW_PROPOSE_PATCH_DISPATCH` | patch writes (`true` to enable) |

`enforce` defaults to **true**. Private deployments that need the legacy
pre-gate behaviour during migration can set `REVIEW_TRIGGER_ENFORCE:
"false"`, but public repositories should leave it enabled.

## Canonical workflow wiring

Safe shape — `pull_request` trigger, no fork secrets exposure, policy
enforced:

```yaml
name: cora
on:
  pull_request:
    types: [opened, synchronize, reopened, labeled]

permissions:
  contents: read
  issues: write          # sticky review comment
  pull-requests: write   # PR metadata and optional Review API path
  checks: write          # the verdict check-run

jobs:
  review:
    # With `pull_request`, fork PRs run with a read-only token and NO
    # secrets — the job below would skip at the missing-key preflight.
    # This gate just avoids burning a runner to find that out.
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4   # YOUR trusted ref, not fork code
      - run: pip install cora && python -m cora
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
          GH_REPO: ${{ github.repository }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          LLM_GATEWAY_KEY: ${{ secrets.LLM_GATEWAY_KEY }}
          REVIEW_TRIGGER_ENFORCE: "true"
          REVIEW_TRIGGER_MAX_RUNS_PER_HOUR: "30"
          REVIEW_TRIGGER_MAX_RUNS_PER_AUTHOR_PER_HOUR: "6"
```

To review **fork PRs from untrusted authors**, keep the `pull_request`
trigger and accept the skip, or route through the approval label: a
maintainer reads the PR, applies `cora:approved`, and the `labeled`
event re-fires the workflow — now passing the gate (still under the
fork comment-only ceiling).

### What NOT to do

**Never** wire cora (or anything with secrets) like this:

```yaml
on: pull_request_target          # ⚠️ secrets + write token, attacker event
# ...
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}   # ⚠️ fork code
```

`pull_request_target` runs with your secrets and a write token on an
event the attacker controls; checking out the fork head hands your
credentials to any PR that edits a build script. cora never needs fork
code checked out — it reads the diff over the API. If you believe you
need `pull_request_target`, you are almost certainly reaching for the
approval-label flow above instead.

## Capability notes

- **Write ceiling.** `propose_patch` (inline suggestions / draft fix
  PRs) dispatch is opt-in via `REVIEW_PROPOSE_PATCH_DISPATCH=true`.
  Even then it only dispatches on non-degraded runs, and its own
  validator denies `.github/` paths so a proposed patch can't edit
  workflows. Degraded runs post the verdict comment + check-run,
  nothing else.
- **Token scope.** The engine works with the workflow's `GITHUB_TOKEN`
  using the least privilege in the canonical workflow:
  `contents: read`, `pull-requests: write`, and `checks: write`. In
  GitHub Actions, `pull-requests: write` is enough for PR metadata/diff
  reads and PR review comments, while `issues: write` is usually covered
  by the repository-scoped `GITHUB_TOKEN` for issue comments. For a
  fine-grained PAT or GitHub App token used via local `GH_TOKEN`, grant:

  | capability | fine-grained permission |
  | --- | --- |
  | read PR metadata, labels, author association, head/base refs | Pull requests: read |
  | fetch PR diffs | Pull requests: read |
  | read repository file contents for patch validation | Contents: read |
  | read check-runs / CI context | Checks: read |
  | create/update the verdict check-run | Checks: read/write |
  | post/edit the sticky review comment | Issues: read/write |
  | remove labels such as `automerge` or add escalation labels | Issues: read/write |
  | post inline PR review comments / suggestions | Pull requests: read/write |

  The optional GitHub App token (`CORA_GH_TOKEN`) is needed when you
  want comments and check-runs attributed to the App instead of
  `github-actions[bot]`, and for draft-fix PR dispatch. In GitHub
  Actions, store the App credentials as `CORA_APP_ID` and
  `CORA_APP_PRIVATE_KEY`; the workflow mints `CORA_GH_TOKEN` from
  those secrets and falls back to `GITHUB_TOKEN` when they are absent.
  If you enable draft-fix PRs, that token also needs
  `contents: read/write` to create/update branches and
  `pull-requests: read/write` to open the draft PR.
- **Filesystem read surface.** The local `grep_repo` / `git_show` tools
  read from the checkout, which IS the reviewed PR's tree — so any path
  the tools derive from repository content is attacker-influenced. Two
  rules follow. A corpus root must be confined: `DEP_SOURCE_ROOTS` is
  operator-set deployment config and trusted as given, but an
  auto-detected root (`vendor/`, `node_modules/`) is refused when it is
  a symlink or resolves outside the checkout — `Path.is_dir()` follows
  symlinks and `os.walk` descends its top-level argument regardless of
  `followlinks`, so `vendor -> /` would otherwise expose the runner
  filesystem. And symlinks encountered during a walk are skipped rather
  than followed. This matters because matched lines are quoted into a
  public verdict comment: a read escape is a disclosure, not just a
  traversal.
- **Merge gating.** The verdict check-run concludes `cancelled` on a
  policy-denied run, so a required-check aggregator that gates on it
  (a required-check aggregator workflow) keeps blocking; a human
  override label is the documented escape hatch.

## Reporting a vulnerability

Open a GitHub security advisory on this repository (Security →
"Report a vulnerability"), or email the maintainer privately. Please
don't open a public issue for an exploitable report.
