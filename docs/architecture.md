# Architecture

cora's public surface is small:

```python
from cora import ReviewerConfig, run_review

result = run_review(ReviewerConfig.from_env())
```

`python -m cora` and the `cora` console script build that config from the
environment and run one review with the default providers.

## Review Flow

At a high level, a run does this:

1. Build `ReviewerConfig` from explicit values or environment variables.
2. Apply trigger policy to decide whether the review can run and at what
   capability level.
3. Gather PR context, including diff, metadata, repository snippets, and
   optional retrieval context.
4. Run the configured model path: quick review, deep tool-loop review,
   continuation, and optional second opinion depending on settings.
5. Strip or suppress unsafe reasoning leakage, parse the verdict, and
   compose the final review body.
6. Report through the configured reporter, normally a GitHub sticky
   comment plus verdict check-run.

The entrypoint treats review verdicts and policy skips as completed runs.
The GitHub comment and check-run carry the review signal; the process exits
nonzero only for hard failures.

## Provider Seams

cora keeps deployment-specific behavior behind replaceable interfaces:

| Seam | Purpose |
| --- | --- |
| `RetrievalProvider` | Supplies extra context, from no-op to local glob/BM25 to TEI/Qdrant retrieval |
| `GitProvider` | Provides read-only repository lookups for tools such as `grep_repo` and `git_show` |
| `Reporter` | Owns side effects such as GitHub comments, check-runs, review comments, and patch suggestions |
| `SecondOpinion` | Optionally runs an independent model and folds verdict disagreement into the final result |

The defaults are a working deployment out of the box; adopters replace
the outer integrations without forking the engine.

## Project Layout

```text
src/cora/
|-- review/        # run_review orchestration phases
|-- core/          # the review engine internals
|-- providers/     # retrieval, git-host, and reporter seams
|-- prompts/       # packaged default prompts
|-- config.py      # ReviewerConfig dataclass and from_env wiring
|-- result.py      # structured review result
|-- trigger.py     # trigger policy and capability decisions
`-- __main__.py    # python -m cora
```

The `core/` package is implementation detail. Prefer the top-level
`cora` imports for adopter-facing code.

## Security Shape

cora reviews attacker-controlled PR content with a tool-using model. The
important security boundary is capability gating:

- PR content is treated as data, not instructions.
- cora reads diffs and repository content; it does not need to execute PR
  code.
- trigger policy is enforced by default.
- untrusted and fork PRs can be skipped or degraded to comment-only mode.
- patch-writing is off unless `REVIEW_PROPOSE_PATCH_DISPATCH=true`.

The complete threat model and workflow guidance live in
[../SECURITY.md](../SECURITY.md).
