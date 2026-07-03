"""`python -m cora` — the minimal workflow entrypoint.

Builds a `ReviewerConfig` from the environment (the workflow env
wiring) and runs one review with the default providers. The richer
`cora review` CLI comes later; this stays thin on purpose.

Exit codes: 0 on any completed run (including soft-fail skips — the
check-run conclusion carries the signal, so the process itself
soft-fails); nonzero only on a hard failure (missing PR identity, or
an unhandled exception out of the engine).
"""

from __future__ import annotations

import sys

from cora.config import ReviewerConfig
from cora.review import run_review


def main() -> int:
    cfg = ReviewerConfig.from_env()
    if not cfg.repo or not cfg.pr_number:
        print(
            "PR_NUMBER and GH_REPO/GITHUB_REPOSITORY must be set",
            file=sys.stderr,
        )
        return 2
    try:
        result = run_review(cfg)
    except Exception as exc:  # noqa: BLE001 — the hard-failure boundary
        print(f"cora: review failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"cora: review finished — conclusion={result.conclusion} "
        f"verdict={result.verdict or 'none'} "
        f"terminated={result.terminated_reason or 'clean'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
