"""BM25-style sparse encoding shared by the indexer + retriever.

Stdlib-only (matches the rest of the agent-review surface). Used by:
  - the knowledge indexer — encodes each chunk's body at
    index time into a Qdrant named-sparse vector ("sparse").
  - ``cora.core.retrieval`` — encodes the retrieval
    query at search time into the same vector space.

Token IDs are derived via ``zlib.crc32`` modulo a large prime so they
are deterministic across processes / Python versions / hash-seed
settings (Python's built-in ``hash()`` is randomised per process — fine
for in-memory dicts, not fine for cross-process sparse-vector
alignment). 1_000_003 is the modulus — a shared indexer/retriever
convention that stays well under Qdrant's 2**32 - 1 ceiling.

For v1 we use uniform IDF (1.0 per unique token) and BM25 term-
frequency saturation only. Proper corpus-derived IDF can land later
once trace data gives us a feel for what the lexical channel
actually contributes.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Shared modulus + BM25 defaults. ``avg_len`` is a rough corpus average
# for the DECISIONS/notes/AGENTS corpus (~500 tokens per chunk after
# tokenisation); off-by-2x is fine for retrieval ranking.
TOKEN_ID_MODULUS = 1_000_003
BM25_K1 = 1.2
BM25_B = 0.75
BM25_AVG_LEN = 500.0


def _stable_token_id(token: str) -> int:
    """Process-independent token-to-id mapping. crc32 is deterministic
    across runs/interpreters; hash() is not (PYTHONHASHSEED randomises)."""
    return zlib.crc32(token.encode("utf-8")) % TOKEN_ID_MODULUS


def _bm25_sparse(
    text: str,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
    avg_len: float = BM25_AVG_LEN,
) -> tuple[list[int], list[float]]:
    """Encode ``text`` as a Qdrant-compatible sparse vector.

    Returns ``(indices, values)``: parallel lists where ``indices[i]``
    is a stable token id and ``values[i]`` is the BM25-saturated weight
    for that token in this document. Empty input → ``([], [])``.

    The encoding is INTENTIONALLY simple — uniform IDF, no stop-word
    list. Refinement (corpus-derived IDF, language-aware stemming) is
    deferred until the trace data tells us if the lexical channel
    is pulling its weight.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return [], []
    doc_len = len(tokens)
    tf = Counter(tokens)
    # Aggregate at INDEX level, not token level. crc32(token) % MODULUS
    # can collide — two different tokens mapping to the same index. A
    # full corpus reindex hit Qdrant's "indices must be unique" 422 on
    # exactly this. We merge collided tokens by summing their BM25
    # contributions, which keeps total mass while preserving uniqueness.
    # Lossy at the collision boundary but matches what every BM25-via-
    # hashing implementation does in practice (e.g., sklearn's
    # HashingVectorizer); the trace data will tell us if the collision
    # rate is high enough to motivate a larger modulus or proper
    # vocab-id mapping.
    by_index: dict[int, float] = {}
    for token, count in tf.items():
        # BM25 term-frequency saturation: as ``count`` grows, the
        # contribution to the score grows sub-linearly. ``b`` controls
        # how much we penalise long documents (1.0 = full penalty,
        # 0.0 = none); 0.75 is the standard value.
        score = (count * (k1 + 1)) / (
            count + k1 * (1 - b + b * doc_len / avg_len)
        )
        idx = _stable_token_id(token)
        by_index[idx] = by_index.get(idx, 0.0) + float(score)
    # Sort by index for deterministic output (helps caching + makes
    # collisions easier to spot during debugging).
    items = sorted(by_index.items())
    indices = [i for i, _ in items]
    values = [v for _, v in items]
    return indices, values
