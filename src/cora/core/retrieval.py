"""TEI + Qdrant retrieval pipeline for PR-review context pre-pack.

Pipeline shape:

  - Query reformulation extracts identifiers (DEC ids, domain vocab),
    changed file paths, and the +/- lines from the diff. The structured
    text feeds both the dense embedder and the sparse BM25 encoder.
  - Hybrid retrieval — Qdrant query against the named ``dense`` vector
    AND the named ``sparse`` vector, fused with RRF.
  - Retired-DEC filter pushed to Qdrant via a payload ``must_not`` clause.
  - Two-stage rerank — cross-encoder on snippets (32 → 16), then on full
    bodies (16 → top_k). Body fetch happens BETWEEN the two stages so
    we only pay it for the 16 survivors.
  - Same-PR re-run disk cache, 5-min TTL, keyed on the query text.
  - Per-PR retrieval trace returned alongside the docs so the caller can
    push it to Loki for dashboard/measurement work.

Soft-fail boundary preserved throughout. Any HTTP failure (TEI down,
Qdrant down, sparse-search 4xx on a collection that hasn't been
migrated to the named-vector schema yet) falls back to the next-best
behaviour; the pipeline never blocks the review. The trace records
which path was taken so dashboard panels can show the fallback rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core._bm25 import _bm25_sparse
from cora.core.config import (
    DEFAULT_RETRIEVAL_VOCAB,
    QDRANT_COLLECTION,
    RETRIEVAL_CACHE_DIR,
    RETRIEVAL_CACHE_TTL_S,
    RETRIEVAL_DOC_CHAR_CAP,
    RETRIEVAL_OVERFETCH_K,
    RETRIEVAL_QUERY_CHAR_CAP,
    RETRIEVAL_STAGE1_SURVIVORS,
    RETRIEVAL_TIMEOUT_S,
    RETRIEVAL_TOP_K,
)

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


_DEC_HEADING_RE = re.compile(
    r"^##\s+(?P<id>DEC-\d+)\s*[—\-:]\s*(?P<title>.+?)\s*$", re.MULTILINE
)
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------

# Structural identifier patterns — generic reference *formats*, not
# deployment vocabulary, so they stay hardcoded:
#   - DEC IDs — exact ADR references in titles/bodies/diffs.
#   - PR references — "#1234"-style citations resolve via notes /
#     post-mortems that cite the same PR by number.
_STRUCTURAL_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bDEC-\d{1,4}\b"),
    re.compile(r"#\d{3,5}"),
)


@lru_cache(maxsize=8)
def _vocab_regex(vocab: tuple[str, ...]) -> re.Pattern[str] | None:
    """Compile a word-bounded, case-insensitive alternation over `vocab`.

    Entries are regex fragments (e.g. ``fetch[-_]?gate``), joined as-is.
    Returns None for an empty vocab so the caller skips it. Cached because
    the vocab tuple is stable across a process (the config default or one
    adopter override), so we compile each distinct vocab at most once.
    """
    if not vocab:
        return None
    return re.compile(r"\b(?:" + "|".join(vocab) + r")\b", re.IGNORECASE)

# Unified diff structure markers. ``+++ b/path/to/file`` gives the
# new-side path for each changed file; ``+``/``-`` lines (excluding the
# four-byte header markers) are the actual edits.
_CHANGED_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
# Match +/- lines but NOT the +++/--- file headers. The negative
# lookahead rejects the header markers; ordinary diff text starting
# with + or - is the +/- change content.
_CHANGED_LINE_RE = re.compile(r"^[+-](?!\+\+|--)(.+)$", re.MULTILINE)


def extract_identifiers(
    text: str, *, vocab: tuple[str, ...] | None = None
) -> list[str]:
    """Pull DEC IDs, PR references, and domain identifiers out of text.

    `vocab` is the domain term list (regex fragments); None falls back to
    the engine default the config field mirrors. DEC/PR-ref formats always
    fire regardless of vocab.

    Dedup is case-insensitive (we don't want both ``Cilium`` and
    ``cilium`` filling the identifier list) but the FIRST occurrence's
    casing is preserved — looks better in the trace.
    """
    if vocab is None:
        vocab = DEFAULT_RETRIEVAL_VOCAB
    patterns: list[re.Pattern[str]] = list(_STRUCTURAL_REGEXES)
    vocab_re = _vocab_regex(vocab)
    if vocab_re is not None:
        patterns.append(vocab_re)
    seen: set[str] = set()
    out: list[str] = []
    for pattern in patterns:
        for m in pattern.finditer(text):
            tok = m.group(0)
            lower = tok.lower()
            if lower in seen:
                continue
            seen.add(lower)
            out.append(tok)
    return out


def parse_changed_paths(diff_text: str, *, cap: int = 50) -> list[str]:
    """Extract the new-side paths from a unified diff. Capped to avoid a
    50-file Renovate PR drowning the query in lockfile noise."""
    return _CHANGED_PATH_RE.findall(diff_text)[:cap]


def extract_changed_lines(diff_text: str, *, cap: int = 50) -> list[str]:
    """Extract +/- content lines from a unified diff (skipping the
    +++/--- file-header markers)."""
    return _CHANGED_LINE_RE.findall(diff_text)[:cap]


def build_retrieval_query(
    metadata: dict,
    diff_text: str,
    *,
    cfg: "ReviewerConfig | None" = None,
) -> dict:
    """Build the retrieval query as both raw text + structured signal.

    Returns a dict with:
      ``text``           — concatenated query for the dense embedder
      ``identifiers``    — DEC IDs + domain vocab from title/body/diff
      ``changed_paths``  — new-side paths from the diff
      ``title``          — PR title (for trace + debugging)

    The ``text`` field is what the embedder sees; the structured fields
    are for the trace + downstream consumers (sparse vector also derives
    from ``text``). `cfg` supplies the query char cap; None falls back
    to the engine constant the config default mirrors.
    """
    query_char_cap = (
        cfg.retrieval_query_char_cap if cfg is not None else RETRIEVAL_QUERY_CHAR_CAP
    )
    vocab = cfg.retrieval_vocab if cfg is not None else None
    title = (metadata.get("title") or "").strip()
    body = (metadata.get("body") or "").strip()
    ids = extract_identifiers(f"{title}\n{body}\n{diff_text}", vocab=vocab)
    paths = parse_changed_paths(diff_text)
    lines = extract_changed_lines(diff_text, cap=50)

    parts: list[str] = []
    if title:
        parts.append(title)
    if ids:
        # Stable ordering for the trace — `extract_identifiers` already
        # dedups but emits in scan order; that's fine for the embedder.
        parts.append("Identifiers: " + " ".join(ids))
    if paths:
        parts.append("Files: " + " ".join(paths))
    if body:
        parts.append(body[:1000])
    if lines:
        parts.append("\n".join(lines))

    # The cap is intentionally 2× the query char cap — the v1
    # heuristic capped at QUERY_CHAR_CAP; the structured-query shape
    # has more high-signal content per char (identifiers + paths +
    # +/- lines, not file headers), so allowing 2× preserves recall on
    # the longer high-quality content. Embedder truncates internally.
    text = "\n\n".join(parts)[: query_char_cap * 2]
    return {
        "text": text,
        "identifiers": ids,
        "changed_paths": paths,
        "title": title,
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_json_post(
    url: str,
    body: dict,
    headers: dict | None = None,
    *,
    timeout_s: int | None = None,
) -> dict | list:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    timeout = timeout_s if timeout_s is not None else RETRIEVAL_TIMEOUT_S
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(repo: str, pr_number: str, head_sha: str, query_text: str) -> str:
    """Stable 32-char hex key for the (repo, PR, head, query) tuple.
    Includes the head SHA so a force-push that changes the diff misses
    cache even though the query MIGHT have ended up identical."""
    h = hashlib.sha256()
    h.update(
        f"{repo}\x00{pr_number}\x00{head_sha}\x00{query_text}".encode("utf-8")
    )
    return h.hexdigest()[:32]


def _cache_dir(cfg: "ReviewerConfig | None") -> Path:
    """Resolve the cache dir: the threaded config wins; the legacy path
    (no config) re-reads the env override at call time, exactly as
    before the threading."""
    if cfg is not None:
        return Path(cfg.retrieval_cache_dir)
    return Path(
        os.environ.get("AGENT_REVIEW_CACHE_DIR") or str(RETRIEVAL_CACHE_DIR)
    )


def _cache_ttl_s(cfg: "ReviewerConfig | None") -> int:
    """Resolve the cache TTL — same config-wins / env-fallback split as
    `_cache_dir`."""
    if cfg is not None:
        return int(cfg.retrieval_cache_ttl_s)
    return int(
        os.environ.get("AGENT_REVIEW_CACHE_TTL_S") or str(RETRIEVAL_CACHE_TTL_S)
    )


def _cache_get(
    key: str, *, cfg: "ReviewerConfig | None" = None
) -> tuple[list[dict] | None, float | None]:
    """Read a cached top-K. Returns ``(docs, age_s)`` on hit, ``(None,
    None)`` on miss / expiry / read error. Read errors are soft-failed —
    a corrupted cache file shouldn't block the review."""
    p = _cache_dir(cfg) / f"{key}.json"
    if not p.is_file():
        return None, None
    age = time.time() - p.stat().st_mtime
    if age > _cache_ttl_s(cfg):
        return None, None
    try:
        return json.loads(p.read_text(encoding="utf-8")), age
    except Exception:  # noqa: BLE001 — soft-fail, corrupt cache file
        return None, None


def _cache_put(
    key: str, docs: list[dict], *, cfg: "ReviewerConfig | None" = None
) -> None:
    """Write the cache file. Best-effort — any failure (disk full,
    permission, etc.) is swallowed; the review proceeds normally."""
    cache_dir = _cache_dir(cfg)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(
            json.dumps(docs), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 — soft-fail
        pass


# ---------------------------------------------------------------------------
# Embedding + Qdrant
# ---------------------------------------------------------------------------


def embed_query(
    text: str, tei_url: str, *, timeout_s: int | None = None
) -> list[float]:
    out = _http_json_post(
        f"{tei_url.rstrip('/')}/embed",
        {"inputs": [text], "normalize": True, "truncate": True},
        timeout_s=timeout_s,
    )
    if not isinstance(out, list) or not out or not isinstance(out[0], list):
        raise RuntimeError(f"TEI /embed: unexpected shape {type(out)}")
    return out[0]


def _retired_filter_clause() -> dict:
    """Qdrant payload filter: exclude points whose payload has
    ``retired: true``. Points without the field (notes / AGENTS chunks,
    or any chunk indexed before the retired flag existed) match
    the filter — Qdrant ``match`` over a missing field is False, so the
    ``must_not`` clause keeps them."""
    return {"must_not": [{"key": "retired", "match": {"value": True}}]}


def qdrant_search_dense(
    vector: list[float],
    qdrant_url: str,
    api_key: str,
    top_k: int,
    *,
    collection: str | None = None,
    timeout_s: int | None = None,
) -> list[dict]:
    """Query the named ``dense`` vector. Filters retired DECs at the
    server side so they don't consume top-K slots."""
    body = {
        "query": vector,
        "using": "dense",
        "limit": top_k,
        "with_payload": True,
        "filter": _retired_filter_clause(),
    }
    out = _http_json_post(
        f"{qdrant_url.rstrip('/')}/collections/"
        f"{collection or QDRANT_COLLECTION}/points/query",
        body,
        headers={"api-key": api_key},
        timeout_s=timeout_s,
    )
    if not isinstance(out, dict):
        raise RuntimeError(f"Qdrant /points/query (dense): unexpected shape {type(out)}")
    return (out.get("result") or {}).get("points") or []


def qdrant_search_sparse(
    sparse_indices: list[int],
    sparse_values: list[float],
    qdrant_url: str,
    api_key: str,
    top_k: int,
    *,
    collection: str | None = None,
    timeout_s: int | None = None,
) -> list[dict]:
    """Query the named ``sparse`` vector. Same retired-filter as the
    dense search. Will fail with a Qdrant 4xx if the collection doesn't
    yet carry a sparse half (not yet reindexed with named sparse
    vectors); the caller is expected to catch and fall back to
    dense-only."""
    body = {
        "query": {"indices": sparse_indices, "values": sparse_values},
        "using": "sparse",
        "limit": top_k,
        "with_payload": True,
        "filter": _retired_filter_clause(),
    }
    out = _http_json_post(
        f"{qdrant_url.rstrip('/')}/collections/"
        f"{collection or QDRANT_COLLECTION}/points/query",
        body,
        headers={"api-key": api_key},
        timeout_s=timeout_s,
    )
    if not isinstance(out, dict):
        raise RuntimeError(f"Qdrant /points/query (sparse): unexpected shape {type(out)}")
    return (out.get("result") or {}).get("points") or []


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    *,
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists by RRF.

    Standard RRF: ``score(d) = sum_i 1 / (k + rank_i(d))`` across the
    input rankings. Ties broken by giving dense a tiny edge (0.99x on
    the sparse contribution) because dense is the higher-quality signal
    in our setup and we want dense to win when both rank a doc at the
    same position. Dedup is by Qdrant point id; first occurrence's
    payload wins.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits):
        pid = str(hit.get("id"))
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads[pid] = hit
    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.get("id"))
        scores[pid] = scores.get(pid, 0.0) + 0.99 / (k + rank + 1)
        payloads.setdefault(pid, hit)
    return [
        payloads[pid]
        for pid, _ in sorted(scores.items(), key=lambda p: p[1], reverse=True)
    ]


def tei_rerank(
    query: str,
    texts: list[str],
    reranker_url: str,
    *,
    timeout_s: int | None = None,
) -> list[float]:
    if not texts:
        return []
    out = _http_json_post(
        f"{reranker_url.rstrip('/')}/rerank",
        {"query": query, "texts": texts, "raw_scores": True},
        timeout_s=timeout_s,
    )
    if not isinstance(out, list):
        raise RuntimeError(f"TEI /rerank: unexpected shape {type(out)}")
    scores = [0.0] * len(texts)
    for r in out:
        scores[int(r["index"])] = float(r["score"])
    return scores


# ---------------------------------------------------------------------------
# Local body resolution (unchanged from v1)
# ---------------------------------------------------------------------------


def _slugify(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-")


def _decisions_body(repo_root: Path, dec_id: str) -> str | None:
    p = repo_root / "DECISIONS.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    matches = list(_DEC_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        if m["id"] != dec_id:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        return text[m.start():end].strip("\n")
    return None


def _note_body(repo_root: Path, name: str, anchor: str | None) -> str | None:
    p = repo_root / "notes" / f"{name}.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    if not anchor:
        return text.strip("\n")
    headings = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(headings):
        level = len(m.group(1))
        if _slugify(m.group(2)) != anchor:
            continue
        end = len(text)
        for j in range(i + 1, len(headings)):
            if len(headings[j].group(1)) <= level:
                end = headings[j].start()
                break
        return text[m.start():end].strip("\n")
    return None


def _agents_body(repo_root: Path, anchor: str) -> str | None:
    p = repo_root / "AGENTS.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    headings = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(headings):
        level = len(m.group(1))
        if _slugify(m.group(2)) != anchor:
            continue
        end = len(text)
        for j in range(i + 1, len(headings)):
            if len(headings[j].group(1)) <= level:
                end = headings[j].start()
                break
        return text[m.start():end].strip("\n")
    return None


def _resolve_doc_body(repo_root: Path, payload: dict) -> str | None:
    kind = payload.get("kind")
    key = payload.get("key") or ""
    if kind == "decision":
        return _decisions_body(repo_root, key.split(":", 1)[-1])
    if kind == "note":
        name_anchor = key.split(":", 1)[-1]
        if "#" in name_anchor:
            name, anchor = name_anchor.split("#", 1)
        else:
            name, anchor = name_anchor, None
        return _note_body(repo_root, name, anchor)
    if kind == "agents":
        anchor = key.split("#", 1)[-1] if "#" in key else None
        return _agents_body(repo_root, anchor) if anchor else None
    return None


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _hit_brief(hit: dict) -> dict:
    """Compact representation of a Qdrant hit for the trace JSON. Trims
    the payload (which can be ~400 chars of snippet) down to just the
    fields the trace needs."""
    p = hit.get("payload") or {}
    return {
        "key": p.get("key", ""),
        "kind": p.get("kind", ""),
        "score": float(hit.get("score", 0.0) or 0.0),
    }


def _count_by_kind(hits: list[dict]) -> dict[str, int]:
    """Count hits by ``payload.kind`` for the trace's per-stage
    breakdown. {} for empty input. Feeds a dashboard's corpus-
    shape breakdown — when a stage's count for a given kind drops to 0,
    something's off with that kind's indexing or query alignment."""
    counts: dict[str, int] = {}
    for h in hits:
        kind = ((h.get("payload") or {}).get("kind")) or "unknown"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def retrieve_relevant_docs(
    *,
    repo: str,
    pr_number: str,
    head_sha: str,
    repo_root: Path,
    metadata: dict,
    diff_text: str,
    qdrant_url: str,
    qdrant_api_key: str,
    tei_url: str,
    reranker_url: str,
    top_k: int | None = None,
    cfg: "ReviewerConfig | None" = None,
) -> tuple[list[dict], dict]:
    """Hybrid + two-stage-rerank retrieval pipeline.

    Returns ``(docs, trace)`` — ``docs`` is the top-K with full bodies
    (same shape the v1 ``retrieve_relevant_docs`` returned), ``trace``
    is a JSON-serialisable dict the caller pushes to Loki for the
    per-PR retrieval dashboard.

    Raises on first-stage failures (no API key, dense embed/search) so
    the caller's soft-fall-back to no-retrieval still works. Sparse-
    search failures are caught and logged in the trace (the collection
    may not yet have sparse vectors on first deploy).

    `cfg` supplies the pipeline tunables (overfetch / stage-1 survivor
    counts, per-doc cap, collection name, HTTP timeout, cache dir+TTL,
    and the `top_k` default); None falls back to the engine constants
    the config defaults mirror — bit-identical.
    """
    overfetch_k = cfg.retrieval_overfetch_k if cfg is not None else RETRIEVAL_OVERFETCH_K
    stage1_survivors = (
        cfg.retrieval_stage1_survivors
        if cfg is not None
        else RETRIEVAL_STAGE1_SURVIVORS
    )
    doc_char_cap = (
        cfg.retrieval_doc_char_cap if cfg is not None else RETRIEVAL_DOC_CHAR_CAP
    )
    collection = cfg.qdrant_collection if cfg is not None else QDRANT_COLLECTION
    timeout_s = cfg.retrieval_timeout_s if cfg is not None else RETRIEVAL_TIMEOUT_S
    if top_k is None:
        top_k = cfg.retrieval_top_k if cfg is not None else RETRIEVAL_TOP_K

    trace: dict = {
        "cache_hit": False,
        "sparse_unavailable": False,
        "stages": {},
    }
    t0 = time.monotonic()

    query = build_retrieval_query(metadata, diff_text, cfg=cfg)
    trace["query_chars"] = len(query["text"])
    trace["query_identifier_count"] = len(query["identifiers"])
    trace["changed_path_count"] = len(query["changed_paths"])
    trace["query_identifiers"] = query["identifiers"][:20]  # cap for the log line
    trace["query_changed_paths"] = query["changed_paths"][:20]

    if not query["text"].strip():
        trace["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        return [], trace

    # Cache check — keyed on the query TEXT so PRs that don't shift the
    # query across re-runs reuse the work.
    cache_key = _cache_key(repo, pr_number, head_sha, query["text"])
    cached, age = _cache_get(cache_key, cfg=cfg)
    if cached is not None:
        trace["cache_hit"] = True
        trace["cache_age_s"] = round(age or 0.0, 1)
        trace["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        return cached, trace

    if not qdrant_api_key:
        raise RuntimeError("QDRANT_API_KEY env not set")

    # Dense + sparse search. EITHER side may fail on the legacy
    # unnamed-vector collection shape (pre-reindex): the new dense query
    # uses `using: "dense"` which 400s against a legacy unnamed
    # collection, and sparse 400s because the named sparse vector
    # doesn't exist yet. Caught + degraded independently so we still
    # return SOME context whenever at least one side works.
    t_dense = time.monotonic()
    dense_vec = embed_query(query["text"], tei_url, timeout_s=timeout_s)
    dense_hits: list[dict] = []
    try:
        dense_hits = qdrant_search_dense(
            dense_vec, qdrant_url, qdrant_api_key, overfetch_k,
            collection=collection, timeout_s=timeout_s,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
        # Collection might not have named dense vectors yet (the
        # indexer hasn't been re-run with --force-recreate). Continue
        # with sparse-only (typically empty too on the legacy shape,
        # but better to soft-fall than to explode the pre-pack).
        trace["dense_unavailable"] = True
        trace["dense_error"] = f"{type(exc).__name__}: {exc}"
    trace["stages"]["dense_ms"] = int((time.monotonic() - t_dense) * 1000)

    sparse_indices, sparse_values = _bm25_sparse(query["text"])
    sparse_hits: list[dict] = []
    if sparse_indices:
        t_sparse = time.monotonic()
        try:
            sparse_hits = qdrant_search_sparse(
                sparse_indices,
                sparse_values,
                qdrant_url,
                qdrant_api_key,
                overfetch_k,
                collection=collection,
                timeout_s=timeout_s,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            # Collection might not have sparse vectors yet (the indexer
            # hasn't been re-run with --force-recreate). Continue with
            # dense-only and surface it in the trace.
            trace["sparse_unavailable"] = True
            trace["sparse_error"] = f"{type(exc).__name__}: {exc}"
            sparse_hits = []
        trace["stages"]["sparse_ms"] = int((time.monotonic() - t_sparse) * 1000)

    trace["dense_candidates"] = [_hit_brief(h) for h in dense_hits]
    trace["sparse_candidates"] = [_hit_brief(h) for h in sparse_hits]
    # Per-kind candidate breakdown — surfaces corpus-shape bias at a
    # glance in a dashboard. An early trace had 0 notes in top-K
    # despite thousands of note chunks indexed; this count makes
    # that pattern legible per-review rather than requiring digging
    # into each candidates list.
    trace["dense_kind_counts"] = _count_by_kind(dense_hits)
    trace["sparse_kind_counts"] = _count_by_kind(sparse_hits)

    merged = reciprocal_rank_fusion(dense_hits, sparse_hits)[:overfetch_k]
    trace["rrf_top_n"] = [_hit_brief(h) for h in merged]
    trace["rrf_kind_counts"] = _count_by_kind(merged)
    if not merged:
        trace["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        # Empty result — still write the cache so a re-run of the same
        # diff doesn't repeat the (mostly) free work.
        _cache_put(cache_key, [], cfg=cfg)
        return [], trace

    # Stage 1 rerank — snippet-only, narrow 32 → 16.
    t_rr1 = time.monotonic()
    snippets = [(h.get("payload") or {}).get("snippet", "") for h in merged]
    stage1_scores = tei_rerank(
        query["text"], snippets, reranker_url, timeout_s=timeout_s
    )
    stage1_ranked = sorted(
        zip(merged, stage1_scores), key=lambda p: p[1], reverse=True
    )[:stage1_survivors]
    trace["stages"]["rerank_stage1_ms"] = int((time.monotonic() - t_rr1) * 1000)
    trace["rerank_stage1_top_n"] = [
        {**_hit_brief(h), "score": float(s)} for h, s in stage1_ranked
    ]

    # Resolve full bodies for the 16 survivors. Hits where the local
    # body can't be resolved (file moved/removed between index time and
    # retrieve time, or a payload shape we don't recognise) drop out.
    docs_with_bodies: list[dict] = []
    for hit, s1_score in stage1_ranked:
        payload = hit.get("payload") or {}
        body = _resolve_doc_body(repo_root, payload)
        if not body:
            continue
        if len(body) > doc_char_cap:
            body = body[:doc_char_cap] + (
                f"\n\n…[truncated at {doc_char_cap} chars]…\n"
            )
        docs_with_bodies.append(
            {
                "kind": payload.get("kind", "?"),
                "key": payload.get("key", ""),
                "title": payload.get("title", ""),
                "path": payload.get("path", ""),
                "body": body,
                "stage1_score": float(s1_score),
                "_qdrant_id": str(hit.get("id")),
            }
        )

    # Stage 2 rerank — full bodies, narrow 16 → top_k. Costs more
    # tokens than stage 1 but only on the survivors.
    if docs_with_bodies:
        t_rr2 = time.monotonic()
        stage2_scores = tei_rerank(
            query["text"], [d["body"] for d in docs_with_bodies], reranker_url,
            timeout_s=timeout_s,
        )
        for d, s in zip(docs_with_bodies, stage2_scores):
            d["score"] = float(s)
        docs_with_bodies.sort(key=lambda d: d["score"], reverse=True)
        docs_with_bodies = docs_with_bodies[:top_k]
        trace["stages"]["rerank_stage2_ms"] = int((time.monotonic() - t_rr2) * 1000)

    trace["rerank_stage2_top_k"] = [
        {"kind": d["kind"], "key": d["key"], "score": d.get("score", 0.0)}
        for d in docs_with_bodies
    ]
    trace["elapsed_ms"] = int((time.monotonic() - t0) * 1000)

    # Strip internal-only fields (the Qdrant id, stage1 intermediate
    # score) before caching + returning. Keeps the cache file small and
    # matches what the prompt-assembly downstream consumer expects.
    cacheable: list[dict] = []
    for d in docs_with_bodies:
        out: dict = {k: v for k, v in d.items() if not k.startswith("_")}
        # stage1_score is intermediate signal — fine to keep in trace
        # but drop from the returned doc shape.
        out.pop("stage1_score", None)
        cacheable.append(out)
    _cache_put(cache_key, cacheable, cfg=cfg)

    return cacheable, trace


def format_retrieved_docs(docs: list[dict]) -> str:
    if not docs:
        return "_(no relevant context retrieved)_"
    chunks: list[str] = []
    for d in docs:
        chunks.append(f"### {d['title']}  _(path: `{d['path']}`)_")
        chunks.append("")
        chunks.append(d["body"])
        chunks.append("")
    return "\n".join(chunks).strip()
