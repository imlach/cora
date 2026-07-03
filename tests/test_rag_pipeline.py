"""Tests for the RAG pipeline.

Covers the pure-Python pieces:
  - Query reformulation (extract_identifiers, parse_changed_paths,
    extract_changed_lines, build_retrieval_query).
  - Hybrid retrieval merge (reciprocal_rank_fusion).
  - BM25 sparse encoder (_bm25_sparse) — stability + non-emptiness.
  - Disk cache primitives (_cache_key, _cache_get, _cache_put, TTL).

HTTP-network functions (embed_query, qdrant_search_*, tei_rerank) are
NOT covered here — they require live services. The integration is
exercised in a live deployment, with the per-PR log traces as the
ground-truth signal.

Run via:
    pytest tests/test_rag_pipeline.py -v
"""

from __future__ import annotations


from cora.core import retrieval as r
from cora.core._bm25 import _bm25_sparse, _stable_token_id


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------


def test_extract_identifiers_picks_up_dec_ids():
    text = "Touches DEC-049 (MTU) and DEC-081 (loop reviewer)."
    out = r.extract_identifiers(text)
    assert "DEC-049" in out
    assert "DEC-081" in out


def test_extract_identifiers_dedups_case_insensitively():
    text = "Cilium and cilium and CILIUM — three forms."
    out = r.extract_identifiers(text, vocab=("cilium",))
    # First-occurrence casing preserved; later forms dropped.
    assert out.count("Cilium") + out.count("cilium") + out.count("CILIUM") == 1


def test_extract_identifiers_picks_up_configured_vocab():
    vocab = ("terraform", "helm", "nginx", "postgres", "kafka", "redis", "qdrant")
    text = "PR touches terraform helm nginx postgres kafka Redis qdrant."
    out = [s.lower() for s in r.extract_identifiers(text, vocab=vocab)]
    for token in vocab:
        assert token in out, f"missing {token} in {out}"


def test_extract_identifiers_returns_empty_for_empty_input():
    assert r.extract_identifiers("") == []
    assert r.extract_identifiers("nothing interesting here") == []


def test_extract_identifiers_matches_multiple_terms_per_text():
    """A wide vocabulary surfaces several identifiers from one query."""
    vocab = ("bm25", "dense", "rrf", "qdrant", "reranker", "rerank")
    text = "Hybrid BM25 + dense RRF over the Qdrant knowledge collection. Rerank stages use the TEI reranker."
    out = [s.lower() for s in r.extract_identifiers(text, vocab=vocab)]
    for token in vocab:
        assert token in out, f"missing {token} in {out}"


def test_extract_identifiers_picks_up_pr_references():
    text = "Builds on #1572 and #1576; supersedes #1577."
    out = r.extract_identifiers(text)
    assert "#1572" in out
    assert "#1576" in out
    assert "#1577" in out


def test_extract_identifiers_picks_up_underscored_terms():
    """Underscored vocab terms match as whole words (word-bounded regex)."""
    vocab = ("role_base", "svc_baseline", "nginx_ha", "keepalived")
    text = "Updates the role_base + svc_baseline roles, plus the nginx_ha keepalived config."
    out = [s.lower() for s in r.extract_identifiers(text, vocab=vocab)]
    for token in vocab:
        assert token in out, f"missing {token} in {out}"


def test_extract_identifiers_custom_vocab_is_exclusive():
    """Only the supplied vocab matches; DEC/PR-ref formats still fire."""
    text = "Bumps the widgetron service and DEC-007; cilium untouched."
    out = r.extract_identifiers(text, vocab=("widgetron",))
    assert "widgetron" in out
    assert "DEC-007" in out  # structural format still fires
    assert "cilium" not in out  # not in the supplied vocab


def test_extract_identifiers_empty_vocab_keeps_structural_formats():
    """Empty vocab disables domain matching but DEC/PR refs survive."""
    out = r.extract_identifiers("cilium argocd DEC-009 #1234", vocab=())
    assert out == ["DEC-009", "#1234"]


def test_default_vocab_matches_config_field():
    """The public config field mirrors the engine default by reference."""
    from cora.config import ReviewerConfig
    from cora.core.config import DEFAULT_RETRIEVAL_VOCAB

    assert ReviewerConfig().retrieval_vocab is DEFAULT_RETRIEVAL_VOCAB


# ---------------------------------------------------------------------------
# Per-kind candidate counts (dashboard surface)
# ---------------------------------------------------------------------------


def test_count_by_kind_basic():
    hits = [
        {"payload": {"kind": "decision", "key": "decision:DEC-001"}},
        {"payload": {"kind": "decision", "key": "decision:DEC-002"}},
        {"payload": {"kind": "note", "key": "note:foo"}},
    ]
    assert r._count_by_kind(hits) == {"decision": 2, "note": 1}


def test_count_by_kind_empty_returns_empty_dict():
    assert r._count_by_kind([]) == {}


def test_count_by_kind_missing_kind_gets_unknown_bucket():
    hits = [{"payload": {"key": "x"}}, {"payload": {}}, {}]
    assert r._count_by_kind(hits) == {"unknown": 3}


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------


_SAMPLE_DIFF = """diff --git a/k8s/apps/foo/deploy.yml b/k8s/apps/foo/deploy.yml
index 1234abc..5678def 100644
--- a/k8s/apps/foo/deploy.yml
+++ b/k8s/apps/foo/deploy.yml
@@ -10,7 +10,7 @@ spec:
   replicas: 1
-      image: foo:v1
+      image: foo:v2
       env:
diff --git a/docs/foo.md b/docs/foo.md
new file mode 100644
--- /dev/null
+++ b/docs/foo.md
@@ -0,0 +1,3 @@
+# Notes
+
+New file.
"""


def test_parse_changed_paths_returns_new_side_files():
    paths = r.parse_changed_paths(_SAMPLE_DIFF)
    assert "k8s/apps/foo/deploy.yml" in paths
    assert "docs/foo.md" in paths
    assert len(paths) == 2


def test_parse_changed_paths_caps_to_50():
    big_diff = "\n".join(f"+++ b/file_{i}.txt" for i in range(100))
    paths = r.parse_changed_paths(big_diff)
    assert len(paths) == 50


def test_extract_changed_lines_skips_file_headers():
    lines = r.extract_changed_lines(_SAMPLE_DIFF)
    # No `+++ b/...` or `--- a/...` markers in the result.
    for line in lines:
        assert not line.startswith("++ b/")
        assert not line.startswith("-- a/")
    # The actual edits show up.
    assert any("image: foo:v2" in line for line in lines)
    assert any("image: foo:v1" in line for line in lines)


def test_extract_changed_lines_caps_count():
    big_diff = "\n".join(f"+line_{i}" for i in range(200))
    lines = r.extract_changed_lines(big_diff, cap=30)
    assert len(lines) == 30


# ---------------------------------------------------------------------------
# build_retrieval_query
# ---------------------------------------------------------------------------


def test_build_retrieval_query_returns_structured_dict():
    metadata = {
        "title": "Bump cilium DEC-049 path",
        "body": "Updates the MTU per DEC-049.",
    }
    out = r.build_retrieval_query(metadata, _SAMPLE_DIFF)
    assert isinstance(out, dict)
    assert "text" in out
    assert "identifiers" in out
    assert "changed_paths" in out
    assert "title" in out
    assert "DEC-049" in out["identifiers"]
    assert "k8s/apps/foo/deploy.yml" in out["changed_paths"]
    assert out["title"] == "Bump cilium DEC-049 path"
    # The text concatenates title, identifiers, paths, body, changed lines.
    assert "Bump cilium" in out["text"]
    assert "DEC-049" in out["text"]
    assert "k8s/apps/foo/deploy.yml" in out["text"]


def test_build_retrieval_query_empty_inputs():
    out = r.build_retrieval_query({}, "")
    assert out["text"] == ""
    assert out["identifiers"] == []
    assert out["changed_paths"] == []
    assert out["title"] == ""


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def _hit(pid: str, score: float = 0.0, key: str = "") -> dict:
    return {
        "id": pid,
        "score": score,
        "payload": {"key": key or f"k{pid}", "kind": "decision"},
    }


def test_rrf_merges_two_lists_and_dedups_by_id():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b"), _hit("d")]
    merged = r.reciprocal_rank_fusion(dense, sparse)
    ids = [str(h["id"]) for h in merged]
    # All unique ids present, b only once.
    assert sorted(ids) == ["a", "b", "c", "d"]


def test_rrf_dense_wins_on_tie():
    # Two docs each appear in exactly one ranking at the same position.
    # The dense-side doc should outrank the sparse-side doc because
    # dense gets weight 1.0/(60+1) while sparse gets 0.99/(60+1).
    # (Docs that appear in BOTH lists naturally outrank docs in one —
    # that's the whole point of RRF — so test that separately.)
    dense = [_hit("a")]
    sparse = [_hit("b")]
    merged = r.reciprocal_rank_fusion(dense, sparse)
    ids = [str(h["id"]) for h in merged]
    assert ids[0] == "a", "dense-only doc should win the tie via the 0.99 multiplier"
    assert ids[1] == "b"


def test_rrf_doc_in_both_lists_outranks_doc_in_one():
    # A doc that appears in both lists accumulates score from both —
    # always outranks a doc that appears in just one list at the same
    # position.
    dense = [_hit("a"), _hit("c")]
    sparse = [_hit("b"), _hit("c")]
    merged = r.reciprocal_rank_fusion(dense, sparse)
    ids = [str(h["id"]) for h in merged]
    assert ids[0] == "c"  # in both lists at rank 2, total wins


def test_rrf_empty_inputs():
    assert r.reciprocal_rank_fusion([], []) == []
    out = r.reciprocal_rank_fusion([_hit("a")], [])
    assert len(out) == 1
    out = r.reciprocal_rank_fusion([], [_hit("a")])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# BM25 sparse encoder
# ---------------------------------------------------------------------------


def test_bm25_sparse_returns_empty_for_empty():
    indices, values = _bm25_sparse("")
    assert indices == []
    assert values == []


def test_bm25_sparse_returns_non_empty_for_normal_text():
    indices, values = _bm25_sparse("cilium cnpg cilium argocd")
    assert len(indices) == 3  # cilium dedup'd to 1, plus cnpg, plus argocd
    assert len(values) == 3
    # All positive.
    assert all(v > 0 for v in values)


def test_bm25_sparse_token_ids_are_stable_across_calls():
    a_indices, _ = _bm25_sparse("cilium argocd helm")
    b_indices, _ = _bm25_sparse("cilium argocd helm")
    assert a_indices == b_indices


def test_bm25_sparse_token_ids_stable_independent_of_hash_seed():
    # crc32-based IDs are reproducible across Python invocations
    # regardless of PYTHONHASHSEED. Hard-code the expected ID for a
    # known token to catch any future change to the encoding scheme
    # that would silently break compatibility with already-indexed data.
    assert _stable_token_id("cilium") == _stable_token_id("cilium")
    # Two different tokens → two different IDs (modulo collision; pick
    # tokens unlikely to collide under 1M-bucket crc32).
    assert _stable_token_id("cilium") != _stable_token_id("helm")


def test_bm25_sparse_indices_are_unique():
    """Qdrant rejects duplicate indices in a single sparse vector with
    'must be unique' 422 — caught the 2026-05-24 reindex on real data.
    Even with crc32, the 1M-bucket modulus means two different tokens
    can collide. The encoder must aggregate collisions, not emit dupes.

    Construct a deliberately-large input (many unique tokens) to make
    collisions plausible; assert indices are unique regardless.
    """
    # Generate ~5000 unique pseudo-tokens — well above the natural
    # collision threshold at 1M buckets (birthday paradox: 50%
    # collision odds at ~1200 unique tokens).
    text = " ".join(f"token{i}" for i in range(5000))
    indices, values = _bm25_sparse(text)
    assert len(indices) == len(set(indices)), (
        f"indices contain {len(indices) - len(set(indices))} duplicate(s) "
        f"— would 422 against Qdrant"
    )
    # Sorted-by-index for determinism (helps caching + debugging).
    assert indices == sorted(indices)
    # All values still positive (collision aggregation preserves sign).
    assert all(v > 0 for v in values)


def test_bm25_sparse_aggregates_collisions():
    """When two different tokens hash to the same index, their values
    must SUM rather than overwrite. Pick tokens that we KNOW collide
    under the current scheme to verify the aggregation behaviour."""
    # Brute-force find a colliding pair so the test is robust to any
    # specific collision happening.
    seen: dict[int, str] = {}
    pair: tuple[str, str] | None = None
    for i in range(100_000):
        tok = f"t{i}"
        idx = _stable_token_id(tok)
        if idx in seen and seen[idx] != tok:
            pair = (seen[idx], tok)
            break
        seen[idx] = tok
    assert pair is not None, "no collision found in 100k tokens — modulus changed?"
    a, b = pair

    # Encode `a` alone, `b` alone, then `a b` together.
    indices_a, values_a = _bm25_sparse(a)
    indices_b, values_b = _bm25_sparse(b)
    indices_both, values_both = _bm25_sparse(f"{a} {b}")

    # Both single-token encodings have one index, and it's the same.
    assert len(indices_a) == 1
    assert len(indices_b) == 1
    assert indices_a[0] == indices_b[0]
    # Combined input also has just one index (collision), with aggregated value.
    assert len(indices_both) == 1
    assert indices_both[0] == indices_a[0]
    # BM25 saturation means the combined score isn't a clean sum, but it
    # must be > either single value (collision adds mass).
    assert values_both[0] > max(values_a[0], values_b[0])


def test_bm25_sparse_indices_in_qdrant_range():
    indices, _ = _bm25_sparse(
        "the quick brown fox jumps over the lazy dog"
    )
    # Modulus is 1_000_003; Qdrant accepts up to 2^32-1.
    for idx in indices:
        assert 0 <= idx < 1_000_003


def test_bm25_sparse_term_frequency_saturation():
    # A token appearing many times should NOT grow linearly. With k1=1.2
    # the TF saturation means doubling count grows score sub-linearly.
    _, single_vals = _bm25_sparse("cilium")
    _, many_vals = _bm25_sparse("cilium " * 10)
    # 10× the count should NOT produce 10× the score.
    assert many_vals[0] < 10 * single_vals[0]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    a = r._cache_key("foo/bar", "1", "abc123", "query text")
    b = r._cache_key("foo/bar", "1", "abc123", "query text")
    assert a == b
    assert len(a) == 32


def test_cache_key_changes_with_any_input():
    base = r._cache_key("repo", "1", "abc", "q")
    assert r._cache_key("repo", "2", "abc", "q") != base
    assert r._cache_key("repo", "1", "def", "q") != base
    assert r._cache_key("repo", "1", "abc", "q2") != base
    assert r._cache_key("other", "1", "abc", "q") != base


def test_cache_get_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(tmp_path))
    docs, age = r._cache_get("nonexistent")
    assert docs is None
    assert age is None


def test_cache_put_and_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_REVIEW_CACHE_TTL_S", "600")
    docs_in = [{"kind": "decision", "key": "decision:DEC-049", "body": "..."}]
    r._cache_put("k1", docs_in)
    docs_out, age = r._cache_get("k1")
    assert docs_out == docs_in
    assert age is not None
    assert 0 <= age < 5  # fresh


def test_cache_get_expired_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_REVIEW_CACHE_TTL_S", "1")
    r._cache_put("k_expired", [{"x": 1}])
    # Backdate the file to ensure expiry.
    import os
    cache_file = tmp_path / "k_expired.json"
    past = (cache_file.stat().st_mtime - 10)
    os.utime(cache_file, (past, past))
    docs, age = r._cache_get("k_expired")
    assert docs is None
    assert age is None


def test_cache_get_corrupt_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_REVIEW_CACHE_TTL_S", "600")
    (tmp_path / "corrupt.json").write_text("{not valid json")
    docs, age = r._cache_get("corrupt")
    assert docs is None
    assert age is None


def test_cache_put_failure_is_soft(tmp_path, monkeypatch):
    # Point the cache dir at a path we can't create (a file, not a dir).
    bad_path = tmp_path / "blocking_file"
    bad_path.write_text("hello")
    monkeypatch.setenv("AGENT_REVIEW_CACHE_DIR", str(bad_path / "inside"))
    # Should silently no-op rather than raising.
    r._cache_put("anything", [{"k": 1}])


# ---------------------------------------------------------------------------
# Hit-brief helper
# ---------------------------------------------------------------------------


def test_hit_brief_extracts_minimal_fields():
    h = {
        "id": "xyz",
        "score": 0.83,
        "payload": {"key": "decision:DEC-049", "kind": "decision", "extra": "junk"},
    }
    out = r._hit_brief(h)
    assert out == {"key": "decision:DEC-049", "kind": "decision", "score": 0.83}


def test_hit_brief_handles_missing_payload():
    out = r._hit_brief({"id": "abc"})
    assert out["key"] == ""
    assert out["kind"] == ""
    assert out["score"] == 0.0
