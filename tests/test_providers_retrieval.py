"""RetrievalProvider seam: the Null zero-infra default, config-driven
selection, and TeiQdrant delegation to the engine pipeline."""

from __future__ import annotations

from pathlib import Path

from cora.config import ReviewerConfig
from cora.providers import (
    GlobRetrievalProvider,
    NullRetrievalProvider,
    RetrievalProvider,
    TeiQdrantRetrievalProvider,
)


def test_null_provider_returns_empty_and_disabled_trace():
    docs, trace = NullRetrievalProvider().retrieve(
        repo="o/r", pr_number="1", head_sha="abc", metadata={}, diff_text="x"
    )
    assert docs == []
    assert trace == {"provider": "null", "disabled": True}


def test_from_config_defaults_to_null_without_api_key():
    # Zero-infra adopter: no QDRANT_API_KEY → Null, no vector store needed.
    provider = RetrievalProvider.from_config(ReviewerConfig())
    assert isinstance(provider, NullRetrievalProvider)


def test_from_config_selects_tei_qdrant_when_fully_configured():
    cfg = ReviewerConfig(qdrant_api_key="secret")  # url/tei/reranker have defaults
    provider = RetrievalProvider.from_config(cfg)
    assert isinstance(provider, TeiQdrantRetrievalProvider)
    assert provider.qdrant_api_key == "secret"
    assert provider.top_k == cfg.retrieval_top_k
    # The full config rides along so the engine pipeline reads every
    # retrieval tunable from it.
    assert provider.cfg is cfg


def test_from_config_stays_null_when_endpoints_blank_even_with_key():
    cfg = ReviewerConfig(qdrant_api_key="secret", tei_url="")
    assert isinstance(RetrievalProvider.from_config(cfg), NullRetrievalProvider)


def test_tei_qdrant_delegates_to_engine_with_provider_wiring(monkeypatch):
    captured: dict = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [{"title": "t", "path": "p", "body": "b"}], {"ok": True}

    monkeypatch.setattr(
        "cora.providers.retrieval._r.retrieve_relevant_docs", fake_retrieve
    )
    provider = TeiQdrantRetrievalProvider(
        qdrant_url="http://q",
        qdrant_api_key="k",
        tei_url="http://t",
        reranker_url="http://rr",
        repo_root=Path("/tmp"),
        top_k=7,
    )
    docs, trace = provider.retrieve(
        repo="o/r", pr_number="2", head_sha="def", metadata={"a": 1}, diff_text="diff"
    )
    assert docs[0]["title"] == "t"
    assert trace == {"ok": True}
    # Provider's static wiring + the per-call PR context both reach the engine.
    assert captured["qdrant_url"] == "http://q"
    assert captured["qdrant_api_key"] == "k"
    assert captured["reranker_url"] == "http://rr"
    assert captured["repo_root"] == Path("/tmp")
    assert captured["top_k"] == 7
    assert captured["repo"] == "o/r"
    assert captured["pr_number"] == "2"
    assert captured["diff_text"] == "diff"
    # No config supplied → engine falls back to its constants.
    assert captured["cfg"] is None


def test_format_docs_shared_default_delegates_to_core():
    assert "no relevant context" in NullRetrievalProvider().format_docs([]).lower()


# ── GlobRetrievalProvider (zero-infra local BM25) ────────────────────


def _glob_tree(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "deploy.md").write_text(
        "# Deploying the gateway\n\n"
        "The litellm gateway deployment rolls via argocd. Rollback by "
        "reverting the configmap.\n"
    )
    (tmp_path / "docs" / "cooking.md").write_text(
        "# Pancakes\n\nFlour, eggs, milk. Whisk and fry.\n"
    )
    (tmp_path / "README.md").write_text(
        "# Project\n\nGeneral overview, nothing specific.\n"
    )
    return tmp_path


def _pr_query_kwargs() -> dict:
    return dict(
        repo="o/r",
        pr_number="7",
        head_sha="abc",
        metadata={"title": "fix litellm gateway rollback", "body": ""},
        diff_text="diff --git a/k8s/litellm/configmap.yml b/...\n+gateway: rollback\n",
    )


def test_glob_provider_ranks_relevant_doc_first(tmp_path):
    p = GlobRetrievalProvider(include=("**/*.md",), root=_glob_tree(tmp_path), top_k=2)
    docs, trace = p.retrieve(**_pr_query_kwargs())
    assert docs and docs[0]["path"] == "docs/deploy.md"
    assert docs[0]["title"] == "Deploying the gateway"
    assert "litellm" in docs[0]["body"]
    assert trace["provider"] == "glob"
    assert trace["returned"] == len(docs) <= 2
    # The irrelevant recipe must not outrank the deploy doc.
    assert all(d["path"] != "docs/cooking.md" or d is not docs[0] for d in docs)


def test_glob_provider_empty_query_short_circuits(tmp_path):
    p = GlobRetrievalProvider(include=("**/*.md",), root=_glob_tree(tmp_path))
    docs, trace = p.retrieve(
        repo="o/r", pr_number="7", head_sha="abc", metadata={}, diff_text=""
    )
    assert docs == []
    assert trace["skip_reason"] == "empty_query"


def test_glob_provider_skips_oversized_and_caps_scan(tmp_path):
    root = _glob_tree(tmp_path)
    (root / "docs" / "huge.md").write_text("litellm gateway " * 50_000)
    p = GlobRetrievalProvider(
        include=("**/*.md",), root=root, max_files=2, max_file_bytes=10_000
    )
    docs, trace = p.retrieve(**_pr_query_kwargs())
    assert trace["files_scanned"] == 2
    assert trace["scan_truncated"] is True
    assert all(d["path"] != "docs/huge.md" for d in docs)


def test_glob_provider_caps_doc_body(tmp_path):
    root = _glob_tree(tmp_path)
    p = GlobRetrievalProvider(include=("**/*.md",), root=root, doc_char_cap=20)
    docs, _ = p.retrieve(**_pr_query_kwargs())
    assert docs and all(len(d["body"]) <= 20 for d in docs)


def test_from_config_selects_glob_when_opted_in(tmp_path):
    cfg = ReviewerConfig(retrieval_glob_include=("docs/**/*.md",))
    p = RetrievalProvider.from_config(cfg)
    assert isinstance(p, GlobRetrievalProvider)
    assert p.include == ("docs/**/*.md",)


def test_from_config_tei_qdrant_outranks_glob():
    cfg = ReviewerConfig(
        qdrant_url="http://q", qdrant_api_key="k", tei_url="http://t",
        reranker_url="http://rr", retrieval_glob_include=("*.md",),
    )
    assert isinstance(
        RetrievalProvider.from_config(cfg), TeiQdrantRetrievalProvider
    )


def test_from_env_parses_glob_csv():
    cfg = ReviewerConfig.from_env({"AGENT_REVIEW_RETRIEVAL_GLOB": "docs/**/*.md, *.md"})
    assert cfg.retrieval_glob_include == ("docs/**/*.md", "*.md")
    assert ReviewerConfig.from_env({}).retrieval_glob_include == ()
