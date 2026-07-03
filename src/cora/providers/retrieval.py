"""Retrieval providers — the RAG seam behind cora's review context.

The engine asks a `RetrievalProvider` for the docs relevant to a diff;
the provider decides where they come from. This is what lets cora run
with **zero** retrieval infrastructure (`NullRetrievalProvider`, the
default) or against a hybrid dense+sparse vector store
(`TeiQdrantRetrievalProvider`).

`retrieve()` returns `(docs, trace)`: `docs` is the ranked top-K with
bodies (the shape `assemble_initial_user_prompt` consumes), `trace` is a
JSON-serialisable dict the caller can log. Implementations must not raise
for an empty result; a hard backend failure (vector store unreachable)
may raise so the caller's soft-fallback to no-retrieval still fires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from cora.core import config as _c
from cora.core import retrieval as _r
from cora.core._bm25 import _bm25_sparse

if TYPE_CHECKING:
    from cora.config import ReviewerConfig


class RetrievalProvider(ABC):
    """Supplies the diff-relevant context bundle for a review."""

    @abstractmethod
    def retrieve(
        self,
        *,
        repo: str,
        pr_number: str,
        head_sha: str,
        metadata: dict,
        diff_text: str,
    ) -> tuple[list[dict], dict]:
        """Return `(docs, trace)` for this PR. Never raises for an empty
        result; may raise on a hard backend failure so the caller can
        soft-fall-back to a retrieval-free review."""

    def format_docs(self, docs: list[dict]) -> str:
        """Render retrieved docs into the prompt block. Shared default;
        override only if a provider returns a different doc shape."""
        return _r.format_retrieved_docs(docs)

    @classmethod
    def from_config(cls, cfg: "ReviewerConfig") -> "RetrievalProvider":
        """Pick a provider from config. Defaults to `NullRetrievalProvider`
        unless a full Qdrant+TEI endpoint set AND an API key are configured
        (the hybrid pipeline) or local glob retrieval is opted
        into via `retrieval_glob_include` — so an adopter with no vector
        store gets working, retrieval-free reviews out of the box and can
        upgrade to zero-infra BM25 over their own docs with one knob."""
        if cfg.qdrant_url and cfg.qdrant_api_key and cfg.tei_url and cfg.reranker_url:
            return TeiQdrantRetrievalProvider(
                qdrant_url=cfg.qdrant_url,
                qdrant_api_key=cfg.qdrant_api_key,
                tei_url=cfg.tei_url,
                reranker_url=cfg.reranker_url,
                top_k=cfg.retrieval_top_k,
                # Full config rides along so the engine pipeline reads
                # every retrieval tunable (overfetch, rerank survivors,
                # doc cap, collection, timeout, cache) from it.
                cfg=cfg,
            )
        if cfg.retrieval_glob_include:
            return GlobRetrievalProvider(
                include=cfg.retrieval_glob_include,
                top_k=cfg.retrieval_top_k,
                doc_char_cap=cfg.retrieval_doc_char_cap,
                max_files=cfg.retrieval_glob_max_files,
                max_file_bytes=cfg.retrieval_glob_max_file_bytes,
                cfg=cfg,
            )
        return NullRetrievalProvider()


class NullRetrievalProvider(RetrievalProvider):
    """No RAG — the review runs on the diff + CLAUDE.md alone. The
    zero-infra default: an adopter needs no vector store to run cora."""

    def retrieve(self, **_kwargs: object) -> tuple[list[dict], dict]:
        return [], {"provider": "null", "disabled": True}


class GlobRetrievalProvider(RetrievalProvider):
    """Zero-infra local retrieval: BM25-rank files
    from the checkout itself against the PR query — no vector store, no
    embedder, no network. The hybrid pipeline's lexical channel
    (`cora.core._bm25`, uniform IDF) applied directly to the doc files
    an adopter already has (`docs/**/*.md`, DECISIONS, runbooks).

    Ranking is dot-product over the shared sparse encoding: the query
    text comes from `build_retrieval_query` (title + identifiers +
    changed paths + diff lines), each candidate file's body is encoded
    once per review. Bounded by `max_files` scanned and the per-file
    byte ceiling (oversized files are skipped, not truncated into
    noise)."""

    def __init__(
        self,
        *,
        include: tuple[str, ...],
        root: Path | None = None,
        top_k: int = _c.RETRIEVAL_TOP_K,
        doc_char_cap: int = _c.RETRIEVAL_DOC_CHAR_CAP,
        max_files: int = _c.RETRIEVAL_GLOB_MAX_FILES,
        max_file_bytes: int = _c.RETRIEVAL_GLOB_MAX_FILE_BYTES,
        cfg: "ReviewerConfig | None" = None,
    ) -> None:
        self.include = include
        self.root = root if root is not None else _c.REPO_ROOT
        self.top_k = top_k
        self.doc_char_cap = doc_char_cap
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.cfg = cfg

    def _candidate_files(self) -> tuple[list[Path], bool]:
        """Matched files in stable order, deduped across globs, and
        whether the `max_files` scan ceiling truncated the walk."""
        seen: set[Path] = set()
        files: list[Path] = []
        truncated = False
        for pattern in self.include:
            for path in sorted(self.root.glob(pattern)):
                if len(files) >= self.max_files:
                    truncated = True
                    return files, truncated
                if path in seen or not path.is_file():
                    continue
                if ".git" in path.relative_to(self.root).parts:
                    continue
                seen.add(path)
                files.append(path)
        return files, truncated

    def retrieve(
        self,
        *,
        repo: str,
        pr_number: str,
        head_sha: str,
        metadata: dict,
        diff_text: str,
    ) -> tuple[list[dict], dict]:
        query = _r.build_retrieval_query(metadata, diff_text, cfg=self.cfg)
        q_idx, q_val = _bm25_sparse(query["text"])
        q_vec = dict(zip(q_idx, q_val))
        trace: dict = {
            "provider": "glob",
            "root": str(self.root),
            "include": list(self.include),
            "query_chars": len(query["text"]),
        }
        if not q_vec:
            trace["skip_reason"] = "empty_query"
            return [], trace

        files, truncated = self._candidate_files()
        scored: list[tuple[float, Path, str]] = []
        skipped_large = 0
        for path in files:
            try:
                if path.stat().st_size > self.max_file_bytes:
                    skipped_large += 1
                    continue
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            d_idx, d_val = _bm25_sparse(body)
            score = sum(q_vec.get(i, 0.0) * v for i, v in zip(d_idx, d_val))
            if score > 0.0:
                scored.append((score, path, body))

        scored.sort(key=lambda t: (-t[0], str(t[1])))
        docs: list[dict] = []
        for score, path, body in scored[: self.top_k]:
            rel = str(path.relative_to(self.root))
            # First markdown heading reads better than a bare path in
            # the prompt block; fall back to the path itself.
            title = next(
                (
                    ln.lstrip("#").strip()
                    for ln in body.splitlines()
                    if ln.startswith("#")
                ),
                rel,
            )
            docs.append({
                "title": title,
                "path": rel,
                "body": body[: self.doc_char_cap],
                "score": round(score, 4),
            })
        trace.update(
            files_scanned=len(files),
            scan_truncated=truncated,
            skipped_large=skipped_large,
            matched=len(scored),
            returned=len(docs),
            top=[{"path": d["path"], "score": d["score"]} for d in docs],
        )
        return docs, trace


class TeiQdrantRetrievalProvider(RetrievalProvider):
    """Hybrid dense+sparse Qdrant retrieval with two-stage TEI reranking.
    Doc bodies are resolved from a local doc
    tree rooted at `repo_root` (defaults to the engine's resolved
    `CORA_REPO_ROOT`)."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str,
        tei_url: str,
        reranker_url: str,
        repo_root: Path | None = None,
        top_k: int = _c.RETRIEVAL_TOP_K,
        cfg: "ReviewerConfig | None" = None,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.tei_url = tei_url
        self.reranker_url = reranker_url
        self.repo_root = repo_root if repo_root is not None else _c.REPO_ROOT
        self.top_k = top_k
        # Optional full ReviewerConfig — threaded into the engine
        # pipeline so the remaining retrieval tunables come from the
        # config object. None keeps the engine-constant defaults.
        self.cfg = cfg

    def retrieve(
        self,
        *,
        repo: str,
        pr_number: str,
        head_sha: str,
        metadata: dict,
        diff_text: str,
    ) -> tuple[list[dict], dict]:
        return _r.retrieve_relevant_docs(
            repo=repo,
            pr_number=pr_number,
            head_sha=head_sha,
            repo_root=self.repo_root,
            metadata=metadata,
            diff_text=diff_text,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            tei_url=self.tei_url,
            reranker_url=self.reranker_url,
            top_k=self.top_k,
            cfg=self.cfg,
        )
