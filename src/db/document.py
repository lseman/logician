# agent_core/db_doc.py
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import _HNSWCollection, create_vector_collection
from ..logging_utils import get_logger


@dataclass
class DocumentDB:
    vector_path: str = "rag_docs.vector"
    embedding_model_name: str = (
        "BAAI/bge-m3|Snowflake/snowflake-arctic-embed-l-v2.0|"
        "Qwen/Qwen3-Embedding-0.6B|nomic-ai/nomic-embed-text-v1.5|"
        "intfloat/e5-mistral-7b-instruct|BAAI/bge-small-en-v1.5"
    )
    collection_name: str = "default"
    vector_backend: str = "usearch"

    vector_enabled: bool = True
    lazy_vector: bool = True

    rerank_enabled: bool = True
    reranker_model_name: str = (
        "BAAI/bge-reranker-v2.5-gemma2-lightweight|"
        "BAAI/bge-reranker-v2-m3|mixedbread-ai/mxbai-rerank-xsmall-v1"
    )
    rerank_fetch_k: int = 30
    min_similarity: float = 0.20
    hnsw_ef_search: int = 128
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    per_source_max_chunks: int = 4
    query_cache_enabled: bool = True
    query_cache_ttl_sec: int = 90
    query_cache_max_entries: int = 256

    _log: Any = field(init=False, repr=False)
    _collection: _HNSWCollection | None = field(default=None, init=False, repr=False)
    _query_cache: dict[str, tuple[float, list[dict[str, Any]]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._log = get_logger("agent.rag")
        if self.vector_enabled and (not self.lazy_vector):
            self._ensure()

    def _ensure(self) -> None:
        if not self.vector_enabled:
            return
        if self._collection is not None:
            return

        self._collection = create_vector_collection(
            backend=self.vector_backend,
            root_path=self.vector_path,
            collection_name=self.collection_name,
            embedding_model_name=self.embedding_model_name,
            rerank_enabled=self.rerank_enabled,
            reranker_model_name=self.reranker_model_name,
            ef_construction=max(16, int(self.hnsw_ef_construction)),
            m=max(8, int(self.hnsw_m)),
            ef_search=max(8, int(self.hnsw_ef_search)),
            min_similarity=self.min_similarity,
        )

    @property
    def collection(self) -> _HNSWCollection:
        self._ensure()
        if self._collection is None:
            raise RuntimeError(
                "Vector store is disabled but a semantic operation was requested."
            )
        return self._collection

    def count(
        self,
        where: dict[str, Any] | None = None,
        *,
        include_deleted: bool = False,
    ) -> int:
        """Count indexed chunks matching optional metadata filter."""
        if not self.vector_enabled:
            return 0
        return int(self.collection.count(where=where, include_deleted=include_deleted))

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: list[str] | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Get indexed chunks by ids/metadata with optional pagination."""
        if not self.vector_enabled:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=include,
            include_deleted=include_deleted,
        )

    def peek(self, limit: int = 10, include: list[str] | None = None) -> dict[str, Any]:
        """Peek first indexed chunks for quick inspection."""
        if not self.vector_enabled:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.peek(limit=limit, include=include)

    @staticmethod
    def _tokenize_for_chunking(text: str) -> list[str]:
        return re.findall(r"\S+", text)

    @staticmethod
    def _chunk_by_token_windows(
        text: str,
        *,
        chunk_size_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        toks = DocumentDB._tokenize_for_chunking(text)
        if not toks:
            return []
        if len(toks) <= chunk_size_tokens:
            return [" ".join(toks)]
        step = max(1, chunk_size_tokens - overlap_tokens)
        chunks: list[str] = []
        for start in range(0, len(toks), step):
            part = toks[start : start + chunk_size_tokens]
            if not part:
                break
            chunks.append(" ".join(part))
            if start + chunk_size_tokens >= len(toks):
                break
        return chunks

    @staticmethod
    def _split_into_semantic_units(text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paras:
            return [text]
        units: list[str] = []
        for p in paras:
            sents = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s and s.strip()
            ]
            if len(sents) > 1:
                units.extend(sents)
            else:
                units.append(p)
        return units

    def _recursive_chunk_text(
        self,
        text: str,
        *,
        chunk_size_tokens: int,
        overlap_ratio: float,
    ) -> list[str]:
        units = self._split_into_semantic_units(text)
        chunks: list[str] = []
        cur: list[str] = []
        cur_tokens = 0

        def _flush_current() -> None:
            nonlocal cur, cur_tokens
            if cur:
                chunks.append(" ".join(cur).strip())
                cur = []
                cur_tokens = 0

        for unit in units:
            t = self._tokenize_for_chunking(unit)
            if not t:
                continue
            tlen = len(t)

            if tlen > chunk_size_tokens:
                _flush_current()
                overlap_tokens = int(round(chunk_size_tokens * overlap_ratio))
                chunks.extend(
                    self._chunk_by_token_windows(
                        unit,
                        chunk_size_tokens=chunk_size_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
                continue

            if cur_tokens + tlen > chunk_size_tokens and cur:
                _flush_current()

            cur.append(unit)
            cur_tokens += tlen

        _flush_current()

        if overlap_ratio <= 0 or len(chunks) <= 1:
            return chunks

        overlap_tokens = int(round(chunk_size_tokens * overlap_ratio))
        if overlap_tokens <= 0:
            return chunks

        overlapped: list[str] = []
        prev_tail: list[str] = []
        for c in chunks:
            toks = self._tokenize_for_chunking(c)
            if prev_tail:
                toks = prev_tail + toks
            if len(toks) > chunk_size_tokens:
                toks = toks[:chunk_size_tokens]
            overlapped.append(" ".join(toks).strip())
            prev_tail = toks[-overlap_tokens:] if overlap_tokens < len(toks) else toks
        return overlapped

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunk_size_tokens: int = 400,
        chunk_overlap_ratio: float = 0.2,
        chunking: str = "recursive",
        chunk_size: int | None = None,  # backward-compatible alias
        add_batch_size: int = 512,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[str]:
        if not self.vector_enabled:
            return []

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if chunk_size is not None:
            chunk_size_tokens = int(chunk_size)
        chunk_size_tokens = max(200, min(600, int(chunk_size_tokens)))
        chunk_overlap_ratio = float(max(0.0, min(0.4, chunk_overlap_ratio)))
        chunking = str(chunking or "recursive").strip().lower()

        chunks: list[str] = []
        chunk_metas: list[dict[str, Any]] = []
        chunk_ids: list[str] = []

        for i, text in enumerate(texts):
            if chunking in ("recursive", "semantic"):
                sub = self._recursive_chunk_text(
                    text,
                    chunk_size_tokens=chunk_size_tokens,
                    overlap_ratio=chunk_overlap_ratio,
                )
            else:
                overlap_tokens = int(round(chunk_size_tokens * chunk_overlap_ratio))
                sub = self._chunk_by_token_windows(
                    text,
                    chunk_size_tokens=chunk_size_tokens,
                    overlap_tokens=overlap_tokens,
                )

            if not sub:
                continue
            for j, chunk in enumerate(sub):
                chunks.append(chunk)
                chunk_metas.append(
                    {
                        **metadatas[i],
                        "chunk": j,
                        "chunking": chunking,
                        "chunk_size_tokens": chunk_size_tokens,
                        "chunk_overlap_ratio": chunk_overlap_ratio,
                    }
                )
                chunk_ids.append(f"{ids[i]}_{j}" if len(sub) > 1 else ids[i])

        self._log.info(
            "Adding %d chunks (strategy=%s size=%d overlap=%.2f)",
            len(chunks),
            chunking,
            chunk_size_tokens,
            chunk_overlap_ratio,
        )

        self.collection.add(
            documents=chunks,
            metadatas=chunk_metas,
            ids=chunk_ids,
            batch_size=max(1, int(add_batch_size)),
            progress_callback=progress_callback,
        )
        self._query_cache.clear()
        return chunk_ids

    def _cache_key(
        self,
        *,
        query: str,
        n_results: int,
        where: dict[str, Any] | None,
        ef_search: int | None,
    ) -> str:
        payload = {
            "q": str(query or "").strip(),
            "k": int(n_results),
            "where": where or {},
            "ef_search": int(ef_search) if ef_search is not None else 0,
        }
        raw = str(payload)
        return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()

    def _cache_get(self, key: str) -> list[dict[str, Any]] | None:
        if not self.query_cache_enabled:
            return None
        ttl = max(1, int(self.query_cache_ttl_sec))
        rec = self._query_cache.get(key)
        if rec is None:
            return None
        ts, rows = rec
        if (time.time() - ts) > ttl:
            self._query_cache.pop(key, None)
            return None
        return [dict(row) for row in rows]

    def _cache_put(self, key: str, rows: list[dict[str, Any]]) -> None:
        if not self.query_cache_enabled:
            return
        self._query_cache[key] = (time.time(), [dict(row) for row in rows])
        cap = max(8, int(self.query_cache_max_entries))
        if len(self._query_cache) <= cap:
            return
        overflow = len(self._query_cache) - cap
        stale_keys = sorted(self._query_cache.items(), key=lambda item: item[1][0])[
            :overflow
        ]
        for stale_key, _ in stale_keys:
            self._query_cache.pop(stale_key, None)

    @staticmethod
    def _source_key(meta: dict[str, Any]) -> str:
        path = str(meta.get("path", "") or "").strip()
        if path:
            return f"path:{path}"
        source = str(meta.get("source", "") or "").strip()
        if source:
            return f"source:{source}"
        return "source:unknown"

    def _diversify_rows(
        self, rows: list[dict[str, Any]], *, n_results: int
    ) -> list[dict[str, Any]]:
        per_source_limit = max(1, int(self.per_source_max_chunks))
        if per_source_limit <= 0 or len(rows) <= n_results:
            return rows[:n_results]

        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        per_source_counts: dict[str, int] = {}

        for idx, row in enumerate(rows):
            meta = row.get("metadata", {}) or {}
            source_key = self._source_key(meta)
            count = per_source_counts.get(source_key, 0)
            if count >= per_source_limit:
                continue
            selected.append(row)
            selected_ids.add(idx)
            per_source_counts[source_key] = count + 1
            if len(selected) >= n_results:
                return selected

        if len(selected) >= n_results:
            return selected[:n_results]

        for idx, row in enumerate(rows):
            if idx in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= n_results:
                break
        return selected[:n_results]

    def query(
        self,
        query: str,
        n_results: int = 20,
        where: dict[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.vector_enabled:
            return []

        n_results = max(1, int(n_results))
        cache_key = self._cache_key(
            query=query,
            n_results=n_results,
            where=where,
            ef_search=ef_search,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached[:n_results]

        fetch_k = max(n_results, min(200, max(self.rerank_fetch_k, n_results * 4)))

        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
            ef_search=ef_search,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        rows = [
            {"content": doc, "metadata": meta, "distance": float(dist)}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        rows = self._diversify_rows(rows, n_results=n_results)
        self._cache_put(cache_key, rows)
        return rows[:n_results]
