# agent_core/db_doc.py
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .db_core import _HNSWCollection
from .logging_utils import get_logger


@dataclass
class DocumentDB:
    vector_path: str = "rag_docs.vector"
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5|BAAI/bge-base-en-v1.5"
    collection_name: str = "default"

    vector_enabled: bool = True
    lazy_vector: bool = True

    rerank_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    rerank_fetch_k: int = 30
    min_similarity: float = 0.20

    _log: Any = field(init=False, repr=False)
    _collection: _HNSWCollection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._log = get_logger("agent.rag")
        if self.vector_enabled and (not self.lazy_vector):
            self._ensure()

    def _ensure(self) -> None:
        if not self.vector_enabled:
            return
        if self._collection is not None:
            return

        self._collection = _HNSWCollection(
            root_path=self.vector_path,
            collection_name=self.collection_name,
            embedding_model_name=self.embedding_model_name,
            rerank_enabled=self.rerank_enabled,
            reranker_model_name=self.reranker_model_name,
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

        self.collection.add(documents=chunks, metadatas=chunk_metas, ids=chunk_ids)
        return chunk_ids

    def query(
        self,
        query: str,
        n_results: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.vector_enabled:
            return []

        n_results = max(1, int(n_results))
        fetch_k = max(n_results, min(60, max(self.rerank_fetch_k, n_results * 3)))

        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        rows = [
            {"content": doc, "metadata": meta, "distance": float(dist)}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        return rows[:n_results]
