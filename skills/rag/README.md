# RAG Skills Layout

This folder contains retrieval and indexing tools for long-context/codebase memory.

## Current modules

- `scripts/ingest.py`: ingestion and indexing commands (`rag_add_file`, `rag_add_text`, `rag_add_dir`, `rag_promote_paths`).
- `scripts/retrieve.py`: retrieval and corpus inspection (`rag_search`, `rag_list`).
- `scripts/tuning.py`: runtime tuning/benchmarking (`rag_tuning_status`, `rag_apply_profile`, `rag_benchmark`).

## Scope

- Keep this folder focused on RAG quality and speed.
- Prefer retrieval/indexing improvements here (chunking, scoring, reranking, ingestion pipelines).
- Keep coding-editing execution tools in `skills/coding`.
