# RAG Skills Layout

This folder contains retrieval and indexing tools for long-context/codebase memory.

## Current modules

- `10_ingest.py`: ingestion and indexing commands (`rag_add_file`, `rag_add_text`, `rag_add_dir`, `rag_promote_paths`).
- `20_retrieve.py`: retrieval and corpus inspection (`rag_search`, `rag_list`).
- `30_tuning.py`: runtime tuning/benchmarking (`rag_tuning_status`, `rag_apply_profile`, `rag_benchmark`).

## Scope

- Keep this folder focused on RAG quality and speed.
- Prefer retrieval/indexing improvements here (chunking, scoring, reranking, ingestion pipelines).
- Keep coding-editing execution tools in `skills/01_coding`.
