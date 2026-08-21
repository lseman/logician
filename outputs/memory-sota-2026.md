# Logician memory modernization brief (2026-08-10)

## Executive finding

Logician already implements most of the practical AgentMemory baseline: grounded turn episodes, asynchronous extraction, SQLite/FTS5, optional dense embeddings, reciprocal-rank fusion, MMR context assembly, versioned memories, relations, provenance IDs, retention tiers, and workspace scoping. Replacing it with a graph/vector product would add operational weight without establishing better task performance.

The next useful step is to make memory *truth-aware*: preserve immutable evidence, represent derived claims with explicit validity and provenance, retrieve with temporal/contradiction awareness, and measure whether recalled evidence improves coding tasks. Current research consistently finds dynamic updates, obsolete facts, long-range consistency, selective forgetting, and poisoning harder than plain similarity search.

## AgentMemory comparison

“AgentMemory” is ambiguous:

- [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory) is the likely TypeScript reference. Its useful ideas are local capture, hybrid BM25/dense/graph retrieval, RRF, lifecycle metadata, hooks, provenance, and retrieval telemetry.
- [JordanMcCann/agentmemory](https://github.com/JordanMcCann/agentmemory) is a separate Python implementation with HNSW, BM25, graph and temporal reranking.
- [jayzeng/agentmemory](https://github.com/jayzeng/agentmemory) is a local Markdown/qmd-oriented CLI.
- [neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) is graph-native.

The rohitg00 headline 95.2% is retrieval `recall_any@5` on LongMemEval-S with a fresh index per question, not end-to-end QA. Its own [methodology](https://github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md) and [comparison caveat](https://github.com/rohitg00/agentmemory/blob/main/benchmark/COMPARISON.md) say competitor figures are not a controlled head-to-head comparison. It is a useful engineering reference, not verified proof of SOTA.

## Techniques worth adopting

1. **Immutable evidence, versioned claims.** Keep raw/semantic episodes append-only. Derive atomic semantic, episodic, procedural, and profile claims carrying source IDs, confidence, extractor/schema version, `validFrom`, `validTo`, and transaction time. Supersede or invalidate claims instead of deleting evidence. This is the strongest transferable lesson from [Zep/Graphiti](https://github.com/getzep/graphiti) and its [temporal knowledge-graph paper](https://arxiv.org/abs/2501.13956).

2. **Cheap capture, asynchronous enrichment.** Interactive writes should be idempotent and local; extraction, embeddings, linking, contradiction checks, and consolidation belong in durable background jobs. Logician already follows this direction. [LightMem](https://github.com/zjunlp/LightMem) supports separating lightweight online memory from offline consolidation.

3. **Hybrid, explainable retrieval.** Classify/decompose the query; apply scope/time/type filters; retrieve FTS, dense, file/entity, temporal, and small graph neighborhoods; fuse ranks with RRF; optionally rerank; then use diversity and hard token quotas. Every returned item should say why it matched and cite its evidence. Keep a deterministic FTS-only mode.

4. **Typed context quotas.** Reserve small budgets for project rules/preferences, current working state, recent episodes, durable claims, and procedural lessons. Letta/MemGPT's [virtual context model](https://arxiv.org/abs/2310.08560) is useful here, but agent-edited core memory needs versioning and rollback.

5. **Utility-aware learning.** Track whether retrieved evidence was expanded, cited, and followed by successful verification. Access frequency alone creates self-reinforcement and can promote poisoned or merely popular memories. Decay should affect ranking, not truth or deletion.

6. **Security as a retrieval property.** Label source/trust, keep memory text as quoted data rather than system instructions, quarantine untrusted writes, redact before persistence and embedding, and propagate deletion to vectors/relations/summaries. Recent memory-poisoning work—[Bad Memory](https://arxiv.org/abs/2607.14611), [Sleeper Memory Poisoning](https://arxiv.org/abs/2605.15338), and [MPBench](https://arxiv.org/abs/2606.04329)—shows persistence turns indirect prompt injection into a cross-session threat.

## What changed now

The initial hardening patch establishes prerequisites for trustworthy retrieval:

- ID-based memory update/delete is scoped to the active workspace.
- Semantic edits invalidate stale vectors; deletion/clear removes vectors.
- exports include only current-workspace sessions, observations, memories, and relations.
- the memory viewer binds to `127.0.0.1` instead of every network interface.
- regression coverage verifies workspace mutation/export isolation and embedding invalidation.

## Roadmap

### P0 — integrity (current patch, then follow-ups)

- Complete workspace checks across access tracking, tiers, relations, imports, and maintenance calls.
- Make export/import preserve IDs, version history, relations, provenance, and workspace exactly.
- Store embedding model, dimensions, content hash, and creation version; re-embed on mismatch.
- Add WAL/busy-timeout and multi-process contention tests.

### P1 — temporal claims and trust

- Add append-only `claims` and `claim_evidence` tables, validity intervals, confidence, status, source/trust channel, extractor/model/schema version, and tombstones.
- Make consolidation emit reversible claim revisions (`ADD`, `SUPERSEDE`, `INVALIDATE`, `NOOP`) while preserving all episodes.
- Never inject untrusted memory as instructions; show citations and validity in expanded records.

### P2 — retrieval depth
s
- Introduce one narrow `MemoryService` interface: capture/remember/recall/expand/forget/explain; keep repository, extractor, embedder, reranker, and maintenance seams internal.
- Replace the newest-4,000 brute-force vector cliff with an ANN adapter or a content-addressed candidate index.
- Add query planning, temporal/entity filters, one-to-two-hop relation expansion, calibrated reranking, typed quotas, and abstention.
- Log a retrieval waterfall (candidate ranks, fusion contributions, final selection, latency, token cost).

### P3 — evaluation before claims

- Build a repo-specific coding-memory corpus and compare recent-only, full-context, FTS-only, FTS+dense, graph, rerank, and consolidation ablations.
- Measure Recall@k/NDCG, evidence sufficiency, end-to-end task success, obsolete-fact rejection, abstention, shift recovery, latency, and tokens.
- Add [LongMemEval](https://github.com/xiaowu0162/LongMemEval), [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2), and poisoning cases. LongMemEval-V2's trajectory results support retaining raw artifacts and agentic evidence gathering rather than relying only on precompressed facts.

## Avoid

- A required Neo4j/vector database for the TUI default.
- LLM calls gating every write.
- Global recompression that destroys raw evidence.
- Treating access count or time decay as truth.
- Copying AgentMemory's broad tool/endpoint surface; deepen Logician's smaller interface instead.
