# Logician memory SOTA roadmap (2026-08-13)

## Decision

Do **not** turn memory into a graph-first system or a mandatory remote server. Keep `packages/memory` as the embeddable core and `apps/memory-mcp` as an adapter. The best next step is to make every client use one traced retrieval pipeline, establish credible evaluation, and make concurrent background work safe. Then add temporal query planning, better vector retrieval, and narrowly measured entity/graph capabilities.

Logician already has stronger truth-integrity foundations than many agent-memory libraries: immutable observations, bitemporal claims, evidence certificates, validity predicates, supersession and contradiction relations, hybrid lexical/dense retrieval, quotas, abstention, outcome receipts, and a shadow-only learned policy. The largest gaps are not missing fashionable memory types. They are path parity, evaluation validity, concurrency, authority boundaries, and retrieval quality at scale.

## What “SOTA” should mean here

For this repository, SOTA should mean measurably better agent outcomes under changing facts, long histories, paraphrased queries, implicit constraints, and hostile content—within declared latency, token, and safety budgets. It should not mean the most storage backends, graph edges, or MCP tools.

Recent benchmarks expose complementary failure modes. [LongMemEval](https://arxiv.org/abs/2410.10813) tests extraction, cross-session reasoning, temporal reasoning, knowledge updates, and abstention. [LoCoMo](https://aclanthology.org/2024.acl-long.747/) covers multi-hop and temporal conversational reasoning, while [LoCoMo-Plus](https://arxiv.org/abs/2602.10715) stresses latent constraints whose later cues do not lexically match their original trigger. [MemoryAgentBench](https://arxiv.org/abs/2507.05257) adds test-time learning, long-range understanding, and selective forgetting. These should be complemented by a private coding-agent stream where success is verified in the environment, not only judged from generated text.

## Immediate findings in this repository

### 1. MCP currently bypasses the best memory path

`memory_search` combines list/search results directly rather than calling the context selector. It therefore misses semantic vectors, claim validity, quotas, temporal handling, retrieval traces, and the shadow policy. It can also return twice the requested limit. Meanwhile `memory_feedback` requires a retrieval-trace ID that MCP search does not create. External agents therefore cannot participate in the outcome loop as designed.

**Required change:** expose one canonical `retrieve()` operation from `packages/memory`. It should return globally ranked typed results, citations, score contributions, and a trace ID. Embedded hooks, TUI, and MCP must all call it. Feedback receipts must join to that trace.

### 2. The benchmark cannot support comparative claims

The current corpus contains four synthetic cases, with claims seeded at the same high importance and query-derived concepts. `recallAt5` does not actually truncate selected IDs at five, and `ndcgAt5` scores only the first relevant item rather than DCG across the first five. The harness checks inclusion in context, not whether the agent correctly answers or acts.

**Required change:** repair the metrics, add fixed public splits and private coding episodes, and evaluate retrieval, answer utility, and environment success separately. Report 95% bootstrap confidence intervals, latency, tokens, write cost, and per-category regressions.

### 3. Multiple MCP processes can duplicate background work

Opening a store globally changes every `running` extraction job back to `pending`. A second MCP process can therefore steal healthy work. Jobs lack owner IDs, lease expiry, heartbeats, and fencing tokens; completion is not owner-fenced. SQLite WAL improves read/write coexistence but does not remove the single-writer and busy-handling constraints documented by [SQLite](https://sqlite.org/wal.html).

**Required change:** add atomic leased claims with `owner_id`, `lease_until`, heartbeat, attempt/backoff, fencing token, idempotent output key, and a dead-letter state. Validate with a multi-process kill-and-recovery soak test.

### 4. Truth controls are stronger than authority controls

Workspace scoping and untrusted-content quarantine are useful, but a path-like workspace ID is not an authorization boundary. Derived summaries can also launder origin or authority if those labels are not propagated. Shared multi-agent memory requires hierarchical scopes and identical enforcement on search and direct reads.

**Required change:** use explicit namespace/principal fields such as organization, user, workspace, agent, session, and memory kind; immutable writer/source/authority provenance; retrieval-time ACL filtering; retention and erasure propagation; and audit records. Retrieved memory must remain quoted evidence, never authority to execute an action.

## Prioritized roadmap

| Priority | Work | Acceptance gate |
|---|---|---|
| P0 | Canonical traced retrieval used by package, hooks, TUI, and MCP | Global limit respected; semantic, temporal, validity, and scope behavior is identical across clients; every accepted feedback receipt joins to a retrieval trace |
| P0 | Credible evaluation suite and corrected metrics | LongMemEval-style update/temporal/abstention slices, LoCoMo-Plus-style implicit constraints, MemoryAgentBench-style forgetting, and private coding tasks; correct Recall@k, MRR, nDCG, context precision, answer/environment success, cost and latency with confidence intervals |
| P0 | Durable leased job runner | No healthy lease theft or lost jobs under 16-process stress and random termination; one committed result per idempotency key; bounded recovery and zero unhandled `SQLITE_BUSY` failures |
| P0 | Namespace, ACL, provenance, retention, and deletion model | Zero cross-scope leakage in randomized tests; authorization enforced on search and direct IDs; origin/authority survives every derivation; deletion propagates to FTS, vectors, summaries, traces, and export |
| P1 | Real model-versioned vector index plus hybrid reranking | Isolate embeddings by model/version; ANN recall@50 at least 0.95 against exact cosine on the declared scale; p95 target declared and met; lexical-only deterministic fallback retained |
| P1 | Typed query planning and temporal retrieval | Parse entities, files, requested memory type, time windows, and `asOf`; use fact-augmented keys and time-aware query expansion; update/temporal QA and obsolete-fact leakage meet predeclared gates |
| P1 | Working-context blocks separate from retrieved history | Bounded project/profile/task/procedure blocks with schema, token budget, ownership, read-only option, audit history, and compare-and-swap updates |
| P1 | Faithful, reversible consolidation | Every derived claim cites exact event/source IDs; raw evidence remains recoverable; extractor/schema versions support re-materialization; entailment, coverage, and contradiction rates are measured |
| P2 | Lightweight entity layer and temporal neighbor expansion | Normalize entity mentions and claim links first; add one-to-two-hop traversal only if held-out multi-hop accuracy improves materially without violating latency/token budgets |
| P2 | Learned memory controller | Keep `ADD/UPDATE/SUPERSEDE/NOOP`, retrieval, and promotion policies replaceable and shadow-only until attributable outcome data yields a statistically significant environment-success gain with no safety regression |

## Retrieval architecture to target

1. Classify and expand the query into scope, entities/files, time intent, memory types, and task phase.
2. Generate independent lexical, dense, temporal, and entity candidates.
3. Fuse candidates with reciprocal-rank fusion and retain every score contribution in the trace.
4. Optionally rerank only a small top set with a replaceable cross-encoder or model adapter.
5. Expand temporal neighbors or relation hops only for queries that need them.
6. Apply validity, ACL, trust, diversity, evidence coverage, and hard token quotas.
7. Abstain when evidence is insufficient, and return stable evidence IDs plus a trace ID.

This follows the strongest common ground across current systems without copying their entire architecture. Graphiti combines immutable episodes, temporal facts, provenance, hybrid search, MMR and graph-aware reranking ([overview](https://help.getzep.com/graphiti/getting-started/overview), [search](https://help.getzep.com/graphiti/working-with-data/searching)). LangGraph separates thread checkpoints from cross-thread namespaced memories and distinguishes semantic, episodic, and procedural memory ([memory concepts](https://docs.langchain.com/oss/python/concepts/memory)). Letta uses small bounded, optionally read-only memory blocks for always-visible state, with larger history outside the immediate context ([memory blocks](https://docs.letta.com/guides/core-concepts/memory/memory-blocks)).

The research supports experimentation, not unconditional adoption. [Zep’s paper](https://arxiv.org/abs/2501.13956) reports strong temporal-graph retrieval, but it is vendor-authored. [Mem0’s paper](https://arxiv.org/abs/2504.19413) reports large token and latency savings from extraction/consolidation, while its graph variant adds a comparatively modest overall gain. [A-MEM](https://arxiv.org/abs/2502.12110) supports dynamically linked structured notes, and [EM-LLM](https://arxiv.org/abs/2407.09450) supports event boundaries plus temporally contiguous retrieval. These justify adapters and ablations—not a graph rewrite.

## Evaluation design

Use three layers:

1. **Retrieval:** Recall@k, nDCG@k, MRR, context precision, evidence coverage, obsolete-memory rejection, abstention calibration, and ANN recall against exact search.
2. **Answer utility:** deterministic exact/F1 or task-specific graders where possible; an LLM judge can be secondary, never the only metric.
3. **Agent outcome:** paired memory-on/memory-off coding runs, verified by tests or environment state, with repeated trials and bootstrap confidence intervals.

Every experiment should record tokens, p50/p95 latency, writes, model calls, index size, and hardware. Required ablations are recent-only, full-context, lexical, lexical+dense, temporal expansion, reranker, relations/graph, consolidation, validity filtering, and learned policy. A feature should ship only if it improves its intended category without material regressions elsewhere.

## What to avoid

- A graph-first rewrite before a graph ablation shows meaningful multi-hop or temporal lift.
- An LLM memory manager on every hot-path write.
- Automatic destructive `DELETE`; prefer append-only supersession, invalidation, archive, and governed erasure.
- Training from global outcomes or model self-ratings without attributable actions, propensities, holdouts, and rollback.
- Treating recency, access count, or decay as truth.
- Deleting raw evidence after lossy consolidation.
- Exposing governance/destructive operations as ordinary model-visible MCP tools.
- Claiming SOTA from a single vendor benchmark, retrieval-only score, or unmatched token/retrieval budget.

## Recommended first implementation slice

Create `MemoryService.retrieve()` and route MCP through it; fix Recall@5/nDCG@5; add trace-linked feedback parity tests; then implement job leases. This slice closes correctness holes, unlocks trustworthy measurements, and provides the data needed to decide whether reranking, temporal expansion, graph traversal, or learned policies are actually worthwhile.
