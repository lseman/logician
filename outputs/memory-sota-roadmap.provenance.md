# Provenance: memory-sota-roadmap.md

Research date: 2026-08-13

Method: direct inspection of `packages/memory`, its benchmark, and `apps/memory-mcp`; parallel academic, production-system, and adversarial review passes; web research restricted to papers, official documentation, specifications, and primary repositories. Recommendations are a synthesis for this repository, not claims that one external architecture is universally SOTA.

## Research and benchmark sources

- LongMemEval: https://arxiv.org/abs/2410.10813
- LoCoMo: https://aclanthology.org/2024.acl-long.747/
- LoCoMo-Plus: https://arxiv.org/abs/2602.10715
- MemoryAgentBench: https://arxiv.org/abs/2507.05257 and https://github.com/HUST-AI-HYZ/MemoryAgentBench
- LongMemEval-V2: https://arxiv.org/abs/2605.12493
- MemBench: https://aclanthology.org/2025.findings-acl.989/
- Zep temporal knowledge graph: https://arxiv.org/abs/2501.13956
- Mem0: https://arxiv.org/abs/2504.19413
- A-MEM: https://arxiv.org/abs/2502.12110
- EM-LLM: https://arxiv.org/abs/2407.09450
- Memory-R1: https://aclanthology.org/2026.acl-long.583/
- AgeMem: https://aclanthology.org/2026.acl-long.981/
- HippoRAG 2: https://arxiv.org/abs/2502.14802

## Official system sources

- Graphiti overview/search: https://help.getzep.com/graphiti/getting-started/overview and https://help.getzep.com/graphiti/working-with-data/searching
- Letta memory blocks/context hierarchy: https://docs.letta.com/guides/core-concepts/memory/memory-blocks and https://docs.letta.com/guides/core-concepts/memory/context-hierarchy
- LangGraph memory/persistence: https://docs.langchain.com/oss/python/concepts/memory and https://docs.langchain.com/oss/python/langgraph/persistence
- Mem0 architecture/graph memory: https://docs.mem0.ai/core-concepts/memory-evaluation and https://docs.mem0.ai/open-source/features/graph-memory
- Cognee memory: https://docs.cognee.ai/core-concepts/main-operations/remember
- Redis Agent Memory: https://github.com/redis/agent-memory-server
- AgentMemory: https://github.com/rohitg00/agentmemory
- SQLite WAL: https://sqlite.org/wal.html
- MCP tools/transports specifications: https://modelcontextprotocol.io/specification/2025-11-25/server/tools and https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

## Verification notes

- Vendor-authored benchmark results are treated as directional and are not assumed to be independently replicated.
- Local evaluation findings were verified against the current working tree: four synthetic corpus cases; Recall@5 and nDCG@5 implementation defects; MCP search bypassing canonical context selection/tracing; and globally resetting running extraction jobs on store initialization.
- No source establishes a universal SOTA architecture across memory workloads. The roadmap therefore requires fixed baselines, ablations, cost accounting, confidence intervals, and category-specific no-regression gates.
- This research task added only the report and provenance files; it did not modify memory implementation code or overwrite the user’s existing worktree changes.
