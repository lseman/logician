# Provenance: memory-sota-2026.md

Research date: 2026-08-10

Method: local package/call-site inspection; package test and typecheck baseline; shallow checkout of `rohitg00/agentmemory`; independent researcher, verifier, and code reviewer passes; primary repositories, papers, benchmark descriptions, and official documentation preferred over secondary summaries.

Primary sources consulted:

- AgentMemory TypeScript repository and benchmark methodology: https://github.com/rohitg00/agentmemory ; https://github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md ; https://github.com/rohitg00/agentmemory/blob/main/benchmark/COMPARISON.md
- Other repositories sharing the name: https://github.com/JordanMcCann/agentmemory ; https://github.com/jayzeng/agentmemory ; https://github.com/neo4j-labs/agent-memory
- Mem0 repository/paper: https://github.com/mem0ai/mem0 ; https://arxiv.org/abs/2504.19413
- Zep/Graphiti repository/paper: https://github.com/getzep/graphiti ; https://arxiv.org/abs/2501.13956
- Letta/MemGPT repository/paper: https://github.com/letta-ai/letta ; https://arxiv.org/abs/2310.08560
- A-MEM repository/paper: https://github.com/agiresearch/A-mem ; https://arxiv.org/abs/2502.12110
- HippoRAG repository: https://github.com/OSU-NLP-Group/HippoRAG
- LightMem repository/paper: https://github.com/zjunlp/LightMem ; https://arxiv.org/abs/2510.18866
- Hindsight repository/paper: https://github.com/vectorize-io/hindsight ; https://arxiv.org/abs/2512.12818
- LongMemEval and LongMemEval-V2: https://github.com/xiaowu0162/LongMemEval ; https://github.com/xiaowu0162/LongMemEval-V2 ; https://arxiv.org/abs/2605.12493
- MemoryAgentBench: https://openreview.net/forum?id=DT7JyQC3MR
- Dynamic/obsolete-memory evaluation: https://arxiv.org/abs/2604.20006
- Distribution-shift evaluation: https://openreview.net/forum?id=CCSztIjmOy
- Memory poisoning/security: https://arxiv.org/abs/2607.14611 ; https://arxiv.org/abs/2605.15338 ; https://arxiv.org/abs/2606.04329 ; https://owasp.org/www-project-agent-memory-guard/

Verification notes:

- Vendor and repository benchmark claims are reported as self-reported unless independently replicated.
- The rohitg00 95.2% result is characterized only as retrieval Recall@5 under its documented setup, not end-to-end accuracy.
- Recommendations are synthesis/inference from sources and the inspected Logician implementation; they are not claims that any single architecture is universally SOTA.
- Local validation after the initial patch: 109 tests passed, zero failures; `tsc --noEmit` passed; `git diff --check` passed.
