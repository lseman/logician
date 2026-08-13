# Improving the Logician agent: deep-research brief

**Date:** 2026-08-13  
**Scope:** agent quality, harness reliability, repository context, tools, security, observability, memory/RAG, subagents, and delivery.

## Executive conclusion

Logician has already implemented much of the infrastructure recommended by its earlier research: a versioned append-only Run Kernel, durable cross-run budgets, semantic no-progress accounting, explicit tool-effect recovery, structured completion state, repository mapping, retrieval traces, and a broad component-test suite. The next bottleneck is no longer missing agent machinery. It is the absence of a representative end-to-end evaluation system that can show which machinery improves real coding outcomes.

The highest-leverage strategy is:

> Freeze the harness, build a private repo-native task set, grade external outcomes, and require every prompt, retrieval, memory, tool, reasoner, or delegation change to earn its complexity through measured quality per unit cost.

Two urgent engineering risks sit beside that evaluation work: stdio MCP servers inherit the parent process's complete environment, and sandbox execution can fall back to an ordinary unsandboxed shell when isolation is unavailable. Release artifacts are also built without a preceding CI gate.

## What the repository already does well

- `agent-core` has a durable Run Kernel with schema-versioned events, replay, leases/fencing, task-spanning budgets, explicit operation intent/result/commit stages, idempotency keys, and conservative crash recovery.
- Completion and continuation are substantially stronger than the 2026-08-11 harness report described: structured task status, a completion gate, persistent continuation limits, trajectory projection, and runtime status are present.
- Memory has temporal/versioned claims, provenance and trust labels, hybrid retrieval, typed quotas, abstention, and retrieval traces.
- RAG has hybrid retrieval and component-level evaluation primitives.
- The test suite is healthy: on 2026-08-13, `bun run typecheck && bun run test` passed across all workspaces.

These are strong foundations. They should now become the substrate for experiments, rather than invitations to add more features.

## Ranked recommendations

### P0 — Build an end-to-end agent evaluation gate

Create an `agent-eval` package and CLI backed by Run Kernel trajectories. Start with 30–50 frozen tasks derived from real Logician failures and feature work, stratified by bug fix, feature, refactor, documentation, investigation, task duration, provider, and permission mode. Keep a smaller 10–12 task smoke set for pull requests.

Each task should pin repository revision, environment, prompt, allowed actions, acceptance tests, and forbidden regressions. Grade the resulting environment—not the agent's prose or self-authored `task_state`. Use deterministic tests and repository assertions as the authority; model trace graders may diagnose strategy and style but must not be the sole success criterion.

Report at least:

- task success and pass@1 across 3–5 trials;
- regression rate, unnecessary edits, and scope violations;
- premature stops and redundant continuations;
- tool-call validity, failures, retries, and approval burden;
- wall time, model/tool calls, tokens, cache use, and cost;
- retrieval precision/recall for files and regions actually needed;
- user corrections, interruption recovery, and rollback rate;
- safety violations and secret-canary exposure.

Slice results by task type and human-duration bucket. METR's work shows that task length strongly predicts agent success, so one aggregate pass rate hides the actual reliability frontier. Anthropic recommends representative tasks, multiple trials and graders, complete transcript retention, and outcome-based grading. NIST similarly emphasizes objective, repeatable, deployment-like, uncertainty-aware TEVV ([Anthropic agent evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [METR task horizons](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/), [NIST TEVV guidance](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)).

Do not make SWE-bench Verified the north star. OpenAI now argues it is contaminated and structurally unreliable for frontier measurement; recent audits also show that newer public coding benchmarks require manual task validation. Use public benchmarks only as secondary comparators, with private post-cutoff tasks and canaries as the primary signal ([OpenAI on SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/), [coding-evaluation signal and noise](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)).

### P0 — Make execution security fail closed

Treat the current permission denylist as a UX guard, not a security boundary.

1. Stdio MCP processes currently receive `{ ...process.env, ...configuredEnv }`. Replace this with a minimal explicit environment allowlist and capability-scoped secret injection. Record which secret references—not values—were made available.
2. When the requested sandbox backend is unavailable, autonomous execution should pause or fail closed. Never silently convert “sandboxed” into an ordinary host shell. User policy must not be overridable by a model-supplied profile argument.
3. Add typed tool capability metadata: `read`, `write`, `execute`, `network`, `credential`; plus reversibility, affected scope, provenance, and recovery semantics. Evaluate canonical argv, paths and hosts rather than a single raw argument string.
4. Label all repository, web, RAG, MCP, plugin, skill, memory, and subagent content by trust/provenance. External content is data, never instruction. Propagate taint into tool authorization and memory writes.
5. Add adversarial fixtures for repository-file injection, malicious tool output/MCP metadata, poisoned memory, credential canaries, command obfuscation, symlink/Unicode paths, approval spoofing, and continuation/replay attacks.

MCP's authorization model requires audience-bound tokens and prohibits token passthrough; its tool guidance calls for input validation, access control, rate limits, output sanitization, confirmations, and auditability. OWASP identifies excessive permissions, functionality, and autonomy as core agent risks ([MCP authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization), [MCP tool security](https://modelcontextprotocol.io/specification/2024-11-05/server/tools), [OWASP excessive agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)).

### P0 — Add CI and release provenance

There is no code PR/push workflow, and tag releases build and publish binaries without running `bun run ci` or Ariadne's Rust checks. Add required Linux/macOS CI for typecheck, lint, format, tests, focused eval smoke tasks, binary startup, and installer verification. Make release jobs depend on the tested commit and add signed artifact provenance/SBOM; GitHub supports build provenance attestations directly ([GitHub artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)).

### P1 — Turn the Run Kernel into the unified trace substrate

Preserve the event ledger as the source of truth, but enrich its evaluation projection with provider/model/config/prompt/tool versions, token and cost accounting, repository base/head/diff manifest, grader evidence, test provenance, memory/RAG selections, user corrections, and rollback outcomes. Correlate sessions, model calls, tools, permissions, compactions, retrievals, and subagents as parent-child spans and support privacy-aware OpenTelemetry export. OpenAI's Agents SDK traces generations, tool calls, handoffs and guardrails, while warning that model/tool inputs can be sensitive ([OpenAI Agents tracing](https://openai.github.io/openai-agents-js/guides/tracing/), [OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)).

Rename or redefine `acceptancePassed`: today it can be derived from the agent's own completed outcome and task state. Distinguish `agentDeclaredComplete`, `contractReportedPass`, and `environmentGradedPass`; only the last should gate quality claims.

### P1 — Evaluate and deepen repository retrieval

The current repository map is a useful lightweight baseline, but its synchronous file listing, regex symbol/import extraction, and substring scoring need an eval before further sophistication. Benchmark:

- lexical/BM25;
- AST/LSP symbols and references;
- import/call/dependency graph expansion;
- changed-file/test pairing;
- region-level reranking and explicit abstention;
- hybrid task-conditional routing.

Log both explored and used files/regions. Recent repository-retrieval research finds no universally dominant method and shows that file discovery remains a major failure source, supporting an evaluated hybrid rather than a vector-everything rewrite ([Agent Retrieval Bench](https://arxiv.org/abs/2607.24882), [SWE-Explore](https://arxiv.org/abs/2606.07297), [ContextBench](https://arxiv.org/abs/2602.05892)).

### P1 — Optimize tools and context through ablation

Build tool-selection cases measuring correct tool choice, argument validity, observation usefulness, and tokens consumed. Simplify or namespace overlapping tools and tune descriptions only when the cases improve. Treat context as a budget allocator: ablate system-prompt sections, skills, repository map, memory, RAG, and recent raw history individually and in combinations. Anthropic's guidance stresses eval-driven tool design and dynamic curation of finite context ([tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), [context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

### P1 — Add state-machine and fault-injection coverage

Concentrate tests on high-complexity seams: `agent-loop-runner`, `harness`, `agent-bridge`, the memory store, MCP, and autoresearch. Cover partial/torn writes, every `intent → result → commit` crash point, lease takeover, cancellation races, compaction interruption, branch/rewind, concurrent steering and children, MCP disconnects, malformed streams, migrations, and provider incompatibilities. Add characterization tests before decomposing the largest modules.

### P2 — Keep advanced reasoning and multi-agent work conditional

Reflection, ToT/SSR/Reflexion, semantic memory, reranking, and delegation should remain optional until equal-budget ablations show higher environment-graded success. Compare parallel agents with a single agent under the same token/time budget and measure duplicated exploration, merge conflicts, parent-integration failures, and partial-child failure. Complex multi-agent designs help when tasks genuinely decompose, but simple loops remain the stronger default ([Anthropic effective agents](https://www.anthropic.com/engineering/building-effective-agents), [Anthropic multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system), [Agentless](https://arxiv.org/abs/2407.01489)).

## First two-week engineering slice

1. Define the task, trial, trajectory, grader, and report schemas.
2. Curate 12 private Logician-native tasks and freeze their repositories/environments.
3. Implement deterministic environment graders and JSON/HTML reporting over Run Kernel traces.
4. Run the present agent with three trials per task to establish quality, cost, duration, continuation, and retrieval baselines.
5. Add CI that runs the component suite plus a small deterministic eval smoke set.
6. Fix MCP environment inheritance and sandbox fail-open behavior.
7. Trial exactly two measured changes: one repository-retrieval improvement and one tool-description/interface improvement. Keep them only if they raise task success without breaching cost, latency, or safety thresholds.

## Decisions to avoid until evidence changes

- Adding more autonomous subagents or self-critique loops.
- Growing the always-on system prompt.
- Making vector or graph memory mandatory.
- Letting an LLM safety reviewer substitute for least privilege.
- Optimizing a public benchmark headline without contamination/task audits.
- Building dashboards before trace semantics and external grading are trustworthy.

## Bottom line

Logician's architecture is ahead of its evidence. The most valuable improvement is an empirical control loop that connects real tasks to independently graded outcomes and the Run Kernel's rich telemetry. Once that exists, repository retrieval, tool design, context allocation, memory, reasoners, and delegation become testable hypotheses rather than architectural bets.
