# Provenance: Improving the Logician agent

**Research date:** 2026-08-13

## Method

- Inspected the current repository architecture, implementation, tests, documentation, prior research briefs, Git history, and release workflows.
- Ran `bun run typecheck && bun run test`; all workspaces completed successfully.
- Used three independent roles required by the deep-research workflow: researcher, verifier, and skeptical reviewer.
- Compared findings and retained recommendations where repo evidence and primary/current sources converged.
- No implementation files were changed; only this research brief and sidecar were added.

## Repository evidence inspected

- `README.md`
- `docs/architecture/{overview,modernization,run-kernel}.md`
- `outputs/{agent-harness-sota-2026,memory-sota-2026,harness-reliability-research}.md`
- `tui/packages/agent-core/src/agent/{run-kernel,run-kernel-events,trajectory,agent-loop-runner,harness}.ts`
- `tui/packages/coding-agent/src/{application/agent-bridge,mcp/client,context/system-prompt}.ts`
- `tui/packages/coding-agent/src/context/repository-map.ts`
- `tui/packages/agent-core/src/permissions.ts` and tool types
- memory and RAG retrieval/evaluation code and test inventories
- `.github/workflows/{release,deploy-pages}.yml`
- root and workspace package scripts

## Primary/current external sources

- Anthropic, *Demystifying evals for AI agents*: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Anthropic, *Writing effective tools for agents*: https://www.anthropic.com/engineering/writing-tools-for-agents
- Anthropic, *Effective context engineering for AI agents*: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, multi-agent research system: https://www.anthropic.com/engineering/multi-agent-research-system
- OpenAI, SWE-bench Verified retirement rationale: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- OpenAI, coding-evaluation signal/noise: https://openai.com/index/separating-signal-from-noise-coding-evaluations/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-js/guides/tracing/
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
- NIST AI RMF TEVV resources: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/
- MCP authorization specification: https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization
- MCP tool security guidance: https://modelcontextprotocol.io/specification/2024-11-05/server/tools
- OWASP excessive agency: https://genai.owasp.org/llmrisk/llm062025-excessive-agency/
- GitHub artifact attestations: https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
- METR task-horizon measurement: https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
- Agent Retrieval Bench: https://arxiv.org/abs/2607.24882
- SWE-Explore: https://arxiv.org/abs/2606.07297
- ContextBench: https://arxiv.org/abs/2602.05892
- Agentless: https://arxiv.org/abs/2407.01489

## Confidence and limitations

- **High confidence:** lack of an end-to-end task-quality eval gate; missing PR CI; release builds not gated by CI; complete environment inheritance by stdio MCP; strong existing Run Kernel foundation.
- **Medium-high confidence:** repository retrieval, context allocation, tool interfaces, memory, reflection, and delegation should be optimized through ablations rather than expanded by default.
- **Needs exploit testing:** practical reachability and severity of sandbox fallback, MCP credential exposure, prompt injection, and memory-poisoning paths.
- Deployed controls, private telemetry, and operational policies outside this repository were not visible.
