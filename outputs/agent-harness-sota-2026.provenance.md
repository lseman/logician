# Provenance: Logician Agent Harness Review

**Collected:** 2026-08-11  
**Method:** primary-source web research plus read-only repository audit. Claims from project READMEs are treated as project claims unless backed by code, documentation, or evaluations.

## External sources

- Pi repository: https://github.com/earendil-works/pi
- Pi sessions: https://pi.dev/docs/latest/sessions
- Pi session format: https://pi.dev/docs/latest/session-format
- Pi compaction: https://pi.dev/docs/latest/compaction
- Pi extensions: https://pi.dev/docs/latest/extensions
- Whale repository: https://github.com/usewhale/Whale
- OpenHands agent architecture: https://docs.openhands.dev/sdk/arch/agent
- OpenHands stuck detector: https://docs.openhands.dev/sdk/guides/agent-stuck-detector
- Codex repository: https://github.com/openai/codex
- Codex app-server protocol: https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md
- Codex Python SDK: https://github.com/openai/codex/blob/main/sdk/python/docs/api-reference.md
- Codex compaction-loop report: https://github.com/openai/codex/issues/13946
- OpenCode repository: https://github.com/opencode-ai/opencode
- OpenCode agents: https://opencode.ai/docs/agents
- OpenCode session specification: https://github.com/anomalyco/opencode/blob/dev/specs/v2/session.md
- SWE-agent repository: https://github.com/SWE-agent/SWE-agent
- SWE-agent trajectories: https://github.com/SWE-agent/SWE-agent/blob/main/docs/usage/trajectories.md
- mini-SWE-agent: https://github.com/SWE-agent/mini-swe-agent
- Aider: https://github.com/Aider-AI/aider
- Cline: https://github.com/cline/cline
- Goose: https://github.com/aaif-goose/goose
- Gemini CLI subagents: https://github.com/google-gemini/gemini-cli/blob/main/docs/core/subagents.md
- Gemini CLI checkpointing: https://google-gemini.github.io/gemini-cli/docs/cli/checkpointing.html

## Local sources audited

- `tui/packages/agent-core/src/agent/harness.ts`
- `tui/packages/agent-core/src/agent/agent-loop-runner.ts`
- `tui/packages/agent-core/src/agent/execution-policy.ts`
- `tui/packages/agent-core/src/agent/intervention-controller.ts`
- `tui/packages/agent-core/src/agent/guards/loop-detector.ts`
- `tui/packages/agent-core/src/agent/guards/response-patterns.ts`
- `tui/packages/agent-core/src/agent/tasks/task-state-controller.ts`
- `tui/packages/agent-core/src/agent/tasks/task-status-state.ts`
- `tui/packages/agent-core/src/agent/session.ts`
- `tui/packages/agent-core/src/compaction/compaction.ts`
- `tui/packages/agent-core/src/hooks/builtin/builtin-hooks.ts`
- `tui/packages/agent-core/src/queue/manager.ts`
- `tui/packages/agent-core/src/runtime/harness-queue-hooks.ts`
- `tui/packages/coding-agent/src/application/agent-bridge.ts`
- `tui/packages/coding-agent/src/application/loop-manager.ts`

## Confidence notes

- **High confidence:** local architecture and gap findings, based on direct source inspection.
- **High confidence:** Pi, OpenHands, Codex, OpenCode, SWE-agent, Gemini, Goose, Aider, and Cline feature descriptions taken from official repositories or documentation.
- **Medium confidence:** Whale performance and cache figures; these are upstream project claims and should be benchmarked independently before use as design targets.
- **Inference:** the recommended architecture and priorities synthesize external patterns with Logician’s current implementation; they are not claims made verbatim by any one source.
