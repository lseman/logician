---
title: Runtime Design Decisions
description: The external evidence and invariants guiding Logician's modern runtime.
---

# Runtime Design Decisions

Logician adopts patterns only when they strengthen a measured invariant. The
runtime is organized around four decisions:

1. **One effective configuration.** User and trusted project layers are
   validated and deeply merged once. Every subsystem receives that resolved
   snapshot. Runtime application is side-effect free; only explicit user
   changes persist, using atomic field updates.
2. **An event ledger, not mutable run-state files.** The Run Kernel records
   typed semantic transitions and reconstructs projections by replay. Streaming
   deltas remain ephemeral; durable writes happen at turn, tool, queue, and
   recovery boundaries.
3. **A responsive presentation boundary.** Input, scrolling, and rendering read
   bounded projections. Network discovery, extraction, persistence, and other
   potentially slow work run outside the input/render path.
4. **Explicit effects and recovery.** Tool intent, result, and commit are
   separate states. Recovery behavior is declared per tool, and unknown
   external effects are quarantined rather than repeated optimistically.

These choices align with current primary-source designs:

- [OpenAI Codex configuration](https://github.com/openai/codex/blob/main/docs/config.md)
  and its [machine-readable schema](https://github.com/openai/codex/blob/main/codex-rs/core/config.schema.json)
  use explicit configuration ownership, profiles, policy controls, and
  inspectable effective settings.
- [Codex app-server approvals](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
  model approvals as structured turn/thread interactions instead of generic
  transcript chatter.
- [Gemini CLI configuration](https://github.com/google-gemini/gemini-cli/blob/main/docs/reference/configuration.md)
  defines ordered settings layers, schema-backed editor validation, and
  restart boundaries. Its release notes attribute responsiveness improvements
  to an event-driven tool scheduler and incremental terminal rendering.
- [LangGraph persistence](https://docs.langchain.com/oss/javascript/langgraph/persistence)
  checkpoints semantic super-steps and supports replay/forks, while
  [interrupts](https://docs.langchain.com/oss/javascript/langgraph/interrupts)
  require side effects around resumable boundaries to be idempotent.

The comparison is a constraint, not a feature checklist. Logician deliberately
keeps MCP discovery non-blocking, suppresses routine automatic permission
allows from the chat transcript while retaining durable audit records, and does
not persist token-by-token streaming events.

## Verification gates

Material runtime changes must include the relevant gate:

- deterministic replay and torn-write/recovery tests for durable state;
- atomic-update and layered-merge tests for configuration;
- permission audit tests that distinguish routine allows from action-required
  decisions;
- keystroke and streaming benchmarks whose cost stays flat with transcript
  history;
- timeout, cancellation, and output-bound tests for subprocesses.
