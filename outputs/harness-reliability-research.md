# Harness reliability research brief

## Findings

The strongest current pattern is not a single “SOTA loop detector.” Reliable
agent harnesses combine bounded execution, explicit resumable state, layered
guardrails, outcome-based verification, and complete trajectory observability.

- OpenAI's Agents SDK uses bounded runner loops, abort signals, lifecycle
  events, tracing, and serializable run state for interruption and resumption.
- Anthropic recommends grading agents against environment outcomes rather than
  their own completion claims, retaining complete trajectories, inspecting
  failures, and evaluating over multiple trials.
- SWE-agent reports that the agent-computer interface materially affects coding
  performance, supporting concise, structured, bounded tool observations.
- OpenHands uses sandboxed execution and an event-stream architecture, which
  supports replay, monitoring, and multi-agent coordination.
- Anthropic's SHADE-Arena demonstrates independent transcript-level monitoring,
  suggesting a separate monitor for high-risk or long-running trajectories.

## Recommended next architecture

1. Replace free-form intervention strings with one durable typed lifecycle:
   `started`, `progress`, `resolved`, and `failed`, carrying an incident ID,
   detector, evidence, attempt, limits, and chosen action.
2. Gate completion on external evidence: acceptance checks, tests, repository
   state, and unresolved plan items—not only model language.
3. Use a staged recovery ladder: warn and nudge, require a strategy change,
   checkpoint, then pause for user input. Do not blindly repeat continuation.
4. Track multiple budgets: turns, model calls, tokens/cost, tool calls, elapsed
   time, consecutive failures, and unchanged iterations.
5. Persist a resumable checkpoint after tool boundaries and record idempotency
   information for external mutations.
6. Evaluate detector false-positive/false-negative rates on replayable real and
   injected-loop trajectories, with multiple trials and task-success/cost data.

## Changes applied in this pass

- Routed builtin guard events to the application/TUI subscriber path.
- Added notices for continuation sources, exhaustion, retry lifecycle,
  compaction recovery, queued next-turn work, and reasoner completion/failure.
- Corrected thinking-loop and diminishing-return stops so they cannot be
  reported as successful completion.
- Removed duplicate loop and max-iteration intervention notices.
- Changed stagnation to require consecutive no-progress turns.
- Cleared stale failure counts after successful tool work.
- Preserved meaningful numeric tool arguments in duplicate signatures.

## Primary sources

- OpenAI Agents SDK runner: https://openai.github.io/openai-agents-js/guides/running-agents/
- OpenAI Agents SDK agents: https://openai.github.io/openai-agents-js/guides/agents/
- OpenAI Agents SDK usage tracking: https://openai.github.io/openai-agents-python/usage/
- OpenAI practical guide to agents: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- Anthropic agent evaluations: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- SWE-agent: https://arxiv.org/abs/2405.15793
- OpenHands: https://arxiv.org/abs/2407.16741
- SHADE-Arena: https://www.anthropic.com/research/shade-arena-sabotage-monitoring
