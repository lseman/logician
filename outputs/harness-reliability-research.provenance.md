# Provenance

| Claim area | Source | Source type | Use |
|---|---|---|---|
| Bounded loops, abort, resumable state, tracing | https://openai.github.io/openai-agents-js/guides/running-agents/ | Official documentation | Architecture comparison |
| Agent lifecycle and guardrails | https://openai.github.io/openai-agents-js/guides/agents/ | Official documentation | Architecture comparison |
| Token usage accounting | https://openai.github.io/openai-agents-python/usage/ | Official documentation | Budget recommendation |
| Layered guardrails and human handoff | https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/ | Official guidance | Escalation recommendation |
| Outcome-based, multi-trial agent evaluation | https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents | Primary lab guidance | Evaluation recommendation |
| Agent-computer interface effects | https://arxiv.org/abs/2405.15793 | Research paper | Tool-interface recommendation |
| Sandboxing and event streams | https://arxiv.org/abs/2407.16741 | Research paper | Runtime architecture comparison |
| Independent trajectory monitoring | https://www.anthropic.com/research/shade-arena-sabotage-monitoring | Primary lab research | Monitor recommendation |

Repository findings were derived from direct inspection of `agent-core`,
`coding-agent`, and TUI runtime event paths on 2026-08-10. Recommendations are
explicitly synthesis/inference unless stated as source behavior.
