// ── Agent Loop ─────────────────────────────────────────────────────────────
// Raw agent loop runner. For the simple API, use agent.ts.
// For full harness control, use harness/agent-harness.ts.

export {
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./harness/utils/agent-loop.ts";
