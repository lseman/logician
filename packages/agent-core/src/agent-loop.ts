// ── Agent Loop ─────────────────────────────────────────────────────────────
// Raw agent loop runner. For the simple API, use agent.ts.
// For full harness control, use core/harness/agent-harness.ts.

export {
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./core/execution/agent-loop-runner.ts";
