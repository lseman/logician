// ── Agent Facade ───────────────────────────────────────────────────────────
// Simple, opinionated agent API matching Pi's agent.ts pattern.
// For full control, import AgentHarness directly from "./core/harness/agent-harness.ts".

import { AgentHarness } from "./core/harness/agent-harness.ts";
import type { LLMBackend } from "./core/provider/backend.ts";
import type { AgentConfig } from "./core/types/types-config.ts";
import type { Message } from "./core/types/types-messages.ts";

export type {
	BranchInfo,
	BranchSummaryData,
} from "./core/harness/agent-harness.ts";
export {
	type AgentRuntimeState,
	type HarnessPhase,
} from "./core/state/runtime-state.ts";
export { HarnessBusyError } from "./core/harness/runtime/phase.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./core/harness/types.ts";
export { AgentHarness };

/**
 * Run the agent with a simple config. This is the entry point for most use cases.
 * For streaming, lifecycle hooks, or queue management, use AgentHarness directly.
 */
export async function runAgent(
	config: AgentConfig,
	backend: LLMBackend,
	prompt: string,
	options?: { cwd?: string; maxIterations?: number },
): Promise<Message[]> {
	const harness = new AgentHarness({
		config,
		backend,
		cwd: options?.cwd,
		maxIterations: options?.maxIterations,
	});

	return harness.prompt(prompt);
}
