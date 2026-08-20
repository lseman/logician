// ── Agent Facade ───────────────────────────────────────────────────────────
// Simple, opinionated agent API matching Pi's agent.ts pattern.
// For full control, import AgentHarness directly from "./harness/agent-harness.ts".

import { AgentHarness } from "./harness/agent-harness.ts";
import type { LLMBackend } from "./harness/utils/backend.ts";
import type { AgentConfig } from "./types/types-config.ts";
import type { Message } from "./types/types-messages.ts";

export type {
	AgentHarnessApi,
	BranchInfo,
	BranchSummaryData,
} from "./harness/agent-harness.ts";
export {
	type AbortResult,
	type AgentRuntimeState,
	HarnessBusyError,
	type HarnessPhase,
	type HarnessQueues,
} from "./harness/result.ts";
export type { AgentHarnessOptions } from "./harness/types.ts";
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
