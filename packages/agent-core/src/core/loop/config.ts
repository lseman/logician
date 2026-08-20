import type { AcceptanceConfig } from "../guards/acceptance-contract.ts";
import type { OutputGuard } from "../guards/output-guard.ts";
import type { HarnessInterventionController } from "../policy/intervention-controller.ts";
import type { LLMBackend } from "../provider/backend.ts";
import type { AgentConfig, Message } from "../types/index.ts";

/** Configuration understood by the agent execution mechanism. */
export type AgentLoopOptions = Pick<
	AgentConfig,
	| "acceptance"
	| "allowAllPaths"
	| "allowedPaths"
	| "cacheSize"
	| "cacheTtlMs"
	| "contextWindowTokens"
	| "convertToLlm"
	| "cwd"
	| "executionProfile"
	| "hookSessionId"
	| "hooks"
	| "inferenceMode"
	| "maxIterations"
	| "maxRetries"
	| "maxTokens"
	| "maxTotalTokens"
	| "model"
	| "onPermissionRequest"
	| "onQuestionRequest"
	| "permissions"
	| "runBudget"
	| "streamOptions"
	| "systemPrompt"
	| "temperature"
	| "thinkingLevel"
	| "toolExecution"
	| "tools"
	| "truncation"
	| "turnTimeoutMs"
>;

export interface AgentLoopConfig extends AgentLoopOptions {
	backend: LLMBackend;
	signal?: AbortSignal;
	onContextCompacted?: (messages: Message[]) => void;
	refreshNextTurnConfig?: () =>
		| Partial<AgentLoopConfig>
		| Promise<Partial<AgentLoopConfig>>;
	outputGuard?: OutputGuard | null;
	getAcceptanceConfig?: () => AcceptanceConfig | undefined;
	interventionController?: HarnessInterventionController;
	durableBudgetState?: {
		providerCalls: number;
		toolCalls: number;
		tokens: number;
		startedAt?: number;
	};
	onBudgetConsumed?: (
		resource: "provider_call" | "tool_call" | "token",
		amount: number,
	) => void;
}
