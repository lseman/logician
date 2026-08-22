import type { AcceptanceConfig } from "../../system/types/acceptance.ts";
import type { OutputGuard } from "../../control/guards/output-guard.ts";
import type { HarnessInterventionController } from "../../control/policy/intervention-controller.ts";
import type { AgentRunController } from "../../control/policy/run-controller.ts";
import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import type { AgentConfig } from "../../system/types/types-config.ts";
import type { Message } from "../../system/types/types-messages.ts";

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
	| "taskLedger"
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
	runController?: AgentRunController;
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
