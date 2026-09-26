import type { OutputGuard } from "../guards/output-guard.ts";
import type { HarnessInterventionController } from "../policy/intervention-controller.ts";
import type { AgentRunController } from "../policy/run-controller.ts";
import type { LLMBackend } from "../provider/backend.ts";
import type { AcceptanceConfig } from "../types/acceptance.ts";
import type { AgentConfig } from "../types/config.ts";
import type { Message } from "../types/messages.ts";

/** Configuration understood by the agent execution mechanism. */
type AgentLoopOptions = Pick<
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
	| "models"
	| "onPermissionRequest"
	| "onQuestionRequest"
	| "permissions"
	| "runBudget"
	| "streamOptions"
	| "stopPolicies"
	| "systemPrompt"
	| "taskLedger"
	| "temperature"
	| "thinkingLevel"
	| "toolExecution"
	| "tools"
	| "truncation"
	| "turnTimeoutMs"
	| "verifiedStopEnabled"
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
		startedAt?: number | undefined;
	};
	onBudgetConsumed?: (
		resource: "provider_call" | "tool_call" | "token",
		amount: number,
	) => void;
}
