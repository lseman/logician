/** Public contracts required to embed and extend the agent loop. */

export type {
	BackendErrorCategory,
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "./capabilities/provider/backend.ts";
export {
	BackendError,
	classifyHttpError,
	classifyNetworkError,
	createLLMBackend,
	normalizeProviderMessages,
	OpenAIBackend,
	parseProviderUsage,
} from "./capabilities/provider/backend.ts";
export {
	parseTextToolCalls,
	stripTextToolCalls,
} from "./capabilities/provider/text-tool-calls.ts";
export type { AcceptanceLedger } from "./control/guards/acceptance-contract.ts";
export { stripAcceptanceReport } from "./control/guards/acceptance-contract.ts";
export {
	type CancellationCleanup,
	CancellationError,
	type CancellationKind,
	CancellationScope,
	type CancellationScopeOptions,
} from "./runtime/control/cancellation-scope.ts";
export {
	EventJournal,
	type EventJournalEntry,
	type EventJournalOptions,
	type EventJournalQuery,
	type EventJournalSubscriptionOptions,
	type JournalEvent,
} from "./runtime/events/event-journal.ts";
export {
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./runtime/harness/agent-harness.ts";
export {
	AdaptiveContextController,
	type AdaptiveContextControllerOptions,
	type AdaptiveContextPlan,
	type AdaptiveContextRequest,
	type ContextOutcome,
} from "./system/context/adaptive-context-controller.ts";
export type { AcceptanceConfig } from "./system/types/acceptance.ts";
export {
	type AgentConfig,
	type AgentModelConfig,
	cycleInferenceMode,
	DEFAULT_INFERENCE_MODE,
	DEFAULT_TRUNCATION,
	getInferenceMode,
	INFERENCE_MODE_ORDER,
	INFERENCE_MODES,
	type InferenceMode,
	isValidInferenceMode,
	type QueueMode,
	THINKING_LEVELS,
	type ThinkingLevel,
	type TruncationConfig,
	type WebSearchConfig,
} from "./system/types/types-config.ts";
export type {
	AgentEvent,
	AgentHooks,
	AskUserContext,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "./system/types/types-messages.ts";
