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
	OpenAIChatCompletionsAdapter,
	type ProviderAdapter,
	type ProviderRequestContext,
} from "./capabilities/provider/provider-adapter.ts";
export {
	parseTextToolCalls,
	stripTextToolCalls,
} from "./capabilities/provider/text-tool-calls.ts";
export type { AcceptanceLedger } from "./control/guards/acceptance-contract.ts";
export { MAX_ESCALATIONS, SoftToolRequirementExceededError, SoftToolRequirementManager } from "./control/guards/soft-tool-requirement.ts";
export { StablePrefix } from "./control/guards/stable-prefix.ts";
export { TextLoopDetector, type TextLoopDetectorOptions } from "./control/guards/text-loop-detector.ts";
export type {
	NamedAgentStopPolicy,
	StopPolicy,
	StopPolicyEvaluation,
	StopPolicyKind,
} from "./control/policy/execution-policy.ts";
export { createVerifiedStopPolicy } from "./control/policy/verified-stop-policy.ts";
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
	type AdaptiveContextLearningState,
	type AdaptiveContextPlan,
	type AdaptiveContextRequest,
	type ContextOutcome,
} from "./system/context/adaptive-context-controller.ts";
export {
	type CancellationCleanup,
	CancellationError,
	type CancellationKind,
	CancellationScope,
	type CancellationScopeOptions,
} from "./system/lifecycle/cancellation-scope.ts";
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
	THINKING_FORMATS,
	THINKING_LEVELS,
	type ThinkingFormat,
	type ThinkingLevel,
	type TruncationConfig,
	type WebSearchConfig,
} from "./system/types/types-config.ts";
export type {
	AgentEvent,
	AgentHooks,
	GetToolChoiceContext,
	SoftToolRequirement,
	SoftToolRequirementState,
	ToolChoiceDirective,
	AskUserContext,
	CompactableMessage,
	Message,
	MutationReceipt,
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "./system/types/types-messages.ts";
// TTSR (Time-Traveling Stream Rules) types
export type {
	TtsrRule,
	TtsrScope,
	TtsrSettings,
	TtsrMatchContext,
	TtsrMatchSource,
	TtsrInjectionEntry,
	TtsrBridgeSettings,
} from "./system/types/ttsr-types.ts";
export { TtsrManager } from "./system/ttsr/ttsr-manager.ts";

