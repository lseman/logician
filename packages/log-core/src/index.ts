/** Public contracts required to embed and extend the agent loop. */

// ── Context compaction ──────────────────────────────────────────────────────
export {
	type CompactionMode,
	type CompactionSettings,
	type CompactionSummarizer,
	type CompactToFitResult,
	compactToFit,
	microCompactCompactableMessages,
	pruneHistoricalToolOutputs,
	shakeCompaction,
} from "./compaction/engine.ts";
export { resolveAgentSettings } from "./config/agent-settings.ts";
export {
	AdaptiveContextController,
	type AdaptiveContextControllerOptions,
	type AdaptiveContextLearningState,
	type AdaptiveContextPlan,
	type AdaptiveContextRequest,
	type ContextOutcome,
} from "./context/adaptive-context-controller.ts";
export {
	EventJournal,
	type EventJournalEntry,
	type EventJournalOptions,
	type EventJournalQuery,
	type EventJournalSubscriptionOptions,
	type JournalEvent,
} from "./events/event-journal.ts";
export type { HarnessPhase } from "./events/runtime-state.ts";
export { loadExtensions } from "./extensions/loader.ts";
export { ExtensionRunner } from "./extensions/runner.ts";
// Public contracts for native Logician extensions.
export type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionEvent,
	ExtensionEventContext,
	ExtensionEventHandler,
	ExtensionEventType,
	RegisteredCommand,
	RegisteredTool,
} from "./extensions/types.ts";
export type { AcceptanceLedger } from "./guards/acceptance-contract.ts";
export {
	MAX_ESCALATIONS,
	SoftToolRequirementExceededError,
	SoftToolRequirementManager,
} from "./guards/soft-tool-requirement.ts";
export {
	TextLoopDetector,
	type TextLoopDetectorOptions,
} from "./guards/text-loop-detector.ts";
export type { AbortResult } from "./harness/types.ts";
export {
	type CancellationCleanup,
	CancellationError,
	type CancellationKind,
	CancellationScope,
	type CancellationScopeOptions,
} from "./lifecycle/cancellation-scope.ts";
export {
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./loop/agent-harness.ts";
export type {
	NamedAgentStopPolicy,
	StopPolicy,
	StopPolicyEvaluation,
	StopPolicyKind,
} from "./policy/execution-policy.ts";
export { createVerifiedStopPolicy } from "./policy/verified-stop-policy.ts";
export type {
	BackendErrorCategory,
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "./provider/backend.ts";
export {
	BackendError,
	classifyHttpError,
	classifyNetworkError,
	createLLMBackend,
	normalizeProviderMessages,
	OpenAIBackend,
	parseProviderUsage,
} from "./provider/backend.ts";
// ── Host embedding surface (formerly @logician/log-core/runtime) ──────────────
export {
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
	estimateChatPayloadTokens,
	estimateChatPayloadTokensHeuristic,
	estimateTokens,
	estimateTokensHeuristic,
} from "./provider/messages.ts";
export {
	OpenAIChatCompletionsAdapter,
	type ProviderAdapter,
	type ProviderRequestContext,
} from "./provider/provider-adapter.ts";
export {
	parseTextToolCalls,
	stripTextToolCalls,
} from "./provider/text-tool-calls.ts";
export {
	type CustomSessionEntry,
	type SessionEntry,
	SessionRegistry,
	SessionStore,
} from "./session/session-store.ts";
export {
	type BashDebugEntry,
	clearBashDebugger,
	getBashDebuggerReport,
	isBashDebuggerEnabled,
	setBashDebugger,
} from "./tools/bash-debugger.ts";
export { ToolRegistry } from "./tools/registry.ts";
export {
	BUILTIN_RULES,
	getBuiltInRuleByName,
	isBuiltInRule,
} from "./ttsr/built-in-rules.ts";
export {
	buildJudgeRequest,
	JUDGED_CONTENT_MAX_CHARS,
	judgeRules,
	parseJudgeVerdicts,
	type TtsrJudge,
} from "./ttsr/judge.ts";
export { RuleLoader } from "./ttsr/rule-loader.ts";
export { TtsrManager } from "./ttsr/ttsr-manager.ts";
export type { AcceptanceConfig } from "./types/acceptance.ts";
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
	VALID_TOOL_EXECUTION,
	type WebSearchConfig,
} from "./types/config.ts";
export type {
	AgentEvent,
	AgentHooks,
	AskUserContext,
	CompactableMessage,
	GetToolChoiceContext,
	Message,
	MutationReceipt,
	SoftToolRequirement,
	SoftToolRequirementState,
	Tool,
	ToolCall,
	ToolChoiceDirective,
	ToolContext,
	ToolResult,
} from "./types/messages.ts";
// TTSR (Time-Traveling Stream Rules) types
export type {
	JudgedCandidate,
	TtsrBridgeSettings,
	TtsrInjectionEntry,
	TtsrMatchContext,
	TtsrMatchSource,
	TtsrOutput,
	TtsrRule,
	TtsrScope,
	TtsrSettings,
} from "./types/ttsr.ts";
