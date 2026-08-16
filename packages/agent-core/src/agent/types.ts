// ── Core types (barrel) ──────────────────────────────────────────────────
// Re-exports from consolidated sub-modules:
//   types-config.ts  : Config, Error, Truncation types
//   types-messages.ts: Message, Tool, Event types
//   types-hooks.ts   : Hook context/result types (extracted from types-messages)
//   types-truncation.ts: Truncation defaults (kept separate — used by non-type code)

export type {
	AgentStopPolicy,
	ExecutionProfile,
	ResolvedExecutionPolicy,
	RunOutcomeStatus,
	StopPolicyContext,
	StopPolicyDecision,
} from "./execution-policy.ts";
export type { AcceptanceConfig } from "./guards/acceptance-contract.ts";
export type {
	HarnessIntervention,
	HarnessInterventionKind,
} from "./intervention-controller.ts";
// ── Config, Error, Truncation ─────────────────────────────────────────────
export type {
	AcceptanceCriterion,
	AcceptanceReview,
	AcceptanceVerification,
	AgentConfig,
	AgentHarnessStreamOptions,
	AgentModelConfig,
	EvidenceKind,
	InferenceMode,
	QueueMode,
	ThinkingLevel,
	WebSearchConfig,
	AgentErrorOptions,
} from "./types/types-config.ts";
export {
	cycleInferenceMode,
	DEFAULT_INFERENCE_MODE,
	getInferenceMode,
	INFERENCE_MODE_ORDER,
	INFERENCE_MODES,
	isValidInferenceMode,
	QUEUE_MODES,
	THINKING_LEVELS,
	VALID_TOOL_EXECUTION,
	AgentError,
	AgentErrorType,
	wrapError,
} from "./types/types-config.ts";
// ── Message, Tool, Event ──────────────────────────────────────────────────
export type {
	AgentEvent,
	AgentEventBody,
	AgentEventEnvelope,
	EventHandler,
	AgentEventSink,
	AgentMessage,
	AskUserContext,
	BashExecutionMessage,
	BranchSummaryMessage,
	CompactableMessage,
	CompactionSummaryMessage,
	CustomAgentMessageMap,
	CustomAgentMessages,
	CustomMessage,
	LlmRole,
	Message,
	MessageRole,
	StopReason,
	Tool,
	ToolCall,
	ToolContext,
	ToolExecutionMode,
	ToolResult,
} from "./types/types-messages.ts";
// ── Hook context/result types (extracted to types-hooks.ts) ───────────────
export type {
	AgentHooks,
	AfterProviderResponseContext,
	AfterToolCallContext,
	AfterToolCallResult,
	BeforeAgentStartContext,
	BeforeAgentStartResult,
	BeforeCompactContext,
	BeforeCompactResult,
	BeforeProviderPayloadContext,
	BeforeProviderPayloadResult,
	BeforeProviderRequestContext,
	BeforeProviderRequestResult,
	BeforeToolCallContext,
	BeforeToolCallResult,
	GetFollowUpMessagesContext,
	GetSteeringMessagesContext,
	PrepareNextTurnContext,
	PrepareNextTurnResult,
	ShouldStopAfterTurnContext,
	TransformContext,
	TransformContextResult,
} from "./types/types-hooks.ts";
export type { TruncationConfig } from "./types/types-truncation.ts";
export {
	DEFAULT_TRUNCATION,
	resolveTruncationConfig,
} from "./types/types-truncation.ts";
