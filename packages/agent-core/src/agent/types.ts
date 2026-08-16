// ── Core types (barrel) ──────────────────────────────────────────────────
// Re-exports from consolidated sub-modules:
//   types-config.ts  : Config, Error, Truncation types
//   types-messages.ts: Message, Tool, Event, Hook types
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
	AgentError,
	AgentErrorType,
	wrapError,
} from "./types/types-config.ts";
// ── Message, Tool, Event, Hook ────────────────────────────────────────────
export type {
	AgentEvent,
	AgentEventBody,
	AgentEventEnvelope,
	EventHandler,
	AgentEventSink,
	AfterProviderResponseContext,
	AfterToolCallContext,
	AfterToolCallResult,
	AgentHooks,
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
	AgentMessage,
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
	AskUserContext,
	Tool,
	ToolCall,
	ToolContext,
	ToolExecutionMode,
	ToolResult,
} from "./types/types-messages.ts";
export type { TruncationConfig } from "./types/types-truncation.ts";
export {
	DEFAULT_TRUNCATION,
	resolveTruncationConfig,
} from "./types/types-truncation.ts";
