// ── Core types (barrel) ──────────────────────────────────────────────────
// Sub-modules:
//   types-messages.ts  : Message, MessageRole, AgentMessage, StopReason
//   types-events.ts    : AgentEvent, AgentEventBody, EventHandler
//   types-hooks.ts     : All hook context/result types, AgentHooks
//   types-tools.ts     : Tool, ToolCall, ToolResult, ToolContext
//   types-config.ts    : AgentConfig, WebSearchConfig, QueueMode, ThinkingLevel
//   types-errors.ts    : AgentErrorType, AgentError, wrapError

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
	HarnessInterventionAction,
	HarnessInterventionEvidence,
	HarnessInterventionKind,
} from "./intervention-controller.ts";
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
} from "./types/types-config.ts";
export {
	AgentError,
	type AgentErrorOptions,
	AgentErrorType,
	BranchSummaryError,
	type BranchSummaryErrorCode,
	CompactionError,
	type CompactionErrorCode,
	ExecutionError,
	type ExecutionErrorCode,
	err,
	FileError,
	type FileErrorCode,
	getOrThrow,
	getOrUndefined,
	ok,
	type Result,
	SessionError,
	type SessionErrorCode,
	toError,
	wrapError,
} from "./types/types-errors.ts";
export type {
	AgentEvent,
	AgentEventBody,
	AgentEventEnvelope,
	EventHandler,
} from "./types/types-events.ts";
export type {
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
	PreToolUseContext,
	PreToolUseResult,
	ShouldStopAfterTurnContext,
	TransformContext,
	TransformContextResult,
} from "./types/types-hooks.ts";
export type {
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
} from "./types/types-messages.ts";
export type {
	AskUserContext,
	Tool,
	ToolCall,
	ToolContext,
	ToolExecutionMode,
	ToolResult,
} from "./types/types-tools.ts";
export type { TruncationConfig } from "./types/types-truncation.ts";
export {
	DEFAULT_TRUNCATION,
	resolveTruncationConfig,
} from "./types/types-truncation.ts";
