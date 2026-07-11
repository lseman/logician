// ── Core types (barrel) ──────────────────────────────────────────────────
// Sub-modules:
//   types-messages.ts  : Message, MessageRole, AgentMessage, StopReason
//   types-events.ts    : AgentEvent, AgentEventBody, EventHandler
//   types-hooks.ts     : All hook context/result types, AgentHooks
//   types-tools.ts     : Tool, ToolCall, ToolResult, ToolContext
//   types-config.ts    : AgentConfig, WebSearchConfig, QueueMode, ThinkingLevel
//   types-errors.ts    : AgentErrorType, AgentError, wrapError

export type {
	MessageRole,
	Message,
	AgentMessage,
	LlmRole,
	CustomAgentMessages,
	CustomAgentMessageMap,
	StopReason,
	CompactionSummaryMessage,
	BranchSummaryMessage,
	BashExecutionMessage,
	CustomMessage,
} from "./types/types-messages.ts";
export type { CompactableMessage } from "./types/types-messages.ts";

export type {
	AgentEventEnvelope,
	AgentEventBody,
	AgentEvent,
	EventHandler,
} from "./types/types-events.ts";

export type {
	BeforeToolCallContext,
	BeforeToolCallResult,
	PreToolUseContext,
	PreToolUseResult,
	AfterToolCallContext,
	AfterToolCallResult,
	PrepareNextTurnContext,
	PrepareNextTurnResult,
	ShouldStopAfterTurnContext,
	GetSteeringMessagesContext,
	TransformContext,
	BeforeProviderRequestContext,
	BeforeProviderRequestResult,
	BeforeProviderPayloadContext,
	BeforeProviderPayloadResult,
	AfterProviderResponseContext,
	TransformContextResult,
	GetFollowUpMessagesContext,
	BeforeCompactContext,
	BeforeCompactResult,
	BeforeAgentStartContext,
	BeforeAgentStartResult,
	AgentHooks,
} from "./types/types-hooks.ts";

export type {
	ToolCall,
	ToolResult,
	ToolExecutionMode,
	Tool,
	AskUserContext,
	ToolContext,
} from "./types/types-tools.ts";

export type {
	QueueMode,
	ThinkingLevel,
	AgentConfig,
	AgentModelConfig,
	AgentHarnessStreamOptions,
	WebSearchConfig,
	EvidenceKind,
	AcceptanceCriterion,
	AcceptanceVerification,
	AcceptanceReview,
} from "./types/types-config.ts";
export type { AcceptanceConfig } from "./acceptance-contract.ts";

export {
	AgentErrorType,
	type AgentErrorOptions,
	AgentError,
	wrapError,
	type Result,
	ok,
	err,
	getOrThrow,
	getOrUndefined,
	toError,
	type FileErrorCode,
	FileError,
	type ExecutionErrorCode,
	ExecutionError,
	type CompactionErrorCode,
	CompactionError,
	type BranchSummaryErrorCode,
	BranchSummaryError,
	type SessionErrorCode,
	SessionError,
} from "./types/types-errors.ts";
