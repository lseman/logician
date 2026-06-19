// ── Core types (barrel) ──────────────────────────────────────────────────
// Re-exports from focused sub-modules for backward compatibility.
//
// Sub-modules:
//   types-messages.ts  : Message, MessageRole, AgentMessage, StopReason
//   types-events.ts    : AgentEvent, AgentEventBody, EventHandler
//   types-hooks.ts     : All hook context/result types, AgentLoopHooks
//   types-tools.ts     : Tool, ToolCall, ToolResult, ToolContext
//   types-config.ts    : AgentConfig, WebSearchConfig, QueueMode, ThinkingLevel
//   types-errors.ts    : AgentErrorType, AgentError, wrapError

// ── Messages ──────────────────────────────────────────────────────────────

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
} from "./types-messages.ts";
export { type CompactableMessage } from "./types-messages.ts";

// ── Events ────────────────────────────────────────────────────────────────

export {
	type AgentEventEnvelope,
	type AgentEventBody,
	type AgentEvent,
	type EventHandler,
} from "./types-events.ts";

// ── Hooks ─────────────────────────────────────────────────────────────────

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
	AgentLoopHooks,
} from "./types-hooks.ts";

// ── Tools ─────────────────────────────────────────────────────────────────

export {
	type ToolCall,
	type ToolResult,
	type ToolExecutionMode,
	type Tool,
	type AskUserContext,
	type ToolContext,
} from "./types-tools.ts";

// ── Config ────────────────────────────────────────────────────────────────

export {
	type QueueMode,
	type ThinkingLevel,
	type AgentConfig,
	type AgentHarnessStreamOptions,
	type WebSearchConfig,
	type EvidenceKind,
	type AcceptanceCriterion,
	type AcceptanceVerification,
	type AcceptanceReview,
	type AcceptanceConfig,
} from "./types-config.ts";

// ── Errors ────────────────────────────────────────────────────────────────

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
} from "./types-errors.ts";
