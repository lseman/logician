// ── Message, Tool, Event, and Hook types ──────────────────────────────────
// The barrel (index.ts) re-exports everything so external import sites are
// unaffected by this file's internal organization.

import type { RunOutcomeStatus } from "../policy/execution-policy.ts";

// ── Message types ─────────────────────────────────────────────────────────

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface Message {
	role: MessageRole;
	content: string | null;
	tool_call_id?: string;
	tool_calls?: ToolCall[];
	name?: string;
	timestamp?: number;
}

/** Loose message type compatible with both Message and AgentMessage. Used by compaction. */
export type CompactableMessage = {
	role: string;
	content?: unknown[] | string | null;
	usage?: Record<string, number>;
	/** UUID for tree-based entry tracking (Pi-compatible). */
	entryId?: string;
};

// ── AgentMessage Abstraction ─────────────────────────────────────────────
// Union of standard LLM messages + custom app messages (notifications,
// status updates, UI-only artifacts). Apps extend via declaration merging.

/** Standard LLM-compatible roles only. */
export type LlmRole = MessageRole;

// ── Custom message types ──────────────────────────────────────────────────

/** Compaction summary text — emitted after context compaction. */
export interface CompactionSummaryMessage {
	role: "compactionSummary";
	summary: string;
	tokensBefore: number;
	timestamp: number;
	/** Files read in the compacted history. */
	readFiles?: string[];
	/** Files modified in the compacted history. */
	modifiedFiles?: string[];
}

/** Branch summary text — emitted after branch recovery. */
export interface BranchSummaryMessage {
	role: "branchSummary";
	summary: string;
	fromId: string;
	timestamp: number;
}

/** Bash execution log — emitted after tool execution. */
export interface BashExecutionMessage {
	role: "bashExecution";
	command: string;
	output: string;
	exitCode: number | undefined;
	cancelled: boolean;
	truncated: boolean;
	fullOutputPath?: string;
	timestamp: number;
	excludeFromContext?: boolean;
}

/** Arbitrary custom message — emitted by tools and hooks. */
export interface CustomMessage {
	role: "custom";
	customType: string;
	content: string;
	display: boolean;
	details?: unknown;
	timestamp: number;
}

/** Custom agent message types — extend via declaration merging. */
export interface CustomAgentMessages {
	compactionSummary?: CompactionSummaryMessage;
	branchSummary?: BranchSummaryMessage;
	bashExecution?: BashExecutionMessage;
	custom?: CustomMessage;
}

/** Helper: map custom keys to message shapes. */
export type CustomAgentMessageMap = {
	[K in keyof CustomAgentMessages]: CustomAgentMessages[K] & {
		role: K extends string ? K : never;
	};
};

/** Union of standard Message + custom app messages. */
export type AgentMessage =
	| Message
	| CustomAgentMessageMap[keyof CustomAgentMessageMap & string];

/** Why the model (or loop) ended its turn. */
export type StopReason =
	| "stop"
	| "length"
	| "tool_calls"
	| "error"
	| "aborted"
	| "loop_detected";

// ── Tool types ────────────────────────────────────────────────────────────

export interface ToolCall {
	id: string;
	name: string;
	arguments: string;
}

/** Structured tool result. */
export interface ToolResult {
	content: string;
	details?: Record<string, unknown>;
	isError?: boolean;
	terminate?: boolean;
}

export type ToolExecutionMode = "sequential" | "parallel";

export interface Tool {
	name: string;
	/** Human-readable label shown in UI/tool lists. */
	label?: string;
	description: string;
	/** One-line description for the "Available tools" section in the system prompt. */
	promptSnippet?: string;
	/** Guideline bullets for the system prompt Guidelines section. */
	promptGuidelines?: string[];
	parameters: Record<string, unknown>;
	prepareArguments?: (args: unknown) => Record<string, unknown>;
	executionMode?: ToolExecutionMode;
	/** Opt-in result caching. Only pure, side-effect-free tools should set this. */
	cacheable?: boolean;
	/** Execution timeout in ms. Overrides the registry default; 0 disables. */
	timeoutMs?: number;
	/**
	 * Per-call timeout in ms derived from the call's arguments (e.g. bash's
	 * timeout parameter). Takes precedence over timeoutMs; return undefined to
	 * fall through.
	 */
	resolveTimeoutMs?: (args: Record<string, unknown>) => number | undefined;
	hookAliases?: string[];
	readOnly?: boolean;
	execute: (
		args: Record<string, unknown>,
		ctx: ToolContext,
	) => Promise<string | ToolResult>;
}

export interface AskUserContext {
	questions: Array<{
		id: string;
		header?: string;
		question: string;
		choices: Array<{ value: string; label: string; description?: string }>;
	}>;
}

export interface ToolContext {
	cwd?: string;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	maxOutputChars?: number;
	signal?: AbortSignal;
	onUpdate?: (partialResult: string) => void;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
}

// ── Event types ───────────────────────────────────────────────────────────

import type { HarnessIntervention } from "../policy/intervention-controller.ts";
import type { AgentConfig, AgentHarnessStreamOptions } from "./types-config.ts";

/**
 * Envelope metadata stamped onto every event at the emit boundary: a
 * monotonic per-loop sequence number and a wall-clock timestamp.
 */
export interface AgentEventEnvelope {
	seq?: number;
	ts?: number;
}

export type AgentEventBody =
	| { type: "agent_start" }
	| ({ type: "harness_intervention" } & HarnessIntervention)
	| {
			type: "agent_end";
			messages?: Message[];
			status?: RunOutcomeStatus;
			summary?: string;
	  }
	| { type: "agent_settled"; nextTurnCount?: number }
	| {
			type: "queue_update";
			steering: readonly string[];
			followUp: readonly string[];
			nextTurn: readonly string[];
	  }
	| { type: "turn_start"; turnId: string }
	| {
			type: "inference_mode_selected";
			configuredMode: "auto";
			effectiveMode: Exclude<AgentConfig["inferenceMode"], "auto">;
			reason: string;
	  }
	| {
			type: "turn_end";
			turnId: string;
			stopReason?: StopReason;
			message?: Message;
			toolResults?: Message[];
	  }
	| { type: "message_start"; turnId: string; role: MessageRole }
	| { type: "text_start"; turnId: string }
	| { type: "text_delta"; turnId: string; delta: string }
	| { type: "text_end"; turnId: string }
	| { type: "message_update"; turnId: string; message: Message }
	| { type: "message_end"; turnId: string; message?: Message }
	| {
			type: "context_update";
			tokens: number;
			maxTokens?: number;
			compacted?: boolean;
			/** null means the provider supplied no cache telemetry. */
			cachedTokens?: number | null;
			/** Tokens sent to the provider (prompt/input); null means unavailable. */
			promptTokens?: number | null;
			/** Tokens generated by the provider (completion/output); null means unavailable. */
			completionTokens?: number | null;
	  }
	| {
			type: "compaction";
			reason: "manual" | "context_full" | "auto" | "overflow" | "threshold";
			tokensBefore?: number;
			tokensAfter?: number;
	  }
	| { type: "thinking_delta"; turnId?: string; delta: string }
	// The backend's coherent mid-stream reasoning accumulation — a snapshot,
	// not a delta, fired alongside message_update. Reasoning is never part of
	// the persisted Message (providers don't echo it back in history), so it
	// travels as its own event instead of a Message field.
	| { type: "message_reasoning_update"; turnId: string; reasoning: string }
	| {
			type: "tool_call_start";
			toolName: string;
			toolCallId: string;
			args: string;
	  }
	| {
			type: "tool_call_delta";
			toolCallId: string;
			delta: string;
	  }
	| {
			type: "tool_call_id_update";
			previousToolCallId: string;
			toolCallId: string;
	  }
	| {
			type: "tool_call_end";
			toolName: string;
			toolCallId: string;
			result: string;
			isError?: boolean;
			details?: Record<string, unknown>;
	  }
	| {
			type: "tool_execution_start";
			toolCallId: string;
			toolName: string;
			args: Record<string, unknown>;
	  }
	| {
			type: "tool_execution_update";
			toolCallId: string;
			toolName: string;
			args: Record<string, unknown>;
			partialResult: string;
	  }
	| {
			type: "tool_execution_end";
			toolCallId: string;
			toolName: string;
			result: string;
			isError: boolean;
	  }
	| {
			type: "repair_nudge";
			turnId?: string;
			repairStage: string;
			toolName?: string;
			message: string;
	  }
	| { type: "phase"; phase: "thinking" | "tool" | "idle" }
	| { type: "model_select"; model: string; index: number }
	| { type: "max_iterations"; iterations: number; limit: number }
	| {
			type: "task_failed";
			reason: string;
			iteration: number;
			lastContent?: string;
	  }
	| {
			type: "loop_detected";
			message: string;
			attempt?: number;
	  }
	| {
			type: "subagent_start";
			agentId: string;
			agent: string;
			task: string;
			/** Position within a spawn_agents batch, if run as part of one. */
			taskIndex?: number;
	  }
	| {
			type: "subagent_event";
			agentId: string;
			event: AgentEvent;
			/** Position within a spawn_agents batch, if run as part of one. */
			taskIndex?: number;
	  }
	| {
			type: "subagent_end";
			agentId: string;
			agent: string;
			result: string;
			isError?: boolean;
			turns?: number;
			/** Position within a spawn_agents batch, if run as part of one. */
			taskIndex?: number;
	  }
	| {
			type: "tool_permission_request";
			toolName: string;
			toolCallId: string;
			args: string;
	  }
	| {
			type: "tool_permission_decision";
			toolName: string;
			toolCallId: string;
			decision: "allow" | "deny" | "always";
			source: "rule" | "mode" | "user" | "hook" | "fail_closed";
	  }
	| { type: "budget_exhausted"; usedTokens: number; limitTokens: number }
	| { type: "error"; message: string; error?: unknown }
	| {
			type: "model_cycle";
			model: string;
			fromModel: string;
			thinkingLevel?: string;
	  }
	| { type: "thinking_level_changed"; level: string }
	| { type: "thinking_level_clamped"; level: string; reason: string }
	| { type: "tools_update"; toolNames: string[] }
	| {
			type: "abort";
			clearedSteering: readonly string[];
			clearedFollowUp: readonly string[];
			clearedNextTurn: readonly string[];
	  }
	| {
			type: "thinking_loop_detected";
			message: string;
			strategy:
				| "thinking_only"
				| "escalation"
				| "meta_reasoning"
				| "budget_exhausted";
			iteration: number;
	  }
	| {
			type: "guard_triggered";
			guard:
				| "duplicate"
				| "failure"
				| "continuation_nudge"
				| "acceptance_retry"
				| "reflection_retry"
				| "loop_detected"
				| "budget_stop"
				| "follow_up"
				| "policy_continue"
				| "continuation_exhausted";
			message: string;
			toolName?: string;
			iteration: number;
	  }
	| {
			type: "thinking_loop_stats";
			consecutiveThinkingOnly: number;
			totalThinkingTurns: number;
			totalThinkingTokens: number;
			metaReasoningHits: number;
	  }
	| {
			type: "acceptance_start";
			level: string;
			criteriaCount: number;
	  }
	| {
			type: "acceptance_check";
			criterionId: string;
			status: "satisfied" | "failed" | "partial";
			severity: string;
	  }
	| {
			type: "acceptance_verify";
			command: string;
			result: "passed" | "failed" | "skipped";
			summary?: string;
	  }
	| {
			type: "acceptance_complete";
			status: "passed" | "failed" | "timeout";
			report?: Record<string, unknown>;
	  }
	| {
			type: "reflection_start";
			turnId: string;
	  }
	| {
			type: "reflection_end";
			turnId: string;
			assessment: "complete" | "incomplete";
			needsMoreWork: boolean;
			issues: string[];
	  }
	// Session tree entries
	| {
			type: "model_change";
			provider: string;
			modelId: string;
	  }
	| {
			type: "active_tools_change";
			activeToolNames: string[];
	  }
	// Retry / error observability
	| {
			type: "agent_retry_start";
			attempt: number;
			maxRetries: number;
			delayMs?: number;
			error: string;
			reason?: "compaction" | "error" | "overflow" | "rate_limit";
	  }
	| {
			type: "agent_retry_end";
			attempt: number;
			success: boolean;
			reason?: "compaction" | "error" | "overflow" | "rate_limit";
	  }
	| {
			type: "agent_error";
			message: string;
			phase: "model" | "tool" | "compaction" | "network" | "other";
			recoverable: boolean;
	  }
	// Session lifecycle
	| {
			type: "session_delete";
			sessionFile: string;
			sessionId: string;
	  };

export type AgentEvent = AgentEventBody & AgentEventEnvelope;
export type EventHandler = (event: AgentEvent) => void;
export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

// ── Hook context/result types ─────────────────────────────────────────────

export interface BeforeToolCallContext {
	toolCall: ToolCall;
	args: Record<string, unknown>;
	iteration: number;
}

export interface BeforeToolCallResult {
	content?: string;
	isError?: boolean;
	args?: Record<string, unknown>;
}

export interface AfterToolCallContext {
	toolCall: ToolCall;
	args: Record<string, unknown>;
	result: string;
	isError: boolean;
	iteration: number;
}

export interface AfterToolCallResult {
	content?: string;
	isError?: boolean;
	terminate?: boolean;
}

export interface PrepareNextTurnContext {
	messages: Message[];
	iteration: number;
	hadToolCalls: boolean;
}

export interface PrepareNextTurnResult {
	messages: Message[];
}

export interface ShouldStopAfterTurnContext {
	messages: Message[];
	iteration: number;
	hadToolCalls: boolean;
}

export interface GetSteeringMessagesContext {
	messages: Message[];
	iteration: number;
}

export interface TransformContext {
	messages: AgentMessage[];
	iteration: number;
	signal?: AbortSignal;
}

export interface BeforeProviderRequestContext {
	model: string;
	sessionId: string;
	iteration: number;
	streamOptions: AgentHarnessStreamOptions;
}

export interface BeforeProviderRequestResult {
	/** Header patch. undefined values delete keys; explicit headers: undefined clears all. */
	headers?: Record<string, string | undefined>;
	/** Timeout override in milliseconds. */
	timeoutMs?: number;
	/** Max retry attempts override. */
	maxRetries?: number;
	/** Cache retention hint (e.g., "transient", "persistent"). */
	cacheRetention?: string;
	/** Additional request headers merged with auth and lifecycle headers. */
	metadata?: Record<string, unknown>;
	/** Provider metadata forwarded with requests. */
	transport?: string;
}

export interface BeforeProviderPayloadContext {
	model: string;
	payload: Record<string, unknown>;
}

export interface BeforeProviderPayloadResult {
	payload: Record<string, unknown>;
}

export interface AfterProviderResponseContext {
	model: string;
	content: string;
	toolCallCount: number;
	stopReason: StopReason;
	usageTokens?: number;
	iteration: number;
}

export interface TransformContextResult {
	messages: AgentMessage[];
}

export interface GetFollowUpMessagesContext {
	messages: Message[];
	iteration: number;
	assistantText: string;
	stopReason?: StopReason;
}

export interface BeforeCompactContext {
	/** Messages that will be summarized. */
	messages: Message[];
	/** Estimated token count before compaction. */
	tokensBefore: number;
	/** "manual" = explicit compact() call; "auto" = threshold-triggered. */
	reason: "manual" | "auto";
}

export interface BeforeCompactResult {
	/** Return true to skip compaction entirely. */
	cancel?: boolean;
	/** Provide a pre-built summary to use instead of generating one. */
	summary?: string;
}

export interface BeforeAgentStartContext {
	prompt: string;
	systemPrompt: string;
	messages: AgentMessage[];
}

export interface BeforeAgentStartResult {
	messages?: AgentMessage[];
	systemPrompt?: string;
}

export interface AgentHooks {
	beforeAgentStart?: (
		ctx: BeforeAgentStartContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeAgentStartResult | undefined>
		| BeforeAgentStartResult
		| undefined;
	beforeToolCall?: (
		ctx: BeforeToolCallContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeToolCallResult | undefined>
		| BeforeToolCallResult
		| undefined;
	afterToolCall?: (
		ctx: AfterToolCallContext,
		signal?: AbortSignal,
	) =>
		| Promise<AfterToolCallResult | undefined>
		| AfterToolCallResult
		| undefined;
	prepareNextTurn?: (
		ctx: PrepareNextTurnContext,
		signal?: AbortSignal,
	) =>
		| Promise<PrepareNextTurnResult | undefined>
		| PrepareNextTurnResult
		| undefined;
	transformContext?: (
		ctx: TransformContext,
		signal?: AbortSignal,
	) =>
		| Promise<TransformContextResult | undefined>
		| TransformContextResult
		| undefined;
	beforeProviderRequest?: (
		ctx: BeforeProviderRequestContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeProviderRequestResult | undefined>
		| BeforeProviderRequestResult
		| undefined;
	beforeProviderPayload?: (
		ctx: BeforeProviderPayloadContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeProviderPayloadResult | undefined>
		| BeforeProviderPayloadResult
		| undefined;
	afterProviderResponse?: (
		ctx: AfterProviderResponseContext,
		signal?: AbortSignal,
	) => Promise<void> | void;
	shouldStopAfterTurn?: (
		ctx: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	) => Promise<boolean | undefined> | boolean | undefined;
	getSteeringMessages?: (
		ctx: GetSteeringMessagesContext,
		signal?: AbortSignal,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	getFollowUpMessages?: (
		ctx: GetFollowUpMessagesContext,
		signal?: AbortSignal,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	beforeCompact?: (
		ctx: BeforeCompactContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeCompactResult | undefined>
		| BeforeCompactResult
		| undefined;
}
