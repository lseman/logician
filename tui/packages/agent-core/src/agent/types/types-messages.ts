// ── Message types ─────────────────────────────────────────────────────────

export type MessageRole = "system" | "user" | "assistant" | "tool";

interface ToolCall {
	id: string;
	name: string;
	arguments: string;
}

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
