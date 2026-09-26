export type ThinkingDisplayStyle = "collapsed" | "summary" | "expanded";

export interface ToolExecution {
	tool_name: string;
	tool_call_id?: string;
	args?: Record<string, unknown>;
	result?: string;
	partialResult?: string;
	/** Human-readable progress emitted while the tool executes. */
	streamOutput?: string;
	details?: Record<string, unknown>;
	isError: boolean;
	isComplete: boolean;
	startedAt?: number;
	durationMs?: number;
}

export type NoticeLevel = "info" | "warn" | "error" | "success";

export interface ChildToolCall {
	agentId: string;
	toolCallId: string;
	toolName: string;
	args: string;
	status?: "running" | "completed" | "failed";
	isError?: boolean;
	resultPreview?: string;
	/** Position within a spawn_agents batch, if run as part of one. */
	taskIndex?: number;
}

export interface ChildChunk {
	seq: number;
	agentId: string;
	type: "thinking" | "content" | "tool";
	contentText?: string;
	tool?: ChildToolCall;
	isComplete: boolean;
	/** Position within a spawn_agents batch, if run as part of one. */
	taskIndex?: number;
}

/** Structured per-task status for a spawn_agents batch. */
export interface SpawnTaskStatus {
	taskIndex: number;
	agentId: string;
	agent: string;
	status: "running" | "completed" | "failed";
	startedAt: number;
	endedAt?: number;
}

export interface AssistantChunk {
	seq: number;
	type: "thinking" | "content" | "tool" | "notice" | "user";
	contentText?: string;
	tool?: ToolExecution;
	notice?: { level: NoticeLevel; label: string; text: string };
	isComplete: boolean;
}

export interface AssistantMessage {
	type: "assistant";
	chunks: AssistantChunk[];
	isComplete: boolean;
}

export interface UserMessage {
	type: "user";
	content: string;
}

export interface SystemMessage {
	type: "system";
	content: string;
}

export type Message = UserMessage | AssistantMessage | SystemMessage;

export interface Turn {
	id: string;
	userMessage: UserMessage | null;
	assistantMessage: AssistantMessage | null;
	isComplete: boolean;
	/** Monotonically increasing revision used by the TUI's render cache. */
	contentRevision?: number;
}

export interface SessionState {
	turns: Turn[];
	currentTurnId: string | null;
	thinkingDisplayMode: ThinkingDisplayStyle;
	thinkingLevel: string;
}

export function createInitialTranscriptState(): SessionState {
	return {
		turns: [],
		currentTurnId: null,
		thinkingDisplayMode: "expanded",
		thinkingLevel: "off",
	};
}
