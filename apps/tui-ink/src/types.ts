// ── Ink TUI Shared Types ──────────────────────────────────────────────────────

// Re-export runtime types
export type { Turn, Message, AssistantMessage, AssistantChunk, UserMessage, SystemMessage, ToolExecution } from "@logician/log-runtime/sessions";
export type { AgentRuntime } from "@logician/log-runtime/application";
export type { TuiSessionService } from "@logician/log-runtime/sessions";
import type { TuiSessionSummary } from "@logician/log-runtime/sessions";

export interface LegacyTurn {
	id: string;
	userMessage?: { content: string };
	assistantMessage?: { content: string; toolCalls?: ToolCall[] };
	systemMessage?: { content: string };
	isComplete: boolean;
	createdAt: number;
	thinking?: string;
}

export interface ToolCall {
	id: string;
	name: string;
	arguments?: string;
	result?: string;
	isError?: boolean;
}

export type NotificationLevel = "info" | "warn" | "error";

export interface Notification {
	id: string;
	message: string;
	level: NotificationLevel;
	createdAt: number;
}

export type TuiPhase =
	| "ready"
	| "thinking"
	| "working"
	| "cancelling"
	| "error"
	| "idle";

export type WorkflowMode = "act" | "plan";

export type InferenceMode = "none" | "deep" | "research" | "creative" | "debug";

export interface GitStatus {
	branch?: string;
	staged: number;
	modified: number;
	untracked: number;
	commit?: string;
	ahead?: number;
	behind?: number;
	addedLines?: number;
	removedLines?: number;
}

export interface ReasonerStatus {
	name: string;
	active: boolean;
}

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export type ThinkingDisplayMode = "collapsed" | "summary" | "expanded";

export interface TodoItem {
	id: string;
	text: string;
	done: boolean;
}

export interface SteerMessage {
	id: string;
	message: string;
	createdAt: number;
}

export interface InferenceSettings {
	mode: InferenceMode;
	profile: "autonomous" | "minimal";
	planMode: boolean;
	thinkingLevel: ThinkingLevel;
}

// Alias for compatibility with legacy code
export type SessionInfo = TuiSessionSummary;

export interface ModelInfo {
	name: string;
	displayName: string;
	contextWindowTokens: number;
}

export interface AppConfig {
	bridge: {
		model?: string;
		contextWindowTokens?: number;
		permissions?: { mode?: string };
		thinkingLevel?: string;
		inferenceMode?: string;
		workflowMode?: string;
		executionProfile?: string;
		rtkProxyEnabled?: boolean;
		legroom?: { mode?: string };
		memoriam?: { mode?: string };
		graphicianEnabled?: boolean;
		fffgrepEnabled?: boolean;
		cwd?: string;
		extraTools?: unknown[];
	};
	source: {
		theme?: string;
		workflowMode?: string;
		transcriptMaxTurns?: number;
		transcriptMaxRenderedLines?: number;
		truncation?: { transcriptMessageMaxChars?: number };
		inferenceMode?: string;
	};
	configPath?: string;
}

export type OverlayKind =
	| "slash"
	| "fileMention"
	| "choice"
	| "permission"
	| "pluginManager"
	| "mcpManager"
	| "sessionManager"
	| "sessionTree"
	| "modelSelector"
	| "themeSelector"
	| "settingsSelector"
	| "thinkingLevelSelector"
	| "inferenceModeSelector"
	| "reasonerSelector"
	| "queueManager"
	| "autoresearchDashboard";

export interface OverlayState {
	kind: OverlayKind | null;
	data?: Record<string, unknown>;
}
