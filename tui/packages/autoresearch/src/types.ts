/**
 * Autoresearch types — self-contained type definitions.
 *
 * These mirror the logician agent-core extension types but are defined locally
 * to avoid transitive dependency issues.
 */

// ============================================================================
// Tool Registration
// ============================================================================

export interface ToolParameterSchema {
	type: string;
	description?: string;
	required?: string[];
	enum?: string[];
	properties?: Record<string, ToolParameterSchema>;
	items?: ToolParameterSchema;
	additionalProperties?: boolean | ToolParameterSchema;
}

export interface RegisteredTool {
	name: string;
	label?: string;
	description: string;
	parameters: ToolParameterSchema;
	execute: (
		toolCallId: string,
		params: Record<string, unknown>,
		ctx: ToolExecutionContext,
	) => Promise<ExtensionToolResult>;
}

export interface ToolExecutionContext {
	toolCallId: string;
	cwd: string;
	sessionId: string;
}

export interface ExtensionToolResult {
	content: string;
	isError?: boolean;
	details?: Record<string, unknown>;
}

// ============================================================================
// Command Registration
// ============================================================================

export interface RegisteredCommand {
	name: string;
	description: string;
	usage?: string;
	acceptsArgs?: boolean;
	handler: (args: string | undefined, ctx: CommandContext) => Promise<string> | string;
}

export interface CommandContext {
	sessionId: string;
	cwd: string;
	ui: ExtensionUI;
}

// ============================================================================
// UI Primitives
// ============================================================================

export interface ExtensionUI {
	notify(message: string, type?: "info" | "warning" | "error"): void;
	confirm(title: string, message: string, opts?: { timeoutMs?: number }): Promise<boolean>;
	input(title: string, placeholder?: string, opts?: { timeoutMs?: number }): Promise<string | undefined>;
	select(title: string, options: Array<{ label: string; description?: string }>, opts?: { timeoutMs?: number }): Promise<string | undefined>;
}

// ============================================================================
// Event System
// ============================================================================

export type ExtensionEventType =
	| "agent_start"
	| "agent_end"
	| "session_start"
	| "session_end"
	| "session_shutdown"
	| "user_prompt_submit"
	| "message_start"
	| "message_update"
	| "message_end"
	| "before_tool_call"
	| "after_tool_call"
	| "tool_call_start"
	| "tool_call_end"
	| "before_compact"
	| "after_compact"
	| "queue_update"
	| "turn_start"
	| "turn_end";

export interface ExtensionEventContext {
	sessionId: string;
	cwd: string;
	[key: string]: unknown;
}

export interface ExtensionEvent {
	type: ExtensionEventType;
	context: ExtensionEventContext;
	[key: string]: unknown;
}

export type ExtensionEventHandler = (
	event: ExtensionEvent,
	ctx: ExtensionContext,
) => Promise<unknown> | unknown;

// ============================================================================
// Extension Context
// ============================================================================

export interface ExtensionContextState {
	counters: Record<string, number>;
	features: Set<string>;
	labels: Record<string, string>;
	data: Record<string, unknown>;
	diagnostics: Array<{ source: string; message: string; severity: "info" | "warning" | "error" }>;
}

export interface ExtensionContextActions {
	incrementCounter: (name: string) => number;
	setFeature: (name: string, value: boolean) => void;
	setLabel: (label: string) => void;
	storeData: (extension: string, key: string, value: unknown) => void;
	fetchData: (extension: string, key: string) => unknown;
	addDiagnostic: (source: string, message: string, severity?: "info" | "warning" | "error") => void;
}

export interface ExtensionContext extends ExtensionContextState, ExtensionContextActions {
	cwd: string;
	sessionId: string;
	ui: ExtensionUI;
	turnIndex: number;
	iteration: number;
	inToolLoop: boolean;
	inThinkingLoop: boolean;
}

// ============================================================================
// Extension API (entry point)
// ============================================================================

export interface EventBus {
	on(event: string, handler: (event: ExtensionEvent) => void): () => void;
	emit(event: ExtensionEvent): Promise<void>;
}

export interface ExtensionAPI {
	on(event: ExtensionEventType, handler: ExtensionEventHandler): () => void;
	registerTool(tool: RegisteredTool): void;
	registerCommand(command: RegisteredCommand): void;
	emit(event: ExtensionEvent): Promise<void>;
	events: EventBus;
	info: { name: string; path: string };
}
