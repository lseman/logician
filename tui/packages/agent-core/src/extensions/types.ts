// ── Extension system types ────────────────────────────────────────────────────
// TypeScript extension API for Logician. Extensions can:
// - Subscribe to agent lifecycle events
// - Register LLM-callable tools
// - Register slash commands
// - Interact with the user via UI primitives
// - Persist state across turns

import type { ToolCall } from "../agent/types.ts";
import type { EventBus } from "./event-bus.ts";

// ============================================================================
// Event System
// ============================================================================

export type ExtensionEventType =
	| "agent_start"
	| "agent_end"
	| "session_start"
	| "session_end"
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

export interface BeforeToolCallExtensionResult {
	block?: boolean;
	reason?: string;
	args?: Record<string, unknown>;
	content?: string;
	isError?: boolean;
}

export interface AfterToolCallExtensionResult {
	content?: string;
	isError?: boolean;
	terminate?: boolean;
}

// ============================================================================
// Tool Registration
// ============================================================================

export interface ToolParameterSchema {
	type: string;
	description?: string;
	required?: boolean;
	properties?: Record<string, ToolParameterSchema>;
	items?: ToolParameterSchema;
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
	toolCall: ToolCall;
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
	handler: (args: string, ctx: CommandContext) => Promise<string> | string;
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
	/** Show a notification to the user. */
	notify(message: string, type?: "info" | "warning" | "error"): void;

	/** Show a confirmation dialog. Returns user choice. */
	confirm(
		title: string,
		message: string,
		opts?: { timeoutMs?: number },
	): Promise<boolean>;

	/** Show a text input dialog. Returns user input or undefined. */
	input(
		title: string,
		placeholder?: string,
		opts?: { timeoutMs?: number },
	): Promise<string | undefined>;

	/** Show a selector (single choice). Returns selected option or undefined. */
	select(
		title: string,
		options: Array<{ label: string; description?: string }>,
		opts?: { timeoutMs?: number },
	): Promise<string | undefined>;
}

// ============================================================================
// State Management
// ============================================================================

export interface ExtensionState {
	/** Get a value from persistent state. */
	get<T>(key: string): Promise<T | undefined>;

	/** Set a value in persistent state. */
	set(key: string, value: unknown): Promise<void>;

	/** Delete a key from persistent state. */
	delete(key: string): Promise<void>;

	/** List all keys in persistent state. */
	keys(): Promise<string[]>;
}

// ============================================================================
// Extension Context
// ============================================================================

export interface ExtensionContext {
	/** UI primitives for user interaction. */
	ui: ExtensionUI;

	/** Persistent state for this extension. */
	state: ExtensionState;

	/** Working directory of the current session. */
	cwd: string;

	/** Current session ID. */
	sessionId: string;
}

// ============================================================================
// Extension API (entry point)
// ============================================================================

export interface ExtensionAPI {
	/** Subscribe to agent lifecycle events. */
	on(event: ExtensionEventType, handler: ExtensionEventHandler): () => void;

	/** Register a custom tool callable by the LLM. */
	registerTool(tool: RegisteredTool): void;

	/** Register a slash command. */
	registerCommand(command: RegisteredCommand): void;

	/** Emit an event to all subscribers (internal use). */
	emit(event: ExtensionEvent): Promise<void>;

	/** Shared event bus — all extensions receive the same instance for cross-extension messaging. */
	events: EventBus;

	/** Get the current extension's name and path. */
	info: { name: string; path: string };
}

// ============================================================================
// Extension Loader
// ============================================================================

export interface ExtensionDefinition {
	path: string;
	name: string;
	source: "user" | "project" | "path";
}

export interface LoadExtensionsResult {
	extensions: ExtensionDefinition[];
	diagnostics: Diagnostic[];
}

export interface Diagnostic {
	type: "warning" | "error";
	message: string;
	path: string;
}
