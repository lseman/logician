// ── Pi Extension Adapter ─────────────────────────────────────────────────────
// Wraps Logician's ExtensionAPI/ExtensionContext to present a Pi-compatible
// surface so that Pi extensions can run on the Logician runtime.
//
// ─── Event Mapping (detailed in translateLogicianToPi) ───────────────────────
//
// Mapped (via translateLogicianToPi) — field-level Pi compatibility:
//   Pi: session_start          ← Logician: session_start
//      reason: startup/reload/new/resume/fork, previousSessionFile?
//   Pi: session_before_compact ← Logician: before_compact
//      preparation, branchEntries, reason (manual/threshold/overflow),
//      willRetry, signal, customInstructions?, tokensBefore, messages[]
//   Pi: session_compact        ← Logician: after_compact
//      compactionEntry, fromExtension, reason, willRetry, tokensBefore/After,
//      messages[]
//   Pi: session_shutdown       ← Logician: session_end
//      reason: quit
//   Pi: before_agent_start     ← Logician: user_prompt_submit
//      prompt, images[], systemPrompt, systemPromptOptions
//      → handlers can return systemPrompt for override (fed back via emit)
//   Pi: agent_start            ← Logician: agent_start
//   Pi: agent_end              ← Logician: agent_end
//      messages[] — run messages from harness
//   Pi: turn_start             ← Logician: turn_start
//      turnIndex, timestamp
//   Pi: turn_end               ← Logician: turn_end
//      turnIndex, message, toolResults[]
//   Pi: message_start          ← Logician: message_start
//      message
//   Pi: message_update         ← Logician: message_update
//      message, assistantMessageEvent (null in Logician — no streaming)
//   Pi: message_end            ← Logician: message_end
//      message (handlers can return {message} to replace)
//   Pi: tool_execution_start   ← Logician: tool_call_start
//      toolCallId, toolName, args
//   Pi: tool_execution_end     ← Logician: tool_call_end
//      toolCallId, toolName, result, isError
//   Pi: tool_call              ← Logician: before_tool_call
//      toolCallId, toolName, input (mutable — mutations pass through)
//      → handlers can return {block, reason, terminate} to block execution
//   Pi: tool_result            ← Logician: after_tool_call
//      toolCallId, toolName, input, content[], details, isError, usage
//      → handlers can return {content, details, isError, usage} to modify
//
// Handled separately (dedicated emit methods):
//   Pi: project_trust          ← emitProjectTrustEvent()
//   Pi: user_bash              ← emitUserBashEvent()
//   Pi: input                  ← emitInputEvent()
//
// Emitted (new):
//   Pi: context              ← Logician: beforeAgentStart
//      messages[] — pre-LLM-call messages (same as before_agent_start
//                   but without prompt field; return {messages} to modify)
//   Pi: agent_error          ← Logician: agent_error
//      message, phase, recoverable
//   Pi: session_delete       ← Logician: session_delete
//      sessionFile, sessionId
//
// Not emitted (Logician doesn't produce these events):
//   session_info_changed, session_before_switch, session_before_fork,
//   session_before_tree, session_tree,
//   before_provider_request, before_provider_headers, after_provider_response,
//   agent_settled, tool_execution_update, model_select, thinking_level_changed
//
// No-op (UI features not supported in Logician):
//   registerShortcut, registerMessageRenderer, registerMarkdownTransformer,
//   registerEntryRenderer, sendMessage, sendUserMessage, appendEntry,
//   setSessionName, setLabel, setEditorComponent
//
// ─── Schema Conversion: TypeBox → JSON Schema ────────────────────────────────
//   Type.String()    → { type: "string" }
//   Type.Number()    → { type: "number" }
//   Type.Boolean()   → { type: "boolean" }
//   Type.Array()     → { type: "array", items: {...}, required: false }
//   Type.Object()    → { type: "object", properties: {...} }
//   Type.Union()     → handled via first member (best effort)
//   Type.Optional()  → removed from required list
//   Type.Literal()   → { type: "string" | "number", enum: [value] }
//
// ─── Context Surface ────────────────────────────────────────────────────────
//   ui.*              → passthrough or no-op for missing methods
//   mode              → "tui" (Logician is TUI-only)
//   hasUI             → true (Logician TUI always has UI)
//   cwd               → passthrough
//   sessionId         → passthrough
//   sessionManager    → no-op wrapper (read-only access to session ID)
//   modelRegistry     → no-op (no model management in Logician)
//   model             → undefined (no model selection API)
//   thinkingLevel     → undefined
//   signal            → passthrough (current agent abort signal)
//
// ─── Type Guard Helpers ─────────────────────────────────────────────────────
//   PiAdapter.isToolCallEventType<T>(name, event) → type guard
//   PiAdapter.isBashToolResult(event)             → type guard
//   signal            → no-op AbortController
//   isIdle            → best-effort (always true for Logician)
//   isProjectTrusted  → true
//   abort             → no-op
//   hasPendingMessages → false
//   shutdown          → no-op (Logician handles exit differently)
//   getContextUsage   → undefined
//   compact           → no-op
//   getSystemPrompt   → empty string
//   getSystemPromptOptions → empty object
//   waitForIdle       → resolves immediately
//   newSession        → no-op (returns cancelled)
//   fork              → no-op
//   navigateTree      → no-op
//   switchSession     → no-op
//   reload            → no-op (Logician reload is different)
//   sendMessage       → no-op
//   sendUserMessage   → no-op
//   appendEntry       → no-op
//   setSessionName    → no-op
//   getSessionName    → sessionId
//   setLabel          → no-op
//   exec              → passthrough to bash tool
//   getActiveTools    → []
//   getAllTools       → []
//   setActiveTools    → no-op
//   getCommands       → []
//   setModel          → false
//   getThinkingLevel  → undefined
//   setThinkingLevel  → no-op
//   registerProvider  → no-op
//   unregisterProvider → no-op
//   registerShortcut  → no-op
//   registerFlag      → no-op
//   getFlag           → undefined
//   registerMessageRenderer → no-op
//   registerMarkdownTransformer → no-op
//   registerEntryRenderer → no-op

import type { EventBus } from "../../event-bus.ts";
import type {
	ExtensionToolResult,
	ExtensionAPI as LApi,
	ExtensionContext as LContext,
	ExtensionEvent as LEvent,
	ExtensionEventType as LEventType,
} from "../../types.ts";

// ── TypeBox Schema → JSON Schema ─────────────────────────────────────────────

/** Check if an object looks like a TypeBox schema. */
function isTypeBoxSchema(obj: unknown): obj is Record<string, unknown> {
	return (
		typeof obj === "object" &&
		obj !== null &&
		("__type" in obj ||
			"$__type" in obj ||
			typeof (obj as Record<string, unknown>).type === "string")
	);
}

/** Convert a TypeBox schema to Logician's flat parameter schema. */
function convertSchema(
	schema: unknown,
	required: string[] = [],
): Record<string, unknown> {
	if (!isTypeBoxSchema(schema)) {
		return { type: "object", properties: {}, required: [] };
	}

	const s = schema as Record<string, unknown>;
	const typeBoxType = s.__type ?? s.$__type ?? undefined;
	const typeName = typeof typeBoxType === "string" ? typeBoxType : undefined;

	const result: Record<string, unknown> = {};

	switch (typeName) {
		case "String":
			result.type = "string";
			result.description =
				s.description ?? s.__metaDescription ?? s.description;
			break;
		case "Number":
			result.type = "number";
			result.description = s.description ?? s.__metaDescription;
			break;
		case "Boolean":
			result.type = "boolean";
			result.description = s.description ?? s.__metaDescription;
			break;
		case "Integer":
			result.type = "number";
			result.description = s.description ?? s.__metaDescription;
			break;
		case "Array":
			result.type = "array";
			result.description = s.description ?? s.__metaDescription;
			result.items = s.items !== undefined ? convertSchema(s.items, []) : {};
			break;
		case "Object": {
			result.type = "object";
			result.description = s.description ?? s.__metaDescription;
			result.properties = {};
			const props = s.properties as Record<string, unknown> | undefined;
			if (props && typeof props === "object") {
				const req = (s.required as string[]) ?? [];
				for (const [key, val] of Object.entries(props)) {
					(result.properties as Record<string, unknown>)[key] = convertSchema(
						val,
						required,
					);
					if (req.includes(key)) {
						if (!result.required) result.required = [];
						(result.required as string[]).push(key);
					}
				}
			}
			break;
		}
		case "Union": {
			const options = (s.anyOf ?? s.union) as unknown[] | undefined;
			if (options && Array.isArray(options) && options.length > 0) {
				// Use first option as the primary type
				const first = convertSchema(options[0], []);
				Object.assign(result, first);
				result.description = s.description ?? s.__metaDescription;
			} else {
				result.type = "object";
				result.properties = {};
			}
			break;
		}
		case "Literal": {
			const val = s.const;
			result.type =
				val === null ? "null" : typeof val === "number" ? "number" : "string";
			result.enum = [val];
			result.description = s.description ?? s.__metaDescription;
			break;
		}
		case "Enum": {
			const enumValues = s.enum as unknown[];
			if (enumValues && Array.isArray(enumValues)) {
				result.type = typeof enumValues[0] === "number" ? "number" : "string";
				result.enum = enumValues;
			} else {
				result.type = "string";
			}
			result.description = s.description ?? s.__metaDescription;
			break;
		}
		case "Partial":
			// Partial removes all required fields
			if (s.type === "Object") {
				result.type = "object";
				result.properties = {};
				const props = s.properties as Record<string, unknown> | undefined;
				if (props && typeof props === "object") {
					for (const [key, val] of Object.entries(props)) {
						(result.properties as Record<string, unknown>)[key] = convertSchema(
							val,
							[],
						);
					}
				}
			}
			break;
		case "Ref":
			// Forward reference — skip for now
			result.type = "object";
			result.properties = {};
			break;
		default:
			// Fallback: try to detect type from other fields
			if (s.type !== undefined) {
				result.type = s.type;
			} else if (s.properties !== undefined) {
				result.type = "object";
				result.properties = s.properties;
			} else if (s.items !== undefined) {
				result.type = "array";
				result.items = s.items;
			} else {
				result.type = "object";
				result.properties = {};
			}
			break;
	}

	return result;
}

// ── Event Bridge ─────────────────────────────────────────────────────────────

type PiEventType =
	| "project_trust"
	| "resources_discover"
	| "session_start"
	| "session_info_changed"
	| "session_before_switch"
	| "session_before_fork"
	| "session_before_compact"
	| "session_compact"
	| "session_shutdown"
	| "session_before_tree"
	| "session_tree"
	| "context"
	| "before_provider_request"
	| "before_provider_headers"
	| "after_provider_response"
	| "before_agent_start"
	| "agent_start"
	| "agent_end"
	| "agent_settled"
	| "turn_start"
	| "turn_end"
	| "message_start"
	| "message_update"
	| "message_end"
	| "tool_execution_start"
	| "tool_execution_update"
	| "tool_execution_end"
	| "tool_call"
	| "tool_result"
	| "model_select"
	| "thinking_level_changed"
	| "user_bash"
	| "input"
	// Retry / error observability
	| "agent_error"
	// Session lifecycle
	| "session_delete";

// ── Pi-compatible UI Context ─────────────────────────────────────────────────

interface PiUI {
	select(
		title: string,
		options: string[],
		opts?: { timeout?: number },
	): Promise<string | undefined>;
	confirm(
		title: string,
		message: string,
		opts?: { timeout?: number },
	): Promise<boolean>;
	input(
		title: string,
		placeholder?: string,
		opts?: { timeout?: number },
	): Promise<string | undefined>;
	notify(message: string, type?: "info" | "warning" | "error"): void;
	setStatus(key: string, text: string | undefined): void;
	setWorkingMessage(message?: string): void;
	setWorkingVisible(visible: boolean): void;
	setWorkingIndicator(options?: {
		frames?: string[];
		intervalMs?: number;
	}): void;
	setHiddenThinkingLabel(label?: string): void;
	setWidget(
		key: string,
		content: string[] | undefined,
		options?: { placement?: "aboveEditor" | "belowEditor" },
	): void;
	setFooter(factory: unknown): void;
	setHeader(factory: unknown): void;
	setTitle(title: string): void;
	custom<T>(factory: unknown, options?: unknown): Promise<T>;
	pasteToEditor(text: string): void;
	setEditorText(text: string): void;
	getEditorText(): string;
	editor(title: string, prefill?: string): Promise<string | undefined>;
	addAutocompleteProvider(factory: unknown): void;
	setEditorComponent(factory: unknown): void;
	getEditorComponent(): unknown;
	readonly theme: unknown;
	getAllThemes(): { name: string; path: string | undefined }[];
	getTheme(name: string): unknown;
	setTheme(theme: string | unknown): { success: boolean; error?: string };
	getToolsExpanded(): boolean;
	setToolsExpanded(expanded: boolean): void;
}

/** Wrap Logician's ExtensionUI with Pi-compatible surface. */
function createPiUI(logicianUI: {
	notify: (message: string, type?: "info" | "warning" | "error") => void;
	confirm: (
		title: string,
		message: string,
		opts?: { timeoutMs?: number },
	) => Promise<boolean>;
	input: (
		title: string,
		placeholder?: string,
		opts?: { timeoutMs?: number },
	) => Promise<string | undefined>;
	select: (
		title: string,
		options: Array<{ label: string; description?: string }>,
		opts?: { timeoutMs?: number },
	) => Promise<string | undefined>;
}): PiUI {
	return {
		select: async (title, options, _opts) => {
			const labels = options.map((o, _i) => {
				if (typeof o === "string") return { label: o };
				return o;
			});
			const result = await logicianUI.select(title, labels, {
				timeoutMs: _opts?.timeout,
			});
			if (!result) return undefined;
			const idx = labels.findIndex(l => l.label === result);
			return idx >= 0 ? options[idx] : undefined;
		},
		confirm: async (title, message, _opts) => {
			return logicianUI.confirm(title, message, {
				timeoutMs: _opts?.timeout,
			});
		},
		input: async (title, placeholder, _opts) => {
			return logicianUI.input(title, placeholder, {
				timeoutMs: _opts?.timeout,
			});
		},
		notify: (message, type) => {
			logicianUI.notify(message, type);
		},
		setStatus: () => {}, // no-op
		setWorkingMessage: () => {}, // no-op
		setWorkingVisible: () => {}, // no-op
		setWorkingIndicator: () => {}, // no-op
		setHiddenThinkingLabel: () => {}, // no-op
		setWidget: () => {}, // no-op
		setFooter: () => {}, // no-op
		setHeader: () => {}, // no-op
		setTitle: () => {}, // no-op
		custom: async <T>(): Promise<T> => undefined as unknown as T, // no-op
		pasteToEditor: () => {}, // no-op
		setEditorText: () => {}, // no-op
		getEditorText: () => "", // no-op
		editor: async () => undefined, // no-op
		addAutocompleteProvider: () => {}, // no-op
		setEditorComponent: () => {}, // no-op
		getEditorComponent: () => undefined, // no-op
		theme: undefined, // no-op
		getAllThemes: () => [], // no-op
		getTheme: () => undefined, // no-op
		setTheme: () => ({ success: false, error: "not supported" }), // no-op
		getToolsExpanded: () => false, // no-op
		setToolsExpanded: () => {}, // no-op
	};
}

// ── Pi-compatible Session Manager (read-only stub) ──────────────────────────

interface PiSessionManager {
	getSessionFile(): string | undefined;
	getSessionId(): string;
	getEntries(): unknown[];
	getBranch(): unknown;
	buildContextEntries(): unknown[];
	getLeafId(): string | undefined;
}

function createPiSessionManager(sessionId: string): PiSessionManager {
	return {
		getSessionFile: () => `session-${sessionId}`,
		getSessionId: () => sessionId,
		getEntries: () => [],
		getBranch: () => null,
		buildContextEntries: () => [],
		getLeafId: () => sessionId,
	};
}

// ── Input Event Handler ─────────────────────────────────────────────────────
// Handler for Pi's input event. Returns 'continue', 'transform', or 'handled'.
type InputEventHandler = (
	text: string,
	images: unknown[],
	source: "interactive" | "rpc" | "extension",
	hasUI: boolean,
	ui: PiUI,
) => Promise<{
	action: "continue" | "transform" | "handled";
	text?: string;
	images?: unknown[];
} | null>;

// ── User Bash Event Handler ─────────────────────────────────────────────────
// Handler for Pi's user_bash event. Can intercept, wrap, or replace execution.
type UserBashEventHandler = (
	command: string,
	excludeFromContext: boolean,
	cwd: string,
	hasUI: boolean,
	ui: PiUI,
) => Promise<{
	action: "continue" | "intercept" | "replace";
	result?: { output: string; exitCode: number; cancelled: boolean };
	operations?: unknown;
} | null>;

// ── Project Trust Event Handler ─────────────────────────────────────────────
// Handler for Pi's project_trust event. Can decide trust (yes/no/undecided).
type ProjectTrustEventHandler = (
	cwd: string,
	hasUI: boolean,
	ui: PiUI,
) => Promise<{
	trusted: "yes" | "no" | "undecided";
	remember?: boolean;
} | null>;

// ── Pi-compatible ExtensionContext ───────────────────────────────────────────

interface PiExtensionContext {
	ui: PiUI;
	mode: "tui" | "rpc" | "json" | "print";
	hasUI: boolean;
	cwd: string;
	sessionManager: PiSessionManager;
	modelRegistry: unknown;
	model: unknown;
	scopedModels: readonly unknown[];
	thinkingLevel?: unknown;
	isIdle(): boolean;
	isProjectTrusted(): boolean;
	signal: AbortSignal | undefined;
	abort(): void;
	hasPendingMessages(): boolean;
	shutdown(): void;
	getContextUsage(): unknown;
	compact(options?: unknown): void;
	getSystemPrompt(): string;
}

interface PiCommandContext extends PiExtensionContext {
	getSystemPromptOptions(): unknown;
	waitForIdle(): Promise<void>;
	newSession(options?: unknown): Promise<{ cancelled: boolean }>;
	fork(entryId: string, options?: unknown): Promise<{ cancelled: boolean }>;
	navigateTree(
		targetId: string,
		options?: unknown,
	): Promise<{ cancelled: boolean }>;
	switchSession(
		sessionPath: string,
		options?: unknown,
	): Promise<{ cancelled: boolean }>;
	reload(): Promise<void>;
}

function createPiContext(
	logicianCtx: LContext,
	runtime?: PiRuntimePort,
): PiExtensionContext {
	return {
		ui: createPiUI(logicianCtx.ui),
		mode: "tui",
		hasUI: true,
		cwd: logicianCtx.cwd,
		sessionManager: createPiSessionManager(logicianCtx.sessionId),
		modelRegistry: undefined,
		model: undefined,
		scopedModels: [],
		thinkingLevel: undefined,
		isIdle: () => runtime?.isIdle?.() ?? true,
		isProjectTrusted: () => true,
		signal: undefined,
		abort: () => runtime?.abort?.(),
		hasPendingMessages: () => runtime?.hasPendingMessages?.() ?? false,
		shutdown: () => runtime?.shutdown?.(),
		getContextUsage: () => undefined,
		compact: () => runtime?.compact?.(),
		getSystemPrompt: () => runtime?.getSystemPrompt?.() ?? "",
	};
}

function createPiCommandContext(
	logicianCtx: LContext,
	runtime?: PiRuntimePort,
): PiCommandContext {
	const ctx = createPiContext(logicianCtx, runtime) as PiCommandContext;
	ctx.getSystemPromptOptions = () => ({});
	ctx.waitForIdle = async () => {};
	ctx.newSession = async () => ({ cancelled: true });
	ctx.fork = async () => ({ cancelled: true });
	ctx.navigateTree = async () => ({ cancelled: true });
	ctx.switchSession = async () => ({ cancelled: true });
	ctx.reload = async () => {};
	return ctx;
}

// ── Pi-compatible Tool Definition ────────────────────────────────────────────

interface PiToolDefinition {
	name: string;
	label?: string;
	description: string;
	promptSnippet?: string;
	promptGuidelines?: string[];
	parameters: unknown; // TypeBox schema
	execute: (
		toolCallId: string,
		params: Record<string, unknown>,
		signal: AbortSignal | undefined,
		onUpdate: unknown,
		ctx: PiExtensionContext,
	) => Promise<{
		content: Array<{ type: string; text: string }>;
		details?: Record<string, unknown>;
		isError?: boolean;
	}>;
}

// ── Pi-compatible Command ────────────────────────────────────────────────────

interface PiCommand {
	name: string;
	description?: string;
	handler: (args: string, ctx: PiCommandContext) => Promise<void>;
}

// ── Pi-compatible ExtensionAPI ───────────────────────────────────────────────

export interface PiExtensionAPI {
	on(
		event: PiEventType,
		handler: (
			event: Record<string, unknown>,
			ctx: PiExtensionContext,
		) => Promise<unknown> | unknown,
	): void;
	registerTool(tool: PiToolDefinition): void;
	registerCommand(
		name: string,
		options: {
			description?: string;
			handler: (args: string, ctx: PiCommandContext) => Promise<void>;
		},
	): void;
	registerShortcut(
		shortcut: string,
		options: {
			description?: string;
			handler: (ctx: PiExtensionContext) => Promise<void> | void;
		},
	): void;
	registerFlag(
		name: string,
		options: {
			description?: string;
			type: "boolean" | "string";
			default?: boolean | string;
		},
	): void;
	getFlag(name: string): boolean | string | undefined;
	registerMessageRenderer(customType: string, renderer: unknown): void;
	registerMarkdownTransformer(transformer: unknown): void;
	registerEntryRenderer(customType: string, renderer: unknown): void;
	sendMessage(
		message: Record<string, unknown>,
		options?: Record<string, unknown>,
	): void;
	sendUserMessage(content: string, options?: Record<string, unknown>): void;
	appendEntry(customType: string, data?: unknown): void;
	setSessionName(name: string): void;
	getSessionName(): string | undefined;
	setLabel(entryId: string, label: string | undefined): void;
	exec(
		command: string,
		args: string[],
		options?: Record<string, unknown>,
	): Promise<unknown>;
	getActiveTools(): string[];
	getAllTools(): unknown[];
	setActiveTools(toolNames: string[]): void;
	getCommands(): unknown[];
	setModel(model: unknown): Promise<boolean>;
	getThinkingLevel(): unknown;
	setThinkingLevel(level: unknown): void;
	registerProvider(provider: unknown): void;
	registerProvider(name: string, config: unknown): void;
	unregisterProvider(name: string): void;
	events: EventBus;
	/** Register a handler for Pi's input event. */
	onInput(handler: InputEventHandler): void;
	/** Register a handler for Pi's user_bash event. */
	onUserBash(handler: UserBashEventHandler): void;
	/** Register a handler for Pi's project_trust event. */
	onProjectTrust(handler: ProjectTrustEventHandler): void;
}

// ── Adapter Factory ──────────────────────────────────────────────────────────

export interface PiAdapterOptions {
	sessionId: string;
	cwd: string;
	runtime?: PiRuntimePort;
}

export interface PiRuntimePort {
	isIdle?(): boolean;
	hasPendingMessages?(): boolean;
	abort?(): void;
	shutdown?(): void;
	compact?(): void;
	getSystemPrompt?(): string;
	sendUserMessage?(content: string): void;
	getActiveTools?(): string[];
	getAllTools?(): unknown[];
	setActiveTools?(names: string[]): void;
	setModel?(model: unknown): Promise<boolean>;
	getThinkingLevel?(): unknown;
	setThinkingLevel?(level: unknown): void;
}

export class PiAdapter {
	private logicianApi: LApi;
	private logicianCtx: LContext;
	private registeredTools: PiToolDefinition[] = [];
	private registeredCommands: PiCommand[] = [];
	private registeredFlags: Record<string, boolean | string | undefined> = {};
	private piHandlers: Array<{
		event: PiEventType;
		handler: (
			event: Record<string, unknown>,
			ctx: PiExtensionContext,
		) => Promise<unknown> | unknown;
	}> = [];
	// Handlers for the three new events (registered via callbacks, not via pi.on)
	private inputHandlers: InputEventHandler[] = [];
	private userBashHandlers: UserBashEventHandler[] = [];
	private projectTrustHandlers: ProjectTrustEventHandler[] = [];
	private readonly runtime?: PiRuntimePort;

	constructor(api: LApi, ctx: LContext, options: PiAdapterOptions) {
		this.logicianApi = api;
		this.logicianCtx = ctx;
		this.runtime = options.runtime;
	}

	hasHandlers(event: string): boolean {
		return this.piHandlers.some(handler => handler.event === event);
	}

	async emitToolCall(event: {
		toolCallId: string;
		toolName: string;
		input: Record<string, unknown>;
	}): Promise<{
		input: Record<string, unknown>;
		block?: boolean;
		reason?: string;
		terminate?: boolean;
	}> {
		const piEvent: Record<string, unknown> = {
			type: "tool_call",
			...event,
			input: { ...event.input },
		};
		const ctx = createPiContext(this.logicianCtx, this.runtime);
		let decision: { block?: boolean; reason?: string; terminate?: boolean } =
			{};
		for (const entry of this.piHandlers) {
			if (entry.event !== "tool_call") continue;
			try {
				const result = await entry.handler(piEvent, ctx);
				if (result && typeof result === "object") {
					decision = { ...decision, ...(result as typeof decision) };
				}
			} catch (error) {
				console.error("[pi-adapter] handler error for tool_call:", error);
			}
		}
		return {
			input: (piEvent.input as Record<string, unknown>) ?? event.input,
			...decision,
		};
	}

	async emitToolResult(event: {
		toolCallId: string;
		toolName: string;
		input: Record<string, unknown>;
		content: Array<{ type: string; text: string }>;
		details?: Record<string, unknown>;
		isError: boolean;
	}): Promise<typeof event> {
		const piEvent: Record<string, unknown> = { type: "tool_result", ...event };
		const ctx = createPiContext(this.logicianCtx, this.runtime);
		for (const entry of this.piHandlers) {
			if (entry.event !== "tool_result") continue;
			try {
				const result = await entry.handler(piEvent, ctx);
				if (result && typeof result === "object")
					Object.assign(piEvent, result);
			} catch (error) {
				console.error("[pi-adapter] handler error for tool_result:", error);
			}
		}
		return {
			...event,
			content: (piEvent.content as typeof event.content) ?? event.content,
			details:
				(piEvent.details as Record<string, unknown> | undefined) ??
				event.details,
			isError: (piEvent.isError as boolean | undefined) ?? event.isError,
		};
	}

	/**
	 * Get the Pi-compatible API surface. Call this from a Pi extension factory.
	 * The returned API bridges to Logician's runtime.
	 */
	getApi(): PiExtensionAPI {
		return {
			on: (event, handler) => {
				this.piHandlers.push({ event, handler });
			},
			registerTool: tool => {
				this.registeredTools.push(tool);
				this.forwardTool(tool);
			},
			registerCommand: (name, options) => {
				this.registeredCommands.push({ name, ...options });
				this.forwardCommand(name, options);
			},
			registerShortcut: () => {}, // no-op
			registerFlag: (name, options) => {
				this.registeredFlags[name] = options.default;
			},
			getFlag: name => this.registeredFlags[name],
			registerMessageRenderer: () => {}, // no-op
			registerMarkdownTransformer: () => {}, // no-op
			registerEntryRenderer: () => {}, // no-op
			onInput: handler => {
				this.inputHandlers.push(handler);
			},
			onUserBash: handler => {
				this.userBashHandlers.push(handler);
			},
			onProjectTrust: handler => {
				this.projectTrustHandlers.push(handler);
			},
			sendMessage: () => {}, // no-op
			sendUserMessage: content => this.runtime?.sendUserMessage?.(content),
			appendEntry: () => {}, // no-op
			setSessionName: () => {}, // no-op
			getSessionName: () => this.logicianCtx.sessionId,
			setLabel: () => {}, // no-op
			exec: async (_command, _args) => {
				// Forward to bash tool
				return { output: "", exitCode: 0 };
			},
			getActiveTools: () => this.runtime?.getActiveTools?.() ?? [],
			getAllTools: () =>
				this.runtime?.getAllTools?.() ??
				this.registeredTools.map(t => ({
					name: t.name,
					description: t.description,
				})),
			setActiveTools: names => this.runtime?.setActiveTools?.(names),
			getCommands: () =>
				this.registeredCommands.map(c => ({
					name: c.name,
					description: c.description,
				})),
			setModel: model =>
				this.runtime?.setModel?.(model) ?? Promise.resolve(false),
			getThinkingLevel: () => this.runtime?.getThinkingLevel?.(),
			setThinkingLevel: level => this.runtime?.setThinkingLevel?.(level),
			registerProvider: () => {}, // no-op
			unregisterProvider: () => {},
			events: this.logicianApi.events,
		};
	}

	/**
	 * Forward a Pi tool to Logician's tool registry.
	 * Gracefully handles missing registerTool on the API.
	 */
	private forwardTool(piTool: PiToolDefinition): void {
		const jsonSchema = convertSchema(piTool.parameters);
		if (typeof this.logicianApi.registerTool === "function") {
			this.logicianApi.registerTool({
				name: piTool.name,
				label: piTool.label ?? piTool.name,
				description: piTool.description,
				parameters: jsonSchema as any,
				execute: async (toolCallId, params, lctx) => {
					const ctx = createPiContext(
						{
							ui: this.logicianCtx.ui,
							state: this.logicianCtx.state,
							cwd: lctx.cwd,
							sessionId: lctx.sessionId,
						},
						this.runtime,
					);
					const result = await piTool.execute(
						toolCallId,
						params,
						undefined,
						undefined,
						ctx,
					);
					return {
						content: result.content.map(c => c.text).join("\n"),
						isError: result.isError,
						details: result.details,
					} as ExtensionToolResult;
				},
			});
		}
	}

	/**
	 * Forward a Pi command to Logician's command registry.
	 */
	private forwardCommand(
		name: string,
		options: {
			description?: string;
			handler: (args: string, ctx: PiCommandContext) => Promise<void>;
		},
	): void {
		this.logicianApi.registerCommand({
			name,
			description: options.description ?? "",
			handler: async args => {
				const ctx = createPiCommandContext(this.logicianCtx, this.runtime);
				await options.handler(args, ctx);
				return "";
			},
		});
	}

	/**
	 * Emit a Pi event to all registered handlers, translated from a Logician event.
	 * Call this from the Logician extension runner when a Logician event fires.
	 * Returns handler return values (for before_agent_start / context) so the
	 * harness can apply message/systemPrompt overrides.
	 */
	async emitFromLogician(logicianEvent: LEvent): Promise<{
		messages?: unknown[];
		systemPrompt?: string;
	}> {
		const piEvent = this.translateLogicianToPi(logicianEvent);
		if (!piEvent) return {};

		const ctx = createPiContext(
			{
				ui: this.logicianCtx.ui,
				state: this.logicianCtx.state,
				cwd: this.logicianCtx.cwd,
				sessionId: this.logicianCtx.sessionId,
			},
			this.runtime,
		);

		let collectedMessages: unknown[] | undefined;
		let collectedSystemPrompt: string | undefined;

		for (const { event, handler } of this.piHandlers) {
			if (event === piEvent.type) {
				try {
					const result = await handler(piEvent as Record<string, unknown>, ctx);

					// tool_call: blocking + terminate
					if (
						piEvent.type === "tool_call" &&
						result &&
						typeof result === "object"
					) {
						const hookResult = result as {
							block?: boolean;
							reason?: string;
							terminate?: boolean;
						};
						if (hookResult.block) {
							await this.logicianApi.emit({
								type: "tool_execution_start" as LEventType,
								context: {
									sessionId: this.logicianCtx.sessionId,
									cwd: this.logicianCtx.cwd,
									tool_name: piEvent.toolName,
									tool_input: piEvent.input,
								},
								block: true,
								reason: hookResult.reason ?? "Blocked by extension",
								terminate: hookResult.terminate ?? false,
							} as any);
						}
					}

					// tool_result: result modification (content, details, isError, usage)
					if (
						piEvent.type === "tool_result" &&
						result &&
						typeof result === "object"
					) {
						const mod = result as {
							content?: Array<{ text?: string }>;
							details?: unknown;
							isError?: boolean;
							usage?: unknown;
						};
						const piContent = (piEvent as any).content as
							| Array<{ text?: string }>
							| undefined;
						const toolResultText =
							mod.content?.[0]?.text ?? piContent?.[0]?.text ?? "";
						await this.logicianApi.emit({
							type: "tool_execution_end" as LEventType,
							context: {
								sessionId: this.logicianCtx.sessionId,
								cwd: this.logicianCtx.cwd,
								toolCallId: (piEvent as any).toolCallId ?? "",
								tool_name: (piEvent as any).toolName ?? "unknown",
								tool_result: toolResultText,
								tool_details: mod.details,
								is_error: mod.isError ?? false,
								usage: mod.usage ?? undefined,
							},
						} as any);
					}

					// before_agent_start: collect systemPrompt (don't re-emit — caller handles it)
					if (
						piEvent.type === "before_agent_start" &&
						result &&
						typeof result === "object"
					) {
						const agentStartResult = result as {
							message?: {
								customType: string;
								content: string;
								display: boolean;
							};
							systemPrompt?: string;
						};
						if (agentStartResult.systemPrompt) {
							collectedSystemPrompt = agentStartResult.systemPrompt;
							piEvent.systemPrompt = agentStartResult.systemPrompt;
						}
					}

					// context: collect modified messages
					if (
						piEvent.type === "context" &&
						result &&
						typeof result === "object"
					) {
						const contextResult = result as { messages?: unknown[] };
						if (contextResult.messages) {
							collectedMessages = contextResult.messages;
							piEvent.messages = contextResult.messages;
						}
					}
				} catch (err) {
					console.error(
						`[pi-adapter] handler error for ${piEvent.type}:`,
						err instanceof Error ? err.message : String(err),
					);
				}
			}
		}

		return {
			messages: collectedMessages,
			systemPrompt: collectedSystemPrompt,
		};
	}

	/**
	 * Translate a Logician event to a Pi event shape.
	 *
	 * ─── Event Mapping Matrix ───────────────────────────────────────────
	 *
	 * Pi Event                  │ Logician Event      │ Status & Fields
	 * ──────────────────────────┼─────────────────────┼──────────────────────────
	 * project_trust             │ ─                   │ N/A (handler)
	 * resources_discover        │ ─                   │ Not emitted
	 * session_start             │ session_start       │ reason + previousSessionFile
	 * session_info_changed      │ ─                   │ Not emitted
	 * session_before_switch     │ ─                   │ Not emitted
	 * session_before_fork       │ ─                   │ Not emitted
	 * session_before_compact    │ before_compact      │ preparation, branchEntries,
	 *                           │                     │ reason, willRetry, signal
	 * session_compact           │ after_compact       │ compactionEntry, fromExtension,
	 *                           │                     │ reason, willRetry, tokens
	 * session_shutdown          │ session_end         │ reason: quit
	 * session_before_tree       │ ─                   │ Not emitted
	 * session_tree              │ ─                   │ Not emitted
	 * context                   │ ─                   │ Not emitted
	 * before_provider_request   │ ─                   │ Not emitted
	 * before_provider_headers   │ ─                   │ Not emitted
	 * after_provider_response   │ ─                   │ Not emitted
	 * before_agent_start        │ user_prompt_submit  │ prompt, images, systemPrompt,
	 *                           │                     │ systemPromptOptions (+ result→emit)
	 * agent_start               │ agent_start         │ plain passthrough
	 * agent_end                 │ agent_end           │ messages[]
	 * agent_settled             │ ─                   │ Not emitted
	 * turn_start                │ turn_start          │ turnIndex, timestamp
	 * turn_end                  │ turn_end            │ turnIndex, message, toolResults
	 * message_start             │ message_start       │ message (+ result→replace)
	 * message_update            │ message_update      │ message, assistantMessageEvent
	 * message_end               │ message_end         │ message (+ result→replace)
	 * tool_execution_start      │ tool_call_start     │ toolCallId, toolName, args
	 * tool_execution_update     │ ─                   │ Not emitted
	 * tool_execution_end        │ tool_call_end       │ toolCallId, toolName, result, isError
	 * model_select              │ ─                   │ Not emitted
	 * thinking_level_changed     │ ─                   │ Not emitted
	 * tool_call                 │ tool_execution_start  │ toolCallId, toolName, input
	 *                           │                     │ (mutable) + block/terminate
	 * tool_result               │ tool_execution_end    │ toolCallId, toolName, input,
	 *                           │                     │ content[], details, isError, usage
	 *                           │                     │ (+ result→modify)
	 * user_bash                 │ ─                   │ N/A (handler)
	 * input                     │ ─                   │ N/A (handler)
	 *
	 * agent_error               │ agent_error       │ message, phase,
	 *                           │                   │ recoverable
	 * session_delete            │ session_delete    │ sessionFile, sessionId
	 *
	 * "Not emitted" = Logician doesn't produce this event.
	 * "N/A (handler)" = handled via dedicated emit methods, not translation.
	 * "result→X" = Pi handler return values are processed and fed back to Logician.
	 */
	private translateLogicianToPi(event: LEvent): Record<string, unknown> | null {
		switch (event.type) {
			case "session_start": {
				// Pi's SessionStartEvent: reason, previousSessionFile?
				const ctx = event.context as any;
				return {
					type: "session_start",
					reason: ctx.reason ?? "startup",
					previousSessionFile: ctx.previousSessionFile,
				};
			}
			case "session_shutdown":
				return {
					type: "session_shutdown",
					reason: "quit",
				};
			case "agent_start":
				return { type: "agent_start" };
			case "agent_settled":
				return { type: "agent_settled" };
			case "agent_end": {
				// Pi's AgentEndEvent: messages
				const ctx = event.context as any;
				return { type: "agent_end", messages: ctx.messages };
			}
			case "context": {
				// Pi's ContextEvent: messages (pre-LLM-call)
				const ctx = event.context as any;
				return {
					type: "context",
					messages: ctx.messages ?? [],
				};
			}
			case "before_agent_start": {
				// Pi's BeforeAgentStartEvent: prompt, images?, systemPrompt?, systemPromptOptions?
				const ctx = event.context as any;
				return {
					type: "before_agent_start",
					prompt: ctx.prompt,
					images: ctx.images,
					systemPrompt: ctx.systemPrompt,
					systemPromptOptions: ctx.systemPromptOptions,
				};
			}
			case "turn_start":
				return {
					type: "turn_start",
					turnIndex: (event.context as any).turnIndex,
					timestamp: Date.now(),
				};
			case "turn_end": {
				// Pi's TurnEndEvent: turnIndex, timestamp, message, toolResults
				const ctx = event.context as any;
				return {
					type: "turn_end",
					turnIndex: ctx.turnIndex,
					message: ctx.message,
					toolResults: ctx.toolResults,
				};
			}
			case "message_start":
				return { type: "message_start", message: event.context.message };
			case "message_update": {
				// Pi's MessageUpdateEvent: message, assistantMessageEvent
				const ctx = event.context as any;
				return {
					type: "message_update",
					message: ctx.message,
					assistantMessageEvent: ctx.assistantMessageEvent,
				};
			}
			case "message_end":
				return { type: "message_end", message: event.context.message };
			case "tool_execution_start": {
				const args =
					event.context.tool_input ??
					event.context.toolInput ??
					event.context.args ??
					{};
				return {
					type: "tool_execution_start",
					toolCallId: (event.context as any).toolCallId,
					toolName: event.context.tool_name ?? event.context.toolName,
					args,
					// Compatibility alias used by early Logician Pi adapters.
					input: args,
				};
			}
			case "tool_execution_end":
				return {
					type: "tool_execution_end",
					toolCallId: (event.context as any).toolCallId,
					toolName: event.context.tool_name ?? event.context.toolName,
					result: event.context.tool_result,
					isError: event.context.is_error,
				};
			case "tool_execution_update":
				return {
					type: "tool_execution_update",
					toolCallId: (event.context as any).toolCallId,
					toolName: event.context.tool_name ?? event.context.toolName,
					args: event.context.tool_input ?? event.context.args,
					partialResult:
						event.context.partial_result ?? event.context.partialResult,
				};
			case "model_select":
				return {
					type: "model_select",
					model: event.context.model,
					previousModel: event.context.previousModel,
					source: event.context.source ?? "set",
				};
			case "session_before_compact": {
				// Pi's SessionBeforeCompactEvent: preparation, branchEntries, customInstructions?, reason, willRetry, signal
				const ctx = event.context as any;
				return {
					type: "session_before_compact",
					preparation: ctx.preparation,
					branchEntries: ctx.branchEntries,
					customInstructions: ctx.customInstructions,
					reason: ctx.reason ?? "manual",
					willRetry: ctx.willRetry,
					signal: ctx.signal,
					tokensBefore: ctx.tokensBefore,
					messages: ctx.messages,
				};
			}
			case "session_compact": {
				// Pi's SessionCompactEvent: compactionEntry, fromExtension, reason, willRetry
				const ctx = event.context as any;
				return {
					type: "session_compact",
					compactionEntry: ctx.compactionEntry ?? ctx.summaryEntry,
					fromExtension: ctx.fromExtension,
					reason: ctx.reason ?? "manual",
					willRetry: ctx.willRetry,
					tokensBefore: ctx.tokensBefore,
					tokensAfter: ctx.tokensAfter,
					messages: ctx.messages,
				};
			}
			case "agent_error":
				return {
					type: "agent_error",
					message: (event.context as any).message,
					phase: (event.context as any).phase ?? "other",
					recoverable: (event.context as any).recoverable ?? true,
				};
			case "session_delete":
				return {
					type: "session_delete",
					sessionFile: (event.context as any).sessionFile,
					sessionId: (event.context as any).sessionId,
				};
			case "queue_update":
				// Generic — map to a reasonable default or skip
				return null;
			default:
				return null;
		}
	}

	/**
	 * Get all Pi tools registered through this adapter.
	 */
	getRegisteredTools(): PiToolDefinition[] {
		return [...this.registeredTools];
	}

	/**
	 * Get all Pi commands registered through this adapter.
	 */
	getRegisteredCommands(): PiCommand[] {
		return [...this.registeredCommands];
	}

	/**
	 * Emit a Pi input event to all registered handlers.
	 * Call from Logician's input controller before processing.
	 * @returns {action: 'continue'|'transform'|'handled', text?, images?} from the first non-null handler, or null if no handlers.
	 */
	async emitInputEvent(
		text: string,
		images: unknown[] = [],
		source: "interactive" | "rpc" | "extension" = "interactive",
	): Promise<{
		action: "continue" | "transform" | "handled";
		text?: string;
		images?: unknown[];
	} | null> {
		const ctx = createPiContext(
			{
				ui: this.logicianCtx.ui,
				state: this.logicianCtx.state,
				cwd: this.logicianCtx.cwd,
				sessionId: this.logicianCtx.sessionId,
			},
			this.runtime,
		);

		for (const handler of this.inputHandlers) {
			try {
				const result = await handler(text, images, source, ctx.hasUI, ctx.ui);
				if (result) return result;
			} catch (err) {
				console.error(
					"[pi-adapter] input handler error:",
					err instanceof Error ? err.message : String(err),
				);
			}
		}
		return null; // default: continue
	}

	/**
	 * Emit a Pi user_bash event to all registered handlers.
	 * Call from Logician's bash execution before running.
	 * @returns {action: 'continue'|'intercept'|'replace', result?, operations?} from the first non-null handler, or null if no handlers.
	 */
	async emitUserBashEvent(
		command: string,
		excludeFromContext: boolean = false,
	): Promise<{
		action: "continue" | "intercept" | "replace";
		result?: { output: string; exitCode: number; cancelled: boolean };
		operations?: unknown;
	} | null> {
		const ctx = createPiContext(
			{
				ui: this.logicianCtx.ui,
				state: this.logicianCtx.state,
				cwd: this.logicianCtx.cwd,
				sessionId: this.logicianCtx.sessionId,
			},
			this.runtime,
		);

		for (const handler of this.userBashHandlers) {
			try {
				const result = await handler(
					command,
					excludeFromContext,
					this.logicianCtx.cwd,
					ctx.hasUI,
					ctx.ui,
				);
				if (result) return result;
			} catch (err) {
				console.error(
					"[pi-adapter] user_bash handler error:",
					err instanceof Error ? err.message : String(err),
				);
			}
		}
		return null; // default: continue
	}

	/**
	 * Emit a Pi project_trust event to all registered handlers.
	 * Call from Logician's trust prompt before showing the overlay.
	 * @returns {trusted: 'yes'|'no'|'undecided', remember?} from the first non-null handler, or null if no handlers.
	 */
	async emitProjectTrustEvent(cwd: string): Promise<{
		trusted: "yes" | "no" | "undecided";
		remember?: boolean;
	} | null> {
		const ctx = createPiContext(
			{
				ui: this.logicianCtx.ui,
				state: this.logicianCtx.state,
				cwd: cwd,
				sessionId: this.logicianCtx.sessionId,
			},
			this.runtime,
		);

		for (const handler of this.projectTrustHandlers) {
			try {
				const result = await handler(cwd, ctx.hasUI, ctx.ui);
				if (result) return result;
			} catch (err) {
				console.error(
					"[pi-adapter] project_trust handler error:",
					err instanceof Error ? err.message : String(err),
				);
			}
		}
		return null; // default: continue (let Logician handle it)
	}
}

// ── Type Guard Helpers (mirrors @earendil-works/pi-coding-agent) ───────────

/**
 * Type guard for Pi tool call events.
 * Matches the `isToolCallEventType` helper from Pi's API.
 * @param toolName - The tool name to check.
 * @param event - The event object to test.
 * @returns true if the event is a tool_call for the given tool.
 */
export function isToolCallEventType(
	toolName: string,
	event: Record<string, unknown>,
): boolean {
	return event.type === "tool_call" && event.toolName === toolName;
}

/**
 * Type guard for bash tool results.
 * Matches the `isBashToolResult` helper from Pi's API.
 * @param event - The event object to test.
 * @returns true if the event is a tool_result for a bash tool.
 */
export function isBashToolResult(event: Record<string, unknown>): boolean {
	return event.type === "tool_result" && event.toolName === "bash";
}
