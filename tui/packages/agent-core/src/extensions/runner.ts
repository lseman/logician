// ── Extension runner ─────────────────────────────────────────────────────────
// Loads and executes extensions, collecting registered tools and commands.
// Wires extension event handlers into the agent runner via the hook bus.
// Provides both the legacy untyped EventBus and a structured typed event system.

import type { AgentHooks } from "../core/types.ts";
import type {
	AfterToolCallExtensionResult,
	BeforeToolCallExtensionResult,
	ExtensionAPI,
	ExtensionContext,
	ExtensionDefinition,
	ExtensionEvent,
	ExtensionEventContext,
	ExtensionEventHandler,
	ExtensionEventType,
	ExtensionState,
	RegisteredCommand,
	RegisteredTool,
} from "./types.ts";
import { createEventBus, type EventBus } from "./event-bus.ts";
import { createExtensionState } from "./state.ts";
import { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";
import { createExtensionContext } from "../hooks/extensions/context.ts";
import type { ExtensionEventName } from "../hooks/extensions/events.ts";

// ============================================================================
// No-op UI (for headless/non-TUI contexts)
// ============================================================================

const noopUI = {
	notify: (_msg: string, _type?: string) => {},
	confirm: async () => false,
	input: async () => undefined,
	select: async () => undefined,
};

// ============================================================================
// State wrapper
// ============================================================================

function createStateWrapper(extId: string): ExtensionState {
	const db = createExtensionState(extId);
	return {
		get: async <T>(key: string): Promise<T | undefined> => {
			const raw = db.get(key);
			if (raw === null) return undefined;
			try {
				return JSON.parse(raw) as T;
			} catch {
				return raw as unknown as T;
			}
		},
		set: async (key: string, value: unknown): Promise<void> => {
			const serialized = typeof value === "string" ? value : JSON.stringify(value);
			db.set(key, serialized);
		},
		delete: async (key: string): Promise<void> => {
			db.delete(key);
		},
		keys: async (): Promise<string[]> => {
			return db.keys();
		},
	};
}

// ============================================================================
// Extension Runner
// ============================================================================

export interface ExtensionRunnerOptions {
	sessionId: string;
	cwd: string;
}

interface HandlerEntry {
	handler: ExtensionEventHandler;
	ctx: ExtensionContext;
	source: string;
}

export class ExtensionRunner {
	private handlers = new Map<ExtensionEventType, HandlerEntry[]>();
	private tools: Array<{ tool: RegisteredTool; source: string }> = [];
	private commands: Array<{ command: RegisteredCommand; source: string }> = [];
	private extensions: Array<{ def: ExtensionDefinition; unload?: () => void }> = [];
	private eventBus: EventBus;
	/** Structured typed event bus for lifecycle events */
	private typedBus: ExtensionEventBus;
	/** Shared context for extension event handlers */
	private extContext: ReturnType<typeof createExtensionContext>;

	constructor(private readonly options: ExtensionRunnerOptions) {
		this.eventBus = createEventBus();
		this.typedBus = new ExtensionEventBus();
		this.extContext = createExtensionContext();
	}

	get events(): EventBus {
		return this.eventBus;
	}

	/**
	 * Load and execute a set of extensions from definitions.
	 * Each extension module is expected to export a default function that
	 * receives an ExtensionAPI instance.
	 */
	async load(definitions: ExtensionDefinition[]): Promise<void> {
		for (const def of definitions) {
			try {
				const mod = await import(/* @vite-ignore */ /* @webpackIgnore: true */ def.path);
				const factory = mod.default as ((api: ExtensionAPI) => void) | undefined;
				if (!factory) {
					console.warn(`[logician] extension "${def.name}" has no default export`);
					continue;
				}

				const api = this.createAPI(def);
				factory(api);
				this.extensions.push({ def, unload: api.unload });
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.error(`[logician] failed to load extension "${def.name}": ${message}`);
			}
		}
	}

	private createAPI(def: ExtensionDefinition): ExtensionAPI & { unload?: () => void } {
		const ownedHandlers: HandlerEntry[] = [];
		const state = createStateWrapper(def.name);
		const ctx: ExtensionContext = {
			ui: noopUI,
			state,
			cwd: this.options.cwd,
			sessionId: this.options.sessionId,
		};

		const on = (event: ExtensionEventType, handler: ExtensionEventHandler): (() => void) => {
			const entry: HandlerEntry = { handler, ctx, source: def.name };
			ownedHandlers.push(entry);
			const list = this.handlers.get(event) ?? [];
			list.push(entry);
			this.handlers.set(event, list);
			return () => {
				const current = this.handlers.get(event) ?? [];
				const filtered = current.filter((h) => h !== entry);
				if (filtered.length === 0) {
					this.handlers.delete(event);
				} else {
					this.handlers.set(event, filtered);
				}
				const ownedIndex = ownedHandlers.indexOf(entry);
				if (ownedIndex >= 0) ownedHandlers.splice(ownedIndex, 1);
			};
		};

		const registerTool = (tool: RegisteredTool): void => {
			this.tools.push({ tool, source: def.name });
		};

		const registerCommand = (command: RegisteredCommand): void => {
			this.commands.push({ command, source: def.name });
		};

		const emit = async (event: ExtensionEvent): Promise<void> => {
			await this.emit(event);
		};

		const unload = (): void => {
			this.tools = this.tools.filter((t) => t.source !== def.name);
			this.commands = this.commands.filter((c) => c.source !== def.name);
			for (const [event, list] of this.handlers) {
				const filtered = list.filter((entry) => entry.source !== def.name);
				if (filtered.length === 0) {
					this.handlers.delete(event);
				} else {
					this.handlers.set(event, filtered);
				}
			}
			ownedHandlers.length = 0;
		};

		return { on, registerTool, registerCommand, emit, events: this.eventBus, info: { name: def.name, path: def.path }, unload };
	}

	/** Get all registered tools from loaded extensions. */
	getTools(): RegisteredTool[] {
		return this.tools.map((entry) => entry.tool);
	}

	/** Get all registered commands from loaded extensions. */
	getCommands(): RegisteredCommand[] {
		return this.commands.map((entry) => entry.command);
	}

	hasHandlers(event: ExtensionEventType): boolean {
		return (this.handlers.get(event)?.length ?? 0) > 0;
	}

	async emit(event: ExtensionEvent): Promise<unknown | undefined> {
		const list = this.handlers.get(event.type) ?? [];
		let firstResult: unknown | undefined;
		for (const { handler, ctx } of list) {
			try {
				const result = await handler(event, ctx);
				if (firstResult === undefined && result !== undefined) {
					firstResult = result;
				}
			} catch {
				// Swallow extension errors so one extension cannot break the runner.
			}
		}
		return firstResult;
	}

	/**
	 * Emit a structured typed event to extension handlers.
	 * Bridges to the legacy ExtensionEvent format for backward-compatible handlers.
	 */
	async emitTyped<T extends ExtensionEventName>(
		event: Extract<{ type: T }, any>,
	): Promise<unknown> {
		// First, notify typed event bus
		await this.typedBus.emit(event as any);

		// Bridge to legacy event format for backward-compatible handlers
		const legacyEventType = this.mapToLegacyEvent(event.type);
		if (!legacyEventType) return undefined;

		const ctx: ExtensionEventContext = {
			sessionId: this.options.sessionId,
			cwd: this.options.cwd,
			...(event as any).toolName ? { tool_name: (event as any).toolName } : {},
			...(event as any).toolCallId ? { tool_call_id: (event as any).toolCallId } : {},
			...(event as any).turnIndex !== undefined ? { turn_index: (event as any).turnIndex } : {},
		};

		const legacyEvent: ExtensionEvent = {
			type: legacyEventType,
			context: ctx,
			...(event as any),
		};

		return this.emit(legacyEvent);
	}

	/** Map typed event name to legacy ExtensionEventType. */
	private mapToLegacyEvent(type: ExtensionEventName): ExtensionEventType | null {
		const mapping: Record<string, ExtensionEventType> = {
			"before_agent_start": "user_prompt_submit",
			"agent_end": "agent_end",
			"turn_start": "turn_start",
			"turn_end": "turn_end",
			"message_start": "message_start",
			"message_update": "message_update",
			"message_end": "message_end",
			"tool_execution_start": "tool_call_start",
			"tool_execution_end": "tool_call_end",
			"session_before_switch": "session_start",
			"session_before_compact": "before_compact",
			"session_compact": "after_compact",
			"session_shutdown": "session_end",
			"before_provider_request": "agent_start",
			"after_provider_response": "agent_end",
		};
		return mapping[type] ?? null;
	}

	/** Access the structured typed event bus for direct extension subscriptions. */
	get typedEvents(): ExtensionEventBus {
		return this.typedBus;
	}

	/** Access the shared extension context. */
	getExtensionContext(): ReturnType<typeof createExtensionContext> {
		return this.extContext;
	}

	/** Update the shared extension context with current run state. */
	updateContext(updates: Partial<ReturnType<typeof createExtensionContext>>): void {
		Object.assign(this.extContext, updates);
	}

	/** Get hooks to wire into the agent runner. */
	getHooks(): AgentHooks | undefined {
		if (this.handlers.size === 0) return undefined;

		const hooks: AgentHooks = {};

		hooks.beforeToolCall = async ({ toolCall, args }) => {
			const ctx: ExtensionEventContext = {
				sessionId: this.options.sessionId,
				cwd: this.options.cwd,
				tool_name: toolCall.name,
				tool_input: args,
			};
			const event: ExtensionEvent = { type: "before_tool_call", context: ctx };
			const result = await this.emit(event);
			if (result && typeof result === "object") {
				const hookResult = result as BeforeToolCallExtensionResult;
				if (hookResult.block) {
					return {
						content: hookResult.reason || hookResult.content || "Blocked by extension",
						isError: true,
					};
				}
				if (hookResult.content !== undefined || hookResult.args !== undefined) {
					return {
						content: hookResult.content,
						args: hookResult.args,
						isError: hookResult.isError,
					};
				}
			}
			return undefined;
		};

		hooks.afterToolCall = async ({ toolCall, result, isError }) => {
			const ctx: ExtensionEventContext = {
				sessionId: this.options.sessionId,
				cwd: this.options.cwd,
				tool_name: toolCall.name,
				tool_result: result,
				is_error: isError,
			};
			const event: ExtensionEvent = { type: "after_tool_call", context: ctx };
			const emitted = await this.emit(event);
			if (emitted && typeof emitted === "object") {
				const hookResult = emitted as AfterToolCallExtensionResult;
				if (
					hookResult.content !== undefined ||
					hookResult.isError !== undefined ||
					hookResult.terminate !== undefined
				) {
					return hookResult;
				}
			}
			return undefined;
		};

		return hooks;
	}

	/** Clean up loaded extensions. */
	destroy(): void {
		for (const ext of this.extensions) {
			ext.unload?.();
		}
		this.handlers.clear();
		this.tools.length = 0;
		this.commands.length = 0;
		this.eventBus.clear();
		this.typedBus.clear();
		this.extContext = createExtensionContext();
	}
}
