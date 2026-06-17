// ── Extension runner ─────────────────────────────────────────────────────────
// Loads and executes extensions, collecting registered tools and commands.
// Wires extension event handlers into the agent loop via the hook bus.

import type { AgentLoopHooks } from "../core/types.ts";
import type {
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

export class ExtensionRunner {
	private handlers = new Map<string, ExtensionEventHandler[]>();
	private tools: RegisteredTool[] = [];
	private commands: RegisteredCommand[] = [];
	private extensions: Array<{ def: ExtensionDefinition; unload?: () => void }> = [];
	private eventBus: EventBus;

	constructor(private readonly options: ExtensionRunnerOptions) {
		this.eventBus = createEventBus();
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
		const handlers = new Map<string, ExtensionEventHandler[]>();
		const state = createStateWrapper(def.name);
		const ctx: ExtensionContext = {
			ui: noopUI,
			state,
			cwd: this.options.cwd,
			sessionId: this.options.sessionId,
		};

		const on = (event: ExtensionEventType, handler: ExtensionEventHandler): (() => void) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			// Return unsubscribe function
			return () => {
				const current = handlers.get(event) ?? [];
				const filtered = current.filter((h) => h !== handler);
				if (filtered.length === 0) {
					handlers.delete(event);
				} else {
					handlers.set(event, filtered);
				}
			};
		};

		const registerTool = (tool: RegisteredTool): void => {
			this.tools.push(tool);
		};

		const registerCommand = (command: RegisteredCommand): void => {
			this.commands.push(command);
		};

		const emit = async (event: ExtensionEvent): Promise<void> => {
			const list = handlers.get(event.type) ?? [];
			for (const handler of list) {
				try {
					await handler(event, ctx);
				} catch {
					// Swallow extension errors — don't break the agent loop
				}
			}
		};

		const unload = (): void => {
			this.tools = this.tools.filter((t) => t.name !== def.name);
			this.commands = this.commands.filter((c) => c.name !== def.name);
		};

		return { on, registerTool, registerCommand, emit, events: this.eventBus, info: { name: def.name, path: def.path }, unload };
	}

	/** Get all registered tools from loaded extensions. */
	getTools(): RegisteredTool[] {
		return this.tools;
	}

	/** Get all registered commands from loaded extensions. */
	getCommands(): RegisteredCommand[] {
		return this.commands;
	}

	/** Get hooks to wire into the agent loop. */
	getHooks(): AgentLoopHooks | undefined {
		if (this.handlers.size === 0) return undefined;

		const hooks: AgentLoopHooks = {};

		hooks.beforeToolCall = async ({ toolCall, args }) => {
			const ctx: ExtensionEventContext = {
				sessionId: this.options.sessionId,
				cwd: this.options.cwd,
				tool_name: toolCall.name,
				tool_input: args,
			};
			const event: ExtensionEvent = { type: "before_tool_call", context: ctx };
			const list = this.handlers.get("before_tool_call") ?? [];
			for (const handler of list) {
				try {
					const result = await handler(event, this.makeCtx());
					if (result && typeof result === "object" && "block" in result && (result as { block: boolean }).block) {
						return {
							content: (result as { reason?: string }).reason || "Blocked by extension",
							isError: true,
						};
					}
				} catch {
					// skip failing handlers
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
			const list = this.handlers.get("after_tool_call") ?? [];
			for (const handler of list) {
				try {
					await handler(event, this.makeCtx());
				} catch {
					// skip failing handlers
				}
			}
			return undefined;
		};

		return hooks;
	}

	private makeCtx(): ExtensionContext {
		return {
			ui: noopUI,
			state: createStateWrapper("global"),
			cwd: this.options.cwd,
			sessionId: this.options.sessionId,
		};
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
	}
}
