// ── Extension runner ─────────────────────────────────────────────────────────
// Loads and executes extensions, collecting registered tools and commands.
// Wires extension event handlers into the agent runner via the hook bus.
// Exposes both an untyped EventBus for cross-extension messaging and a
// structured typed event bus for agent lifecycle events.

import type { ExtensionHooks } from "../hooks/contracts.ts";
import { createExtensionContext } from "./context.ts";
import { createEventBus, type EventBus } from "./event-bus.ts";
import { createExtensionState } from "./state.ts";
import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionDefinition,
	ExtensionEvent,
	ExtensionEventHandler,
	ExtensionEventType,
	ExtensionState,
	RegisteredCommand,
	RegisteredTool,
} from "./types.ts";

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
			} catch (_e: unknown) {
				return raw as unknown as T;
			}
		},
		set: async (key: string, value: unknown): Promise<void> => {
			const serialized =
				typeof value === "string" ? value : JSON.stringify(value);
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
	private extensions: Array<{ def: ExtensionDefinition; unload?: () => void }> =
		[];
	private eventBus: EventBus;
	/** Shared context for extension event handlers */
	private extContext: ReturnType<typeof createExtensionContext>;

	constructor(private readonly options: ExtensionRunnerOptions) {
		this.eventBus = createEventBus();
		this.extContext = createExtensionContext();
	}

	get events(): EventBus {
		return this.eventBus;
	}

	/**
	 * Load and execute a set of extensions from definitions.
	 * Each extension module is expected to export a default function that
	 * receives an ExtensionAPI instance.
	 *
	 */
	async load(definitions: ExtensionDefinition[]): Promise<void> {
		for (const def of definitions) {
			try {
				await this.loadNativeExtension(def);
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.error(
					`[logician] failed to load extension "${def.name}": ${message}`,
				);
			}
		}
	}

	/**
	 * Load a native Logician extension.
	 */
	private async loadNativeExtension(def: ExtensionDefinition): Promise<void> {
		const mod = await import(
			/* @vite-ignore */ /* @webpackIgnore: true */ def.path
		);
		const factory = mod.default as ((api: ExtensionAPI) => void) | undefined;
		if (!factory) {
			console.warn(`[logician] extension "${def.name}" has no default export`);
			return;
		}

		const api = this.createAPI(def);
		factory(api);
		this.extensions.push({ def, unload: api.unload });
	}

	private createAPI(
		def: ExtensionDefinition,
	): ExtensionAPI & { unload?: () => void } {
		const ownedHandlers: HandlerEntry[] = [];
		const state = createStateWrapper(def.name);
		const ctx: ExtensionContext = {
			ui: noopUI,
			state,
			cwd: this.options.cwd,
			sessionId: this.options.sessionId,
		};

		const on = (
			event: ExtensionEventType,
			handler: ExtensionEventHandler,
		): (() => void) => {
			const entry: HandlerEntry = { handler, ctx, source: def.name };
			ownedHandlers.push(entry);
			const list = this.handlers.get(event) ?? [];
			list.push(entry);
			this.handlers.set(event, list);
			return () => {
				const current = this.handlers.get(event) ?? [];
				const filtered = current.filter(h => h !== entry);
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
			this.tools = this.tools.filter(t => t.source !== def.name);
			this.commands = this.commands.filter(c => c.source !== def.name);
			for (const [event, list] of this.handlers) {
				const filtered = list.filter(entry => entry.source !== def.name);
				if (filtered.length === 0) {
					this.handlers.delete(event);
				} else {
					this.handlers.set(event, filtered);
				}
			}
			ownedHandlers.length = 0;
		};

		return {
			on,
			registerTool,
			registerCommand,
			emit,
			events: this.eventBus,
			info: { name: def.name, path: def.path },
			unload,
		};
	}

	/** Get all registered tools from loaded extensions. */
	getTools(): RegisteredTool[] {
		return this.tools.map(entry => entry.tool);
	}

	/** Get all registered commands from loaded extensions. */
	getCommands(): RegisteredCommand[] {
		return this.commands.map(entry => entry.command);
	}

	async executeCommand(
		name: string,
		args: string,
	): Promise<string | undefined> {
		const command = this.commands.find(
			entry => entry.command.name.toLowerCase() === name.toLowerCase(),
		)?.command;
		if (!command) return undefined;
		return command.handler(args, {
			sessionId: this.options.sessionId,
			cwd: this.options.cwd,
			ui: noopUI,
		});
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
			} catch (_e: unknown) {
				// Swallow extension errors so one extension cannot break the runner.
				console.error("[extensions] handler error:", _e);
			}
		}
		return firstResult;
	}

	/** Access the shared extension context. */
	getExtensionContext(): ReturnType<typeof createExtensionContext> {
		return this.extContext;
	}

	/** Update the shared extension context with current run state. */
	updateContext(
		updates: Partial<ReturnType<typeof createExtensionContext>>,
	): void {
		Object.assign(this.extContext, updates);
	}

	/** Get hooks to wire into the agent runner. */
	getHooks(): ExtensionHooks | undefined {
		if (this.handlers.size === 0) return undefined;

		const hooks: ExtensionHooks = {};

		hooks.transformContext = async ({ messages, iteration }) => {
			if (!this.hasHandlers("context")) return undefined;
			const result = await this.emitToAll({
				type: "context",
				context: {
					sessionId: this.options.sessionId,
					cwd: this.options.cwd,
					messages: [...messages],
					iteration,
				},
			});
			if (result && typeof result === "object") {
				const transformed = (result as { messages?: unknown[] }).messages;
				if (Array.isArray(transformed))
					return { messages: transformed as typeof messages };
			}
			return undefined;
		};

		hooks.beforeToolCall = async ({ toolCall, args }) => {
			// Use typed event name: tool_execution_start
			const event: ExtensionEvent = {
				type: "tool_execution_start",
				context: {
					sessionId: this.options.sessionId,
					cwd: this.options.cwd,
					toolCallId: toolCall.id,
					tool_name: toolCall.name,
					tool_input: args,
				},
			} as unknown as ExtensionEvent;
			const result = await this.emit(event);
			if (result && typeof result === "object") {
				const hookResult = result as {
					block?: boolean;
					reason?: string;
					args?: Record<string, unknown>;
					content?: string;
					isError?: boolean;
				};
				if (hookResult.block) {
					return {
						content:
							hookResult.reason || hookResult.content || "Blocked by extension",
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
			// Use typed event name: tool_execution_end
			const event: ExtensionEvent = {
				type: "tool_execution_end",
				context: {
					sessionId: this.options.sessionId,
					cwd: this.options.cwd,
					toolCallId: toolCall.id,
					tool_name: toolCall.name,
					tool_input: JSON.parse(toolCall.arguments || "{}"),
					tool_result: result,
					is_error: isError,
				},
			} as unknown as ExtensionEvent;
			const emitted = await this.emit(event);
			if (emitted && typeof emitted === "object") {
				const hookResult = emitted as {
					content?: string;
					isError?: boolean;
					terminate?: boolean;
				};
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
		this.extContext = createExtensionContext();
	}

	/**
	 * Emit a Logician event to all registered native extensions.
	 */
	async emitToAll(event: ExtensionEvent): Promise<unknown | undefined> {
		return this.emit(event);
	}
}
