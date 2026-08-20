// ── Extension runner ─────────────────────────────────────────────────────────
// Loads and executes extensions, collecting registered tools and commands.
// Wires extension event handlers into the agent runner via the hook bus.
// Exposes both an untyped EventBus for cross-extension messaging and a
// structured typed event bus for agent lifecycle events.

import type { AgentHooks } from "../types/index.ts";
import { createExtensionContext } from "./context.ts";
import { createEventBus, type EventBus } from "./event-bus.ts";
import type { PiRuntimePort } from "../../adapters/pi/index.ts";
import { PiAdapter } from "../../adapters/pi/index.ts";
import { createExtensionState } from "./state.ts";
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
	piRuntime?: PiRuntimePort;
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
	private piAdapters: PiAdapter[] = [];
	private eventBus: EventBus;
	/** Shared context for extension event handlers */
	private extContext: ReturnType<typeof createExtensionContext>;
	/** Session ID and cwd for adapter context */
	private adapterSessionId = "";
	private adapterCwd = "";

	constructor(private readonly options: ExtensionRunnerOptions) {
		this.eventBus = createEventBus();
		this.extContext = createExtensionContext();
		this.adapterSessionId = options.sessionId;
		this.adapterCwd = options.cwd;
	}

	get events(): EventBus {
		return this.eventBus;
	}

	/**
	 * Detect if an extension is a Pi-style extension by inspecting its source.
	 * Checks for TypeBox imports and @earendil-works/pi-coding-agent imports.
	 */
	private isPiExtension(content: string): boolean {
		return (
			content.includes("@earendil-works/pi-coding-agent") ||
			content.includes("typebox") ||
			content.includes("Type.Object") ||
			content.includes("Type.String") ||
			content.includes("Type.Number") ||
			content.includes("Type.Boolean") ||
			content.includes("Type.Array")
		);
	}

	/**
	 * Extract file content from a path, handling file:// URLs.
	 */
	private async readExtensionSource(path: string): Promise<string | null> {
		try {
			const fs = await import("node:fs/promises");
			// Handle file:// URLs (from pathToFileURL)
			const cleanPath = path.startsWith("file://")
				? path.replace(/^file:\/\//, "")
				: path;
			return await fs.readFile(cleanPath, "utf-8");
		} catch {
			return null;
		}
	}

	/**
	 * Load and execute a set of extensions from definitions.
	 * Each extension module is expected to export a default function that
	 * receives an ExtensionAPI instance.
	 *
	 * Pi-style extensions are auto-detected and loaded through the PiAdapter.
	 */
	async load(definitions: ExtensionDefinition[]): Promise<void> {
		for (const def of definitions) {
			try {
				// Read file content to detect Pi extensions (best-effort)
				const content = await this.readExtensionSource(def.path);

				if (
					def.compatibility === "pi" ||
					(def.compatibility !== "native" &&
						content &&
						this.isPiExtension(content))
				) {
					// Load as Pi extension through the adapter
					await this.loadPiExtension(def, content);
				} else {
					// Load as native Logician extension
					await this.loadNativeExtension(def);
				}
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.error(
					`[logician] failed to load extension "${def.name}": ${message}`,
				);
			}
		}
	}

	/**
	 * Load a Pi extension through the PiAdapter.
	 */
	private async loadPiExtension(
		def: ExtensionDefinition,
		_content: string | null,
	): Promise<void> {
		// Give the adapter the same live registration interface native extensions
		// use. Pi contributions then enter the actual tool/command registries at
		// factory time instead of being stranded in adapter-local bookkeeping.
		const logicianApi = this.createAPI(def);
		const adapter = new PiAdapter(
			logicianApi,
			{
				ui: noopUI,
				state: {
					get: async () => undefined,
					set: async () => {},
					delete: async () => {},
					keys: async () => [],
				},
				cwd: this.adapterCwd,
				sessionId: this.adapterSessionId,
			},
			{
				sessionId: this.adapterSessionId,
				cwd: this.adapterCwd,
				runtime: this.options.piRuntime,
			},
		);

		try {
			const mod = await import(
				/* @vite-ignore */ /* @webpackIgnore: true */ def.path
			);
			const factory = mod.default as ((api: any) => void) | undefined;
			if (!factory) {
				console.warn(
					`[logician] pi-extension "${def.name}" has no default export`,
				);
				return;
			}

			const piApi = adapter.getApi();
			factory(piApi);

			this.piAdapters.push(adapter);
			this.extensions.push({ def, unload: logicianApi.unload });
			console.log(
				`[logician] loaded Pi extension "${def.name}" with ${adapter.getRegisteredTools().length} tool(s) and ${adapter.getRegisteredCommands().length} command(s)`,
			);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			console.error(
				`[logician] failed to load Pi extension "${def.name}": ${message}`,
			);
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

	/**
	 * Set the Logician API reference for Pi adapters.
	 * Called once when the runner is fully initialized.
	 */
	setLogicianApi(_api: ExtensionAPI): void {
		// Update all Pi adapters with the real API
		for (const _adapter of this.piAdapters) {
			// Re-create with real API (adapters store reference, not value)
			// For now, just log — full re-binding requires a more sophisticated approach
			console.debug(
				`[logician] Pi extension context bound to Logician API (session: ${this.adapterSessionId})`,
			);
		}
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
		return (
			(this.handlers.get(event)?.length ?? 0) > 0 ||
			this.piAdapters.some(adapter => adapter.hasHandlers(event))
		);
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
	getHooks(): AgentHooks | undefined {
		if (this.handlers.size === 0 && this.piAdapters.length === 0)
			return undefined;

		const hooks: AgentHooks = {};

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
			let piArgs = args;
			for (const adapter of this.piAdapters) {
				const piResult = await adapter.emitToolCall({
					toolCallId: toolCall.id,
					toolName: toolCall.name,
					input: piArgs,
				});
				piArgs = piResult.input;
				if (piResult.block) {
					return {
						content: piResult.reason ?? "Blocked by extension",
						isError: true,
					};
				}
			}
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
			return piArgs !== args ? { args: piArgs } : undefined;
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
			let piResult: {
				toolCallId: string;
				toolName: string;
				input: Record<string, unknown>;
				content: Array<{ type: string; text: string }>;
				isError: boolean;
				details?: Record<string, unknown>;
			} = {
				toolCallId: toolCall.id,
				toolName: toolCall.name,
				input: JSON.parse(toolCall.arguments || "{}") as Record<
					string,
					unknown
				>,
				content: [{ type: "text", text: result }],
				isError,
			};
			for (const adapter of this.piAdapters) {
				piResult = await adapter.emitToolResult(piResult);
			}
			const content = piResult.content.map(part => part.text).join("\n");
			if (
				content !== result ||
				piResult.isError !== isError ||
				piResult.details
			) {
				return {
					content,
					isError: piResult.isError,
					details: piResult.details,
				};
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
		this.piAdapters.length = 0;
	}

	/**
	 * Emit a Logician event to all registered systems — both native extensions
	 * and Pi extensions via the adapter.
	 */
	async emitToAll(event: ExtensionEvent): Promise<unknown | undefined> {
		const nativeResult = await this.emit(event);
		let combinedResult = nativeResult;

		// Also emit to all Pi adapters
		for (const adapter of this.piAdapters) {
			const result = await adapter.emitFromLogician(event);
			if (result.messages || result.systemPrompt) {
				combinedResult = {
					...(typeof combinedResult === "object" && combinedResult !== null
						? combinedResult
						: {}),
					...result,
				};
			}
		}

		return combinedResult;
	}

	/** Get the number of loaded Pi extensions. */
	getPiExtensionCount(): number {
		return this.piAdapters.length;
	}

	/** Get all Pi-registered tools from adapters. */
	getPiTools(): Array<{ name: string; description: string }> {
		const tools: Array<{ name: string; description: string }> = [];
		for (const adapter of this.piAdapters) {
			for (const tool of adapter.getRegisteredTools()) {
				tools.push({ name: tool.name, description: tool.description });
			}
		}
		return tools;
	}

	/** Get all Pi-registered commands from adapters. */
	getPiCommands(): Array<{ name: string; description?: string }> {
		const commands: Array<{ name: string; description?: string }> = [];
		for (const adapter of this.piAdapters) {
			for (const cmd of adapter.getRegisteredCommands()) {
				commands.push({ name: cmd.name, description: cmd.description });
			}
		}
		return commands;
	}

	/**
	 * Emit an event that can return handler results (messages, systemPrompt).
	 * Used for context / before_agent_start events where Pi extensions return
	 * modifications that the harness needs to apply.
	 */
	async emitWithContext(
		eventType: ExtensionEventType,
		context: ExtensionEventContext,
	): Promise<{ messages?: unknown[]; systemPrompt?: string } | undefined> {
		if (this.piAdapters.length === 0) return undefined;

		const event: ExtensionEvent = {
			type: eventType,
			context,
		};

		// Collect return values from all Pi adapters
		let mergedMessages: unknown[] | undefined;
		let mergedSystemPrompt: string | undefined;

		for (const adapter of this.piAdapters) {
			const result = await adapter.emitFromLogician(event as any);
			if (result?.messages) mergedMessages = result.messages;
			if (result?.systemPrompt) mergedSystemPrompt = result.systemPrompt;
		}

		if (mergedMessages || mergedSystemPrompt) {
			return { messages: mergedMessages, systemPrompt: mergedSystemPrompt };
		}
		return undefined;
	}

	/**
	 * Emit a Pi input event to all loaded Pi adapters.
	 * Call from the input controller before processing user input.
	 * @returns {action: 'continue'|'transform'|'handled', text?, images?} from the first non-null handler.
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
		for (const adapter of this.piAdapters) {
			const result = await adapter.emitInputEvent(text, images, source);
			if (result) return result;
		}
		return null; // default: continue
	}

	/**
	 * Emit a Pi user_bash event to all loaded Pi adapters.
	 * Call from the bash execution layer before running.
	 * @returns {action: 'continue'|'intercept'|'replace', result?, operations?} from the first non-null handler.
	 */
	async emitUserBashEvent(
		command: string,
		excludeFromContext: boolean = false,
	): Promise<{
		action: "continue" | "intercept" | "replace";
		result?: { output: string; exitCode: number; cancelled: boolean };
		operations?: unknown;
	} | null> {
		for (const adapter of this.piAdapters) {
			const result = await adapter.emitUserBashEvent(
				command,
				excludeFromContext,
			);
			if (result) return result;
		}
		return null; // default: continue
	}

	/**
	 * Emit a Pi project_trust event to all loaded Pi adapters.
	 * Call from the trust prompt before showing the overlay.
	 * @returns {trusted: 'yes'|'no'|'undecided', remember?} from the first non-null handler.
	 */
	async emitProjectTrustEvent(cwd: string): Promise<{
		trusted: "yes" | "no" | "undecided";
		remember?: boolean;
	} | null> {
		for (const adapter of this.piAdapters) {
			const result = await adapter.emitProjectTrustEvent(cwd);
			if (result) return result;
		}
		return null; // default: continue (let Logician handle it)
	}
}
