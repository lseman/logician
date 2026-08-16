// ── Typed extension event bus ──────────────────────────────────────────
// Emits ExtensionEvent to registered handlers per event type. Supports
// per-handler timeouts, error isolation, and structured result merging.
//
// Can be used standalone OR wired to a HookBus as an observer layer:
//   const hookBus = new HookBus();
//   const extBus = ExtensionEventBus.fromHookBus(hookBus);
//   // extBus now auto-emits typed events from every HookBus event.
//
// Standalone usage:
//   const bus = new ExtensionEventBus({ onError, defaultTimeoutMs: 5000 });
//   const off = bus.on("turn_start", (event) => { /* handle */ });
//   await bus.emit({ type: "turn_start", turnIndex: 5 });
//   off(); // unsubscribe

import type { HookBus, HookEventName } from "../native/hook-bus.ts";
import type {
	ExtensionErrorHandler,
	ExtensionEvent,
	ExtensionEventHandler,
	ExtensionEventName,
	ExtensionEventResult,
} from "./events.ts";

export interface ExtensionEventBusOptions {
	/** Called when a handler throws. Default: log to console.error. */
	onError?: ExtensionErrorHandler;
	/** Default per-handler timeout. 0 = no timeout. */
	defaultTimeoutMs?: number;
}

interface Registration {
	handler: ExtensionEventHandler<ExtensionEventName>;
	timeoutMs?: number;
}

export class ExtensionEventBus {
	private handlers: Map<ExtensionEventName, Registration[]> = new Map();
	private onError: ExtensionErrorHandler;
	private defaultTimeoutMs: number;
	private _hookBusObserver: (() => void) | null = null;

	constructor(options: ExtensionEventBusOptions = {}) {
		this.onError = options.onError ?? this.defaultOnError;
		this.defaultTimeoutMs = options.defaultTimeoutMs ?? 0;
	}

	/**
	 * Factory: creates an ExtensionEventBus that emits typed events from a
	 * HookBus's observer firehose. The bus still accepts direct .on() registrations
	 * in addition to the auto-emitted typed events.
	 *
	 * The returned bus owns the HookBus subscription; call dispose() to clean up.
	 */
	static fromHookBus(
		hookBus: HookBus,
		options: ExtensionEventBusOptions = {},
	): ExtensionEventBus {
		const bus = new ExtensionEventBus(options);
		bus._hookBusObserver = hookBus.observe((event, ctx) => {
			const typed = hookToExtensionEvent(event, ctx);
			if (typed) bus.emit(typed as any); // type-safe by construction
		});
		return bus;
	}

	/**
	 * Manually emit a legacy event that isn't typed in the ExtensionEvent contract.
	 * Used by builtin hooks for internal diagnostics (e.g. thinking_loop_detected).
	 */
	async emitLegacy(event: { type: string; [key: string]: unknown }): Promise<void> {
		const registrations =
			this.handlers.get(event.type as ExtensionEventName) ?? [];
		for (const { handler, timeoutMs } of registrations) {
			await this.guardHandler(
				handler,
				event as unknown as Extract<ExtensionEvent, { type: ExtensionEventName }>,
				timeoutMs,
			);
		}
	}

	/** Register a handler for a specific event type. Returns unsubscribe function. */
	on<T extends ExtensionEventName>(
		eventType: T,
		handler: ExtensionEventHandler<T>,
		options?: { timeoutMs?: number },
	): () => void {
		const registrations = this.handlers.get(eventType) ?? [];
		const registration: Registration = {
			handler: handler as unknown as ExtensionEventHandler<ExtensionEventName>,
			timeoutMs: options?.timeoutMs,
		};
		registrations.push(registration);
		this.handlers.set(eventType, registrations);

		return () => {
			const idx = registrations.indexOf(registration);
			if (idx >= 0) registrations.splice(idx, 1);
		};
	}

	/** Register handlers for multiple event types at once. Returns batch unsubscribe. */
	onMultiple(
		registrations: Array<{
			eventType: ExtensionEventName;
			handler: ExtensionEventHandler<ExtensionEventName>;
			timeoutMs?: number;
		}>,
	): () => void {
		const unsubscribes = registrations.map(r =>
			this.on(r.eventType, r.handler, { timeoutMs: r.timeoutMs }),
		);
		return () => unsubscribes.forEach(u => u());
	}

	/** Check if any handlers are registered for an event type. */
	hasHandlers(eventType: ExtensionEventName): boolean {
		return (this.handlers.get(eventType)?.length ?? 0) > 0;
	}

	/** Emit an event to all registered handlers. Returns merged results. */
	async emit<T extends ExtensionEventName>(
		event: Extract<ExtensionEvent, { type: T }>,
	): Promise<ExtensionEventResult<T>> {
		const registrations = this.handlers.get(event.type) ?? [];
		if (registrations.length === 0) return undefined as ExtensionEventResult<T>;

		const results: ExtensionEventResult<T>[] = [];
		for (const { handler, timeoutMs } of registrations) {
			const result = await this.guardHandler(handler, event, timeoutMs);
			if (result !== undefined) results.push(result);
		}
		// Last handler's result wins (like hook reducer semantics)
		return (
			results.length > 0 ? results[results.length - 1] : undefined
		) as ExtensionEventResult<T>;
	}



	/** Get all registered event types. */
	getRegisteredEvents(): ExtensionEventName[] {
		return Array.from(this.handlers.keys());
	}

	/** Get handler count for a specific event type. */
	getHandlerCount(eventType: ExtensionEventName): number {
		return this.handlers.get(eventType)?.length ?? 0;
	}

	/** Clear all handlers. */
	clear(): void {
		this.handlers.clear();
	}

	// ── Lifecycle ───────────────────────────────────────────────────────

	/**
	 * Clean up resources. When wired to a HookBus, unsubscribes the observer.
	 * When used standalone, clears all handlers.
	 */
	async dispose(): Promise<void> {
		if (this._hookBusObserver) {
			this._hookBusObserver();
			this._hookBusObserver = null;
		} else {
			this.clear();
		}
	}

	// ── Internals ───────────────────────────────────────────────────────

	private async guardHandler<T extends ExtensionEventName>(
		handler: ExtensionEventHandler<ExtensionEventName>,
		event: Extract<ExtensionEvent, { type: T }>,
		timeoutMs?: number,
	): Promise<ExtensionEventResult<T> | undefined> {
		const effectiveTimeout = timeoutMs ?? this.defaultTimeoutMs;
		try {
			const run = Promise.resolve(handler(event as ExtensionEvent));
			const result =
				effectiveTimeout > 0
					? await this.withTimeout(run, effectiveTimeout)
					: await run;
			return result as ExtensionEventResult<T>;
		} catch (error) {
			this.onError(error as Error, event.type);
			return undefined;
		}
	}

	private defaultOnError(error: Error, event: ExtensionEventName): void {
		console.error(
			`[ExtensionEventBus] Handler error (${event}):`,
			error.message,
		);
	}

	private async withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
		return new Promise((resolve, reject) => {
			const timer = setTimeout(
				() => reject(new Error(`Extension handler timeout after ${ms}ms`)),
				ms,
			);
			promise.then(
				value => {
					clearTimeout(timer);
					resolve(value);
				},
				error => {
					clearTimeout(timer);
					reject(error);
				},
			);
		});
	}
}

// ── Mapping: HookBus event → ExtensionEvent payload ─────────────────────
// Maps the 12 HookBus event types to typed ExtensionEvent payloads.
// Returns undefined when the HookBus event has no extension counterpart.

import type { Message, StopReason } from "../../agent/types.ts";

function hookToExtensionEvent(
	event: HookEventName,
	ctx: unknown,
): ExtensionEvent | undefined {
	switch (event) {
		case "beforeAgentStart": {
			const c = ctx as {
				prompt: string;
				systemPrompt: string;
				messages: Message[];
			};
			return { type: "before_agent_start", prompt: c.prompt, systemPrompt: c.systemPrompt };
		}
		case "beforeToolCall": {
			const c = ctx as {
				toolCall: { id: string; name: string; arguments: string };
				args: Record<string, unknown>;
				iteration: number;
			};
			return {
				type: "tool_execution_start",
				toolCallId: c.toolCall.id,
				toolName: c.toolCall.name,
				args: c.args,
			};
		}
		case "afterToolCall": {
			const c = ctx as {
				toolCall: { id: string; name: string };
				result: string;
				isError: boolean;
				iteration: number;
			};
			return {
				type: "tool_execution_end",
				toolCallId: c.toolCall.id,
				toolName: c.toolCall.name,
				result: c.result,
				isError: c.isError,
			};
		}
		case "prepareNextTurn":
		case "transformContext":
			// No direct extension counterpart — these transform messages in-place.
			return undefined;
		case "beforeProviderRequest": {
			const c = ctx as {
				model: string;
				sessionId: string;
				iteration: number;
				streamOptions: Record<string, unknown>;
			};
			return {
				type: "before_provider_request",
				model: c.model,
				sessionId: c.sessionId,
				iteration: c.iteration,
				streamOptions: { stream: true, ...c.streamOptions },
			};
		}
		case "beforeProviderPayload":
			// Internal only — no extension counterpart.
			return undefined;
		case "afterProviderResponse": {
			const c = ctx as {
				model: string;
				content: string;
				toolCallCount: number;
				stopReason: StopReason;
				usageTokens?: number;
				iteration: number;
			};
			return {
				type: "after_provider_response",
				model: c.model,
				content: c.content,
				toolCallCount: c.toolCallCount,
				stopReason: c.stopReason,
				usageTokens: c.usageTokens,
				iteration: c.iteration,
			};
		}
		case "shouldStopAfterTurn":
			// No extension counterpart — decision hook.
			return undefined;
		case "getSteeringMessages":
		case "getFollowUpMessages":
			return { type: "queue_update", steering: [], followUp: [] };
		case "beforeCompact": {
			const c = ctx as {
				tokensBefore: number;
				reason: "manual" | "auto";
			};
			return {
				type: "session_before_compact",
				tokensBefore: c.tokensBefore,
				reason: c.reason === "auto" ? "auto" : "manual",
			};
		}
		default:
			return undefined;
	}
}
