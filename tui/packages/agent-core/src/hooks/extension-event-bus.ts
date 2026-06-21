// ── Typed extension event bus ──────────────────────────────────────────
// Emits ExtensionEvent to registered handlers per event type. Supports
// per-handler timeouts, error isolation, and structured result merging.
//
// Usage:
//   const bus = new ExtensionEventBus({ onError, defaultTimeoutMs: 5000 });
//   const off = bus.on("turn_start", (event) => { /* handle */ });
//   await bus.emit({ type: "turn_start", turnIndex: 5 });
//   off(); // unsubscribe

import type {
	ExtensionEvent,
	ExtensionEventName,
	ExtensionEventResult,
	ExtensionEventHandler,
	ExtensionErrorHandler,
} from "./extension-events.ts";

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

	constructor(options: ExtensionEventBusOptions = {}) {
		this.onError = options.onError ?? this.defaultOnError;
		this.defaultTimeoutMs = options.defaultTimeoutMs ?? 0;
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
	onMultiple(registrations: Array<{
		eventType: ExtensionEventName;
		handler: ExtensionEventHandler<ExtensionEventName>;
		timeoutMs?: number;
	}>): () => void {
		const unsubscribes = registrations.map((r) =>
			this.on(r.eventType, r.handler, { timeoutMs: r.timeoutMs }),
		);
		return () => unsubscribes.forEach((u) => u());
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
		return (results.length > 0 ? results[results.length - 1] : undefined) as ExtensionEventResult<T>;
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

	// ── Internals ───────────────────────────────────────────────────────

	private async guardHandler<T extends ExtensionEventName>(
		handler: ExtensionEventHandler<ExtensionEventName>,
		event: Extract<ExtensionEvent, { type: T }>,
		timeoutMs?: number,
	): Promise<ExtensionEventResult<T> | undefined> {
		const effectiveTimeout = timeoutMs ?? this.defaultTimeoutMs;
		try {
			const run = Promise.resolve(handler(event as ExtensionEvent));
			const result = effectiveTimeout > 0
				? await this.withTimeout(run, effectiveTimeout)
				: await run;
			return result as ExtensionEventResult<T>;
		} catch (error) {
			this.onError(error as Error, event.type);
			return undefined;
		}
	}

	private defaultOnError(error: Error, event: ExtensionEventName): void {
		console.error(`[ExtensionEventBus] Handler error (${event}):`, error.message);
	}

	private async withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => reject(new Error(`Extension handler timeout after ${ms}ms`)), ms);
			promise.then(
				(value) => { clearTimeout(timer); resolve(value); },
				(error) => { clearTimeout(timer); reject(error); },
			);
		});
	}
}
