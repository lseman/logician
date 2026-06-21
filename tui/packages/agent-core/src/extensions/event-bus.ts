// ── Event bus ────────────────────────────────────────────────────────────────
// Simple pub/sub for extension communication.
// All loaded extensions share the same event bus instance, enabling cross-extension
// messaging. Handler errors are caught and logged so one failing handler doesn't
// block others.

export interface EventBus {
	/** Subscribe to events on a channel. Returns unsubscribe function. */
	on(channel: string, handler: (data: unknown) => void | Promise<void>): () => void;

	/** Emit an event on a channel. */
	emit(channel: string, data?: unknown): void;

	/** Clear all listeners. */
	clear(): void;
}

export function createEventBus(): EventBus {
	const listeners = new Map<string, Set<(data: unknown) => void>>();

	const on = (channel: string, handler: (data: unknown) => void | Promise<void>): (() => void) => {
		const safeHandler = (data: unknown) => {
			try {
				Promise.resolve(handler(data)).catch((err) => {
					console.error(`[logician] event bus handler error (${channel}):`, err);
				});
			} catch (err) {
				console.error(`[logician] event bus handler error (${channel}):`, err);
			}
		};

		const list = listeners.get(channel) ?? new Set();
		list.add(safeHandler);
		listeners.set(channel, list);

		return () => {
			const current = listeners.get(channel);
			if (current) {
				current.delete(safeHandler);
				if (current.size === 0) listeners.delete(channel);
			}
		};
	};

	const emit = (channel: string, data?: unknown): void => {
		const list = listeners.get(channel);
		if (!list) return;
		for (const handler of list) {
			handler(data);
		}
	};

	const clear = (): void => {
		listeners.clear();
	};

	return { on, emit, clear };
}
