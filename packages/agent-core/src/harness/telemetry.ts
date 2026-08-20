// ── Event bus for cross-extension communication ──────────────────────────
// Replaces extension/event-bus.ts — simple pub/sub for telemetry and events.

export interface EventBus {
	on(
		channel: string,
		handler: (data: unknown) => void | Promise<void>,
	): () => void;
	emit(channel: string, data?: unknown): void;
	clear(): void;
}

export function createEventBus(): EventBus {
	const listeners = new Map<string, Set<(data: unknown) => void>>();

	return {
		on(channel, handler) {
			const safeHandler = (data: unknown) => {
				try {
					Promise.resolve(handler(data)).catch(err => {
						console.error(`[telemetry] event handler error (${channel}):`, err);
					});
				} catch (err) {
					console.error(`[telemetry] event handler error (${channel}):`, err);
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
		},
		emit(channel, data) {
			const list = listeners.get(channel);
			if (!list) return;
			for (const handler of list) {
				handler(data);
			}
		},
		clear() {
			listeners.clear();
		},
	};
}
