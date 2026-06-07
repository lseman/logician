// ── Event system ──────────────────────────────────────────────────────────────────
// Mirrors Python AgentEvent + EventEmitter. TUI subscribes to events.

import type { AgentEvent, EventHandler } from "./types.ts";

export class EventEmitter {
	private listeners = new Set<EventHandler>();
	private history: AgentEvent[] = [];
	private maxHistory = 1000;

	on(handler: EventHandler): () => void {
		this.listeners.add(handler);
		return () => this.listeners.delete(handler);
	}

	emit(event: AgentEvent): void {
		this.history.push(event);
		if (this.history.length > this.maxHistory) {
			this.history = this.history.slice(-this.maxHistory);
		}
		for (const listener of this.listeners) {
			try {
				listener(event);
			} catch (e) {
				// eslint-disable-next-line no-console
				console.error("Event handler error:", e);
			}
		}
	}

	getHistory(): AgentEvent[] {
		return [...this.history];
	}

	clearHistory(): void {
		this.history = [];
	}
}

// ── Helper factories ─────────────────────────────────────────────────────────────

export function createEventEmitter(): EventEmitter {
	return new EventEmitter();
}
