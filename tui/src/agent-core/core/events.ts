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
				// A listener throwing is a consumer bug, not a loop failure — never
				// abort the emit. Don't use console.* here: in a TUI it corrupts the
				// rendered frame. Write to stderr only when stdout is not a TTY (tests,
				// piped runs); otherwise drop silently.
				if (!process.stdout.isTTY) {
					process.stderr.write(`Event handler error: ${String(e)}\n`);
				}
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
