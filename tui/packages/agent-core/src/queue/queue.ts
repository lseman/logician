// ── Message queue ─────────────────────────────────────────────────────────────
// Queue for steering and follow-up messages submitted during agent execution.
//
// Steering messages (Enter): delivered after the current turn finishes tool
//   calls but before the next provider request.
// Follow-up messages (Alt+Enter): delivered only after the agent finishes all
//   work (idle state).
//
// Escape aborts and restores queued messages to the editor.

export type MessageType = "steering" | "followUp";

export interface QueuedMessage {
	id: string;
	type: MessageType;
	content: string;
	timestamp: number;
}

export class MessageQueue {
	private queue: QueuedMessage[] = [];
	private nextId = 0;

	/** Submit a steering message (Enter). Delivered after current turn. */
	steering(content: string): QueuedMessage {
		const msg: QueuedMessage = {
			id: `msg_${Date.now()}_${this.nextId++}`,
			type: "steering",
			content,
			timestamp: Date.now(),
		};
		this.queue.push(msg);
		return msg;
	}

	/** Submit a follow-up message (Alt+Enter). Delivered after agent finishes. */
	followUp(content: string): QueuedMessage {
		const msg: QueuedMessage = {
			id: `msg_${Date.now()}_${this.nextId++}`,
			type: "followUp",
			content,
			timestamp: Date.now(),
		};
		this.queue.push(msg);
		return msg;
	}

	/** Get all queued messages, ordered by submission time. */
	getAll(): QueuedMessage[] {
		return [...this.queue];
	}

	/** Get steering messages. */
	getSteering(): QueuedMessage[] {
		return this.queue.filter((m) => m.type === "steering");
	}

	/** Get follow-up messages. */
	getFollowUp(): QueuedMessage[] {
		return this.queue.filter((m) => m.type === "followUp");
	}

	/** Peek at the next steering message without removing it. */
	peekSteering(): QueuedMessage | undefined {
		return this.queue.find((m) => m.type === "steering");
	}

	/** Remove and return all steering messages. */
	dequeueSteering(): QueuedMessage[] {
		const steering = this.queue.filter((m) => m.type === "steering");
		this.queue = this.queue.filter((m) => m.type !== "steering");
		return steering;
	}

	/** Remove and return the oldest steering message. */
	dequeueOneSteering(): QueuedMessage[] {
		const index = this.queue.findIndex((message) => message.type === "steering");
		if (index < 0) return [];
		return this.queue.splice(index, 1);
	}

	/** Remove and return all follow-up messages. */
	dequeueFollowUp(): QueuedMessage[] {
		const followUp = this.queue.filter((m) => m.type === "followUp");
		this.queue = this.queue.filter((m) => m.type !== "followUp");
		return followUp;
	}

	/** Remove and return the oldest follow-up message. */
	dequeueOneFollowUp(): QueuedMessage[] {
		const index = this.queue.findIndex((message) => message.type === "followUp");
		if (index < 0) return [];
		return this.queue.splice(index, 1);
	}

	remove(id: string): QueuedMessage | undefined {
		const index = this.queue.findIndex((message) => message.id === id);
		if (index < 0) return undefined;
		return this.queue.splice(index, 1)[0];
	}

	/** Clear all queued messages (used on abort). */
	clear(): QueuedMessage[] {
		const all = [...this.queue];
		this.queue = [];
		return all;
	}

	/** Check if there are any queued messages. */
	hasMessages(): boolean {
		return this.queue.length > 0;
	}

	/** Count of queued messages. */
	get size(): number {
		return this.queue.length;
	}
}
