// ── Message queue ─────────────────────────────────────────────────────────────
// Queue for steering and follow-up messages submitted during agent execution.
//
// Steering messages (Enter): delivered after the current turn finishes tool
//   calls but before the next provider request.
// Follow-up messages (Alt+Enter): delivered only after the agent finishes all
//   work (idle state).
//
// Escape aborts and restores queued messages to the editor.

export type MessageType = "steering" | "followUp" | "nextTurn";
export type DeliveryMode = "one-at-a-time" | "all";

export interface MessageQueueOptions {
	steeringMode?: DeliveryMode;
	followUpMode?: DeliveryMode;
}

export interface QueuedMessage {
	id: string;
	type: MessageType;
	content: string;
	timestamp: number;
}

export class MessageQueue {
	private queue: QueuedMessage[] = [];
	private nextId = 0;
	private steeringMode: DeliveryMode;
	private followUpMode: DeliveryMode;

	constructor(options: MessageQueueOptions = {}) {
		this.steeringMode = options.steeringMode ?? "one-at-a-time";
		this.followUpMode = options.followUpMode ?? "one-at-a-time";
	}

	afterTurn(): QueuedMessage[] {
		return this.steeringMode === "all"
			? this.dequeueSteering()
			: this.dequeueOneSteering();
	}

	onIdle(): QueuedMessage[] {
		return [
			...(this.steeringMode === "all"
				? this.dequeueSteering()
				: this.dequeueOneSteering()),
			...(this.followUpMode === "all"
				? this.dequeueFollowUp()
				: this.dequeueOneFollowUp()),
		];
	}

	setMode(type: "steering" | "followUp", mode: DeliveryMode): void {
		if (type === "steering") this.steeringMode = mode;
		else this.followUpMode = mode;
	}

	getMode(type: "steering" | "followUp"): DeliveryMode {
		return type === "steering" ? this.steeringMode : this.followUpMode;
	}

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

	/** Queue guidance for the next user-initiated turn. */
	nextTurn(content: string): QueuedMessage {
		const msg: QueuedMessage = {
			id: `msg_${Date.now()}_${this.nextId++}`,
			type: "nextTurn",
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
		return this.queue.filter(m => m.type === "steering");
	}

	/** Get follow-up messages. */
	getFollowUp(): QueuedMessage[] {
		return this.queue.filter(m => m.type === "followUp");
	}

	getNextTurn(): QueuedMessage[] {
		return this.queue.filter(message => message.type === "nextTurn");
	}

	/** Peek at the next steering message without removing it. */
	peekSteering(): QueuedMessage | undefined {
		return this.queue.find(m => m.type === "steering");
	}

	/** Remove and return all steering messages. */
	dequeueSteering(): QueuedMessage[] {
		const steering = this.queue.filter(m => m.type === "steering");
		this.queue = this.queue.filter(m => m.type !== "steering");
		return steering;
	}

	/** Remove and return the oldest steering message. */
	dequeueOneSteering(): QueuedMessage[] {
		const index = this.queue.findIndex(message => message.type === "steering");
		if (index < 0) return [];
		return this.queue.splice(index, 1);
	}

	/** Remove and return all follow-up messages. */
	dequeueFollowUp(): QueuedMessage[] {
		const followUp = this.queue.filter(m => m.type === "followUp");
		this.queue = this.queue.filter(m => m.type !== "followUp");
		return followUp;
	}

	/** Remove and return the oldest follow-up message. */
	dequeueOneFollowUp(): QueuedMessage[] {
		const index = this.queue.findIndex(message => message.type === "followUp");
		if (index < 0) return [];
		return this.queue.splice(index, 1);
	}

	dequeueNextTurn(): QueuedMessage[] {
		const messages = this.getNextTurn();
		this.queue = this.queue.filter(message => message.type !== "nextTurn");
		return messages;
	}

	restore(snapshot: {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	}): void {
		this.queue = [];
		for (const content of snapshot.steering) this.steering(content);
		for (const content of snapshot.followUp) this.followUp(content);
		for (const content of snapshot.nextTurn) this.nextTurn(content);
	}

	remove(id: string): QueuedMessage | undefined {
		const index = this.queue.findIndex(message => message.id === id);
		if (index < 0) return undefined;
		return this.queue.splice(index, 1)[0];
	}

	/** Clear all queued messages (used on abort). */
	clear(): QueuedMessage[] {
		const all = [...this.queue];
		this.queue = [];
		return all;
	}

	/** Clear steering/follow-up messages while preserving next-turn guidance. */
	clearCurrentTurn(): QueuedMessage[] {
		const cleared = this.queue.filter(message => message.type !== "nextTurn");
		this.queue = this.queue.filter(message => message.type === "nextTurn");
		return cleared;
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
