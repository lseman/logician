// ── Message queue manager ─────────────────────────────────────────────────────
// Manages message delivery semantics based on user configuration.
// Supports "one-at-a-time" (delivers one message then waits) or "all" (delivers
// all queued messages at once) modes for both steering and follow-up.

import { MessageQueue, type QueuedMessage } from "./queue.ts";

export type DeliveryMode = "one-at-a-time" | "all";

export interface MessageDeliveryOptions {
	steeringMode?: DeliveryMode;
	followUpMode?: DeliveryMode;
}

export class MessageDeliveryManager {
	private msgQueue: MessageQueue;
	private steeringMode: DeliveryMode;
	private followUpMode: DeliveryMode;

	constructor(options: MessageDeliveryOptions = {}) {
		this.msgQueue = new MessageQueue();
		this.steeringMode = options.steeringMode ?? "one-at-a-time";
		this.followUpMode = options.followUpMode ?? "one-at-a-time";
	}

	get queue(): MessageQueue {
		return this.msgQueue;
	}

	/**
	 * Called when the agent finishes a turn.
	 * Returns steering messages to deliver (if any).
	 * "one-at-a-time" mode: one message per drain; "all" drains everything.
	 */
	afterTurn(): QueuedMessage[] {
		if (this.steeringMode === "all") {
			return this.queue.dequeueSteering();
		}
		// one-at-a-time: take the oldest message
		const msgs = this.queue.dequeueSteering();
		return msgs.length > 0 ? [msgs[0]] : [];
	}

	/**
	 * Called when the loop would stop — drains steering + follow-up.
	 * Steering has priority (user guidance overrides follow-up).
	 * "one-at-a-time" mode: one message from each queue per call.
	 */
	onIdle(): QueuedMessage[] {
		const results: QueuedMessage[] = [];

		// Drain steering first (mutates the queue in-place).
		if (this.steeringMode === "all") {
			results.push(...this.queue.dequeueSteering());
		} else {
			const msgs = this.queue.dequeueSteering();
			if (msgs.length > 0) results.push(msgs[0]);
		}

		// Then drain follow-up.
		if (this.followUpMode === "all") {
			results.push(...this.queue.dequeueFollowUp());
		} else {
			const msgs = this.queue.dequeueFollowUp();
			if (msgs.length > 0) results.push(msgs[0]);
		}

		return results;
	}

	/**
	 * Update delivery mode at runtime.
	 */
	setMode(type: "steering" | "followUp", mode: DeliveryMode): void {
		if (type === "steering") {
			this.steeringMode = mode;
		} else {
			this.followUpMode = mode;
		}
	}

	getMode(type: "steering" | "followUp"): DeliveryMode {
		return type === "steering" ? this.steeringMode : this.followUpMode;
	}
}
