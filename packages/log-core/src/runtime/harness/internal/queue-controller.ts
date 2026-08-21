import type { QueueMode } from "../../../system/types/types-config.ts";
import {
	type DeliveryMode,
	MessageQueue,
	type QueuedMessage,
} from "../queue/message-queue.ts";
import type { AbortResult, HarnessQueues } from "../types.ts";

export class HarnessQueueController {
	private readonly queue: MessageQueue;

	constructor(
		options: { steeringMode?: QueueMode; followUpMode?: QueueMode },
		private readonly changed: (queues: HarnessQueues) => void,
	) {
		this.queue = new MessageQueue({
			steeringMode: options.steeringMode as DeliveryMode | undefined,
			followUpMode: options.followUpMode as DeliveryMode | undefined,
		});
	}

	afterTurn(): QueuedMessage[] {
		const messages = this.queue.afterTurn();
		if (messages.length) this.publish();
		return messages;
	}

	onIdle(): QueuedMessage[] {
		const messages = this.queue.onIdle();
		if (messages.length) this.publish();
		return messages;
	}

	dequeueNextTurn(): QueuedMessage[] {
		const messages = this.queue.dequeueNextTurn();
		if (messages.length) this.publish();
		return messages;
	}

	clearCurrentTurn(): void {
		this.queue.clearCurrentTurn();
		this.publish();
	}

	steer(text: string, interrupt: boolean, abort: () => void): void {
		if (interrupt) {
			this.queue.nextTurn(text);
			this.publish();
			abort();
			return;
		}
		this.queue.steering(text);
		this.publish();
	}

	flushSteering(abort: () => void): number {
		const queued = this.queue.dequeueSteering();
		for (const message of queued) this.queue.nextTurn(message.content);
		if (queued.length) {
			this.publish();
			abort();
		}
		return queued.length;
	}

	drop(displayIndex: number): string | undefined {
		const target = [...this.queue.getSteering(), ...this.queue.getFollowUp()][
			displayIndex
		];
		if (!target) return undefined;
		const removed = this.queue.remove(target.id);
		if (removed) this.publish();
		return removed?.content;
	}

	followUp(text: string): void {
		this.queue.followUp(text);
		this.publish();
	}

	nextTurn(text: string): void {
		this.queue.nextTurn(text);
		this.publish();
	}

	abortSnapshot(): AbortResult {
		const queues = this.snapshot();
		this.queue.clearCurrentTurn();
		this.publish();
		return {
			clearedSteering: queues.steering,
			clearedFollowUp: queues.followUp,
			clearedNextTurn: [],
		};
	}

	snapshot(): HarnessQueues {
		return {
			steering: this.queue.getSteering().map(message => message.content),
			followUp: this.queue.getFollowUp().map(message => message.content),
			nextTurn: this.queue.getNextTurn().map(message => message.content),
		};
	}

	clear(): HarnessQueues {
		const queues = this.snapshot();
		this.queue.clear();
		this.publish();
		return queues;
	}

	setMode(type: "steering" | "followUp", mode: QueueMode): void {
		this.queue.setMode(type, mode as DeliveryMode);
	}

	getMode(type: "steering" | "followUp"): QueueMode {
		return this.queue.getMode(type) as QueueMode;
	}

	private publish(): void {
		this.changed(this.snapshot());
	}
}
