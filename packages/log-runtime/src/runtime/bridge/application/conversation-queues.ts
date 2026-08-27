import type { QueueMode } from "@logician/log-core";
import type { AgentSession } from "@logician/log-core/session";

export interface ConversationQueueSnapshot {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}

const emptyQueueSnapshot = (): ConversationQueueSnapshot => ({
	steering: [],
	followUp: [],
	nextTurn: [],
});

/** Owns queue-specific delegation and empty-session defaults. */
export class ConversationQueues {
	constructor(private readonly session: () => AgentSession | null) {}

	steer(message: string): void {
		this.session()?.steer(message);
	}

	steerQueue(message: string): void {
		this.session()?.steerQueue(message);
	}

	steerNow(message: string): void {
		this.session()?.steerNow(message);
	}

	followUp(message: string): void {
		this.session()?.followUp(message);
	}

	nextTurn(message: string): void {
		this.session()?.nextTurn(message);
	}

	setSteeringMode(mode: QueueMode): void {
		this.session()?.setSteeringMode(mode);
	}

	setFollowUpMode(mode: QueueMode): void {
		this.session()?.setFollowUpMode(mode);
	}

	snapshot(): ConversationQueueSnapshot {
		return this.session()?.getQueues() ?? emptyQueueSnapshot();
	}

	flushSteeringNow(): number {
		return this.session()?.flushSteeringNow() ?? 0;
	}

	clear(): ConversationQueueSnapshot {
		return this.session()?.clearQueues() ?? emptyQueueSnapshot();
	}

	drop(displayIndex: number): string | undefined {
		return this.session()?.dropQueuedMessage(displayIndex);
	}
}
