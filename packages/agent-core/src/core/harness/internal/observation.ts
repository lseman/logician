import type { HarnessPhase } from "../../state/runtime-state.ts";
import type { AgentEvent } from "../../types/index.ts";
import type { HarnessObserver, HarnessQueues } from "../types.ts";

/** Owns harness observation fan-out independently of turn orchestration. */
export class HarnessObservation {
	private readonly observers = new Set<HarnessObserver>();

	constructor(initial: HarnessObserver[] = []) {
		for (const observer of initial) this.observers.add(observer);
	}

	observe(observer: HarnessObserver): () => void {
		this.observers.add(observer);
		return () => this.observers.delete(observer);
	}

	event(event: AgentEvent): void {
		for (const observer of this.observers) observer.event?.(event);
	}

	phase(phase: HarnessPhase, previous: HarnessPhase): void {
		if (phase === previous) return;
		for (const observer of this.observers) {
			observer.phaseChange?.(phase, previous);
		}
	}

	settled(nextTurnCount: number): void {
		for (const observer of this.observers) observer.settled?.(nextTurnCount);
	}

	queue(queues: HarnessQueues): void {
		for (const observer of this.observers) observer.queueChange?.(queues);
	}
}
