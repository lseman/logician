// ── Steering/follow-up/next-turn queue operations for AgentHarness ───────────
// Pure(ish) helpers operating on explicit state passed in by the harness.
// The harness owns the mutable fields (msgManager, nextTurnQueue, session,
// abortController) and supplies them here; these functions contain the logic
// that used to live directly on the AgentHarness class.

import { createSteeringInterruptReason } from "../agent-loop-runner.ts";
import type { MessageDeliveryManager, DeliveryMode } from "../../queue/manager.ts";
import type { Session } from "../session.ts";
import type { HarnessPhase } from "../runtime-state.ts";
import type { QueueMode } from "../types.ts";
import { HarnessBusyError } from "./phase.ts";
import type { AbortResult, HarnessQueues } from "./contracts.ts";

export interface QueueOpsDeps {
	msgManager: MessageDeliveryManager;
	getPhase: () => HarnessPhase;
	getNextTurnQueue: () => string[];
	setNextTurnQueue: (queue: string[]) => void;
	emitQueueChange: () => void;
}

export function getQueues(deps: QueueOpsDeps): HarnessQueues {
	const q = deps.msgManager.queue;
	return {
		steering: q.getSteering().map((m) => m.content),
		followUp: q.getFollowUp().map((m) => m.content),
		nextTurn: [...deps.getNextTurnQueue()],
	};
}

export function clearQueues(deps: QueueOpsDeps): HarnessQueues {
	const cleared = getQueues(deps);
	deps.msgManager.queue.clear();
	deps.setNextTurnQueue([]);
	deps.emitQueueChange();
	return cleared;
}

/** Queue steering text for the current turn. Requires phase === "turn". */
export function steer(
	deps: QueueOpsDeps,
	text: string,
	steeringInterrupt: boolean | undefined,
	abortController: AbortController | null,
): void {
	if (deps.getPhase() !== "turn") {
		throw new HarnessBusyError("steer", deps.getPhase(), "turn");
	}
	deps.msgManager.queue.steering(text);
	deps.emitQueueChange();
	if (steeringInterrupt) {
		abortController?.abort(createSteeringInterruptReason());
	}
}

/** Promote queued steering into the immediate next turn and interrupt the current step. */
export function flushSteeringNow(
	deps: QueueOpsDeps,
	abortController: AbortController | null,
): number {
	if (deps.getPhase() !== "turn") {
		throw new HarnessBusyError("flush steering", deps.getPhase(), "turn");
	}
	const queued = deps.msgManager.queue.dequeueSteering();
	if (queued.length === 0) return 0;
	deps.setNextTurnQueue([
		...deps.getNextTurnQueue(),
		...queued.map((message) => message.content),
	]);
	deps.emitQueueChange();
	abortController?.abort(createSteeringInterruptReason());
	return queued.length;
}

export function dropQueuedMessage(
	deps: QueueOpsDeps,
	displayIndex: number,
): string | undefined {
	const queue = deps.msgManager.queue;
	const displayed = [...queue.getSteering(), ...queue.getFollowUp()];
	const target = displayed[displayIndex];
	if (!target) return undefined;
	const removed = queue.remove(target.id);
	if (removed) deps.emitQueueChange();
	return removed?.content;
}

export function followUp(deps: QueueOpsDeps, text: string): void {
	deps.msgManager.queue.followUp(text);
	deps.emitQueueChange();
}

export function nextTurn(deps: QueueOpsDeps, text: string): void {
	deps.setNextTurnQueue([...deps.getNextTurnQueue(), text]);
	deps.emitQueueChange();
}

export function setSteeringMode(deps: QueueOpsDeps, mode: QueueMode): void {
	deps.msgManager.setMode("steering", mode as DeliveryMode);
}

export function getSteeringMode(deps: QueueOpsDeps): QueueMode {
	return deps.msgManager.getMode("steering") as QueueMode;
}

export function setFollowUpMode(deps: QueueOpsDeps, mode: QueueMode): void {
	deps.msgManager.setMode("followUp", mode as DeliveryMode);
}

export function getFollowUpMode(deps: QueueOpsDeps): QueueMode {
	return deps.msgManager.getMode("followUp") as QueueMode;
}

export interface AbortDeps extends QueueOpsDeps {
	abortController: AbortController | null;
	activeOperationId: string | undefined;
	session: Session | undefined;
	setAbortRequested: () => void;
	waitForIdle: () => Promise<void>;
	emitAbortEvent: (result: AbortResult) => void;
	emitSessionEnd: (reason: string) => Promise<void>;
}

export async function abort(deps: AbortDeps): Promise<AbortResult> {
	const q = deps.msgManager.queue;
	const clearedSteering = q.getSteering().map((m) => m.content);
	const clearedFollowUp = q.getFollowUp().map((m) => m.content);
	if (deps.activeOperationId) {
		deps.session?.appendJournalEvent({
			type: "operation_interrupted",
			operationId: deps.activeOperationId,
			status: "aborted",
		});
	}
	deps.setAbortRequested();
	deps.abortController?.abort();
	deps.msgManager.queue.clear();
	deps.emitQueueChange();
	await deps.waitForIdle();
	const result: AbortResult = {
		clearedSteering,
		clearedFollowUp,
		clearedNextTurn: [],
	};
	deps.emitAbortEvent(result);
	await deps.emitSessionEnd("abort");
	return result;
}
