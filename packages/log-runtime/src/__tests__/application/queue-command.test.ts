import { expect, test } from "bun:test";
import { runQueueCommand } from "../../runtime/bridge/queue-command.ts";

function target(overrides: Record<string, unknown> = {}) {
	return {
		flushSteeringNow: () => 0,
		getQueues: () => ({ steering: [], followUp: [], nextTurn: [] }),
		clearQueues: () => ({ steering: [], followUp: [], nextTurn: [] }),
		dropQueuedMessage: () => undefined,
		...overrides,
	};
}

test("queue commands remain runtime policy over core queue primitives", () => {
	expect(
		runQueueCommand(
			target({
				getQueues: () => ({
					steering: ["focus"],
					followUp: ["verify"],
					nextTurn: [],
				}),
			}),
			"/queue",
		),
	).toEqual({ text: "1. ▸ focus\n2. ↳ verify", level: "info" });
	expect(runQueueCommand(target(), "/unknown")).toBeNull();
});

test("queue mutation commands report their outcome", () => {
	expect(
		runQueueCommand(target({ flushSteeringNow: () => 2 }), "/steer-now"),
	).toEqual({
		text: "Processing 2 queued steering messages now.",
		level: "info",
	});
	expect(
		runQueueCommand(
			target({
				dropQueuedMessage: (index: number) =>
					index === 1 ? "later" : undefined,
			}),
			"/queue-drop 2",
		),
	).toEqual({ text: "Removed: later", level: "info" });
});
