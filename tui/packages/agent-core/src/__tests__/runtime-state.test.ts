import assert from "node:assert/strict";
import { test } from "node:test";
import { createRuntimeState, reduceRuntimeState } from "../core/runtime-state.ts";

void test("runtime reducer is immutable and deterministic", () => {
	const initial = createRuntimeState("turn");
	const started = reduceRuntimeState(
		initial,
		{ type: "agent_start", seq: 1, ts: 100 },
		"turn",
	);
	const turning = reduceRuntimeState(
		started,
		{ type: "turn_start", turnId: "turn_1", seq: 2, ts: 120 },
	);
	const ended = reduceRuntimeState(
		turning,
		{
			type: "turn_end",
			turnId: "turn_1",
			stopReason: "stop",
			toolResults: [],
			seq: 3,
			ts: 170,
		},
	);
	assert.notEqual(started, initial);
	assert.equal(initial.isStreaming, false);
	assert.equal(ended.lastTurnDurationMs, 50);
	assert.equal(ended.lastEventSeq, 3);
});
