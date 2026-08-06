import assert from "node:assert/strict";
import { test } from "node:test";
import type { ParsedBridgeEvent } from "@logician/coding-agent/runtime";
import { INITIAL_TURN_STATE, reduceTurnState } from "../state/turn-state.ts";

const event = (value: Record<string, unknown>): ParsedBridgeEvent =>
	value as unknown as ParsedBridgeEvent;

void test("turn state follows a complete tool lifecycle", () => {
	let state = reduceTurnState(
		INITIAL_TURN_STATE,
		event({
			type: "turn_start",
			turn_id: "turn-1",
		}),
		1,
	);
	assert.equal(state.phase, "thinking");
	state = reduceTurnState(state, event({ type: "token", token: "a" }), 2);
	assert.equal(state.phase, "streaming");
	state = reduceTurnState(
		state,
		event({
			type: "tool_execution_start",
			tool: "bash",
			tool_name: "bash",
			tool_args: { command: "npm test" },
		}),
		3,
	);
	assert.equal(state.phase, "verifying");
	state = reduceTurnState(
		state,
		event({
			type: "tool_execution_end",
			tool: "bash",
			tool_name: "bash",
			is_error: false,
		}),
		4,
	);
	assert.equal(state.phase, "thinking");
	state = reduceTurnState(state, event({ type: "phase", state: "ready" }), 5);
	assert.equal(state.phase, "complete");
	assert.equal(state.settledAt, 5);
});

void test("approval and failures are explicit states", () => {
	const started = reduceTurnState(
		INITIAL_TURN_STATE,
		event({
			type: "turn_start",
			turn_id: "turn-1",
		}),
	);
	assert.equal(
		reduceTurnState(
			started,
			event({
				type: "permission_request",
				tool_name: "bash",
				tool_call_id: "1",
				args: {},
			}),
		).phase,
		"approval",
	);
	assert.equal(
		reduceTurnState(
			started,
			event({
				type: "notice",
				level: "error",
				label: "Error",
				text: "failed",
			}),
		).phase,
		"failed",
	);
});

void test("duplicate provider and execution starts count one running tool", () => {
	const started = reduceTurnState(
		INITIAL_TURN_STATE,
		event({ type: "turn_start", turn_id: "turn-1" }),
	);
	const toolStart = event({
		type: "tool_execution_start",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call-1",
		tool_args: { path: "file.ts" },
	});
	const first = reduceTurnState(started, toolStart);
	const duplicate = reduceTurnState(first, toolStart);
	assert.equal(duplicate.runningTools, 1);
	assert.deepEqual(duplicate.runningToolIds, ["call-1"]);

	const ended = reduceTurnState(
		duplicate,
		event({
			type: "tool_execution_end",
			tool: "read_file",
			tool_name: "read_file",
			tool_call_id: "call-1",
			result: "done",
		}),
	);
	assert.equal(ended.runningTools, 0);
	assert.equal(ended.phase, "thinking");
});
