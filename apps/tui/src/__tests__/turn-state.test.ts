import { test } from "bun:test";
import assert from "node:assert/strict";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	beginPendingTurn,
	INITIAL_TURN_STATE,
	reduceTurnState,
} from "../state/turn-state.ts";

const event = (value: Record<string, unknown>): RuntimeEvent =>
	value as unknown as RuntimeEvent;

void test("an accepted submission stays thinking through pre-turn runtime events", () => {
	const pending = beginPendingTurn(INITIAL_TURN_STATE, 1);
	assert.equal(pending.phase, "thinking");
	assert.equal(pending.startedAt, 1);

	const afterMcpNotice = reduceTurnState(
		pending,
		event({ type: "notice", level: "info", label: "MCP", text: "Loaded" }),
		2,
	);
	assert.equal(afterMcpNotice.phase, "thinking");

	const afterLateStartupReady = reduceTurnState(
		afterMcpNotice,
		event({ type: "phase", state: "ready" }),
		3,
	);
	assert.equal(afterLateStartupReady.phase, "thinking");
	assert.equal(afterLateStartupReady.settledAt, undefined);
});

void test("turn state follows a complete tool lifecycle", () => {
	let state = reduceTurnState(
		INITIAL_TURN_STATE,
		event({
			type: "turn_start",
			turnId: "turn-1",
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
			toolName: "bash",
			args: { command: "npm test" },
		}),
		3,
	);
	assert.equal(state.phase, "verifying");
	state = reduceTurnState(
		state,
		event({
			type: "tool_execution_end",
			tool: "bash",
			toolName: "bash",
			isError: false,
		}),
		4,
	);
	assert.equal(state.phase, "thinking");
	state = reduceTurnState(
		state,
		event({ type: "turn_end", turnId: "turn-1" }),
		5,
	);
	assert.equal(state.phase, "complete");
	assert.equal(state.settledAt, 5);
});

void test("generic ready phases never settle an accepted or active turn", () => {
	const pending = beginPendingTurn(INITIAL_TURN_STATE, 1);
	assert.equal(
		reduceTurnState(pending, event({ type: "phase", state: "ready" }), 2).phase,
		"thinking",
	);

	const active = reduceTurnState(
		pending,
		event({ type: "turn_start", turnId: "turn-1" }),
		3,
	);
	const afterReady = reduceTurnState(
		active,
		event({ type: "phase", state: "ready" }),
		4,
	);
	assert.equal(afterReady.phase, "thinking");
	assert.equal(afterReady.settledAt, undefined);
});

void test("approval and failures are explicit states", () => {
	const started = reduceTurnState(
		INITIAL_TURN_STATE,
		event({
			type: "turn_start",
			turnId: "turn-1",
		}),
	);
	assert.equal(
		reduceTurnState(
			started,
			event({
				type: "permission_request",
				toolName: "bash",
				toolCallId: "1",
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
	assert.equal(
		reduceTurnState(
			started,
			event({
				type: "agent_error",
				message: "provider failed",
				phase: "model",
				recoverable: false,
			}),
		).phase,
		"failed",
	);
});

void test("a steerNow abort settles a mid-stream turn instead of leaving it streaming", () => {
	let state = reduceTurnState(
		INITIAL_TURN_STATE,
		event({ type: "turn_start", turnId: "turn-1" }),
		1,
	);
	state = reduceTurnState(state, event({ type: "token", token: "a" }), 2);
	assert.equal(state.phase, "streaming");

	// steerNow aborts the turn; the runner suppresses turn_end/agent_end for
	// this case and the mapping layer surfaces a "Steering" notice instead.
	state = reduceTurnState(
		state,
		event({
			type: "notice",
			level: "info",
			label: "Steering",
			text: "Steering the agent...",
		}),
		3,
	);
	assert.equal(state.phase, "complete");
	assert.equal(state.settledAt, 3);

	// An unrelated info notice must not be treated as terminal.
	const restarted = reduceTurnState(
		state,
		event({ type: "turn_start", turnId: "turn-2" }),
		4,
	);
	const streaming = reduceTurnState(
		restarted,
		event({ type: "token", token: "b" }),
		5,
	);
	const afterOtherNotice = reduceTurnState(
		streaming,
		event({ type: "notice", level: "info", label: "MCP", text: "Loaded" }),
		6,
	);
	assert.equal(afterOtherNotice.phase, "streaming");
});

void test("duplicate provider and execution starts count one running tool", () => {
	const started = reduceTurnState(
		INITIAL_TURN_STATE,
		event({ type: "turn_start", turnId: "turn-1" }),
	);
	const toolStart = event({
		type: "tool_execution_start",
		tool: "read_file",
		toolName: "read_file",
		toolCallId: "call-1",
		args: { path: "file.ts" },
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
			toolName: "read_file",
			toolCallId: "call-1",
			result: "done",
		}),
	);
	assert.equal(ended.runningTools, 0);
	assert.equal(ended.phase, "thinking");
});
