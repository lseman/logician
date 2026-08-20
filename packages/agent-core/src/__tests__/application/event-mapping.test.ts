import { test } from "bun:test";
import assert from "node:assert/strict";
import { mapAgentEvent } from "../../application/events/event-mapping.ts";
import { STEERING_INTERRUPT_SUMMARY } from "../../core/execution/agent-loop-runner.ts";

void test("context updates preserve unavailable provider telemetry", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "context_update",
			tokens: 1_000,
			maxTokens: 32_768,
			cachedTokens: null,
			promptTokens: null,
			completionTokens: null,
		}),
		{
			type: "context_update",
			tokens: 1_000,
			maxTokens: 32_768,
			compacted: undefined,
			cachedTokens: null,
			promptTokens: null,
			completionTokens: null,
		},
	);
});

void test("harness interventions map to one evidence-rich notice", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "harness_intervention",
			id: "intervention-1",
			kind: "loop",
			cause: "stagnation",
			action: "change_strategy",
			severity: "warning",
			detector: "output_guard",
			attempt: 2,
			evidence: { summary: "No verified progress" },
			nextAction: "Try another approach.",
			iteration: 4,
		}),
		{
			type: "notice",
			level: "warn",
			label: "loop: change_strategy",
			text: "No verified progress (attempt 2, incident intervention-1) Next: Try another approach.",
		},
	);
});

void test("agent_end produces no notice for normal outcomes", () => {
	assert.equal(
		mapAgentEvent({
			type: "agent_end",
			status: "completed",
			summary: "No more work remains.",
		}),
		null,
	);
});

void test("steering cancellation maps to one informational notice", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "agent_end",
			status: "cancelled",
			summary: STEERING_INTERRUPT_SUMMARY,
		}),
		{
			type: "notice",
			level: "info",
			label: "Steering",
			text: STEERING_INTERRUPT_SUMMARY,
		},
	);
});

void test("routine automatic permission allows stay out of the transcript", () => {
	for (const source of ["mode", "rule"] as const) {
		assert.equal(
			mapAgentEvent({
				type: "tool_permission_decision",
				toolName: "task_status",
				toolCallId: `call-${source}`,
				decision: "allow",
				source,
			}),
			null,
		);
	}
});

void test("permission denials and user-mediated approvals remain visible", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "tool_permission_decision",
			toolName: "bash",
			toolCallId: "call-denied",
			decision: "deny",
			source: "rule",
		}),
		{
			type: "notice",
			level: "warn",
			label: "Permission",
			text: "bash: deny (rule)",
		},
	);
	assert.deepEqual(
		mapAgentEvent({
			type: "tool_permission_decision",
			toolName: "write_file",
			toolCallId: "call-user",
			decision: "always",
			source: "user",
		}),
		{
			type: "notice",
			level: "info",
			label: "Permission",
			text: "write_file: always (user)",
		},
	);
});

void test("tool preparation and execution remain distinct lifecycle phases", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "tool_call_start",
			toolName: "read_file",
			toolCallId: "call-1",
			args: '{"path":"a.ts"}',
		}),
		{
			type: "tool_call_start",
			toolName: "read_file",
			toolCallId: "call-1",
			args: { path: "a.ts" },
		},
	);
	assert.deepEqual(
		mapAgentEvent({
			type: "tool_call_delta",
			toolCallId: "call-1",
			delta: '"}',
		}),
		{ type: "tool_call_update", toolCallId: "call-1", delta: '"}' },
	);
	assert.deepEqual(
		mapAgentEvent({
			type: "tool_call_id_update",
			previousToolCallId: "tool_0",
			toolCallId: "call-1",
		}),
		{
			type: "tool_call_id_update",
			previousToolCallId: "tool_0",
			toolCallId: "call-1",
		},
	);
	assert.equal(
		mapAgentEvent({
			type: "tool_call_end",
			toolName: "read_file",
			toolCallId: "call-1",
			result: "ok",
		}),
		null,
	);
});

void test("subagent tool_execution_start/end map into subagent_chunk activity", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "subagent_event",
			agentId: "agent_1",
			taskIndex: 0,
			event: {
				type: "tool_execution_start",
				toolCallId: "tc1",
				toolName: "bash",
				args: { command: "ls" },
				seq: 3,
			},
		}),
		{
			type: "subagent_chunk",
			agentId: "agent_1",
			seq: 3,
			kind: "tool_execution_start",
			toolCallId: "tc1",
			toolName: "bash",
			args: JSON.stringify({ command: "ls" }),
			taskIndex: 0,
		},
	);
	assert.deepEqual(
		mapAgentEvent({
			type: "subagent_event",
			agentId: "agent_1",
			event: {
				type: "tool_execution_end",
				toolCallId: "tc1",
				toolName: "bash",
				result: "ok",
				isError: false,
				seq: 4,
			},
		}),
		{
			type: "subagent_chunk",
			agentId: "agent_1",
			seq: 4,
			kind: "tool_execution_end",
			toolCallId: "tc1",
			toolName: "bash",
			result: "ok",
			isError: false,
			taskIndex: undefined,
		},
	);
});
