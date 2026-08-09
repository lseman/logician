import assert from "node:assert/strict";
import { test } from "node:test";
import { STEERING_INTERRUPT_SUMMARY } from "@logician/agent-core";
import { mapAgentEvent } from "../runtime/event-mapping.ts";

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

void test("steering cancellation maps to one informational notice", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "run_outcome",
			status: "cancelled",
			summary: STEERING_INTERRUPT_SUMMARY,
			source: "runtime",
		}),
		{
			type: "notice",
			level: "info",
			label: "Steering",
			text: STEERING_INTERRUPT_SUMMARY,
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
