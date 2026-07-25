import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

void test("transcript instances do not share turn state", () => {
	const first = new Transcript();
	first.addTurn("First session");

	const second = new Transcript();
	assert.equal(second.getTurns().length, 0);

	second.addTurn("Second session");
	first.clear();
	assert.equal(first.getTurns().length, 0);
	assert.equal(second.getTurns().length, 1);
});

void test("subagent tool notices become one integrated lifecycle entry", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: '▶ call-1 read_file path=src/index.ts',
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-1 read_file 120 lines",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	assert.ok(assistant);
	assert.equal(
		assistant.chunks.some((chunk) => chunk.type === "notice"),
		false,
		"child activity should not duplicate into top-level notices",
	);
	const calls = assistant.chunks.find(
		(chunk) => chunk.tool?.tool_name === "spawn_agent",
	)?.tool?.details?.childToolCalls as Array<Record<string, unknown>>;
	assert.deepEqual(calls, [
		{
			agentId: "explorer-1",
			toolCallId: "call-1",
			toolName: "read_file",
			args: "path=src/index.ts",
			status: "completed",
			isError: false,
			resultPreview: "120 lines",
		},
	]);
});

void test("concurrent same-name child tool calls resolve to the correct call by id", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	// Two concurrent calls to the same tool name, interleaved arrival: start A,
	// start B, end B, end A — a name-only match would wrongly resolve "end B"
	// against the first running "read_file" (call A) instead of call B.
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: "▶ call-A read_file path=a.ts",
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: "▶ call-B read_file path=b.ts",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-B read_file 40 lines",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-A read_file 120 lines",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	const calls = assistant?.chunks.find(
		(chunk) => chunk.tool?.tool_name === "spawn_agent",
	)?.tool?.details?.childToolCalls as Array<Record<string, unknown>>;

	const callA = calls.find((c) => c.toolCallId === "call-A");
	const callB = calls.find((c) => c.toolCallId === "call-B");
	assert.equal(callA?.status, "completed");
	assert.equal(callA?.resultPreview, "120 lines");
	assert.equal(callB?.status, "completed");
	assert.equal(callB?.resultPreview, "40 lines");
});

void test("multi-line child tool results do not break marker/id/name parsing", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: "▶ fU2Tm5kWJkVSygsPXRBBSvBAwPT5h25A list_files path=.",
	});
	// A real multi-line file listing result — previously the payload group
	// used `.*` which cannot cross newlines, so the whole regex failed to
	// match and fell back to raw text.slice(0, 40), leaking the marker and
	// tool_call_id straight into the rendered tool name.
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ fU2Tm5kWJkVSygsPXRBBSvBAwPT5h25A list_files a.ts\nb.ts\nc.ts",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	const calls = assistant?.chunks.find(
		(chunk) => chunk.tool?.tool_name === "spawn_agent",
	)?.tool?.details?.childToolCalls as Array<Record<string, unknown>>;

	assert.equal(calls.length, 1);
	assert.equal(calls[0].toolCallId, "fU2Tm5kWJkVSygsPXRBBSvBAwPT5h25A");
	assert.equal(calls[0].toolName, "list_files");
	assert.equal(calls[0].status, "completed");
	assert.equal(calls[0].resultPreview, "a.ts\nb.ts\nc.ts");
});

void test("subagent lifecycle notices update the parent card without duplication", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "Subagent explorer",
		text: "started: Inspect the workspace",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "Subagent explorer",
		text: "done in 2 turns",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	assert.ok(assistant);
	assert.equal(
		assistant.chunks.some((chunk) => chunk.type === "notice"),
		false,
	);
	const parent = assistant.chunks.find(
		(chunk) => chunk.tool?.tool_call_id === "parent-tool",
	)?.tool;
	assert.equal(parent?.details?.status, "completed");
	assert.equal(parent?.details?.lifecycleSummary, "done in 2 turns");
});

void test("final batch details preserve collected child tool activity", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate both tasks");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agents",
		tool_name: "spawn_agents",
		tool_call_id: "batch",
		tool_args: {
			tasks: [
				{ agent: "explorer", task: "Inspect files" },
				{ agent: "reviewer", task: "Review tests" },
			],
		},
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ agent-1",
		text: "▶ read_file path=src/index.ts",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agents",
		tool_name: "spawn_agents",
		tool_call_id: "batch",
		result: "",
		details: { total: 2, completed: 2, failed: 0, results: [] },
	});

	const batch = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	const activity = batch.details?.childToolCalls as Array<Record<string, unknown>>;
	assert.equal(activity.length, 1);
	assert.equal(activity[0].agentId, "agent-1");
	assert.equal(batch.details?.total, 2);
});

void test("completed subagents retain their live transcript", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate");
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "agent",
		tool_args: { agent: "general", task: "Implement it" },
	});
	transcript.handleEvent({
		type: "tool_execution_update",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "agent",
		update_kind: "output",
		partial_result: "Inspecting...\n```ts\nconst answer = 42;\n```\n",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "agent",
		result: "Implemented successfully.",
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.match(String(tool.details?.streamTranscript), /const answer = 42/);
	assert.equal(tool.streamOutput, undefined);
	assert.equal(tool.result, "Implemented successfully.");
});

void test("subagent chunks retain thinking, tool calls, and responses in order", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate");
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "agent",
		tool_args: { agent: "explorer", task: "Inspect it" },
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "explorer-1",
		seq: 1,
		kind: "thinking",
		delta: "I should inspect first.",
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "explorer-1",
		seq: 2,
		kind: "content",
		delta: "Inspecting now.",
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "explorer-1",
		seq: 3,
		kind: "tool_start",
		toolCallId: "read-1",
		toolName: "read_file",
		args: "{\"path\":\"src/index.ts\"}",
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "explorer-1",
		seq: 4,
		kind: "tool_end",
		toolCallId: "read-1",
		toolName: "read_file",
		result: "file contents",
		isError: false,
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "explorer-1",
		seq: 5,
		kind: "content",
		delta: "Inspection complete.",
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	const chunks = tool.details?.childChunks as Array<{
		type: string;
		contentText?: string;
		tool?: { toolName: string; resultPreview?: string };
	}>;
	assert.deepEqual(chunks.map((chunk) => chunk.type), [
		"thinking",
		"content",
		"tool",
		"content",
	]);
	assert.equal(chunks[0].contentText, "I should inspect first.");
	assert.equal(chunks[2].tool?.toolName, "read_file");
	assert.equal(chunks[2].tool?.resultPreview, "file contents");
	assert.equal(chunks[3].contentText, "Inspection complete.");
});
