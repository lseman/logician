import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

/** Direct-mode /spawn: tool_end arrives before subagent_lifecycle events. */
void test("direct-mode /spawn: lifecycle summary written after tool_end closes chunk", () => {
	const transcript = new Transcript();
	// turn_start creates synthetic assistant-only turn (no user message)
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });

	// Tool execution completes before any lifecycle events
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		result: "Agent done in 2 turns",
	});

	// Lifecycle events arrive after tool_end — this is the bug case
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "explorer-1",
		agent: "explorer",
		task: "Inspect workspace",
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "explorer-1",
		agent: "explorer",
		result: "done in 2 turns",
		isError: false,
		turns: 2,
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	assert.ok(assistant, "assistant message should exist");
	assert.equal(assistant.isComplete, false); // not closed by turn_end yet

	const toolChunk = assistant.chunks.find(
		c => c.tool?.tool_name === "spawn_agent",
	);
	assert.ok(toolChunk?.tool, "tool chunk should exist");

	// The lifecycle summary must be written even though tool_end came first
	assert.equal(toolChunk.tool.details?.status, "completed");
	assert.equal(toolChunk.tool.details?.lifecycleSummary, "done in 2 turn(s)");
	assert.equal(toolChunk.tool.isComplete, true);
	// Only one spawn card — no leftover "running" twin.
	assert.equal(
		assistant.chunks.filter(c => c.tool?.tool_name === "spawn_agent").length,
		1,
	);
});

/** Normal direct /spawn order: start → lifecycle → end → turn_end. */
void test("direct-mode /spawn: lifecycle end marks tool done before tool_end", () => {
	const transcript = new Transcript();
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-direct" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		tool_args: { task: "list files", agent: "general" },
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "agent_1",
		agent: "general",
		task: "list files",
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "agent_1",
		agent: "general",
		result: "file a, file b",
		isError: false,
		turns: 1,
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.ok(tool);
	assert.equal(tool.isComplete, true);
	assert.equal(tool.details?.status, "completed");
	assert.equal(tool.details?.lifecycleSummary, "done in 1 turn(s)");

	// tool_end should reuse the same card and prefer metrics duration
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		result: "file a, file b",
		details: {
			agent: "general",
			status: "completed",
			metrics: { durationMs: 1500, turns: 1 },
		},
	});
	assert.equal(tool.durationMs, 1500);
	assert.equal(tool.result, "file a, file b");
	assert.equal(
		transcript
			.getTurns()[0]
			?.assistantMessage?.chunks.filter(
				c => c.tool?.tool_name === "spawn_agent",
			).length,
		1,
	);

	transcript.handleEvent({
		type: "turn_end",
		turn_id: "turn-direct",
		message: "",
	});
	assert.equal(transcript.getTurns()[0]?.isComplete, true);
});

/** spawn_agents batch: lifecycle arrives after all tasks end. */
void test("direct-mode /spawn-test: batch lifecycle captured after tool_end", () => {
	const transcript = new Transcript();
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-2" });

	// Tool ends first
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agents",
		tool_name: "spawn_agents",
		tool_call_id: "batch",
		result: "",
		details: { total: 2, completed: 2 },
	});

	// Lifecycle events for individual tasks arrive after
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "general-1",
		agent: "general",
		task: "Task 1",
		taskIndex: 0,
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "general-1",
		agent: "general",
		result: "done",
		isError: false,
		taskIndex: 0,
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "reviewer-1",
		agent: "reviewer",
		task: "Task 2",
		taskIndex: 1,
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "reviewer-1",
		agent: "reviewer",
		result: "something failed",
		isError: true,
		taskIndex: 1,
	});

	const batch = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.ok(batch);
	const statusMap = batch.details?.taskStatus as Record<string, unknown>;
	assert.ok(statusMap, "taskStatus should exist on batch details");
	assert.equal((statusMap["0"] as { status: string }).status, "completed");
	assert.equal((statusMap["1"] as { status: string }).status, "failed");
});

/** /spawn after addTurn must keep tool_start + stream on the same turn. */
void test("direct-mode /spawn: stream and result stay on the user command turn", () => {
	const transcript = new Transcript();
	// TUI addTurn happens before spawnAgentDirectly after the slash-popup fix
	transcript.addTurn("/spawn list files");
	transcript.handleEvent({ type: "turn_start", turn_id: "spawn_turn" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		tool_args: { task: "list files", agent: "general" },
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "agent_1",
		agent: "general",
		task: "list files",
	});
	transcript.handleEvent({
		type: "tool_execution_update",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		partial_result: "Listing files…\n",
		update_kind: "output",
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "agent_1",
		seq: 1,
		kind: "content",
		delta: "Listing files…\n",
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "agent_1",
		seq: 2,
		kind: "tool_start",
		toolCallId: "tc1",
		toolName: "bash",
		args: '{"command":"ls"}',
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "agent_1",
		seq: 3,
		kind: "tool_end",
		toolCallId: "tc1",
		toolName: "bash",
		result: "a.md\nb.md",
		isError: false,
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "agent_1",
		agent: "general",
		result: "Two markdown files.",
		isError: false,
		turns: 2,
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		result: "Two markdown files.",
		details: {
			agent: "general",
			status: "completed",
			metrics: { turns: 2, durationMs: 900, toolCalls: 1 },
		},
	});
	transcript.handleEvent({
		type: "turn_end",
		turn_id: "spawn_turn",
		message: "",
	});

	assert.equal(transcript.getTurns().length, 1, "single turn");
	const turn = transcript.getTurns()[0];
	assert.equal(turn.userMessage?.content, "/spawn list files");
	const tools = transcript.getAssistantTools(turn);
	assert.equal(tools.length, 1, "single spawn card");
	const tool = tools[0];
	assert.equal(tool.args?.task, "list files");
	assert.equal(tool.result, "Two markdown files.");
	assert.equal(tool.details?.streamTranscript, "Listing files…\n");
	const metrics = tool.details?.metrics as { turns?: number } | undefined;
	assert.equal(metrics?.turns, 2);
	const chunks = tool.details?.childChunks as Array<{ type: string }>;
	assert.ok(Array.isArray(chunks) && chunks.length >= 2);
	assert.ok(chunks.some(c => c.type === "tool"));
	assert.ok(chunks.some(c => c.type === "content"));
});

/** Dual tool_call_* + tool_execution_* events must not double child tools. */
void test("child tool_call and tool_execution events dedupe by toolCallId", () => {
	const transcript = new Transcript();
	transcript.handleEvent({ type: "turn_start", turn_id: "t1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "spawn_1",
		tool_args: { task: "list", agent: "general" },
	});
	// Streaming start
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "a1",
		seq: 1,
		kind: "tool_start",
		toolCallId: "tc_bash",
		toolName: "bash",
		args: '{"command":"ls"}',
	});
	// Execution start (same id) — must not create a second row
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "a1",
		seq: 2,
		kind: "tool_start",
		toolCallId: "tc_bash",
		toolName: "bash",
		args: JSON.stringify({ command: "ls" }),
	});
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "a1",
		seq: 3,
		kind: "tool_end",
		toolCallId: "tc_bash",
		toolName: "bash",
		result: "a.md",
		isError: false,
	});
	// Second end (tool_execution_end) — must update, not duplicate
	transcript.handleEvent({
		type: "subagent_chunk",
		agentId: "a1",
		seq: 4,
		kind: "tool_end",
		toolCallId: "tc_bash",
		toolName: "bash",
		result: "a.md\nb.md",
		isError: false,
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	const chunks = (tool.details?.childChunks ?? []) as Array<{
		type: string;
		tool?: { toolCallId?: string; resultPreview?: string; status?: string };
	}>;
	const tools = chunks.filter(c => c.type === "tool");
	assert.equal(tools.length, 1, "exactly one child tool row");
	assert.equal(tools[0].tool?.toolCallId, "tc_bash");
	assert.equal(tools[0].tool?.status, "completed");
	assert.equal(tools[0].tool?.resultPreview, "a.md\nb.md");
});

/** turn_start must not rebind onto a completed prior turn. */
void test("turn_start without pending turn opens a fresh synthetic turn", () => {
	const transcript = new Transcript();
	transcript.addTurn("old prompt");
	transcript.handleEvent({ type: "turn_start", turn_id: "old" });
	transcript.handleEvent({ type: "turn_end", turn_id: "old", message: "" });
	assert.equal(transcript.getTurns()[0]?.isComplete, true);

	transcript.handleEvent({ type: "turn_start", turn_id: "spawn_fresh" });
	assert.equal(transcript.getTurns().length, 2);
	assert.equal(transcript.getTurns()[1]?.id, "spawn_fresh");
	assert.equal(transcript.getTurns()[1]?.isComplete, false);
});

/** Normal flow still works: lifecycle before tool_end. */
void test("normal flow: lifecycle before tool_end still captures summary", () => {
	const transcript = new Transcript();
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-3" });

	// Tool start creates the chunk first (as agent-loop does)
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect files" },
	});

	// Lifecycle events arrive before tool_end
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "start",
		agentId: "explorer-1",
		agent: "explorer",
		task: "Inspect files",
	});
	transcript.handleEvent({
		type: "subagent_lifecycle",
		phase: "end",
		agentId: "explorer-1",
		agent: "explorer",
		result: "done in 3 turns",
		isError: false,
		turns: 3,
	});

	// Tool ends after
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		result: "done",
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.equal(tool?.details?.status, "completed");
	assert.equal(tool?.details?.lifecycleSummary, "done in 3 turn(s)");
});
