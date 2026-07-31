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
		(c) => c.tool?.tool_name === "spawn_agent",
	);
	assert.ok(toolChunk?.tool, "tool chunk should exist");

	// The lifecycle summary must be written even though tool_end came first
	assert.equal(toolChunk.tool.details?.status, "completed");
	assert.equal(
		toolChunk.tool.details?.lifecycleSummary,
		"done in 2 turn(s)",
	);
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
