import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
} from "../adapters/claude-code/hook-layer.ts";

void test("Claude hook matcher names cover Logician-native tools", () => {
	assert.equal(claudeToolMatcherName("bash"), "Bash");
	assert.equal(claudeToolMatcherName("read_file"), "Read");
	assert.equal(claudeToolMatcherName("spawn_agent"), "Agent");
	assert.equal(claudeToolMatcherName("ask_user"), "AskUserQuestion");
	assert.equal(
		claudeToolMatcherName("mcp__plugin_context_mode__ctx_execute"),
		"mcp__plugin_context_mode__ctx_execute",
	);
});

void test("non-blocking pre-tool guidance survives until the tool result", async () => {
	const seenMatchers: string[] = [];
	const layer = createClaudeCodeHookLayer({
		enabled: true,
		sessionId: "session",
		transcriptPath: "/tmp/transcript.jsonl",
		cwd: "/tmp",
		getMatcherValue: claudeToolMatcherName,
		runHookEvent: async (event, payload) => {
			if (event === "PreToolUse") {
				seenMatchers.push(String(payload?.matcher_value));
				return {
					additional_contexts: [
						"Use ctx_execute_file when analyzing a file instead of editing it.",
					],
				};
			}
			return {};
		},
	});
	const toolCall = {
		id: "read-1",
		name: "read_file",
		arguments: '{"path":"large.log"}',
	};

	const before = await layer.hooks?.beforeToolCall?.({
		toolCall,
		args: { path: "large.log" },
		iteration: 1,
	});
	assert.equal(before, undefined);

	const after = await layer.hooks?.afterToolCall?.({
		toolCall,
		args: { path: "large.log" },
		result: "file contents",
		isError: false,
		iteration: 1,
	});

	assert.deepEqual(seenMatchers, ["Read"]);
	assert.match(after?.content ?? "", /^file contents/);
	assert.match(after?.content ?? "", /ctx_execute_file/);
});

void test("blocking hook guidance prevents the original tool call", async () => {
	const layer = createClaudeCodeHookLayer({
		enabled: true,
		sessionId: "session",
		transcriptPath: "/tmp/transcript.jsonl",
		cwd: "/tmp",
		getMatcherValue: claudeToolMatcherName,
		runHookEvent: async () => ({
			permission_decision: "deny",
			permission_reason: "Route this through ctx_batch_execute.",
		}),
	});

	const result = await layer.hooks?.beforeToolCall?.({
		toolCall: { id: "bash-1", name: "bash", arguments: "{}" },
		args: { command: "find . -type f" },
		iteration: 1,
	});

	assert.equal(result?.isError, true);
	assert.match(result?.content ?? "", /ctx_batch_execute/);
});
