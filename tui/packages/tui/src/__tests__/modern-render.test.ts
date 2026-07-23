import assert from "node:assert/strict";
import { test } from "node:test";
import type { Turn } from "@logician/coding-agent/transcript";
import { InputBar } from "../components/input-bar.ts";
import { StatusBar } from "../components/status-bar.ts";
import { TranscriptDisplay } from "../components/transcript-display.ts";
import { CURSOR_MARKER, visibleWidth } from "../layers/core/tui-core.ts";
import { initTheme } from "../layers/theme/theme.ts";

initTheme("dark");

const plain = (value: string): string =>
	value.replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");

void test("transcript renders clear speaker hierarchy and compact tool activity", () => {
	const display = new TranscriptDisplay();
	const turn: Turn = {
		id: "turn-1",
		userMessage: { type: "user", content: "Inspect the runtime bridge." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "content",
					contentText: "I found the startup issue.",
					isComplete: true,
				},
				{
					seq: 2,
					type: "tool",
					tool: {
						tool: "read_file",
						tool_name: "read_file",
						args: { path: "runtime/bridge.ts" },
						result: "ok",
						isError: false,
						isComplete: true,
						durationMs: 18,
					},
					isComplete: true,
				},
			],
		},
		isComplete: true,
	};
	display.setTurns([turn]);
	const lines = display.render(80);
	const output = plain(lines.join("\n"));

	assert.match(output, /╭─ YOU/);
	assert.match(output, /◆ LOGICIAN/);
	assert.match(output, /✓ read_file done/);
	assert.match(output, /18ms/);
	assert.ok(lines.every((line) => visibleWidth(line) <= 80));
});

void test("status bar drops optional sections instead of clipping ANSI text", () => {
	const status = new StatusBar();
	status.update({
		phase: "ready",
		model: "a-very-long-model-name",
		cwd: "/workspace/logician",
		branch: "feature/modern-ui",
		gitModified: 12,
		gitUntracked: 4,
		contextTokens: 45_000,
		contextMaxTokens: 150_000,
		thinkingLevel: "high",
		reasoner: "reflection",
	});

	for (const width of [32, 60, 120]) {
		const [line] = status.render(width);
		assert.ok(visibleWidth(line) <= width);
		assert.match(plain(line), /READY/);
		assert.doesNotMatch(plain(line), /…/);
	}
	assert.match(plain(status.render(120)[0]), /feature\/modern-ui/);
});

void test("input prompt has stable inset modern chrome", () => {
	const input = new InputBar();
	input.focused = true;
	const [header, line] = input.render(40);
	assert.match(plain(header), /MESSAGE/);
	assert.match(plain(header), /Enter send/);
	assert.match(plain(line).replace(CURSOR_MARKER, ""), /^  › Type a message/);
	assert.equal(visibleWidth(header), 40);
	assert.equal(visibleWidth(line), 40);
});

void test("input composer collapses to one line on narrow terminals", () => {
	const input = new InputBar();
	const lines = input.render(30);
	assert.equal(lines.length, 1);
	assert.equal(visibleWidth(lines[0]), 30);
});

void test("message editor accepts terminal arrow variants and batched movement", () => {
	const input = new InputBar();
	input.valueText = "abcd";
	input.handleInput("\x1b[1;2D");
	input.handleInput("X");
	assert.equal(input.valueText, "abcXd");

	input.valueText = "abcd";
	input.handleInput("\x1b[D\x1b[D");
	input.handleInput("X");
	assert.equal(input.valueText, "abXcd");
});

void test("up and down arrows navigate within multiline message text", () => {
	const input = new InputBar();
	input.valueText = "abcd\nxy";
	input.handleInput("\x1b[A");
	input.handleInput("X");
	assert.equal(input.valueText, "abXcd\nxy");

	input.handleInput("\x1b[B");
	input.handleInput("Z");
	assert.equal(input.valueText, "abXcd\nxyZ");
});

void test("expanded agent tools separate task arguments from live output", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([{
		id: "agent-turn",
		userMessage: { type: "user", content: "Audit the repository" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: false,
				tool: {
					tool: "spawn_agent",
					tool_name: "spawn_agent",
					args: { task: "Inspect architecture and tests", agent: "explorer" },
					partialResult: '{"task":"Inspect architecture and tests"}',
					streamOutput: "I am inspecting the core files now.",
					isError: false,
					isComplete: false,
				},
			}],
		},
		isComplete: false,
	}]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /agent explorer · running/);
	assert.match(output, /TASK/);
	assert.match(output, /Inspect architecture and tests/);
	assert.match(output, /LIVE PROGRESS/);
	assert.match(output, /I am inspecting the core files now/);
	assert.doesNotMatch(output, /\{"task":/);
});

void test("expanded agent progress is never character-truncated", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	const longProgress = `BEGIN-${"x".repeat(8_000)}-END`;
	display.setTurns([{
		id: "long-agent-output",
		userMessage: { type: "user", content: "Run a long audit" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: false,
				tool: {
					tool: "spawn_agent",
					tool_name: "spawn_agent",
					args: { task: "Audit everything", agent: "explorer" },
					streamOutput: longProgress,
					isError: false,
					isComplete: false,
				},
			}],
		},
		isComplete: false,
	}]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /BEGIN-/);
	assert.match(output, /-END/);
	assert.doesNotMatch(output, /truncated|earlier progress hidden/i);
});

void test("collapsed agent streaming is never character or line truncated", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 14 });
	const longProgress = `BEGIN\n${Array.from(
		{ length: 80 },
		(_, index) => `stream-line-${index}-${"x".repeat(80)}`,
	).join("\n")}\nEND`;
	display.setTurns([{
		id: "long-collapsed-agent-output",
		userMessage: { type: "user", content: "Run a long audit" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: false,
				tool: {
					tool: "spawn_agent",
					tool_name: "spawn_agent",
					args: { task: "Audit everything", agent: "explorer" },
					streamOutput: longProgress,
					isError: false,
					isComplete: false,
				},
			}],
		},
		isComplete: false,
	}]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /BEGIN/);
	assert.match(output, /stream-line-79/);
	assert.match(output, /END/);
	assert.doesNotMatch(output, /truncated|earlier lines not shown/i);
});

void test("post-edit diagnostics render as a dedicated formatted block", () => {
	const display = new TranscriptDisplay();
	display.setTurns([{
		id: "diagnostic-turn",
		userMessage: { type: "user", content: "Update runtime config" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: true,
				tool: {
					tool: "edit_file",
					tool_name: "edit_file",
					args: { path: "src/runtime-config.ts", edits: [] },
					result: [
						"Successfully replaced 1 block.",
						'<post_edit_diagnostics file="/workspace/src/runtime-config.ts">',
						"Fix these project diagnostics before continuing:",
						"- /workspace/src/runtime-config.ts:78:4 TS2353: Object literal may only specify known properties.",
						"</post_edit_diagnostics>",
					].join("\n"),
					isError: false,
					isComplete: true,
				},
			}],
		},
		isComplete: true,
	}]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /◆ DIAGNOSTICS 1 issue/);
	assert.match(output, /\/workspace\/src\/runtime-config\.ts/);
	assert.match(output, /× 78:4 TS2353/);
	assert.match(output, /Object literal may only specify known properties/);
	assert.doesNotMatch(output, /post_edit_diagnostics|Fix these project diagnostics/);
});

void test("transcript line limits discard oldest turns and retain newest messages", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 14 });
	const turns: Turn[] = Array.from({ length: 8 }, (_, index) => ({
		id: `turn-${index}`,
		userMessage: { type: "user", content: `user-${index}` },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "content",
				contentText: `assistant-${index}`,
				isComplete: true,
			}],
		},
		isComplete: true,
	}));
	display.setTurns(turns);
	const output = plain(display.render(80).join("\n"));

	assert.match(output, /older turn\(s\) not shown/);
	assert.match(output, /user-7/);
	assert.match(output, /assistant-7/);
	assert.doesNotMatch(output, /user-0/);
});

void test("Ctrl+O expansion keeps a bottom-anchored viewport on newest content", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 200 });
	display.setViewportHeight(7);
	display.setTurns([{
		id: "expanded-at-bottom",
		userMessage: { type: "user", content: "Run the command" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "tool",
					isComplete: true,
					tool: {
						tool: "bash",
						tool_name: "bash",
						args: { command: "build" },
						result: Array.from({ length: 30 }, (_, i) => `build-line-${i}`).join("\n"),
						isError: false,
						isComplete: true,
					},
				},
				{
					seq: 2,
					type: "content",
					contentText: "LATEST RESPONSE",
					isComplete: true,
				},
			],
		},
		isComplete: true,
	}]);
	display.scrollToBottom();
	display.render(90);
	display.toggleToolsExpanded();
	const output = plain(display.render(90).join("\n"));

	assert.match(output, /LATEST RESPONSE/);
});

void test("empty think wrappers do not render a THINK section", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([{
		id: "empty-thinking",
		userMessage: { type: "user", content: "Answer" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "thinking",
					contentText: "  <think>\n</think>  ",
					isComplete: true,
				},
				{
					seq: 2,
					type: "content",
					contentText: "Final answer",
					isComplete: true,
				},
			],
		},
		isComplete: true,
	}]);
	const output = plain(display.render(80).join("\n"));

	assert.doesNotMatch(output, /THINK|<\/?think>/);
	assert.match(output, /Final answer/);
});

void test("think wrappers are removed without hiding real reasoning", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([{
		id: "wrapped-thinking",
		userMessage: { type: "user", content: "Answer" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "thinking",
				contentText: "<think>Useful reasoning</think>",
				isComplete: true,
			}],
		},
		isComplete: true,
	}]);
	const output = plain(display.render(80).join("\n"));

	assert.match(output, /THINK.*reasoning/);
	assert.match(output, /Useful reasoning/);
	assert.doesNotMatch(output, /<\/?think>/);
});

void test("thinking display hides textual tool-call markup", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([{
		id: "thinking-tool-call",
		userMessage: { type: "user", content: "Inspect config" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "thinking",
				contentText: `I need to inspect the defaults.
<tool_call>
<function=grep>
<parameter=path>
config.ts
</parameter>
<parameter=pattern>
cfg.defaults
</parameter>
</function>
</tool_call>`,
				isComplete: false,
			}],
		},
		isComplete: false,
	}]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /I need to inspect the defaults/);
	assert.doesNotMatch(
		output,
		/tool_call|function=grep|parameter=path|cfg\.defaults/,
	);
});

void test("expanded subagent details show child tool calls", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([{
		id: "child-tools-turn",
		userMessage: { type: "user", content: "Run a subagent" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: true,
				tool: {
					tool: "spawn_agent",
					tool_name: "spawn_agent",
					args: { task: "Inspect files", agent: "explorer" },
					result: "Done inspecting.",
					details: {
						agent: "explorer",
						status: "completed",
						metrics: { turns: 5, toolCalls: 3 },
						childToolCalls: [
							{ agentId: "explorer", toolName: "read_file", args: '{"path":"src/index.ts"}', isError: false },
							{ agentId: "explorer", toolName: "grep", args: '{"pattern":"export"}', isError: false },
							{ agentId: "explorer", toolName: "bash", args: '{"command":"ls"}', isError: false },
						],
					},
					isError: false,
					isComplete: true,
				},
			}],
		},
		isComplete: true,
	}]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /tool calls/i);
	assert.match(output, /3 tool calls/);
	assert.match(output, /read_file/);
	assert.match(output, /grep/);
	assert.match(output, /bash/);
	assert.match(output, /explorer/);
});

void test("collapsed subagent card shows a compact recent tool timeline", () => {
	const display = new TranscriptDisplay();
	// toolsExpanded defaults to false
	display.setTurns([{
		id: "child-tools-turn",
		userMessage: { type: "user", content: "Run a subagent" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: true,
				tool: {
					tool: "spawn_agent",
					tool_name: "spawn_agent",
					args: { task: "Inspect files", agent: "explorer" },
					result: "Done inspecting.",
					details: {
						agent: "explorer",
						status: "completed",
						metrics: { turns: 5, toolCalls: 3 },
						childToolCalls: [
							{ agentId: "explorer", toolName: "read_file", args: '{"path":"src/index.ts"}', status: "completed", isError: false },
						],
					},
					isError: false,
					isComplete: true,
				},
			}],
		},
		isComplete: true,
	}]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /ACTIVITY.*1 tool call/);
	assert.match(output, /read_file.*path=src\/index\.ts/);
});

void test("edited TypeScript previews are syntax highlighted", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([{
		id: "highlight-edit",
		userMessage: { type: "user", content: "Edit the file" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: true,
				tool: {
					tool: "edit_file",
					tool_name: "edit_file",
					args: {
						path: "src/example.ts",
						oldText: 'const answer = "no";',
						newText: 'const answer = "yes";',
					},
					isError: false,
					isComplete: true,
				},
			}],
		},
		isComplete: true,
	}]);
	const rendered = display.render(100).join("\n");

	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
	assert.match(plain(rendered), /const answer = "yes";/);
});

void test("internal post-tool hook guidance stays out of the transcript", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([{
		id: "hidden-hook",
		userMessage: { type: "user", content: "Run a tool" },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				isComplete: true,
				tool: {
					tool: "bash",
					tool_name: "bash",
					args: { command: "printf visible" },
					result:
						"visible output\n\n<post-tool-use-hook>\n<context_guidance><tip>internal only</tip></context_guidance>\n</post-tool-use-hook>",
					isError: false,
					isComplete: true,
				},
			}],
		},
		isComplete: true,
	}]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /visible output/);
	assert.doesNotMatch(output, /post-tool-use-hook|context_guidance|internal only/);
});
