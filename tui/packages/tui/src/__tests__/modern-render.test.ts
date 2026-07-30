import assert from "node:assert/strict";
import { test } from "node:test";
import type { Turn } from "@logician/coding-agent/sessions";
import { InputBar } from "../input/input-bar.ts";
import { NotificationCenter } from "../status/notification-center.ts";
import { SteerQueue } from "../status/steer-queue.ts";
import { StatusBar } from "../status/status-bar.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { CURSOR_MARKER, visibleWidth } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

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
						args: { path: "application/agent-bridge.ts" },
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

	assert.match(output, /› YOU/);
	assert.doesNotMatch(output, /╭─|╰─/);
	assert.match(output, /◆ LOGICIAN/);
	assert.match(output, /✓ read_file done/);
	assert.match(output, /output ok/);
	assert.match(output, /18ms/);
	assert.ok(lines.every((line) => visibleWidth(line) <= 80));
});

void test("collapsed running tools show live output without expanding details", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "turn-live-tool",
			userMessage: { type: "user", content: "Run the build." },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						tool: {
							tool: "bash",
							tool_name: "bash",
							tool_call_id: "build-1",
							args: { command: "npm test" },
							streamOutput: "compiling packages...\nsecond line",
							isError: false,
							isComplete: false,
						},
						isComplete: false,
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(100).join("\n"));
	assert.match(output, /bash streaming/);
	assert.match(output, /live compiling packages\.\.\./);
	assert.doesNotMatch(output, /second line/);
});

void test("tool output cannot inject terminal control sequences", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([{
		id: "terminal-injection",
		userMessage: { type: "user", content: "Show the output." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "bash",
					tool_name: "bash",
					args: { command: "printf untrusted" },
					result:
						"safe\x1b[2J text\x1b]0;owned title\x07 visible",
					isComplete: true,
					isError: false,
				},
				isComplete: true,
			}],
		},
		isComplete: true,
	}]);

	const rendered = display.render(100).join("\n");
	assert.doesNotMatch(rendered, /\x1b\[2J|\x1b\]0;/);
	assert.match(plain(rendered), /safe text visible/);
});

void test("growing tool streams sanitize only the appended suffix", () => {
	const display = new TranscriptDisplay();
	const turn: Turn = {
		id: "incremental-sanitize",
		userMessage: { type: "user", content: "Compile." },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "bash",
					tool_name: "bash",
					args: { command: "compile" },
					streamOutput: "x".repeat(100_000),
					isComplete: false,
					isError: false,
				},
				isComplete: false,
			}],
		},
		isComplete: false,
	};
	display.setTurns([turn]);
	display.render(100);
	const before = display.getSanitizationMetrics().scannedCharacters;
	const tool = turn.assistantMessage?.chunks[0].tool;
	assert.ok(tool);
	tool.streamOutput += "y";
	display.setTurns([turn]);

	display.render(100);

	const scanned =
		display.getSanitizationMetrics().scannedCharacters - before;
	assert.equal(scanned, 1);
});

void test("clicking a tool card toggles only that tool's details", () => {
	const display = new TranscriptDisplay();
	display.setViewportHeight(20);
	display.setTurns([{
		id: "clickable-tools",
		userMessage: { type: "user", content: "Run both commands." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "tool",
					tool: {
						tool: "bash",
						tool_name: "bash",
						tool_call_id: "first-tool",
						args: { command: "echo first" },
						result: "first",
						isComplete: true,
						isError: false,
					},
					isComplete: true,
				},
				{
					seq: 2,
					type: "tool",
					tool: {
						tool: "bash",
						tool_name: "bash",
						tool_call_id: "second-tool",
						args: { command: "echo second" },
						result: "second",
						isComplete: true,
						isError: false,
					},
					isComplete: true,
				},
			],
		},
		isComplete: true,
	}]);

	const collapsed = display.render(100);
	const firstToolRow = collapsed.findIndex((line) =>
		plain(line).includes("echo first"),
	);
	assert.notEqual(firstToolRow, -1);
	assert.equal(display.handleMouse(4, firstToolRow), true);

	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /COMMAND[\s\S]*echo first/);
	assert.doesNotMatch(expanded, /COMMAND[\s\S]*echo second[\s\S]*OUTPUT/);

	const rerendered = display.render(100);
	const expandedFirstRow = rerendered.findIndex((line) =>
		plain(line).includes("echo first"),
	);
	assert.equal(display.handleMouse(4, expandedFirstRow), true);
	assert.doesNotMatch(plain(display.render(100).join("\n")), /◆ details/);
});

void test("keyboard navigation focuses and toggles individual tool cards", () => {
	const display = new TranscriptDisplay();
	display.setViewportHeight(20);
	display.setTurns([{
		id: "keyboard-tools",
		userMessage: { type: "user", content: "Run both." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: ["first", "second"].map((name, index) => ({
				seq: index + 1,
				type: "tool" as const,
				tool: {
					tool: "bash",
					tool_name: "bash",
					tool_call_id: name,
					args: { command: `echo ${name}` },
					result: `${name} output`,
					isComplete: true,
					isError: false,
				},
				isComplete: true,
			})),
		},
		isComplete: true,
	}]);

	display.render(80);
	assert.deepEqual(display.focusTool(1), { index: 1, total: 2 });
	assert.match(plain(display.render(80).join("\n")), /› ✓ bash done/);
	assert.equal(display.toggleFocusedTool(), true);
	assert.match(
		plain(display.render(80).join("\n")),
		/COMMAND[\s\S]*echo first[\s\S]*OUTPUT[\s\S]*first output/,
	);

	assert.deepEqual(display.focusTool(1), { index: 2, total: 2 });
	assert.equal(display.toggleFocusedTool(), true);
	const expanded = plain(display.render(80).join("\n"));
	assert.match(expanded, /echo second[\s\S]*second output/);
});

void test("write_file streams live line counts and expanded content", () => {
	const display = new TranscriptDisplay();
	display.setTurns([{
		id: "streaming-write-file",
		userMessage: { type: "user", content: "Create the module." },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "write_file",
					tool_name: "write_file",
					tool_call_id: "live-write",
					args: {},
					partialResult:
						"{\"path\":\"src/live.ts\",\"content\":\"const one = 1;\\nconst two = 2;\\nconst three",
					isComplete: false,
					isError: false,
				},
				isComplete: false,
			}],
		},
		isComplete: false,
	}]);

	const collapsed = plain(display.render(100).join("\n"));
	assert.match(collapsed, /write_file src\/live\.ts streaming/);
	assert.match(collapsed, /3 lines written so far/);
	assert.doesNotMatch(collapsed, /const one = 1/);
	assert.doesNotMatch(collapsed, /"content"/);

	display.setToolsExpanded(true);
	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /CONTENT.*3 lines · streaming/);
	assert.match(expanded, /const one = 1/);
	assert.match(expanded, /const two = 2/);
	assert.match(expanded, /const three/);
});

void test("click-expanded write_file shows every line without a Ctrl+O hint", () => {
	const content = Array.from({ length: 24 }, (_, index) => `line-${index + 1}`)
		.join("\n");
	const display = new TranscriptDisplay();
	display.setViewportHeight(40);
	display.setTurns([{
		id: "large-clicked-write",
		userMessage: { type: "user", content: "Write the fixture." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "write_file",
					tool_name: "write_file",
					tool_call_id: "large-write",
					args: { path: "fixture.txt", content },
					result: "Created fixture.txt",
					isComplete: true,
					isError: false,
				},
				isComplete: true,
			}],
		},
		isComplete: true,
	}]);

	const collapsed = display.render(100);
	const toolRow = collapsed.findIndex((line) =>
		plain(line).includes("write_file fixture.txt"),
	);
	assert.notEqual(toolRow, -1);
	assert.match(plain(collapsed.join("\n")), /24 lines written/);
	assert.equal(display.handleMouse(4, toolRow), true);

	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /24│line-24/);
	assert.doesNotMatch(expanded, /more lines · ctrl\+o to expand/);
});

void test("write_file_append streams live line counts and expanded content", () => {
	const display = new TranscriptDisplay();
	display.setTurns([{
		id: "streaming-write-file-append",
		userMessage: { type: "user", content: "Append the next chunk." },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "write_file_append",
					tool_name: "write_file_append",
					tool_call_id: "live-append",
					args: {},
					partialResult:
						"{\"path\":\"src/live.ts\",\"content\":\"const four = 4;\\nconst five = 5;\\nconst six",
					isComplete: false,
					isError: false,
				},
				isComplete: false,
			}],
		},
		isComplete: false,
	}]);

	const collapsed = plain(display.render(100).join("\n"));
	assert.match(collapsed, /write_file_append src\/live\.ts streaming/);
	assert.match(collapsed, /3 lines appended so far/);
	assert.doesNotMatch(collapsed, /const four = 4/);
	assert.doesNotMatch(collapsed, /"content"/);

	display.setToolsExpanded(true);
	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /APPEND CONTENT.*3 lines · streaming/);
	assert.match(expanded, /const four = 4/);
	assert.match(expanded, /const five = 5/);
	assert.match(expanded, /const six/);
});

void test("click-expanded write_file_append shows every appended line", () => {
	const content = Array.from(
		{ length: 24 },
		(_, index) => `appended-${index + 1}`,
	).join("\n");
	const display = new TranscriptDisplay();
	display.setViewportHeight(40);
	display.setTurns([{
		id: "large-clicked-append",
		userMessage: { type: "user", content: "Append the fixture." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 1,
				type: "tool",
				tool: {
					tool: "write_file_append",
					tool_name: "write_file_append",
					tool_call_id: "large-append",
					args: { path: "fixture.txt", content },
					result:
						"Appended to fixture.txt (+279 bytes, 558 bytes total · 48 lines total).",
					isComplete: true,
					isError: false,
				},
				isComplete: true,
			}],
		},
		isComplete: true,
	}]);

	const collapsed = display.render(100);
	const toolRow = collapsed.findIndex((line) =>
		plain(line).includes("write_file_append fixture.txt"),
	);
	assert.notEqual(toolRow, -1);
	assert.match(plain(collapsed.join("\n")), /24 lines appended/);
	assert.equal(display.handleMouse(4, toolRow), true);

	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /24│appended-24/);
	assert.doesNotMatch(expanded, /more lines · ctrl\+o to expand/);
});

void test("skill activations render as a compact dedicated status line", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "skill-activation",
			userMessage: { type: "user", content: "Debug this TypeScript error" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "notice",
						notice: {
							level: "info",
							label: "Skills",
							text: "TypeScript Debugging · matched “TypeScript error”",
						},
						isComplete: true,
					},
				],
			},
			isComplete: false,
		},
	]);
	const rendered = display.render(100).join("\n");
	const output = plain(rendered);

	assert.match(
		output,
		/✦ NOTICE Skills  TypeScript Debugging · matched “TypeScript error”/,
	);
	assert.doesNotMatch(output, /Skills:/);
	assert.match(rendered, /\x1b\[/);
});

void test("assistant chunks render as distinct semantic blocks", () => {
	const display = new TranscriptDisplay();
	display.setThinkingMode("summary");
	display.setTurns([
		{
			id: "semantic-blocks",
			userMessage: { type: "user", content: "Explain the result." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "thinking",
						contentText: "Compare both execution paths.",
						isComplete: true,
					},
					{
						seq: 1,
						type: "notice",
						notice: { level: "warn", label: "Context", text: "Near limit" },
						isComplete: true,
					},
					{
						seq: 2,
						type: "content",
						contentText: "The minimal path delegates continuation.",
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(100).join("\n"));
	assert.match(output, /REASONING.*Compare both execution paths/);
	assert.match(output, /⚠ NOTICE Context  Near limit/);
	assert.match(output, /RESPONSE/);
	assert.match(output, /The minimal path delegates continuation/);
});

void test("expanded reasoning renders fenced code as one labeled block", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([{
		id: "reasoning-code-block",
		userMessage: { type: "user", content: "Inspect this implementation." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [{
				seq: 0,
				type: "thinking",
				contentText:
					"Check the handler:\n```typescript\nconst value = 1;\nreturn value;\n```",
				isComplete: true,
			}],
		},
		isComplete: true,
	}]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /┌─ typescript · 2 lines/);
	assert.match(output, /│ const value = 1;/);
	assert.match(output, /│ return value;/);
	assert.match(output, /└─/);
	assert.equal(output.match(/typescript/g)?.length, 1);
	assert.doesNotMatch(output, /```/);
});

void test("notifications are transient, bounded, and width-safe", () => {
	const notifications = new NotificationCenter();
	notifications.show("Execution policy: minimal", "success", 60_000);
	notifications.show("Theme: dark", "info", 60_000);
	notifications.show("Invalid temperature", "error", 60_000);
	notifications.show("Only the newest three remain", "warning", 60_000);
	const lines = notifications.render(32);
	const output = plain(lines.join("\n"));
	assert.equal(lines.length, 3);
	assert.doesNotMatch(output, /Execution policy/);
	assert.match(output, /Only the newest three remain/);
	assert.ok(lines.every((line) => visibleWidth(line) <= 32));
	notifications.clear();
	assert.deepEqual(notifications.render(32), []);
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

void test("status bar renders cached tokens and unknown telemetry", () => {
	const status = new StatusBar();
	status.update({
		contextTokens: 20_000,
		contextMaxTokens: 32_768,
		cacheReadTokens: 12_400,
	});
	assert.match(plain(status.render(160)[0]), /cache read: 12\.4k/);

	status.update({ contextTokens: 21_000 });
	assert.match(plain(status.render(160)[0]), /cache read: 12\.4k/);

	status.update({ cacheReadTokens: undefined });
	assert.match(plain(status.render(160)[0]), /cache read: unknown/);
});

void test("status bar renders RTK when restored as enabled", () => {
	const status = new StatusBar();
	assert.doesNotMatch(plain(status.render(200)[0]), /\brtk on\b/);

	status.update({ rtkProxyEnabled: true });
	assert.match(plain(status.render(200)[0]), /\brtk on\b/);
});

void test("input prompt has stable inset modern chrome", () => {
	const input = new InputBar();
	input.focused = true;
	const [header, line] = input.render(40);
	assert.match(plain(header), /Enter send/);
	assert.match(plain(line).replace(CURSOR_MARKER, ""), /^  › Ask Logician/);
	assert.equal(visibleWidth(header), 40);
	assert.equal(visibleWidth(line), 40);
});

void test("composer preserves explicit steer-now submission intent", () => {
	const input = new InputBar();
	let submitted: { text: string; intent: string } | undefined;
	input.onSubmit = (text, intent) => {
		submitted = { text, intent };
	};
	input.valueText = "change direction";

	assert.equal(input.submit("steer-now"), true);
	assert.deepEqual(submitted, {
		text: "change direction",
		intent: "steer-now",
	});
	assert.equal(input.valueText, "");
	assert.equal(input.submit("steer-now"), false);
});

void test("steering queue distinguishes queued and later delivery", () => {
	const queue = new SteerQueue();
	queue.setItems(["inspect the parser"], ["run the complete test suite"]);
	const lines = queue.render(72);
	const rendered = plain(lines.join("\n"));

	assert.match(rendered, /STEERING\s+1 queued · 1 follow-up/);
	assert.match(rendered, /QUEUE\s+inspect the parser/);
	assert.match(rendered, /LATER\s+run the complete test suite/);
	assert.match(rendered, /Ctrl\+Enter steer now/);
	assert.ok(lines.every((line) => visibleWidth(line) <= 72));
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
	display.setTurns([
		{
			id: "agent-turn",
			userMessage: { type: "user", content: "Audit the repository" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 1,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: {
								task: "Inspect architecture and tests",
								agent: "explorer",
							},
							partialResult: "{\"task\":\"Inspect architecture and tests\"}",
							streamOutput: "I am inspecting the core files now.",
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /subagent explorer streaming/);
	assert.match(output, /Inspect architecture and tests/);
	assert.match(output, /I am inspecting the core files now/);
	assert.doesNotMatch(output, /TASK|ACTIVITY|LIVE PROGRESS/);
	assert.doesNotMatch(output, /\{"task":/);
});

void test("expanded subagent streams render fenced code with syntax highlighting", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "agent-code-stream",
			userMessage: { type: "user", content: "Inspect code" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Inspect code", agent: "explorer" },
							streamOutput:
								"Found this:\n```typescript\nconst answer = 42;\n```",
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const rendered = display.render(100).join("\n");

	assert.match(plain(rendered), /Found this/);
	assert.match(plain(rendered), /const answer = 42/);
	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
});

void test("expanded agent progress is never character-truncated", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	const longProgress = `BEGIN-${"x".repeat(8_000)}-END`;
	display.setTurns([
		{
			id: "long-agent-output",
			userMessage: { type: "user", content: "Run a long audit" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
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
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /BEGIN-/);
	assert.match(output, /-END/);
	assert.doesNotMatch(output, /truncated|earlier progress hidden/i);
});

void test("collapsed agent card hides its stream until expanded", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 14 });
	const longProgress = `BEGIN\n${Array.from(
		{ length: 80 },
		(_, index) => `stream-line-${index}-${"x".repeat(80)}`,
	).join("\n")}\nEND`;
	display.setTurns([
		{
			id: "long-collapsed-agent-output",
			userMessage: { type: "user", content: "Run a long audit" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
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
					},
				],
			},
			isComplete: false,
		},
	]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /subagent explorer streaming/);
	assert.doesNotMatch(output, /BEGIN|stream-line-79|END/);
});

void test("expanded completed subagent keeps its streaming transcript", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "completed-agent-stream",
			userMessage: { type: "user", content: "Run an audit" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Audit everything", agent: "explorer" },
							result: "Audit complete.",
							details: {
								streamTranscript:
									"Inspecting files...\n```ts\nconst ok = true;\n```",
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const rendered = display.render(100).join("\n");
	const output = plain(rendered);

	assert.match(output, /Inspecting files/);
	assert.match(output, /const ok = true/);
	assert.match(output, /Audit complete/);
	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
});

void test("expanded completed subagent does not repeat its final report", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "deduplicated-agent-report",
			userMessage: { type: "user", content: "Review it" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Review it", agent: "reviewer" },
							result: "**Final report:** all checks passed.",
							details: {
								streamTranscript:
									"Inspecting files...\n\n**Final report:** all checks passed.\n\n" +
									"```acceptance-report\n{\"criteriaSatisfied\":[]}\n```",
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(100).join("\n"));

	assert.equal(output.match(/Final report:/g)?.length, 1);
	assert.match(output, /Inspecting files/);
});

void test("collapsed completed subagent formats its final report as markdown", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "markdown-agent-report",
			userMessage: { type: "user", content: "Review it" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Review it", agent: "reviewer" },
							result:
								"**Approved** with `zero errors`.\n\n```ts\nconst valid = true;\n```",
							details: {
								streamTranscript:
									"Working...\n```acceptance-report\n{\"criteriaSatisfied\":[]}\n```",
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const rendered = display.render(100).join("\n");
	const output = plain(rendered);

	assert.match(output, /Approved.*zero errors/);
	assert.match(rendered, /\x1b\[1mApproved/);
	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
	assert.doesNotMatch(output, /acceptance-report|criteriaSatisfied/);
});

void test("post-edit diagnostics render as a dedicated formatted block", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "diagnostic-turn",
			userMessage: { type: "user", content: "Update runtime config" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "edit_file",
							tool_name: "edit_file",
							args: { path: "src/runtime-config.ts", edits: [] },
							result: [
								"Successfully replaced 1 block.",
								"<post_edit_diagnostics file=\"/workspace/src/runtime-config.ts\">",
								"Fix these project diagnostics before continuing:",
								"- /workspace/src/runtime-config.ts:78:4 TS2353: Object literal may only specify known properties.",
								"</post_edit_diagnostics>",
							].join("\n"),
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /◆ DIAGNOSTICS 1 issue/);
	assert.match(output, /\/workspace\/src\/runtime-config\.ts/);
	assert.match(output, /× 78:4 TS2353/);
	assert.match(output, /Object literal may only specify known properties/);
	assert.doesNotMatch(
		output,
		/post_edit_diagnostics|Fix these project diagnostics/,
	);
});

void test("post-edit diagnostics render clangd source and symbolic codes", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "clang-diagnostic-turn",
			userMessage: { type: "user", content: "Update native extension" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "edit_file",
							tool_name: "edit_file",
							args: { path: "/data/dev/solvers/python/qp_ext.cpp", edits: [] },
							result: [
								"Successfully replaced 1 block.",
								"<post_edit_diagnostics file=\"/data/dev/solvers/python/qp_ext.cpp\">",
								"Fix these project diagnostics before continuing:",
								"- /data/dev/solvers/python/qp_ext.cpp:42:7 clang ovl_no_viable_function_in_call: No matching function for call.",
								"</post_edit_diagnostics>",
							].join("\n"),
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);

	const output = plain(display.render(100).join("\n"));
	assert.match(output, /◆ DIAGNOSTICS 1 issue/);
	assert.match(output, /× 42:7 clang ovl_no_viable_function_in_call/);
	assert.match(output, /No matching function for call/);
	assert.doesNotMatch(output, /could not be parsed/);
});

void test("transcript line limits discard oldest turns and retain newest messages", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 14 });
	const turns: Turn[] = Array.from({ length: 8 }, (_, index) => ({
		id: `turn-${index}`,
		userMessage: { type: "user", content: `user-${index}` },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "content",
					contentText: `assistant-${index}`,
					isComplete: true,
				},
			],
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
	display.setTurns([
		{
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
							result: Array.from(
								{ length: 30 },
								(_, i) => `build-line-${i}`,
							).join("\n"),
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
		},
	]);
	display.scrollToBottom();
	display.render(90);
	display.toggleToolsExpanded();
	const output = plain(display.render(90).join("\n"));

	assert.match(output, /LATEST RESPONSE/);
});

void test("wheel-down reaches the new bottom while a streaming update awaits render", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 200 });
	display.setViewportHeight(6);
	const streamingTurn = (contentText: string): Turn => ({
		id: "streaming-scroll",
		userMessage: { type: "user", content: "Keep streaming" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [
				{
					seq: 1,
					type: "content",
					contentText,
					isComplete: false,
				},
			],
		},
		isComplete: false,
	});

	display.setTurns([
		streamingTurn(Array.from({ length: 16 }, (_, i) => `line-${i}`).join("\n")),
	]);
	display.render(80);
	display.scroll(4);
	assert.equal(display.isAtBottom, false);

	display.setTurns([
		streamingTurn(
			[
				...Array.from({ length: 16 }, (_, i) => `line-${i}`),
				"NEWEST STREAMED LINE",
			].join("\n"),
		),
	]);
	display.scroll(-100);
	const output = plain(display.render(80).join("\n"));

	assert.equal(display.isAtBottom, true);
	assert.match(output, /NEWEST STREAMED LINE/);
});

void test("new streamed output is signaled while the user is scrolled up", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 200 });
	display.setViewportHeight(6);
	const streamingTurn = (contentText: string): Turn => ({
		id: "streaming-indicator",
		userMessage: { type: "user", content: "Keep streaming" },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [{
				seq: 1,
				type: "content",
				contentText,
				isComplete: false,
			}],
		},
		isComplete: false,
	});

	display.setTurns([
		streamingTurn(Array.from({ length: 18 }, (_, i) => `line-${i}`).join("\n")),
	]);
	display.render(80);
	display.scroll(4);
	assert.doesNotMatch(plain(display.render(80).join("\n")), /new output below/);

	display.setTurns([
		streamingTurn(
			[
				...Array.from({ length: 18 }, (_, i) => `line-${i}`),
				"line-18",
			].join("\n"),
		),
	]);
	const scrolledOutput = plain(display.render(80).join("\n"));
	assert.equal(display.isAtBottom, false);
	assert.match(scrolledOutput, /↓ new output below/);

	assert.equal(display.handleMouse(4, 5), true);
	const bottomOutput = plain(display.render(80).join("\n"));
	assert.doesNotMatch(bottomOutput, /new output below/);
	assert.match(bottomOutput, /line-18/);
});

void test("empty think wrappers do not render a THINK section", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([
		{
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
		},
	]);
	const output = plain(display.render(80).join("\n"));

	assert.doesNotMatch(output, /THINK|<\/?think>/);
	assert.match(output, /Final answer/);
});

void test("think wrappers are removed without hiding real reasoning", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([
		{
			id: "wrapped-thinking",
			userMessage: { type: "user", content: "Answer" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "thinking",
						contentText: "<think>Useful reasoning</think>",
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(80).join("\n"));

	assert.match(output, /REASONING/);
	assert.match(output, /Useful reasoning/);
	assert.doesNotMatch(output, /<\/?think>/);
});

void test("thinking display hides textual tool-call markup", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([
		{
			id: "thinking-tool-call",
			userMessage: { type: "user", content: "Inspect config" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
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
					},
				],
			},
			isComplete: false,
		},
	]);

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
	display.setTurns([
		{
			id: "child-tools-turn",
			userMessage: { type: "user", content: "Run a subagent" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
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
									{
										agentId: "explorer",
										toolName: "read_file",
										args: "{\"path\":\"src/index.ts\"}",
										isError: false,
									},
									{
										agentId: "explorer",
										toolName: "grep",
										args: "{\"pattern\":\"export\"}",
										isError: false,
									},
									{
										agentId: "explorer",
										toolName: "bash",
										args: "{\"command\":\"ls\"}",
										isError: false,
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /3 tool call\(s\)/);
	assert.match(output, /read_file/);
	assert.match(output, /grep/);
	assert.match(output, /bash/);
	assert.match(output, /explorer/);
});

void test("expanded subagent renders thinking, tools, and responses in call order", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "ordered-child-flow",
			userMessage: { type: "user", content: "Run a subagent" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Inspect files", agent: "explorer" },
							result: "Summary: implementation verified successfully.",
							details: {
								agent: "explorer",
								status: "completed",
								childChunks: [
									{
										seq: 1,
										agentId: "explorer-1",
										type: "thinking",
										contentText: "I should inspect the entry point.",
										isComplete: true,
									},
									{
										seq: 2,
										agentId: "explorer-1",
										type: "content",
										contentText: "I am checking the implementation.",
										isComplete: true,
									},
									{
										seq: 3,
										agentId: "explorer-1",
										type: "tool",
										tool: {
											agentId: "explorer-1",
											toolCallId: "child-tool-1",
											toolName: "read_file",
											args: "{\"path\":\"src/index.ts\"}",
											status: "completed",
											resultPreview: "export const ready = true;",
										},
										isComplete: true,
									},
									{
										seq: 4,
										agentId: "explorer-1",
										type: "content",
										contentText: "The implementation is correct.",
										isComplete: true,
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);

	const output = plain(display.render(120).join("\n"));
	const thinking = output.indexOf("I should inspect the entry point.");
	const progress = output.indexOf("I am checking the implementation.");
	const tool = output.indexOf("read_file");
	const result = output.indexOf("export const ready = true;");
	const response = output.indexOf("The implementation is correct.");

	assert.ok(thinking >= 0);
	assert.ok(progress > thinking);
	assert.ok(tool > progress);
	assert.ok(result > tool);
	assert.ok(response > result);
	assert.equal(output.match(/The implementation is correct\./g)?.length, 1);
	assert.doesNotMatch(
		output,
		/Summary: implementation verified successfully\./,
	);
	assert.doesNotMatch(output, /ACTIVITY/);
});

void test("collapsed completed subagent shows its final summary", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "collapsed-child-summary",
			userMessage: { type: "user", content: "Run a subagent" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Inspect files", agent: "explorer" },
							result: "Summary: the implementation is correct.",
							details: {
								agent: "explorer",
								status: "completed",
								childChunks: [
									{
										seq: 1,
										agentId: "explorer-1",
										type: "thinking",
										contentText: "Private reasoning.",
										isComplete: true,
									},
									{
										seq: 2,
										agentId: "explorer-1",
										type: "content",
										contentText: "Intermediate progress.",
										isComplete: true,
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);

	const output = plain(display.render(120).join("\n"));

	assert.match(output, /Summary: the implementation is correct\./);
	assert.match(output, /Private reasoning/);
	assert.match(output, /Intermediate progress/);
});

void test("collapsed subagent shows ordered flow with child tools collapsed", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([
		{
			id: "collapsed-child-flow",
			userMessage: { type: "user", content: "Run a subagent" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Inspect files", agent: "explorer" },
							result: "Inspection complete.",
							details: {
								agent: "explorer",
								status: "completed",
								childChunks: [
									{
										seq: 1,
										agentId: "explorer-1",
										type: "thinking",
										contentText: "I should inspect first.",
										isComplete: true,
									},
									{
										seq: 2,
										agentId: "explorer-1",
										type: "content",
										contentText: "Inspecting now.",
										isComplete: true,
									},
									{
										seq: 3,
										agentId: "explorer-1",
										type: "tool",
										tool: {
											agentId: "explorer-1",
											toolCallId: "read-1",
											toolName: "read_file",
											args: "{\"path\":\"src/index.ts\"}",
											status: "completed",
											resultPreview: "private file contents",
										},
										isComplete: true,
									},
									{
										seq: 4,
										agentId: "explorer-1",
										type: "content",
										contentText: "Inspection complete.",
										isComplete: true,
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);

	const collapsed = plain(display.render(120).join("\n"));
	assert.match(collapsed, /I should inspect first\./);
	assert.match(collapsed, /Inspecting now\./);
	assert.match(collapsed, /read_file/);
	assert.match(collapsed, /Inspection complete\./);
	assert.match(collapsed, /SUBAGENT · explorer-1/);
	assert.match(collapsed, /RETURN TO PARENT/);
	assert.match(collapsed, /output private file contents/);
	assert.equal(collapsed.match(/Inspection complete\./g)?.length, 1);

	display.setToolsExpanded(true);
	const expanded = plain(display.render(120).join("\n"));
	assert.match(expanded, /private file contents/);
	assert.equal(expanded.match(/Inspection complete\./g)?.length, 1);
});

void test("collapsed subagent card shows a compact recent tool timeline", () => {
	const display = new TranscriptDisplay();
	// toolsExpanded defaults to false
	display.setTurns([
		{
			id: "child-tools-turn",
			userMessage: { type: "user", content: "Run a subagent" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
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
									{
										agentId: "explorer",
										toolName: "read_file",
										args: "{\"path\":\"src/index.ts\"}",
										status: "completed",
										isError: false,
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /read_file.*path=src\/index\.ts/);
	assert.doesNotMatch(output, /ACTIVITY/);
	assert.equal(output.match(/explorer/g)?.length, 1);
});

void test("completed subagent card has one parent success indicator", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "completed-subagent",
			userMessage: { type: "user", content: "Delegate inspection" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agent",
							tool_name: "spawn_agent",
							args: { task: "Inspect files", agent: "explorer" },
							result: "Inspection complete.",
							details: {
								agent: "explorer",
								status: "completed",
								metrics: { turns: 2, toolCalls: 0, durationMs: 1400 },
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.equal(output.match(/✓/g)?.length, 1);
	assert.match(output, /✓ subagent explorer done/);
	assert.match(output, /2 turn.*0 tool call.*1\.4s/);
	assert.equal(
		output.match(/Inspection complete\./g)?.length,
		1,
		"the final report should render once",
	);
	assert.doesNotMatch(output, /◆ subagent|◆ agent/);
});

void test("spawn_agents renders ordered live task status", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "running-agent-batch",
			userMessage: { type: "user", content: "Inspect in parallel" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agents",
							tool_name: "spawn_agents",
							args: {
								tasks: [
									{ agent: "explorer", task: "Inspect the API" },
									{ agent: "reviewer", task: "Review the tests" },
									{ agent: "general", task: "Check documentation" },
								],
							},
							streamOutput: "▶ 0 explorer\n✓ 0 explorer\n▶ 1 reviewer\n",
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /subagents 2\/3 running.*3 tasks/);
	assert.match(output, /✓ 1\. explorer.*Inspect the API/);
	assert.match(output, /⠋ 2\. reviewer.*Review the tests/);
	assert.match(output, /· 3\. general.*Check documentation/);
});

void test("spawn_agents never renders a positive count over zero while arguments stream", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "streaming-agent-batch-args",
			userMessage: { type: "user", content: "Inspect in parallel" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agents",
							tool_name: "spawn_agents",
							partialResult: "{\"tasks\":[{\"agent\":\"explorer\"",
							streamOutput: "▶ 0 explorer\n",
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /subagents 1\/1 running/);
	assert.doesNotMatch(output, /\/0 running/);
});

void test("spawn_agents repairs an inconsistent structured total", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "inconsistent-agent-batch-total",
			userMessage: { type: "user", content: "Inspect in parallel" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agents",
							tool_name: "spawn_agents",
							streamOutput: "▶ 2 explorer\n",
							details: { total: 0 },
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /subagents 1\/3 running/);
	assert.doesNotMatch(output, /\/0 running/);
});

void test("expanded spawn_agents keeps concurrent text streams attributed", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "streaming-agent-batch",
			userMessage: { type: "user", content: "Inspect in parallel" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: false,
						tool: {
							tool: "spawn_agents",
							tool_name: "spawn_agents",
							args: {
								tasks: [
									{ agent: "explorer", task: "Inspect API" },
									{ agent: "reviewer", task: "Inspect tests" },
								],
							},
							streamOutput: [
								"▶ 0 explorer",
								"↳ 0 \"API stream\\n```ts\\nconst api = true;\\n```\"",
								"▶ 1 reviewer",
								"↳ 1 \"Test stream\"",
								"",
							].join("\n"),
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	const rendered = display.render(120).join("\n");
	const output = plain(rendered);

	assert.match(output, /1\. explorer.*Inspect API[\s\S]*API stream/);
	assert.match(output, /2\. reviewer.*Inspect tests[\s\S]*Test stream/);
	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
});

void test("spawn_agents shows partial failures and expanded reports", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "completed-agent-batch",
			userMessage: { type: "user", content: "Inspect in parallel" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "spawn_agents",
							tool_name: "spawn_agents",
							args: {
								tasks: [
									{ agent: "explorer", task: "Inspect the API" },
									{ agent: "reviewer", task: "Review the tests" },
								],
							},
							result: "",
							details: {
								total: 2,
								completed: 1,
								failed: 1,
								results: [
									{ index: 0, content: "API looks good.", isError: false },
									{ index: 1, content: "Tests failed.", isError: true },
								],
								childToolCalls: [
									{
										agentId: "agent-reviewer",
										toolName: "bash",
										args: "{\"command\":\"npm test\"}",
										status: "failed",
										isError: true,
										resultPreview: "1 test failed",
									},
								],
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(120).join("\n"));

	assert.match(output, /! subagents partial · 1 failed/);
	assert.match(output, /✓ 1\. explorer/);
	assert.match(output, /× 2\. reviewer/);
	assert.match(output, /API looks good/);
	assert.match(output, /Tests failed/);
	assert.match(output, /bash.*agent-reviewer.*command=npm test/);
});

void test("edited TypeScript previews are syntax highlighted", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "highlight-edit",
			userMessage: { type: "user", content: "Edit the file" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "edit_file",
							tool_name: "edit_file",
							args: {
								path: "src/example.ts",
								oldText: "const answer = \"no\";",
								newText: "const answer = \"yes\";",
							},
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const rendered = display.render(100).join("\n");

	assert.match(rendered, /\x1b\[38;5;\d+mconst/);
	assert.match(plain(rendered), /const answer = "yes";/);
});

void test("edit_file result highlights code inside the diff", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(false);
	display.setTurns([
		{
			id: "highlight-edit-result",
			userMessage: { type: "user", content: "Edit the file" },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						isComplete: true,
						tool: {
							tool: "edit_file",
							tool_name: "edit_file",
							args: { path: "src/example.ts" },
							result: [
								"Successfully replaced 1 occurrence.",
								"Diff:",
								"--- a/edit",
								"+++ b/edit",
								"@@ -1 +1 @@",
								"-const answer = \"no\";",
								"+const answer = \"yes\";",
							].join("\n"),
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const rendered = display.render(100).join("\n");

	assert.match(
		rendered,
		/\x1b\[38;5;\d+m\+\x1b\[0m\x1b\[48;5;\d+m\x1b\[0m\x1b\[38;5;141mconst/,
	);
	assert.match(plain(rendered), /\+const answer = "yes";/);
});

void test("internal post-tool hook guidance stays out of the transcript", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	display.setTurns([
		{
			id: "hidden-hook",
			userMessage: { type: "user", content: "Run a tool" },
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
							args: { command: "printf visible" },
							result:
								"visible output\n\n<post-tool-use-hook>\n<context_guidance><tip>internal only</tip></context_guidance>\n</post-tool-use-hook>",
							isError: false,
							isComplete: true,
						},
					},
				],
			},
			isComplete: true,
		},
	]);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /visible output/);
	assert.doesNotMatch(
		output,
		/post-tool-use-hook|context_guidance|internal only/,
	);
});
