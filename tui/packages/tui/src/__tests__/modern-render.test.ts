import assert from "node:assert/strict";
import { test } from "node:test";
import type { Turn } from "@logician/coding-agent/sessions";
import { InputBar } from "../input/input-bar.ts";
import { renderLayoutFrame } from "../rendering/layout.ts";
import { ScrollView } from "../rendering/scroll-view.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { NewOutputIndicator } from "../rendering/transcript/new-output-indicator.ts";
import { NotificationCenter } from "../status/notification-center.ts";
import { StatusBar } from "../status/status-bar.ts";
import { SteerQueue } from "../status/steer-queue.ts";
import { CURSOR_MARKER, visibleWidth } from "../terminal/core.ts";
import { initTheme, theme } from "../terminal/theme.ts";

/** Drive a TranscriptDisplay through a real ScrollView + layout pass, the
 * same integration path app/tui.ts wires up in buildLayout(). Returns the
 * ScrollView so tests can scroll/inspect follow state, plus a render()
 * helper that produces the clipped, scrollbar-painted viewport lines a real
 * frame would show. */
function mountInScrollView(
	display: TranscriptDisplay,
	options: { width: number; height: number },
): { scrollView: ScrollView; render: () => string[] } {
	const scrollView = new ScrollView(display, {
		follow: "end",
		primary: true,
		scrollbar: "auto",
	});
	display.setScrollView(scrollView);
	return {
		scrollView,
		render: () =>
			renderLayoutFrame(scrollView, options.width, options.height, () => {})
				.lines,
	};
}

initTheme("dark");

const plain = (value: string): string =>
	value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

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
	assert.ok(
		lines.some(line => line.includes(`${theme.fgRaw("userLabel")}\x1b[1mYOU`)),
	);
	assert.doesNotMatch(output, /╭─|╰─/);
	assert.match(output, /◆ LOGICIAN/);
	assert.match(output, /✓ read_file done/);
	assert.match(output, /output ok/);
	assert.match(output, /18ms/);
	assert.ok(lines.every(line => visibleWidth(line) <= 80));
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
	display.setTurns([
		{
			id: "terminal-injection",
			userMessage: { type: "user", content: "Show the output." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						tool: {
							tool_name: "bash",
							args: { command: "printf untrusted" },
							result: "safe\x1b[2J text\x1b]0;owned title\x07 visible",
							isComplete: true,
							isError: false,
						},
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);

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
			chunks: [
				{
					seq: 1,
					type: "tool",
					tool: {
						tool_name: "bash",
						args: { command: "compile" },
						streamOutput: "x".repeat(100_000),
						isComplete: false,
						isError: false,
					},
					isComplete: false,
				},
			],
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

	const scanned = display.getSanitizationMetrics().scannedCharacters - before;
	assert.equal(scanned, 1);
});

void test("streaming updates reuse the rendered completed-turn prefix", () => {
	const display = new TranscriptDisplay();
	display.setToolsExpanded(true);
	const completed: Turn = {
		id: "completed-prefix",
		userMessage: { type: "user", content: "Run once." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 0,
					type: "tool",
					tool: {
						tool_name: "bash",
						args: { command: "build" },
						result: "completed output",
						isComplete: true,
						isError: false,
					},
					isComplete: true,
				},
			],
		},
		isComplete: true,
	};
	const streaming: Turn = {
		id: "active-turn",
		userMessage: { type: "user", content: "Explain it." },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [
				{ seq: 0, type: "content", contentText: "First", isComplete: false },
			],
		},
		isComplete: false,
	};
	const turns = [completed, streaming];
	display.setTurns(turns);
	display.render(100);
	const cacheHitsAfterFirstFrame = display.getSanitizationMetrics().cacheHits;

	const streamingChunk = streaming.assistantMessage?.chunks[0];
	assert.ok(streamingChunk);
	streamingChunk.contentText = "First second";
	display.setTurns(turns);
	const output = plain(display.render(100).join("\n"));

	assert.match(output, /First second/);
	assert.equal(
		display.getSanitizationMetrics().cacheHits,
		cacheHitsAfterFirstFrame,
	);
});

void test("clicking a tool card toggles only that tool's details", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
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
		},
	]);

	const collapsed = display.render(100);
	const firstToolRow = collapsed.findIndex(line =>
		plain(line).includes("echo first"),
	);
	assert.notEqual(firstToolRow, -1);
	assert.equal(display.handleMouse(4, firstToolRow), true);

	const expanded = plain(display.render(100).join("\n"));
	assert.match(expanded, /COMMAND[\s\S]*echo first/);
	assert.doesNotMatch(expanded, /COMMAND[\s\S]*echo second[\s\S]*OUTPUT/);

	const rerendered = display.render(100);
	const expandedFirstRow = rerendered.findIndex(line =>
		plain(line).includes("echo first"),
	);
	assert.equal(display.handleMouse(4, expandedFirstRow), true);
	assert.doesNotMatch(plain(display.render(100).join("\n")), /◆ details/);
});

void test("keyboard navigation focuses and toggles individual tool cards", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "keyboard-tools",
			userMessage: { type: "user", content: "Run both." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: ["first", "second"].map((name, index) => ({
					seq: index + 1,
					type: "tool" as const,
					tool: {
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
		},
	]);

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

void test("clicking a tool card in a non-first turn expands the right card", () => {
	// Per-turn caching builds each turn's hitRegions relative to that turn's
	// OWN lines (see buildTurnLines in display.ts), then render() rebases them
	// to absolute offsets by adding that turn's turnStart. A single-turn test
	// can't catch a rebasing bug — turnStart is always the same small constant
	// with only one turn. This test's second turn has a non-trivial turnStart,
	// so a broken rebase (e.g. forgetting to add turnStart) would expand the
	// wrong card or fail to expand anything.
	const display = new TranscriptDisplay();
	const makeToolTurn = (id: string, name: string): Turn => ({
		id,
		userMessage: { type: "user", content: `Run ${name}.` },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "tool" as const,
					tool: {
						tool_name: "bash",
						tool_call_id: name,
						args: { command: `echo ${name}` },
						result: `${name} output`,
						isComplete: true,
						isError: false,
					},
					isComplete: true,
				},
			],
		},
		isComplete: true,
	});
	display.setTurns([
		makeToolTurn("turn-a", "first"),
		makeToolTurn("turn-b", "second"),
	]);

	const collapsed = display.render(80);
	const secondToolRow = collapsed.findIndex(line =>
		plain(line).includes("echo second"),
	);
	assert.notEqual(secondToolRow, -1);
	assert.equal(display.handleMouse(4, secondToolRow), true);

	const expanded = plain(display.render(80).join("\n"));
	assert.match(
		expanded,
		/COMMAND[\s\S]*echo second[\s\S]*OUTPUT[\s\S]*second output/,
	);
	// The first turn's card must stay collapsed — only the clicked card toggled.
	assert.doesNotMatch(expanded, /COMMAND[\s\S]*echo first/);

	// focusTool's scroll-into-view math also depends on absolute offsets —
	// cover it too, using key-based toggling for the first turn's card.
	assert.deepEqual(display.focusTool(-1), { index: 2, total: 2 });
	assert.deepEqual(display.focusTool(-1), { index: 1, total: 2 });
	assert.equal(display.toggleFocusedTool(), true);
	assert.match(
		plain(display.render(80).join("\n")),
		/echo first[\s\S]*first output/,
	);
});

void test("streaming a new turn does not disturb a completed turn's cached lines", () => {
	// Per-turn caching means a streaming turn's cache entry should churn every
	// token while sibling turns' entries stay untouched — the opposite of the
	// old single prefix-blob cache, which required the ENTIRE prefix to be
	// simultaneously stable before caching anything.
	const display = new TranscriptDisplay();
	const completedTurn: Turn = {
		id: "completed",
		userMessage: { type: "user", content: "First question." },
		assistantMessage: {
			type: "assistant",
			isComplete: true,
			chunks: [
				{
					seq: 1,
					type: "content",
					contentText: "COMPLETED_TURN_MARKER answer.",
					isComplete: true,
				},
			],
		},
		isComplete: true,
	};
	const streamingTurn = (text: string): Turn => ({
		id: "streaming",
		userMessage: { type: "user", content: "Second question." },
		assistantMessage: {
			type: "assistant",
			isComplete: false,
			chunks: [
				{ seq: 1, type: "content", contentText: text, isComplete: false },
			],
		},
		isComplete: false,
	});

	display.setTurns([completedTurn, streamingTurn("partial")]);
	const first = plain(display.render(80).join("\n"));
	assert.match(first, /COMPLETED_TURN_MARKER/);
	const completedLineIndex = first
		.split("\n")
		.findIndex(line => line.includes("COMPLETED_TURN_MARKER"));

	display.setTurns([
		completedTurn,
		streamingTurn("partial more tokens arrived"),
	]);
	const second = plain(display.render(80).join("\n"));
	const secondLines = second.split("\n");
	assert.match(second, /COMPLETED_TURN_MARKER/);
	assert.equal(
		secondLines[completedLineIndex],
		first.split("\n")[completedLineIndex],
	);
	assert.match(second, /partial more tokens arrived/);
});

void test("write_file streams live line counts and expanded content", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "streaming-write-file",
			userMessage: { type: "user", content: "Create the module." },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 1,
						type: "tool",
						tool: {
							tool_name: "write_file",
							tool_call_id: "live-write",
							args: {},
							partialResult:
								'{"path":"src/live.ts","content":"const one = 1;\\nconst two = 2;\\nconst three',
							isComplete: false,
							isError: false,
						},
						isComplete: false,
					},
				],
			},
			isComplete: false,
		},
	]);

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
	const content = Array.from(
		{ length: 24 },
		(_, index) => `line-${index + 1}`,
	).join("\n");
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "large-clicked-write",
			userMessage: { type: "user", content: "Write the fixture." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						tool: {
							tool_name: "write_file",
							tool_call_id: "large-write",
							args: { path: "fixture.txt", content },
							result: "Created fixture.txt",
							isComplete: true,
							isError: false,
						},
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);

	const collapsed = display.render(100);
	const toolRow = collapsed.findIndex(line =>
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
	display.setTurns([
		{
			id: "streaming-write-file-append",
			userMessage: { type: "user", content: "Append the next chunk." },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 1,
						type: "tool",
						tool: {
							tool_name: "write_file_append",
							tool_call_id: "live-append",
							args: {},
							partialResult:
								'{"path":"src/live.ts","content":"const four = 4;\\nconst five = 5;\\nconst six',
							isComplete: false,
							isError: false,
						},
						isComplete: false,
					},
				],
			},
			isComplete: false,
		},
	]);

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
	display.setTurns([
		{
			id: "large-clicked-append",
			userMessage: { type: "user", content: "Append the fixture." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 1,
						type: "tool",
						tool: {
							tool_name: "write_file_append",
							tool_call_id: "large-append",
							args: { path: "fixture.txt", content },
							result:
								"Appended to fixture.txt (+279 bytes, 558 bytes total · 48 lines total).",
							isComplete: true,
							isError: false,
						},
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);

	const collapsed = display.render(100);
	const toolRow = collapsed.findIndex(line =>
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
		/✦ NOTICE Skills {2}TypeScript Debugging · matched “TypeScript error”/,
	);
	assert.doesNotMatch(output, /Skills:/);
	assert.match(rendered, /\x1b\[/);
});

void test("notices render their heading and message on separate themed lines", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "notice-lines",
			userMessage: { type: "user", content: "Continue" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "notice",
						notice: {
							level: "warn",
							label: "Run needs input",
							text: "Agent is waiting for the user's answer.",
						},
						isComplete: true,
					},
				],
			},
			isComplete: false,
		},
	]);
	const lines = display.render(100).map(plain);
	const heading = lines.findIndex(line =>
		line.includes("⚠ NOTICE Run needs input"),
	);
	assert.ok(heading >= 0);
	assert.equal(lines[heading].includes("Agent is waiting"), false);
	assert.match(
		lines[heading + 1],
		/^\s{9}Agent is waiting for the user's answer\./,
	);
});

void test("notice reasons use a distinct color and align beneath the label", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "notice-reason",
			userMessage: { type: "user", content: "Continue" },
			assistantMessage: {
				type: "assistant",
				isComplete: false,
				chunks: [
					{
						seq: 0,
						type: "notice",
						notice: {
							level: "warn",
							label: "**Guard: continuation\\_nudge**",
							text: "[continuation-nudge:structured-conclusion] Do not stop yet.",
						},
						isComplete: true,
					},
				],
			},
			isComplete: false,
		},
	]);

	const lines = display.render(100);
	const heading = lines.findIndex(line =>
		plain(line).includes("⚠ NOTICE Guard: continuation_nudge"),
	);
	assert.ok(heading >= 0);
	assert.match(
		plain(lines[heading + 1]),
		/^\s{9}\[continuation-nudge:structured-conclusion\] Do not stop yet\./,
	);
	assert.ok(
		lines[heading + 1].includes(
			`${theme.fgRaw("accent")}[continuation-nudge:structured-conclusion]`,
		),
	);
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
	const rendered = display.render(100).join("\n");
	const output = plain(rendered);
	assert.ok(
		rendered.includes(`${theme.fgRaw("reasoningLabel")}\x1b[1mREASONING`),
	);
	assert.ok(
		rendered.includes(`${theme.fgRaw("responseLabel")}\x1b[1mRESPONSE`),
	);
	assert.match(output, /REASONING.*Compare both execution paths/);
	assert.match(output, /⚠ NOTICE Context\s+Near limit/);
	assert.match(output, /RESPONSE/);
	assert.match(output, /The minimal path delegates continuation/);
});

void test("expanded reasoning renders fenced code as one labeled block", () => {
	const display = new TranscriptDisplay({ thinkingMode: "expanded" });
	display.setTurns([
		{
			id: "reasoning-code-block",
			userMessage: { type: "user", content: "Inspect this implementation." },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "thinking",
						contentText:
							"Check the handler:\n```typescript\nconst value = 1;\nreturn value;\n```",
						isComplete: true,
					},
				],
			},
			isComplete: true,
		},
	]);

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
	assert.ok(lines.every(line => visibleWidth(line) <= 32));
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
	assert.match(plain(line).replace(CURSOR_MARKER, ""), /^ {2}› Ask Logician/);
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
	assert.ok(lines.every(line => visibleWidth(line) <= 72));
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

void test("multiline composer keeps a bounded readable window around the cursor", () => {
	const input = new InputBar();
	input.focused = true;
	input.valueText = "one\ntwo\nthree\nfour\nfive\nsix";
	const lines = input.render(60);
	const rendered = plain(lines.join("\n")).replace(CURSOR_MARKER, "");

	assert.equal(lines.length, 6, "header plus at most five prompt lines");
	assert.doesNotMatch(rendered, /one/);
	assert.match(rendered, /two[\s\S]*six/);
	assert.match(rendered, /↑/);
	assert.ok(lines.every(line => visibleWidth(line) === 60));
});

void test("bracketed paste replays input batched after its closing marker", () => {
	const input = new InputBar();
	input.handleInput("\x1b[200~first\nsecond\x1b[201~!");
	assert.equal(input.valueText, "first\nsecond!");
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
							tool_name: "spawn_agent",
							args: {
								task: "Inspect architecture and tests",
								agent: "explorer",
							},
							partialResult: '{"task":"Inspect architecture and tests"}',
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

void test("collapsed agent card shows only the header while running", () => {
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
	// /spawn starts collapsed: header only until the user expands the card.
	assert.match(output, /subagent explorer streaming|subagent explorer running/);
	assert.doesNotMatch(output, /BEGIN|stream-line-|END|truncated/i);

	display.toolsExpanded = true;
	display.invalidate();
	const expanded = plain(display.render(100).join("\n"));
	// The render buffer is capped to maxRenderedLines even while streaming (so
	// a long-running turn doesn't force an ever-growing full-history re-render
	// on every spinner tick), so only the tail of this 82-line card survives —
	// same as any other over-budget content. Assert on the tail, not "BEGIN".
	assert.match(expanded, /END/);
	assert.match(expanded, /stream-line-79-/);
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
							tool_name: "spawn_agent",
							args: { task: "Review it", agent: "reviewer" },
							result: "**Final report:** all checks passed.",
							details: {
								streamTranscript:
									"Inspecting files...\n\n**Final report:** all checks passed.\n\n" +
									'```acceptance-report\n{"criteriaSatisfied":[]}\n```',
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
							tool_name: "spawn_agent",
							args: { task: "Review it", agent: "reviewer" },
							result:
								"**Approved** with `zero errors`.\n\n```ts\nconst valid = true;\n```",
							details: {
								streamTranscript:
									'Working...\n```acceptance-report\n{"criteriaSatisfied":[]}\n```',
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
	// Collapsed: single header line with status and task summary
	const renderedCollapsed = display.render(100).join("\n");
	const outputCollapsed = plain(renderedCollapsed);
	assert.match(outputCollapsed, /✓ subagent reviewer done/);
	assert.doesNotMatch(outputCollapsed, /Approved|zero errors/);

	// Expanded: full detail block with markdown rendering
	display.toolsExpanded = true;
	display.invalidate();
	const renderedExpanded = display.render(100).join("\n");
	const outputExpanded = plain(renderedExpanded);
	assert.match(outputExpanded, /Approved.*zero errors/);
	assert.match(renderedExpanded, /\x1b\[1mApproved/);
	assert.match(renderedExpanded, /\x1b\[38;5;\d+mconst/);
	assert.doesNotMatch(outputExpanded, /acceptance-report|criteriaSatisfied/);
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
							tool_name: "edit_file",
							args: { path: "/data/dev/solvers/python/qp_ext.cpp", edits: [] },
							result: [
								"Successfully replaced 1 block.",
								'<post_edit_diagnostics file="/data/dev/solvers/python/qp_ext.cpp">',
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
	const { scrollView, render } = mountInScrollView(display, {
		width: 90,
		height: 7,
	});
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
	render();
	scrollView.scrollToEnd();
	display.toggleToolsExpanded();
	const output = plain(render().join("\n"));

	assert.match(output, /LATEST RESPONSE/);
});

void test("wheel-down reaches the new bottom while a streaming update awaits render", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 200 });
	const { scrollView, render } = mountInScrollView(display, {
		width: 80,
		height: 6,
	});
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
	render();
	scrollView.scrollBy(-4);
	assert.equal(scrollView.isFollowingEnd, false);

	display.setTurns([
		streamingTurn(
			[
				...Array.from({ length: 16 }, (_, i) => `line-${i}`),
				"NEWEST STREAMED LINE",
			].join("\n"),
		),
	]);
	scrollView.scrollBy(100);
	const output = plain(render().join("\n"));

	assert.equal(scrollView.isFollowingEnd, true);
	assert.match(output, /NEWEST STREAMED LINE/);
});

void test("new streamed output is signaled while the user is scrolled up", () => {
	const display = new TranscriptDisplay({ maxRenderedLines: 200 });
	const { scrollView, render } = mountInScrollView(display, {
		width: 80,
		height: 6,
	});
	const indicator = new NewOutputIndicator(display);
	const streamingTurn = (contentText: string): Turn => ({
		id: "streaming-indicator",
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
		streamingTurn(Array.from({ length: 18 }, (_, i) => `line-${i}`).join("\n")),
	]);
	render();
	scrollView.scrollBy(-4);
	assert.equal(display.hasNewOutputBelow(), false);
	assert.deepEqual(indicator.render(80), []);

	display.setTurns([
		streamingTurn(
			[...Array.from({ length: 18 }, (_, i) => `line-${i}`), "line-18"].join(
				"\n",
			),
		),
	]);
	render();
	assert.equal(scrollView.isFollowingEnd, false);
	assert.equal(display.hasNewOutputBelow(), true);
	assert.match(plain(indicator.render(80).join("\n")), /↓ new output below/);

	// Same click-to-catch-up app/tui.ts wires onto the indicator's overlay.
	scrollView.scrollToEnd();
	display.clearNewOutputIndicator();
	const bottomOutput = plain(render().join("\n"));
	assert.deepEqual(indicator.render(80), []);
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
										args: '{"path":"src/index.ts"}',
										isError: false,
									},
									{
										agentId: "explorer",
										toolName: "grep",
										args: '{"pattern":"export"}',
										isError: false,
									},
									{
										agentId: "explorer",
										toolName: "bash",
										args: '{"command":"ls"}',
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
											args: '{"path":"src/index.ts"}',
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
	// The ordered content appears once in the child flow.
	assert.equal(output.match(/The implementation is correct\./g)?.length, 1);
	// The final result is also shown separately.
	assert.match(output, /Summary: implementation verified successfully\./);
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

	// Collapsed: single header line with status/summary only
	const collapsed = plain(display.render(120).join("\n"));
	assert.match(collapsed, /✓ subagent explorer done/);
	assert.doesNotMatch(collapsed, /Private reasoning|Intermediate progress/);

	// Expanded: full detail block with child chunks
	display.toolsExpanded = true;
	display.invalidate();
	const expanded = plain(display.render(120).join("\n"));
	assert.match(expanded, /Summary: the implementation is correct\./);
	assert.match(expanded, /Private reasoning/);
	assert.match(expanded, /Intermediate progress/);
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
											args: '{"path":"src/index.ts"}',
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

	// Collapsed: single header line with status only
	const collapsed = plain(display.render(120).join("\n"));
	assert.match(collapsed, /✓ subagent explorer done/);
	assert.doesNotMatch(collapsed, /I should inspect first|Inspecting now/);
	assert.doesNotMatch(collapsed, /SUBAGENT · explorer-1/);

	// Expanded: full detail block with ordered flow
	display.toolsExpanded = true;
	display.invalidate();
	const expanded = plain(display.render(120).join("\n"));
	assert.match(expanded, /I should inspect first\./);
	assert.match(expanded, /Inspecting now\./);
	assert.match(expanded, /read_file/);
	assert.match(expanded, /Inspection complete\./);
	assert.match(expanded, /SUBAGENT · explorer-1/);
	assert.match(expanded, /RETURN TO PARENT/);
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
										args: '{"path":"src/index.ts"}',
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
	// Collapsed: single header line with status only
	const collapsed = plain(display.render(120).join("\n"));
	assert.match(collapsed, /✓ subagent explorer done/);
	assert.doesNotMatch(collapsed, /read_file|ACTIVITY/);
	assert.equal(collapsed.match(/explorer/g)?.length, 1);

	// Expanded: full detail block with child tool calls
	display.toolsExpanded = true;
	display.invalidate();
	const expanded = plain(display.render(120).join("\n"));
	assert.match(expanded, /read_file.*path=src\/index\.ts/);
	assert.doesNotMatch(expanded, /ACTIVITY/);
	assert.equal(expanded.match(/explorer/g)?.length, 1);
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

	// Collapsed: single header line with one ✓
	const collapsed = plain(display.render(120).join("\n"));
	assert.equal(collapsed.match(/✓/g)?.length, 1);
	assert.match(collapsed, /✓ subagent explorer done/);
	assert.doesNotMatch(collapsed, /2 turn.*0 tool call|Inspection complete/);

	// Expanded: detail block with metadata
	display.toolsExpanded = true;
	display.invalidate();
	const expandedOutput = plain(display.render(120).join("\n"));
	assert.equal(expandedOutput.match(/✓/g)?.length, 1);
	assert.match(expandedOutput, /✓ subagent explorer done/);
	assert.match(expandedOutput, /2 turn.*0 tool call.*1\.4s/);
	assert.equal(
		expandedOutput.match(/Inspection complete\./g)?.length,
		1,
		"the final report should render once",
	);
	assert.doesNotMatch(expandedOutput, /◆ subagent|◆ agent/);
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
							tool_name: "spawn_agents",
							args: {
								tasks: [
									{ agent: "explorer", task: "Inspect the API" },
									{ agent: "reviewer", task: "Review the tests" },
									{ agent: "general", task: "Check documentation" },
								],
							},
							details: {
								taskStatus: {
									0: {
										taskIndex: 0,
										agentId: "agent_1",
										agent: "explorer",
										status: "completed",
										startedAt: 1000,
										endedAt: 1500,
									},
									1: {
										taskIndex: 1,
										agentId: "agent_2",
										agent: "reviewer",
										status: "running",
										startedAt: 1500,
									},
								},
							},
							isError: false,
							isComplete: false,
						},
					},
				],
			},
			isComplete: false,
		},
	]);
	// Collapsed: task rows are always visible so each is individually
	// clickable to expand, even before the whole tool is expanded.
	const collapsed = plain(display.render(120).join("\n"));
	assert.match(collapsed, /subagents 2\/3 running.*3 tasks/);
	assert.match(collapsed, /✓ 1\. explorer.*Inspect the API/);
	assert.match(collapsed, /⠋ 2\. reviewer.*Review the tests/);
	assert.match(collapsed, /· 3\. general.*Check documentation/);

	// Expanded: task breakdown with per-task status
	display.toolsExpanded = true;
	display.invalidate();
	const expanded = plain(display.render(120).join("\n"));
	assert.match(expanded, /subagents 2\/3 running.*3 tasks/);
	assert.match(expanded, /✓ 1\. explorer.*Inspect the API/);
	assert.match(expanded, /⠋ 2\. reviewer.*Review the tests/);
	assert.match(expanded, /· 3\. general.*Check documentation/);
});

void test("clicking a spawn_agents task row expands that exact task, not a neighbor", () => {
	const display = new TranscriptDisplay();
	display.setTurns([
		{
			id: "click-target-batch",
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
							tool_name: "spawn_agents",
							tool_call_id: "batch-1",
							args: {
								tasks: [
									{ agent: "explorer", task: "Task Alpha" },
									{ agent: "reviewer", task: "Task Beta" },
									{ agent: "general", task: "Task Gamma" },
								],
							},
							details: {
								childChunks: [
									{
										seq: 0,
										agentId: "agent_1",
										taskIndex: 0,
										type: "content",
										contentText: "ALPHA-ONLY-CONTENT",
										isComplete: true,
									},
									{
										seq: 1,
										agentId: "agent_2",
										taskIndex: 1,
										type: "content",
										contentText: "BETA-ONLY-CONTENT",
										isComplete: true,
									},
									{
										seq: 2,
										agentId: "agent_3",
										taskIndex: 2,
										type: "content",
										contentText: "GAMMA-ONLY-CONTENT",
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

	const findRow = (needle: string) => {
		const rendered = display.render(120);
		const row = rendered.findIndex(line => plain(line).includes(needle));
		assert.notEqual(row, -1, `row for "${needle}" not found`);
		return row;
	};

	// Clicking the parent header should do nothing — only task rows toggle.
	const headerRow = findRow("subagents");
	assert.equal(display.handleMouse(4, headerRow), false);
	assert.doesNotMatch(
		plain(display.render(120).join("\n")),
		/ALPHA-ONLY-CONTENT/,
	);

	// Click task 1 (Alpha) — only Alpha's content should appear.
	const alphaRow = findRow("Task Alpha");
	assert.equal(display.handleMouse(4, alphaRow), true);
	const afterAlpha = plain(display.render(120).join("\n"));
	assert.match(afterAlpha, /ALPHA-ONLY-CONTENT/);
	assert.doesNotMatch(afterAlpha, /BETA-ONLY-CONTENT/);
	assert.doesNotMatch(afterAlpha, /GAMMA-ONLY-CONTENT/);

	// Click task 2 (Beta) — Alpha stays expanded (independent toggles), Beta
	// joins it, Gamma remains collapsed.
	const betaRow = findRow("Task Beta");
	assert.equal(display.handleMouse(4, betaRow), true);
	const afterBeta = plain(display.render(120).join("\n"));
	assert.match(afterBeta, /ALPHA-ONLY-CONTENT/);
	assert.match(afterBeta, /BETA-ONLY-CONTENT/);
	assert.doesNotMatch(afterBeta, /GAMMA-ONLY-CONTENT/);

	// Click task 3 (Gamma) — all three now expanded.
	const gammaRow = findRow("Task Gamma");
	assert.equal(display.handleMouse(4, gammaRow), true);
	const afterGamma = plain(display.render(120).join("\n"));
	assert.match(afterGamma, /ALPHA-ONLY-CONTENT/);
	assert.match(afterGamma, /BETA-ONLY-CONTENT/);
	assert.match(afterGamma, /GAMMA-ONLY-CONTENT/);

	// Click task 1 (Alpha) again — it collapses back, Beta/Gamma stay expanded.
	const alphaRowAgain = findRow("Task Alpha");
	assert.equal(display.handleMouse(4, alphaRowAgain), true);
	const afterAlphaCollapse = plain(display.render(120).join("\n"));
	assert.doesNotMatch(afterAlphaCollapse, /ALPHA-ONLY-CONTENT/);
	assert.match(afterAlphaCollapse, /BETA-ONLY-CONTENT/);
	assert.match(afterAlphaCollapse, /GAMMA-ONLY-CONTENT/);
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
							tool_name: "spawn_agents",
							partialResult: '{"tasks":[{"agent":"explorer"',
							details: {
								taskStatus: {
									0: {
										taskIndex: 0,
										agentId: "agent_1",
										agent: "explorer",
										status: "running",
										startedAt: 1000,
									},
								},
							},
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
							tool_name: "spawn_agents",
							details: {
								total: 0,
								taskStatus: {
									2: {
										taskIndex: 2,
										agentId: "agent_1",
										agent: "explorer",
										status: "running",
										startedAt: 1000,
									},
								},
							},
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
							tool_name: "spawn_agents",
							args: {
								tasks: [
									{ agent: "explorer", task: "Inspect API" },
									{ agent: "reviewer", task: "Inspect tests" },
								],
							},
							details: {
								taskStatus: {
									0: {
										taskIndex: 0,
										agentId: "agent_1",
										agent: "explorer",
										status: "running",
										startedAt: 1000,
									},
									1: {
										taskIndex: 1,
										agentId: "agent_2",
										agent: "reviewer",
										status: "running",
										startedAt: 1000,
									},
								},
								childChunks: [
									{
										seq: 0,
										agentId: "agent_1",
										taskIndex: 0,
										type: "content",
										contentText: "API stream\n```ts\nconst api = true;\n```",
										isComplete: false,
									},
									{
										seq: 1,
										agentId: "agent_2",
										taskIndex: 1,
										type: "content",
										contentText: "Test stream",
										isComplete: false,
									},
								],
							},
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
										args: '{"command":"npm test"}',
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
	assert.match(output, /bash.*command=npm test/);
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
							tool_name: "edit_file",
							args: {
								path: "src/example.ts",
								oldText: 'const answer = "no";',
								newText: 'const answer = "yes";',
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
							tool_name: "edit_file",
							args: { path: "src/example.ts" },
							result: [
								"Successfully replaced 1 occurrence.",
								"Diff:",
								"--- a/edit",
								"+++ b/edit",
								"@@ -1 +1 @@",
								'-const answer = "no";',
								'+const answer = "yes";',
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
