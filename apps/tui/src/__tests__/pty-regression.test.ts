import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

// Minimal dark theme that satisfies the TUI's initTheme() requirement.
const DARK_THEME_JSON = JSON.stringify({
	name: "dark",
	vars: {
		cyan: "#00d7ff",
		blue: "#5f87ff",
		green: "#b5bd68",
		red: "#cc6666",
		yellow: "#ffff00",
		text: "#d4d4d4",
		gray: "#808080",
		dimGray: "#666666",
		darkGray: "#505050",
		accent: "#8abeb7",
	},
	colors: {
		accent: "accent",
		border: "blue",
		borderMuted: "darkGray",
		success: "green",
		error: "red",
		warning: "yellow",
		muted: "gray",
		dim: "dimGray",
		text: "text",
		userText: "text",
		assistantText: "text",
		systemText: "text",
		mdHeading: "#f0c674",
		mdCode: "accent",
		mdCodeBlock: "green",
		mdCodeBlockBg: "#1e1e24",
		mdCodeBlockBorder: "gray",
		mdLink: "#81a2be",
		mdQuote: "gray",
		mdListBullet: "accent",
		toolTitle: "text",
		toolRunning: "#f0c674",
		toolSuccess: "green",
		toolError: "red",
		toolStreaming: "#f0c674",
		toolOutput: "gray",
		thinkingText: "gray",
		separator: "gray",
		prompt: "green",
		inputText: "text",
		phaseReady: "green",
		phaseThinking: "#81a2be",
		phaseTool: "#f0c674",
		phaseError: "red",
		phaseStreaming: "#f0c674",
		phaseCompacting: "#81a2be",
		phaseBranching: "#81a2be",
		contextGood: "green",
		contextWarning: "yellow",
		contextCritical: "red",
		levelOff: "#505050",
		levelLow: "#6e6e6e",
		levelMedium: "#5f87af",
		levelHigh: "#5f87af",
		levelXhigh: "#81a2be",
		diffAdded: "green",
		diffRemoved: "red",
		diffContext: "gray",
		diffHunk: "#f0c674",
		diffMeta: "gray",
		terminalOutput: "text",
		memoryTag: "#81a2be",
		memoryId: "#81a2be",
		memoryContent: "text",
		memoryCount: "#81a2be",
		pluginStartup: "green",
		header: "accent",
		active: "accent",
		selected: "#3a3a4a",
		inputPlaceholder: "dimGray",
		jsonKey: "#81a2be",
		jsonKeyword: "#c678dd",
		jsonNumber: "#b5bd68",
		jsonString: "#abb2bf",
		jsonPunctuation: "#808080",
	},
});

import { InputBar } from "../input/input-bar.ts";
import {
	type Component,
	normalizeKeyboardInput,
	TUI,
	visibleWidth,
} from "../terminal/core.ts";
import { runInPty, screenFromPtyResult } from "../testing/pty-harness.ts";

const repoRoot = path.resolve(import.meta.dirname, "../../../..");
const bun = process.execPath;
const entry = path.join(repoRoot, "apps", "tui", "src", "index.ts");
void test("TUI starts in a real terminal and Ctrl+M opens mode selection", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	const result = await runInPty({
		command: bun,
		args: ["run", entry],
		cwd: repoRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
			LOGICIAN_MODEL: "test-model",
		},
		actions: [{ afterMs: 300, send: "\x1b[109;5u" }],
		timeoutMs: 4_000,
		columns: 120,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 120, 32).text();
	assert.match(screen, /Enter commands/);
	assert.match(screen, /Inference Mode \(10\)/);
	assert.match(screen, /Provider ✓/);
	assert.doesNotMatch(result.output, /TypeError|TUI render error/);
});

void test("startup applies inference preferences without rewriting settings", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-settings-"));
	const settingsDir = path.join(home, ".logician");
	const themeDir = path.join(settingsDir, "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	const settingsPath = path.join(settingsDir, "settings.json");
	const original = `${JSON.stringify(
		{
			theme: "dark",
			thinkingLevel: "high",
			inferenceMode: "thinking-coding",
			customPreference: { preserve: true },
		},
		null,
		2,
	)}\n`;
	writeFileSync(settingsPath, original);

	await runInPty({
		command: bun,
		args: ["run", entry],
		cwd: repoRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
			LOGICIAN_MODEL: "test-model",
		},
		actions: [],
		timeoutMs: 1_000,
		columns: 120,
		rows: 32,
	});

	assert.equal(readFileSync(settingsPath, "utf8"), original);
});

void test("Kitty Ctrl+O and Ctrl+C reports reach legacy TUI keybindings", () => {
	assert.equal(normalizeKeyboardInput("\x1b[27u"), "\x1b");
	assert.equal(normalizeKeyboardInput("\x1b[27;1u"), "\x1b");
	assert.equal(normalizeKeyboardInput("\x1b[111;5u"), "\x0f");
	assert.equal(normalizeKeyboardInput("\x1b[99;5u"), "\x03");
	assert.equal(normalizeKeyboardInput("\x1b[116;6u"), "\x14");
	assert.equal(
		normalizeKeyboardInput("\x1b[105;5u"),
		"\x1b[105;5u",
		"Ctrl+I must remain distinct from Tab",
	);
	assert.equal(
		normalizeKeyboardInput("\x1b[109;5u"),
		"\x1b[109;5u",
		"Ctrl+M must remain distinct from Enter",
	);

	const input = new InputBar();
	let cancelled = 0;
	input.onCancel = () => {
		cancelled++;
	};
	input.valueText = "draft message";
	input.handleInput(normalizeKeyboardInput("\x1b[99;5u"));
	assert.equal(input.valueText, "");
	assert.equal(cancelled, 1);
});

void test("Ctrl+I changes and persists the execution profile", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	const result = await runInPty({
		command: bun,
		args: ["run", entry],
		cwd: repoRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
			LOGICIAN_MODEL: "test-model",
			LOGICIAN_PERSIST_CONFIG: "1",
		},
		actions: [{ afterMs: 500, send: "\x1b[105;5u" }],
		timeoutMs: 4_000,
		columns: 140,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 140, 32).text();
	const settings = JSON.parse(
		readFileSync(path.join(home, ".logician", "settings.json"), "utf8"),
	) as { executionProfile?: string };

	assert.match(screen, /Execution mode: auto|exec: auto/);
	assert.equal(settings.executionProfile, "autonomous");
	assert.doesNotMatch(result.output, /TUI render error/);
});

void test("Ctrl+P changes and persists act/plan mode", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	const result = await runInPty({
		command: bun,
		args: ["run", entry],
		cwd: repoRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
			LOGICIAN_MODEL: "test-model",
			LOGICIAN_PERSIST_CONFIG: "1",
		},
		actions: [{ afterMs: 500, send: "\x10" }],
		timeoutMs: 4_000,
		columns: 140,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 140, 32).text();
	const settings = JSON.parse(
		readFileSync(path.join(home, ".logician", "settings.json"), "utf8"),
	) as { workflowMode?: string };

	assert.match(screen, /Mode: plan|\bplan\b/);
	assert.equal(settings.workflowMode, "plan");
	assert.doesNotMatch(result.output, /TUI render error/);
});

void test("Escape clears first and cancels the active turn on second press", () => {
	const input = new InputBar();
	let cancelled = 0;
	input.onCancel = () => {
		cancelled++;
	};
	input.valueText = "draft message";

	input.handleInput("\x1b");
	assert.equal(input.valueText, "");
	assert.equal(cancelled, 0);

	input.handleInput("\x1b");
	assert.equal(cancelled, 1);

	input.handleInput("x");
	input.handleInput("\x1b");
	input.handleInput("y");
	input.handleInput("\x1b");
	assert.equal(
		cancelled,
		1,
		"typing between Escape presses must disarm turn cancellation",
	);
});

void test("Escape reaches the dialog owner before the core overlay fallback", () => {
	const tui = new TUI({} as NodeJS.WriteStream);
	const dialog: Component & {
		visible: boolean;
		handleInput(data: string): { type: "close" } | null;
	} = {
		visible: true,
		render: () => [],
		handleInput: data => (data === "\x1b" ? { type: "close" } : null),
	};
	let dismissed = false;
	tui.showOverlay(dialog);
	tui.addInputListener(data => {
		const action = dialog.handleInput(data);
		if (action?.type === "close") {
			dismissed = true;
			tui.removeOverlay(dialog);
			return { consume: true };
		}
		return undefined;
	});

	(
		tui as unknown as {
			handleInput(data: string): void;
		}
	).handleInput(normalizeKeyboardInput("\x1b[27u"));

	assert.equal(dismissed, true);
	assert.equal(dialog.visible, false);
});

void test("bringToFront moves an existing overlay above later registrations", () => {
	const tui = new TUI({} as NodeJS.WriteStream);
	const dashboard: Component = { render: () => ["dashboard"] };
	const currentRenderer: Component = { render: () => ["current"] };

	tui.showOverlay(dashboard);
	tui.showOverlay(currentRenderer);
	tui.bringToFront(dashboard);

	const lines = (
		tui as unknown as {
			composeOverlays(
				lines: string[],
				termWidth: number,
				termHeight: number,
				transcriptHeight: number,
			): string[];
		}
	).composeOverlays([" ".repeat(40)], 40, 1, 1);
	assert.ok(lines[0].includes("dashboard"));
	assert.ok(!lines[0].includes("current"));
});

void test("overlays replace the occupied screen cells instead of rendering past them", () => {
	const tui = new TUI({} as NodeJS.WriteStream);
	const dashboard: Component = { render: () => ["dashboard"] };
	tui.showOverlay(dashboard, { row: 0, col: 0, width: 9 });

	const lines = (
		tui as unknown as {
			composeOverlays(
				lines: string[],
				termWidth: number,
				termHeight: number,
				transcriptHeight: number,
			): string[];
		}
	).composeOverlays(["background content".padEnd(80)], 80, 1, 1);

	assert.equal(visibleWidth(lines[0]), 80);
	assert.ok(lines[0].includes("dashboard"));
	assert.ok(!lines[0].includes("background content"));
});

void test("TUI handles a Kitty Ctrl+O sequence without corrupting the render", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	const result = await runInPty({
		command: bun,
		args: ["run", entry],
		cwd: repoRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
			LOGICIAN_MODEL: "test-model",
		},
		actions: [
			{ afterMs: 100, send: "s\n" },
			{ afterMs: 500, send: "\x1b[111;5u" },
		],
		timeoutMs: 4_000,
		columns: 120,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 120, 32).text();
	assert.match(screen, /Enter commands/);
	assert.doesNotMatch(result.output, /TUI render error/);
});
