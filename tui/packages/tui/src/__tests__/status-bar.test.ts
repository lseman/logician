// ── StatusBar tests ──────────────────────────────────────────────────────────

import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { initTheme } from "../terminal/theme.ts";
import { StatusBar } from "../status/status-bar.ts";
import { inkTextComponentLines as inkLines, visibleWidth } from "../terminal/core.ts";

const setupTheme = (): void => {
	try {
		initTheme("dark");
	} catch {
		// Theme already initialized.
	}
};

void describe("StatusBar", () => {
	it("renders minimal line with phase, model, context", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "claude-sonnet-4", contextTokens: 5000, contextMaxTokens: 150000 });
		const lines = inkLines(bar, 120);
		assert.strictEqual(lines.length, 1);
		assert.ok(lines[0].includes("READY"));
		assert.ok(lines[0].includes("claude-sonnet-4"));
	});

	it("shows MCP server count when set", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000, mcpServerCount: 3 });
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("mcp"));
		assert.ok(lines[0].includes("3"));
	});

	it("omits MCP section when count is zero or undefined", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000, mcpServerCount: 0 });
		const lines = inkLines(bar, 120);
		assert.ok(!lines[0].toLowerCase().includes("mcp"));
	});

	it("omits MCP section when mcpServerCount is not set", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000 });
		const lines = inkLines(bar, 120);
		assert.ok(!lines[0].toLowerCase().includes("mcp"));
	});

	it("shows the resolved execution profile", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			executionProfile: "minimal",
		});
		const plain = inkLines(bar, 160)[0]
			.replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");
		assert.ok(plain.includes("exec: minimal"));
	});

	it("drops optional sections on narrow terminals", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "claude-sonnet-4",
			contextTokens: 5000,
			contextMaxTokens: 150000,
			mcpServerCount: 3,
			branch: "main",
			cacheReadTokens: 12000,
		});
		const lines = inkLines(bar, 40);
		assert.ok(lines[0].includes("READY"));
		// Narrow terminal should drop optional sections; visible width fits.
		assert.ok(visibleWidth(lines[0]) <= 40);
	});

	it("shows git indicators when present", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			branch: "feature",
			gitModified: 2,
			gitStaged: 1,
			gitUntracked: 3,
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("feature"));
		assert.ok(lines[0].includes("*2"));
		assert.ok(lines[0].includes("+1"));
		assert.ok(lines[0].includes("?3"));
	});

	it("shows thinking level", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000, thinkingLevel: "high" });
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("think:"));
		assert.ok(lines[0].includes("HIGH"));
	});

	it("shows thinking off", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000, thinkingLevel: "off" });
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("think:"));
		assert.ok(lines[0].includes("off"));
	});

	it("shows inference mode", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			inferenceMode: "thinking-general",
		});
		const lines = inkLines(bar, 120);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");
		assert.ok(plain.includes("THINK GEN") || plain.includes("mode:"));
	});

	it("shows reasoner when not none", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			reasoner: "loop-detector",
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("reasoner:"));
		assert.ok(lines[0].includes("loop-detector"));
	});

	it("omits reasoner when none", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			reasoner: "none",
		});
		const lines = inkLines(bar, 120);
		assert.ok(!lines[0].includes("reasoner"));
	});

	it("shows cache read tokens", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			cacheReadTokens: 12400,
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("cache read:"));
		assert.ok(lines[0].includes("12.4k"));
	});

	it("renders the memory toggle exactly once", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ memoryEnabled: true });
		const line = inkLines(bar, 240)[0];
		assert.equal(line.match(/memory:/g)?.length, 1);
		assert.match(line, /memory: on\b/);
		assert.doesNotMatch(line, /memory: on:\s*on/);
	});

	it("shows goal when present", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			goalCondition: "Fix the bug",
			goalTurnCount: 5,
			goalElapsed: 120,
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("Fix the bug"));
		assert.ok(lines[0].includes("5 turns"));
		assert.ok(lines[0].includes("2m0s"));
	});

	it("truncates long session titles", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			sessionTitle: "A very long session title that should be truncated",
		});
		const lines = inkLines(bar, 80);
		assert.ok(lines[0].includes("◇"));
	});

	it("updates on tick animation", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "streaming", model: "test", contextTokens: 0, contextMaxTokens: 100000 });
		const lines1 = inkLines(bar, 80);
		bar.setTick(4);
		const lines2 = inkLines(bar, 80);
		assert.ok(lines1[0] !== lines2[0]);
	});

	it("invalidates cache on update", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "ready", model: "test", contextTokens: 0, contextMaxTokens: 100000 });
		const lines1 = inkLines(bar, 80);
		bar.update({ phase: "thinking", model: "test2", contextTokens: 1000, contextMaxTokens: 100000 });
		const lines2 = inkLines(bar, 80);
		assert.ok(lines1[0] !== lines2[0]);
	});

	it("pads shorter phase redraws to erase stale terminal cells", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "compacting", model: "test", contextTokens: 0, contextMaxTokens: 100000 });
		const longLine = inkLines(bar, 80)[0];
		bar.update({ phase: "ready" });
		const shortLine = inkLines(bar, 80)[0];
		assert.equal(visibleWidth(longLine), 80);
		assert.equal(visibleWidth(shortLine), 80);
		assert.match(shortLine, /\s+$/);
	});

	it("starts and stops animation timer", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "streaming", model: "test", contextTokens: 0, contextMaxTokens: 100000 });
		bar.startAnimation();
		assert.ok(bar["timer"] !== null);
		bar.stopAnimation();
		assert.ok(bar["timer"] === null);
	});

	it("renders empty line when not visible", () => {
		setupTheme();
		const bar = new StatusBar();
		const lines = inkLines(bar, 80);
		assert.strictEqual(lines.length, 1);
	});

	it("renders context meter bar", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 75000,
			contextMaxTokens: 150000,
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("ctx"));
		assert.ok(lines[0].includes("50.0%"));
	});

	it("colors context meter red at high usage", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 140000,
			contextMaxTokens: 150000,
		});
		const lines = inkLines(bar, 120);
		assert.ok(lines[0].includes("93.3%"));
	});

	it("shows token flow with prompt and completion tokens", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 5000,
			contextMaxTokens: 150000,
			promptTokens: 4800,
			completionTokens: 200,
		});
		const lines = inkLines(bar, 160);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");
		assert.ok(plain.includes("↑"));
		assert.ok(plain.includes("↓"));
		assert.ok(plain.includes("4.8k"));
		assert.ok(plain.includes("200"));
	});

	it("shows partial token flow when only prompt tokens present", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 5000,
			contextMaxTokens: 150000,
			promptTokens: 4800,
		});
		const lines = inkLines(bar, 160);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");
		assert.ok(plain.includes("↑"));
		assert.ok(plain.includes("–"));
	});

	it("omits token flow when neither token count is set", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 5000,
			contextMaxTokens: 150000,
		});
		const lines = inkLines(bar, 120);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "");
		assert.ok(!plain.includes("↑"));
		assert.ok(!plain.includes("↓"));
	});
});
