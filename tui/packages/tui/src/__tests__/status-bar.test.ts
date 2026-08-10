// ── StatusBar tests ──────────────────────────────────────────────────────────

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { StatusBar } from "../footer/layout.ts";
import { createDefaultConfig, DEFAULT_WIDGET_LAYOUTS } from "../footer/types.ts";
import { visibleWidth } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

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
		bar.update({
			phase: "ready",
			model: "claude-sonnet-4",
			contextTokens: 5000,
			contextMaxTokens: 150000,
		});
		const lines = bar.render(120);
		assert.strictEqual(lines.length, 1);
		assert.ok(lines[0].includes("READY"));
		assert.ok(lines[0].includes("claude-sonnet-4"));
	});

	it("shows MCP server count when set", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			mcpServerCount: 3,
		});
		const lines = bar.render(120);
		assert.ok(lines[0].includes("mcp"));
		assert.ok(lines[0].includes("3"));
	});

	it("omits MCP section when count is zero or undefined", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			mcpServerCount: 0,
		});
		const lines = bar.render(120);
		assert.ok(!lines[0].toLowerCase().includes("mcp"));
	});

	it("omits MCP section when mcpServerCount is not set", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
		});
		const lines = bar.render(120);
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
		const plain = bar.render(160)[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
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
		const lines = bar.render(40);
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
		const lines = bar.render(120);
		assert.ok(lines[0].includes("feature"));
		assert.ok(lines[0].includes("*2"));
		assert.ok(lines[0].includes("+1"));
		assert.ok(lines[0].includes("?3"));
	});

	it("shows thinking level", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			thinkingLevel: "high",
		});
		const lines = bar.render(120);
		assert.ok(lines[0].includes("think:"));
		assert.ok(lines[0].includes("HIGH"));
	});

	it("shows thinking off", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
			thinkingLevel: "off",
		});
		const lines = bar.render(120);
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
		const lines = bar.render(120);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
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
		const lines = bar.render(120);
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
		const lines = bar.render(120);
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
		const lines = bar.render(120);
		assert.ok(lines[0].includes("cache read:"));
		assert.ok(lines[0].includes("12.4k"));
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
		const lines = bar.render(120);
		assert.ok(lines[0].includes("Fix the bug"));
		assert.ok(lines[0].includes("5 turns"));
		assert.ok(lines[0].includes("2m0s"));
	});

	it("updates on tick animation", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "streaming",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
		});
		const lines1 = bar.render(80);
		bar.setTick(4);
		const lines2 = bar.render(80);
		assert.ok(lines1[0] !== lines2[0]);
	});

	it("invalidates cache on update", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "ready",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
		});
		const lines1 = bar.render(80);
		bar.update({
			phase: "thinking",
			model: "test2",
			contextTokens: 1000,
			contextMaxTokens: 100000,
		});
		const lines2 = bar.render(80);
		assert.ok(lines1[0] !== lines2[0]);
	});

	it("starts and stops animation timer", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({
			phase: "streaming",
			model: "test",
			contextTokens: 0,
			contextMaxTokens: 100000,
		});
		bar.startAnimation();
		assert.ok(bar.timer !== null);
		bar.stopAnimation();
		assert.ok(bar.timer === null);
	});

	it("renders empty line when not visible", () => {
		setupTheme();
		const bar = new StatusBar();
		const lines = bar.render(80);
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
		const lines = bar.render(120);
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
		const lines = bar.render(120);
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
		const lines = bar.render(160);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
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
		const lines = bar.render(160);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
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
		const lines = bar.render(120);
		const plain = lines[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
		assert.ok(!plain.includes("↑"));
		assert.ok(!plain.includes("↓"));
	});

	it("preserves legacy phase markers", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.update({ phase: "approval" });
		assert.ok(bar.render(80)[0].includes("◆ APPROVAL"));
		bar.update({ phase: "error" });
		assert.ok(bar.render(80)[0].includes("× ERROR"));
		bar.update({ phase: "verifying" });
		assert.ok(bar.render(80)[0].includes("⠋ VERIFYING"));
	});

	it("keeps configured rows separate when returning a cached render", () => {
		setupTheme();
		const bar = new StatusBar();
		const config = createDefaultConfig();
		config.rows = 2;
		config.widgets.memory = {
			enabled: true,
			row: 1,
			position: 0,
			align: "left",
			fill: "none",
		};
		bar.setConfig(config);
		bar.update({ memoryEnabled: true });
		assert.strictEqual(bar.render(160).length, 2);
		assert.strictEqual(bar.render(160).length, 2);
	});
});

void describe("configurable footer widgets", () => {
	const emptyConfig = () => {
		const config = createDefaultConfig();
		for (const id of Object.keys(DEFAULT_WIDGET_LAYOUTS)) {
			config.widgets[id] = { ...DEFAULT_WIDGET_LAYOUTS[id as keyof typeof DEFAULT_WIDGET_LAYOUTS], enabled: false };
		}
		return config;
	};

	it("positions contributed widgets in left, middle, and right groups", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.setConfig(emptyConfig());
		bar.upsertWidget({ id: "test.left", text: "LEFT", layout: { row: 0, align: "left" } });
		bar.upsertWidget({ id: "test.middle", text: "MIDDLE", layout: { row: 0, align: "middle" } });
		bar.upsertWidget({ id: "test.right", text: "RIGHT", layout: { row: 0, align: "right" } });
		const line = bar.render(50)[0].replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
		assert.ok(line.startsWith("LEFT"));
		assert.ok(line.indexOf("MIDDLE") >= 20 && line.indexOf("MIDDLE") <= 24);
		assert.ok(line.endsWith("RIGHT"));
	});

	it("applies icons, style overrides, minimum width, and grow fill", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.setConfig(emptyConfig());
		bar.upsertWidget({
			id: "test.grow",
			text: "build",
			icon: "!",
			layout: { row: 0, align: "left", fill: "grow", minWidth: 12 },
			style: { iconColor: "warning", textColor: "error" },
		});
		bar.upsertWidget({ id: "test.edge", text: "END", layout: { row: 0, align: "right" } });
		const line = bar.render(40)[0];
		const plain = line.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
		assert.ok(plain.startsWith("! build"));
		assert.ok(plain.endsWith("END"));
		assert.strictEqual(visibleWidth(line), 40);
		assert.ok(line.includes("\x1b["));
	});

	it("sanitizes and removes contributed widget snapshots", () => {
		setupTheme();
		const bar = new StatusBar();
		bar.setConfig(emptyConfig());
		bar.upsertWidget({
			id: "test.safe",
			text: "safe\x1b[2J\ntext",
			layout: { row: 0 },
		});
		const rendered = bar.render(80)[0];
		assert.ok(rendered.includes("safe text"));
		assert.ok(!rendered.includes("\x1b[2J"));
		assert.strictEqual(bar.removeWidget("test.safe"), true);
		assert.ok(!bar.render(80)[0].includes("safe text"));
	});
});
