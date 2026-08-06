// ── SettingsOverlay tests ────────────────────────────────────────────────────

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
	type SettingDef,
	SettingsSelectorOverlay,
} from "../overlays/settings-overlay.ts";
import { initTheme } from "../terminal/theme.ts";

// Initialize theme before any overlay rendering.
const setupTheme = (): void => {
	try {
		initTheme("dark");
	} catch {
		// Theme already initialized or error — ignore.
	}
};

const makeSettings = (): SettingDef[] => [
	{
		name: "Model",
		currentValue: "claude-sonnet-4",
		description: "AI model",
		options: [
			{ label: "claude-sonnet-4", value: "claude-sonnet-4", current: true },
			{ label: "gpt-4", value: "gpt-4" },
		],
	},
	{
		name: "Thinking Level",
		currentValue: "medium",
		description: "Reasoning depth",
		options: [
			{ label: "off", value: "off" },
			{ label: "low", value: "low" },
			{ label: "medium", value: "medium", current: true },
			{ label: "high", value: "high" },
		],
	},
	{
		name: "Loop Detection",
		currentValue: "ON",
		description: "Detect loops",
		options: [
			{ label: "ON", value: "on", toggleOn: true, current: true },
			{ label: "OFF", value: "off", toggleOn: false },
		],
	},
];

describe("SettingsSelectorOverlay", () => {
	it("shows empty when not visible", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		const lines = overlay.render(80);
		assert.strictEqual(lines.length, 0);
	});

	it("shows overlay when visible", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();
		const lines = overlay.render(80);
		assert.ok(lines.length > 5);
		assert.ok(lines[0].includes("─"));
		assert.ok(lines.at(-1)?.includes("─"));
	});

	it("navigates menu with arrow keys", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Move down
		overlay.handleInput("\x1b[B");
		assert.strictEqual(overlay.selectedIndex, 1);

		// Wrap to top
		overlay.handleInput("\x1b[A");
		assert.strictEqual(overlay.selectedIndex, 0);

		// Wrap to bottom
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\x1b[B");
		assert.strictEqual(overlay.selectedIndex, 2);
	});

	it("escapes to close", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		const action = overlay.handleInput("\x1b");
		assert.deepStrictEqual(action, { type: "close" });
	});

	it("opens detail view on enter", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Model opens the dedicated model selector instead of inline options.
		const action = overlay.handleInput("\r");
		assert.deepStrictEqual(action, { type: "open", settingName: "Model" });
		assert.strictEqual(overlay.inDetailView, false);
	});

	it("applies option in detail view on enter", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Open the inline options for Thinking Level.
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\r");
		// Select a different option.
		overlay.handleInput("\x1b[B");
		// Apply
		const action = overlay.handleInput("\r");
		assert.strictEqual(action?.type, "change");
		if (action?.type === "change") {
			assert.strictEqual(action.settingName, "Thinking Level");
			assert.strictEqual(action.value, "high");
		}
	});

	it("goes back to menu on tab", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Open a setting with inline options (Model has a dedicated selector).
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\r");
		assert.strictEqual(overlay.inDetailView, true);

		// Go back
		const action = overlay.handleInput("\t");
		assert.strictEqual(action, null);
		assert.strictEqual(overlay.inDetailView, false);
	});

	it("renders toggle indicators", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Open detail view for Loop Detection (index 2)
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\r");

		const lines = overlay.render(80);
		const rendered = lines.join("\n");
		assert.ok(rendered.includes("[on]"));
		assert.ok(rendered.includes("[off]"));
	});

	it("renders current value indicator", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		const lines = overlay.render(80);
		const rendered = lines.join("\n");
		assert.ok(rendered.includes("(claude-sonnet-4)"));
		assert.ok(rendered.includes("(medium)"));
	});

	it("handles empty settings", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings([]);
		overlay.show();

		const lines = overlay.render(80);
		const rendered = lines.join("\n");
		assert.ok(rendered.includes("No settings available"));
	});

	it("scrolls with page up/down", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		const manySettings: SettingDef[] = Array.from({ length: 20 }, (_, i) => ({
			name: `Setting ${i}`,
			currentValue: "val",
			description: "desc",
			options: [{ label: "opt", value: "opt", current: i === 0 }],
		}));
		overlay.setSettings(manySettings);
		overlay.show();

		// Page down
		overlay.handleInput("\x1b[6~");
		assert.ok(overlay.selectedIndex >= 8);

		// Page up
		overlay.handleInput("\x1b[5~");
		assert.ok(overlay.selectedIndex < 8);
	});

	it("cycles option with j/k keys", () => {
		setupTheme();
		const overlay = new SettingsSelectorOverlay();
		overlay.setSettings(makeSettings());
		overlay.show();

		// Open detail view for Thinking Level (index 1)
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\r");
		assert.strictEqual(overlay.selectedOptionIndex, 2); // medium is current at index 2

		// Move down with j
		overlay.handleInput("j");
		assert.strictEqual(overlay.selectedOptionIndex, 3); // high

		// Move up with k
		overlay.handleInput("k");
		assert.strictEqual(overlay.selectedOptionIndex, 2); // medium
	});
});
