import { describe, expect, test } from "bun:test";
import { validateConfig } from "../../config/config.ts";
import {
	getDefault,
	getEnumValues,
	getKnownConfigKeys,
	getNestedKeys,
	getSettingSpec,
	getUiSettings,
	getUiSettingsForTab,
	SETTING_TABS,
	type SettingTab,
} from "../../config/settings-schema.ts";

// ── Registry integrity ────────────────────────────────────────────────────

describe("settings schema registry", () => {
	test("every top-level key is a known config key", () => {
		const known = new Set(getKnownConfigKeys());
		for (const key of [
			"baseUrl",
			"model",
			"models",
			"temperature",
			"maxTokens",
			"thinkingLevel",
			"compaction",
			"truncation",
			"lsp",
			"mcp",
			"mcpServers",
			"plugins",
			"legroom",
			"memoriam",
			"webSearch",
			"permissions",
			"tools",
			"simpleTools",
			"allowedPaths",
			"todoEnabled",
		]) {
			expect(known.has(key)).toBe(true);
		}
	});

	test("nested key sets match the validation surface", () => {
		expect(getNestedKeys("tools")).toEqual(["xdev"]);
		expect(getNestedKeys("compaction")).toEqual([
			"enabled",
			"mode",
			"reserveTokens",
			"keepRecentTokens",
		]);
		expect(getNestedKeys("truncation")).toEqual([
			"toolResultMaxChars",
			"maxLines",
			"grepLineMaxChars",
			"subagentResultMaxChars",
			"compactionSummaryMaxChars",
			"microCompactMaxChars",
			"transcriptMessageMaxChars",
		]);
		expect(getNestedKeys("truncation.microCompactMaxChars")).toEqual([
			"tool",
			"assistant",
			"default",
		]);
		expect(getNestedKeys("webSearch")).toEqual(["baseUrl", "maxResults"]);
		expect(getNestedKeys("permissions")).toEqual(["allow", "deny"]);
		expect(getNestedKeys("legroom")).toEqual([
			"mode",
			"python",
			"args",
			"failOpen",
			"timeoutMs",
			"config",
		]);
		expect(getNestedKeys("memoriam")).toEqual([
			"mode",
			"python",
			"args",
			"failOpen",
			"timeoutMs",
			"config",
		]);
	});

	test("enum values match the harness contracts", () => {
		expect(getEnumValues("thinkingLevel")).toEqual([
			"off",
			"minimal",
			"low",
			"medium",
			"high",
			"xhigh",
		]);
		expect(getEnumValues("thinkingFormat")).toEqual([
			"qwen",
			"qwen-chat-template",
		]);
		expect(getEnumValues("executionProfile")).toEqual([
			"autonomous",
			"minimal",
		]);
		expect(getEnumValues("toolExecution")).toEqual(["sequential", "parallel"]);
		expect(getEnumValues("permissionMode")).toEqual([
			"acceptAll",
			"acceptEdits",
			"ask",
			"plan",
		]);
		expect(getEnumValues("workflowMode")).toEqual(["act", "plan"]);
		expect(getEnumValues("inferenceMode")).toHaveLength(10);
		expect(getEnumValues("legroom.mode")).toEqual(["off", "sdk"]);
		expect(getEnumValues("memoriam.mode")).toEqual(["off", "sdk"]);
	});

	test("boolean defaults match the validation defaults", () => {
		expect(getDefault("duplicateGuardEnabled")).toBe(true);
		expect(getDefault("continuationEnabled")).toBe(true);
		expect(getDefault("postEditDiagnostics")).toBe(true);
		expect(getDefault("autoRetryEnabled")).toBe(true);
		expect(getDefault("graphicianEnabled")).toBe(true);
		expect(getDefault("fffgrepEnabled")).toBe(true);
		expect(getDefault("todoEnabled")).toBe(false);
		// No config-level default — the runtime resolves these.
		expect(getDefault("failureGuardEnabled")).toBeUndefined();
		expect(getDefault("progressStopEnabled")).toBeUndefined();
		expect(getDefault("guardsEnabled")).toBeUndefined();
	});

	test("numeric ranges match the validation rules", () => {
		expect(getSettingSpec("temperature")?.min).toBe(0);
		expect(getSettingSpec("temperature")?.max).toBe(2);
		expect(getSettingSpec("maxTokens")?.minExclusive).toBe(true);
		expect(getSettingSpec("turnTimeoutMs")?.minExclusive).toBe(true);
		expect(getSettingSpec("maxRetries")?.minExclusive).toBeUndefined();
		expect(getSettingSpec("webSearch.maxResults")?.min).toBe(1);
		expect(getSettingSpec("webSearch.maxResults")?.max).toBe(100);
	});

	// ── UI metadata ─────────────────────────────────────────────────────────

	test("ui settings are ordered and carry complete metadata", () => {
		const ui = getUiSettings();
		expect(ui.length).toBeGreaterThan(0);
		for (const { spec } of ui) {
			const meta = spec.ui;
			expect(meta).toBeDefined();
			expect(meta?.tab).toBeDefined();
			expect(meta?.group).toBeDefined();
			expect(meta?.label.length).toBeGreaterThan(0);
			expect(meta?.description.length).toBeGreaterThan(0);
		}
		const labels = ui.map(({ spec }) => spec.ui?.label);
		expect(new Set(labels).size).toBe(labels.length);
	});

	test("ui settings group into the known tabs in display order", () => {
		const tabs: SettingTab[] = [];
		for (const { spec } of getUiSettings()) {
			const tab = spec.ui?.tab;
			if (tab && !tabs.includes(tab)) tabs.push(tab);
		}
		expect(tabs).toEqual(["Model", "Behavior", "Tools", "Guards"]);
		for (const tab of tabs) expect(SETTING_TABS).toContain(tab);
	});

	test("number settings expose presets for the overlay", () => {
		for (const name of ["Temperature", "Max tokens", "Max iterations"]) {
			const entry = getUiSettings().find(({ spec }) => spec.ui?.label === name);
			expect(entry).toBeDefined();
			expect(entry?.spec.ui?.presets?.length).toBeGreaterThan(0);
		}
	});

	test("virtual settings are marked for the TUI adapter", () => {
		const virtual = getUiSettings()
			.filter(({ spec }) => spec.ui?.virtual)
			.map(({ spec }) => spec.ui?.label);
		expect(virtual).toContain("Model");
		expect(virtual).toContain("Guards");
		expect(virtual).toContain("Compaction");
	});

	test("per-tab queries match the flat list", () => {
		for (const tab of SETTING_TABS) {
			const flat = getUiSettings().filter(({ spec }) => spec.ui?.tab === tab);
			expect(getUiSettingsForTab(tab)).toEqual(flat);
		}
	});
});

// ── validateConfig behavior against the registry ──────────────────────────

describe("validateConfig with registry-derived keys", () => {
	test("simpleTools is a known key (no unknown-key warning)", () => {
		const warnings: string[] = [];
		const cfg = validateConfig({ simpleTools: ["bash"] }, warnings);
		expect(warnings.filter(w => w.includes("Unknown config key"))).toEqual([]);
		expect(cfg.simpleTools).toEqual(["bash"]);
	});

	test("tools.xdev is a known key (no unknown-key warning)", () => {
		const warnings: string[] = [];
		const cfg = validateConfig({ tools: { xdev: false } }, warnings);
		expect(warnings.filter(w => w.includes("Unknown config key"))).toEqual([]);
		expect(cfg.tools).toEqual({ xdev: false });
	});

	test("unknown top-level keys still warn", () => {
		const warnings: string[] = [];
		validateConfig({ notARealKey: true }, warnings);
		expect(warnings.some(w => w.includes('"notARealKey"'))).toBe(true);
	});

	test("unknown nested keys still warn", () => {
		const warnings: string[] = [];
		validateConfig({ compaction: { bogus: 1 } }, warnings);
		expect(
			warnings.some(w => w.includes('Unknown compaction key: "bogus"')),
		).toBe(true);
	});

	test("enum values sourced from the registry are enforced", () => {
		const warnings: string[] = [];
		const cfg = validateConfig({ permissionMode: "yolo" }, warnings) as {
			permissionMode?: string;
		};
		expect(cfg.permissionMode).toBeUndefined();
		expect(warnings.some(w => w.includes('"permissionMode"'))).toBe(true);
	});

	test("boolean defaults are applied when keys are absent", () => {
		const warnings: string[] = [];
		const cfg = validateConfig({}, warnings);
		expect(cfg.duplicateGuardEnabled).toBe(true);
		expect(cfg.continuationEnabled).toBe(true);
		expect(cfg.postEditDiagnostics).toBe(true);
		expect(cfg.autoRetryEnabled).toBe(true);
		expect(cfg.graphicianEnabled).toBe(true);
		expect(cfg.fffgrepEnabled).toBe(true);
		expect(cfg.todoEnabled).toBe(false);
		expect(cfg.failureGuardEnabled).toBeUndefined();
	});
});
