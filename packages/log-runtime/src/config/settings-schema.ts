/**
 * Settings schema registry — single source of truth for config keys,
 * defaults, enum values, numeric ranges, and settings-UI metadata.
 *
 * Mirrors the oh-my-pi settings-schema pattern: a flat record of config
 * paths (top-level keys plus dotted nested keys), each with a type and —
 * for keys exposed in the TUI settings overlay — `ui` metadata (tab,
 * group, label, description, presets). `config.ts` derives its known-key
 * sets and enum lists from this registry, and the TUI generates its
 * settings list from `getUiSettings()`.
 *
 * Entry order is meaningful for the UI: `getUiSettings()` returns entries
 * in declaration order, which is the settings-overlay display order (and
 * the order tabs first appear).
 */

import {
	INFERENCE_MODE_ORDER,
	INFERENCE_MODES,
	THINKING_FORMATS,
	THINKING_LEVELS,
	VALID_TOOL_EXECUTION,
} from "@logician/log-core";

export type SettingTab =
	| "Model"
	| "Behavior"
	| "Tools"
	| "Guards"
	| "Appearance";

export const SETTING_TABS: readonly SettingTab[] = [
	"Model",
	"Behavior",
	"Tools",
	"Guards",
	"Appearance",
];

export type SettingType =
	| "string"
	| "url"
	| "number"
	| "boolean"
	| "enum"
	| "array"
	| "object"
	| "models";

/** UI metadata for settings exposed in the TUI settings overlay. */
export interface SettingUi {
	tab: SettingTab;
	/** Section name within the tab. */
	group: string;
	/** Display label (the settings overlay keys action dispatch on this). */
	label: string;
	description: string;
	/** Display label overrides for enum values. */
	labels?: Readonly<Record<string, string>>;
	/** Preset values offered for number/text settings (free input still allowed). */
	presets?: readonly number[];
	/**
	 * The setting is read/written through the runtime view rather than a
	 * flat config key (e.g. "Guards" projects onto `guardsEnabled`,
	 * "Legroom SDK" onto `legroom.mode`). The TUI adapter owns the mapping.
	 */
	virtual?: boolean;
}

export interface SettingSpec {
	type: SettingType;
	/** Value `validateConfig` applies when the key is absent. */
	default?: string | number | boolean;
	/** Allowed values for `enum` settings. */
	enum?: readonly string[];
	/** Inclusive lower bound for `number` settings. */
	min?: number;
	/** Inclusive upper bound for `number` settings. */
	max?: number;
	/** When true, `min` is exclusive (setting must be > min). */
	minExclusive?: boolean;
	/**
	 * Range violations are dropped without a warning (legacy alias or
	 * "0 means unset" soft knobs).
	 */
	silent?: boolean;
	ui?: SettingUi;
}

const INFER_MODE_LABELS: Readonly<Record<string, string>> = Object.fromEntries(
	INFERENCE_MODE_ORDER.map(mode => [
		mode,
		INFERENCE_MODES.get(mode)?.label ?? mode,
	]),
);

/**
 * The full config schema. The first section is the UI surface in display
 * order; the remainder are config-only keys (order insignificant).
 */
export const SETTINGS_SCHEMA: Readonly<Record<string, SettingSpec>> = {
	// ── UI surface (settings-overlay display order) ──────────────────────
	model: {
		type: "string",
		ui: {
			tab: "Model",
			group: "Model",
			label: "Model",
			description: "LLM model to use",
			virtual: true,
		},
	},
	temperature: {
		type: "number",
		min: 0,
		max: 2,
		default: 0.5,
		ui: {
			tab: "Model",
			group: "Sampling",
			label: "Temperature",
			description: "Sampling temperature (0–2)",
			presets: [0, 0.3, 0.5, 0.7, 1.0],
		},
	},
	maxTokens: {
		type: "number",
		min: 0,
		minExclusive: true,
		default: 4096,
		ui: {
			tab: "Model",
			group: "Response limits",
			label: "Max tokens",
			description: "Maximum response tokens",
			presets: [1024, 2048, 4096, 8192, 16384],
		},
	},
	maxIterations: {
		type: "number",
		min: 0,
		minExclusive: true,
		ui: {
			tab: "Model",
			group: "Response limits",
			label: "Max iterations",
			description: "Maximum tool-use iterations per turn",
			presets: [10, 20, 30, 50, 100],
		},
	},
	thinkingLevel: {
		type: "enum",
		enum: THINKING_LEVELS,
		ui: {
			tab: "Model",
			group: "Reasoning",
			label: "Thinking level",
			description: "Depth of reasoning before responding",
			labels: Object.fromEntries(
				THINKING_LEVELS.map(level => [
					level,
					level.charAt(0).toUpperCase() + level.slice(1),
				]),
			),
		},
	},
	workflowMode: {
		type: "enum",
		enum: ["act", "plan"],
		ui: {
			tab: "Behavior",
			group: "Workflow",
			label: "Workflow mode",
			description: "Act with tools or produce a read-only plan",
			labels: { act: "Act", plan: "Plan" },
		},
	},
	guardsEnabled: {
		type: "boolean",
		ui: {
			tab: "Behavior",
			group: "Safety",
			label: "Guards",
			description: "Loop guards: auto uses safe defaults; off disables all",
			virtual: true,
		},
	},
	compaction: { type: "object" },
	"compaction.enabled": {
		type: "boolean",
		ui: {
			tab: "Behavior",
			group: "Context",
			label: "Compaction",
			description: "Auto-compact context to save tokens",
			virtual: true,
		},
	},
	inferenceMode: {
		type: "enum",
		enum: INFERENCE_MODE_ORDER,
		ui: {
			tab: "Model",
			group: "Sampling",
			label: "Inference mode",
			description: "Pre-defined sampling parameter set (Alt+M to cycle)",
			labels: INFER_MODE_LABELS,
		},
	},
	postEditDiagnostics: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Tools",
			group: "Editing",
			label: "Post-edit diagnostics",
			description: "Check edited files against the project",
		},
	},
	rtkProxyEnabled: {
		type: "boolean",
		ui: {
			tab: "Tools",
			group: "Command output",
			label: "RTK CLI proxy",
			description:
				"Prefix all bash commands with `rtk` for 60-90% output compression",
		},
	},
	legroom: { type: "object" },
	"legroom.mode": {
		type: "enum",
		enum: ["off", "sdk"],
		ui: {
			tab: "Tools",
			group: "Integrations",
			label: "Legroom SDK",
			description: "Compress outbound context through the local Legroom worker",
			virtual: true,
		},
	},
	memoriam: { type: "object" },
	"memoriam.mode": {
		type: "enum",
		enum: ["off", "sdk"],
		ui: {
			tab: "Tools",
			group: "Integrations",
			label: "Memoriam SDK",
			description:
				"Retrieve memory context for every turn (SQLite-backed store)",
			virtual: true,
		},
	},
	graphicianEnabled: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Tools",
			group: "Integrations",
			label: "Graphician",
			description:
				"Expose the Graphician code-graph tool for semantic repository analysis",
		},
	},
	fffgrepEnabled: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Tools",
			group: "Integrations",
			label: "fffgrep",
			description: "Prefer the fff indexed MCP grep tool over local grep",
		},
	},
	executionProfile: {
		type: "enum",
		enum: ["autonomous", "minimal"],
		ui: {
			tab: "Behavior",
			group: "Workflow",
			label: "Execution policy",
			description:
				"Auto continues bounded work; minimal performs one direct pass",
			labels: { autonomous: "Auto" },
		},
	},
	duplicateGuardEnabled: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Guards",
			group: "Loop protection",
			label: "Duplicate-call guard",
			description: "Block repeated identical tool calls",
		},
	},
	failureGuardEnabled: {
		type: "boolean",
		ui: {
			tab: "Guards",
			group: "Loop protection",
			label: "Failure-loop guard",
			description: "Block repeated equivalent tool failures",
		},
	},
	continuationEnabled: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Guards",
			group: "Loop protection",
			label: "Continuation",
			description: "Continue bounded unfinished autonomous work",
		},
	},
	autoRetryEnabled: {
		type: "boolean",
		default: true,
		ui: {
			tab: "Guards",
			group: "Loop protection",
			label: "Auto-compact on full context",
			description: "Compact and retry automatically when context fills up",
		},
	},
	progressStopEnabled: {
		type: "boolean",
		ui: {
			tab: "Guards",
			group: "Loop protection",
			label: "Budget early-stop",
			description: "Stop when useful token growth flattens",
		},
	},

	// ── Config-only keys ──────────────────────────────────────────────────
	baseUrl: { type: "url" },
	llmUrl: { type: "url" },
	models: { type: "models" },
	theme: { type: "string" },
	systemPrompt: { type: "string" },
	chatTemplate: { type: "string" },
	thinkingFormat: { type: "enum", enum: THINKING_FORMATS },
	toolExecution: { type: "enum", enum: VALID_TOOL_EXECUTION },
	contextWindow: { type: "number", min: 0, minExclusive: true, silent: true },
	contextWindowTokens: {
		type: "number",
		min: 0,
		minExclusive: true,
		silent: true,
	},
	hooks: { type: "boolean" },
	mcp: { type: "object" },
	mcpServers: { type: "object" },
	plugins: { type: "object" },
	permissionMode: {
		type: "enum",
		enum: ["acceptAll", "acceptEdits", "ask", "plan"],
	},
	permissions: { type: "object" },
	"permissions.allow": { type: "array" },
	"permissions.deny": { type: "array" },
	steeringInterrupt: { type: "boolean" },
	maxTotalTokens: { type: "number", min: 0, minExclusive: true },
	duplicateToolThreshold: { type: "number", min: 0 },
	toolFailureLoopThreshold: { type: "number", min: 0 },
	verifiedStopEnabled: { type: "boolean" },
	maxRetries: { type: "number", min: 0 },
	retryBaseDelayMs: { type: "number", min: 0 },
	turnTimeoutMs: { type: "number", min: 0, minExclusive: true },
	cacheSize: { type: "number", min: 0, minExclusive: true },
	cacheTtlMs: { type: "number", min: 0, minExclusive: true },
	allowedPaths: { type: "array" },
	allowAllPaths: { type: "boolean" },
	cwd: { type: "string" },
	lsp: { type: "object" },
	"lsp.enabled": { type: "boolean" },
	"lsp.timeoutMs": { type: "number", min: 0, minExclusive: true },
	"lsp.serverOverrides": { type: "object" },
	"compaction.mode": {
		type: "enum",
		enum: ["auto", "llm", "snapcompact", "shake", "remote"],
	},
	"compaction.reserveTokens": {
		type: "number",
		min: 0,
		minExclusive: true,
		silent: true,
	},
	"compaction.keepRecentTokens": {
		type: "number",
		min: 0,
		minExclusive: true,
		silent: true,
	},
	truncation: { type: "object" },
	"truncation.toolResultMaxChars": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.maxLines": { type: "number", min: 0, minExclusive: true },
	"truncation.grepLineMaxChars": { type: "number", min: 0, minExclusive: true },
	"truncation.subagentResultMaxChars": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.compactionSummaryMaxChars": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.microCompactMaxChars": { type: "object" },
	"truncation.microCompactMaxChars.tool": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.microCompactMaxChars.assistant": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.microCompactMaxChars.default": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	"truncation.transcriptMessageMaxChars": {
		type: "number",
		min: 0,
		minExclusive: true,
	},
	maxParallelAgents: { type: "number", min: 0, minExclusive: true },
	transcriptMaxTurns: { type: "number" },
	transcriptMaxRenderedLines: { type: "number" },
	reasoner: { type: "string" },
	reasonerConfig: { type: "object" },
	simpleTools: { type: "array" },
	tools: { type: "object" },
	"tools.xdev": { type: "boolean" },
	todoEnabled: { type: "boolean", default: false },
	"legroom.python": { type: "string" },
	"legroom.args": { type: "array" },
	"legroom.failOpen": { type: "boolean" },
	"legroom.timeoutMs": { type: "number", min: 0, minExclusive: true },
	"legroom.config": { type: "object" },
	"memoriam.python": { type: "string" },
	"memoriam.args": { type: "array" },
	"memoriam.failOpen": { type: "boolean" },
	"memoriam.timeoutMs": { type: "number", min: 0, minExclusive: true },
	"memoriam.config": { type: "object" },
	ttsr: { type: "object" },
	"ttsr.enabled": { type: "boolean" },
	"ttsr.builtinRules": { type: "boolean" },
	"ttsr.judge": { type: "boolean" },
	"ttsr.interruptMode": {
		type: "enum",
		enum: ["always", "prose-only", "tool-only", "never"],
	},
	"ttsr.repeatMode": { type: "enum", enum: ["once", "gap"] },
	"ttsr.repeatGap": { type: "number", min: 1 },
	"ttsr.disabledRules": { type: "array" },
	webSearch: { type: "object" },
	"webSearch.baseUrl": { type: "url" },
	"webSearch.maxResults": { type: "number", min: 1, max: 100 },
};

/** Top-level config keys, in declaration order. */
export function getKnownConfigKeys(): readonly string[] {
	return Object.keys(SETTINGS_SCHEMA).filter(key => !key.includes("."));
}

/**
 * Direct sub-keys under a top-level key (e.g. `getNestedKeys("truncation")`),
 * in declaration order. Deeper paths are excluded — query their own prefix
 * (e.g. `getNestedKeys("truncation.microCompactMaxChars")`).
 */
export function getNestedKeys(prefix: string): readonly string[] {
	const needle = `${prefix}.`;
	return Object.keys(SETTINGS_SCHEMA)
		.filter(
			key => key.startsWith(needle) && !key.slice(needle.length).includes("."),
		)
		.map(key => key.slice(needle.length));
}

export function getSettingSpec(key: string): SettingSpec | undefined {
	return SETTINGS_SCHEMA[key];
}

export function getDefault(key: string): string | number | boolean | undefined {
	return SETTINGS_SCHEMA[key]?.default;
}

export function getEnumValues(key: string): readonly string[] | undefined {
	return SETTINGS_SCHEMA[key]?.enum;
}

/** Settings exposed in the TUI settings overlay, in display order. */
export function getUiSettings(): readonly {
	key: string;
	spec: SettingSpec;
}[] {
	return Object.entries(SETTINGS_SCHEMA)
		.filter(([, spec]) => spec.ui !== undefined)
		.map(([key, spec]) => ({ key, spec }));
}

/** UI settings for one tab, in display order. */
export function getUiSettingsForTab(tab: SettingTab): readonly {
	key: string;
	spec: SettingSpec;
}[] {
	return getUiSettings().filter(({ spec }) => spec.ui?.tab === tab);
}
