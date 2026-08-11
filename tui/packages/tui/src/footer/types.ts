// ── Configurable Footer — types & defaults ───────────────────────────────────
// Widget-based footer inspired by pi-fancy-footer. Each data source is a
// configurable widget with position (row/position), alignment
// (left/middle/right), visibility toggle, fill mode (none/grow), and per-
// widget icon/text coloring.

import type { ThemeColor } from "../terminal/theme.ts";

/* ════════════════════════════════════════════════════════════════════════════
 *  Alignment & fill
 * ════════════════════════════════════════════════════════════════════════════ */

export type Alignment = "left" | "middle" | "right";
export type FillMode = "none" | "grow";

/* ════════════════════════════════════════════════════════════════════════════
 *  Widget layout config — per-widget overrides the user saves
 * ════════════════════════════════════════════════════════════════════════════ */

export interface WidgetLayout {
	enabled: boolean; // whether to show this widget
	row: number; // 0 = top row, 1 = bottom row
	position: number; // ordering within the alignment group (lower = left)
	align: Alignment; // left | middle | right
	fill: FillMode; // none = compact, grow = fill remaining space
	minWidth?: number; // minimum width in columns
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Per-widget style overrides (colors)
 * ════════════════════════════════════════════════════════════════════════════ */

export interface WidgetStyle {
	iconColor?: ThemeColor; // color for the widget's icon glyph
	textColor?: ThemeColor; // color for the widget's text content
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Known widget IDs — must match the widget factory registry
 * ════════════════════════════════════════════════════════════════════════════ */

export type BuiltinWidgetId =
	// Core status
	| "model" // active model name
	| "thinking" // thinking level (off/low/medium/high/xhigh)
	| "phase" // READY, THINKING, STREAMING, etc. with spinner
	// Context & tokens
	| "context-bar" // mini gauge of used context (can grow to full bar)
	| "context-capacity" // total context window size (e.g. 150k)
	| "token-flow" // combined token flow: ↑ in │ ↓ out (shows – for missing)
	// Cache stats
	| "cache-read" // cumulative cache-read tokens
	| "cache-write" // cumulative cache-write tokens
	| "cache-hit-rate" // latest turn's cache hit rate
	// Git / repo
	| "location" // current directory (abbreviated)
	| "virtual-env" // active Python virtual environment
	| "branch" // git branch name
	| "commit" // short commit SHA (opt-in)
	| "git-diff-added" // +N modified lines
	| "git-diff-removed" // -N removed lines
	| "git-status" // ahead/behind indicators (^, _, <>)
	// PRs (requires gh CLI)
	| "pull-request" // current PR number + status
	| "pull-request-review-threads" // unresolved review threads on PR
	| "pull-request-ci-status" // CI check status
	// Reasoner / inference
	| "reasoner" // active reasoner name
	| "inference-mode" // thinking-general, instruct-general, etc.
	| "sandbox" // sandbox mode (code / file / none / etc.)
	| "permission" // act / plan mode
	| "mcp" // MCP server count
	// Memory
	| "rtk" // RTK proxy status
	| "ariadne" // Ariadne code-graph tool status
	| "fffgrep" // fff indexed grep status
	| "memory" // memory subsystem status
	// Misc / config
	| "goal" // active goal condition with turns/time
	| "execution-profile" // autonomous / minimal execution profile
	// Cost (future)
	| "total-cost"; // cumulative session cost

/** Built-ins plus namespaced IDs contributed by extensions. */
export type WidgetId = BuiltinWidgetId | (string & {});

/* ════════════════════════════════════════════════════════════════════════════
 *  Widget data — what a provider returns each render cycle
 * ════════════════════════════════════════════════════════════════════════════ */

export interface WidgetData {
	id: WidgetId;
	label?: string; // optional label prefix (e.g. "ctx", "dir")
	text: string; // main visible text (may contain ANSI)
	icon?: string; // icon glyph to prepend
	empty?: boolean; // if true, widget renders as nothing
}

/** Complete snapshot published by an extension or other runtime producer. */
export interface ContributedWidget {
	id: string;
	text: string;
	label?: string;
	icon?: string;
	empty?: boolean;
	layout?: Partial<WidgetLayout>;
	style?: WidgetStyle;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Global footer config — user-editable settings
 * ════════════════════════════════════════════════════════════════════════════ */

export interface FooterConfig {
	// Layout: how many rows (1 or 2)
	rows: number;

	// Per-widget layout overrides (keys = widgetId)
	widgets: Partial<Record<WidgetId, WidgetLayout>>;

	// Per-widget style overrides (keys = widgetId)
	widgetStyles: Partial<Record<WidgetId, WidgetStyle>>;

	// Default colors when a widget doesn't specify overrides
	defaultTextColor?: ThemeColor;
	defaultIconColor?: ThemeColor;

	// Animation
	animationIntervalMs?: number; // phase spinner refresh rate (default 150)
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Default configuration — sensible out-of-the-box layout matching current
 *  StatusBar behavior: single row, sections separated by │.
 * ════════════════════════════════════════════════════════════════════════════ */

/** Build the default WidgetId → WidgetLayout map. Uses quoted keys for
 * kebab-case WidgetIds so TypeScript is happy. */
function buildDefaultLayouts(): Record<BuiltinWidgetId, WidgetLayout> {
	return {
		// Row 0 — left group (always shown when relevant)
		phase: { enabled: true, row: 0, position: 0, align: "left", fill: "none" },
		model: { enabled: true, row: 0, position: 1, align: "left", fill: "none" },
		"context-bar": {
			enabled: true,
			row: 0,
			position: 2,
			align: "left",
			fill: "none",
		},
		location: {
			enabled: true,
			row: 0,
			position: 3,
			align: "left",
			fill: "none",
		},
		"virtual-env": {
			enabled: true,
			row: 0,
			position: 4,
			align: "left",
			fill: "none",
		},
		branch: { enabled: true, row: 0, position: 5, align: "left", fill: "none" },
		"context-capacity": {
			enabled: false,
			row: 0,
			position: 6,
			align: "left",
			fill: "none",
		},

		// Row 0 — middle group (optional details)
		thinking: {
			enabled: true,
			row: 0,
			position: 0,
			align: "middle",
			fill: "none",
		},
		reasoner: {
			enabled: true,
			row: 0,
			position: 1,
			align: "middle",
			fill: "none",
		},
		"inference-mode": {
			enabled: true,
			row: 0,
			position: 2,
			align: "middle",
			fill: "none",
		},
		sandbox: {
			enabled: true,
			row: 0,
			position: 3,
			align: "middle",
			fill: "none",
		},
		permission: {
			enabled: true,
			row: 0,
			position: 4,
			align: "middle",
			fill: "none",
		},

		// Row 0 — right group (telemetry)
		"token-flow": {
			enabled: true,
			row: 0,
			position: 0,
			align: "right",
			fill: "none",
		},
		"cache-read": {
			enabled: true,
			row: 0,
			position: 2,
			align: "right",
			fill: "none",
		},
		"cache-write": {
			enabled: false,
			row: 0,
			position: 3,
			align: "right",
			fill: "none",
		},
		"cache-hit-rate": {
			enabled: false,
			row: 0,
			position: 4,
			align: "right",
			fill: "none",
		},
		mcp: { enabled: true, row: 0, position: 5, align: "right", fill: "none" },
		"total-cost": {
			enabled: false,
			row: 0,
			position: 6,
			align: "right",
			fill: "none",
		},

		// Row 1 — opt-in widgets (git / PRs)
		"git-diff-added": {
			enabled: false,
			row: 1,
			position: 0,
			align: "left",
			fill: "none",
		},
		"git-diff-removed": {
			enabled: false,
			row: 1,
			position: 1,
			align: "left",
			fill: "none",
		},
		"git-status": {
			enabled: false,
			row: 1,
			position: 2,
			align: "left",
			fill: "none",
		},
		commit: {
			enabled: false,
			row: 1,
			position: 3,
			align: "left",
			fill: "none",
		},
		"pull-request": {
			enabled: false,
			row: 1,
			position: 4,
			align: "left",
			fill: "none",
		},
		"pull-request-ci-status": {
			enabled: false,
			row: 1,
			position: 5,
			align: "left",
			fill: "none",
		},
		"pull-request-review-threads": {
			enabled: false,
			row: 1,
			position: 6,
			align: "right",
			fill: "none",
		},
		rtk: { enabled: true, row: 0, position: 7, align: "right", fill: "none" },
		ariadne: {
			enabled: true,
			row: 0,
			position: 8,
			align: "right",
			fill: "none",
		},
		fffgrep: {
			enabled: true,
			row: 0,
			position: 9,
			align: "right",
			fill: "none",
		},
		memory: {
			enabled: true,
			row: 0,
			position: 10,
			align: "right",
			fill: "none",
		},
		goal: { enabled: true, row: 0, position: 5, align: "middle", fill: "grow" },
		"execution-profile": {
			enabled: true,
			row: 0,
			position: 6,
			align: "middle",
			fill: "none",
		},
	};
}

export const DEFAULT_WIDGET_LAYOUTS: Record<BuiltinWidgetId, WidgetLayout> =
	buildDefaultLayouts();

/** The default config — single row, current StatusBar's section set */
export function createDefaultConfig(): FooterConfig {
	return {
		rows: 1,
		widgets: {}, // no overrides — use defaults above
		widgetStyles: {},
		defaultTextColor: "text",
		defaultIconColor: "dim",
		animationIntervalMs: 150,
	};
}
