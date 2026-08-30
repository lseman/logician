// ── Ink TUI Footer Widget Types ─────────────────────────────────────────────
// Configurable footer widget system: types, layout config, and data structures.

/* ════════════════════════════════════════════════════════════════════════════
 *  Alignment & fill mode
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

import type { ThemeColor } from "../theme.ts";

export interface WidgetStyle {
	iconColor?: ThemeColor; // color for the widget's icon glyph
	textColor?: ThemeColor; // color for the widget's text content
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Known widget IDs — must match the widget factory registry
 * ════════════════════════════════════════════════════════════════════════════ */

export type BuiltinWidgetId =
	// Core status
	| "model"
	| "thinking"
	| "phase"
	| "runtime-status"
	// Context & tokens
	| "context-bar"
	| "context-capacity"
	| "token-flow"
	// Cache stats
	| "cache-read"
	| "cache-write"
	| "cache-hit-rate"
	// Git / repo
	| "location"
	| "virtual-env"
	| "branch"
	| "commit"
	| "git-diff-added"
	| "git-diff-removed"
	| "git-status"
	// PRs (requires gh CLI)
	| "pull-request"
	| "pull-request-review-threads"
	| "pull-request-ci-status"
	// Reasoner / inference
	| "reasoner"
	| "inference-mode"
	| "sandbox"
	| "permission"
	| "mcp"
	// Memory / tool toggles
	| "rtk"
	| "legroom"
	| "memoriam"
	| "graphician"
	| "fffgrep"
	// Misc / config
	| "goal"
	| "execution-profile"
	// Cost (future)
	| "total-cost";

/** Built-ins plus namespaced IDs contributed by extensions. */
export type WidgetId = BuiltinWidgetId | (string & {});

/* ════════════════════════════════════════════════════════════════════════════
 *  Widget data — what a provider returns each render cycle
 * ════════════════════════════════════════════════════════════════════════════ */

export interface WidgetData {
	id: WidgetId;
	label?: string; // optional label prefix
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
	// Layout: how many rows (1-5)
	rows: number;

	// Per-widget layout overrides (keys = widgetId)
	widgets: Partial<Record<WidgetId, WidgetLayout>>;

	// Per-widget style overrides (keys = widgetId)
	widgetStyles: Partial<Record<WidgetId, WidgetStyle>>;

	// Default colors when a widget doesn't specify overrides
	defaultTextColor?: ThemeColor;
	defaultIconColor?: ThemeColor;

	// Animation: phase spinner refresh rate (default 150ms)
	animationIntervalMs?: number;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Default configuration — sensible out-of-the-box layout
 * ════════════════════════════════════════════════════════════════════════════ */

function buildDefaultLayouts(): Record<BuiltinWidgetId, WidgetLayout> {
	return {
		// Row 0 — left group (always shown when relevant)
		phase: { enabled: true, row: 0, position: 0, align: "left", fill: "none" },
		"runtime-status": {
			enabled: true,
			row: 0,
			position: 0,
			align: "right",
			fill: "none",
		},
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
			enabled: false,
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
			enabled: false,
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
			enabled: false,
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
			enabled: false,
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

		// Tool toggles (right group)
		rtk: { enabled: false, row: 0, position: 7, align: "right", fill: "none" },
		legroom: {
			enabled: true,
			row: 0,
			position: 8,
			align: "right",
			fill: "none",
		},
		memoriam: {
			enabled: true,
			row: 0,
			position: 9,
			align: "right",
			fill: "none",
		},
		graphician: {
			enabled: true,
			row: 0,
			position: 10,
			align: "right",
			fill: "none",
		},
		fffgrep: {
			enabled: true,
			row: 0,
			position: 11,
			align: "right",
			fill: "none",
		},

		// Goal (grow middle)
		goal: { enabled: false, row: 0, position: 5, align: "middle", fill: "grow" },

		// Execution profile
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

/** The default config — single row, minimal widgets */
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
