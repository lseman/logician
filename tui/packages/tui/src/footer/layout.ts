// ── Configurable Footer — layout & render engine ───────────────────────────
// Takes a config (per-widget enable/align/row/position) + WidgetData[], groups
// widgets by row then alignment (left | middle | right), joins with │ separators,
// and produces the exact string[] output the TUI expects.

import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { sanitizeTerminalText } from "../rendering/terminal-sanitize.ts";
import { type Component, DIM, RESET, visibleWidth } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	type ContributedWidget,
	createDefaultConfig,
	DEFAULT_WIDGET_LAYOUTS,
	type FooterConfig,
	type WidgetLayout,
} from "./types.ts";
import {
	produceWidgets,
	type WidgetData,
	type WidgetFactoryStatus,
	type WidgetId,
} from "./widget-factory.ts";

/* ════════════════════════════════════════════════════════════════════════════
 *  Config loading — reads ~/.logician/footer-config.json if present
 * ════════════════════════════════════════════════════════════════════════════ */

function resolveConfigPath(): string {
	const home = process.env.HOME || "";
	return `${home}/.logician/footer-config.json`;
}

function configMtimeMs(): number {
	try {
		return statSync(resolve(resolveConfigPath())).mtimeMs;
	} catch {
		return 0;
	}
}

function loadFooterConfig(): FooterConfig {
	try {
		const fullPath = resolve(resolveConfigPath());
		if (existsSync(fullPath)) {
			const raw = readFileSync(fullPath, "utf-8");
			const parsed = JSON.parse(raw);
			return mergeWithDefaults(parsed);
		}
	} catch {
		// Config file missing or invalid — fall back to defaults
	}
	return createDefaultConfig();
}

/** Merge a raw config object with the default layout, validating against known
 * WidgetIds. Unknown keys are silently dropped. */
function mergeWithDefaults(raw: unknown): FooterConfig {
	const base = createDefaultConfig();
	if (!raw || typeof raw !== "object") return base;
	const obj = raw as Record<string, unknown>;

	if (
		typeof obj.rows === "number" &&
		Number.isInteger(obj.rows) &&
		obj.rows >= 1 &&
		obj.rows <= 5
	) {
		base.rows = obj.rows;
	}
	if (
		typeof obj.animationIntervalMs === "number" &&
		obj.animationIntervalMs > 0
	) {
		base.animationIntervalMs = obj.animationIntervalMs;
	}
	if (typeof obj.defaultTextColor === "string") {
		base.defaultTextColor = obj.defaultTextColor as any;
	}
	if (typeof obj.defaultIconColor === "string") {
		base.defaultIconColor = obj.defaultIconColor as any;
	}

	// Merge per-widget layouts
	if (obj.widgets && typeof obj.widgets === "object") {
		for (const [key, val] of Object.entries(
			obj.widgets as Record<string, unknown>,
		)) {
			if (!isValidWidgetId(key)) continue;
			if (val && typeof val === "object") {
				const w = val as Record<string, unknown>;
				const layout: Partial<WidgetLayout> = {};
				if (typeof w.enabled === "boolean") layout.enabled = w.enabled;
				if (
					typeof w.row === "number" &&
					Number.isInteger(w.row) &&
					w.row >= 0 &&
					w.row < base.rows
				)
					layout.row = w.row;
				if (typeof w.position === "number") layout.position = w.position;
				if (w.align === "left" || w.align === "middle" || w.align === "right") {
					layout.align = w.align as any;
				}
				if (w.fill === "none" || w.fill === "grow") {
					layout.fill = w.fill as any;
				}
				if (typeof w.minWidth === "number" && w.minWidth > 0) {
					layout.minWidth = w.minWidth;
				}
				base.widgets[key as WidgetId] = {
					...(DEFAULT_WIDGET_LAYOUTS[
						key as keyof typeof DEFAULT_WIDGET_LAYOUTS
					] ?? {
						enabled: true,
						row: 1,
						position: 0,
						align: "left",
						fill: "none",
					}),
					...layout,
				};
			}
		}
	}

	// Merge per-widget styles
	if (obj.widgetStyles && typeof obj.widgetStyles === "object") {
		for (const [key, val] of Object.entries(
			obj.widgetStyles as Record<string, unknown>,
		)) {
			if (!isValidWidgetId(key)) continue;
			if (val && typeof val === "object") {
				const s = val as Record<string, unknown>;
				const style: { iconColor?: string; textColor?: string } = {};
				if (s.iconColor && isValidThemeColor(s.iconColor)) {
					style.iconColor = s.iconColor as string;
				}
				if (s.textColor && isValidThemeColor(s.textColor)) {
					style.textColor = s.textColor as string;
				}
				base.widgetStyles[key as WidgetId] =
					style as FooterConfig["widgetStyles"][WidgetId];
			}
		}
	}

	return base;
}

function isValidWidgetId(id: string): boolean {
	return (
		id in DEFAULT_WIDGET_LAYOUTS || /^[a-z0-9][a-z0-9._:-]{0,127}$/i.test(id)
	);
}

function isValidThemeColor(c: unknown): boolean {
	const colors = [
		"text",
		"accent",
		"muted",
		"dim",
		"success",
		"error",
		"warning",
	];
	return typeof c === "string" && colors.includes(c);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Layout engine — groups widgets into rows + alignment buckets
 * ════════════════════════════════════════════════════════════════════════════ */

interface GroupedWidget {
	id: string;
	align: "left" | "middle" | "right";
	position: number;
	data: WidgetData;
	layout: WidgetLayout;
}

/** Sort widgets into rows and alignment groups, ordered by position within
 * each group. Returns an array of row-groups, each containing left/middle/right
 * sections. */
function layoutWidgets(
	widgets: WidgetData[],
	config: FooterConfig,
): GroupedWidget[][] {
	const byRow = new Map<number, Map<string, GroupedWidget[]>>();

	// Sort widgets by config ordering (row → align order → position)
	const sortedIds = [
		...Object.keys(DEFAULT_WIDGET_LAYOUTS),
		...Object.keys(config.widgets).filter(
			id => !(id in DEFAULT_WIDGET_LAYOUTS),
		),
	] as WidgetId[];

	for (const widgetId of sortedIds) {
		const layout =
			config.widgets[widgetId] ??
			DEFAULT_WIDGET_LAYOUTS[widgetId as keyof typeof DEFAULT_WIDGET_LAYOUTS];
		if (!layout) continue;
		if (!layout.enabled) continue;

		// Find the matching WidgetData
		const data = widgets.find(w => w.id === widgetId);
		if (!data || data.empty) continue;

		const alignKey = layout.align;
		const rowWidgets =
			byRow.get(layout.row) ?? new Map<string, GroupedWidget[]>();
		const group = rowWidgets.get(alignKey) ?? [];
		group.push({
			id: widgetId,
			align: alignKey,
			position: layout.position,
			data,
			layout,
		});

		rowWidgets.set(
			alignKey,
			group.sort((a, b) => a.position - b.position),
		);
		byRow.set(layout.row, rowWidgets);
	}

	// Convert to array of rows in order
	const result: GroupedWidget[][] = [];
	for (let rowNum = 0; rowNum < config.rows; rowNum++) {
		const row = byRow.get(rowNum);
		if (!row) {
			result.push([]);
			continue;
		}
		const sections: Array<{ align: string; widgets: GroupedWidget[] }> = [];
		for (const alignKey of ["left", "middle", "right"]) {
			const widgets = row.get(alignKey);
			if (widgets && widgets.length > 0) {
				sections.push({ align: alignKey, widgets });
			}
		}
		result.push(sections.flatMap(s => s.widgets));
	}

	return result;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Render — produce final output lines from grouped widgets
 * ════════════════════════════════════════════════════════════════════════════ */

const SEP = ` ${DIM}│${RESET} `;
const SEP_WIDTH = 3;
const ANSI_RE = /\x1b\[[0-?]*[ -/]*[@-~]/g;

function renderWidget(
	widget: GroupedWidget,
	config: FooterConfig,
	allocatedWidth?: number,
): string {
	const style = config.widgetStyles[widget.data.id];
	const iconColor = style?.iconColor ?? config.defaultIconColor ?? "dim";
	let text = widget.data.text;
	const textColor =
		style?.textColor ??
		(!text.includes("\x1b[") ? config.defaultTextColor : undefined);
	if (textColor) {
		text = theme.fg(textColor, text.replace(ANSI_RE, ""));
	}
	const icon = widget.data.icon
		? `${theme.fg(iconColor, widget.data.icon)} `
		: "";
	const rendered = `${icon}${text}`;
	const minimum = Math.max(widget.layout.minWidth ?? 0, allocatedWidth ?? 0);
	return rendered + " ".repeat(Math.max(0, minimum - visibleWidth(rendered)));
}

function joinWidgets(
	widgets: GroupedWidget[],
	config: FooterConfig,
	allocations = new Map<string, number>(),
): string {
	return widgets
		.map(widget => renderWidget(widget, config, allocations.get(widget.id)))
		.join(SEP);
}

function renderGroupedRow(
	group: GroupedWidget[],
	width: number,
	config: FooterConfig,
): string {
	const visibleWidgets = [...group];
	let parts = visibleWidgets.map(widget => renderWidget(widget, config));
	let line = parts.join(SEP);

	// Match the legacy footer's width behavior: optional sections disappear as
	// whole units, so ANSI sequences and labels are never clipped midway.
	const retentionPriority: Partial<Record<WidgetId, number>> = {
		phase: 0,
		model: 1,
		"context-bar": 2,
		location: 3,
		branch: 3,
		thinking: 4,
		reasoner: 5,
		"cache-read": 6,
		"token-flow": 7,
		goal: 8,
		"inference-mode": 9,
		mcp: 10,
		sandbox: 11,
		"execution-profile": 12,
		permission: 13,
		rtk: 14,
		memory: 15,
	};
	while (visibleWidth(line) > width && visibleWidgets.length > 3) {
		let dropIndex = -1;
		let lowestPriority = -1;
		for (let i = 0; i < visibleWidgets.length; i++) {
			const priority = retentionPriority[visibleWidgets[i].data.id] ?? 100;
			if (priority > lowestPriority) {
				lowestPriority = priority;
				dropIndex = i;
			}
		}
		if (dropIndex < 0 || lowestPriority <= 2) break;
		visibleWidgets.splice(dropIndex, 1);
		parts = visibleWidgets.map(widget => renderWidget(widget, config));
		line = parts.join(SEP);
	}
	if (visibleWidth(line) > width) {
		const phase =
			group.find(widget => widget.id === "phase")?.data.text ?? parts[0] ?? "";
		const context =
			group.find(widget => widget.id === "context-bar")?.data.text ?? "";
		const compact = [phase, context].filter(Boolean).join(SEP);
		line =
			visibleWidth(compact) <= width ? compact : truncateVisible(phase, width);
		return line + RESET;
	}

	const allocations = new Map<string, number>();
	const growWidgets = visibleWidgets.filter(
		widget => widget.layout.fill === "grow",
	);
	const unused = Math.max(0, width - visibleWidth(line));
	if (growWidgets.length > 0 && unused > 0) {
		const each = Math.floor(unused / growWidgets.length);
		let remainder = unused % growWidgets.length;
		for (const widget of growWidgets) {
			const natural = visibleWidth(renderWidget(widget, config));
			allocations.set(widget.id, natural + each + (remainder-- > 0 ? 1 : 0));
		}
	}

	const left = joinWidgets(
		visibleWidgets.filter(widget => widget.align === "left"),
		config,
		allocations,
	);
	const middle = joinWidgets(
		visibleWidgets.filter(widget => widget.align === "middle"),
		config,
		allocations,
	);
	const right = joinWidgets(
		visibleWidgets.filter(widget => widget.align === "right"),
		config,
		allocations,
	);
	const leftWidth = visibleWidth(left);
	const middleWidth = visibleWidth(middle);
	const rightWidth = visibleWidth(right);

	if (!left && !middle)
		return `${" ".repeat(Math.max(0, width - rightWidth))}${right}${RESET}`;
	if (!left && !right) {
		const before = Math.max(0, Math.floor((width - middleWidth) / 2));
		return `${" ".repeat(before)}${middle}${RESET}`;
	}
	if (!middle && !right) return `${left}${RESET}`;
	if (!middle) {
		return `${left}${" ".repeat(Math.max(SEP_WIDTH, width - leftWidth - rightWidth))}${right}${RESET}`;
	}
	if (!left) {
		const before = Math.max(
			0,
			Math.min(
				Math.floor((width - middleWidth) / 2),
				width - rightWidth - SEP_WIDTH - middleWidth,
			),
		);
		const after = Math.max(
			SEP_WIDTH,
			width - before - middleWidth - rightWidth,
		);
		return `${" ".repeat(before)}${middle}${" ".repeat(after)}${right}${RESET}`;
	}
	if (!right) {
		const before = Math.max(
			SEP_WIDTH,
			Math.floor((width - middleWidth) / 2) - leftWidth,
		);
		return `${left}${" ".repeat(before)}${middle}${RESET}`;
	}

	const middleStart = Math.max(
		leftWidth + SEP_WIDTH,
		Math.min(
			Math.floor((width - middleWidth) / 2),
			width - rightWidth - SEP_WIDTH - middleWidth,
		),
	);
	const beforeMiddle = middleStart - leftWidth;
	const afterMiddle = Math.max(
		SEP_WIDTH,
		width - middleStart - middleWidth - rightWidth,
	);
	return `${left}${" ".repeat(beforeMiddle)}${middle}${" ".repeat(afterMiddle)}${right}${RESET}`;
}

function truncateVisible(text: string, width: number): string {
	if (visibleWidth(text) <= width) return text;
	const ellipsis = "…";
	let out = "";
	let inEscape = false;
	let visible = 0;
	const target = Math.max(0, width - visibleWidth(ellipsis));
	for (let i = 0; i < text.length && visible < target; i++) {
		const ch = text[i];
		if (ch === "\x1b" && text[i + 1] === "[") {
			inEscape = true;
			out += ch;
			continue;
		}
		if (inEscape) {
			out += ch;
			if (ch === "m") inEscape = false;
			continue;
		}
		const chWidth = visibleWidth(ch);
		if (chWidth > 0) {
			out += ch;
			visible += chWidth;
		}
	}
	return out + ellipsis;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  StatusBar — public component, wraps the widget system
 * ════════════════════════════════════════════════════════════════════════════ */

const DEFAULT_INFO: WidgetFactoryStatus = {
	thinkingLevel: "off",
	inferenceMode: "instruct-general",
	cacheReadTokens: undefined,
	turnCount: 0,
	messageCount: 0,
	phase: "ready",
	model: "local",
	cwd: process.cwd(),
	virtualEnv: undefined,
	virtualEnvPythonVersion: undefined,
	branch: "",
	gitModified: 0,
	gitStaged: 0,
	gitUntracked: 0,
	gitCommit: "",
	gitAhead: 0,
	gitBehind: 0,
	gitAddedLines: 0,
	gitRemovedLines: 0,
	contextTokens: 0,
	contextMaxTokens: undefined,
	contextCompacted: false,
	reasoner: "none",
	sessionTitle: "",
	mcpServerCount: 0,
	mcpLoading: false,
	sandboxMode: "code",
	permissionMode: "acceptAll",
	executionProfile: "autonomous",
	promptTokens: undefined,
	completionTokens: undefined,
	rtkProxyEnabled: false,
	ariadneEnabled: true,
	fffgrepEnabled: true,
};

export class StatusBar implements Component {
	private info: WidgetFactoryStatus = { ...DEFAULT_INFO };
	private tick = 0;
	private _timer: ReturnType<typeof setInterval> | null = null;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onInvalidate: (() => void) | null = null;

	// Config — loaded from disk once, re-readable via reloadConfig()
	private config: FooterConfig;
	private configMtime = 0;
	private lastConfigCheck = 0;
	private readonly contributedWidgets = new Map<string, WidgetData>();
	private readonly contributedLayouts = new Map<
		string,
		Partial<WidgetLayout>
	>();
	private readonly contributedStyles = new Map<
		string,
		NonNullable<ContributedWidget["style"]>
	>();

	/** Widgets produced by the factory, refreshed each render cycle */
	private widgets: WidgetData[] = [];

	constructor() {
		this.config = loadFooterConfig();
		this.configMtime = configMtimeMs();
	}

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	update(info: Partial<WidgetFactoryStatus>): void {
		Object.assign(this.info, info);
		this._invalidate();
	}

	setTick(tick: number): void {
		this.tick = tick;
		this._invalidate();
	}

	invalidate(): void {
		this._invalidate();
	}

	reloadConfig(): FooterConfig {
		this.config = loadFooterConfig();
		this.applyContributedConfig();
		this.configMtime = configMtimeMs();
		this._invalidate();
		return this.config;
	}

	setConfig(config: FooterConfig): void {
		this.config = config;
		this.applyContributedConfig();
		this._invalidate();
	}

	getConfig(): FooterConfig {
		return structuredClone(this.config);
	}

	/** Add or replace a complete runtime widget snapshot. Extension IDs should
	 * be namespaced (for example `acme.build-status`) to avoid collisions. */
	upsertWidget(widget: ContributedWidget): void {
		if (!isValidWidgetId(widget.id) || widget.id in DEFAULT_WIDGET_LAYOUTS) {
			throw new Error(`Invalid or reserved footer widget id: ${widget.id}`);
		}
		const text = sanitizeTerminalText(widget.text)
			.replace(/\s*\n\s*/g, " ")
			.slice(0, 512);
		const icon = widget.icon
			? sanitizeTerminalText(widget.icon).replace(/\s+/g, " ").slice(0, 16)
			: undefined;
		this.contributedWidgets.set(widget.id, {
			id: widget.id,
			text,
			icon,
			label: widget.label,
			empty: widget.empty,
		});
		if (widget.layout) this.contributedLayouts.set(widget.id, widget.layout);
		if (widget.style) this.contributedStyles.set(widget.id, widget.style);
		if (!this.config.widgets[widget.id]) {
			this.config.widgets[widget.id] = {
				enabled: true,
				row: 1,
				position: this.contributedWidgets.size - 1,
				align: "left",
				fill: "none",
				...widget.layout,
			};
		} else if (widget.layout) {
			this.config.widgets[widget.id] = {
				...(this.config.widgets[widget.id] as WidgetLayout),
				...widget.layout,
			};
		}
		if (widget.style) this.config.widgetStyles[widget.id] = widget.style;
		this._invalidate();
	}

	removeWidget(id: string): boolean {
		const removed = this.contributedWidgets.delete(id);
		this.contributedLayouts.delete(id);
		this.contributedStyles.delete(id);
		if (removed) this._invalidate();
		return removed;
	}

	private applyContributedConfig(): void {
		let position = 0;
		for (const id of this.contributedWidgets.keys()) {
			const layout = this.contributedLayouts.get(id);
			this.config.widgets[id] = {
				enabled: true,
				row: 1,
				position: position++,
				align: "left",
				fill: "none",
				...this.config.widgets[id],
				...layout,
			};
			const style = this.contributedStyles.get(id);
			if (style) this.config.widgetStyles[id] = style;
		}
	}

	get timer(): ReturnType<typeof setInterval> | null {
		return this._timer;
	}

	private _invalidate(): void {
		this.cachedLines = null;
		this.onInvalidate?.();
	}

	// ── Animation timer ────────────────────────────────────────────────────

	startAnimation(): void {
		if (this._timer) return;
		const interval = this.config.animationIntervalMs ?? 150;
		this._timer = setInterval(() => {
			this.tick = (this.tick + 1) % 8;
			this.cachedLines = null;
			this.onInvalidate?.();
		}, interval);
	}

	stopAnimation(): void {
		if (this._timer) {
			clearInterval(this._timer);
			this._timer = null;
		}
		this.tick = 0;
		this._invalidate();
	}

	// ── Component.render ───────────────────────────────────────────────────

	render(width: number): string[] {
		const now = Date.now();
		if (now - this.lastConfigCheck >= 1000) {
			this.lastConfigCheck = now;
			const mtime = configMtimeMs();
			if (mtime !== this.configMtime) this.reloadConfig();
		}
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;

		// Produce fresh widget data from current status
		this.widgets = [
			...produceWidgets({ ...this.info, tick: this.tick }),
			...this.contributedWidgets.values(),
		];

		// Layout widgets into rows per config
		const rows = layoutWidgets(this.widgets, this.config);

		// Render each row (should be 1-2 lines max)
		const lines: string[] = [];
		for (const row of rows) {
			lines.push(renderGroupedRow(row, width, this.config));
		}

		if (lines.length === 0) {
			lines.push(" ".repeat(width));
		}

		this.cachedLines = lines;
		return lines;
	}
}
