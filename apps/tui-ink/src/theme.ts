// ── Ink TUI Theme System ──────────────────────────────────────────────────────
// Semantic color model matching the old tui. Supports hex (#rrggbb), 256-color
// (0-255), variable references, and default ("").

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// ── Color token names (mirrors old tui) ──────────────────────────────────────

export type ThemeColor =
	// Core UI
	| "accent"
	| "success"
	| "error"
	| "warning"
	| "info"
	| "muted"
	| "dim"
	| "text"
	| "border"
	| "borderMuted"
	// Speaker colors
	| "userText"
	| "assistantText"
	| "systemText"
	// Transcript section labels
	| "userLabel"
	| "responseLabel"
	| "reasoningLabel"
	// Markdown
	| "mdHeading"
	| "mdCode"
	| "mdCodeBlock"
	| "mdCodeBlockBg"
	| "mdCodeBlockBorder"
	| "mdLink"
	| "mdQuote"
	| "mdListBullet"
	// Tool display
	| "toolTitle"
	| "toolRunning"
	| "toolSuccess"
	| "toolError"
	| "toolStreaming"
	| "toolOutput"
	// Plugin startup messages
	| "pluginStartup"
	// Diff
	| "diffAdded"
	| "diffRemoved"
	| "diffContext"
	| "diffHunk"
	| "diffMeta"
	// Terminal output
	| "terminalOutput"
	// Thinking
	| "thinkingText"
	| "thinkingLabel"
	// Status phases
	| "phaseReady"
	| "phaseThinking"
	| "phaseTool"
	| "phaseError"
	| "phaseStreaming"
	| "phaseCompacting"
	| "phaseBranching"
	// Status indicators
	| "contextGood"
	| "contextWarning"
	| "contextCritical"
	// Thinking levels
	| "levelOff"
	| "levelLow"
	| "levelMedium"
	| "levelHigh"
	| "levelXhigh"
	// UI elements
	| "separator"
	| "prompt"
	| "inputText"
	| "inputPlaceholder"
	| "selected"
	| "header"
	| "active"
	| "jsonKey"
	| "jsonString"
	| "jsonNumber"
	| "jsonKeyword"
	| "jsonPunctuation"
	// Memory / observations
	| "memoryTag"
	| "memoryId"
	| "memoryContent"
	| "memoryCount";

// ── Schema types ──────────────────────────────────────────────────────────────

interface ThemeJson {
	name: string;
	vars?: Record<string, string | number>;
	colors: Record<string, string | number>;
}

// ── Color resolution helpers ──────────────────────────────────────────────────

function hexToRgb(hex: string): { r: number; g: number; b: number } {
	const cleaned = hex.replace("#", "");
	if (cleaned.length !== 6) throw new Error(`Invalid hex color: ${hex}`);
	return {
		r: parseInt(cleaned.substring(0, 2), 16),
		g: parseInt(cleaned.substring(2, 4), 16),
		b: parseInt(cleaned.substring(4, 6), 16),
	};
}

const CUBE_VALUES = [0, 95, 135, 175, 215, 255];
const GRAY_VALUES = Array.from({ length: 24 }, (_, i) => 8 + i * 10);

// Pre-computed 256-color → hex lookup table for truecolor mode
const _256_TO_HEX: Record<number, string> = (() => {
	const table: Record<number, string> = {};
	// Colors 0-7: standard
	const stdColors: Record<number, string> = {
		0: "#000000", 1: "#800000", 2: "#008000", 3: "#808000", 4: "#000080",
		5: "#800080", 6: "#008080", 7: "#c0c0c0",
	};
	// Colors 8-15: high intensity
	const hiColors: Record<number, string> = {
		8: "#808080", 9: "#ff0000", 10: "#00ff00", 11: "#ffff00", 12: "#0000ff",
		13: "#ff00ff", 14: "#00ffff", 15: "#ffffff",
	};
	Object.assign(table, stdColors, hiColors);
	// Colors 16-231: 6×6×6 color cube
	for (let r = 0; r < 6; r++) {
		for (let g = 0; g < 6; g++) {
			for (let b = 0; b < 6; b++) {
				const idx = 16 + 36 * r + 6 * g + b;
				const rv = CUBE_VALUES[r];
				const gv = CUBE_VALUES[g];
				const bv = CUBE_VALUES[b];
				table[idx] = `#${rv.toString(16).padStart(2, "0")}${gv.toString(16).padStart(2, "0")}${bv.toString(16).padStart(2, "0")}`;
			}
		}
	}
	// Colors 232-255: grayscale ramp
	for (let i = 0; i < 24; i++) {
		const idx = 232 + i;
		const v = GRAY_VALUES[i];
		table[idx] = `#${v.toString(16).padStart(2, "0")}${v.toString(16).padStart(2, "0")}${v.toString(16).padStart(2, "0")}`;
	}
	return table;
})();

function findClosest(arr: number[], value: number): number {
	let minDist = Infinity;
	let minIdx = 0;
	for (let i = 0; i < arr.length; i++) {
		const dist = Math.abs(value - arr[i]);
		if (dist < minDist) {
			minDist = dist;
			minIdx = i;
		}
	}
	return minIdx;
}

function rgbTo256(r: number, g: number, b: number): number {
	const rI = findClosest(CUBE_VALUES, r);
	const gI = findClosest(CUBE_VALUES, g);
	const bI = findClosest(CUBE_VALUES, b);
	const cubeIdx = 16 + 36 * rI + 6 * gI + bI;
	const cubeDist =
		(r - CUBE_VALUES[rI]) ** 2 * 0.299 +
		(g - CUBE_VALUES[gI]) ** 2 * 0.587 +
		(b - CUBE_VALUES[bI]) ** 2 * 0.114;

	const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
	const gI2 = findClosest(GRAY_VALUES, gray);
	const grayIdx = 232 + gI2;
	const grayDist =
		(r - GRAY_VALUES[gI2]) ** 2 * 0.299 +
		(g - GRAY_VALUES[gI2]) ** 2 * 0.587 +
		(b - GRAY_VALUES[gI2]) ** 2 * 0.114;

	const spread = Math.max(r, g, b) - Math.min(r, g, b);
	if (spread < 10 && grayDist < cubeDist) return grayIdx;
	return cubeIdx;
}

function resolveVarRefs(
	value: string | number,
	vars: Record<string, string | number> | undefined,
	visited = new Set<string>(),
): string | number {
	if (typeof value === "number" || value === "" || value.startsWith("#")) {
		return value;
	}
	if (visited.has(value)) {
		throw new Error(`Circular variable reference: ${value}`);
	}
	if (!vars || !(value in vars)) {
		throw new Error(`Variable not found: ${value}`);
	}
	visited.add(value);
	const resolvedValue = vars[value] ?? (() => { throw new Error(`Variable not found: ${value}`); })();
	return resolveVarRefs(resolvedValue, vars, visited);
}

function valueToInkColor(
	value: string | number,
	mode: "truecolor" | "256color",
): string | undefined {
	if (value === "") return undefined;
	if (typeof value === "number") {
		// 256-color index → hex for truecolor, ansiN for 256color mode
		if (mode === "truecolor") {
			return _256_TO_HEX[value] ?? `#000000`;
		}
		return `ansi${value}`;
	}
	// hex string
	if (mode === "truecolor") {
		return value; // ink accepts hex directly
	}
	// 256color mode: convert hex → nearest 256-index → ansiN
	const idx = rgbTo256(...(Object.values(hexToRgb(value)) as [number, number, number]));
	return `ansi${idx}`;
}

// ── Theme Class ───────────────────────────────────────────────────────────────

export class Theme {
	readonly name: string;
	readonly mode: "truecolor" | "256color";

	/** Resolved foreground color for each semantic key. */
	private readonly _fgColors: Map<ThemeColor, string>;
	/** Resolved background color for each semantic key. */
	private readonly _bgColors: Map<"mdCodeBlockBg", string>;

	constructor(
		name: string,
		mode: "truecolor" | "256color",
		fgColors: Map<ThemeColor, string>,
		bgColors: Map<"mdCodeBlockBg", string>,
	) {
		this.name = name;
		this.mode = mode;
		this._fgColors = fgColors;
		this._bgColors = bgColors;
	}

	// ── Public color methods ──────────────────────────────────────────────────

	get fgColors(): Map<ThemeColor, string> {
		return this._fgColors;
	}

	get bgColors(): Map<"mdCodeBlockBg", string> {
		return this._bgColors;
	}

	fg(color: ThemeColor): string | undefined {
		return this._fgColors.get(color);
	}

	hasColor(color: string): color is ThemeColor {
		return this._fgColors.has(color as ThemeColor);
	}

	bg(color: "mdCodeBlockBg"): string | undefined {
		return this._bgColors.get(color);
	}

	/** Derive a background color from a foreground color. */
	fgAsBg(color: ThemeColor): string | undefined {
		const fg = this._fgColors.get(color);
		if (!fg) return undefined;
		// If it's a hex color, convert RRGGBB → bg hex
		if (fg.startsWith("#")) {
			return fg; // ink handles bg separately; caller manages
		}
		return fg;
	}

	// ── Convenience helpers ───────────────────────────────────────────────────

	/** Get the Ink-compatible color string for a semantic key. */
	colorFor(color: ThemeColor): string | undefined {
		return this._fgColors.get(color);
	}

	phaseColor(phase: string): string | undefined {
		const map: Record<string, ThemeColor> = {
			ready: "phaseReady",
			thinking: "phaseThinking",
			tool: "phaseTool",
			error: "phaseError",
			streaming: "phaseStreaming",
			compacting: "phaseCompacting",
			branching: "phaseBranching",
		};
		const color = map[phase.toLowerCase()] ?? "muted";
		return this._fgColors.get(color);
	}

	thinkingBorderColor(level: "off" | "low" | "medium" | "high" | "xhigh"): string | undefined {
		const map = {
			off: "levelOff",
			low: "levelLow",
			medium: "levelMedium",
			high: "levelHigh",
			xhigh: "levelXhigh",
		} as const;
		return this._fgColors.get(map[level]);
	}
}

// ── Global Theme Instance ─────────────────────────────────────────────────────

const THEME_KEY = Symbol.for("logician:theme");

function getGlobalTheme(): Theme | undefined {
	return (globalThis as Record<symbol, Theme | undefined>)[THEME_KEY];
}

function setGlobalTheme(t: Theme): void {
	(globalThis as Record<symbol, Theme | undefined>)[THEME_KEY] = t;
}

// Lazy proxy so code can import `theme` without initialization concerns
export const theme: Theme = new Proxy({} as Theme, {
	get(_target, prop) {
		const t = getGlobalTheme();
		if (!t)
			throw new Error(
				"Logician theme not initialized. Call initTheme() first.",
			);
		return (t as unknown as Record<string | symbol, unknown>)[prop];
	},
});

// ── Theme Discovery ───────────────────────────────────────────────────────────

const BUNDLED_THEMES_DIR = join(
	dirname(fileURLToPath(import.meta.url)),
	"themes",
);

function loadThemeJson(_name: string, path: string): ThemeJson {
	const content = readFileSync(path, "utf-8");
	let json: unknown;
	try {
		json = JSON.parse(content);
	} catch (_e: unknown) {
		throw new Error(`Failed to parse theme file: ${path}: ${_e}`);
	}
	if (
		!json ||
		typeof json !== "object" ||
		!("name" in json) ||
		!("colors" in json)
	) {
		throw new Error(`Invalid theme format in: ${path}`);
	}
	return json as ThemeJson;
}

function detectColorMode(): "truecolor" | "256color" {
	const colorterm = process.env.COLORTERM || "";
	if (colorterm.includes("truecolor") || colorterm.includes("24bit")) {
		return "truecolor";
	}
	return "256color";
}

function buildThemeFromJson(
	json: ThemeJson,
	mode: "truecolor" | "256color",
): Theme {
	const vars = json.vars || {};
	const fgColors = new Map<ThemeColor, string>();
	const bgColors = new Map<"mdCodeBlockBg", string>();

	const fgKeys: ThemeColor[] = [
		"accent",
		"success",
		"error",
		"warning",
		"muted",
		"dim",
		"text",
		"border",
		"borderMuted",
		"userText",
		"assistantText",
		"systemText",
		"userLabel",
		"responseLabel",
		"reasoningLabel",
		"mdHeading",
		"mdCode",
		"mdCodeBlock",
		"mdCodeBlockBg",
		"mdCodeBlockBorder",
		"mdLink",
		"mdQuote",
		"mdListBullet",
		"toolTitle",
		"toolRunning",
		"toolSuccess",
		"toolError",
		"toolStreaming",
		"toolOutput",
		"pluginStartup",
		"diffAdded",
		"diffRemoved",
		"diffContext",
		"diffHunk",
		"diffMeta",
		"terminalOutput",
		"thinkingText",
		"thinkingLabel",
		"phaseReady",
		"phaseThinking",
		"phaseTool",
		"phaseError",
		"phaseStreaming",
		"phaseCompacting",
		"phaseBranching",
		"contextGood",
		"contextWarning",
		"contextCritical",
		"levelOff",
		"levelLow",
		"levelMedium",
		"levelHigh",
		"levelXhigh",
		"separator",
		"prompt",
		"inputText",
		"inputPlaceholder",
		"selected",
		"header",
		"active",
		"jsonKey",
		"jsonString",
		"jsonNumber",
		"jsonKeyword",
		"jsonPunctuation",
		"memoryTag",
		"memoryId",
		"memoryContent",
		"memoryCount",
	];

	const labelFallbacks: Partial<Record<ThemeColor, ThemeColor>> = {
		userLabel: "accent",
		responseLabel: "assistantText",
		reasoningLabel: "thinkingText",
	};

	for (const key of fgKeys) {
		const colors = json.colors as Record<string, unknown>;
		const raw = colors[key] ?? colors[labelFallbacks[key] ?? ""];
		if (raw !== undefined && raw !== null) {
			const resolved = resolveVarRefs(raw as string | number, vars);
			const isBg = key === "mdCodeBlockBg";
			const inkColor = valueToInkColor(resolved, mode);
			if (inkColor) {
				if (isBg) {
					bgColors.set(key, inkColor);
				} else {
					fgColors.set(key, inkColor);
				}
			}
		}
	}

	return new Theme(json.name, mode, fgColors, bgColors);
}

export function getAvailableThemes(): string[] {
	const themes: string[] = [];
	const seen = new Set<string>();

	try {
		if (existsSync(BUNDLED_THEMES_DIR)) {
			for (const file of readdirSync(BUNDLED_THEMES_DIR)) {
				if (file.endsWith(".json")) {
					const name = file.replace(".json", "");
					if (!seen.has(name)) {
						seen.add(name);
						themes.push(name);
					}
				}
			}
		}
	} catch (_e: unknown) {
		// themes dir doesn't exist
	}

	return themes.sort();
}

function loadTheme(name: string): Theme {
	const path = join(BUNDLED_THEMES_DIR, `${name}.json`);
	if (!existsSync(path)) {
		throw new Error(
			`Theme not found: ${name} (looked in ${BUNDLED_THEMES_DIR})`,
		);
	}
	const json = loadThemeJson(name, path);
	const mode = detectColorMode();
	return buildThemeFromJson(json, mode);
}

export function initTheme(name?: string): void {
	const themeName = name || process.env.LOGICIAN_THEME || "dark";
	try {
		const t = loadTheme(themeName);
		setGlobalTheme(t);
	} catch (_e: unknown) {
		console.error("[theme] initTheme failed:", _e);
		setGlobalTheme(loadTheme("dark"));
	}
}

export function getCurrentThemeName(): string {
	const t = getGlobalTheme();
	return t?.name ?? "unknown";
}

export function setCurrentTheme(name: string): boolean {
	try {
		const t = loadTheme(name);
		setGlobalTheme(t);
		return true;
	} catch (_e: unknown) {
		return false;
	}
}

// ── Legacy API (for backward compat) ─────────────────────────────────────────

/**
 * Get the current theme as a plain object with fg/bg/modifiers records.
 * This provides backward compatibility with the old theme API.
 */
export function getCurrentTheme(): {
	name: string;
	fg: Record<string, string | undefined>;
	bg: Record<string, string | undefined>;
	modifiers: Record<string, string>;
} {
	const t = getGlobalTheme();
	if (!t) {
		return {
			name: "unknown",
			fg: {},
			bg: {},
			modifiers: { bold: "bold", italic: "italic", underline: "underline", dim: "dim" },
		};
	}

	const fg: Record<string, string | undefined> = {};
	for (const [key, val] of t.fgColors) {
		fg[key] = val;
	}

	const bg: Record<string, string | undefined> = {};
	for (const [key, val] of t.bgColors) {
		bg[key] = val;
	}

	return {
		name: t.name,
		fg,
		bg,
		modifiers: { bold: "bold", italic: "italic", underline: "underline", dim: "dim" },
	};
}
