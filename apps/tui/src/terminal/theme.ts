// ── Theme System ──────────────────────────────────────────────────────────────
// Color themes for the Logician TUI, modeled after pi's theme system.
// Supports hex (#rrggbb), 256-color (0-255), variable references, and default ("").

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// ── Color token names ─────────────────────────────────────────────────────────

export type ThemeColor =
	// Core UI
	| "accent"
	| "success"
	| "error"
	| "warning"
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

export type ThemeBg = "mdCodeBlockBg";

// ── Schema types ──────────────────────────────────────────────────────────────

interface ThemeJson {
	name: string;
	vars?: Record<string, string | number>;
	colors: Record<string, string | number>;
}

// ── Color resolution ──────────────────────────────────────────────────────────

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
	const resolvedValue =
		vars[value] ??
		(() => {
			throw new Error(`Variable not found: ${value}`);
		})();
	return resolveVarRefs(resolvedValue, vars, visited);
}

function valueToAnsi(
	value: string | number,
	mode: "truecolor" | "256color",
	isBg: boolean,
): string {
	if (value === "") return isBg ? "\x1b[49m" : "\x1b[39m";
	if (typeof value === "number") {
		return isBg ? `\x1b[48;5;${value}m` : `\x1b[38;5;${value}m`;
	}
	// hex
	if (mode === "truecolor") {
		const { r, g, b } = hexToRgb(value);
		return isBg ? `\x1b[48;2;${r};${g};${b}m` : `\x1b[38;2;${r};${g};${b}m`;
	} else {
		const idx = rgbTo256(
			...(Object.values(hexToRgb(value)) as [number, number, number]),
		);
		return isBg ? `\x1b[48;5;${idx}m` : `\x1b[38;5;${idx}m`;
	}
}

// ── Theme Class ───────────────────────────────────────────────────────────────

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const UNDERLINE = "\x1b[4m";

export class Theme {
	readonly name: string;
	readonly mode: "truecolor" | "256color";

	private fgCache = new Map<ThemeColor, string>();
	private bgCache = new Map<ThemeBg, string>();

	constructor(
		name: string,
		mode: "truecolor" | "256color",
		fgColors: Map<ThemeColor, string>,
		bgColors: Map<ThemeBg, string>,
	) {
		this.name = name;
		this.mode = mode;
		this.fgCache = fgColors;
		this.bgCache = bgColors;
	}

	// ── Public color methods ──────────────────────────────────────────────────

	fg(color: ThemeColor, text: string): string {
		const ansi = this.fgCache.get(color);
		if (!ansi) throw new Error(`Unknown theme color: ${color}`);
		return `${ansi}${text}${RESET}`;
	}

	bg(color: ThemeBg, text: string): string {
		const ansi = this.bgCache.get(color);
		if (!ansi) throw new Error(`Unknown theme bg: ${color}`);
		return `${ansi}${text}${RESET}`;
	}

	/** Get raw ANSI color code without trailing reset. Use for composing custom styles. */
	fgRaw(color: ThemeColor): string {
		const ansi = this.fgCache.get(color);
		if (!ansi) throw new Error(`Unknown theme color: ${color}`);
		return ansi;
	}

	bgRaw(color: ThemeBg): string {
		const ansi = this.bgCache.get(color);
		if (!ansi) throw new Error(`Unknown theme bg: ${color}`);
		return ansi;
	}

	/** Derive a background ANSI code from a foreground color's fg code (38;... → 48;...). */
	fgAsBg(color: ThemeColor): string {
		const ansi = this.fgCache.get(color);
		if (!ansi) throw new Error(`Unknown theme color: ${color}`);
		return ansi.replace("\x1b[38;", "\x1b[48;");
	}

	bold(text: string): string {
		return `${BOLD}${text}${RESET}`;
	}

	dim(text: string): string {
		return `${DIM}${text}${RESET}`;
	}

	underline(text: string): string {
		return `${UNDERLINE}${text}${RESET}`;
	}

	// ── Convenience helpers ───────────────────────────────────────────────────

	codeBlockBg(text: string): string {
		return `${this.bgCache.get("mdCodeBlockBg") ?? ""}${DIM}${text}${RESET}`;
	}

	thinkingBorderColor(
		level: "off" | "low" | "medium" | "high" | "xhigh",
	): string {
		const map = {
			off: "levelOff",
			low: "levelLow",
			medium: "levelMedium",
			high: "levelHigh",
			xhigh: "levelXhigh",
		} as const;
		return this.fg(map[level], "");
	}

	phaseColor(phase: string): string {
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
		return this.fg(color, "");
	}

	/** Create a styled text token: <color><text></color> */
	style(color: ThemeColor, text: string): string {
		return this.fg(color, text);
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

// Export a lazy proxy so code can import `theme` without initialization concerns
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

function getThemesDir(): string {
	const home = process.env.HOME || "";
	return join(home, ".logician", "themes");
}

// Themes bundled with the package, used when a theme isn't found under the
// user's ~/.logician/themes (fresh installs, CI, sandboxed HOME dirs).
const BUNDLED_THEMES_DIR = join(
	dirname(fileURLToPath(import.meta.url)),
	"..",
	"..",
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
	const bgColors = new Map<ThemeBg, string>();

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

	const bgKeys: ThemeBg[] = ["mdCodeBlockBg"];
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
			fgColors.set(key, valueToAnsi(resolved, mode, false));
		}
	}

	for (const key of bgKeys) {
		const raw = (json.colors as Record<string, unknown>)[key];
		if (raw !== undefined && raw !== null) {
			const resolved = resolveVarRefs(raw as string | number, vars);
			bgColors.set(key, valueToAnsi(resolved, mode, true));
		}
	}

	return new Theme(json.name, mode, fgColors, bgColors);
}

export function getAvailableThemes(): string[] {
	const themes: string[] = [];
	const seen = new Set<string>();

	for (const dir of [getThemesDir(), BUNDLED_THEMES_DIR]) {
		try {
			if (existsSync(dir)) {
				for (const file of readdirSync(dir)) {
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
	}

	return themes.sort();
}

function loadTheme(name: string): Theme {
	const userPath = join(getThemesDir(), `${name}.json`);
	const path = existsSync(userPath)
		? userPath
		: join(BUNDLED_THEMES_DIR, `${name}.json`);
	if (!existsSync(path)) {
		throw new Error(
			`Theme not found: ${name} (looked in ${getThemesDir()} and ${BUNDLED_THEMES_DIR})`,
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
		// Fallback to dark
		console.error("[theme] initTheme failed:", _e);
		setGlobalTheme(loadTheme("dark"));
	}
}

export function getCurrentThemeName(): string {
	const t = getGlobalTheme();
	return t?.name ?? "unknown";
}

export function setTheme(name: string): boolean {
	try {
		const t = loadTheme(name);
		setGlobalTheme(t);
		return true;
	} catch (_e: unknown) {
		return false;
	}
}
