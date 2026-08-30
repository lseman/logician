// ── Ink TUI Theme System ──────────────────────────────────────────────────────

import type { ForegroundColorName } from "ansi-styles";

export interface ThemeDefinition {
	name: string;
	fg: Record<string, string | ForegroundColorName>;
	bg: Record<string, string | ForegroundColorName>;
	modifiers: Record<string, string>;
}

// Logician dark theme
export const defaultTheme: ThemeDefinition = {
	name: "logician-dark",
	fg: {
		primary: "#e0e0e0",
		secondary: "#888888",
		accent: "#4fc3f7",
		success: "#81c784",
		warning: "#ffb74d",
		error: "#e57373",
		info: "#64b5f6",
		selected: "#1565c0",
		highlight: "#ffeb3b",
		muted: "#555555",
		link: "#82b1ff",
	},
	bg: {
		primary: "#1a1a2e",
		secondary: "#16213e",
		accent: "#0f3460",
		selected: "#1a3a5c",
		surface: "#22223a",
		overlay: "#0a0a1a",
	},
	modifiers: {
		bold: "bold",
		italic: "italic",
		underline: "underline",
		reverse: "reverse",
		dim: "dim",
	},
};

export const lightTheme: ThemeDefinition = {
	name: "logician-light",
	fg: {
		primary: "#212121",
		secondary: "#616161",
		accent: "#0277bd",
		success: "#2e7d32",
		warning: "#ef6c00",
		error: "#c62828",
		info: "#0288d1",
		selected: "#1565c0",
		highlight: "#f57f17",
		muted: "#9e9e9e",
		link: "#0277bd",
	},
	bg: {
		primary: "#fafafa",
		secondary: "#eeeeee",
		accent: "#e3f2fd",
		selected: "#bbdefb",
		surface: "#ffffff",
		overlay: "#ffffff",
	},
	modifiers: {
		bold: "bold",
		italic: "italic",
		underline: "underline",
		reverse: "reverse",
		dim: "dim",
	},
};

export const themes: Record<string, ThemeDefinition> = {
	"default": defaultTheme,
	"logician-dark": defaultTheme,
	"logician-light": lightTheme,
};

let currentTheme: ThemeDefinition = defaultTheme;

export function getCurrentTheme(): ThemeDefinition {
	return currentTheme;
}

export function setCurrentTheme(name: string): boolean {
	const theme = themes[name];
	if (!theme) return false;
	currentTheme = theme;
	return true;
}

export function getThemeColor(scope: "fg" | "bg" | "modifiers", key: string): string | ForegroundColorName {
	return currentTheme[scope][key] ?? currentTheme[scope].primary;
}

export function getAvailableThemes(): string[] {
	return Object.keys(themes);
}
