import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

// Minimal dark theme that satisfies the TUI's initTheme() requirement.
const DARK_THEME_JSON = JSON.stringify({
	name: "dark",
	vars: {
		"cyan": "#00d7ff",
		"blue": "#5f87ff",
		"green": "#b5bd68",
		"red": "#cc6666",
		"yellow": "#ffff00",
		"text": "#d4d4d4",
		"gray": "#808080",
		"dimGray": "#666666",
		"darkGray": "#505050",
		"accent": "#8abeb7",
	},
	colors: {
		"accent": "accent",
		"border": "blue",
		"borderMuted": "darkGray",
		"success": "green",
		"error": "red",
		"warning": "yellow",
		"muted": "gray",
		"dim": "dimGray",
		"text": "text",
		"userText": "text",
		"assistantText": "text",
		"systemText": "text",
		"mdHeading": "#f0c674",
		"mdCode": "accent",
		"mdCodeBlock": "green",
		"mdCodeBlockBg": "#1e1e24",
		"mdCodeBlockBorder": "gray",
		"mdLink": "#81a2be",
		"mdQuote": "gray",
		"mdListBullet": "accent",
		"toolTitle": "text",
		"toolRunning": "#f0c674",
		"toolSuccess": "green",
		"toolError": "red",
		"toolStreaming": "#f0c674",
		"toolOutput": "gray",
		"thinkingText": "gray",
		"separator": "gray",
		"prompt": "green",
		"inputText": "text",
		"phaseReady": "green",
		"phaseThinking": "#81a2be",
		"phaseTool": "#f0c674",
		"phaseError": "red",
		"phaseStreaming": "#f0c674",
		"phaseCompacting": "#81a2be",
		"phaseBranching": "#81a2be",
		"contextGood": "green",
		"contextWarning": "yellow",
		"contextCritical": "red",
		"levelOff": "#505050",
		"levelLow": "#6e6e6e",
		"levelMedium": "#5f87af",
		"levelHigh": "#5f87af",
		"levelXhigh": "#81a2be",
		"diffAdded": "green",
		"diffRemoved": "red",
		"diffContext": "gray",
		"diffHunk": "#f0c674",
		"diffMeta": "gray",
		"terminalOutput": "text",
		"memoryTag": "#81a2be",
		"memoryId": "#81a2be",
		"memoryContent": "text",
		"memoryCount": "#81a2be",
		"pluginStartup": "green",
		"header": "accent",
		"active": "accent",
		"selected": "#3a3a4a",
		"inputPlaceholder": "dimGray",
		"jsonKey": "#81a2be",
		"jsonKeyword": "#c678dd",
		"jsonNumber": "#b5bd68",
		"jsonString": "#abb2bf",
		"jsonPunctuation": "#808080",
	},
});

/**
 * Create an isolated HOME directory with a valid theme file, so PTY tests
 * that boot the real LogicianTUI app (real AgentCoreBridge, real SQLite
 * memory store under ~/.logician) don't collide with each other or with the
 * developer's own ~/.logician when run concurrently. Without this, parallel
 * PTY tests hit "SQLiteError: database is locked" against a shared DB file.
 */
export function createPtyAppHome(): string {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	writeFileSync(path.join(themeDir, "dark.json"), DARK_THEME_JSON);
	return home;
}
