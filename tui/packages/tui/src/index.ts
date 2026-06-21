#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { initTheme, theme, getAvailableThemes } from "./layers/theme/theme.ts";
import { LogicianTUI } from "./layers/presentation/tui.ts";
import { loadLogicianConfig } from "@logician/coding-agent/config";

// Initialize theme before any component rendering
const config = loadLogicianConfig(process.cwd()).config;
initTheme(config.theme);

// Display theme info at startup
const themes = getAvailableThemes();
if (themes.length > 0) {
	// eslint-disable-next-line no-console -- startup theme info
	console.error(`Theme: ${theme.name} (available: ${themes.join(", ")})`);
}

const tui = new LogicianTUI();

// Graceful shutdown
let stopping = false;
const shutdown = async (): Promise<void> => {
	if (stopping) return;
	stopping = true;
	await tui.stop();
	process.exit(0);
};

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());

tui.start();
