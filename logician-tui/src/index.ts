#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { LogicianTUI } from "./tui.ts";

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
