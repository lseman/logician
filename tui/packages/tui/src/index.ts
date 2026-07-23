#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { initTheme, theme, getAvailableThemes } from "./layers/theme/theme.ts";
import { loadLogicianConfig } from "@logician/coding-agent/config";
import {
	AgentCoreBridge,
	buildDoctorReport,
	formatDoctorReport,
	resolveRuntimeConfig,
} from "@logician/coding-agent/runtime";
import { parseExecArgs, runHeadlessExec } from "./headless-exec.ts";

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	if (args[0] === "exec") {
		try {
			const execArgs = parseExecArgs(args.slice(1));
			const runtimeConfig = resolveRuntimeConfig(process.cwd());
			for (const warning of runtimeConfig.warnings) {
				process.stderr.write(`warning: ${warning}\n`);
			}
			const bridge = new AgentCoreBridge(runtimeConfig.bridge);
			process.exitCode = await runHeadlessExec(bridge, {
				...execArgs,
				cwd: process.cwd(),
				stdout: process.stdout,
				stderr: process.stderr,
			});
		} catch (error: unknown) {
			const message = error instanceof Error ? error.message : String(error);
			process.stderr.write(`${message}\n`);
			process.exitCode = 2;
		}
		return;
	}
	if (args[0] === "doctor" || args.includes("--doctor")) {
		const report = await buildDoctorReport(process.cwd());
		process.stdout.write(
			args.includes("--json")
				? `${JSON.stringify(report, null, 2)}\n`
				: `${formatDoctorReport(report)}\n`,
		);
		process.exitCode = report.config.valid && report.workspace.present ? 0 : 1;
		return;
	}

	// Initialize theme before any component rendering
	const config = loadLogicianConfig(process.cwd()).config;
	initTheme(config.theme);

	// Display theme info at startup
	const themes = getAvailableThemes();
	if (themes.length > 0) {
		// eslint-disable-next-line no-console -- startup theme info
		console.error(`Theme: ${theme.name} (available: ${themes.join(", ")})`);
	}

	const { LogicianTUI } = await import("./layers/presentation/tui.ts");
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
}

void main();
