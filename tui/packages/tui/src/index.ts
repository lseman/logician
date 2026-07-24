#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { createInterface } from "node:readline/promises";
import { initTheme, theme, getAvailableThemes } from "./layers/theme/theme.ts";
import {
	AgentCoreBridge,
	buildDoctorReport,
	formatDoctorReport,
	resolveRuntimeConfig,
} from "@logician/coding-agent/runtime";
import { resolveTrust } from "@logician/coding-agent/trust";
import { parseExecArgs, runHeadlessExec } from "./headless-exec.ts";

function defaultProjectTrust(): "ask" | "always" | "never" {
	const value = process.env.LOGICIAN_TRUST?.trim().toLowerCase();
	if (value === "always" || value === "1" || value === "true") return "always";
	if (value === "never" || value === "0" || value === "false") return "never";
	return "ask";
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	const cwd = process.cwd();
	if (args[0] === "exec") {
		try {
			const execArgs = parseExecArgs(args.slice(1));
			const trust = await resolveTrust({
				cwd,
				hasUI: false,
				defaultProjectTrust: defaultProjectTrust(),
			});
			const runtimeConfig = resolveRuntimeConfig(cwd, process.env, {
				loadProjectConfig: trust.trusted,
			});
			for (const warning of runtimeConfig.warnings) {
				process.stderr.write(`warning: ${warning}\n`);
			}
			const bridge = new AgentCoreBridge(runtimeConfig.bridge);
			process.exitCode = await runHeadlessExec(bridge, {
				...execArgs,
				cwd,
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
		const report = await buildDoctorReport(cwd);
		process.stdout.write(
			args.includes("--json")
				? `${JSON.stringify(report, null, 2)}\n`
				: `${formatDoctorReport(report)}\n`,
		);
		process.exitCode = report.config.valid && report.workspace.present ? 0 : 1;
		return;
	}
	const trust = await resolveTrust({
		cwd,
		hasUI: Boolean(process.stdin.isTTY && process.stdout.isTTY),
		defaultProjectTrust: defaultProjectTrust(),
		onSelectTrust: async (prompt) => {
			const readline = createInterface({
				input: process.stdin,
				output: process.stdout,
			});
			try {
				const answer = (
					await readline.question(
						`${prompt}\n\n[y] trust  [p] trust parent  [s] session only  [n] deny: `,
					)
				).trim().toLowerCase();
				if (answer === "y" || answer === "yes") return "trust";
				if (answer === "p" || answer === "parent") return "trust-parent";
				if (answer === "s" || answer === "session") return "session-only";
				return "deny-session";
			} finally {
				readline.close();
			}
		},
	});
	const runtimeConfig = resolveRuntimeConfig(cwd, process.env, {
		loadProjectConfig: trust.trusted,
	});

	// Initialize theme before any component rendering
	initTheme(runtimeConfig.source.theme);

	// Display theme info at startup
	const themes = getAvailableThemes();
	if (themes.length > 0) {
		// eslint-disable-next-line no-console -- startup theme info
		console.error(`Theme: ${theme.name} (available: ${themes.join(", ")})`);
	}

	const { LogicianTUI } = await import("./layers/presentation/tui.ts");
	const tui = new LogicianTUI(runtimeConfig);

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
