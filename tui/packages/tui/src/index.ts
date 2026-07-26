#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { initTheme, theme, getAvailableThemes } from "./layers/theme/theme.ts";
import {
	AgentCoreBridge,
	buildDoctorReport,
	formatDoctorReport,
	resolveRuntimeConfig,
} from "@logician/coding-agent/runtime";
import {
	applyTrustChoice,
	resolveTrust,
	resolveTrustInfo,
	TrustStore,
} from "@logician/coding-agent/trust";
import { parseExecArgs, runHeadlessExec } from "./headless-exec.ts";
import { TrustPromptOverlay, type TrustChoice } from "./components/trust-prompt-overlay.ts";
import { LogicianTUI } from "./layers/presentation/tui.ts";

/** Show the trust prompt overlay visually, then use readline for input. */
async function showTrustOverlay(
	cwd: string,
	paths: string[],
): Promise<TrustChoice> {
	return new Promise((resolve) => {
		// Initialize a default theme for overlay rendering
		initTheme();

		const { createInterface } = require("node:readline");
		const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
		const overlay = new TrustPromptOverlay();
		overlay.setOptions({ cwd, paths });
		overlay.show();

		const width = process.stdout.columns ?? 80;
		const lines = overlay.render(width);

		// Save terminal state and show the overlay
		process.stdout.write(`\x1b[?25l\x1b[H${lines.join("\n")}\n`);

		rl.question(
			"\n[y] trust  [p] trust parent  [s] session  [n] deny  [N] deny session → ",
			(answer: string) => {
				const choice = parseTrustAnswer(answer.trim().toLowerCase());
				// Restore terminal state
				process.stdout.write(`\x1b[?25h\x1b[${lines.length + 2}A`);
				resolve(choice);
			},
		);
	});
}

function parseTrustAnswer(answer: string): TrustChoice {
	if (answer === "y" || answer === "yes") return "trust";
	if (answer === "p" || answer === "parent") return "trust-parent";
	if (answer === "s" || answer === "session") return "session-only";
	if (answer === "n" || answer === "no" || answer === "deny") return "deny";
	return "deny-session";
}

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
	// Detect terminal mode
	const hasUI = Boolean(process.stdin.isTTY && process.stdout.isTTY);

	// ── Resolve trust ─────────────────────────────────────────────────────

	let loadProjectConfig = false;

	if (hasUI) {
		// ── TUI mode: show trust overlay ─────────────────────────────────
		const trustInfo = resolveTrustInfo(cwd, defaultProjectTrust());

		if (trustInfo.preDecided) {
			loadProjectConfig = trustInfo.preDecidedResult!.trusted;
		} else {
			// Show trust overlay via readline (visually formatted)
			const choice = await showTrustOverlay(cwd, trustInfo.paths);
			const store = new TrustStore();
			const result = applyTrustChoice(store, choice, cwd);
			if (!result.trusted) {
				process.exit(1);
			}
			loadProjectConfig = true;
		}
	} else {
		// ── CLI mode: resolve trust silently or with readline ────────────
		const trust = await resolveTrust({
			cwd,
			hasUI: false,
			defaultProjectTrust: defaultProjectTrust(),
		});
		loadProjectConfig = trust.trusted;
	}

	const runtimeConfig = resolveRuntimeConfig(cwd, process.env, {
		loadProjectConfig,
	});

	// Initialize theme before any component rendering
	initTheme(runtimeConfig.source.theme);

	// Display theme info at startup
	const themes = getAvailableThemes();
	if (themes.length > 0) {
		// eslint-disable-next-line no-console -- startup theme info
		console.error(`Theme: ${theme.name} (available: ${themes.join(", ")})`);
	}

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
