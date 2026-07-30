#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

import { getAvailableThemes, initTheme, theme } from "./terminal/theme.ts";
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
import { parseExecArgs, runHeadlessExec } from "./app/headless-exec.ts";
import {
	TrustPromptOverlay,
	type TrustChoice,
} from "./overlays/trust-prompt-overlay.ts";
import { visibleWidth } from "./terminal/core.ts";
import { LogicianTUI } from "./app/tui.ts";

/** Show the trust prompt as an interactive terminal card before the main TUI. */
async function showTrustOverlay(
	cwd: string,
	paths: string[],
): Promise<TrustChoice> {
	return new Promise((resolve) => {
		initTheme();
		const overlay = new TrustPromptOverlay();
		overlay.setOptions({ cwd, paths });
		overlay.show();
		const stdin = process.stdin;
		const wasRaw = stdin.isRaw === true;

		const render = (): void => {
			const width = Math.max(24, process.stdout.columns ?? 80);
			const height = Math.max(12, process.stdout.rows ?? 24);
			const lines = overlay.render(width);
			const left = Math.max(
				0,
				Math.floor((width - Math.max(...lines.map(visibleWidth))) / 2),
			);
			const top = Math.max(0, height - lines.length - 3);
			const inset = " ".repeat(left);
			process.stdout.write(
				`\x1b[?25l\x1b[2J\x1b[H${"\n".repeat(top)}${lines
					.map((line) => inset + line)
					.join("\n")}`,
			);
		};
		const cleanup = (): void => {
			stdin.off("data", onData);
			process.stdout.off("resize", render);
			if (stdin.setRawMode) stdin.setRawMode(wasRaw);
			process.stdout.write("\x1b[?25h\x1b[2J\x1b[H");
		};
		const onData = (chunk: Buffer | string): void => {
			const input = String(chunk);
			let action = overlay.handleInput(input);
			// PTYs and some terminals batch a shortcut with Enter. Preserve
			// escape sequences as one key, but replay ordinary character batches.
			if (!action && input.length > 1 && !input.startsWith("\x1b[")) {
				for (const character of input) {
					action = overlay.handleInput(character);
					if (action) break;
				}
			}
			if (!action) {
				render();
				return;
			}
			cleanup();
			resolve(action.choice);
		};

		if (stdin.setRawMode) stdin.setRawMode(true);
		stdin.resume();
		stdin.on("data", onData);
		process.stdout.on("resize", render);
		render();
	});
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
