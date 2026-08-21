#!/usr/bin/env node
// ── Logician TUI — Entry point ────────────────────────────────────────────────

// Force truecolor output before terminal theme initialization.
// transitively) — chalk reads COLORTERM/TERM once at module-init time via
// `supports-color`, and terminals that report bare `TERM=xterm` with no
// COLORTERM get downgraded to basic 16-color, which mangles our hex theme
// colors (e.g. a light gray can render as the terminal's black/dark slot).
if (!process.env.FORCE_COLOR) process.env.FORCE_COLOR = "3";

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Load ~/.logician/.env so MCP servers can resolve env-var placeholders.
(function loadHomeEnv(): void {
	const home = process.env.HOME || "/";
	const envPath = join(home, ".logician", ".env");
	if (!existsSync(envPath)) return;
	const lines = readFileSync(envPath, "utf8").split("\n");
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const eq = trimmed.indexOf("=");
		if (eq < 1) continue;
		const key = trimmed.slice(0, eq);
		let value = trimmed.slice(eq + 1);
		// Strip surrounding quotes (single or double)
		if (
			value.length >= 2 &&
			((value[0] === '"' && value.at(-1) === '"') ||
				(value[0] === "'" && value.at(-1) === "'"))
		) {
			value = value.slice(1, -1);
		}
		process.env[key] = value;
	}
})();

import { AgentRuntime } from "@logician/agent-runtime/application";
import { resolveRuntimeConfig } from "@logician/agent-runtime/configuration/runtime";
import {
	buildDoctorReport,
	formatDoctorReport,
} from "@logician/agent-runtime/developer-tools";
import { activateProjectVirtualEnv } from "@logician/agent-runtime/tools";
import {
	applyTrustChoice,
	resolveTrust,
	resolveTrustInfo,
	TrustStore,
} from "@logician/agent-runtime/trust";
import { parseExecArgs, runHeadlessExec } from "./app/headless-exec.ts";
import { LogicianTUI } from "./app/tui.ts";
import {
	type TrustChoice,
	TrustPromptOverlay,
} from "./overlays/trust-prompt-overlay.ts";
import { visibleWidth } from "./terminal/core.ts";
import { getAvailableThemes, initTheme, theme } from "./terminal/theme.ts";

/** Show the trust prompt as an interactive terminal card before the main TUI. */
async function showTrustOverlay(
	cwd: string,
	paths: string[],
): Promise<TrustChoice> {
	return new Promise(resolve => {
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
			process.stdout.write(
				`\x1b[?25l\x1b[2J\x1b[H${"\n".repeat(top)}${lines.map(line => " ".repeat(left) + line).join("\n")}`,
			);
		};
		const cleanup = (): void => {
			stdin.off("data", onData);
			process.stdout.off("resize", render);
			stdin.setRawMode?.(wasRaw);
			process.stdout.write("\x1b[?25h\x1b[2J\x1b[H");
		};
		const onData = (chunk: Buffer | string): void => {
			const input = String(chunk);
			let action = overlay.handleInput(input);
			if (!action && input.length > 1 && !input.startsWith("\x1b[")) {
				for (const character of input) {
					action = overlay.handleInput(character);
					if (action) break;
				}
			}
			if (!action) return render();
			cleanup();
			resolve(action.choice);
		};
		stdin.setRawMode?.(true);
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
	activateProjectVirtualEnv(cwd);

	// ── Parse --session <id> flag ────────────────────────────────────────
	let resumeSessionId: string | undefined;
	for (let i = 0; i < args.length; i++) {
		if (args[i] === "--session" && i + 1 < args.length) {
			resumeSessionId = args[i + 1];
			break;
		}
	}

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
			const bridge = new AgentRuntime(runtimeConfig.bridge);
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
			loadProjectConfig = trustInfo.preDecidedResult?.trusted ?? false;
		} else {
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

	// Release the trust-only extension runtime before the real bridge starts.
	// In particular, do not leave its MCP clients or other lifecycle resources
	// alive alongside the application bridge.

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

	// Load explicit session if --session <id> was passed
	if (resumeSessionId) {
		try {
			const turns = tui.loadTurns(resumeSessionId);
			if (turns && turns.length > 0) {
				// Drop the empty session auto-created on startup — new turns
				// must accumulate onto the resumed session, not a fresh one.
				const staleSessionId = tui.currentSessionId;
				tui.restoreSession(turns);
				tui.sessionService.setCurrentSession(resumeSessionId);
				tui.currentSessionId = resumeSessionId;
				if (staleSessionId && staleSessionId !== resumeSessionId) {
					tui.sessionService.deleteSession(staleSessionId);
				}
			}
		} catch (error: unknown) {
			const message = error instanceof Error ? error.message : String(error);
			process.stderr.write(
				`error: failed to load session ${resumeSessionId}: ${message}\n`,
			);
		}
	}

	tui.start();
	let stopping = false;
	const shutdown = async (): Promise<void> => {
		if (stopping) return;
		stopping = true;
		await tui.stop();
		const recoveryTip = tui.getSessionRecoveryTip();
		if (recoveryTip) process.stderr.write(recoveryTip);
		process.exit(0);
	};
	tui.setExitHandler(() => void shutdown());
	process.on("SIGINT", () => void shutdown());
	process.on("SIGTERM", () => void shutdown());
}

void main();
