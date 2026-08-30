// ── Ink TUI — slash command wiring ───────────────────────────────────────────
// Builds the real runtime slash-command registry and a compact dispatcher.
// The full custom-terminal TUI special-cases ~20 commands (see
// apps/tui/src/app/commands/submit-handler.ts); the Ink MVP covers the common
// dispatch paths (bridge / local / state / quit) plus a few high-value opens.

import {
	createSlashCommands,
	filterSlashCommands,
	type SlashCommandDef,
} from "@logician/log-runtime/commands";
import type { AgentRuntime } from "@logician/log-runtime/application";
import type { Transcript } from "@logician/log-runtime/sessions";
import type { TuiState } from "./state.ts";
import type { OverlayKind } from "./types.ts";

/** Slash commands that just open an Ink overlay (when given no args). */
const OVERLAY_OPENERS: Record<string, OverlayKind> = {
	"/model": "modelSelector",
	"/models": "modelSelector",
	"/session": "sessionManager",
	"/sessions": "sessionManager",
	"/queue": "queueManager",
	"/theme": "themeSelector",
	"/plugins": "pluginManager",
	"/mcp": "mcpManager",
	"/reasoner": "reasonerSelector",
	"/settings": "settingsSelector",
	"/thinking-steps": "thinkingLevelSelector",
};

export interface SlashContext {
	bridge: AgentRuntime;
	transcript: Transcript;
	state: TuiState;
	onExit: () => void;
}

/** Local handlers passed to createSlashCommands(). Partial — missing keys
 * degrade to the built-in fallback strings (all call sites use `?.`). */
function localHandlers(ctx: SlashContext): Record<string, (...a: unknown[]) => unknown> {
	const { bridge, transcript, state } = ctx;
	return {
		clear: () => {
			transcript.clear();
			state.setTranscriptTurns(transcript.getTurns());
			return "";
		},
		version: () => "Logician (Ink TUI)",
		setThinking: (level: unknown) => {
			state.setThinkingLevel(String(level) as never);
			return `Thinking level: ${String(level)}`;
		},
		setThinkingMode: (mode: unknown) => {
			state.setThinkingDisplayMode(String(mode) as never);
			return `Thinking display: ${String(mode)}`;
		},
		cycleThinking: () => {
			const order = ["collapsed", "summary", "expanded"] as const;
			const next = order[(order.indexOf(state.thinkingDisplayMode) + 1) % order.length]!;
			state.setThinkingDisplayMode(next);
			return `Thinking display: ${next}`;
		},
		setPermissionMode: (mode: unknown) => {
			bridge.setPermissionMode(String(mode) as never);
			state.setPermissionMode(String(mode) as never);
			return `Permission mode: ${String(mode)}`;
		},
		getPermissionMode: () => bridge.getPermissionMode(),
		openModelSelector: () => state.setOverlay("modelSelector"),
		openSessionManager: () => state.setOverlay("sessionManager"),
		openQueueManager: () => state.setOverlay("queueManager"),
		openSettingsSelector: () => state.setOverlay("settingsSelector"),
	};
}

export function buildSlashCommands(ctx: SlashContext): SlashCommandDef[] {
	return createSlashCommands(ctx.bridge, localHandlers(ctx));
}

export function filterCommands(
	commands: SlashCommandDef[],
	query: string,
): SlashCommandDef[] {
	return filterSlashCommands(commands, query);
}

/** Run a selected/typed slash command. `raw` is the full line incl. leading "/". */
export function dispatchSlash(
	ctx: SlashContext,
	commands: SlashCommandDef[],
	raw: string,
): void {
	const { bridge, transcript, state } = ctx;
	const line = raw.trim();
	if (!line) return;

	const name = line.split(/\s+/)[0]!.toLowerCase();
	const args = line.split(/\s+/).slice(1).join(" ");
	const match = commands.find(c => c.command.toLowerCase() === name);

	// Pure overlay openers: don't echo, just open the panel.
	if (!args && OVERLAY_OPENERS[name]) {
		state.setOverlay(OVERLAY_OPENERS[name]!);
		return;
	}

	// Echo the command into the transcript so async output shares the turn.
	transcript.addTurn(line);
	state.setTranscriptTurns(transcript.getTurns());

	if (!match) {
		transcript.addSystemMessage(`Unknown command: ${name}`);
		state.setTranscriptTurns(transcript.getTurns());
		return;
	}

	const finish = (): void => state.setTranscriptTurns(transcript.getTurns());

	switch (match.dispatch) {
		case "quit":
			ctx.onExit();
			return;
		case "local": {
			const res = match.handler?.(args);
			if (res) transcript.addSystemMessage(String(res));
			finish();
			return;
		}
		case "state":
			void bridge.getState().then(snapshot => {
				transcript.addSystemMessage(
					`Runtime state:\n${JSON.stringify(snapshot, null, 2)}`,
				);
				finish();
			});
			return;
		default:
			// "bridge" — forward verbatim; the runtime handles /new, /compact, …
			void Promise.resolve(bridge.sendSlash(line)).then(finish, finish);
			return;
	}
}
