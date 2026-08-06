// ── Slash-command submission dispatcher ────────────────────────────────────

import { formatContextSize } from "@logician/coding-agent";
import { GoalManager } from "@logician/coding-agent/application";
import type { SlashCommandDef } from "@logician/coding-agent/commands";
import { describeSandboxProfile, parseLoopInterval } from "../tui-helpers.ts";
import {
	handleMcp,
	handlePlugins,
	handleReasoner,
	handleStatus,
	handleTheme,
} from "./async-handlers.ts";
import type { SlashCommandsCtx } from "./context.ts";

export function createSlashSubmitHandler(
	ctx: SlashCommandsCtx,
): (
	result: unknown,
	dispatch: string | undefined,
	command: string | undefined,
) => Promise<void> {
	return async (result, dispatch, command) => {
		if (dispatch === "quit") {
			ctx.requestExit();
			return;
		}
		// Add slash command as user message to transcript. Called before local
		// handlers that spawn async work so tool/stream events share this turn.
		if (command?.trim()) {
			ctx.transcript.addTurn(command.trim());
		}
		// Handler return text (may arrive in a second onSubmit with no command).
		if (result) {
			ctx.transcript.addSystemMessage(String(result));
		}
		if (command?.trim()) {
			const cmdName = command.trim().split(/\s+/)[0]?.toLowerCase() || "";
			const args = command.trim().split(/\s+/).slice(1).join(" ");
			const allCmds = ctx.slashPopup.getCommands() as SlashCommandDef[];
			const match = allCmds?.find(
				(c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
			);
			if (match && match.command === "/plugins") {
				void handlePlugins(ctx, args);
			}
			if (match && match.command === "/mcp") {
				void handleMcp(ctx, args);
			}
			if (match && match.command === "/reasoner") {
				void handleReasoner(ctx, args);
			}
			if (match && match.command === "/theme") {
				void handleTheme(ctx, args);
			}
			if (match && match.command === "/compact") {
				void ctx.bridge.compact().then(result => {
					if (result === null) {
						ctx.transcript.addSystemMessage("Nothing to compact.");
					} else {
						ctx.transcript.addSystemMessage(
							`Context compacted (${formatContextSize(
								result.tokensBefore,
							)} -> ${formatContextSize(
								result.tokensAfter,
							)}). Saved ${formatContextSize(result.tokensSaved)}.`,
						);
					}
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
				});
				return;
			}
			if (match && match.command === "/fork") {
				const id = ctx.bridge.fork();
				ctx.transcript.addSystemMessage(
					id ? `Forked conversation (${id}).` : "Nothing to fork.",
				);
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/branch-summary") {
				void ctx.bridge.branchSummary().then(summary => {
					ctx.transcript.addSystemMessage(
						summary === null
							? "No active branch to summarize."
							: `Branch merged: ${summary}`,
					);
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
				});
				return;
			}
			if (match && match.command === "/discard-branch") {
				const ok = ctx.bridge.discardBranch();
				ctx.transcript.addSystemMessage(
					ok ? "Branch discarded." : "No active branch.",
				);
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			// Plain /sessions opens the browser. Subcommands such as
			// /sessions clean are handled by the local command handler.
			if (match && match.command === "/sessions" && !args.trim()) {
				ctx.openSessionManager();
				return;
			}
			if (match && match.command === "/loop") {
				const args = command.trim().split(/\s+/).slice(1).join(" ");
				if (args.toLowerCase() === "stop") {
					ctx.loopManager.stop();
					ctx.loopActive = false;
					ctx.transcript.addSystemMessage("Loop stopped.");
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
					return;
				}
				const parsedInterval = parseLoopInterval(args);
				if (parsedInterval) {
					const { value, unit, prompt, ms } = parsedInterval;
					ctx.loopManager.start(prompt, ms);
					ctx.loopActive = true;
					ctx.transcript.addSystemMessage(
						`🔄 Loop started: "${prompt}" every ${value}${unit}`,
					);
				} else if (args) {
					// No interval specified — default to 5 minutes
					ctx.loopManager.start(args, 5 * 60 * 1000);
					ctx.loopActive = true;
					ctx.transcript.addSystemMessage(
						`🔄 Loop started: "${args}" (default 5m interval)`,
					);
				} else {
					ctx.transcript.addSystemMessage(
						"Usage: /loop <prompt> [interval] — e.g. /loop 5m check the deploy\n" +
							"Or: /loop stop",
					);
				}
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/goal") {
				const args = command.trim().split(/\s+/).slice(1).join(" ");
				if (args.toLowerCase() === "clear") {
					ctx.goalManager.cancel();
					ctx.goalActive = false;
					ctx.transcript.addSystemMessage("Goal cleared.");
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
					return;
				}
				if (!args) {
					// Show goal status
					const state = ctx.goalManager.getState();
					if (!state) {
						ctx.transcript.addSystemMessage("No goal set.");
					} else if (state.status === "achieved") {
						const dur = Math.round(
							((state.achievedAt ?? Date.now()) - state.startedAt) / 1000,
						);
						ctx.transcript.addSystemMessage(
							`Goal achieved: "${state.condition}"\n` +
								`Duration: ${dur}s, Turns: ${state.turnCount}, Reason: ${state.lastReason || "N/A"}`,
						);
					} else if (state.status === "cancelled") {
						ctx.transcript.addSystemMessage(
							`Goal cancelled: "${state.condition}"\n` +
								`Turns: ${state.turnCount}, Reason: ${state.lastReason || "N/A"}`,
						);
					} else {
						const elapsed = Math.round((Date.now() - state.startedAt) / 1000);
						ctx.transcript.addSystemMessage(
							`Goal active: "${state.condition}"\n` +
								`Running: ${elapsed}s, Turns: ${state.turnCount}${state.maxTurns ? ` / ${state.maxTurns}` : ""}\n` +
								`Last: ${state.lastReason || "evaluating..."}`,
						);
					}
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
					return;
				}
				// Parse condition, extracting optional turn limit
				const parsed = GoalManager.parseCondition(args);
				ctx.goalManager.set(parsed.condition, parsed.maxTurns);
				ctx.goalActive = true;
				const info = parsed.maxTurns ? ` (max ${parsed.maxTurns} turns)` : "";
				ctx.transcript.addSystemMessage(
					`◎ Goal set: "${parsed.condition}"${info}`,
				);
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/sandbox") {
				const parts = args.trim().split(/\s+/);
				const sub = parts[0]?.toLowerCase() ?? "";
				const subArgs = parts.slice(1).join(" ");

				// /sandbox status
				if (sub === "status") {
					const { spawnSync } = await import("node:child_process");
					const { existsSync } = await import("node:fs");
					const pathMod = await import("node:path");
					const bwrapPath = process.env.PATH?.split(pathMod.delimiter).find(d =>
						existsSync(pathMod.join(d, "bwrap")),
					)
						? pathMod.join(
								process.env.PATH?.split(pathMod.delimiter).find(d =>
									existsSync(pathMod.join(d, "bwrap")),
								)!,
								"bwrap",
							)
						: null;

					let bwrapVersion = "unknown";
					if (bwrapPath) {
						const result = spawnSync(bwrapPath, ["--version"], {
							timeout: 5000,
							stdio: ["ignore", "pipe", "pipe"],
						});
						if (result.status === 0 && result.stdout) {
							bwrapVersion = result.stdout.toString().trim();
						}
					}

					const isLinux = process.platform === "linux";
					const available = !!bwrapPath && isLinux;

					ctx.transcript.addSystemMessage(
						`Sandbox availability: ${available ? "OK" : "unavailable"}` +
							`${bwrapPath ? ` (bwrap found at ${bwrapPath})` : " (bwrap not found)"}` +
							`${!isLinux ? ` (not on Linux: ${process.platform})` : ""}` +
							`${bwrapVersion !== "unknown" ? ` — ${bwrapVersion}` : ""}` +
							"\n\nProfiles: none (no isolation), code (read-only host fs, writable /tmp, no network/devices), file (code + bind mounts), dev (code + /dev), full (code + namespaces)",
					);
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
					return;
				}

				// /sandbox profile <level>
				if (sub === "profile") {
					ctx.transcript.addSystemMessage(describeSandboxProfile(subArgs));
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
					return;
				}

				// /sandbox <command> — dispatch to sandbox tool via bridge
				const cmd = subArgs || args.trim();
				if (!cmd) {
					ctx.transcript.addSystemMessage(
						"Usage:\n" +
							"  /sandbox <command>       — run with CODE profile\n" +
							"  /sandbox profile <level> — show profile info\n" +
							"  /sandbox status          — check sandbox availability",
					);
				} else {
					// Check if first word is a profile name
					const profileNames = ["none", "code", "full"];
					const firstWord = cmd.split(/\s+/)[0]?.toLowerCase();
					let actualCommand = cmd;
					let profileHint = "code";

					if (firstWord && profileNames.includes(firstWord)) {
						profileHint = firstWord;
						actualCommand = cmd.slice(firstWord.length).trim();
					}

					ctx.transcript.addSystemMessage(
						`Running in sandbox (profile: ${profileHint}): ${actualCommand}`,
					);
					// Dispatch to the sandbox tool via the bridge
					void ctx.bridge.sendSlash(`/sandbox ${actualCommand}`);
				}
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/new") {
				ctx._autoSaveTurn();
				ctx.currentSessionId = ctx.sessionStore.createSession({
					title: "New Session",
				});
				ctx.bridge.useConversationSession(ctx.currentSessionId);
				ctx.transcript.clear();
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.statusPanel.update({ sessionTitle: "New Session" });
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/save") {
				ctx._autoSaveTurn();
				ctx.transcript.addSystemMessage("Session saved.");
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.command === "/rename") {
				if (ctx.currentSessionId && args.trim()) {
					ctx.sessionStore.renameSession(ctx.currentSessionId, args.trim());
					ctx.bridge.renameConversationSession(
						ctx.currentSessionId,
						args.trim(),
					);
					ctx.statusPanel.update({ sessionTitle: args.trim() });
					ctx.transcript.addSystemMessage(
						`Session renamed to "${args.trim()}"`,
					);
				}
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.tui.requestRender();
				return;
			}
			if (match && match.dispatch === "bridge") {
				void ctx.bridge.sendSlash(command.trim());
			}
			if (match && match.dispatch === "state") {
				void handleStatus(ctx);
			}
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	};
}
