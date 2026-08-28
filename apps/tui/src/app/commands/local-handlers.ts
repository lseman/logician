// ── Local slash-command registry ───────────────────────────────────────────

import {
	saveConfigField,
	saveConfigNestedField,
} from "@logician/log-runtime/configuration";
import { theme } from "../../terminal/theme.ts";
import type { SlashCommandsCtx } from "./context.ts";

/** Minimal shape of a Memoriam observation as returned by the SDK worker. */
interface MemoriamObservation {
	id: string;
	type: string;
	title?: string;
	narrative?: string;
	importance?: number;
	timestamp?: string;
}

function compactObservationLine(
	observation: MemoriamObservation,
	index: number,
): string {
	const number = theme.fg("memoryCount", `#${index + 1}`);
	const shortId =
		observation.id.length > 12
			? `${observation.id.slice(0, 12)}…`
			: observation.id;
	const id = theme.fg("memoryId", shortId);
	const title =
		observation.title || observation.narrative?.slice(0, 100) || "No title";
	return `${number} · ${id} · importance ${observation.importance ?? 0}/10 · ${observation.type} · ${(observation.timestamp ?? "").slice(0, 19)} · ${title.replace(/\s+/g, " ")}`;
}

export function createLocalHandlers(
	ctx: SlashCommandsCtx,
): Record<string, (...args: unknown[]) => unknown> {
	const setStatusPhase = (phase: string) => {
		ctx.statusPanel.update({ phase });
	};

	return {
		setThinking: (level: unknown) => {
			const lvl = typeof level === "string" ? level : String(level);
			ctx.applyThinkingLevel(lvl);
			setStatusPhase("ready");
		},
		setInferenceMode: (mode: unknown) => {
			const m = typeof mode === "string" ? mode : String(mode);
			ctx.setInferenceMode(m);
			setStatusPhase("ready");
		},
		setThinkingMode: (mode: unknown) => {
			const m = typeof mode === "string" ? mode : String(mode);
			ctx.thinkingDisplayMode = m as typeof ctx.thinkingDisplayMode;
			ctx.transcript.setThinkingDisplayMode(
				m as "collapsed" | "summary" | "expanded",
			);
			setStatusPhase("ready");
		},
		cycleThinking: () => {
			ctx.transcript.cycleThinkingDisplayMode();
			setStatusPhase("ready");
		},
		clear: () => {
			ctx.transcript.clear();
			setStatusPhase("ready");
		},
		askPreview: () => {
			ctx.choicePopupPreview = true;
			ctx.choicePopup.setQuestionId("");
			ctx.choicePopup.setQuestions([
				{
					id: "approach",
					header: "Approach",
					question: "How should we approach the next implementation?",
					choices: [
						{
							value: "focused",
							label: "Focused fix",
							description:
								"Make the smallest safe change and keep the current structure.",
						},
						{
							value: "balanced",
							label: "Balanced refactor",
							description:
								"Improve the design while keeping the scope practical.",
						},
						{
							value: "redesign",
							label: "Full redesign",
							description:
								"Rework the experience without preserving the current layout.",
						},
					],
				},
				{
					id: "validation",
					header: "Validation",
					question: "How much validation should we run?",
					choices: [
						{
							value: "focused",
							label: "Focused tests",
							description: "Run the tests closest to the changed behavior.",
						},
						{
							value: "full",
							label: "Full suite",
							description: "Run all repository checks before handing off.",
						},
					],
				},
			]);
			ctx.choicePopup.show();
			const overlay = ctx.tui.showOverlay(ctx.choicePopup, {
				anchor: "aboveInput",
				align: "left",
				maxHeight: 22,
			});
			overlay.focus();
			ctx.tui.requestRender();
		},
		version: () => "Logician 0.2.0 (TypeScript runtime)",
		eoh: (raw: unknown) => ctx.bridge.eohCommand(String(raw ?? "")),
		settings: (raw: unknown) => {
			const args = String(raw ?? "").trim();
			if (!args) {
				void ctx.openSettingsSelector();
				return "";
			}
			const [key, value = ""] = args.split(/\s+/, 2);
			const on = value.toLowerCase() === "on";
			switch (key.toLowerCase()) {
				case "thinking":
					if (!value) return "Usage: /settings thinking <level>";
					ctx.applyThinkingLevel(value);
					return `Thinking level: ${value}`;
				case "model":
					if (!value) return "Usage: /settings model <name>";
					ctx.bridge.models.select(value);
					saveConfigField("model", value);
					return `Model: ${value}`;
				case "model-cycle":
				case "model_cycle":
					return `Model: ${ctx.bridge.models.cycle() ?? "unchanged"}`;
				case "temp": {
					const number = Number(value);
					if (!Number.isFinite(number) || number < 0 || number > 2)
						return "Temperature must be between 0 and 2.";
					ctx.bridge.updateSettings({ temperature: number });
					saveConfigField("temperature", number);
					return `Temperature: ${number}`;
				}
				case "max-tokens":
				case "max_tokens": {
					const number = Number.parseInt(value, 10);
					if (!Number.isFinite(number) || number < 1)
						return "Max tokens must be a positive integer.";
					ctx.bridge.updateSettings({ maxTokens: number });
					saveConfigField("maxTokens", number);
					return `Max tokens: ${number}`;
				}
				case "max-iterations":
				case "max_iterations": {
					const number = Number.parseInt(value, 10);
					if (!Number.isFinite(number) || number < 1)
						return "Max iterations must be a positive integer.";
					ctx.bridge.updateSettings({ maxIterations: number });
					saveConfigField("maxIterations", number);
					return `Max iterations: ${number}`;
				}
				case "permissions":
					if (!value) return "Usage: /settings permissions <mode>";
					ctx.bridge.setPermissionMode(
						value as "acceptAll" | "acceptEdits" | "ask" | "plan",
					);
					saveConfigField("permissionMode", value);
					return `Permission mode: ${value}`;
				case "guards":
					if (!["on", "off", "auto"].includes(value.toLowerCase()))
						return "Usage: /settings guards <auto|on|off>";
					ctx.bridge.updateSettings({
						guardMode: value.toLowerCase() as "auto" | "on" | "off",
					});
					saveConfigField(
						"guardsEnabled",
						value.toLowerCase() === "auto" ? undefined : on,
					);
					return `Guards: ${value.toLowerCase()}`;
				case "compaction":
					ctx.bridge.updateSettings({ proactiveCompactionEnabled: on });
					saveConfigNestedField("compaction", "enabled", on);
					return `Compaction: ${on ? "on" : "off"}`;
				case "diagnostics":
				case "post-edit-diagnostics":
					ctx.bridge.updateSettings({ postEditDiagnostics: on });
					saveConfigField("postEditDiagnostics", on);
					return `Post-edit diagnostics: ${on ? "on" : "off"}`;
				case "inference-mode":
				case "inference_mode": {
					const modes = [
						"auto",
						"none",
						"thinking-general",
						"thinking-coding",
						"instruct-general",
						"instruct-reasoning",
						"instruct-coding",
						"deterministic",
						"creative",
						"analytical",
					];
					if (!value) {
						return `Usage: /settings inference-mode <mode>\n\nValid: ${modes.join(", ")}`;
					}
					if (!modes.includes(value.toLowerCase())) {
						return `Invalid mode "${value}". Valid: ${modes.join(", ")}`;
					}
					ctx.setInferenceMode(value);
					return `Inference mode: ${value}`;
				}
				case "execution-policy":
				case "execution_policy": {
					const normalized = value === "auto" ? "autonomous" : value;
					const valid: Array<"autonomous" | "minimal"> = [
						"autonomous",
						"minimal",
					];
					if (!value) {
						return `Usage: /settings execution-policy <mode>\n\nValid: ${valid.join(", ")}`;
					}
					if (!valid.includes(normalized as (typeof valid)[number])) {
						return `Invalid policy "${value}". Valid: ${valid.join(", ")}`;
					}
					ctx.setExecutionProfile(normalized as "autonomous" | "minimal");
					return `Execution mode: ${normalized === "autonomous" ? "auto" : "minimal"}`;
				}
				default:
					return `Unknown setting "${key}". Use /settings to list available settings.`;
			}
		},
		getContext: () => {
			return ctx.bridge.getContext();
		},
		sessions: async (raw: unknown) => {
			const args =
				typeof raw === "string" ? raw.trim() : String(raw ?? "").trim();
			if (args.toLowerCase() !== "clean") {
				ctx.openSessionManager();
				return;
			}
			const currentSessionId =
				ctx.currentSessionId || ctx.sessionService.getCurrentSessionId();
			const olderSessions = ctx.sessionService
				.listSessions()
				.filter(session => session.id !== currentSessionId);
			let removedSessions = 0;
			for (const session of olderSessions) {
				if (ctx.sessionService.deleteSession(session.id)) removedSessions++;
			}
			if (ctx.bridge.getConfig()?.memoriamEnabled) {
				try {
					await ctx.bridge.memoriamClearSessions(currentSessionId || undefined);
				} catch {
					// Memoriam session cleanup is best-effort.
				}
			}
			if (!removedSessions) {
				return "No older sessions to remove from this folder.";
			}
			return `Removed ${removedSessions} older sessions from this folder.`;
		},
		newSession: () => {
			ctx._autoSaveTurn();
			ctx.currentSessionId = ctx.sessionService.createSession("New Session");
			ctx.bridge.useConversationSession(
				ctx.currentSessionId,
				ctx.sessionService.getRawSession(ctx.currentSessionId) ?? undefined,
			);
			ctx.transcript.clear();
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.statusPanel.update({ sessionTitle: "New Session" });
			setStatusPhase("ready");
		},
		saveSession: () => {
			ctx._autoSaveTurn();
			ctx.statusPanel.update({
				phase: "saved",
			});
			ctx.transcript.addSystemMessage("Session saved.");
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
		},
		renameSession: (title: unknown) => {
			if (!ctx.currentSessionId) return;
			const newTitle = typeof title === "string" ? title : String(title || "");
			if (!newTitle.trim()) return;
			ctx.sessionService.renameSession(ctx.currentSessionId, newTitle.trim());
			void ctx.bridge.renameConversationSession(
				ctx.currentSessionId,
				newTitle.trim(),
			);
			ctx.statusPanel.update({ sessionTitle: newTitle.trim() });
			setStatusPhase("ready");
		},
		setPermissionMode: (mode: unknown) => {
			const selected = String(mode) as
				| "acceptAll"
				| "acceptEdits"
				| "ask"
				| "plan";
			if (selected === "plan") {
				ctx.setPlanMode(true);
			} else {
				ctx.normalPermissionMode = selected;
				if (ctx.workflowMode === "act") ctx.bridge.setPermissionMode(selected);
			}
			setStatusPhase("ready");
		},
		getPermissionMode: () => ctx.bridge.getPermissionMode(),
		togglePlanMode: () => {
			const next = ctx.togglePlanMode();
			return next === "plan"
				? "Plan mode ON — plan read-only, then pause for approval before execution."
				: "Act mode ON — execute directly without a required planning phase.";
		},
		rewind: () => {
			const restored = ctx.bridge.rewind();
			if (restored === null) return "Nothing to rewind.";
			return (
				`Rewound to checkpoint: ${restored.messages} message(s) in ` +
				`context, ${restored.filesRestored} file(s) restored ` +
				"(bash mutations are not captured)."
			);
		},
		fork: () => {
			const id = ctx.bridge.fork();
			if (!id) return "Fork unavailable.";
			setStatusPhase("ready");
			return `Forked conversation: branch ${id}. Use /branch-summary to merge or /discard-branch to abandon.`;
		},
		branchSummary: async () => {
			const summary = await ctx.bridge.branchSummary();
			setStatusPhase("ready");
			if (summary === null)
				return "No active branch (or nothing to summarize).";
			return `Branch merged. Summary: ${summary}`;
		},
		discardBranch: () => {
			const discarded = ctx.bridge.discardBranch();
			setStatusPhase("ready");
			return discarded
				? "Branch discarded — conversation restored to fork point."
				: "No active branch to discard.";
		},
		toggleRtkProxy: () => {
			const current = ctx.bridge.getConfig()?.rtkProxyEnabled ?? false;
			const next = !current;
			ctx.bridge.updateSettings({ rtkProxyEnabled: next });
			saveConfigField("rtkProxyEnabled", next);
			ctx.statusPanel.update({ rtkProxyEnabled: next });
			return next;
		},
		toggleLegroom: () => {
			const current = ctx.bridge.getConfig()?.legroomEnabled ?? false;
			const next = !current;
			ctx.bridge.updateSettings({ legroomEnabled: next });
			saveConfigNestedField("legroom", "mode", next ? "sdk" : "off");
			ctx.statusPanel.update({ legroomEnabled: next });
			return next;
		},
		toggleMemoriam: () => {
			const current = ctx.bridge.getConfig()?.memoriamEnabled ?? false;
			const next = !current;
			ctx.bridge.updateSettings({ memoriamEnabled: next });
			saveConfigNestedField("memoriam", "mode", next ? "sdk" : "off");
			ctx.statusPanel.update({ memoriamEnabled: next });
			return next;
		},
		openModelSelector: () => {
			ctx.openModelSelector();
		},
		openQueueManager: () => {
			ctx.openQueueManager();
		},
		openSessionManager: () => {
			ctx.openSessionManager();
		},
		cycleSandboxMode: () => {
			const mode = ctx.bridge.cycleSandboxMode();
			ctx.statusPanel.update({ sandboxMode: mode });
			return `Sandbox mode: ${mode}`;
		},
		cycleExecutionProfile: () => {
			const next = ctx.cycleExecutionProfile();
			return `Execution mode: ${next === "autonomous" ? "auto" : "minimal"}`;
		},
		cycleInferenceMode: () => {
			ctx.cycleInferenceMode();
			return `Inference mode: ${ctx.inferenceMode}`;
		},
		sendSpawnPrompt: (prompt: unknown) => {
			const text = typeof prompt === "string" ? prompt : String(prompt);
			ctx.bridge.sendMessage(text);
		},
		spawnAgentDirectly: (args: unknown) => {
			const raw = typeof args === "string" ? args : String(args ?? "");
			const task = raw || "Investigate the codebase and report findings";
			ctx.bridge.spawnAgentDirectly(task);
		},
		memory: async (raw: unknown) => {
			if (!ctx.bridge.getConfig()?.memoriamEnabled)
				return 'Memoriam is not enabled. Set "memoriam": { "mode": "sdk" } in settings.';
			const args = typeof raw === "string" ? raw : String(raw ?? "");
			const trimmed = args.trim();
			const sessionId =
				ctx.currentSessionId || ctx.sessionService.getCurrentSessionId();
			const notSupported = (feature: string) =>
				`${feature} is not supported on the memoriam-py backend yet.`;
			if (!trimmed) {
				try {
					const stats = (await ctx.bridge.getMemoriamStats()) as Record<
						string,
						unknown
					>;
					const lines = Object.entries(stats).map(
						([k, v]) => `${k}: ${String(v)}`,
					);
					return [
						"Memoriam: on",
						...lines,
						"",
						"Subcommands: list [type] [n] | search <q> [n] | obs <q> [n] | forget <id> | consolidate [sid] | context <sid> [budget] | clean",
					].join("\n");
				} catch (error) {
					return `Memoriam stats unavailable: ${(error as Error).message}`;
				}
			}
			const [rawSub, ...rest] = trimmed.split(/\s+/);
			const sub = rawSub.toLowerCase();
			const subArgs = rest.join(" ");
			try {
				switch (sub) {
					case "list": {
						const parts = subArgs.split(/\s+/).filter(Boolean);
						const type =
							parts[0] && !/^\d+$/.test(parts[0]) ? parts[0] : undefined;
						const limitRaw = type ? parts[1] : parts[0];
						const limit = limitRaw ? Math.min(Number(limitRaw), 100) : 20;
						const memories = (await ctx.bridge.memoriamListMemories(
							type ? { type, limit } : { limit },
						)) as Array<Record<string, unknown>>;
						if (!memories.length) return "No memories found.";
						const items = memories
							.slice(0, limit)
							.map(
								m =>
									`[${m.strength ?? "?"}/10] ${m.type} | ${String(m.createdAt ?? "").slice(0, 10)}\n${String(m.content ?? "").slice(0, 200)}`,
							);
						return `Showing ${items.length} of ${memories.length} memories:\n\n${items.join("\n\n---\n\n")}`;
					}
					case "search": {
						const parts = subArgs.split(/\s+/);
						const query = parts[0] || "";
						const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 10;
						if (!query) return "Usage: /memory search <query> [limit]";
						const result = await ctx.bridge.memoriamRecall(
							{ search: query, limit },
							"markdown",
						);
						return result || `No memories found matching "${query}"`;
					}
					case "obs":
					case "observations": {
						const parts = subArgs.split(/\s+/);
						const query = parts[0] || "";
						const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 20;
						if (!query) return "Usage: /memory obs <query> [limit]";
						const results = (await ctx.bridge.memoriamSearchObservations(
							query,
							limit,
						)) as Array<Record<string, unknown>>;
						if (!results.length)
							return `No observations found matching "${query}"`;
						const items = results
							.slice(0, limit)
							.map(
								(r, index) =>
									`#${index + 1} · ${String(r.id ?? "").slice(0, 12)}… · ${r.type ?? ""}\n${r.title ?? ""}\n${String(r.content ?? r.narrative ?? "").slice(0, 300)}`,
							);
						return `Found ${items.length} observations:\n\n${items.join("\n\n---\n\n")}`;
					}
					case "forget": {
						if (!subArgs) return "Usage: /memory forget <id>";
						const deleted = await ctx.bridge.memoriamRemoveMemory(subArgs);
						return deleted
							? `Memory ${subArgs} deleted.`
							: `Memory ${subArgs} not found.`;
					}
					case "clean": {
						const removed = await ctx.bridge.memoriamClearObservations();
						return removed
							? `Removed ${removed} observations.`
							: "No observations to remove.";
					}
					case "consolidate": {
						const target = subArgs.trim() || sessionId;
						if (!target) return "No active memory session for this folder.";
						const memories = (await ctx.bridge.memoriamConsolidate(
							target,
						)) as Array<Record<string, unknown>>;
						if (!memories.length)
							return `No unconsolidated high-signal observations for session ${target.slice(0, 12)}.`;
						return `Consolidated ${memories.length} memories:\n${memories
							.map(m => `[${m.strength ?? "?"}/10] ${m.title ?? ""}`)
							.join("\n")}`;
					}
					case "context": {
						const parts = subArgs.split(/\s+/).filter(Boolean);
						const target = parts[0] || sessionId;
						if (!target) return "Usage: /memory context <session-id> [budget]";
						const budget = parts[1] ? Number(parts[1]) : 4000;
						const context = await ctx.bridge.memoriamGetContext(
							target,
							"recent",
							budget,
						);
						return context || `No context for session ${target}.`;
					}
					case "ws":
						return notSupported("Workspace switching");
					case "stats":
						return notSupported("Per-workspace stats");
					case "tiers":
					case "tier":
					case "auto-tier":
						return notSupported("Working-memory tiers");
					case "retention":
					case "scores":
						return notSupported("Retention scoring");
					default:
						return `Unknown memory subcommand: ${sub}.\n\nUsage:\n  /memory                         Show memoriam worker stats\n  /memory list [type] [limit]     List memories\n  /memory search <query> [n]      Search memories\n  /memory obs <query> [n]         Search observations\n  /memory forget <id>             Delete memory\n  /memory clean                   Remove all observations\n  /memory consolidate [sid]       Consolidate a session\n  /memory context <sid> [budget]  Get context for a session`;
				}
			} catch (error) {
				return `Memoriam error: ${(error as Error).message}`;
			}
		},
		obs: async (raw: unknown) => {
			if (!ctx.bridge.getConfig()?.memoriamEnabled)
				return 'Memoriam is not enabled. Set "memoriam": { "mode": "sdk" } in settings.';
			const args = typeof raw === "string" ? raw : String(raw ?? "");
			const trimmed = args.trim();
			const sessionId =
				ctx.currentSessionId || ctx.sessionService.getCurrentSessionId();
			const notSupported = (feature: string) =>
				`${feature} is not supported on the memoriam-py backend yet.`;
			if (!trimmed) {
				return [
					"Observations (memoriam-py)",
					"",
					"Subcommands: list [n] | search <q> [n] | clean",
				].join("\n");
			}
			const [rawSub, ...rest] = trimmed.split(/\s+/);
			const sub = rawSub.toLowerCase();
			const subArgs = rest.join(" ");
			try {
				switch (sub) {
					case "list": {
						if (!sessionId) return "No active session.";
						const parts = subArgs ? subArgs.split(/\s+/) : [];
						const limitRaw = parts.find(p => /^\d+$/.test(p));
						const limit = limitRaw
							? Math.min(Math.max(Number(limitRaw), 1), 100)
							: 50;
						const allObs = (await ctx.bridge.memoriamListObservations(
							sessionId,
							limit,
						)) as MemoriamObservation[];
						if (!allObs.length) return "No observations found.";
						return `Observations for session ${sessionId.slice(0, 12)} (${allObs.length} recent):\n${allObs
							.map(compactObservationLine)
							.join("\n")}`;
					}
					case "search":
					case "find": {
						const parts = subArgs.split(/\s+/);
						const query = parts[0] || "";
						const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 30;
						if (!query) return "Usage: /obs search <query> [limit]";
						const results = (await ctx.bridge.memoriamSearchObservations(
							query,
							limit,
						)) as Array<Record<string, unknown>>;
						if (!results.length)
							return `No observations found matching "${query}"`;
						const items = results
							.slice(0, limit)
							.map(
								(r, index) =>
									`#${index + 1} · ${String(r.id ?? "").slice(0, 12)}… · ${r.type ?? ""}\n${r.title ?? ""}\n${String(r.content ?? r.narrative ?? "").slice(0, 300)}`,
							);
						return `Found ${items.length} observations:\n\n${items.join("\n\n---\n\n")}`;
					}
					case "clean": {
						const removed = await ctx.bridge.memoriamClearObservations();
						return removed
							? `Removed ${removed} observations.`
							: "No observations to remove.";
					}
					case "stats":
						return notSupported("Observation statistics");
					case "sessions":
						return notSupported("Per-session observation listing");
					case "by-session":
					case "session":
						return notSupported("Cross-workspace session lookup");
					default:
						return `Unknown obs subcommand: ${sub}.\n\nUsage:\n  /obs                           Show observation summary\n  /obs list [n]                  List observations for the current session\n  /obs search <query> [n]        Search observations\n  /obs clean                     Remove all observations`;
				}
			} catch (error) {
				return `Memoriam error: ${(error as Error).message}`;
			}
		},
		notifications: () => {
			const history = ctx.notifications.history();
			if (history.length === 0) return "No notifications yet.";
			const icons: Record<string, string> = {
				info: "●",
				success: "✓",
				warning: "⚠",
				error: "×",
			};
			return history
				.map(n => `${icons[n.level] ?? "●"} ${n.message}`)
				.join("\n");
		},
	};
}
