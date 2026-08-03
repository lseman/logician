// ── Local slash-command registry ───────────────────────────────────────────

import { saveConfigField } from "@logician/coding-agent/configuration";
import type { SlashCommandsCtx } from "./context.ts";
import type { CompressedObservation, MemoryStore, ObservationType } from "@logician/memory";
import { theme } from "../../rendering/transcript/semantic-markup.ts";

function observationLabel(observation: CompressedObservation, index?: number): string {
	const ordinal = index === undefined ? "" : `#${index + 1} · `;
	const shortId = observation.id.length > 12
		? `${observation.id.slice(0, 12)}…`
		: observation.id;
	return `${ordinal}${shortId} · importance ${observation.importance}/10`;
}

function compactObservationLine(observation: CompressedObservation, index: number): string {
	const number = theme.fg("memoryCount", `#${index + 1}`);
	const shortId = observation.id.length > 12
		? `${observation.id.slice(0, 12)}…`
		: observation.id;
	const id = theme.fg("memoryId", shortId);
	const title = observation.title || observation.narrative?.slice(0, 100) || "No title";
	return `${number} · ${id} · importance ${observation.importance}/10 · ${observation.type} · ${observation.timestamp.slice(0, 19)} · ${title.replace(/\s+/g, " ")}`;
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
					ctx.bridge.setModel(value);
					return `Model: ${value}`;
				case "model-cycle":
				case "model_cycle":
					return `Model: ${ctx.bridge.cycleModel() ?? "unchanged"}`;
				case "temp": {
					const number = Number(value);
					if (!Number.isFinite(number) || number < 0 || number > 2)
						return "Temperature must be between 0 and 2.";
					ctx.bridge.setTemperature(number);
					return `Temperature: ${number}`;
				}
				case "max-tokens":
				case "max_tokens": {
					const number = Number.parseInt(value, 10);
					if (!Number.isFinite(number) || number < 1)
						return "Max tokens must be a positive integer.";
					ctx.bridge.setMaxTokens(number);
					return `Max tokens: ${number}`;
				}
				case "max-iterations":
				case "max_iterations": {
					const number = Number.parseInt(value, 10);
					if (!Number.isFinite(number) || number < 1)
						return "Max iterations must be a positive integer.";
					ctx.bridge.setMaxIterations(number);
					return `Max iterations: ${number}`;
				}
				case "permissions":
					if (!value) return "Usage: /settings permissions <mode>";
					ctx.bridge.setPermissionMode(
						value as "acceptAll" | "acceptEdits" | "ask" | "plan",
					);
					return `Permission mode: ${value}`;
				case "guards":
					ctx.bridge.setRuntimeToggle("guardsEnabled", on);
					return `Guards: ${on ? "on" : "off"}`;
				case "compaction":
					ctx.bridge.setRuntimeToggle("proactiveCompactionEnabled", on);
					return `Compaction: ${on ? "on" : "off"}`;
				case "diagnostics":
				case "post-edit-diagnostics":
					ctx.bridge.setRuntimeToggle("postEditDiagnostics", on);
					saveConfigField("postEditDiagnostics", on);
					return `Post-edit diagnostics: ${on ? "on" : "off"}`;
				case "inference-mode":
				case "inference_mode": {
					const modes = [
						"auto",
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
					const valid: Array<"autonomous" | "minimal"> = [
						"autonomous",
						"minimal",
					];
					if (!value) {
						return `Usage: /settings execution-policy <mode>\n\nValid: ${valid.join(", ")}`;
					}
					if (
						!valid.includes(value.toLowerCase() as (typeof valid)[number])
					) {
						return `Invalid policy "${value}". Valid: ${valid.join(", ")}`;
					}
					ctx.setExecutionProfile(value as "autonomous" | "minimal");
					return `Execution policy: ${value}`;
				}
				default:
					return `Unknown setting "${key}". Use /settings to list available settings.`;
			}
		},
		getContext: () => {
			return ctx.bridge.getContext();
		},
		sessions: (raw: unknown) => {
			const args = typeof raw === "string" ? raw.trim() : String(raw ?? "").trim();
			if (args.toLowerCase() !== "clean") {
				ctx.openSessionManager();
				return;
			}
			const currentSessionId = ctx.currentSessionId || ctx.sessionStore.getCurrentSessionId();
			const olderSessions = ctx.sessionStore.listSessions()
				.filter((session) => session.id !== currentSessionId);
			let removedSessions = 0;
			for (const session of olderSessions) {
				if (ctx.sessionStore.deleteSession(session.id)) removedSessions++;
			}
			const memoryStore = ctx.bridge.getMemoryStore();
			const memoryResult = memoryStore?.clearSessions(currentSessionId || undefined) || { sessions: 0, observations: 0 };
			if (!removedSessions && !memoryResult.sessions) {
				return "No older sessions to remove from this folder.";
			}
			return `Removed ${Math.max(removedSessions, memoryResult.sessions)} older sessions and ${memoryResult.observations} associated observations from this folder.`;
		},
		newSession: () => {
			ctx._autoSaveTurn();
			ctx.currentSessionId = ctx.sessionStore.createSession({
				title: "New Session",
			});
			ctx.bridge.useConversationSession(ctx.currentSessionId);
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
			const newTitle =
				typeof title === "string" ? title : String(title || "");
			if (!newTitle.trim()) return;
			ctx.sessionStore.renameSession(ctx.currentSessionId, newTitle.trim());
			ctx.bridge.renameConversationSession(ctx.currentSessionId, newTitle.trim());
			ctx.statusPanel.update({ sessionTitle: newTitle.trim() });
			setStatusPhase("ready");
		},
		setPermissionMode: (mode: unknown) => {
			ctx.bridge.setPermissionMode(
				String(mode) as "acceptAll" | "acceptEdits" | "ask" | "plan",
			);
			setStatusPhase("ready");
		},
		getPermissionMode: () => ctx.bridge.getPermissionMode(),
		togglePlanMode: () => {
			const next =
				ctx.bridge.getPermissionMode() === "plan" ? "acceptAll" : "plan";
			ctx.bridge.setPermissionMode(next);
			ctx.statusPanel.update({ permissionMode: next });
			return next === "plan"
				? "Plan mode ON — only read-only tools; the agent should present a plan."
				: "Plan mode OFF — permission mode back to acceptAll.";
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
			ctx.bridge.setRuntimeToggle("rtkProxyEnabled", next);
			saveConfigField("rtkProxyEnabled", next);
			ctx.statusPanel.update({ rtkProxyEnabled: next });
			return next;
		},
		openModelSelector: () => {
			ctx.openModelSelector();
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
			return `Execution policy: ${next}`;
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
		memory: (raw: unknown) => {
			const store = ctx.bridge.getMemoryStore();
			if (!store) return "Memory is not enabled. Set \"memory\": true in settings.";
			const args = typeof raw === "string" ? raw : String(raw ?? "");
			const trimmed = args.trim();
			const workspace = store.getCurrentWorkspace();
			if (!trimmed) {
				const stats = ctx.bridge.getMemoryStats();
				return [
					`Memory: ${stats.memoryEnabled ? "on" : "off"}`,
					`Workspace: ${workspace || "(unset)"}`,
					`Memories: ${stats.memoryCount}`,
					`Sessions: ${stats.sessionCount}`,
					`Observations: ${stats.observationCount}`,
					stats.viewerPort ? `Viewer: :${stats.viewerPort}` : undefined,
					"",
					"Subcommands: list [type] | search <q> | obs <q> | stats | ws [name] | tiers | auto-tier | forget <id> | clean | consolidate [all|sid] | context <sid>",
				].filter(Boolean).join("\n");
			}
			const [rawSub, ...rest] = trimmed.split(/\s+/);
			const sub = rawSub.toLowerCase();
			const subArgs = rest.join(" ");
			switch (sub) {
				case "list": {
					const parts = subArgs.split(/\s+/);
					const type = parts[0] || undefined;
					const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 20;
					const memories = store.list({ type: (type || undefined) as any, limit });
					if (!memories.length) return "No memories found.";
					const items = memories.slice(0, limit).map(
						(m) => `[${m.strength}/10] ${m.type} | ${m.createdAt.slice(0, 10)} [${m.workspace || "global"}]\n${m.content.slice(0, 200)}`,
					);
					return `Showing ${items.length} of ${memories.length} memories (workspace: ${workspace || "(default)"}):\n\n${items.join("\n\n---\n\n")}`;
				}
				case "ws": {
					if (!subArgs) {
						return `Current workspace: ${workspace || "(none)"}\n\nUsage: /memory ws <workspace-name>`;
					}
					store.setCurrentWorkspace(subArgs);
					return `Workspace set to: ${store.getCurrentWorkspace()}`;
				}
				case "search": {
					const parts = subArgs.split(/\s+/);
					const query = parts[0] || "";
					const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 10;
					if (!query) return "Usage: /memory search <query> [limit]";
					const result = store.recall({ search: query, limit }, { format: "markdown" });
					return result || `No memories found matching "${query}"`;
				}
				case "obs":
				case "observations": {
					const parts = subArgs.split(/\s+/);
					const query = parts[0] || "";
					const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 20;
					if (!query) return "Usage: /memory obs <query> [limit]";
					const results = store.searchObservations(query, limit);
					if (!results.length) return `No observations found matching "${query}"`;
					const items = results.slice(0, limit).map(
						(r, index) => `${observationLabel(r.observation, index)} · ${r.observation.type}\n${r.observation.title}\n${r.observation.narrative.slice(0, 300)}`,
					);
					return `Found ${items.length} observations:\n\n${items.join("\n\n---\n\n")}`;
				}
				case "stats": {
					const stats = ctx.bridge.getMemoryStats();
					const allMemories = store.list({ limit: 1000 });
					const workspaceMemories = store.list({ limit: 1000 });
					const sessions = store.listSessions();
					const workspaceSessions = store.listSessions();
					const tierCounts = { hot: 0, warm: 0, cold: 0, archived: 0 };
					const typeCounts: Record<string, number> = {};
					for (const m of workspaceMemories) {
						tierCounts[store.getWorkingMemoryTier(m.id)] =
							(tierCounts[store.getWorkingMemoryTier(m.id)] || 0) + 1;
						typeCounts[m.type] = (typeCounts[m.type] || 0) + 1;
					}
					return [
						`Workspace: ${workspace || "(none)"}`,
						`Memories: ${workspaceMemories.length}`,
						`Sessions: ${workspaceSessions.length}`,
						`Type breakdown: ${Object.entries(typeCounts).map(([t, c]) => `${t}: ${c}`).join(", ")}`,
						`Working tiers: hot=${tierCounts.hot}, warm=${tierCounts.warm}, cold=${tierCounts.cold}, archived=${tierCounts.archived}`,
					].join("\n");
				}
				case "tiers":
				case "tier": {
					const memories = store.list({ limit: 50 });
					if (!memories.length) return "No memories to tier.";
					const tiers = memories.map((m) => ({
						id: m.id,
						content: m.content.slice(0, 80),
						tier: store.getWorkingMemoryTier(m.id),
					}));
					const lines = tiers.slice(0, 20).map(
						(t) => `[${t.tier}] ${t.id.slice(0, 12)}… ${t.content}`,
					);
					return `Working memory tiers (${tiers.length} total):\n${lines.join("\n")}`;
				}
				case "auto-tier": {
					const result = store.autoTierMemories();
					const summary = Object.entries(result)
						.map(([tier, ids]) => `  ${tier}: ${ids.length}`)
						.join("\n");
					return `Auto-tier complete:\n${summary}`;
				}
				case "forget": {
					if (!subArgs) return "Usage: /memory forget <id>";
					const deleted = store.remove(subArgs);
					return deleted ? `Memory ${subArgs} deleted.` : `Memory ${subArgs} not found.`;
				}
				case "clean": {
					const removed = store.clearMemories();
					return removed
						? `Removed ${removed} memories from ${workspace}.`
						: `No memories to remove from ${workspace}.`;
				}
				case "consolidate": {
					const target = subArgs.trim();
					const sessionIds = /^(?:all|folder)$/i.test(target)
						? store.listSessions().map((session) => session.id)
						: [target || store.getCurrentSessionId()].filter((id): id is string => Boolean(id));
					if (!sessionIds.length) return "No active memory session for this folder.";
					const memories = sessionIds.flatMap((sessionId) => store.consolidate(sessionId));
					const scope = sessionIds.length > 1 || /^(?:all|folder)$/i.test(target)
						? `folder ${workspace}`
						: `current session ${sessionIds[0]!.slice(0, 12)}`;
					if (!memories.length) return `No unconsolidated high-signal observations in the ${scope}.`;
					return `Consolidated ${memories.length} memories from the ${scope}:\n${memories.map((m) => `[${m.strength}/10] ${m.title}`).join("\n")}`;
				}
				case "context": {
					if (!subArgs) return "Usage: /memory context <session-id>";
					const ctxParts = subArgs.split(/\s+/);
					const sessionId = ctxParts[0];
					const budget = ctxParts[1] ? Number(ctxParts[1]) : 4000;
					const context = store.getContext(sessionId, budget);
					return context || `No context for session ${sessionId}.`;
				}
				case "retention":
				case "scores": {
					const scores = store.listByRetentionScore(undefined, 20);
					if (!scores.length) return "No memories to score.";
					const lines = scores.map(
						(s) => `[${s.score.toFixed(2)}] ${s.id.slice(0, 12)}… type:${s.type} strength:${s.strength}`,
					);
					return `Retention scores (top ${scores.length}):\n${lines.join("\n")}`;
				}
				default:
					return `Unknown memory subcommand: ${sub}.\n\nUsage:\n  /memory                         Show memory stats\n  /memory list [type] [limit]     List memories\n  /memory search <query> [n]      Search memories\n  /memory obs <query> [n]         Search observations\n  /memory stats                   Detailed stats\n  /memory tiers                   Show working memory tiers\n  /memory auto-tier               Auto-classify tiers\n  /memory forget <id>             Delete memory\n  /memory clean                   Remove memories in this folder\n  /memory consolidate             Consolidate the current session\n  /memory consolidate all         Consolidate sessions in this folder\n  /memory context <sid> [budget]  Get context for session\n  /memory retention               Show retention scores`;
			}
		},
		obs: (raw: unknown) => {
			const store = ctx.bridge.getMemoryStore();
			if (!store) return "Memory is not enabled. Set \"memory\": true in settings.";
			const args = typeof raw === "string" ? raw : String(raw ?? "");
			const trimmed = args.trim();
			if (!trimmed) {
				// Default: show recent observations summary
				const sessions = store.listSessions();
				const totalObs = sessions.reduce((sum, s) => sum + (s.observationCount || 0), 0);
				const workspace = store.getCurrentWorkspace();
				return [
					`Observations: ${totalObs} total`,
					`Workspace: ${workspace || "(none)"}`,
					`Sessions: ${sessions.length}`,
					"",
					"Subcommands: list [type] [n] | search <q> [n] | stats | sessions | by-session <sid> [n] | clean",
				].filter(Boolean).join("\n");
			}
			const [rawSub, ...rest] = trimmed.split(/\s+/);
			const sub = rawSub.toLowerCase();
			const subArgs = rest.join(" ");
			switch (sub) {
				case "list": {
					const parts = subArgs ? subArgs.split(/\s+/) : [];
					const firstIsLimit = parts[0] !== undefined && /^\d+$/.test(parts[0]);
					const type = firstIsLimit ? undefined : parts[0] || undefined;
					const rawLimit = firstIsLimit ? parts[0] : parts[1];
					const limit = rawLimit ? Math.min(Math.max(Number(rawLimit), 1), 100) : 50;
					const allObs = store.listRecentObservations(limit, type as ObservationType | undefined);
					if (!allObs.length) return "No observations found.";
					const items = allObs.map(compactObservationLine);
					return `Observations for ${store.getCurrentWorkspace()} (${items.length} recent, type: ${type || "all"}):\n${items.join("\n")}`;
				}
				case "search":
				case "find": {
					const parts = subArgs.split(/\s+/);
					const query = parts[0] || "";
					const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 30;
					if (!query) return "Usage: /obs search <query> [limit]";
					const results = store.searchObservations(query, limit);
					if (!results.length) return `No observations found matching "${query}"`;
					const items = results.slice(0, limit).map(
						(r, index) => `${observationLabel(r.observation, index)} · ${r.observation.type}\n${r.observation.title}\n${r.observation.narrative.slice(0, 300)}`,
					);
					return `Found ${items.length} observations:\n\n${items.join("\n\n---\n\n")}`;
				}
				case "stats": {
					const sessions = store.listSessions();
					const typeCounts: Record<string, number> = {};
					const workspaceCounts: Record<string, number> = {};
					let totalObs = 0;
					for (const session of sessions) {
						const obs = store.listObservations(session.id, 1000);
						totalObs += obs.length;
						for (const o of obs) {
							typeCounts[o.type] = (typeCounts[o.type] || 0) + 1;
							const ws = o.workspace || session.workspace || "(none)";
							workspaceCounts[ws] = (workspaceCounts[ws] || 0) + 1;
						}
					}
					const typeBreakdown = Object.entries(typeCounts).map(([t, c]) => `${t}: ${c}`).join(", ");
					const wsBreakdown = Object.entries(workspaceCounts).map(([w, c]) => `${w}: ${c}`).join(", ");
					return [
						`Total observations: ${totalObs}`,
						`Sessions: ${sessions.length}`,
						`Type breakdown: ${typeBreakdown || "none"}`,
						`Workspace breakdown: ${wsBreakdown || "none"}`,
					].join("\n");
				}
				case "sessions": {
					const sessions = store.listSessions();
					if (!sessions.length) return "No sessions with observations.";
					const items = sessions.slice(0, 30).map(
						(s) => `  ${s.id.slice(0, 12)} | ${s.observationCount} obs | ${s.workspace || "(none)"} | ${s.startedAt?.slice(0, 19) || ""}`,
					);
					return `Sessions (${sessions.length} total, showing ${items.length}):\n${items.join("\n")}`;
				}
				case "by-session":
				case "session": {
					if (!subArgs) return "Usage: /obs by-session <session-id> [limit]";
					const parts = subArgs.split(/\s+/);
					const sessionId = parts[0];
					const limit = parts[1] ? Math.min(Number(parts[1]), 100) : 50;
					const session = store.getSession(sessionId);
					if (!session || session.workspace !== store.getCurrentWorkspace()) {
						return `Session ${sessionId} not found in ${store.getCurrentWorkspace()}.`;
					}
					const obs = store.listObservations(sessionId, limit);
					if (!obs.length) return `No observations for session ${sessionId.slice(0, 12)}`;
					const items = obs.slice(0, limit).map(
						(o, index) => `${observationLabel(o, index)} · ${o.type} · ${o.timestamp?.slice(0, 19) || ""}\n  ${o.title || o.narrative?.slice(0, 100) || "No title"}`,
					);
					return `Observations for session ${sessionId.slice(0, 12)} (${obs.length} total, showing ${items.length}):\n\n${items.join("\n\n---\n\n")}`;
				}
				case "clean": {
					const workspace = store.getCurrentWorkspace();
					const removed = store.clearObservations();
					return removed
						? `Removed ${removed} observations from ${workspace}.`
						: `No observations to remove from ${workspace}.`;
				}
				default:
					return `Unknown obs subcommand: ${sub}.\n\nUsage:\n  /obs                           Show observation summary\n  /obs list [type] [n]           List observations by type\n  /obs search <query> [n]        Search observations\n  /obs stats                     Observation statistics\n  /obs sessions                  List sessions with observations\n  /obs by-session <sid> [n]      Observations for a specific session\n  /obs clean                     Remove observations in this folder`;
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
				.map((n) => `${icons[n.level] ?? "●"} ${n.message}`)
				.join("\n");
		},
	};
}
