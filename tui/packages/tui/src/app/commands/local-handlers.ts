// ── Local slash-command registry ───────────────────────────────────────────

import { saveConfigField } from "@logician/coding-agent/configuration";
import type { SlashCommandsCtx } from "./context.ts";

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
						"thinking-general",
						"thinking-coding",
						"instruct-general",
						"instruct-reasoning",
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
		sessions: () => {
			ctx.openSessionManager();
		},
		newSession: () => {
			ctx._autoSaveTurn();
			ctx.currentSessionId = ctx.sessionStore.createSession({
				title: "New Session",
			});
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
