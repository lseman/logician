// ── TUI input controller ───────────────────────────────────────────────────

import {
	filterSlashCommands,
	type SlashCommandDef,
} from "@logician/log-runtime/commands";
import { beginPendingTurn } from "../state/turn-state.ts";
import { logInputTrace } from "../terminal/input-protocol.ts";
import { theme } from "../terminal/theme.ts";
import type { LogicianTUI } from "./tui.ts";

/** Ctrl+E encodings emitted by terminals with CSI-u or modifyOtherKeys. */
export function isCtrlE(data: string): boolean {
	return data === "\x05" || data === "\x1b[5;5u" || data === "\x1b[27;5;5~";
}

// ── Overlay dispatcher ─────────────────────────────────────────────────────

/**
 * Dispatch input to a visible overlay. Returns true if consumed.
 * Overlays are checked in priority order; the first visible one handles the input.
 */
function handleOverlayInput(ctx: LogicianTUI, data: string): boolean {
	const check = <T>(
		overlay: { isVisibleOverlay(): boolean; handleInput(data: string): T | null },
		actionHandler: (action: T) => void,
	): boolean => {
		if (overlay.isVisibleOverlay()) {
			const action = overlay.handleInput(data);
			if (action) actionHandler(action);
			ctx.tui.requestRender();
			return true;
		}
		return false;
	};

	// SessionTree has no action handler — it handles input directly.
	if (ctx.sessionTree.isVisibleOverlay()) {
		ctx.sessionTree.handleInput(data);
		ctx.tui.requestRender();
		return true;
	}

	if (check(ctx.pluginManager, a => ctx.handlePluginManagerAction(a))) return true;
	if (check(ctx.mcpManager, a => ctx.handleMcpManagerAction(a))) return true;
	if (check(ctx.autoresearchDashboard, a => ctx.handleAutoresearchDashboardAction(a))) return true;
	if (check(ctx.reasonerSelector, a => ctx.handleReasonerSelectorAction(a))) return true;
	if (check(ctx.queueManager, a => ctx.handleQueueManagerAction(a))) return true;
	if (check(ctx.modelSelector, a => ctx.handleModelSelectorAction(a))) return true;
	if (check(ctx.inferenceModeSelector, a => ctx.handleInferenceModeSelectorAction(a))) return true;
	if (check(ctx.themeSelector, a => ctx.handleThemeSelectorAction(a))) return true;
	if (check(ctx.settingsSelector, a => ctx.handleSettingsSelectorAction(a))) return true;
	if (check(ctx.thinkingLevelSelector, a => ctx.handleThinkingLevelSelectorAction(a))) return true;

	return false;
}

// ── Autocomplete popup handling ──────────────────────────────────────────────

/**
 * Handle input for inline autocomplete popups (@-mention, skill://, slash).
 * Returns true if the popup was visible and the key was consumed.
 */
function handleAutocompleteInput(ctx: LogicianTUI, data: string): boolean {
	// File mention popup: up/down navigation, tab/enter accept, escape dismiss.
	if (ctx.fileMentionPopup.isVisibleOverlay()) {
		if (data === "\x1b[A" || data === "\x1bOA") {
			ctx.fileMentionPopup.moveSelection(-1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\x1b[B" || data === "\x1bOB") {
			ctx.fileMentionPopup.moveSelection(1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\t" || data === "\r" || data === "\n") {
			const file = ctx.fileMentionPopup.currentFile();
			if (file) ctx.inputBar.insertMention(file);
			ctx.fileMentionPopup.hide();
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\x1b") {
			ctx.fileMentionPopup.hide();
			ctx.tui.requestRender();
			return true;
		}
		// Everything else goes to the input bar; onChange re-syncs the popup.
		return true;
	}

	// Skill popup: same pattern as file mention.
	if (ctx.skillPopup.isVisibleOverlay()) {
		if (data === "\x1b[A" || data === "\x1bOA") {
			ctx.skillPopup.moveSelection(-1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\x1b[B" || data === "\x1bOB") {
			ctx.skillPopup.moveSelection(1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\t" || data === "\r" || data === "\n") {
			const name = ctx.skillPopup.currentSkill();
			if (name) ctx.inputBar.insertSkill(name);
			ctx.skillPopup.hide();
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\x1b") {
			ctx.skillPopup.hide();
			ctx.tui.requestRender();
			return true;
		}
		return true;
	}

	// Slash popup: up/down navigation, tab complete, escape dismiss/stop loop,
	// enter accept command and fall through to input bar.
	if (ctx.slashPopup.isVisibleOverlay()) {
		if (data === "\x1b[A" || data === "\x1bOA") {
			ctx.slashPopup.moveSelection(-1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\x1b[B" || data === "\x1bOB") {
			ctx.slashPopup.moveSelection(1);
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\t") {
			const cmd = ctx.slashPopup.currentCommand();
			if (cmd) {
				ctx.inputBar.valueText = `${cmd} `;
				ctx.tui.requestRender();
			}
			return true;
		}
		if (data === "\x1b") {
			ctx.slashPopup.hide();
			ctx.inputBar.handleInput(data);
			if (ctx.loopActive) {
				ctx.loopManager.stop();
				ctx.loopActive = false;
				ctx.transcript.addSystemMessage("Loop stopped (Esc).");
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			}
			ctx.tui.requestRender();
			return true;
		}
		if (data === "\r" || data === "\n") {
			const cmd = ctx.slashPopup.currentCommand();
			if (cmd && ctx.inputBar.valueText.trim() !== cmd) {
				if (/^\/\S+\s+\S+/.test(cmd)) {
					ctx.inputBar.valueText = cmd;
				} else {
					const typedArgs = ctx.inputBar.valueText.replace(/^\/\S*\s*/, "");
					ctx.inputBar.valueText = typedArgs ? `${cmd} ${typedArgs}` : cmd;
				}
			}
			ctx.slashPopup.hide();
			return false; // fall through to input bar submission
		}
		return true; // typing/backspace goes to input bar
	}

	return false;
}

// ── Key bindings ─────────────────────────────────────────────────────────────

/** Handle special key bindings. Returns true if consumed. */
function handleKeyBinding(ctx: LogicianTUI, data: string): boolean {
	// Ctrl+H — thinking level selector
	if (data === "\x08") {
		ctx.openThinkingLevelSelector();
		return true;
	}
	// Ctrl+L — model selector
	if (data === "\x0c") {
		ctx.openModelSelector();
		return true;
	}
	// Ctrl+G — jump to file from working set
	if (data === "\x07") {
		const files = ctx.workSurface.getWorkingSet();
		if (files.length === 0) {
			ctx.notify("Working set is empty.", "info");
			return true;
		}
		if (ctx.inputBar.getActiveMentionQuery() === null) {
			ctx.inputBar.valueText = `${ctx.inputBar.valueText}@`;
		}
		ctx.fileMentionPopup.setFiles(files);
		ctx.fileMentionPopup.setQuery("");
		ctx.fileMentionPopup.show();
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+O — toggle tool expansion
	if (data === "\x0f") {
		const expanded = ctx.transcriptDisplay.toggleToolsExpanded();
		ctx.statusPanel.update({
			phase: expanded ? "tools expanded" : "tools collapsed",
		});
		ctx.tui.requestRender();
		return true;
	}
	// Alt+J/K — tool card navigation; Alt+Enter — toggle focused tool
	if (data === "\x1bj" || data === "\x1bk") {
		const position = ctx.transcriptDisplay.focusTool(
			data === "\x1bj" ? 1 : -1,
		);
		if (position) {
			ctx.notify(`Tool ${position.index}/${position.total}`, "info");
			ctx.tui.requestRender();
		}
		return true;
	}
	if (data === "\x1b\r" || data === "\x1b\n") {
		const expanded = ctx.transcriptDisplay.toggleFocusedTool();
		if (expanded !== null) {
			ctx.notify(expanded ? "Tool expanded" : "Tool collapsed", "info");
			ctx.tui.requestRender();
		}
		return true;
	}
	// Ctrl+Shift+T — cycle thinking display mode
	if (data === "\x14") {
		ctx.transcript.cycleThinkingDisplayMode();
		ctx.transcriptDisplay.setThinkingMode(ctx.transcript.getThinkingDisplayMode());
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+S — session tree
	if (data === "\x13") {
		ctx.openSessionTree();
		return true;
	}
	// Ctrl+Q — queue manager
	if (data === "\x11") {
		ctx.openQueueManager();
		return true;
	}
	// Ctrl+K — cycle sandbox mode
	if (data === "\x0b") {
		const mode = ctx.bridge.cycleSandboxMode();
		ctx.statusPanel.update({ sandboxMode: mode });
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+P — toggle plan mode
	if (data === "\x10") {
		const next = ctx.togglePlanMode();
		ctx.notify(next === "plan" ? "Mode: plan" : "Mode: act", "success");
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+E — steering flush
	if (isCtrlE(data)) {
		if (ctx.inputBar.submit("steer-now")) return true;
		const count = ctx.bridge.flushSteeringNow();
		if (count > 0) {
			ctx.transcript.addSystemMessage(
				`Flushed ${count} steering message${count === 1 ? "" : "s"} to the active turn.`,
			);
		} else {
			ctx.transcript.addSystemMessage("No queued steering messages to flush.");
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+M — inference mode selector
	if (
		data === "\x1bm" ||
		data === "\x1bM" ||
		data === "\x1b[109;5u" ||
		data === "\x1b[109;6u"
	) {
		ctx.openInferenceModeSelector();
		return true;
	}
	// Ctrl+I — cycle execution mode
	if (data === "\x1b[105;5u" || data === "\x1b[105;6u") {
		const next = ctx.cycleExecutionProfile();
		ctx.notify(
			`Execution mode: ${next === "autonomous" ? "auto" : "minimal"}`,
			"success",
		);
		ctx.tui.requestRender();
		return true;
	}
	// Ctrl+A — autoresearch dashboard
	if (data === "\x1b[97;4u" || data === "\x01") {
		ctx.openAutoresearchDashboard();
		return true;
	}

	return false;
}

// ── Choice popup helpers ─────────────────────────────────────────────────────

function handleChoicePopupSubmit(ctx: LogicianTUI): void {
	const qid = ctx.choicePopup.getQuestionId();
	const answers = ctx.choicePopup.getAnswers();

	if (qid === "__plan_approval__") {
		ctx.choicePopup.hide();
		if (ctx.choicePopup.getResponseValue() === "approve") {
			ctx.planPhase = "executing";
			ctx.bridge.setPermissionMode(ctx.normalPermissionMode);
			ctx.transcript.addSystemMessage("Plan approved — executing now.");
			ctx.statusPanel.update({ phase: "streaming" });
			void ctx.bridge
				.sendMessage(
					"The user approved the plan. Execute the approved plan now. Do not create another plan or ask for approval again.",
				)
				.catch(err => ctx.bridge.events.reportError(err));
		} else {
			ctx.planPhase = "idle";
			ctx.bridge.setPermissionMode(ctx.normalPermissionMode);
			ctx.transcript.addSystemMessage("Plan rejected — nothing was executed.");
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}

	if (ctx.choicePopupPreview) {
		ctx.choicePopupPreview = false;
		ctx.transcript.addSystemMessage(`Ask preview: ${JSON.stringify(answers)}`);
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return;
	}

	if (qid && ctx.bridge.respondToQuestion(qid, ctx.choicePopup.getResponseValue())) {
		ctx.transcript.addSystemMessage(
			`Questions answered: ${Object.keys(answers).length}`,
		);
	}
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();
}

function handleChoicePopupDismiss(ctx: LogicianTUI): void {
	if (ctx.choicePopupPreview) {
		ctx.choicePopupPreview = false;
		ctx.tui.requestRender();
		return;
	}

	const qid = ctx.choicePopup.getQuestionId();
	if (qid === "__plan_approval__") {
		ctx.choicePopup.hide();
		ctx.planPhase = "idle";
		ctx.bridge.setPermissionMode(ctx.normalPermissionMode);
		ctx.transcript.addSystemMessage("Plan approval dismissed — nothing was executed.");
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}

	if (qid) {
		ctx.bridge.respondToQuestion(qid, "__dismissed__");
		ctx.transcript.addSystemMessage("Question dismissed.");
	}
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();
}

// ── Permission popup handler ─────────────────────────────────────────────────

function handlePermissionPopupInput(
	ctx: LogicianTUI,
	action: { type: string; choice?: { value: string } } | null,
): void {
	if (!action) return;
	if (action.type === "close") {
		ctx.pendingPermission = null;
		ctx.transcript.addSystemMessage("Permission request dismissed.");
	} else {
		const value = action.choice?.value;
		if (!value) return;
		const decision =
			value === "y" || value === "yes" || value === "allow"
				? "allow"
				: value === "a" || value === "always"
					? "always"
					: "deny";
		ctx.bridge.respondToPermission(
			ctx.pendingPermission?.toolCallId ?? "",
			decision,
		);
		ctx.transcript.addSystemMessage(
			`Permission ${decision}: ${ctx.pendingPermission?.toolName ?? "unknown"}`,
		);
	}
	ctx.pendingPermission = null;
	ctx.permissionPopup.hide();
	ctx.tui.removeOverlay(ctx.permissionPopup);
	ctx.statusPanel.update({ phase: action.type !== "close" ? "streaming" : "ready" });
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
}

// ── Submit handler ───────────────────────────────────────────────────────────

function handleInputSubmit(ctx: LogicianTUI, text: string, intent: string): void {
	// Pending permission answer takes priority.
	if (ctx.pendingPermission) {
		const answer = text.trim().toLowerCase();
		const decision =
			answer === "y" || answer === "yes" || answer === "allow"
				? "allow"
				: answer === "a" || answer === "always"
					? "always"
					: "deny";
		ctx.bridge.respondToPermission(ctx.pendingPermission.toolCallId, decision);
		ctx.transcript.addSystemMessage(`Permission ${decision}: ${ctx.pendingPermission.toolName}`);
		ctx.pendingPermission = null;
		ctx.statusPanel.update({ phase: "streaming" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return;
	}

	ctx.inputBar.pushHistory(text);

	// Bash: `!command` or `!!command` (exclude from context).
	if (text.startsWith("!")) {
		const excludeFromContext = text.startsWith("!!");
		const command = excludeFromContext ? text.slice(2) : text.slice(1);
		if (command.trim()) {
			executeBashCommand(ctx, command.trim(), excludeFromContext);
			return;
		}
	}

	// Python: `$code` or `$$code`.
	if (text.startsWith("$")) {
		const trimmed = text.trimStart();
		const prefixLength = pythonCommandPrefixLength(trimmed);
		if (
			prefixLength > 0 &&
			!looksLikePastedShellPrompt(trimmed.slice(prefixLength).trim())
		) {
			const excludeFromContext = prefixLength === 2;
			const code = trimmed.slice(prefixLength).trim();
			if (code) {
				executePythonCommand(ctx, code, excludeFromContext);
				return;
			}
		}
	}

	// Slash commands.
	if (text.startsWith("/")) {
		handleSlashCommand(ctx, text);
		return;
	}

	// Steering or new turn.
	if (ctx.bridge.isActive()) {
		handleSteering(ctx, text, intent);
		return;
	}

	// Start a new turn.
	startNewTurn(ctx, text);
}

function executeBashCommand(ctx: LogicianTUI, command: string, excludeFromContext: boolean): void {
	ctx.statusPanel.update({ phase: "bash" });
	ctx.statusPanel.startAnimation();
	ctx.tui.renderNow();
	setImmediate(async () => {
		try {
			const output = await ctx.bridge.executeBashCommand(command);
			ctx.transcript.addSystemMessage(
				`!${excludeFromContext ? "!" : ""}${command}: exit ${output.exitCode}\n${output.output}`,
			);
		} catch (err) {
			ctx.transcript.addSystemMessage(
				`!${excludeFromContext ? "!" : ""}${command}: Error: ${err instanceof Error ? err.message : String(err)}`,
			);
		} finally {
			ctx.statusPanel.update({ phase: "ready" });
			ctx.tui.requestRender();
		}
	});
}

function executePythonCommand(ctx: LogicianTUI, code: string, excludeFromContext: boolean): void {
	ctx.statusPanel.update({ phase: "python" });
	ctx.statusPanel.startAnimation();
	ctx.tui.renderNow();
	setImmediate(async () => {
		try {
			const result = await ctx.bridge.executePythonCommand(code);
			ctx.transcript.addSystemMessage(
				`$$${excludeFromContext ? "" : "$"}${code}: ${result.error ? `Error: ${result.error}` : `Output:\n${result.output}`}`,
			);
		} catch (err) {
			ctx.transcript.addSystemMessage(
				`$$${excludeFromContext ? "" : "$"}${code}: Error: ${err instanceof Error ? err.message : String(err)}`,
			);
		} finally {
			ctx.statusPanel.update({ phase: "ready" });
			ctx.tui.requestRender();
		}
	});
}

function handleSlashCommand(ctx: LogicianTUI, text: string): void {
	const parts = text.trim().split(/\s+/);
	const cmdName = parts[0]?.toLowerCase() ?? "";
	const args = parts.slice(1).join(" ");
	const allCmds = ctx.slashPopup.getCommands() as SlashCommandDef[];
	const match = allCmds?.find(
		(c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
	);

	if (match) {
		ctx.slashPopup.submitRaw(text.trim());
		return;
	}

	// Unknown command — skill invocation?
	if (ctx.bridge.invokeSkill(cmdName.slice(1), args)) {
		ctx.transcript.addTurn(text.trim());
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.statusPanel.update({ phase: "streaming" });
		ctx.statusPanel.startAnimation();
		ctx.tui.requestRender();
		return;
	}

	// Unknown command — suggest corrections.
	ctx.transcript.addTurn(text.trim());
	const suggestions = filterSlashCommands(allCmds, cmdName, 3).map(
		command => command.command,
	);
	ctx.transcript.addSystemMessage(
		`Unknown command: ${cmdName}.` +
			(suggestions.length > 0
				? ` Did you mean ${suggestions.join(", ")}?`
				: "") +
			" Use /help to list available commands.",
	);
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();
}

function handleSteering(ctx: LogicianTUI, text: string, intent: string): void {
	const preview = oneLineSteerPreview(text);
	const label =
		intent === "steer-now"
			? `Steering now: ${preview}`
			: `Steering queued: ${preview}`;
	ctx.notify(label, "info");
	try {
		if (intent === "steer-now") {
			ctx.bridge.steerNow(text);
		} else {
			ctx.bridge.steerQueue(text);
		}
	} catch (err) {
		ctx.bridge.events.reportError(err as Error);
	}
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender(false, true);
}

function startNewTurn(ctx: LogicianTUI, text: string): void {
	ctx.transcript.addTurn(text);
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.turnState = beginPendingTurn(ctx.turnState);
	ctx.workSurface.setPhase(ctx.turnState.phase);
	ctx.statusPanel.update({ phase: "thinking" });
	ctx.statusPanel.startAnimation();
	ctx.tui.renderNow();

	const prompt =
		ctx.workflowMode === "plan"
			? `[PLAN MODE]\nFirst investigate using read-only tools and produce a concrete implementation plan. Do not modify files or execute mutating commands. End after presenting the plan and wait for explicit user approval.\n\nUser request:\n${text}`
			: text;

	if (ctx.workflowMode === "plan") {
		ctx.planPhase = "planning";
		ctx.bridge.setPermissionMode("plan");
	}

	setImmediate(() => {
		void ctx.bridge
			.sendMessage(prompt)
			.catch(err => ctx.bridge.events.reportError(err));
	});
}

// ── Change handler ───────────────────────────────────────────────────────────

function handleInputChange(ctx: LogicianTUI, text: string): void {
	const isCommandPrefix = text.startsWith("/");
	if (isCommandPrefix) {
		ctx.slashPopup.setQuery(text);
		if (ctx.slashPopup.hasMatches()) {
			if (!ctx.slashPopup.isVisibleOverlay()) ctx.slashPopup.show();
		} else {
			ctx.slashPopup.hide();
		}
	} else if (ctx.slashPopup.isVisibleOverlay()) {
		ctx.slashPopup.hide();
	}

	const mentionQuery = ctx.inputBar.getActiveMentionQuery();
	if (mentionQuery !== null) {
		void ctx.updateFileMentionPopup(mentionQuery);
	} else if (ctx.fileMentionPopup.isVisibleOverlay()) {
		ctx.fileMentionPopup.hide();
	} else {
		const skillQuery = ctx.inputBar.getActiveSkillQuery();
		if (skillQuery !== null) {
			void ctx.updateSkillPopup(skillQuery);
		} else if (ctx.skillPopup.isVisibleOverlay()) {
			ctx.skillPopup.hide();
		}
	}

	// Update input bar mode color for bash (!) and python ($) prefixes.
	const trimmed = text.trimStart();
	if (trimmed.startsWith("!")) {
		ctx.inputBar.modeColor = theme.fgRaw("bashMode");
	} else if (trimmed.startsWith("$")) {
		const prefixLen = pythonCommandPrefixLength(trimmed);
		if (
			prefixLen > 0 &&
			!looksLikePastedShellPrompt(trimmed.slice(prefixLen).trim())
		) {
			ctx.inputBar.modeColor = theme.fgRaw("pythonMode");
		} else {
			ctx.inputBar.modeColor = null;
		}
	} else {
		ctx.inputBar.modeColor = null;
	}

	ctx.tui.requestRender();
}

// ── Main input handler ───────────────────────────────────────────────────────

export function setupInputHandler(ctx: LogicianTUI): void {
	// Global input listener
	ctx.tui.addInputListener((data: string) => {
		if (data === "\x03" || data === "\x1b") {
			logInputTrace("interrupt-route", data, {
				active: ctx.hasActiveTurn(),
				bridgeActive: ctx.bridge.isActive(),
				phase: ctx.turnState.phase,
			});
		}

		// Raw-mode Ctrl+C: interrupt active work; exit if idle.
		if (data === "\x03") {
			if (ctx.hasActiveTurn()) void ctx.cancelActiveTurn();
			else ctx.requestExit();
			return { consume: true };
		}

		// Escape: interrupt active run; otherwise let overlays/focus routes handle it.
		if (data === "\x1b" && ctx.hasActiveTurn()) {
			void ctx.cancelActiveTurn();
			return { consume: true };
		}

		// 1. Overlay dispatch (plugin, mcp, dashboard, selectors, etc.)
		if (handleOverlayInput(ctx, data)) return { consume: true };

		// 2. Choice popup
		if (ctx.choicePopup.isVisibleOverlay()) {
			const action = ctx.choicePopup.handleInput(data);
			if (action) {
				if (action.type === "submit") handleChoicePopupSubmit(ctx);
				else handleChoicePopupDismiss(ctx);
				ctx.tui.removeOverlay(ctx.choicePopup);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}

		// 3. Permission popup
		if (ctx.permissionPopup.isVisibleOverlay()) {
			const action = ctx.permissionPopup.handleInput(data);
			if (action) {
				handlePermissionPopupInput(ctx, action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}

		// 4. Autocomplete popups (file, skill, slash)
		if (handleAutocompleteInput(ctx, data)) {
			return { consume: true };
		}

		// 5. Key bindings
		if (handleKeyBinding(ctx, data)) return { consume: true };

		// 6. Ctrl+Backspace handled by InputBar directly
		return { consume: false };
	});

	// Live slash autocomplete and mode color updates
	ctx.inputBar.onChange = (text: string) => handleInputChange(ctx, text);

	// Input bar submission
	ctx.inputBar.onSubmit = (text: string, intent: string) => handleInputSubmit(ctx, text, intent);

	// Cancel handler
	ctx.inputBar.onCancel = () => {
		void ctx.cancelActiveTurn();
	};
}

// ── Utility helpers ──────────────────────────────────────────────────────────

/** Collapse whitespace and cap length for a one-line steer toast preview. */
function oneLineSteerPreview(text: string, maxLength = 60): string {
	const flat = text.replace(/\s+/g, " ").trim();
	return flat.length > maxLength ? `${flat.slice(0, maxLength)}…` : flat;
}

/**
 * OMP-style `$code` / `$$code` prefix detection.
 * Returns 0 (not a python command), 1 (single `$`), or 2 (double `$$`).
 */
function pythonCommandPrefixLength(trimmedText: string): 0 | 1 | 2 {
	if (trimmedText.charCodeAt(0) !== 36 /* $ */) return 0;
	if (trimmedText.charCodeAt(1) === 123 /* { */) return 0;

	const prefixLength = trimmedText.charCodeAt(1) === 36 /* $ */ ? 2 : 1;
	const next = trimmedText.charCodeAt(prefixLength);
	if (Number.isNaN(next)) return prefixLength;
	return next === 32 || next === 9 || next === 10 || next === 13
		? prefixLength
		: 0;
}

// Regex patterns to detect pasted shell prompts that should NOT trigger Python mode.
const SHELL_PROMPT_COMMAND_RE =
	/^(?:\.{0,2}\/|~\/|cd(?:\s|$)|sudo(?:\s|$)|git(?:\s|$)|bun(?:\s|$)|npm(?:\s|$)|pnpm(?:\s|$)|yarn(?:\s|$)|node(?:\s|$)|python\d*(?:\s|$)|cargo(?:\s|$)|go(?:\s|$)|make(?:\s|$)|docker(?:\s|$)|kubectl(?:\s|$))/;
const SHELL_PROMPT_OPERATOR_RE =
	/(?:^|\s)(?:&&|\|\||\||2>&1|[<>]{1,2})(?:\s|$)/;

function looksLikePastedShellPrompt(code: string): boolean {
	const firstLine = code.split("\n", 1)[0]?.trimStart() ?? "";
	return (
		SHELL_PROMPT_COMMAND_RE.test(firstLine) ||
		SHELL_PROMPT_OPERATOR_RE.test(firstLine)
	);
}
// ── End of input controller ──────────────────────────────────────────────────
