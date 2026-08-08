// ── TUI input controller ───────────────────────────────────────────────────

import {
	filterSlashCommands,
	type SlashCommandDef,
} from "@logician/coding-agent/commands";
import type { LogicianTUI } from "./tui.ts";

export function setupInputHandler(ctx: LogicianTUI): void {
	// ── Choice popup handlers ──────────────────────────────────────
	const handleChoicePopupSubmit = (): void => {
		const qid = ctx.choicePopup.getQuestionId();
		const answers = ctx.choicePopup.getAnswers();
		if (ctx.choicePopupPreview) {
			ctx.choicePopupPreview = false;
			ctx.transcript.addSystemMessage(
				`Ask preview: ${JSON.stringify(answers)}`,
			);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			return;
		}
		if (
			qid &&
			ctx.bridge.respondToQuestion(qid, ctx.choicePopup.getResponseValue())
		) {
			ctx.transcript.addSystemMessage(
				`Questions answered: ${Object.keys(answers).length}`,
			);
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	};

	const handleChoicePopupDismiss = (): void => {
		if (ctx.choicePopupPreview) {
			ctx.choicePopupPreview = false;
			ctx.tui.requestRender();
			return;
		}
		const qid = ctx.choicePopup.getQuestionId();
		if (qid) {
			ctx.bridge.respondToQuestion(qid, "__dismissed__");
			ctx.transcript.addSystemMessage("Question dismissed.");
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	};

	// Global input listener
	ctx.tui.addInputListener((data: string) => {
		if (ctx.pluginManager.isVisibleOverlay()) {
			const action = ctx.pluginManager.handleInput(data);
			if (action) {
				ctx.handlePluginManagerAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.mcpManager.isVisibleOverlay()) {
			const action = ctx.mcpManager.handleInput(data);
			if (action) {
				ctx.handleMcpManagerAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.reasonerSelector.isVisibleOverlay()) {
			const action = ctx.reasonerSelector.handleInput(data);
			if (action) {
				ctx.handleReasonerSelectorAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.modelSelector.isVisibleOverlay()) {
			const action = ctx.modelSelector.handleInput(data);
			if (action) {
				ctx.handleModelSelectorAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.inferenceModeSelector.isVisibleOverlay()) {
			const action = ctx.inferenceModeSelector.handleInput(data);
			if (action) {
				ctx.handleInferenceModeSelectorAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.themeSelector.isVisibleOverlay()) {
			const action = ctx.themeSelector.handleInput(data);
			if (action) {
				ctx.handleThemeSelectorAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}
		if (ctx.settingsSelector.isVisibleOverlay()) {
			const action = ctx.settingsSelector.handleInput(data);
			if (action) {
				ctx.handleSettingsSelectorAction(action);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}

		// ChoicePopup — agent Q&A popup
		if (ctx.choicePopup.isVisibleOverlay()) {
			const action = ctx.choicePopup.handleInput(data);
			if (action) {
				if (action.type === "submit") {
					handleChoicePopupSubmit();
				} else {
					handleChoicePopupDismiss();
				}
				ctx.tui.removeOverlay(ctx.choicePopup);
			}
			ctx.tui.requestRender();
			return { consume: true };
		}

		// PermissionPopup — tool permission overlay
		if (ctx.permissionPopup.isVisibleOverlay()) {
			const action = ctx.permissionPopup.handleInput(data);
			if (action) {
				if (action.type === "close") {
					ctx.pendingPermission = null;
					ctx.transcript.addSystemMessage("Permission request dismissed.");
				} else {
					ctx.bridge.respondToPermission(
						ctx.pendingPermission?.toolCallId ?? "",
						action.choice.value,
					);
					ctx.transcript.addSystemMessage(
						`Permission ${action.choice.value}: ${ctx.pendingPermission?.toolName ?? "unknown"}`,
					);
				}
				ctx.pendingPermission = null;
				ctx.permissionPopup.hide();
				ctx.tui.removeOverlay(ctx.permissionPopup);
				if (action.type !== "close") {
					ctx.statusPanel.update({ phase: "streaming" });
				} else {
					ctx.statusPanel.update({ phase: "ready" });
				}
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			}
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Inline @-mention autocomplete: same pattern as the slash popup below —
		// the input bar keeps focus, we only intercept nav/accept keys.
		if (ctx.fileMentionPopup.isVisibleOverlay()) {
			if (data === "\x1b[A" || data === "\x1bOA") {
				ctx.fileMentionPopup.moveSelection(-1);
				ctx.tui.requestRender();
				return { consume: true };
			}
			if (data === "\x1b[B" || data === "\x1bOB") {
				ctx.fileMentionPopup.moveSelection(1);
				ctx.tui.requestRender();
				return { consume: true };
			}
			if (data === "\t" || data === "\r" || data === "\n") {
				const file = ctx.fileMentionPopup.currentFile();
				if (file) {
					ctx.inputBar.insertMention(file);
				}
				ctx.fileMentionPopup.hide();
				ctx.tui.requestRender();
				return { consume: true };
			}
			if (data === "\x1b") {
				ctx.fileMentionPopup.hide();
				ctx.tui.requestRender();
				return { consume: true };
			}
			// Everything else (typing, backspace, etc.) goes to the input bar; the
			// onChange hook re-syncs the popup query afterwards.
		}

		// Inline slash autocomplete: while the popup is showing matches, the
		// input bar keeps focus and ordinary typing flows through to it. We only
		// intercept the navigation/accept keys here.
		if (ctx.slashPopup.isVisibleOverlay()) {
			// Up / Down — move highlight
			if (data === "\x1b[A" || data === "\x1bOA") {
				ctx.slashPopup.moveSelection(-1);
				ctx.tui.requestRender();
				return { consume: true };
			}
			if (data === "\x1b[B" || data === "\x1bOB") {
				ctx.slashPopup.moveSelection(1);
				ctx.tui.requestRender();
				return { consume: true };
			}
			// Tab — complete input to the highlighted command
			if (data === "\t") {
				const cmd = ctx.slashPopup.currentCommand();
				if (cmd) {
					ctx.inputBar.valueText = `${cmd} `;
					ctx.tui.requestRender();
				}
				return { consume: true };
			}
			// Escape — dismiss the menu, clear/arm the composer, and stop an
			// active loop. A following Escape cancels the active model turn.
			if (data === "\x1b") {
				ctx.slashPopup.hide();
				// Let the composer consume the first Escape too: it clears the
				// slash draft and arms the second Escape for turn cancellation.
				ctx.inputBar.handleInput(data);
				if (ctx.loopActive) {
					ctx.loopManager.stop();
					ctx.loopActive = false;
					ctx.transcript.addSystemMessage("Loop stopped (Esc).");
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
				}
				ctx.tui.requestRender();
				return { consume: true };
			}
			// Enter — accept highlighted command (submit it directly)
			if (data === "\r" || data === "\n") {
				const cmd = ctx.slashPopup.currentCommand();
				if (cmd && ctx.inputBar.valueText.trim() !== cmd) {
					// If the typed text isn't already an exact command, accept the
					// highlighted one (carrying over any args the user typed).
					if (/^\/\S+\s+\S+/.test(cmd)) {
						ctx.inputBar.valueText = cmd;
					} else {
						const typedArgs = ctx.inputBar.valueText.replace(/^\/\S*\s*/, "");
						ctx.inputBar.valueText = typedArgs ? `${cmd} ${typedArgs}` : cmd;
					}
				}
				ctx.slashPopup.hide();
				// Fall through to the input bar so it submits the value.
				return { consume: false };
			}
			// Everything else (typing, backspace, etc.) goes to the input bar; the
			// onChange hook re-syncs the popup query afterwards.
		}

		// Ctrl+L — open model selector
		if (data === "\x0c") {
			ctx.openModelSelector();
			return { consume: true };
		}

		// Ctrl+G — jump to a file from the current working set (files touched
		// this session), inserting it as an @-mention in the composer.
		if (data === "\x07") {
			const files = ctx.workSurface.getWorkingSet();
			if (files.length === 0) {
				ctx.notify("Working set is empty.", "info");
				return { consume: true };
			}
			if (ctx.inputBar.getActiveMentionQuery() === null) {
				ctx.inputBar.valueText = `${ctx.inputBar.valueText}@`;
			}
			ctx.fileMentionPopup.setFiles(files);
			ctx.fileMentionPopup.setQuery("");
			ctx.fileMentionPopup.show();
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+O — expand/collapse tool execution details
		if (data === "\x0f") {
			const expanded = ctx.transcriptDisplay.toggleToolsExpanded();
			ctx.statusPanel.update({
				phase: expanded ? "tools expanded" : "tools collapsed",
			});
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Alt+J / Alt+K — move between tool cards. Alt+Enter toggles only the
		// focused card, providing keyboard parity with mouse clicks.
		if (data === "\x1bj" || data === "\x1bk") {
			const position = ctx.transcriptDisplay.focusTool(
				data === "\x1bj" ? 1 : -1,
			);
			if (position) {
				ctx.notify(`Tool ${position.index}/${position.total}`, "info");
				ctx.tui.requestRender();
			}
			return { consume: true };
		}
		if (data === "\x1b\r" || data === "\x1b\n") {
			const expanded = ctx.transcriptDisplay.toggleFocusedTool();
			if (expanded !== null) {
				ctx.notify(expanded ? "Tool expanded" : "Tool collapsed", "info");
				ctx.tui.requestRender();
			}
			return { consume: true };
		}

		// Ctrl+Shift+T — cycle thinking display mode
		if (data === "\x14") {
			ctx.transcript.cycleThinkingDisplayMode();
			ctx.transcriptDisplay.setThinkingMode(
				ctx.transcript.getThinkingDisplayMode(),
			);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+S — open session manager
		if (data === "\x13") {
			ctx.openSessionManager();
			return { consume: true };
		}

		// Ctrl+K — cycle sandbox mode (off / code / full)
		if (data === "\x0b") {
			const mode = ctx.bridge.cycleSandboxMode();
			ctx.statusPanel.update({ sandboxMode: mode });
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+P — toggle plan mode (plan ↔ act)
		if (data === "\x10") {
			const next =
				ctx.bridge.getPermissionMode() === "plan" ? "acceptAll" : "plan";
			ctx.bridge.setPermissionMode(next);
			ctx.statusPanel.update({ permissionMode: next });
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+Enter — submit the composer as immediate steering. With an
		// empty composer, retain the shortcut for flushing an existing queue.
		if (data === "\x1b[13;5u") {
			if (ctx.inputBar.submit("steer-now")) {
				return { consume: true };
			}
			const count = ctx.bridge.flushSteeringNow();
			if (count > 0) {
				ctx.transcript.addSystemMessage(
					`Flushed ${count} steering message${count === 1 ? "" : "s"} to the active turn.`,
				);
			} else {
				ctx.transcript.addSystemMessage(
					"No queued steering messages to flush.",
				);
			}
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+M — open inference-mode selector (replaces old cycle behavior).
		// Requires CSI-u because legacy terminals encode it like Enter.
		// Alt+M remains the portable fallback.
		if (
			data === "\x1bm" ||
			data === "\x1bM" ||
			data === "\x1b[109;5u" ||
			data === "\x1b[109;6u"
		) {
			ctx.openInferenceModeSelector();
			return { consume: true };
		}

		// Ctrl+I — cycle execution policy (autonomous ↔ minimal)
		if (data === "\x1b[105;5u" || data === "\x1b[105;6u") {
			const next = ctx.cycleExecutionProfile();
			ctx.notify(`Execution policy: ${next}`, "success");
			ctx.tui.requestRender();
			return { consume: true };
		}

		// Ctrl+Backspace in input bar is handled by InputBar directly
		return { consume: false };
	});

	// Live slash autocomplete: show/hide + filter the popup as the input text
	// changes. Commands with declared subcommands continue offering matches after
	// the first space (for example, `/mcp li` offers `/mcp list`).
	ctx.inputBar.onChange = (text: string) => {
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
		}

		ctx.tui.requestRender();
	};

	// Input bar handler
	ctx.inputBar.onSubmit = (text: string, intent) => {
		// A pending permission request captures the next submission:
		// y/a/n (or allow/always/deny) answers it instead of becoming a message.
		if (ctx.pendingPermission) {
			const answer = text.trim().toLowerCase();
			const decision =
				answer === "y" || answer === "yes" || answer === "allow"
					? "allow"
					: answer === "a" || answer === "always"
						? "always"
						: "deny";
			ctx.bridge.respondToPermission(
				ctx.pendingPermission.toolCallId,
				decision,
			);
			ctx.transcript.addSystemMessage(
				`Permission ${decision}: ${ctx.pendingPermission.toolName}`,
			);
			ctx.pendingPermission = null;
			ctx.statusPanel.update({ phase: "streaming" });
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			return;
		}

		// Always push to history (both slash and regular messages)
		ctx.inputBar.pushHistory(text);

		// Check for slash commands
		if (text.startsWith("/")) {
			const parts = text.trim().split(/\s+/);
			const cmdName = parts[0].toLowerCase();
			const args = parts.slice(1).join(" ");
			const allCmds = ctx.slashPopup.getCommands() as SlashCommandDef[];
			const match = allCmds?.find(
				(c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
			);

			if (match) {
				ctx.slashPopup.submitRaw(text.trim());
				return;
			}

			// Unknown command — a skill invocation? (/<skill-name> args)
			if (ctx.bridge.invokeSkill(cmdName.slice(1), args)) {
				ctx.transcript.addTurn(text.trim());
				ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
				ctx.statusPanel.update({ phase: "streaming" });
				ctx.statusPanel.startAnimation();
				ctx.tui.requestRender();
				return;
			}

			// Unknown command — do not silently turn a typo into an agent prompt.
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
			return;
		}

		// While a turn is running, a plain message steers it instead of
		// starting a new run. The bridge emits a `steered` event that
		// renders the message, so skip the normal turn/animation setup.
		if (ctx.bridge.isActive()) {
			ctx.notify(
				intent === "steer-now" ? "Steering now…" : "Steering queued…",
				"info",
			);
			ctx.tui.renderNow();
			setImmediate(() => {
				void ctx.bridge
					.sendMessage(text)
					.catch(err => ctx.bridge.reportError(err));
				if (intent === "steer-now") {
					const count = ctx.bridge.flushSteeringNow();
					ctx.notify(
						`Steering now with ${count} message${count === 1 ? "" : "s"}.`,
						"info",
					);
					ctx.tui.requestRender();
				}
			});
			return;
		}

		ctx.transcript.addTurn(text);
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.statusPanel.update({ phase: "streaming" });
		ctx.statusPanel.startAnimation();
		// Paint the submitted turn and active status before bridge setup. Model,
		// plugin, and skill initialization can do synchronous work before their
		// first await; deferring it keeps Enter-to-feedback latency near one frame.
		ctx.tui.renderNow();
		setImmediate(() => {
			void ctx.bridge
				.sendMessage(text)
				.catch(err => ctx.bridge.reportError(err));
		});
	};

	ctx.inputBar.onCancel = () => {
		void ctx.cancelActiveTurn();
	};
}
