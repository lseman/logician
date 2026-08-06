// ── Bridge setup & event dispatch ───────────────────────────────────────────
// Wires the AgentCoreBridge event stream and error stream, and fuses
// bridge-event handling with overlay-opening side effects for
// permission/question requests.

import { formatContextSize } from "@logician/coding-agent";
import type {
	AgentCoreBridge,
	GoalManager,
	GoalState,
} from "@logician/coding-agent/application";
import type { SlashCommandDef } from "@logician/coding-agent/commands";
import type { ParsedBridgeEvent } from "@logician/coding-agent/runtime";
import type { Transcript } from "@logician/coding-agent/sessions";
import type { ChoicePopup } from "../overlays/choice-popup.ts";
import type { PermissionPopup } from "../overlays/permission-popup.ts";
import type { SlashPopup } from "../overlays/slash-popup.ts";
import type { TranscriptDisplay } from "../rendering/transcript/display.ts";
import {
	reduceTurnState,
	type TurnState,
	turnPhaseIsActive,
	turnPhaseLabel,
} from "../state/turn-state.ts";
import type { StatusBar } from "../status/status-bar.ts";
import type { SteerQueue } from "../status/steer-queue.ts";
import type { TodoBar } from "../status/todo-bar.ts";
import type { WorkSurface } from "../status/work-surface.ts";
import type { TUI } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { getGitVersion } from "./git-status.ts";
import { formatStartupMessage } from "./startup/message.ts";

export interface BridgeEventHandlerCtx {
	tui: TUI;
	bridge: AgentCoreBridge;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	todoBar: TodoBar;
	workSurface: WorkSurface;
	steerQueue: SteerQueue;
	slashPopup: SlashPopup;
	choicePopup: ChoicePopup;
	choicePopupPreview: boolean;
	permissionPopup: PermissionPopup;
	pendingPermission: { toolCallId: string; toolName: string } | null;
	turnState: TurnState;
	goalManager: GoalManager;
	goalActive: boolean;
	configPath?: string;
	_autoSaveTurn: () => void;
	evaluateGoal: (goalState: Readonly<GoalState>) => Promise<void>;
}

export function setupBridge(ctx: BridgeEventHandlerCtx): void {
	const eventHandler = (event: ParsedBridgeEvent): void => {
		handleEvent(ctx, event);
	};

	ctx.bridge.on(eventHandler);
	ctx.bridge.onError(err => {
		// Also display in transcript so the user sees connection/server errors
		ctx.transcript.addSystemMessage(`Connection error: ${err.message}`);
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	});

	// Initialize bridge
	ctx.bridge
		.init()
		.then(state => {
			ctx.statusPanel.update({
				contextTokens: Number(state.context_tokens || 0),
				contextMaxTokens: Number(state.context_max_tokens || 0) || undefined,
				sandboxMode: ctx.bridge.getSandboxMode(),
				memoryEnabled: ctx.bridge.getSettingsData().memoryEnabled,
			});
			// Don't add startup message when restoring a session — user history
			// is already visible; prepending startup text just causes rendering
			// overlap.
			const turns = ctx.transcript.getTurns();
			if (turns.length === 0) {
				const message = formatStartupMessage(state, {
					configPath: ctx.configPath,
					project: getGitVersion() || "-",
					themeName: theme.name,
				});
				if (message) {
					ctx.transcript.addSystemMessage(message);
					ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
					ctx.tui.requestRender();
				}
			}
			// Surface discovered skills as /<skill-name> commands in the popup.
			const skills = ctx.bridge.getSkills();
			if (skills.length) {
				const existing = ctx.slashPopup.getCommands() as SlashCommandDef[];
				const taken = new Set(existing.map(c => c.command));
				const skillCmds: SlashCommandDef[] = skills
					.map(s => ({
						command: `/${s.slashName}`,
						usage: `/${s.slashName}${s.argumentHint ? ` ${s.argumentHint}` : ""}`,
						description: `Skill: ${s.description.slice(0, 80)}`,
						dispatch: "local" as const,
						acceptsArgs: true,
						bridgeHandler: (args: string) => {
							ctx.bridge.invokeSkill(s.name, args);
						},
					}))
					.filter(c => !taken.has(c.command));
				if (skillCmds.length) {
					ctx.slashPopup.setCommands([...existing, ...skillCmds]);
				}
			}
			// Surface discovered prompts as /<prompt-name> commands in the popup.
			const prompts = ctx.bridge.getPrompts();
			if (prompts.length) {
				const existing = ctx.slashPopup.getCommands() as SlashCommandDef[];
				const taken = new Set(existing.map(c => c.command));
				const promptCmds: SlashCommandDef[] = prompts
					.map(p => ({
						command: `/${p.slashName}`,
						usage: `/${p.slashName}${p.argumentHint ? ` ${p.argumentHint}` : ""}`,
						description: `Prompt: ${p.description.slice(0, 80)}`,
						dispatch: "local" as const,
						acceptsArgs: true,
						bridgeHandler: (args: string) => {
							ctx.bridge.invokePrompt(p.name, args);
						},
					}))
					.filter(c => !taken.has(c.command));
				if (promptCmds.length) {
					ctx.slashPopup.setCommands([...existing, ...promptCmds]);
				}
			}
		})
		.catch(err => {
			// Display init/connection errors in transcript so the user knows
			// the agent couldn't start (e.g. server unreachable).
			ctx.transcript.addSystemMessage(`Failed to start agent: ${err.message}`);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
		});
}

export function handleEvent(
	ctx: BridgeEventHandlerCtx,
	event: ParsedBridgeEvent,
): void {
	// Update transcript state
	ctx.transcript.handleEvent(event);
	ctx.turnState = reduceTurnState(ctx.turnState, event);
	ctx.workSurface.setPhase(ctx.turnState.phase);
	ctx.statusPanel.update({ phase: turnPhaseLabel(ctx.turnState.phase) });
	if (turnPhaseIsActive(ctx.turnState.phase)) {
		ctx.statusPanel.startAnimation();
		ctx.transcriptDisplay.startAnimation();
	} else {
		ctx.statusPanel.stopAnimation();
		ctx.transcriptDisplay.stopAnimation();
	}

	switch (event.type) {
		case "todos":
			ctx.todoBar.setTodos(event.todos);
			ctx.tui.requestRender();
			break;
		case "queue_update":
			ctx.steerQueue.setItems(event.steering || [], event.followUp || []);
			ctx.tui.requestRender();
			break;
		case "permission_request": {
			ctx.pendingPermission = {
				toolCallId: event.tool_call_id,
				toolName: event.tool_name,
			};
			const preview = JSON.stringify(event.args ?? {}).slice(0, 500);
			ctx.permissionPopup.setToolInfo(event.tool_name, preview);
			ctx.permissionPopup.setChoices([
				{
					value: "allow",
					label: "Allow once",
					description: "Run this tool for this call only",
				},
				{
					value: "always",
					label: "Always allow",
					description: `Allow ${event.tool_name} without asking`,
				},
				{ value: "deny", label: "Deny", description: "Block this tool call" },
			]);
			ctx.permissionPopup.show();
			const overlay = ctx.tui.showOverlay(ctx.permissionPopup, {
				anchor: "aboveInput",
				align: "left",
				maxHeight: 14,
			});
			overlay.focus();
			ctx.statusPanel.update({ phase: "permission" });
			ctx.statusPanel.stopAnimation();
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			break;
		}
		case "question_request": {
			ctx.choicePopupPreview = false;
			ctx.choicePopup.setQuestionId(event.question_id);
			ctx.choicePopup.setQuestions(event.questions);
			ctx.choicePopup.show();
			const overlay = ctx.tui.showOverlay(ctx.choicePopup, {
				anchor: "aboveInput",
				align: "left",
				maxHeight: 24,
			});
			overlay.focus();
			ctx.tui.requestRender();
			break;
		}
		case "token":
			break;
		case "tool_start":
		case "tool_execution_start":
			ctx.workSurface.recordToolStart(
				event.tool_call_id,
				event.tool_name || event.tool,
				event.tool_args,
			);
			break;
		case "tool_end":
		case "tool_execution_end":
			ctx.workSurface.recordToolEnd(
				event.tool_call_id,
				event.tool_name || event.tool,
				event.result,
				event.is_error,
			);
			break;
		case "turn_end":
			// Auto-save the completed turn
			ctx._autoSaveTurn();
			ctx.statusPanel.update({
				turnCount: ctx.transcript.getTurns().length,
				messageCount: ctx.transcript.getMessageCount(),
			});
			// Goal evaluation: if a goal is active, evaluate after each turn
			if (ctx.goalActive && ctx.goalManager.isSet()) {
				const goalState = ctx.goalManager.getState();
				if (goalState && goalState.status === "active") {
					void ctx.evaluateGoal(goalState);
				}
			}
			break;
		case "turn_start":
			ctx.workSurface.startTurn();
			break;
		case "phase":
			if (event.state === "ready") {
				ctx.statusPanel.update({
					turnCount: ctx.transcript.getTurns().length,
					messageCount: ctx.transcript.getMessageCount(),
				});
			}
			break;
		case "context_update":
			ctx.workSurface.setContext(
				Number(event.tokens || 0),
				Number(event.max_tokens || 0) || undefined,
			);
			ctx.statusPanel.update({
				contextTokens: Number(event.tokens || 0),
				contextMaxTokens: Number(event.max_tokens || 0) || undefined,
				contextCompacted: event.compacted === true,
				...(typeof event.cached_tokens === "number" && {
					cacheReadTokens: event.cached_tokens,
				}),
				...("prompt_tokens" in event && {
					promptTokens:
						typeof event.prompt_tokens === "number"
							? event.prompt_tokens
							: undefined,
				}),
				...("completion_tokens" in event && {
					completionTokens:
						typeof event.completion_tokens === "number"
							? event.completion_tokens
							: undefined,
				}),
			});
			break;
		case "compaction":
			ctx.transcript.addSystemMessage(
				`Context compacted (${formatContextSize(
					Number(event.tokens_before || 0),
				)} -> ${formatContextSize(Number(event.tokens_after || 0))}).`,
			);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.statusPanel.update({
				phase: "compacted",
				contextTokens: Number(event.tokens_after || 0),
				contextCompacted: true,
			});
			break;
		case "model_select":
			ctx.statusPanel.update({ model: event.model });
			break;
		case "notice":
			if (event.label === "MCP") {
				void ctx.bridge.getState().then(state => {
					ctx.statusPanel.update({
						mcpServerCount: Number(state.mcp_servers || 0),
						mcpLoading: false,
					});
				});
			} else if (event.label === "Memory") {
				ctx.statusPanel.update({
					memoryEnabled: ctx.bridge.getSettingsData().memoryEnabled,
				});
			}
			break;
		case "repair_nudge":
			ctx.transcript.addSystemMessage(
				`Tool-call repair: ${event.message || "recovered malformed tool call"}`,
			);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			break;
		case "memory_update": {
			if (event.kind === "observations_added") {
				const previews = (event.items ?? [])
					.slice(0, 3)
					.map(item => `[${item.id}] ${item.content.slice(0, 120)}`)
					.join("\n");
				ctx.transcript.addSystemMessage(
					`Memory added: ${event.count} observation${event.count === 1 ? "" : "s"}` +
						(previews ? `\n${previews}` : ""),
				);
			} else if (event.kind === "reflections_added") {
				ctx.transcript.addSystemMessage(
					`Memory synthesized: ${event.count} reflection${event.count === 1 ? "" : "s"}`,
				);
			} else if (event.kind === "reflections_evolved") {
				ctx.transcript.addSystemMessage(
					`Memory evolved: ${event.count} existing reflection${event.count === 1 ? "" : "s"}`,
				);
			} else if (event.kind === "observations_dropped") {
				ctx.transcript.addSystemMessage(
					`Memory compacted: ${event.count} observation${event.count === 1 ? "" : "s"} archived.`,
				);
			}
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			break;
		}
		case "steered":
			// Steering is part of the active run rather than a new turn, but it must
			// remain visible after the transient queue widget drains. Otherwise a
			// successfully queued user message looks as though it was discarded.
			ctx.transcript.addSystemMessage(
				`You steered the active turn:\n${String(event.message || "")}`,
			);
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			break;
	}

	ctx.tui.requestRender();
}
