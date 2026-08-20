// ── Bridge setup & event dispatch ───────────────────────────────────────────
// Wires the AgentCoreBridge event stream and error stream, and fuses
// bridge-event handling with overlay-opening side effects for
// permission/question requests.

import { formatContextSize } from "@logician/agent-core";
import type {
	AgentCoreBridge,
	GoalManager,
	GoalState,
} from "@logician/agent-core/application";
import {
	isTranscriptEvent,
	type RuntimeEvent,
} from "@logician/agent-core/runtime";
import type { Transcript } from "@logician/agent-core/sessions";
import type { AutoresearchSession } from "@logician/autoresearch";
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
import type { TuiHandle } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { getGitStatus, getGitVersion } from "./git-status.ts";
import { registerRuntimeCommandContributions } from "./runtime/command-contributions.ts";
import { formatStartupMessage } from "./startup/message.ts";

export interface BridgeEventHandlerCtx {
	tui: TuiHandle;
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
	researchManager: AutoresearchSession;
	configPath?: string;
	workflowMode: "act" | "plan";
	planPhase: "idle" | "planning" | "awaiting_approval" | "executing";
	normalPermissionMode: "acceptAll" | "acceptEdits" | "ask";
	_autoSaveTurn: () => void;
	evaluateGoal: (goalState: Readonly<GoalState>) => Promise<void>;
}

export function setupBridge(ctx: BridgeEventHandlerCtx): void {
	const eventHandler = (event: RuntimeEvent): void => {
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
			registerRuntimeCommandContributions(ctx);
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
	event: RuntimeEvent,
): void {
	// Update transcript state
	if (isTranscriptEvent(event)) ctx.transcript.handleEvent(event);
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
			break;
		case "queue_update":
			ctx.steerQueue.setItems(
				event.steering || [],
				event.followUp || [],
				event.nextTurn || [],
			);
			break;
		case "permission_request": {
			ctx.pendingPermission = {
				toolCallId: event.toolCallId,
				toolName: event.toolName,
			};
			const preview = JSON.stringify(event.args ?? {}).slice(0, 500);
			ctx.permissionPopup.setToolInfo(event.toolName, preview);
			ctx.permissionPopup.setChoices([
				{
					value: "allow",
					label: "Allow once",
					description: "Run this tool for this call only",
				},
				{
					value: "always",
					label: "Always allow",
					description: `Allow ${event.toolName} without asking`,
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
			break;
		}
		case "question_request": {
			ctx.choicePopupPreview = false;
			ctx.choicePopup.setQuestionId(event.questionId);
			ctx.choicePopup.setQuestions(event.questions);
			ctx.choicePopup.show();
			const overlay = ctx.tui.showOverlay(ctx.choicePopup, {
				anchor: "aboveInput",
				align: "left",
				maxHeight: 24,
			});
			overlay.focus();
			break;
		}
		case "token":
		case "thinking_token":
		case "message_update":
		case "message_reasoning_update":
		case "tool_call_start":
		case "tool_call_update":
		case "tool_call_id_update":
		case "tool_execution_update":
		case "subagent_chunk":
		case "subagent_lifecycle":
			break;
		case "agent_retry_start":
			ctx.statusPanel.update({ phase: "retrying" });
			break;
		case "agent_retry_end":
			ctx.statusPanel.update({ phase: event.success ? "thinking" : "error" });
			break;
		case "runtime_status":
			ctx.statusPanel.update({
				runPhase: event.runPhase,
				runtimeRetry: event.retry,
				runtimeRepair: event.repair,
				activeSubagents: event.activeSubagents,
			});
			break;
		case "agent_error":
			ctx.transcript.handleEvent({
				type: "notice",
				level: event.recoverable ? "warn" : "error",
				label: `Agent error: ${event.phase}`,
				text: event.message,
			});
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			break;
		case "tool_execution_start":
			ctx.workSurface.recordToolStart(
				event.toolCallId,
				event.toolName,
				event.args,
			);
			break;
		case "tool_execution_end":
			ctx.workSurface.recordToolEnd(
				event.toolCallId,
				event.toolName,
				event.result,
				event.isError,
			);
			if (
				!event.isError &&
				[
					"edit_file",
					"write_file",
					"write_file_append",
					"git",
					"bash",
				].includes(event.toolName)
			) {
				updateGitFooter(ctx);
			}
			break;
		case "turn_end":
			// Auto-save the completed turn
			ctx._autoSaveTurn();
			updateGitFooter(ctx);
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
			ctx.researchManager.onAgentEnd();
			if (ctx.planPhase === "planning") {
				ctx.planPhase = "awaiting_approval";
				ctx.choicePopupPreview = false;
				ctx.choicePopup.setQuestionId("__plan_approval__");
				ctx.choicePopup.setQuestions([
					{
						id: "decision",
						header: "Plan approval",
						question: "Approve this plan and execute it?",
						choices: [
							{ value: "approve", label: "Approve and execute" },
							{
								value: "reject",
								label: "Reject",
								description: "Do not execute the plan",
							},
						],
					},
				]);
				ctx.choicePopup.show();
				ctx.tui
					.showOverlay(ctx.choicePopup, {
						anchor: "aboveInput",
						align: "left",
						maxHeight: 14,
					})
					.focus();
			} else if (ctx.planPhase === "executing") {
				ctx.planPhase = "idle";
				ctx.bridge.setPermissionMode(ctx.normalPermissionMode);
			}
			break;
		case "turn_start":
			ctx.workSurface.startRun();
			ctx.researchManager.onAgentStart();
			break;
		case "agent_iteration_start":
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
				Number(event.maxTokens || 0) || undefined,
			);
			ctx.statusPanel.update({
				contextTokens: Number(event.tokens || 0),
				contextMaxTokens: Number(event.maxTokens || 0) || undefined,
				contextCompacted: event.compacted === true,
				...(typeof event.cachedTokens === "number" && {
					cacheReadTokens: event.cachedTokens,
				}),
				...(event.promptTokens !== undefined && {
					promptTokens:
						typeof event.promptTokens === "number"
							? event.promptTokens
							: undefined,
				}),
				...(event.completionTokens !== undefined && {
					completionTokens:
						typeof event.completionTokens === "number"
							? event.completionTokens
							: undefined,
				}),
			});
			break;
		case "compaction":
			ctx.transcript.handleEvent({
				type: "notice",
				level: "info",
				label: "Compaction",
				text: `Context compacted (${formatContextSize(
					Number(event.tokensBefore || 0),
				)} -> ${formatContextSize(Number(event.tokensAfter || 0))}).`,
			});
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.statusPanel.update({
				phase: "compacted",
				contextTokens: Number(event.tokensAfter || 0),
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
					ctx.tui.requestRender();
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
		default:
			assertNever(event);
	}

	ctx.tui.requestRender();
}

function updateGitFooter(ctx: BridgeEventHandlerCtx): void {
	const git = getGitStatus();
	ctx.statusPanel.update({
		branch: git.branch,
		gitCommit: git.commit,
		gitModified: git.modified,
		gitStaged: git.staged,
		gitUntracked: git.untracked,
		gitAhead: git.ahead,
		gitBehind: git.behind,
		gitAddedLines: git.addedLines,
		gitRemovedLines: git.removedLines,
	});
}

function assertNever(value: never): never {
	throw new Error(
		`Unhandled runtime event: ${String((value as { type?: unknown }).type)}`,
	);
}
