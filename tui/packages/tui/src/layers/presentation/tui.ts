// ── Main TUI ──────────────────────────────────────────────────────────────────
// Wires agent-core, transcript, and components together.

import {
	getReasonerIds,
	getReasonerMeta,
	type ReasonerMeta,
} from "@logician/agent-capabilities/reasoners/registry";
import type { Message as CoreMessage } from "@logician/agent-core";
import { formatContextSize } from "@logician/coding-agent";
import { AgentCoreBridge } from "@logician/coding-agent/bridge";
import { saveConfigField } from "@logician/coding-agent/config";
import type { ParsedBridgeEvent } from "@logician/coding-agent/events";
import { LoopManager } from "@logician/coding-agent/loop-manager";
import {
	GoalManager,
	type GoalState,
	resolveRuntimeConfig,
} from "@logician/coding-agent/runtime";
import { SessionStore } from "@logician/coding-agent/session-store";
import {
	createSlashCommands,
	filterSlashCommands,
	type SlashCommandDef,
} from "@logician/coding-agent/slash-commands";
import { Transcript, type Turn } from "@logician/coding-agent/transcript";
import { ChoicePopup } from "../../components/choice-popup.ts";
import { InputBar } from "../../components/input-bar.ts";
import {
	type McpManagerAction,
	McpManagerOverlay,
} from "../../components/mcp-manager.ts";
import {
	type ModelInfo,
	type ModelSelectorAction,
	ModelSelectorOverlay,
} from "../../components/model-selector.ts";
import {
	type PluginManagerAction,
	PluginManagerOverlay,
} from "../../components/plugin-manager.ts";
import {
	type ReasonerInfo,
	type ReasonerSelectorAction,
	ReasonerSelectorOverlay,
} from "../../components/reasoner-selector.ts";
import { SessionBrowserOverlay } from "../../components/session-manager.ts";
import { getGitStatus, getGitVersion } from "./git-status.ts";
import {
	type SettingDef,
	type SettingsSelectorAction,
	SettingsSelectorOverlay,
} from "../../components/settings-overlay.ts";
import { SlashPopup } from "../../components/slash-popup.ts";
import { FileMentionPopup } from "../../components/file-mention-popup.ts";
import { listProjectFiles } from "@logician/coding-agent/file-mentions";
import { StatusBar } from "../../components/status-bar.ts";
import { SteerQueue } from "../../components/steer-queue.ts";
import {
	type ThemeInfo,
	type ThemeSelectorAction,
	ThemeSelectorOverlay,
} from "../../components/theme-selector.ts";
import { PermissionPopup } from "../../components/permission-popup.ts";
import { TodoBar } from "../../components/todo/todo-bar.ts";
import { TranscriptDisplay } from "../../components/transcript-display.ts";
import { WorkSurface } from "../../components/work-surface.ts";
import {
	INITIAL_TURN_STATE,
	reduceTurnState,
	type TurnState,
	turnPhaseIsActive,
	turnPhaseLabel,
} from "../../state/turn-state.ts";
import { Container, TUI } from "../core/tui-core.ts";
import { KillRing } from "../input/kill-ring.ts";
import { UndoStack } from "../input/undo-stack.ts";
import { getAvailableThemes, setTheme, theme } from "../theme/theme.ts";
import { formatStartupMemory } from "./startup-memory.ts";

// ── Main TUI ─────────────────────────────────────────────────────────────────

export class LogicianTUI {
	private tui: TUI;
	private bridge: AgentCoreBridge;
	private transcript: Transcript;
	private statusPanel: StatusBar;
	private todoBar: TodoBar;
	private workSurface: WorkSurface;
	private steerQueue: SteerQueue;
	private inputBar: InputBar;
	private slashPopup: SlashPopup;
	private fileMentionPopup: FileMentionPopup;
	private choicePopup: ChoicePopup;
	private permissionPopup: PermissionPopup;
	private pluginManager: PluginManagerOverlay;
	private mcpManager: McpManagerOverlay;
	private reasonerSelector: ReasonerSelectorOverlay;
	private modelSelector: ModelSelectorOverlay;
	private themeSelector: ThemeSelectorOverlay;
	private settingsSelector: SettingsSelectorOverlay;
	private transcriptDisplay: TranscriptDisplay;
	private sessionManager: SessionBrowserOverlay;
	private sessionStore: SessionStore;
	private killRing: KillRing;
	private undoStack: UndoStack<{ value: string; cursor: number }>;
	private loopManager: LoopManager;
	private goalManager: GoalManager;
	private turnState: TurnState = INITIAL_TURN_STATE;
	private loopActive = false;
	private goalActive = false;
	private goalEvaluationPending = false;
	private configPath?: string;
	private thinkingLevel = "off";
	private inferenceMode:
		| "thinking-general"
		| "thinking-coding"
		| "instruct-general"
		| "instruct-reasoning" = "instruct-general";
	private thinkingDisplayMode: "collapsed" | "summary" | "expanded" =
		"expanded";
	private traceOn = false;
	private currentSessionId: string | null = null;
	// Tool call awaiting an interactive allow/deny answer in the input bar.
	private pendingPermission: { toolCallId: string; toolName: string } | null =
		null;

	// Inference mode helper — used by the keyboard shortcut and /settings.
	private setInferenceMode(mode: string): void {
		const valid = [
			"thinking-general",
			"thinking-coding",
			"instruct-general",
			"instruct-reasoning",
		];
		if (!valid.includes(mode)) return;
		const oldMode = this.inferenceMode;
		this.inferenceMode = mode as typeof this.inferenceMode;
		this.bridge.setInferenceMode(mode);
		this.statusPanel.update({ inferenceMode: mode });
		if (oldMode !== mode) {
			const labels: Record<string, string> = {
				"thinking-general": "Thinking (General)",
				"thinking-coding": "Thinking (Precise Code)",
				"instruct-general": "Instruct (General)",
				"instruct-reasoning": "Instruct (Reasoning)",
			};
			this.transcript.addSystemMessage(
				`Inference mode: ${labels[mode] ?? mode}`,
			);
			saveConfigField("inferenceMode", mode);
		}
	}

	private cycleInferenceMode(): void {
		const modes: Array<
			| "thinking-general"
			| "thinking-coding"
			| "instruct-general"
			| "instruct-reasoning"
		> = [
			"thinking-general",
			"thinking-coding",
			"instruct-general",
			"instruct-reasoning",
		];
		const currentIndex = modes.indexOf(this.inferenceMode);
		this.setInferenceMode(modes[(currentIndex + 1) % modes.length]);
		this.tui.requestRender();
	}

	// eslint-disable-next-line max-lines-per-function -- wires up entire TUI (bridge, transcript, components, keybindings, overlays)
	constructor(
		runtimeConfig = resolveRuntimeConfig(process.cwd(), process.env, {
			loadProjectConfig: false,
		}),
	) {
		this.configPath = runtimeConfig.configPath;
		this.bridge = new AgentCoreBridge(runtimeConfig.bridge);
		this.transcript = new Transcript();
		this.statusPanel = new StatusBar();
		this.todoBar = new TodoBar();
		this.workSurface = new WorkSurface();
		this.steerQueue = new SteerQueue();
		this.inputBar = new InputBar();
		this.slashPopup = new SlashPopup();
		this.fileMentionPopup = new FileMentionPopup();
		this.choicePopup = new ChoicePopup();
		this.permissionPopup = new PermissionPopup();
		this.pluginManager = new PluginManagerOverlay();
		this.mcpManager = new McpManagerOverlay();
		this.reasonerSelector = new ReasonerSelectorOverlay();
		this.modelSelector = new ModelSelectorOverlay();
		this.themeSelector = new ThemeSelectorOverlay();
		this.settingsSelector = new SettingsSelectorOverlay();
		this.transcriptDisplay = new TranscriptDisplay({
			thinkingMode: this.thinkingDisplayMode,
			maxMessageLength:
				runtimeConfig.source.truncation?.transcriptMessageMaxChars,
		});
		this.transcriptDisplay.setOnAnimationTick(() => this.tui.requestRender());
		// Apply inference mode only after its transcript/status dependencies exist.
		if (runtimeConfig.source.inferenceMode) {
			this.setInferenceMode(runtimeConfig.source.inferenceMode);
		}
		this.killRing = new KillRing();
		this.undoStack = new UndoStack();
		this.loopManager = new LoopManager();
		this.loopManager.setOnTick(async (iteration, prompt) => {
			this.loopActive = true;
			this.transcript.addSystemMessage(
				`🔄 Loop iteration ${iteration}: ${prompt}`,
			);
			this.transcript.addTurn(prompt);
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
			await this.bridge.sendMessage(prompt);
		});
		this.loopManager.setOnStateChange((state) => {
			this.loopActive = state !== null;
			if (state?.lastError) {
				this.transcript.addSystemMessage(
					`Loop iteration ${state.iteration} failed: ${state.lastError}`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			}
		});

		this.goalManager = new GoalManager();
		this.goalManager.setOnStateChange((state: Readonly<GoalState> | null) => {
			if (state?.lastReason?.startsWith("Evaluation error:")) {
				this.transcript.addSystemMessage(
					`Goal evaluation #${state.turnCount} failed: ${state.lastReason}`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			}
			// Update status bar with goal indicator
			if (state && state.status === "active") {
				const elapsed = Math.round((Date.now() - state.startedAt) / 1000);
				this.statusPanel.update({
					goalCondition: state.condition,
					goalTurnCount: state.turnCount,
					goalElapsed: elapsed,
				});
			} else if (state && state.status === "achieved") {
				this.statusPanel.update({
					goalCondition: undefined,
					goalTurnCount: undefined,
					goalElapsed: undefined,
				});
			}
		});

		// Create the TUI with hardware cursor support
		this.tui = new TUI(process.stdout, true);
		this.statusPanel.setOnInvalidate(() => this.tui.requestRender());
		this.todoBar.setOnInvalidate(() => this.tui.requestRender());
		this.workSurface.setOnInvalidate(() => this.tui.requestRender());

		// ── Session store ────────────────────────────────────────────────────
		this.sessionStore = new SessionStore(process.cwd());
		this.sessionManager = new SessionBrowserOverlay();
		this.sessionManager.setStore(this.sessionStore);
		this.sessionManager.setActionCallback((action) =>
			this.handleSessionAction(action),
		);
		// Create initial session if none exists
		if (this.sessionStore.listSessions().length === 0) {
			this.currentSessionId = this.sessionStore.createSession({
				title: "New Session",
			});
			this.statusPanel.update({ sessionTitle: "New Session" });
		} else {
			// Resume the most recent session
			const sessions = this.sessionStore.listSessions();
			if (sessions.length > 0) {
				this.currentSessionId = sessions[0].id;
				const turns = this.sessionStore.loadTurns(this.currentSessionId);
				if (turns.length > 0) {
					this.restoreSession(turns);
					this.statusPanel.update({ sessionTitle: sessions[0].title });
					this.tui.requestRender();
				}
			}
		}

		// Wire up dependencies
		this.inputBar.setKillRing(this.killRing);
		this.inputBar.setUndoStack(this.undoStack);

		// Setup bridge event handling
		this.setupBridge();

		// Setup transcript change handling
		this.setupTranscript();

		// Wire up scrollable component
		this.tui.setScrollableComponent(this.transcriptDisplay);

		// ── Async helpers (must be defined before setupInputHandler) ─────────

		const setStatusPhase = (phase: string) => {
			this.statusPanel.update({ phase });
		};

		const handleStatus = async () => {
			try {
				const state = await this.bridge.getState();
				const runtime = (state.runtime_state ?? {}) as {
					phase?: string;
					isStreaming?: boolean;
					pendingToolCalls?: string[];
					retry?: { attempt?: number; maxRetries?: number };
					lastError?: string;
					outcome?: { status?: string; summary?: string; source?: string };
					lastTurnDurationMs?: number;
					lastRunDurationMs?: number;
				};
				const lines = [
					`Agent: ${state.agent_name || "unknown"}`,
					`Model: ${state.model || "unknown"}`,
					`Base URL: ${state.base_url || "unknown"}`,
					`Project: ${getGitVersion() || "-"}`,
					`Tools: ${(state.tools as string[])?.length || 0} loaded`,
					`MCP: ${state.mcp_servers || 0} server(s), ${state.mcp_tools || 0} tool(s)`,
					`Context: ${formatContextSize(
						Number(state.context_tokens || 0),
						Number(state.context_max_tokens || 0) || undefined,
					)}`,
					`Runtime: ${runtime.phase || "idle"}${runtime.isStreaming ? " (streaming)" : ""}`,
					`Active tools: ${runtime.pendingToolCalls?.length || 0}`,
					...(runtime.lastTurnDurationMs !== undefined
						? [`Last turn: ${runtime.lastTurnDurationMs}ms`]
						: []),
					...(runtime.lastRunDurationMs !== undefined
						? [`Last run: ${runtime.lastRunDurationMs}ms`]
						: []),
					...(runtime.retry
						? [
								`Retry: ${runtime.retry.attempt || 0}/${runtime.retry.maxRetries || 0}`,
							]
						: []),
					...(runtime.lastError ? [`Last error: ${runtime.lastError}`] : []),
					...(runtime.outcome
						? [
								`Outcome: ${runtime.outcome.status || "unknown"} (${runtime.outcome.source || "unknown"})`,
								...(runtime.outcome.summary
									? [`Outcome summary: ${runtime.outcome.summary}`]
									: []),
							]
						: []),
					`Hooks: ${state.hooks_enabled === false ? "disabled" : "enabled"}`,
					`Hook transcript: ${state.hook_transcript_path || "-"}`,
					`Config: ${state.config_path || "-"}`,
					`Connected: ${state.connected !== false}`,
				];
				const mcpErrors = Array.isArray(state.mcp_errors)
					? state.mcp_errors
							.map((item) => String(item || "").trim())
							.filter(Boolean)
					: [];
				if (mcpErrors.length) {
					lines.push("", "MCP errors:", ...mcpErrors.map((err) => `- ${err}`));
				}
				this.transcript.addSystemMessage(lines.join("\n"));
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			} catch (e: unknown) {
				this.transcript.addSystemMessage(
					`Status error: ${e instanceof Error ? e.message : String(e)}`,
				);
				this.tui.requestRender();
			}
		};

		const handlePlugins = async (args: string) => {
			try {
				const normalized = args.trim().toLowerCase();
				if (!normalized || normalized === "list") {
					await this.openPluginManager();
					return;
				}
				const result = await this.bridge.runPluginCommand(args);
				this.transcript.addSystemMessage(result);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			} catch (e: unknown) {
				this.transcript.addSystemMessage(
					`Plugins error: ${e instanceof Error ? e.message : String(e)}`,
				);
				this.tui.requestRender();
			}
		};

		const handleMcp = async (_args: string) => {
			try {
				await this.openMcpManager();
			} catch (e: unknown) {
				this.transcript.addSystemMessage(
					`MCP error: ${e instanceof Error ? e.message : String(e)}`,
				);
			}
		};

		const handleReasoner = async (args: string) => {
			try {
				const normalized = args.trim().toLowerCase();
				if (!normalized || normalized === "list") {
					await this.openReasonerSelector();
					return;
				}
				// Direct set: /reasoner ssr, /reasoner none, etc.
				// reasoner removed;
				const meta = getReasonerMeta(normalized);
				const label = meta ? meta.name : normalized;
				this.transcript.addSystemMessage(`Reasoning mode: ${label}`);
				this.statusPanel.update({ reasoner: normalized });
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			} catch (e: unknown) {
				this.transcript.addSystemMessage(
					`Reasoner error: ${e instanceof Error ? e.message : String(e)}`,
				);
				this.tui.requestRender();
			}
		};

		const handleTheme = async (args: string) => {
			try {
				const normalized = args.trim().toLowerCase();
				if (!normalized || normalized === "list") {
					await this.openThemeSelector();
					return;
				}
				// Direct set: /theme dark, /theme light, etc.
				const ok = this.setThemeByName(normalized);
				if (ok) {
					this.transcript.addSystemMessage(`Theme: ${normalized}`);
				} else {
					const available = getAvailableThemes();
					this.transcript.addSystemMessage(
						`Unknown theme "${normalized}". Available: ${available.join(", ")}`,
					);
				}
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			} catch (e: unknown) {
				this.transcript.addSystemMessage(
					`Theme error: ${e instanceof Error ? e.message : String(e)}`,
				);
				this.tui.requestRender();
			}
		};

		// Setup keyboard shortcuts
		this.setupInputHandler();

		// Focus input bar by default
		this.tui.setFocus(this.inputBar);

		// Initial state
		const gitStatus = getGitStatus();
		this.statusPanel.update({
			thinkingLevel: this.thinkingLevel,
			cacheReadTokens: undefined,
			phase: "ready",
			model: runtimeConfig.bridge.model || "local",
			cwd: process.cwd(),
			branch: gitStatus.branch,
			gitModified: gitStatus.modified,
			gitStaged: gitStatus.staged,
			gitUntracked: gitStatus.untracked,
			contextTokens: 0,
			reasoner: "none",
			contextMaxTokens: runtimeConfig.bridge.contextWindowTokens,
		});

		// Setup slash commands
		const localHandlers: Record<string, (...args: unknown[]) => unknown> = {
			setThinking: (level: unknown) => {
				const lvl = typeof level === "string" ? level : String(level);
				this.thinkingLevel = lvl;
				this.bridge.setThinkingLevel(lvl);
				this.statusPanel.update({ thinkingLevel: lvl });
				setStatusPhase("ready");
			},
			setInferenceMode: (mode: unknown) => {
				const m = typeof mode === "string" ? mode : String(mode);
				this.setInferenceMode(m);
				setStatusPhase("ready");
			},
			setThinkingMode: (mode: unknown) => {
				const m = typeof mode === "string" ? mode : String(mode);
				this.thinkingDisplayMode = m as typeof this.thinkingDisplayMode;
				this.transcript.setThinkingDisplayMode(
					m as "collapsed" | "summary" | "expanded",
				);
				setStatusPhase("ready");
			},
			cycleThinking: () => {
				this.transcript.cycleThinkingDisplayMode();
				setStatusPhase("ready");
			},
			setTrace: (on: unknown) => {
				this.traceOn =
					typeof on === "boolean" ? on : on === "true" || on === true;
				setStatusPhase("ready");
			},
			clear: () => {
				this.transcript.clear();
				setStatusPhase("ready");
			},
			version: () => "Logician 0.2.0 (TypeScript runtime)",
			eoh: (raw: unknown) => this.bridge.eohCommand(String(raw ?? "")),
			settings: (raw: unknown) => {
				const args = String(raw ?? "").trim();
				if (!args) {
					void this.openSettingsSelector();
					return "";
				}
				const [key, value = ""] = args.split(/\s+/, 2);
				const on = value.toLowerCase() === "on";
				switch (key.toLowerCase()) {
					case "thinking":
						if (!value) return "Usage: /settings thinking <level>";
						this.bridge.setThinkingLevel(value);
						return `Thinking level: ${value}`;
					case "model":
						if (!value) return "Usage: /settings model <name>";
						this.bridge.setModel(value);
						return `Model: ${value}`;
					case "model-cycle":
					case "model_cycle":
						return `Model: ${this.bridge.cycleModel() ?? "unchanged"}`;
					case "temp": {
						const number = Number(value);
						if (!Number.isFinite(number) || number < 0 || number > 2)
							return "Temperature must be between 0 and 2.";
						this.bridge.setTemperature(number);
						return `Temperature: ${number}`;
					}
					case "max-tokens":
					case "max_tokens": {
						const number = Number.parseInt(value, 10);
						if (!Number.isFinite(number) || number < 1)
							return "Max tokens must be a positive integer.";
						this.bridge.setMaxTokens(number);
						return `Max tokens: ${number}`;
					}
					case "max-iterations":
					case "max_iterations": {
						const number = Number.parseInt(value, 10);
						if (!Number.isFinite(number) || number < 1)
							return "Max iterations must be a positive integer.";
						this.bridge.setMaxIterations(number);
						return `Max iterations: ${number}`;
					}
					case "permissions":
						if (!value) return "Usage: /settings permissions <mode>";
						this.bridge.setPermissionMode(
							value as "acceptAll" | "acceptEdits" | "ask" | "plan",
						);
						return `Permission mode: ${value}`;
					case "loop-detection":
						this.bridge.setRuntimeToggle("loopDetectionEnabled", on);
						return `Loop detection: ${on ? "on" : "off"}`;
					case "guards":
						this.bridge.setRuntimeToggle("guardsEnabled", on);
						return `Guards: ${on ? "on" : "off"}`;
					case "compaction":
						this.bridge.setRuntimeToggle("proactiveCompactionEnabled", on);
						return `Compaction: ${on ? "on" : "off"}`;
					case "diagnostics":
					case "post-edit-diagnostics":
						this.bridge.setRuntimeToggle("postEditDiagnostics", on);
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
						this.setInferenceMode(value);
						return `Inference mode: ${value}`;
					}
					default:
						return `Unknown setting "${key}". Use /settings to list available settings.`;
				}
			},
			getContext: () => {
				return this.bridge.getContext();
			},
			sessions: () => {
				this.openSessionManager();
			},
			newSession: () => {
				this._autoSaveTurn();
				this.currentSessionId = this.sessionStore.createSession({
					title: "New Session",
				});
				this.transcript.clear();
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.statusPanel.update({ sessionTitle: "New Session" });
				setStatusPhase("ready");
			},
			saveSession: () => {
				this._autoSaveTurn();
				this.statusPanel.update({
					phase: "saved",
				});
				this.transcript.addSystemMessage("Session saved.");
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			},
			renameSession: (title: unknown) => {
				if (!this.currentSessionId) return;
				const newTitle =
					typeof title === "string" ? title : String(title || "");
				if (!newTitle.trim()) return;
				this.sessionStore.renameSession(this.currentSessionId, newTitle.trim());
				this.statusPanel.update({ sessionTitle: newTitle.trim() });
				setStatusPhase("ready");
			},
			setPermissionMode: (mode: unknown) => {
				this.bridge.setPermissionMode(
					String(mode) as "acceptAll" | "acceptEdits" | "ask" | "plan",
				);
				setStatusPhase("ready");
			},
			getPermissionMode: () => this.bridge.getPermissionMode(),
			togglePlanMode: () => {
				const next =
					this.bridge.getPermissionMode() === "plan" ? "acceptAll" : "plan";
				this.bridge.setPermissionMode(next);
				this.statusPanel.update({ permissionMode: next });
				return next === "plan"
					? "Plan mode ON — only read-only tools; the agent should present a plan."
					: "Plan mode OFF — permission mode back to acceptAll.";
			},
			rewind: () => {
				const restored = this.bridge.rewind();
				if (restored === null) return "Nothing to rewind.";
				return (
					`Rewound to checkpoint: ${restored.messages} message(s) in ` +
					`context, ${restored.filesRestored} file(s) restored ` +
					"(bash mutations are not captured)."
				);
			},
			fork: () => {
				const id = this.bridge.fork();
				if (!id) return "Fork unavailable.";
				setStatusPhase("ready");
				return `Forked conversation: branch ${id}. Use /branch-summary to merge or /discard-branch to abandon.`;
			},
			branchSummary: async () => {
				const summary = await this.bridge.branchSummary();
				setStatusPhase("ready");
				if (summary === null)
					return "No active branch (or nothing to summarize).";
				return `Branch merged. Summary: ${summary}`;
			},
			discardBranch: () => {
				const discarded = this.bridge.discardBranch();
				setStatusPhase("ready");
				return discarded
					? "Branch discarded — conversation restored to fork point."
					: "No active branch to discard.";
			},
		};

		const slashCommands = createSlashCommands(this.bridge, localHandlers);
		this.slashPopup.setCommands(slashCommands);

		// Wire up slash popup submit to handle quit dispatch
		// eslint-disable-next-line @typescript-eslint/no-misused-promises -- callback body uses await import() for dynamic imports
		this.slashPopup.setOnSubmit(async (result, dispatch, command) => {
			if (dispatch === "quit") {
				void this.stop().then(() => process.exit(0));
				return;
			}
			// Add slash command as user message to transcript
			if (command?.trim()) {
				this.transcript.addTurn(command.trim());
				if (result) {
					this.transcript.addSystemMessage(String(result));
				}
				const cmdName = command.trim().split(/\s+/)[0]?.toLowerCase() || "";
				const args = command.trim().split(/\s+/).slice(1).join(" ");
				const allCmds = this.slashPopup.getCommands() as SlashCommandDef[];
				const match = allCmds?.find(
					(c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
				);
				if (match && match.command === "/plugins") {
					void handlePlugins(args);
				}
				if (match && match.command === "/mcp") {
					void handleMcp(args);
				}
				if (match && match.command === "/reasoner") {
					void handleReasoner(args);
				}
				if (match && match.command === "/theme") {
					void handleTheme(args);
				}
				if (match && match.command === "/compact") {
					void this.bridge.compact().then((result) => {
						if (result === null) {
							this.transcript.addSystemMessage("Nothing to compact.");
						} else {
							this.transcript.addSystemMessage(
								`Context compacted (${formatContextSize(
									result.tokensBefore,
								)} -> ${formatContextSize(
									result.tokensAfter,
								)}). Saved ${formatContextSize(result.tokensSaved)}.`,
							);
						}
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
					});
					return;
				}
				if (match && match.command === "/fork") {
					const id = this.bridge.fork();
					this.transcript.addSystemMessage(
						id ? `Forked conversation (${id}).` : "Nothing to fork.",
					);
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/branch-summary") {
					void this.bridge.branchSummary().then((summary) => {
						this.transcript.addSystemMessage(
							summary === null
								? "No active branch to summarize."
								: `Branch merged: ${summary}`,
						);
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
					});
					return;
				}
				if (match && match.command === "/discard-branch") {
					const ok = this.bridge.discardBranch();
					this.transcript.addSystemMessage(
						ok ? "Branch discarded." : "No active branch.",
					);
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/sessions") {
					this.openSessionManager();
					return;
				}
				if (match && match.command === "/loop") {
					const args = command.trim().split(/\s+/).slice(1).join(" ");
					if (args.toLowerCase() === "stop") {
						this.loopManager.stop();
						this.loopActive = false;
						this.transcript.addSystemMessage("Loop stopped.");
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
						return;
					}
					const intervalMatch = args.match(/^(\d+)(s|m|h|d)\s+(.+)$/);
					if (intervalMatch) {
						const [, value, unit, prompt] = intervalMatch;
						const mult: Record<string, number> = {
							s: 1000,
							m: 60000,
							h: 3600000,
							d: 86400000,
						};
						const ms = parseInt(value, 10) * (mult[unit] ?? 60000);
						this.loopManager.start(prompt, ms);
						this.loopActive = true;
						this.transcript.addSystemMessage(
							`🔄 Loop started: "${prompt}" every ${value}${unit}`,
						);
					} else if (args) {
						// No interval specified — default to 5 minutes
						this.loopManager.start(args, 5 * 60 * 1000);
						this.loopActive = true;
						this.transcript.addSystemMessage(
							`🔄 Loop started: "${args}" (default 5m interval)`,
						);
					} else {
						this.transcript.addSystemMessage(
							"Usage: /loop <prompt> [interval] — e.g. /loop 5m check the deploy\n" +
								"Or: /loop stop",
						);
					}
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/goal") {
					const args = command.trim().split(/\s+/).slice(1).join(" ");
					if (args.toLowerCase() === "clear") {
						this.goalManager.cancel();
						this.goalActive = false;
						this.transcript.addSystemMessage("Goal cleared.");
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
						return;
					}
					if (!args) {
						// Show goal status
						const state = this.goalManager.getState();
						if (!state) {
							this.transcript.addSystemMessage("No goal set.");
						} else if (state.status === "achieved") {
							const dur = Math.round(
								((state.achievedAt ?? Date.now()) - state.startedAt) / 1000,
							);
							this.transcript.addSystemMessage(
								`Goal achieved: "${state.condition}"\n` +
									`Duration: ${dur}s, Turns: ${state.turnCount}, Reason: ${state.lastReason || "N/A"}`,
							);
						} else if (state.status === "cancelled") {
							this.transcript.addSystemMessage(
								`Goal cancelled: "${state.condition}"\n` +
									`Turns: ${state.turnCount}, Reason: ${state.lastReason || "N/A"}`,
							);
						} else {
							const elapsed = Math.round((Date.now() - state.startedAt) / 1000);
							this.transcript.addSystemMessage(
								`Goal active: "${state.condition}"\n` +
									`Running: ${elapsed}s, Turns: ${state.turnCount}${state.maxTurns ? ` / ${state.maxTurns}` : ""}\n` +
									`Last: ${state.lastReason || "evaluating..."}`,
							);
						}
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
						return;
					}
					// Parse condition, extracting optional turn limit
					const parsed = GoalManager.parseCondition(args);
					this.goalManager.set(parsed.condition, parsed.maxTurns);
					this.goalActive = true;
					const info = parsed.maxTurns ? ` (max ${parsed.maxTurns} turns)` : "";
					this.transcript.addSystemMessage(
						`◎ Goal set: "${parsed.condition}"${info}`,
					);
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
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
						const bwrapPath = process.env.PATH?.split(pathMod.delimiter)
							.find((d) => existsSync(pathMod.join(d, "bwrap")))
							? pathMod.join(
								process.env.PATH?.split(pathMod.delimiter).find((d) =>
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

						this.transcript.addSystemMessage(
							`Sandbox availability: ${available ? "OK" : "unavailable"}` +
								`${bwrapPath ? ` (bwrap found at ${bwrapPath})` : " (bwrap not found)"}` +
								`${!isLinux ? ` (not on Linux: ${process.platform})` : ""}` +
								`${bwrapVersion !== "unknown" ? ` — ${bwrapVersion}` : ""}` +
								"\n\nProfiles: none (no isolation), code (read-only host fs, writable /tmp, no network/devices), file (code + bind mounts), dev (code + /dev), full (code + namespaces)",
						);
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
						return;
					}

					// /sandbox profile <level>
					if (sub === "profile") {
						const profiles = {
							none: "No isolation — chroot to tmpdir only",
							code: "Read-only host fs, writable /tmp, no network, no devices",
							file: "CODE + read-only bind-mount of specified directories",
							dev: "CODE + limited /dev (null, zero, random, tty)",
							full: "CODE + user namespace + mount namespace + no new privs",
						};
						if (!subArgs) {
							this.transcript.addSystemMessage(
								"Available profiles:\n" +
									Object.entries(profiles)
										.map(([k, v]) => `  ${k}: ${v}`)
										.join("\n"),
							);
						} else if (profiles[subArgs as keyof typeof profiles]) {
							this.transcript.addSystemMessage(
								`${subArgs}: ${profiles[subArgs as keyof typeof profiles]}`,
							);
						} else {
							this.transcript.addSystemMessage(
								`Unknown profile: ${subArgs}. Use one of: ${Object.keys(profiles).join(", ")}`,
							);
						}
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
						return;
					}

					// /sandbox <command> — dispatch to sandbox tool via bridge
					const cmd = subArgs || args.trim();
					if (!cmd) {
						this.transcript.addSystemMessage(
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

						this.transcript.addSystemMessage(
							`Running in sandbox (profile: ${profileHint}): ${actualCommand}`,
						);
						// Dispatch to the sandbox tool via the bridge
						void this.bridge.sendSlash(
							`/sandbox ${actualCommand}`,
						);
					}
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/new") {
					this._autoSaveTurn();
					this.currentSessionId = this.sessionStore.createSession({
						title: "New Session",
					});
					this.transcript.clear();
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.statusPanel.update({ sessionTitle: "New Session" });
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/save") {
					this._autoSaveTurn();
					this.transcript.addSystemMessage("Session saved.");
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.command === "/rename") {
					if (this.currentSessionId && args.trim()) {
						this.sessionStore.renameSession(this.currentSessionId, args.trim());
						this.statusPanel.update({ sessionTitle: args.trim() });
						this.transcript.addSystemMessage(
							`Session renamed to "${args.trim()}"`,
						);
					}
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender();
					return;
				}
				if (match && match.dispatch === "bridge") {
					void this.bridge.sendSlash(command.trim());
				}
				if (match && match.dispatch === "state") {
					void handleStatus();
				}
			}
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		});
	}

	// ── Bridge setup ─────────────────────────────────────────────────────────

	private setupBridge(): void {
		const eventHandler = (event: ParsedBridgeEvent): void => {
			this.handleEvent(event);
		};

		this.bridge.on(eventHandler);
		this.bridge.onError((err) => {
			// eslint-disable-next-line no-console
			console.error(`Bridge error: ${err.message}`);
			// Also display in transcript so the user sees connection/server errors
			this.transcript.addSystemMessage(`Connection error: ${err.message}`);
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		});

		// Initialize bridge
		this.bridge
			.init()
			.then((state) => {
				this.statusPanel.update({
					contextTokens: Number(state.context_tokens || 0),
					contextMaxTokens: Number(state.context_max_tokens || 0) || undefined,
					sandboxMode: this.bridge.getSandboxMode(),
				});
				const message = this.formatStartupMessage(state);
				if (message) {
					this.transcript.addSystemMessage(message);
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.tui.requestRender(true);
				}
				// Surface discovered skills as /<skill-name> commands in the popup.
				const skills = this.bridge.getSkills();
				if (skills.length) {
					const existing = this.slashPopup.getCommands() as SlashCommandDef[];
					const taken = new Set(existing.map((c) => c.command));
					const skillCmds: SlashCommandDef[] = skills
						.map((s) => ({
							command: `/${s.slashName}`,
							usage: `/${s.slashName}${s.argumentHint ? ` ${s.argumentHint}` : ""}`,
							description: `Skill: ${s.description.slice(0, 80)}`,
							dispatch: "local" as const,
							acceptsArgs: true,
							bridgeHandler: (args: string) => {
								this.bridge.invokeSkill(s.name, args);
							},
						}))
						.filter((c) => !taken.has(c.command));
					if (skillCmds.length) {
						this.slashPopup.setCommands([...existing, ...skillCmds]);
					}
				}
			})
			.catch((err) => {
				// eslint-disable-next-line no-console
				console.error(`Bridge init failed: ${err.message}`);
				// Display init/connection errors in transcript so the user knows
				// the agent couldn't start (e.g. server unreachable).
				this.transcript.addSystemMessage(
					`Failed to start agent: ${err.message}`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
			});
	}

	private handleEvent(event: ParsedBridgeEvent): void {
		// Update transcript state
		this.transcript.handleEvent(event);
		this.turnState = reduceTurnState(this.turnState, event);
		this.workSurface.setPhase(this.turnState.phase);
		this.statusPanel.update({ phase: turnPhaseLabel(this.turnState.phase) });
		if (turnPhaseIsActive(this.turnState.phase)) {
			this.statusPanel.startAnimation();
			this.transcriptDisplay.startAnimation();
		} else {
			this.statusPanel.stopAnimation();
			this.transcriptDisplay.stopAnimation();
		}

		switch (event.type) {
			case "todos":
				this.todoBar.setTodos(event.todos);
				this.tui.requestRender();
				break;
			case "queue_update":
				this.steerQueue.setItems(event.steering || [], event.followUp || []);
				this.tui.requestRender();
				break;
			case "permission_request": {
				this.pendingPermission = {
					toolCallId: event.tool_call_id,
					toolName: event.tool_name,
				};
				const preview = JSON.stringify(event.args ?? {}).slice(0, 500);
				this.permissionPopup.setToolInfo(event.tool_name, preview);
				this.permissionPopup.setChoices([
					{ value: "allow", label: "Allow once", description: "Run this tool for this call only" },
					{ value: "always", label: "Always allow", description: `Allow ${event.tool_name} without asking` },
					{ value: "deny", label: "Deny", description: "Block this tool call" },
				]);
				this.permissionPopup.show();
				const overlay = this.tui.showOverlay(this.permissionPopup, {
					anchor: "aboveInput",
					align: "left",
					maxHeight: 14,
				});
				overlay.focus();
				this.statusPanel.update({ phase: "permission" });
				this.statusPanel.stopAnimation();
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
				break;
			}
			case "question_request": {
				this.choicePopup.setQuestionId(event.question_id);
				this.choicePopup.setQuestion(event.question);
				this.choicePopup.setChoices(event.choices);
				this.choicePopup.show();
				const overlay = this.tui.showOverlay(this.choicePopup, {
					anchor: "aboveInput",
					align: "left",
					maxHeight: 18,
				});
				overlay.focus();
				this.tui.requestRender();
				break;
			}
			case "token":
				break;
			case "tool_start":
			case "tool_execution_start":
				this.workSurface.recordToolStart(
					event.tool_call_id,
					event.tool_name || event.tool,
					event.tool_args,
				);
				break;
			case "tool_end":
			case "tool_execution_end":
				this.workSurface.recordToolEnd(
					event.tool_call_id,
					event.tool_name || event.tool,
					event.result,
					event.is_error,
				);
				break;
			case "turn_end":
				// Auto-save the completed turn
				this._autoSaveTurn();
				this.statusPanel.update({
					turnCount: this.transcript.getTurns().length,
					messageCount: this.transcript.getMessageCount(),
				});
				// Goal evaluation: if a goal is active, evaluate after each turn
				if (this.goalActive && this.goalManager.isSet()) {
					const goalState = this.goalManager.getState();
					if (goalState && goalState.status === "active") {
						void this.evaluateGoal(goalState);
					}
				}
				break;
			case "turn_start":
				this.workSurface.startTurn();
				break;
			case "phase":
				if (event.state === "ready") {
					this.statusPanel.update({
						turnCount: this.transcript.getTurns().length,
						messageCount: this.transcript.getMessageCount(),
					});
				}
				break;
			case "context_update":
				this.workSurface.setContext(
					Number(event.tokens || 0),
					Number(event.max_tokens || 0) || undefined,
				);
				this.statusPanel.update({
					contextTokens: Number(event.tokens || 0),
					contextMaxTokens: Number(event.max_tokens || 0) || undefined,
					contextCompacted: event.compacted === true,
					...("cached_tokens" in event && {
						cacheReadTokens:
							typeof event.cached_tokens === "number"
								? event.cached_tokens
								: undefined,
					}),
				});
				break;
			case "compaction":
				this.transcript.addSystemMessage(
					`Context compacted (${formatContextSize(
						Number(event.tokens_before || 0),
					)} -> ${formatContextSize(Number(event.tokens_after || 0))}).`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.statusPanel.update({
					phase: "compacted",
					contextTokens: Number(event.tokens_after || 0),
					contextCompacted: true,
				});
				break;
			case "model_select":
				this.statusPanel.update({ model: event.model });
				break;
			case "notice":
				if (event.label === "MCP") {
					void this.bridge.getState().then((state) => {
						this.statusPanel.update({
							mcpServerCount: Number(state.mcp_servers || 0),
						});
					});
				}
				break;
			case "repair_nudge":
				this.transcript.addSystemMessage(
					`Tool-call repair: ${event.message || "recovered malformed tool call"}`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				break;
			case "memory_update": {
				if (event.kind === "observations_added") {
					const previews = (event.items ?? [])
						.slice(0, 3)
						.map((item) => `[${item.id}] ${item.content.slice(0, 120)}`)
						.join("\n");
					this.transcript.addSystemMessage(
						`Memory added: ${event.count} observation${event.count === 1 ? "" : "s"}` +
							(previews ? `\n${previews}` : ""),
					);
				} else if (event.kind === "reflections_added") {
					this.transcript.addSystemMessage(
						`Memory synthesized: ${event.count} reflection${event.count === 1 ? "" : "s"}`,
					);
				} else if (event.kind === "observations_dropped") {
					this.transcript.addSystemMessage(
						`Memory compacted: ${event.count} observation${event.count === 1 ? "" : "s"} archived.`,
					);
				}
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				break;
			}
			case "steered":
				// Steering is part of the active run rather than a new turn, but it must
				// remain visible after the transient queue widget drains. Otherwise a
				// successfully queued user message looks as though it was discarded.
				this.transcript.addSystemMessage(
					`You steered the active turn:\n${String(event.message || "")}`,
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				break;
		}

		this.tui.requestRender();
	}

	private formatStartupMessage(state: Record<string, unknown>): string {
		const pluginCount = Number(state.startup_plugins_loaded || 0);
		const hookCount = Number(state.startup_hooks_loaded || 0);
		const mcpServerCount = Number(state.mcp_servers_loaded || 0);
		const mcpToolCount = Number(state.mcp_tools_loaded || 0);
		const plugins = Array.isArray(state.startup_plugins)
			? state.startup_plugins
					.map((item) => String(item || "").trim())
					.filter(Boolean)
			: [];
		const skills = Array.isArray(state.loaded_skills)
			? state.loaded_skills
					.map((item) => {
						if (!item || typeof item !== "object") return null;
						const skill = item as Record<string, unknown>;
						const slashName = String(
							skill.slash_name || skill.name || "",
						).trim();
						const description = String(skill.description || "").trim();
						return slashName ? { slashName, description } : null;
					})
					.filter(
						(item): item is { slashName: string; description: string } =>
							item !== null,
					)
			: [];
		const contexts = Array.isArray(state.startup_hook_contexts)
			? state.startup_hook_contexts
					.map((item) => String(item || "").trim())
					.filter(Boolean)
			: [];
		const hookMessages = Array.isArray(state.startup_hook_messages)
			? state.startup_hook_messages
					.map(normalizeStartupHookMessage)
					.filter((item) => item.content)
			: [];
		const initialMessage = String(
			state.startup_hook_initial_message || "",
		).trim();
		const errors = Array.isArray(state.startup_hook_errors)
			? state.startup_hook_errors
					.map((item) => String(item || "").trim())
					.filter(Boolean)
			: [];
		const mcpErrors = Array.isArray(state.mcp_errors)
			? state.mcp_errors
					.map((item) => String(item || "").trim())
					.filter(Boolean)
			: [];

		const dim = "\x1b[2m";
		const reset = "\x1b[0m";
		const model = String(state.model || "unknown");
		const agent = String(state.agent_name || "logician");
		const project = getGitVersion() || "-";
		const mcpState = state.mcp_loading
			? "MCP loading"
			: state.mcp_deferred
				? "MCP deferred"
				: `MCP ${mcpServerCount}/${mcpToolCount}`;
		const searchUrl = String(state.web_search_url || "");
		const searchEnabled =
			state.web_search_enabled === true ||
			(Array.isArray(state.tools) && state.tools.includes("web_search"));
		const searchState = searchEnabled ? searchUrl || "enabled" : "disabled";

		const lines = [
			"# Logician",
			`${agent} is ready · ${model}`,
			"",
			`${dim}${project} · theme ${theme.name} · plugins ${pluginCount} · skills ${skills.length} · hooks ${hookCount} · ${mcpState}${reset}`,
			`${dim}web search ${searchState}${reset}`,
			`${dim}config ${this.configPath || "-"} · ${String(state.base_url || "unknown")}${reset}`,
		];

		if (initialMessage) {
			lines.push("", "## Startup message", initialMessage);
		}

		if (contexts.length) {
			lines.push("", "## Plugin startup messages");
			if (hookMessages.length) {
				hookMessages.forEach((message) => {
					lines.push("", `### ${message.title}`, message.content);
				});
			} else {
				contexts.forEach((context, idx) => {
					lines.push("", `### Startup hook ${idx + 1}`, context);
				});
			}
		}

		if (errors.length) {
			lines.push(
				"",
				"## Startup hook errors",
				...errors.map((err) => `- ${err}`),
			);
		}

		if (mcpErrors.length) {
			lines.push("", "## MCP errors", ...mcpErrors.map((err) => `- ${err}`));
		}

		lines.push(...formatStartupMemory(state));

		// Keep the inventory last so the transcript's initial scroll position shows
		// what was loaded instead of ending on verbose startup-hook context.
		lines.push("", "## Loaded resources");
		lines.push(
			`${dim}Plugins (${plugins.length})${reset}`,
			plugins.length ? plugins.join(" · ") : `${dim}None${reset}`,
			"",
			`${dim}Skills (${skills.length})${reset}`,
			skills.length
				? skills.map((skill) => `/${skill.slashName}`).join(" · ")
				: `${dim}None${reset}`,
		);

		return lines.join("\n");
	}

	// ── Transcript setup ─────────────────────────────────────────────────────

	private setupTranscript(): void {
		this.transcript.onChange(() => {
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.transcriptDisplay.setThinkingMode(
				this.transcript.getThinkingDisplayMode(),
			);
			// Steer queue is driven directly by queue_update events (see
			// handleBridgeEvent), not transcript state.
			// Auto-scroll to bottom only when already at bottom
			if (this.tui.isAtBottom) {
				this.transcriptDisplay.scrollToBottom();
				this.tui.scrollToBottom();
			}
			this.tui.requestRender();
		});
	}

	// ── Input handling ─────────────────────────────────────────────────────

	private setupInputHandler(): void {
		// ── Choice popup handlers ──────────────────────────────────────
		const handleChoicePopupSubmit = (): void => {
			const qid = this.choicePopup.getQuestionId();
			const selected = this.choicePopup.getSelected();
			if (
				qid &&
				selected &&
				this.bridge.respondToQuestion(qid, selected.value)
			) {
				this.transcript.addSystemMessage(
					`Question answered: ${selected.label}`,
				);
			}
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		};

		const handleChoicePopupDismiss = (): void => {
			const qid = this.choicePopup.getQuestionId();
			if (qid) {
				this.bridge.respondToQuestion(qid, "__dismissed__");
				this.transcript.addSystemMessage("Question dismissed.");
			}
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		};

		// Global input listener
		this.tui.addInputListener((data: string) => {
			if (this.pluginManager.isVisibleOverlay()) {
				const action = this.pluginManager.handleInput(data);
				if (action) {
					this.handlePluginManagerAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}
			if (this.mcpManager.isVisibleOverlay()) {
				const action = this.mcpManager.handleInput(data);
				if (action) {
					this.handleMcpManagerAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}
			if (this.reasonerSelector.isVisibleOverlay()) {
				const action = this.reasonerSelector.handleInput(data);
				if (action) {
					this.handleReasonerSelectorAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}
			if (this.modelSelector.isVisibleOverlay()) {
				const action = this.modelSelector.handleInput(data);
				if (action) {
					this.handleModelSelectorAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}
			if (this.themeSelector.isVisibleOverlay()) {
				const action = this.themeSelector.handleInput(data);
				if (action) {
					this.handleThemeSelectorAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}
			if (this.settingsSelector.isVisibleOverlay()) {
				const action = this.settingsSelector.handleInput(data);
				if (action) {
					this.handleSettingsSelectorAction(action);
				}
				this.tui.requestRender();
				return { consume: true };
			}

			// ChoicePopup — agent Q&A popup
			if (this.choicePopup.isVisibleOverlay()) {
				const action = this.choicePopup.handleInput(data);
				if (action) {
					if (action.type === "select") {
						handleChoicePopupSubmit();
					} else {
						handleChoicePopupDismiss();
					}
					this.tui.removeOverlay(this.choicePopup);
				}
				this.tui.requestRender();
				return { consume: true };
			}

			// PermissionPopup — tool permission overlay
			if (this.permissionPopup.isVisibleOverlay()) {
				const action = this.permissionPopup.handleInput(data);
				if (action) {
					if (action.type === "close") {
						this.pendingPermission = null;
						this.transcript.addSystemMessage("Permission request dismissed.");
					} else {
						this.bridge.respondToPermission(
							this.pendingPermission?.toolCallId ?? "",
							action.choice.value,
						);
						this.transcript.addSystemMessage(
							`Permission ${action.choice.value}: ${this.pendingPermission?.toolName ?? "unknown"}`,
						);
					}
					this.pendingPermission = null;
					this.permissionPopup.hide();
					this.tui.removeOverlay(this.permissionPopup);
					if (action.type !== "close") {
						this.statusPanel.update({ phase: "streaming" });
					} else {
						this.statusPanel.update({ phase: "ready" });
					}
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
				}
				this.tui.requestRender();
				return { consume: true };
			}

			// Inline @-mention autocomplete: same pattern as the slash popup below —
			// the input bar keeps focus, we only intercept nav/accept keys.
			if (this.fileMentionPopup.isVisibleOverlay()) {
				if (data === "\x1b[A" || data === "\x1bOA") {
					this.fileMentionPopup.moveSelection(-1);
					this.tui.requestRender();
					return { consume: true };
				}
				if (data === "\x1b[B" || data === "\x1bOB") {
					this.fileMentionPopup.moveSelection(1);
					this.tui.requestRender();
					return { consume: true };
				}
				if (data === "\t" || data === "\r" || data === "\n") {
					const file = this.fileMentionPopup.currentFile();
					if (file) {
						this.inputBar.insertMention(file);
					}
					this.fileMentionPopup.hide();
					this.tui.requestRender();
					return { consume: true };
				}
				if (data === "\x1b") {
					this.fileMentionPopup.hide();
					this.tui.requestRender();
					return { consume: true };
				}
				// Everything else (typing, backspace, etc.) goes to the input bar; the
				// onChange hook re-syncs the popup query afterwards.
			}

			// Inline slash autocomplete: while the popup is showing matches, the
			// input bar keeps focus and ordinary typing flows through to it. We only
			// intercept the navigation/accept keys here.
			if (this.slashPopup.isVisibleOverlay()) {
				// Up / Down — move highlight
				if (data === "\x1b[A" || data === "\x1bOA") {
					this.slashPopup.moveSelection(-1);
					this.tui.requestRender();
					return { consume: true };
				}
				if (data === "\x1b[B" || data === "\x1bOB") {
					this.slashPopup.moveSelection(1);
					this.tui.requestRender();
					return { consume: true };
				}
				// Tab — complete input to the highlighted command
				if (data === "\t") {
					const cmd = this.slashPopup.currentCommand();
					if (cmd) {
						this.inputBar.valueText = `${cmd} `;
						this.tui.requestRender();
					}
					return { consume: true };
				}
				// Escape — dismiss the menu, clear/arm the composer, and stop an
				// active loop. A following Escape cancels the active model turn.
				if (data === "\x1b") {
					this.slashPopup.hide();
					// Let the composer consume the first Escape too: it clears the
					// slash draft and arms the second Escape for turn cancellation.
					this.inputBar.handleInput(data);
					if (this.loopActive) {
						this.loopManager.stop();
						this.loopActive = false;
						this.transcript.addSystemMessage("Loop stopped (Esc).");
						this.transcriptDisplay.setTurns(this.transcript.getTurns());
						this.tui.requestRender();
					}
					this.tui.requestRender();
					return { consume: true };
				}
				// Enter — accept highlighted command (submit it directly)
				if (data === "\r" || data === "\n") {
					const cmd = this.slashPopup.currentCommand();
					if (cmd && this.inputBar.valueText.trim() !== cmd) {
						// If the typed text isn't already an exact command, accept the
						// highlighted one (carrying over any args the user typed).
						const typedArgs = this.inputBar.valueText.replace(/^\/\S*\s*/, "");
						this.inputBar.valueText = typedArgs ? `${cmd} ${typedArgs}` : cmd;
					}
					this.slashPopup.hide();
					// Fall through to the input bar so it submits the value.
					return { consume: false };
				}
				// Everything else (typing, backspace, etc.) goes to the input bar; the
				// onChange hook re-syncs the popup query afterwards.
			}

			// Ctrl+L — open model selector
			if (data === "\x0c") {
				this.openModelSelector();
				return { consume: true };
			}

			// Ctrl+O — expand/collapse tool execution details
			if (data === "\x0f") {
				const expanded = this.transcriptDisplay.toggleToolsExpanded();
				this.statusPanel.update({
					phase: expanded ? "tools expanded" : "tools collapsed",
				});
				this.tui.requestRender();
				return { consume: true };
			}

			// Ctrl+Shift+T — cycle thinking display mode
			if (data === "\x14") {
				this.transcript.cycleThinkingDisplayMode();
				this.transcriptDisplay.setThinkingMode(
					this.transcript.getThinkingDisplayMode(),
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
				return { consume: true };
			}

			// Ctrl+S — open session manager
			if (data === "\x13") {
				this.openSessionManager();
				return { consume: true };
			}

			// Ctrl+K — cycle sandbox mode (off / code / full)
			if (data === "\x0b") {
				const mode = this.bridge.cycleSandboxMode();
				this.statusPanel.update({ sandboxMode: mode });
				this.tui.requestRender();
				return { consume: true };
			}

			// Ctrl+P — toggle plan mode (plan ↔ act)
			if (data === "\x10") {
				const next =
					this.bridge.getPermissionMode() === "plan" ? "acceptAll" : "plan";
				this.bridge.setPermissionMode(next);
				this.statusPanel.update({ permissionMode: next });
				this.tui.requestRender();
				return { consume: true };
			}

			// Ctrl+M requires an enhanced keyboard protocol because legacy
			// terminals encode it exactly like Enter. TUI.start() requests CSI-u;
			// Alt+M remains the portable fallback for terminals that ignore it.
			if (
				data === "\x1bm" ||
				data === "\x1bM" ||
				data === "\x1b[109;5u" ||
				data === "\x1b[109;6u" ||
				data === "\x1b[13;5u"
			) {
				this.cycleInferenceMode();
				return { consume: true };
			}

			// Ctrl+Backspace in input bar is handled by InputBar directly
			return { consume: false };
		});

		// Live slash autocomplete: show/hide + filter the popup as the input text
		// changes. The popup only appears while the line begins with "/" and has no
		// space yet (i.e. the user is still picking a command, not typing args).
		this.inputBar.onChange = (text: string) => {
			const isCommandPrefix = text.startsWith("/") && !text.includes(" ");
			if (isCommandPrefix) {
				this.slashPopup.setQuery(text);
				if (this.slashPopup.hasMatches()) {
					if (!this.slashPopup.isVisibleOverlay()) this.slashPopup.show();
				} else {
					this.slashPopup.hide();
				}
			} else if (this.slashPopup.isVisibleOverlay()) {
				this.slashPopup.hide();
			}

			const mentionQuery = this.inputBar.getActiveMentionQuery();
			if (mentionQuery !== null) {
				void this.updateFileMentionPopup(mentionQuery);
			} else if (this.fileMentionPopup.isVisibleOverlay()) {
				this.fileMentionPopup.hide();
			}

			this.tui.requestRender();
		};

		// Input bar handler
		this.inputBar.onSubmit = (text: string) => {
			// A pending permission request captures the next submission:
			// y/a/n (or allow/always/deny) answers it instead of becoming a message.
			if (this.pendingPermission) {
				const answer = text.trim().toLowerCase();
				const decision =
					answer === "y" || answer === "yes" || answer === "allow"
						? "allow"
						: answer === "a" || answer === "always"
							? "always"
							: "deny";
				this.bridge.respondToPermission(
					this.pendingPermission.toolCallId,
					decision,
				);
				this.transcript.addSystemMessage(
					`Permission ${decision}: ${this.pendingPermission.toolName}`,
				);
				this.pendingPermission = null;
				this.statusPanel.update({ phase: "streaming" });
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
				return;
			}

			// Always push to history (both slash and regular messages)
			this.inputBar.pushHistory(text);

			// Check for slash commands
			if (text.startsWith("/")) {
				const parts = text.trim().split(/\s+/);
				const cmdName = parts[0].toLowerCase();
				const args = parts.slice(1).join(" ");
				const allCmds = this.slashPopup.getCommands() as SlashCommandDef[];
				const match = allCmds?.find(
					(c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
				);

				if (match) {
					this.slashPopup.submitRaw(text.trim());
					return;
				}

				// Unknown command — a skill invocation? (/<skill-name> args)
				if (this.bridge.invokeSkill(cmdName.slice(1), args)) {
					this.transcript.addTurn(text.trim());
					this.transcriptDisplay.setTurns(this.transcript.getTurns());
					this.statusPanel.update({ phase: "streaming" });
					this.statusPanel.startAnimation();
					this.tui.requestRender();
					return;
				}

				// Unknown command — do not silently turn a typo into an agent prompt.
				this.transcript.addTurn(text.trim());
				const suggestions = filterSlashCommands(allCmds, cmdName, 3).map(
					(command) => command.command,
				);
				this.transcript.addSystemMessage(
					`Unknown command: ${cmdName}.` +
						(suggestions.length > 0
							? ` Did you mean ${suggestions.join(", ")}?`
							: "") +
						" Use /help to list available commands.",
				);
				this.transcriptDisplay.setTurns(this.transcript.getTurns());
				this.tui.requestRender();
				return;
			}

			// While a turn is running, a plain message steers it instead of
			// starting a new run. The bridge emits a `steered` event that
			// renders the message, so skip the normal turn/animation setup.
			if (this.bridge.isActive()) {
				this.bridge
					.sendMessage(text)
					.catch((err) => this.bridge.reportError(err));
				return;
			}

			this.transcript.addTurn(text);
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
			this.bridge
				.sendMessage(text)
				.catch((err) => this.bridge.reportError(err));
			this.statusPanel.update({ phase: "streaming" });
			this.statusPanel.startAnimation();
		};

		this.inputBar.onCancel = () => {
			this.pendingPermission = null;
			this.bridge.cancel();
			this.statusPanel.update({ phase: "ready" });
		};
	}

	// ── Session management ───────────────────────────────────────────────

	/**
	 * Restore a session into BOTH the UI transcript and the model context.
	 * Without the bridge restore, a resumed session renders its history but the
	 * model starts cold ("continue" loses everything). Pass [] for a fresh
	 * session (clears both).
	 */
	private restoreSession(turns: Turn[]): void {
		this.transcript.loadTurns(turns);
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.bridge.restoreHistory(turnsToMessages(turns));
	}

	/** Auto-save the latest turn to the current session. */
	private _autoSaveTurn(): void {
		if (!this.currentSessionId) return;
		const turns = this.transcript.getTurns();
		const latestTurn = turns[turns.length - 1];
		if (latestTurn && latestTurn.isComplete) {
			this.sessionStore.saveTurn(latestTurn);
			// Update the session title from the first user message
			if (latestTurn.userMessage.content.length > 0) {
				const title = latestTurn.userMessage.content
					.replace(/\n+/g, " ")
					.slice(0, 60);
				const current = this.sessionStore.getSession(this.currentSessionId);
				if (current && current.title === "Untitled Session") {
					this.sessionStore.renameSession(this.currentSessionId, title);
				}
			}
		}
	}

	/** Handle session manager actions (select, rename, delete, new). */
	private handleSessionAction(action: {
		type: "close" | "select" | "rename" | "delete" | "new";
		sessionId?: string;
		title?: string;
	}): void {
		switch (action.type) {
			case "close":
				this.tui.removeOverlay(this.sessionManager);
				break;

			case "select": {
				if (!action.sessionId) return;
				const session = this.sessionStore.getSession(action.sessionId);
				if (!session) return;

				// Save current session
				this._autoSaveTurn();

				// Load new session turns into transcript + model context
				const turns = this.sessionStore.loadTurns(action.sessionId);
				this.currentSessionId = action.sessionId;
				this.restoreSession(turns);
				this.statusPanel.update({
					sessionTitle: session.title,
					turnCount: turns.length,
				});
				this.tui.removeOverlay(this.sessionManager);
				this.tui.requestRender();
				break;
			}

			case "rename":
				if (!action.sessionId || !action.title) return;
				this.sessionStore.renameSession(action.sessionId, action.title);
				this.tui.removeOverlay(this.sessionManager);
				this.tui.requestRender();
				break;

			case "delete":
				if (!action.sessionId) return;
				this.sessionStore.deleteSession(action.sessionId);
				if (this.currentSessionId === action.sessionId) {
					// Switch to the next most recent session or create new
					const remaining = this.sessionStore.listSessions();
					if (remaining.length > 0) {
						this.currentSessionId = remaining[0].id;
						const turns = this.sessionStore.loadTurns(this.currentSessionId);
						this.restoreSession(turns);
						this.statusPanel.update({ sessionTitle: remaining[0].title });
					} else {
						this.currentSessionId = this.sessionStore.createSession({
							title: "New Session",
						});
						this.restoreSession([]);
					}
				}
				this.tui.removeOverlay(this.sessionManager);
				this.tui.requestRender();
				break;

			case "new":
				this._autoSaveTurn();
				this.currentSessionId = this.sessionStore.createSession({
					title: "New Session",
				});
				this.restoreSession([]);
				this.statusPanel.update({ sessionTitle: "New Session" });
				this.tui.removeOverlay(this.sessionManager);
				this.tui.requestRender();
				break;
		}
	}

	/** Open the session manager overlay. */
	private openSessionManager(): void {
		this.statusPanel.update({ phase: "sessions" });
		this.sessionManager.refresh();
		this.sessionManager.show();
		const overlay = this.tui.showOverlay(this.sessionManager, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		overlay.focus();
	}

	// ── Layout ─────────────────────────────────────────────────────────────

	private buildLayout(): void {
		// Fixed layout: transcript (scrollable, top) + separator + input bar (fixed) + status bar (fixed, bottom)
		this.tui.setInputBarComponent(this.inputBar);
		this.tui.setScrollableComponent(this.transcriptDisplay);
		this.tui.setFixedBottomComponent(this.statusPanel);

		// Stack todo bar + steer queue + question handler above the input bar
		// (both render empty when there's nothing to show, so they only take
		// space when active).
		const pinnedContainer = new Container();
		pinnedContainer.addChild(this.todoBar);
		pinnedContainer.addChild(this.workSurface);
		pinnedContainer.addChild(this.steerQueue);
		this.tui.setFixedAboveInputComponent(pinnedContainer);

		// Interactive pickers join the fixed composer stack. They consume layout
		// space above the input like the TODO/queue region instead of floating
		// over transcript content.
		this.tui.showOverlay(this.slashPopup, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 12,
		});
		this.tui.showOverlay(this.fileMentionPopup, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 12,
		});
		this.tui.showOverlay(this.pluginManager, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		this.tui.showOverlay(this.mcpManager, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
	}

	private async openPluginManager(): Promise<void> {
		this.statusPanel.update({ phase: "plugins" });
		try {
			const snapshot = await this.bridge.getPluginSnapshot();
			this.pluginManager.setSnapshot({
				pluginsDir: String(snapshot.plugins_dir || ""),
				plugins: snapshot.plugins || [],
				sessionStartHooks: snapshot.session_start_hooks || {},
			});
			this.pluginManager.setMessage(
				"Space toggles enabled state in the Claude plugin registry.",
			);
			this.pluginManager.show();
		} catch (e: unknown) {
			this.transcript.addSystemMessage(
				`Plugins error: ${e instanceof Error ? e.message : String(e)}`,
			);
		} finally {
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		}
	}

	private handlePluginManagerAction(action: PluginManagerAction): void {
		if (action.type === "close") {
			// The plugin manager is registered once in buildLayout().
			this.pluginManager.hide();
			return;
		}
		if (action.type === "refresh") {
			void this.openPluginManager();
			return;
		}

		const plugin = action.plugin;
		const nextEnabled = !plugin.enabled;
		this.pluginManager.setBusy(plugin.pluginId);
		this.pluginManager.setMessage(
			`${nextEnabled ? "Enabling" : "Disabling"} ${plugin.pluginId}...`,
		);
		this.tui.requestRender();
		void this.bridge
			.setPluginEnabled(plugin.pluginId, nextEnabled)
			.then(async (result) => {
				this.pluginManager.setMessage(
					String(result.message || `${plugin.pluginId} updated.`),
				);
				const snapshot = await this.bridge.getPluginSnapshot();
				this.pluginManager.setSnapshot({
					pluginsDir: String(snapshot.plugins_dir || ""),
					plugins: snapshot.plugins || [],
					sessionStartHooks: snapshot.session_start_hooks || {},
				});
			})
			.catch((e: unknown) => {
				this.pluginManager.setMessage(
					`Plugin update failed: ${e instanceof Error ? e.message : String(e)}`,
				);
			})
			.finally(() => {
				this.pluginManager.setBusy(null);
				this.statusPanel.update({ phase: "ready" });
				this.tui.requestRender();
			});
	}

	// ── MCP manager ───────────────────────────────────────────────────────

	private async openMcpManager(): Promise<void> {
		this.statusPanel.update({ phase: "mcp" });
		try {
			const snapshot = await this.bridge.getMcpSnapshot();
			this.mcpManager.setSnapshot({
				configPath: snapshot.configPath,
				servers: snapshot.servers.map((s) => ({
					server_name: s.serverName,
					server: s.server,
					url: s.server.url || "",
					command: s.server.command || "",
					type: s.server.type || (s.server.url ? "http" : "stdio"),
					enabled: s.enabled,
				})),
				loadedServers: snapshot.loadedServers,
				errors: snapshot.errors,
			});
			this.mcpManager.setMessage(
				"Space toggles enabled state in the MCP config file.",
			);
			this.mcpManager.show();
		} catch (e: unknown) {
			this.transcript.addSystemMessage(
				`MCP error: ${e instanceof Error ? e.message : String(e)}`,
			);
		} finally {
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
		}
	}

	private handleMcpManagerAction(action: McpManagerAction): void {
		if (action.type === "close") {
			// The MCP manager is registered once in buildLayout(). Keep it in the
			// overlay stack so a later `/mcp list` can show the same component.
			this.mcpManager.hide();
			return;
		}
		if (action.type === "refresh") {
			void this.openMcpManager();
			return;
		}

		const server = action.server;
		const nextEnabled = !server.enabled;
		this.mcpManager.setBusy(server.serverName);
		this.mcpManager.setMessage(
			`${nextEnabled ? "Enabling" : "Disabling"} ${server.serverName}...`,
		);
		this.tui.requestRender();
		void this.bridge
			.setMcpServerEnabled(server.serverName, nextEnabled)
			.then(async (result) => {
				this.mcpManager.setMessage(result.message);
				const snapshot = await this.bridge.getMcpSnapshot();
				this.mcpManager.setSnapshot({
					configPath: snapshot.configPath,
					servers: snapshot.servers.map((s) => ({
						server_name: s.serverName,
						server: s.server,
						url: s.server.url || "",
						command: s.server.command || "",
						type: s.server.type || (s.server.url ? "http" : "stdio"),
						enabled: s.enabled,
					})),
					loadedServers: snapshot.loadedServers,
					errors: snapshot.errors,
				});
			})
			.catch((e: unknown) => {
				this.mcpManager.setMessage(
					`MCP update failed: ${e instanceof Error ? e.message : String(e)}`,
				);
			})
			.finally(() => {
				this.mcpManager.setBusy(null);
				this.statusPanel.update({ phase: "ready" });
				this.tui.requestRender();
			});
	}

	// ── Reasoner selector ───────────────────────────────────────────────────

	private async openReasonerSelector(): Promise<void> {
		this.statusPanel.update({ phase: "reasoner" });
		const currentId = "none";
		const reasoners: ReasonerInfo[] = getReasonerIds().map((id) => {
			const meta = getReasonerMeta(id) as ReasonerMeta;
			return {
				id,
				name: meta.name,
				description: meta.description,
				active: id === currentId,
			};
		});
		this.reasonerSelector.setReasoners(reasoners);
		this.reasonerSelector.setMessage(
			"Enter selects reasoning mode for the next turn.",
		);
		this.reasonerSelector.show();
		const overlay = this.tui.showOverlay(this.reasonerSelector, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		overlay.focus();
	}

	private handleReasonerSelectorAction(action: ReasonerSelectorAction): void {
		if (action.type === "close") {
			this.tui.removeOverlay(this.reasonerSelector);
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			return;
		}
		const reasoner = action.reasoner;
		this.reasonerSelector.setMessage(`Setting: ${reasoner.name}...`);
		this.tui.requestRender();
		// reasoner removed;
		this.tui.removeOverlay(this.reasonerSelector);
		this.statusPanel.update({ phase: "ready" });
		this.transcript.addSystemMessage(`Reasoning mode: ${reasoner.name}`);
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
	}

	// ── File mention autocomplete ────────────────────────────────────────

	private fileMentionListedCwd: string | null = null;
	private fileMentionListing: Promise<string[]> | null = null;

	private async updateFileMentionPopup(query: string): Promise<void> {
		const cwd = process.cwd();
		if (this.fileMentionListedCwd !== cwd || !this.fileMentionListing) {
			this.fileMentionListedCwd = cwd;
			this.fileMentionListing = listProjectFiles(cwd);
		}
		const files = await this.fileMentionListing;

		// The user may have kept typing (or dismissed the mention) while the
		// listing was in flight; only apply this result if still relevant.
		if (this.inputBar.getActiveMentionQuery() !== query) return;

		this.fileMentionPopup.setFiles(files);
		this.fileMentionPopup.setQuery(query);
		if (this.fileMentionPopup.hasMatches()) {
			if (!this.fileMentionPopup.isVisibleOverlay()) this.fileMentionPopup.show();
		} else {
			this.fileMentionPopup.hide();
		}
		this.tui.requestRender();
	}

	// ── Model selector ───────────────────────────────────────────────────

	private openModelSelector(): void {
		this.statusPanel.update({ phase: "model" });
		const modelInfos: ModelInfo[] = this.bridge
			.getModelOptions()
			.map((option) => ({
				id: option.key,
				name: option.name,
				active: option.active,
				url: `${option.model} · ${option.url}`,
			}));
		this.modelSelector.setModels(modelInfos);
		this.modelSelector.setMessage(
			"Enter selects model for the current session.",
		);
		this.modelSelector.show();
		const overlay = this.tui.showOverlay(this.modelSelector, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		overlay.focus();
	}

	private handleModelSelectorAction(action: ModelSelectorAction): void {
		if (action.type === "close") {
			this.tui.removeOverlay(this.modelSelector);
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			return;
		}
		const selected = action.model;
		this.modelSelector.setMessage(`Switching to ${selected.name}...`);
		this.tui.requestRender();
		// Switch the model via the bridge (handles url switching too)
		const applied = this.bridge.setModelOption(selected.id);
		if (!applied) return;
		// Save to global settings
		saveConfigField("model", applied.model);
		saveConfigField("baseUrl", applied.url);
		// Update status
		this.tui.removeOverlay(this.modelSelector);
		this.statusPanel.update({ phase: "ready", model: applied.model });
		this.transcript.addSystemMessage(`Switched model: ${selected.name}`);
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
	}

	// ── Theme selector ───────────────────────────────────────────────────

	private async openThemeSelector(): Promise<void> {
		const available = getAvailableThemes();
		const themes: ThemeInfo[] = available.map((name) => ({
			name,
			description: `${name.charAt(0).toUpperCase() + name.slice(1)} theme`,
		}));
		this.themeSelector.setThemes(themes);
		this.themeSelector.setMessage("Enter selects a color theme.");
		this.themeSelector.show();
		const overlay = this.tui.showOverlay(this.themeSelector, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		overlay.focus();
	}

	private handleThemeSelectorAction(action: ThemeSelectorAction): void {
		if (action.type === "close") {
			this.tui.removeOverlay(this.themeSelector);
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			return;
		}
		const themeInfo = action.theme;
		this.themeSelector.setMessage(`Setting: ${themeInfo.name}...`);
		this.tui.requestRender();
		const ok = this.setThemeByName(themeInfo.name);
		this.tui.removeOverlay(this.themeSelector);
		this.statusPanel.update({ phase: "ready" });
		if (ok) {
			this.transcript.addSystemMessage(`Theme: ${themeInfo.name}`);
		} else {
			this.transcript.addSystemMessage(`Unknown theme: ${themeInfo.name}`);
		}
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
	}

	private setThemeByName(name: string): boolean {
		const available = getAvailableThemes();
		if (!available.includes(name)) return false;
		setTheme(name);
		saveConfigField("theme", name);
		return true;
	}

	private async openSettingsSelector(): Promise<void> {
		try {
			const data = this.bridge.getSettingsData();
			const thinkingLevels = [
				"off",
				"minimal",
				"low",
				"medium",
				"high",
				"xhigh",
			];
			const permissionModes = ["acceptAll", "acceptEdits", "ask", "plan"];
			const settings: SettingDef[] = [
				{
					name: "Model",
					currentValue: data.model,
					description: "LLM model to use",
					options: [{ label: data.model, value: data.model, current: true }],
				},
				{
					name: "Temperature",
					currentValue: String(data.temperature),
					description: "Sampling temperature (0–2)",
					options: [0.0, 0.3, 0.5, 0.7, 1.0].map((v) => ({
						label: String(v),
						value: String(v),
						current: Math.abs(data.temperature - v) < 0.001,
					})),
				},
				{
					name: "Max tokens",
					currentValue: String(data.maxTokens),
					description: "Maximum response tokens",
					options: [1024, 2048, 4096, 8192, 16384].map((v) => ({
						label: String(v),
						value: String(v),
						current: data.maxTokens === v,
					})),
				},
				{
					name: "Max iterations",
					currentValue: String(data.maxIterations),
					description: "Maximum tool-use iterations per turn",
					options: [10, 20, 30, 50, 100].map((v) => ({
						label: String(v),
						value: String(v),
						current: data.maxIterations === v,
					})),
				},
				{
					name: "Thinking level",
					currentValue: data.thinkingLevel,
					description: "Depth of reasoning before responding",
					options: thinkingLevels.map((v) => ({
						label: v.charAt(0).toUpperCase() + v.slice(1),
						value: v,
						current: data.thinkingLevel === v,
					})),
				},
				{
					name: "Permission mode",
					currentValue: data.permissionMode,
					description: "How the agent handles tool permissions",
					options: permissionModes.map((v) => ({
						label: v,
						value: v,
						current: data.permissionMode === v,
					})),
				},
				{
					name: "Loop detection",
					currentValue: data.loopDetectionEnabled ? "on" : "off",
					description: "Detect and break infinite agent loops",
					options: [
						{
							label: "on",
							value: "true",
							current: data.loopDetectionEnabled,
							toggleOn: true,
						},
						{
							label: "off",
							value: "false",
							current: !data.loopDetectionEnabled,
							toggleOn: false,
						},
					],
				},
				{
					name: "Guards",
					currentValue: data.guardsEnabled ? "on" : "off",
					description: "Safety guards against harmful tool use",
					options: [
						{
							label: "on",
							value: "true",
							current: data.guardsEnabled,
							toggleOn: true,
						},
						{
							label: "off",
							value: "false",
							current: !data.guardsEnabled,
							toggleOn: false,
						},
					],
				},
				{
					name: "Compaction",
					currentValue: data.proactiveCompactionEnabled ? "on" : "off",
					description: "Auto-compact context to save tokens",
					options: [
						{
							label: "on",
							value: "true",
							current: data.proactiveCompactionEnabled,
							toggleOn: true,
						},
						{
							label: "off",
							value: "false",
							current: !data.proactiveCompactionEnabled,
							toggleOn: false,
						},
					],
				},
				{
					name: "Inference mode",
					currentValue: data.inferenceMode,
					description: "Pre-defined sampling parameter set (Alt+M to cycle)",
					options: [
						{
							label: "Think General",
							value: "thinking-general",
							current: data.inferenceMode === "thinking-general",
						},
						{
							label: "Think Code",
							value: "thinking-coding",
							current: data.inferenceMode === "thinking-coding",
						},
						{
							label: "Instruct",
							value: "instruct-general",
							current: data.inferenceMode === "instruct-general",
						},
						{
							label: "Reason",
							value: "instruct-reasoning",
							current: data.inferenceMode === "instruct-reasoning",
						},
					],
				},
				{
					name: "Post-edit diagnostics",
					currentValue: data.postEditDiagnostics ? "on" : "off",
					description: "Check edited files against the project",
					options: [
						{
							label: "on",
							value: "true",
							current: data.postEditDiagnostics,
							toggleOn: true,
						},
						{
							label: "off",
							value: "false",
							current: !data.postEditDiagnostics,
							toggleOn: false,
						},
					],
				},
			];
			this.settingsSelector.setSettings(settings);
			this.settingsSelector.setMessage(
				"Enter selects a setting · Enter in detail applies",
			);
			this.settingsSelector.show();
			const overlay = this.tui.showOverlay(this.settingsSelector, {
				anchor: "aboveInput",
				align: "left",
				maxHeight: 18,
			});
			overlay.focus();
		} catch (e: unknown) {
			this.transcript.addSystemMessage(
				`Settings error: ${e instanceof Error ? e.message : String(e)}`,
			);
		}
	}

	private handleSettingsSelectorAction(action: SettingsSelectorAction): void {
		if (action.type === "close") {
			this.tui.removeOverlay(this.settingsSelector);
			this.statusPanel.update({ phase: "ready" });
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
			return;
		}
		if (
			action.type === "open" &&
			action.settingName.toLowerCase() === "model"
		) {
			this.tui.removeOverlay(this.settingsSelector);
			this.openModelSelector();
			return;
		}
		if (action.type !== "change") return;
		// action.type === "change"
		const { settingName, value } = action;
		this.settingsSelector.setMessage(`Applying ${settingName}...`);
		this.tui.requestRender();

		// Apply the setting via the bridge
		switch (settingName.toLowerCase()) {
			case "model":
				this.bridge.setModel(value);
				this.transcript.addSystemMessage(`Model: ${value}`);
				break;
			case "temperature": {
				const num = Number(value);
				if (Number.isFinite(num) && num >= 0 && num <= 2) {
					this.bridge.setTemperature(num);
					this.transcript.addSystemMessage(`Temperature: ${num}`);
				} else {
					this.transcript.addSystemMessage(
						"Temperature must be between 0 and 2.",
					);
				}
				break;
			}
			case "max tokens": {
				const num = Number.parseInt(value, 10);
				if (Number.isFinite(num) && num >= 1) {
					this.bridge.setMaxTokens(num);
					this.transcript.addSystemMessage(`Max tokens: ${num}`);
				} else {
					this.transcript.addSystemMessage(
						"Max tokens must be a positive integer.",
					);
				}
				break;
			}
			case "max iterations": {
				const num = Number.parseInt(value, 10);
				if (Number.isFinite(num) && num >= 1) {
					this.bridge.setMaxIterations(num);
					this.transcript.addSystemMessage(`Max iterations: ${num}`);
				} else {
					this.transcript.addSystemMessage(
						"Max iterations must be a positive integer.",
					);
				}
				break;
			}
			case "thinking level":
				this.bridge.setThinkingLevel(value);
				this.transcript.addSystemMessage(`Thinking level: ${value}`);
				break;
			case "permission mode":
				this.bridge.setPermissionMode(
					value as "acceptAll" | "acceptEdits" | "ask" | "plan",
				);
				this.transcript.addSystemMessage(`Permission mode: ${value}`);
				break;
			case "loop detection": {
				const on = value === "true";
				this.bridge.setRuntimeToggle("loopDetectionEnabled", on);
				this.transcript.addSystemMessage(
					`Loop detection: ${on ? "on" : "off"}`,
				);
				break;
			}
			case "guards": {
				const on = value === "true";
				this.bridge.setRuntimeToggle("guardsEnabled", on);
				this.transcript.addSystemMessage(`Guards: ${on ? "on" : "off"}`);
				break;
			}
			case "compaction": {
				const on = value === "true";
				this.bridge.setRuntimeToggle("proactiveCompactionEnabled", on);
				this.transcript.addSystemMessage(`Compaction: ${on ? "on" : "off"}`);
				break;
			}
			case "post-edit diagnostics": {
				const on = value === "true";
				this.bridge.setRuntimeToggle("postEditDiagnostics", on);
				saveConfigField("postEditDiagnostics", on);
				this.transcript.addSystemMessage(
					`Post-edit diagnostics: ${on ? "on" : "off"}`,
				);
				break;
			}
			case "inference mode": {
				const valid = [
					"thinking-general",
					"thinking-coding",
					"instruct-general",
					"instruct-reasoning",
				];
				if (!valid.includes(value)) {
					this.transcript.addSystemMessage(
						`Invalid inference mode: ${value}. Valid: ${valid.join(", ")}`,
					);
				} else {
					this.setInferenceMode(value);
				}
				break;
			}
			default:
				this.transcript.addSystemMessage(`Unknown setting: ${settingName}`);
		}

		this.tui.removeOverlay(this.settingsSelector);
		this.statusPanel.update({ phase: "ready" });
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
	}

	// ── Start ──────────────────────────────────────────────────────────────

	start(): void {
		this.buildLayout();
		this.tui.enableMouse();
		this.tui.start();
	}

	// ── Public accessors for external integration ──────────────────────────

	getSlashPopup(): SlashPopup {
		return this.slashPopup;
	}

	getInputBar(): InputBar {
		return this.inputBar;
	}

	async stop(): Promise<void> {
		this.tui.stop();
		await this.bridge.stop();
	}

	// ── Goal evaluation ──────────────────────────────────────────────────

	private async evaluateGoal(goalState: Readonly<GoalState>): Promise<void> {
		if (this.goalEvaluationPending) return;
		this.goalEvaluationPending = true;
		// Build conversation snapshot from transcript turns
		const turns = this.transcript.getTurns();
		const snapshot = turns
			.map((t) => {
				const parts: string[] = [];
				if (t.userMessage) parts.push(`User: ${t.userMessage}`);
				if (t.assistantMessage) parts.push(`Assistant: ${t.assistantMessage}`);
				return parts.join("\n");
			})
			.filter(Boolean)
			.join("\n\n");

		const evaluatorPrompt = GoalManager.buildEvaluatorPrompt(
			goalState.condition,
			snapshot,
		);

		this.transcript.addSystemMessage(
			`◎ Goal evaluation #${goalState.turnCount}: "${goalState.condition}"`,
		);
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();

		// Call LLM directly for evaluation (like dropper.ts does)
		const { baseUrl, model } = this.bridge.getConfig();
		const apiKey =
			process.env.ANTHROPIC_API_KEY ??
			process.env.OPENAI_API_KEY ??
			process.env.LLM_API_KEY ??
			"sk-no-key";

		let response: string;
		try {
			const res = await fetch(
				`${(baseUrl ?? "https://api.openai.com").replace(/\/+$/, "")}/v1/chat/completions`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${apiKey}`,
						"x-api-key": apiKey,
					},
					body: JSON.stringify({
						model: model || "gpt-4o",
						messages: [{ role: "system", content: evaluatorPrompt }],
						max_tokens: 256,
						temperature: 0,
					}),
				},
			);

			if (!res.ok) {
				const errText = await res.text().catch(() => "");
				throw new Error(
					`LLM API error ${res.status}: ${errText.slice(0, 200)}`,
				);
			}

			const data = (await res.json()) as {
				choices: Array<{ message: { content: string } }>;
			};
			response = data.choices?.[0]?.message?.content ?? "";
		} catch (e: unknown) {
			const err = e instanceof Error ? e.message : String(e);
			this.goalManager.handleAction({ type: "clear" });
			this.goalActive = false;
			this.transcript.addSystemMessage(
				`Goal evaluation failed: ${err}. Goal cancelled.`,
			);
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.tui.requestRender();
			this.goalEvaluationPending = false;
			return;
		}

		const { met, reason } = GoalManager.parseEvaluatorResponse(response);

		if (met) {
			this.goalManager.recordEvaluation(true, reason);
			this.goalActive = false;
			this.transcript.addSystemMessage(
				`✓ Goal achieved: "${goalState.condition}" — ${reason}`,
			);
		} else {
			this.goalManager.recordEvaluation(false, reason);
			const stillActive = this.goalManager.isActive();
			this.goalActive = stillActive;
			this.transcript.addSystemMessage(
				stillActive
					? `◎ Goal not yet met: ${reason} — continuing...`
					: `Goal stopped: ${this.goalManager.getState()?.lastReason || reason}`,
			);
			if (stillActive) {
				const reminder = `Goal reminder: "${goalState.condition}". ${reason}. Continue working toward the goal.`;
				void this.bridge.sendMessage(reminder).catch((error: unknown) => {
					this.bridge.reportError(error);
				});
			}
		}

		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
		this.goalEvaluationPending = false;
	}
}

function normalizeStartupHookMessage(item: unknown): {
	title: string;
	content: string;
} {
	if (!item || typeof item !== "object") {
		return { title: "Startup hook", content: String(item || "").trim() };
	}
	const raw = item as Record<string, unknown>;
	const pluginName = String(raw.plugin_name || "").trim();
	const pluginId = String(raw.plugin_id || "").trim();
	const matcher = String(raw.matcher || "").trim();
	const label = pluginName || pluginId || "Startup hook";
	const suffix =
		pluginName && pluginId && pluginName !== pluginId ? ` (${pluginId})` : "";
	const matcherText = matcher && matcher !== "*" ? ` · ${matcher}` : "";
	return {
		title: `${label}${suffix}${matcherText}`,
		content: String(raw.content || "").trim(),
	};
}

// Convert stored transcript turns into agent-core messages so a restored
// session's conversation re-enters the model context. Tool chunks are stored
// as names only (no results), so only user/assistant text is restorable.
function turnsToMessages(turns: Turn[]): CoreMessage[] {
	const messages: CoreMessage[] = [];
	for (const turn of turns) {
		if (turn.userMessage.content) {
			messages.push({ role: "user", content: turn.userMessage.content });
		}
		const assistantText = (turn.assistantMessage?.chunks ?? [])
			.filter((c) => c.type === "content" && c.contentText)
			.map((c) => c.contentText)
			.join("");
		if (assistantText) {
			messages.push({ role: "assistant", content: assistantText });
		}
	}
	return messages;
}
