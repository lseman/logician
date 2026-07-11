// ── Main TUI ──────────────────────────────────────────────────────────────────
// Wires agent-core, transcript, and components together.
import { formatContextSize, envNumber } from "@logician/coding-agent";

// Re-export markdownTableCell (same as escapeTable for table use)

import { execSync } from "node:child_process";
import { AgentCoreBridge } from "@logician/coding-agent/bridge";
import { InputBar } from "../../components/input-bar.ts";
import { ChoicePopup } from "../../components/choice-popup.ts";
import {
	type McpManagerAction,
	McpManagerOverlay,
} from "../../components/mcp-manager.ts";
import {
	type PluginManagerAction,
	PluginManagerOverlay,
} from "../../components/plugin-manager.ts";
import {
	type ReasonerInfo,
	type ReasonerSelectorAction,
	ReasonerSelectorOverlay,
} from "../../components/reasoner-selector.ts";
import {
	type ModelInfo,
	type ModelSelectorAction,
	ModelSelectorOverlay,
} from "../../components/model-selector.ts";
import { SessionManager } from "../../components/session-manager.ts";
import {
	type ThemeInfo,
	type ThemeSelectorAction,
	ThemeSelectorOverlay,
} from "../../components/theme-selector.ts";
import {
	type SettingDef,
	type SettingsSelectorAction,
	SettingsSelectorOverlay,
} from "../../components/settings-overlay.ts";
import { SlashPopup } from "../../components/slash-popup.ts";
import { StatusBar } from "../../components/status-bar.ts";
import { SteerQueue } from "../../components/steer-queue.ts";
import { ThinkingPanel } from "../../components/thinking-panel.ts";
import { TodoBar } from "../../components/todo/todo-bar.ts";
import { TranscriptDisplay } from "../../components/transcript-display.ts";
import { SessionStore } from "@logician/coding-agent/session-store";
import {
	configBool,
	configNumber,
	configString,
	loadLogicianConfig,
	saveConfigField,
} from "@logician/coding-agent/config";
import type { ParsedBridgeEvent } from "@logician/coding-agent/events";
import { KillRing } from "../input/kill-ring.ts";
import { LoopManager } from "@logician/coding-agent/loop-manager";
import {
	getReasonerIds,
	getReasonerMeta,
	type ReasonerMeta,
} from "@logician/agent-capabilities/reasoners/registry";
import type { Message as CoreMessage } from "@logician/agent-core";
import {
	createSlashCommands,
	filterSlashCommands,
	type SlashCommandDef,
} from "@logician/coding-agent/slash-commands";
import { setTheme, getAvailableThemes, theme } from "../theme/theme.ts";
import { Transcript, type Turn } from "@logician/coding-agent/transcript";
import { Container, TUI } from "../core/tui-core.ts";
import { UndoStack } from "../input/undo-stack.ts";

// ── Main TUI ─────────────────────────────────────────────────────────────────

export class LogicianTUI {
	private tui: TUI;
	private bridge: AgentCoreBridge;
	private transcript: Transcript;
	private statusPanel: StatusBar;
	private todoBar: TodoBar;
	private steerQueue: SteerQueue;
	private thinkingPanel: ThinkingPanel;
	private inputBar: InputBar;
	private slashPopup: SlashPopup;
	private choicePopup: ChoicePopup;
	private pluginManager: PluginManagerOverlay;
	private mcpManager: McpManagerOverlay;
	private reasonerSelector: ReasonerSelectorOverlay;
	private modelSelector: ModelSelectorOverlay;
	private themeSelector: ThemeSelectorOverlay;
	private settingsSelector: SettingsSelectorOverlay;
	private transcriptDisplay: TranscriptDisplay;
	private sessionManager: SessionManager;
	private sessionStore: SessionStore;
	private killRing: KillRing;
	private undoStack: UndoStack<{ value: string; cursor: number }>;
	private loopManager: LoopManager;
	private streaming = false;
	private loopActive = false;
	private configPath?: string;
	private thinkingLevel = "off";
	private cacheEnabled = true;
	private thinkingDisplayMode: "collapsed" | "summary" | "expanded" =
		"expanded";
	private traceOn = false;
	private currentSessionId: string | null = null;
	// Tool call awaiting an interactive allow/deny answer in the input bar.
	private pendingPermission: { toolCallId: string; toolName: string } | null =
		null;

	// eslint-disable-next-line max-lines-per-function -- wires up entire TUI (bridge, transcript, components, keybindings, overlays)
	constructor() {
		const loadedConfig = loadLogicianConfig(process.cwd());
		this.configPath = loadedConfig.path;
		const config = loadedConfig.config;
		const modelName =
			process.env.LOGICIAN_MODEL || configString(config.model) || "";
		this.bridge = new AgentCoreBridge({
			baseUrl:
				process.env.LOGICIAN_LLM_URL ||
				configString(config.baseUrl) ||
				configString(config.llmUrl) ||
				"http://127.0.0.1:8080",
			model: modelName,
			models: config.models,
			systemPrompt:
				process.env.LOGICIAN_SYSTEM_PROMPT || configString(config.systemPrompt),
			chatTemplate: configString(config.chatTemplate),
			temperature: configNumber(config.temperature),
			maxTokens: configNumber(config.maxTokens),
			maxIterations: configNumber(config.maxIterations),
			toolExecution:
				configString(config.toolExecution) === "sequential"
					? "sequential"
					: configString(config.toolExecution) === "parallel"
						? "parallel"
						: undefined,
			contextWindowTokens:
				envNumber("LOGICIAN_CONTEXT_WINDOW") ||
				envNumber("LOGICIAN_CTX_SIZE") ||
				configNumber(config.contextWindowTokens) ||
				configNumber(config.contextWindow),
			runtimeHooksEnabled:
				process.env.LOGICIAN_HOOKS !== undefined
					? process.env.LOGICIAN_HOOKS !== "0"
					: configBool(config.hooks),
			mcpEager:
				process.env.LOGICIAN_MCP_EAGER !== undefined
					? process.env.LOGICIAN_MCP_EAGER !== "0"
					: configBool(config.mcpEager),
			webSearch: config.webSearch
				? {
						baseUrl: configString(config.webSearch.baseUrl),
						maxResults: configNumber(config.webSearch.maxResults),
					}
				: undefined,
			permissionMode: configString(config.permissionMode) as
				| "acceptAll"
				| "acceptEdits"
				| "ask"
				| "plan"
				| undefined,
			permissionRules:
				config.permissions &&
				typeof config.permissions === "object" &&
				!Array.isArray(config.permissions)
					? (config.permissions as { allow?: string[]; deny?: string[] })
					: undefined,
			steeringInterrupt: configBool(config.steeringInterrupt),
			maxTotalTokens: configNumber(config.maxTotalTokens),
			// Safeguard options: loop detection and guards default OFF (match pi's trust-model).
			// Continuation defaults ON — prevents premature stopping when model says "done" mid-task.
			loopDetectionEnabled: configBool(config.loopDetectionEnabled),
			guardsEnabled: configBool(config.guardsEnabled),
			continuationEnabled: configBool(config.continuationEnabled),
			cwd: process.cwd(),
		});
		this.transcript = new Transcript();
		this.statusPanel = new StatusBar();
		this.todoBar = new TodoBar();
		this.steerQueue = new SteerQueue();
		this.thinkingPanel = new ThinkingPanel();
		this.inputBar = new InputBar();
		this.slashPopup = new SlashPopup();
		this.choicePopup = new ChoicePopup();
		this.pluginManager = new PluginManagerOverlay();
		this.mcpManager = new McpManagerOverlay();
		this.reasonerSelector = new ReasonerSelectorOverlay();
		this.modelSelector = new ModelSelectorOverlay();
		this.themeSelector = new ThemeSelectorOverlay();
		this.settingsSelector = new SettingsSelectorOverlay();
		this.transcriptDisplay = new TranscriptDisplay({
			thinkingMode: this.thinkingDisplayMode,
		});
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

		// Create the TUI with hardware cursor support
		this.tui = new TUI(process.stdout, true);
		this.statusPanel.setOnInvalidate(() => this.tui.requestRender());
		this.todoBar.setOnInvalidate(() => this.tui.requestRender());

		// ── Session store ────────────────────────────────────────────────────
		this.sessionStore = new SessionStore(process.cwd());
		this.sessionManager = new SessionManager();
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
					`Project: ${this.getGitVersion() || "-"}`,
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
		this.setupInputHandler(
			handleStatus,
			handlePlugins,
			handleMcp,
			handleReasoner,
			handleTheme,
		);

		// Focus input bar by default
		this.tui.setFocus(this.inputBar);

		// Initial state
		const gitStatus = this.getGitStatus();
		this.statusPanel.update({
			thinkingLevel: this.thinkingLevel,
			cacheEnabled: this.cacheEnabled,
			phase: "ready",
			model: modelName || "local",
			cwd: process.cwd(),
			branch: gitStatus.branch,
			gitModified: gitStatus.modified,
			gitStaged: gitStatus.staged,
			gitUntracked: gitStatus.untracked,
			contextTokens: 0,
			reasoner: "none",
			contextMaxTokens:
				envNumber("LOGICIAN_CONTEXT_WINDOW") ||
				envNumber("LOGICIAN_CTX_SIZE") ||
				configNumber(config.contextWindowTokens) ||
				configNumber(config.contextWindow),
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
			setCache: (enabled: unknown) => {
				const en =
					typeof enabled === "boolean"
						? enabled
						: enabled === "true" || enabled === true;
				this.cacheEnabled = en;
				this.transcript.setCacheEnabled(en);
				this.bridge.sendSlash(`/cache ${en ? "enable" : "disable"}`);
				this.statusPanel.update({ cacheEnabled: en });
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
				this.thinkingPanel.clear();
				setStatusPhase("ready");
			},
			version: () => "Logician 0.2.0 (TypeScript runtime)",
			memory: (raw: unknown) => this.bridge.memoryCommand(String(raw ?? "")),
			settings: (raw: unknown) => {
				const args = String(raw ?? "").trim();
				if (!args) {
					this.openSettingsSelector();
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
				this.statusPanel.update({
					phase: next === "plan" ? "plan" : "ready",
				});
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
		this.slashPopup.setOnSubmit((result, dispatch, command) => {
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
				const preview = JSON.stringify(event.args ?? {}).slice(0, 200);
				this.transcript.addSystemMessage(
					`Permission needed: ${event.tool_name} ${preview}\n` +
						"Reply y (allow once), a (always allow this tool), or n (deny).",
				);
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
					anchor: "center",
					maxHeight: 18,
				});
				overlay.focus();
				this.tui.requestRender();
				break;
			}
			case "token":
				if (!this.streaming) {
					this.streaming = true;
					this.statusPanel.update({ phase: "streaming" });
					this.statusPanel.startAnimation();
				}
				break;
			case "tool_start":
			case "tool_execution_start":
				this.statusPanel.update({ phase: "tool" });
				this.statusPanel.startAnimation();
				break;
			case "turn_end":
				this.streaming = false;
				this.statusPanel.stopAnimation();
				// Auto-save the completed turn
				this._autoSaveTurn();
				this.statusPanel.update({
					phase: "ready",
					turnCount: this.transcript.getTurns().length,
					messageCount: this.transcript.getMessageCount(),
				});
				break;
			case "turn_start":
				this.statusPanel.update({ phase: "thinking" });
				this.statusPanel.startAnimation();
				break;
			case "phase":
				this.statusPanel.update({ phase: event.state });
				if (event.state !== "ready") {
					this.statusPanel.startAnimation();
				} else {
					this.streaming = false;
					this.statusPanel.stopAnimation();
					this.statusPanel.update({
						turnCount: this.transcript.getTurns().length,
						messageCount: this.transcript.getMessageCount(),
					});
				}
				break;
			case "context_update":
				this.statusPanel.update({
					contextTokens: Number(event.tokens || 0),
					contextMaxTokens: Number(event.max_tokens || 0) || undefined,
					contextCompacted: event.compacted === true,
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
				// No inline confirmation — the SteerQueue widget already reflects
				// pending messages, and the injected content shows up in the turn.
				break;
		}

		this.tui.requestRender();
	}

	private formatStartupMessage(state: Record<string, unknown>): string {
		const pluginCount = Number(state.startup_plugins_loaded || 0);
		const hookCount = Number(state.startup_hooks_loaded || 0);
		const mcpServerCount = Number(state.mcp_servers_loaded || 0);
		const mcpToolCount = Number(state.mcp_tools_loaded || 0);
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

		const runtimeRows: Array<[string, string]> = [
			["Agent", String(state.agent_name || "logician")],
			["Model", String(state.model || "unknown")],
			["Theme", theme.name],
			["Base URL", String(state.base_url || "unknown")],
			["Project", this.getGitVersion() || "-"],
			["Config", this.configPath || "-"],
		];

		const dim = "\x1b[2m";
		const reset = "\x1b[0m";
		const tableHeader = `${dim}| Runtime | Value |${reset}`;
		const tableSep = `${dim}| --- | --- |${reset}`;
		const tableRows = runtimeRows.map(
			([label, value]) =>
				`| ${dim}${markdownTableCell(label)}${reset} | ${markdownTableCell(value)} |`,
		);

		const lines = [
			"# Logician",
			"Runtime ready.",
			"",
			tableHeader,
			tableSep,
			...tableRows,
			"",
			"## Startup",
			`Plugins loaded: ${pluginCount}`,
			`Startup hooks: ${hookCount}`,
			state.mcp_deferred
				? "MCP: deferred until first agent turn or /status"
				: `MCP: ${mcpServerCount} server(s), ${mcpToolCount} tool(s)`,
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

	private setupInputHandler(
		handleStatus: () => Promise<void>,
		handlePlugins: (args: string) => Promise<void>,
		handleMcp: (args: string) => Promise<void>,
		handleReasoner: (args: string) => Promise<void>,
		handleTheme: (args: string) => Promise<void>,
	): void {
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
				// Escape — dismiss the menu but keep what was typed;
				// if a loop is active, stop it
				if (data === "\x1b") {
					this.slashPopup.hide();
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
			anchor: "center",
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
		pinnedContainer.addChild(this.steerQueue);
		this.tui.setFixedAboveInputComponent(pinnedContainer);

		// Slash popup as overlay anchored to the bottom of the transcript area, so
		// the suggestion list sits directly above the input bar like an inline
		// autocomplete menu.
		this.tui.showOverlay(this.slashPopup, {
			anchor: "bottom",
			align: "left",
			maxHeight: 12,
		});
		this.tui.showOverlay(this.pluginManager, {
			anchor: "center",
			maxHeight: 18,
		});
		this.tui.showOverlay(this.mcpManager, {
			anchor: "center",
			maxHeight: 18,
		});
		this.tui.showOverlay(this.sessionManager, {
			anchor: "center",
			maxHeight: 18,
		});
		this.tui.showOverlay(this.settingsSelector, {
			anchor: "center",
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
			this.tui.removeOverlay(this.pluginManager);
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
			this.tui.removeOverlay(this.mcpManager);
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
			anchor: "center",
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

	// ── Model selector ───────────────────────────────────────────────────

	private openModelSelector(): void {
		this.statusPanel.update({ phase: "model" });
		const currentModel = this.bridge.getCurrentModel();
		const models = this.bridge.getModels();
		const baseUrl = this.bridge.getCurrentBaseUrl();
		const modelInfos: ModelInfo[] = models.map((id) => {
			const url = this.bridge.getModelUrl(id);
			const showUrl = url && url !== baseUrl ? url : undefined;
			return {
				id,
				name: id,
				active: id === currentModel,
				url: showUrl,
			};
		});
		this.modelSelector.setModels(modelInfos);
		this.modelSelector.setMessage(
			"Enter selects model for the current session.",
		);
		this.modelSelector.show();
		const overlay = this.tui.showOverlay(this.modelSelector, {
			anchor: "center",
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
		this.bridge.setModel(selected.id);
		// Save to global settings
		saveConfigField("model", selected.id);
		// Update status
		this.tui.removeOverlay(this.modelSelector);
		this.statusPanel.update({ phase: "ready", model: selected.id });
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
			anchor: "center",
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
			const thinkingLevels = ["off", "minimal", "low", "medium", "high", "xhigh"];
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
						{ label: "on", value: "true", current: data.loopDetectionEnabled, toggleOn: true },
						{ label: "off", value: "false", current: !data.loopDetectionEnabled, toggleOn: false },
					],
				},
				{
					name: "Guards",
					currentValue: data.guardsEnabled ? "on" : "off",
					description: "Safety guards against harmful tool use",
					options: [
						{ label: "on", value: "true", current: data.guardsEnabled, toggleOn: true },
						{ label: "off", value: "false", current: !data.guardsEnabled, toggleOn: false },
					],
				},
				{
					name: "Compaction",
					currentValue: data.proactiveCompactionEnabled ? "on" : "off",
					description: "Auto-compact context to save tokens",
					options: [
						{ label: "on", value: "true", current: data.proactiveCompactionEnabled, toggleOn: true },
						{ label: "off", value: "false", current: !data.proactiveCompactionEnabled, toggleOn: false },
					],
				},
			];
			this.settingsSelector.setSettings(settings);
			this.settingsSelector.setMessage("Enter selects a setting · Enter in detail applies");
			this.settingsSelector.show();
			const overlay = this.tui.showOverlay(this.settingsSelector, {
				anchor: "center",
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
					this.transcript.addSystemMessage("Temperature must be between 0 and 2.");
				}
				break;
			}
			case "max tokens": {
				const num = Number.parseInt(value, 10);
				if (Number.isFinite(num) && num >= 1) {
					this.bridge.setMaxTokens(num);
					this.transcript.addSystemMessage(`Max tokens: ${num}`);
				} else {
					this.transcript.addSystemMessage("Max tokens must be a positive integer.");
				}
				break;
			}
			case "max iterations": {
				const num = Number.parseInt(value, 10);
				if (Number.isFinite(num) && num >= 1) {
					this.bridge.setMaxIterations(num);
					this.transcript.addSystemMessage(`Max iterations: ${num}`);
				} else {
					this.transcript.addSystemMessage("Max iterations must be a positive integer.");
				}
				break;
			}
			case "thinking level":
				this.bridge.setThinkingLevel(value);
				this.transcript.addSystemMessage(`Thinking level: ${value}`);
				break;
			case "permission mode":
				this.bridge.setPermissionMode(value as "acceptAll" | "acceptEdits" | "ask" | "plan");
				this.transcript.addSystemMessage(`Permission mode: ${value}`);
				break;
			case "loop detection": {
				const on = value === "true";
				this.bridge.setRuntimeToggle("loopDetectionEnabled", on);
				this.transcript.addSystemMessage(`Loop detection: ${on ? "on" : "off"}`);
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
			default:
				this.transcript.addSystemMessage(`Unknown setting: ${settingName}`);
		}

		this.tui.removeOverlay(this.settingsSelector);
		this.statusPanel.update({ phase: "ready" });
		this.transcriptDisplay.setTurns(this.transcript.getTurns());
		this.tui.requestRender();
	}

	private getGitBranch(): string {
		try {
			return execSync("git branch --show-current", {
				cwd: process.cwd(),
				encoding: "utf8",
				stdio: ["ignore", "pipe", "ignore"],
			}).trim();
		} catch {
			return "";
		}
	}

	private getGitStatus(): {
		branch: string;
		modified: number;
		staged: number;
		untracked: number;
	} {
		const branch = this.getGitBranch();
		let modified = 0;
		let staged = 0;
		let untracked = 0;
		try {
			modified =
				parseInt(
					execSync("git diff --quiet || git diff --name-only | wc -l", {
						cwd: process.cwd(),
						encoding: "utf8",
						stdio: ["ignore", "pipe", "ignore"],
					}).trim(),
				) || 0;
			staged =
				parseInt(
					execSync(
						"git diff --cached --quiet || git diff --cached --name-only | wc -l",
						{
							cwd: process.cwd(),
							encoding: "utf8",
							stdio: ["ignore", "pipe", "ignore"],
						},
					).trim(),
				) || 0;
			untracked =
				parseInt(
					execSync("git ls-files --others --exclude-standard | wc -l", {
						cwd: process.cwd(),
						encoding: "utf8",
						stdio: ["ignore", "pipe", "ignore"],
					}).trim(),
				) || 0;
		} catch {
			// ignore
		}
		return { branch, modified, staged, untracked };
	}

	private getGitVersion(): string {
		try {
			const branch =
				this.getGitBranch() ||
				execSync("git rev-parse --short HEAD", {
					cwd: process.cwd(),
					encoding: "utf8",
					stdio: ["ignore", "pipe", "ignore"],
				}).trim();
			const sha = execSync("git rev-parse --short HEAD", {
				cwd: process.cwd(),
				encoding: "utf8",
				stdio: ["ignore", "pipe", "ignore"],
			}).trim();
			let dirty = "";
			try {
				execSync("git diff --quiet && git diff --cached --quiet", {
					cwd: process.cwd(),
					stdio: "ignore",
				});
			} catch {
				dirty = " dirty";
			}
			return `${branch}@${sha}${dirty}`;
		} catch {
			return "";
		}
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

function markdownTableCell(value: string): string {
	return value.replace(/\\/g, "\\\\").replace(/\|/g, "\\|");
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
