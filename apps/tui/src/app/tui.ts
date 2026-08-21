// ── Main TUI ──────────────────────────────────────────────────────────────────
// Wires agent-core, transcript, and components together.

import {
	AgentRuntime,
	GoalManager,
	type GoalState,
	LoopManager,
} from "@logician/agent-runtime/application";
import { createSlashCommands } from "@logician/agent-runtime/commands";
import { resolveRuntimeConfig } from "@logician/agent-runtime/configuration/runtime";
import {
	Transcript,
	TuiSessionService,
	type Turn,
} from "@logician/agent-runtime/sessions";
import {
	createAutoresearchTools,
	getVirtualEnvPythonVersion,
} from "@logician/agent-runtime/tools";
import { AutoresearchSession } from "@logician/autoresearch";
import { StatusBar } from "../footer/layout.ts";
import { InputBar } from "../input/input-bar.ts";
import { KillRing } from "../input/kill-ring.ts";
import { UndoStack } from "../input/undo-stack.ts";
import {
	type AutoresearchDashboardAction,
	AutoresearchDashboardOverlay,
} from "../overlays/autoresearch-dashboard.ts";
import { ChoicePopup } from "../overlays/choice-popup.ts";
import { FileMentionPopup } from "../overlays/file-mention-popup.ts";
import {
	InferenceModeSelector,
	type InferenceModeSelectorAction,
} from "../overlays/inference-mode-selector.ts";
import {
	type McpManagerAction,
	McpManagerOverlay,
} from "../overlays/mcp-manager.ts";
import {
	type ModelSelectorAction,
	ModelSelectorOverlay,
} from "../overlays/model-selector.ts";
import { PermissionPopup } from "../overlays/permission-popup.ts";
import {
	type PluginManagerAction,
	PluginManagerOverlay,
} from "../overlays/plugin-manager.ts";
import {
	type ReasonerSelectorAction,
	ReasonerSelectorOverlay,
} from "../overlays/reasoner-selector.ts";
import { SessionBrowserOverlay } from "../overlays/session-manager.ts";
import {
	type SettingsSelectorAction,
	SettingsSelectorOverlay,
} from "../overlays/settings-overlay.ts";
import { SlashPopup } from "../overlays/slash-popup.ts";
import {
	type ThemeSelectorAction,
	ThemeSelectorOverlay,
} from "../overlays/theme-selector.ts";
import { Flex } from "../rendering/flex.ts";
import { ScrollView } from "../rendering/scroll-view.ts";
import { Separator } from "../rendering/separator.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { NewOutputIndicator } from "../rendering/transcript/new-output-indicator.ts";
import { INITIAL_TURN_STATE, type TurnState } from "../state/turn-state.ts";
import {
	NotificationCenter,
	type NotificationLevel,
} from "../status/notification-center.ts";
import { ResearchWidget } from "../status/research-widget.ts";
import { SteerQueue } from "../status/steer-queue.ts";
import { TodoBar } from "../status/todo-bar.ts";
import { WorkSurface } from "../status/work-surface.ts";
import { Container, TUI } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { setupBridge as setupBridgeImpl } from "./bridge-event-handler.ts";
import {
	createLocalHandlers,
	createSlashSubmitHandler,
} from "./commands/index.ts";
import { getGitStatus } from "./git-status.ts";
import { evaluateGoal as evaluateGoalImpl } from "./goal-runner.ts";
import {
	applyThinkingLevel as applyThinkingLevelImpl,
	cycleExecutionProfile as cycleExecutionProfileImpl,
	cycleInferenceMode as cycleInferenceModeImpl,
	type InferenceMode,
	setExecutionProfile as setExecutionProfileImpl,
	setInferenceMode as setInferenceModeImpl,
	setPlanMode as setPlanModeImpl,
	togglePlanMode as togglePlanModeImpl,
} from "./inference-settings.ts";
import { setupInputHandler as setupInputHandlerImpl } from "./input-controller.ts";
import {
	handleAutoresearchDashboardAction as handleAutoresearchDashboardActionImpl,
	handleInferenceModeSelectorAction as handleInferenceModeSelectorActionImpl,
	handleMcpManagerAction as handleMcpManagerActionImpl,
	handleModelSelectorAction as handleModelSelectorActionImpl,
	handlePluginManagerAction as handlePluginManagerActionImpl,
	handleReasonerSelectorAction as handleReasonerSelectorActionImpl,
	handleSettingsSelectorAction as handleSettingsSelectorActionImpl,
	handleThemeSelectorAction as handleThemeSelectorActionImpl,
	openAutoresearchDashboard as openAutoresearchDashboardImpl,
	openInferenceModeSelector as openInferenceModeSelectorImpl,
	openMcpManager as openMcpManagerImpl,
	openModelSelector as openModelSelectorImpl,
	openPluginManager as openPluginManagerImpl,
	openReasonerSelector as openReasonerSelectorImpl,
	openSettingsSelector as openSettingsSelectorImpl,
	openThemeSelector as openThemeSelectorImpl,
	setThemeByName as setThemeByNameImpl,
	updateFileMentionPopup as updateFileMentionPopupImpl,
} from "./overlay-controllers/index.ts";
import {
	autoSaveTurn,
	handleSessionAction as handleSessionActionImpl,
	openSessionManager as openSessionManagerImpl,
	restoreSession as restoreSessionImpl,
} from "./session/controller.ts";

// ── Main TUI ─────────────────────────────────────────────────────────────────

export class LogicianTUI {
	private exitHandler: (() => void) | null = null;
	private stopPromise: Promise<void> | null = null;
	// Not private: read/written by the extracted app/*.ts functions through
	// their Ctx interfaces, which LogicianTUI satisfies structurally. Typed as
	// the narrow surface TUI implements — see TuiHandle in terminal/core.ts
	// for what's actually called across app/*.ts.
	tui: TUI;
	bridge: AgentRuntime;
	transcript: Transcript;
	statusPanel: StatusBar;
	todoBar: TodoBar;
	workSurface: WorkSurface;
	researchWidget: ResearchWidget;
	notifications: NotificationCenter;
	steerQueue: SteerQueue;
	inputBar: InputBar;
	slashPopup: SlashPopup;
	fileMentionPopup: FileMentionPopup;
	choicePopup: ChoicePopup;
	choicePopupPreview = false;
	workflowMode: "act" | "plan";
	planPhase: "idle" | "planning" | "awaiting_approval" | "executing" = "idle";
	normalPermissionMode: "acceptAll" | "acceptEdits" | "ask";
	permissionPopup: PermissionPopup;
	pluginManager: PluginManagerOverlay;
	autoresearchDashboard: AutoresearchDashboardOverlay;
	mcpManager: McpManagerOverlay;
	reasonerSelector: ReasonerSelectorOverlay;
	modelSelector: ModelSelectorOverlay;
	inferenceModeSelector: InferenceModeSelector;
	themeSelector: ThemeSelectorOverlay;
	settingsSelector: SettingsSelectorOverlay;
	transcriptDisplay: TranscriptDisplay;
	sessionManager: SessionBrowserOverlay;
	sessionService: TuiSessionService;
	private killRing: KillRing;
	private undoStack: UndoStack<{ value: string; cursor: number }>;
	loopManager: LoopManager;
	goalManager: GoalManager;
	researchManager: AutoresearchSession;
	turnState: TurnState = INITIAL_TURN_STATE;
	loopActive = false;
	goalActive = false;
	goalEvaluationPending = false;
	private cancellationPending = false;
	configPath?: string;
	thinkingLevel = "off";
	inferenceMode: InferenceMode = "none";
	thinkingDisplayMode: "collapsed" | "summary" | "expanded" = "expanded";
	currentSessionId: string | null = null;
	// Tool call awaiting an interactive allow/deny answer in the input bar.
	pendingPermission: { toolCallId: string; toolName: string } | null = null;

	// Inference mode helper — used by the keyboard shortcut and /settings.
	setInferenceMode(mode: string, persist = true): void {
		setInferenceModeImpl(this, mode, { persist });
	}

	notify(message: string, level: NotificationLevel = "info"): void {
		this.notifications.show(message, level);
	}

	async cancelActiveTurn(): Promise<void> {
		if (this.cancellationPending || !this.bridge.isActive()) return;
		this.cancellationPending = true;
		this.pendingPermission = null;
		const activeTurn = this.transcript.getTurns().at(-1);
		const recoveryPrompt =
			activeTurn && !activeTurn.isComplete
				? (activeTurn.userMessage?.content ?? "")
				: "";
		this.statusPanel.update({ phase: "cancelling" });
		this.statusPanel.startAnimation();
		this.notify("Stopping after the active operation settles…", "info");
		this.tui.requestRender();

		try {
			const cleared = await this.bridge.cancel();
			const clearedCount =
				(cleared?.clearedSteering.length ?? 0) +
				(cleared?.clearedFollowUp.length ?? 0);
			// Do not overwrite anything the user typed while cancellation settled.
			if (recoveryPrompt && this.inputBar.valueText.length === 0) {
				this.inputBar.valueText = recoveryPrompt;
			}
			this.transcript.addSystemMessage(
				`Turn interrupted safely.${recoveryPrompt ? " The prompt was restored to the composer for editing or retry." : ""}` +
					(clearedCount > 0
						? ` Cleared ${clearedCount} queued message${clearedCount === 1 ? "" : "s"}; next-turn messages were preserved.`
						: ""),
			);
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.statusPanel.update({ phase: "ready" });
		} catch (error) {
			this.statusPanel.update({ phase: "error" });
			this.notify(
				`Could not confirm interruption: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		} finally {
			this.cancellationPending = false;
			this.statusPanel.stopAnimation();
			this.tui.requestRender();
		}
	}

	cycleInferenceMode(): void {
		cycleInferenceModeImpl(this);
	}

	setExecutionProfile(profile: "autonomous" | "minimal"): void {
		setExecutionProfileImpl(this, profile);
	}

	cycleExecutionProfile(): "autonomous" | "minimal" {
		return cycleExecutionProfileImpl(this);
	}

	setPlanMode(planning: boolean): "acceptEdits" | "plan" {
		return setPlanModeImpl(this, planning);
	}

	togglePlanMode(): "acceptEdits" | "plan" {
		return togglePlanModeImpl(this);
	}

	constructor(
		runtimeConfig = resolveRuntimeConfig(process.cwd(), process.env, {
			loadProjectConfig: false,
		}),
	) {
		this.workflowMode =
			runtimeConfig.source.workflowMode ??
			(runtimeConfig.bridge.permissionMode === "plan" ? "plan" : "act");
		this.normalPermissionMode =
			runtimeConfig.bridge.permissionMode === "acceptAll" ||
			runtimeConfig.bridge.permissionMode === "ask"
				? runtimeConfig.bridge.permissionMode
				: "acceptEdits";
		this.configPath = runtimeConfig.configPath;
		this.thinkingLevel = runtimeConfig.bridge.thinkingLevel ?? "off";
		this.inferenceMode =
			(runtimeConfig.bridge.inferenceMode as InferenceMode | undefined) ??
			"none";
		this.researchManager = new AutoresearchSession(
			runtimeConfig.bridge.cwd ?? process.cwd(),
			(message, level) => this.notifications.show(message, level),
		);
		// Reload persisted .auto/log.jsonl state now — equivalent to
		// pi-autoresearch's session_start hook, since this app has one
		// AutoresearchSession per process rather than per pi-session.
		this.researchManager.reload();
		this.researchWidget = new ResearchWidget(this.researchManager);
		this.bridge = new AgentRuntime({
			...runtimeConfig.bridge,
			extraTools: [
				...(runtimeConfig.bridge.extraTools ?? []),
				...createAutoresearchTools(this.researchManager),
			],
		});
		this.transcript = new Transcript();
		this.statusPanel = new StatusBar();
		this.todoBar = new TodoBar();
		this.workSurface = new WorkSurface();
		this.notifications = new NotificationCenter();
		this.steerQueue = new SteerQueue();
		this.inputBar = new InputBar();
		this.slashPopup = new SlashPopup();
		this.fileMentionPopup = new FileMentionPopup();
		this.choicePopup = new ChoicePopup();
		this.permissionPopup = new PermissionPopup();
		this.pluginManager = new PluginManagerOverlay();
		this.autoresearchDashboard = new AutoresearchDashboardOverlay(
			this.researchManager,
		);
		this.mcpManager = new McpManagerOverlay();
		this.reasonerSelector = new ReasonerSelectorOverlay();
		this.modelSelector = new ModelSelectorOverlay();
		this.inferenceModeSelector = new InferenceModeSelector();
		this.themeSelector = new ThemeSelectorOverlay();
		this.settingsSelector = new SettingsSelectorOverlay();
		this.transcriptDisplay = new TranscriptDisplay({
			thinkingMode: this.thinkingDisplayMode,
			maxMessageLength:
				runtimeConfig.source.truncation?.transcriptMessageMaxChars,
			// Both caps default to unbounded in every mode. Fullscreen mode's
			// ScrollView already clips painted output to the viewport (only
			// termHeight rows are walked per frame — see paintBox), so dropping
			// old turns/lines here doesn't save render cost, it just stops the
			// user from scrolling back to see them and adds a truncation banner
			// nobody asked for. "regular" mode has no fixed viewport at all —
			// printed lines are handed off to the terminal's own scrollback and
			// never re-rendered, so unbounded is the only sensible behavior
			// there too. transcriptMaxTurns/transcriptMaxRenderedLines in
			// settings.json remain available as an explicit opt-in cap in either
			// mode, for anyone who wants one.
			maxTurns: runtimeConfig.source.transcriptMaxTurns,
			maxRenderedLines: runtimeConfig.source.transcriptMaxRenderedLines,
		});
		this.transcriptDisplay.setOnAnimationTick(() => this.tui.requestRender());
		// Apply inference mode only after its transcript/status dependencies exist.
		if (runtimeConfig.source.inferenceMode) {
			this.setInferenceMode(runtimeConfig.source.inferenceMode, false);
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
		this.loopManager.setOnStateChange(state => {
			this.loopActive = state !== null;
			this.workSurface.setLoopIteration(state?.iteration ?? 0);
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
		this.sessionService = new TuiSessionService(process.cwd());
		this.sessionManager = new SessionBrowserOverlay();
		this.sessionManager.setStore(this.sessionService);
		this.sessionManager.setActionCallback(action =>
			this.handleSessionAction(action),
		);
		// Only create initial session — never auto-resume. Sessions are loaded
		// explicitly via the --session CLI flag in index.ts.
		this.currentSessionId = this.sessionService.createSession("New Session");
		this.bridge.useConversationSession(
			this.currentSessionId,
			this.sessionService.getRawSession(this.currentSessionId) ?? undefined,
		);
		this.statusPanel.update({ sessionTitle: "New Session" });

		// Wire up dependencies
		this.inputBar.setKillRing(this.killRing);
		this.inputBar.setUndoStack(this.undoStack);

		// Setup bridge event handling
		this.setupBridge();

		// Setup transcript change handling
		this.setupTranscript();

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
			virtualEnv: process.env.VIRTUAL_ENV,
			virtualEnvPythonVersion: getVirtualEnvPythonVersion(
				process.env.VIRTUAL_ENV,
			),
			branch: gitStatus.branch,
			gitModified: gitStatus.modified,
			gitStaged: gitStatus.staged,
			gitUntracked: gitStatus.untracked,
			gitCommit: gitStatus.commit,
			gitAhead: gitStatus.ahead,
			gitBehind: gitStatus.behind,
			gitAddedLines: gitStatus.addedLines,
			gitRemovedLines: gitStatus.removedLines,
			contextTokens: 0,
			reasoner: this.bridge.getReasonerStatus(),
			contextMaxTokens: runtimeConfig.bridge.contextWindowTokens,
			permissionMode: runtimeConfig.bridge.permissionMode ?? "acceptEdits",
			workflowMode: this.workflowMode,
			executionProfile: runtimeConfig.bridge.executionProfile ?? "minimal",
			rtkProxyEnabled: runtimeConfig.bridge.rtkProxyEnabled ?? false,
			ariadneEnabled: runtimeConfig.bridge.ariadneEnabled ?? true,
			fffgrepEnabled: runtimeConfig.bridge.fffgrepEnabled ?? true,
			memoryEnabled: runtimeConfig.bridge.memoryEnabled ?? false,
			// MCP discovery starts in the background the moment the bridge is
			// constructed (ToolRouter's constructor), so it's typically already
			// in flight by the time this status line renders. The "MCP" notice
			// handler in bridge-event-handler.ts flips this back off and fills
			// in the server count once loading actually finishes.
			mcpLoading: this.bridge.isMcpLoading(),
		});

		// Setup slash commands
		const localHandlers = createLocalHandlers(this);
		this.slashPopup.setCommands(
			createSlashCommands(this.bridge, localHandlers),
		);
		const submitSlashCommand = createSlashSubmitHandler(this);
		this.slashPopup.setOnSubmit((result, dispatch, command) => {
			void submitSlashCommand(result, dispatch, command);
		});
	}

	// ── Bridge setup ─────────────────────────────────────────────────────────

	private setupBridge(): void {
		setupBridgeImpl(this);
	}

	// ── Transcript setup ─────────────────────────────────────────────────────

	private setupTranscript(): void {
		this.transcript.onChange(() => {
			this.transcriptDisplay.setTurns(this.transcript.getTurns());
			this.transcriptDisplay.setThinkingMode(
				this.transcript.getThinkingDisplayMode(),
			);
			// Steer queue is driven directly by queue_update events (see
			// handleBridgeEvent), not transcript state. Auto-scroll to bottom
			// while already at bottom is handled by the transcript ScrollView's
			// own `follow: "end"` behavior (see buildLayout()) — it re-pins to
			// the new max scroll position on every layout pass as content grows.
			this.tui.requestRender();
		});
	}

	// ── Input handling ─────────────────────────────────────────────────────

	private setupInputHandler(): void {
		setupInputHandlerImpl(this);
	}

	/** Auto-save the latest turn to the current session. */
	_autoSaveTurn(): void {
		autoSaveTurn(this);
	}

	/** Handle session manager actions (select, rename, delete, new). */
	private handleSessionAction(action: {
		type: "close" | "select" | "rename" | "delete" | "new";
		sessionId?: string;
		title?: string;
	}): void {
		handleSessionActionImpl(this, action);
	}

	/** Open the session manager overlay. */
	openSessionManager(): void {
		openSessionManagerImpl(this);
	}

	/** Load turns for a specific session ID (used by --session CLI flag). */
	loadTurns(sessionId: string): Turn[] | undefined {
		return this.sessionService.loadTurns(sessionId);
	}

	/** Restore turns into transcript and bridge (public for CLI --session usage). */
	restoreSession(turns: Turn[]): void {
		restoreSessionImpl(this, turns);
	}

	// ── Layout ─────────────────────────────────────────────────────────────

	private buildLayout(): void {
		// Stack todo bar + steer queue + question handler above the input bar
		// (both render empty when there's nothing to show, so they only take
		// space when active).
		const pinnedContainer = new Container();
		this.notifications.setOnInvalidate(() => this.tui.requestRender());
		pinnedContainer.addChild(this.notifications);
		pinnedContainer.addChild(this.todoBar);
		pinnedContainer.addChild(this.workSurface);
		pinnedContainer.addChild(this.researchWidget);
		pinnedContainer.addChild(this.steerQueue);

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
		// Center-anchored and large, unlike the aboveInput popups above — this
		// is meant to read as a fullscreen dashboard (README: "Ctrl+A opens a
		// scrollable full-terminal dashboard"), not a compact picker.
		this.tui.showOverlay(this.autoresearchDashboard, {
			anchor: "center",
			width: "90%",
			maxHeight: "90%",
		});

		// The dock (input bar + status bar + aboveInput overlays) is OUTSIDE
		// the ScrollView so it stays fixed at the bottom of the viewport even
		// when the user scrolls the transcript up. The ScrollView clips only
		// the TranscriptDisplay to its viewport; the dock renders at a
		// separate y position below the clip region.
		const dock = new Container();
		dock.addChild(new Separator());
		dock.addChild(pinnedContainer);
		dock.addChild(this.tui.getAboveInputOverlaysComponent());
		dock.addChild(this.inputBar);
		dock.addChild(new Separator());
		dock.addChild(this.statusPanel);

		// Transcript scrolls and follows newly streamed output; scrolling away
		// disables follow until the user returns to the bottom (Home/End/PageDown
		// or the new-output indicator's click-to-catch-up).
		const transcriptScroll = new ScrollView(this.transcriptDisplay, {
			follow: "end",
			primary: true,
			overscroll: "chain",
			scrollbar: "always",
			scrollbarStyle: (glyph, isThumb) =>
				isThumb ? theme.fg("selected", glyph) : theme.fg("separator", glyph),
		});
		this.transcriptDisplay.setScrollView(transcriptScroll);
		this.tui.showOverlay(new NewOutputIndicator(this.transcriptDisplay), {
			anchor: "bottom",
			align: "left",
			onClick: () => {
				transcriptScroll.scrollToEnd();
				this.transcriptDisplay.clearNewOutputIndicator();
			},
		});

		// Root is a Flex (column) with ScrollView (grow: 1) + dock (auto).
		// The Flex allocates the remaining height to the ScrollView after the
		// dock takes its natural height. The ScrollView clips its content
		// (TranscriptDisplay) to its rect; the dock is at a different y
		// position, so it's never clipped — always visible at the bottom.
		const root = new Flex([
			{ component: transcriptScroll, basis: 0, grow: 1, shrink: 1, minSize: 1 },
			{ component: dock, basis: "auto", grow: 0, shrink: 1, minSize: 1 },
		]);
		this.tui.setLayoutRoot(root);
		this.tui.setInputBarComponent(this.inputBar);
	}

	async openPluginManager(): Promise<void> {
		await openPluginManagerImpl(this);
	}

	handlePluginManagerAction(action: PluginManagerAction): void {
		handlePluginManagerActionImpl(this, action);
	}

	// ── MCP manager ───────────────────────────────────────────────────────

	async openMcpManager(): Promise<void> {
		await openMcpManagerImpl(this);
	}

	handleMcpManagerAction(action: McpManagerAction): void {
		handleMcpManagerActionImpl(this, action);
	}

	// ── Autoresearch dashboard ───────────────────────────────────────────────

	openAutoresearchDashboard(): void {
		openAutoresearchDashboardImpl(this);
	}

	handleAutoresearchDashboardAction(action: AutoresearchDashboardAction): void {
		handleAutoresearchDashboardActionImpl(this, action);
	}

	// ── Reasoner selector ───────────────────────────────────────────────────

	async openReasonerSelector(): Promise<void> {
		await openReasonerSelectorImpl(this);
	}

	handleReasonerSelectorAction(action: ReasonerSelectorAction): void {
		handleReasonerSelectorActionImpl(this, action);
	}

	// ── File mention autocomplete ────────────────────────────────────────

	fileMentionListedCwd: string | null = null;
	fileMentionListing: Promise<string[]> | null = null;

	async updateFileMentionPopup(query: string): Promise<void> {
		await updateFileMentionPopupImpl(this, query);
	}

	// ── Model selector ───────────────────────────────────────────────────

	openModelSelector(): void {
		openModelSelectorImpl(this);
	}

	handleModelSelectorAction(action: ModelSelectorAction): void {
		handleModelSelectorActionImpl(this, action);
	}

	openInferenceModeSelector(): void {
		openInferenceModeSelectorImpl(this);
	}

	handleInferenceModeSelectorAction(action: InferenceModeSelectorAction): void {
		handleInferenceModeSelectorActionImpl(this, action);
	}

	// ── Theme selector ───────────────────────────────────────────────────

	async openThemeSelector(): Promise<void> {
		await openThemeSelectorImpl(this);
	}

	handleThemeSelectorAction(action: ThemeSelectorAction): void {
		handleThemeSelectorActionImpl(this, action);
	}

	setThemeByName(name: string): boolean {
		return setThemeByNameImpl(name);
	}

	async openSettingsSelector(): Promise<void> {
		await openSettingsSelectorImpl(this);
	}

	handleSettingsSelectorAction(action: SettingsSelectorAction): void {
		handleSettingsSelectorActionImpl(this, action);
	}

	applyThinkingLevel(level: string): void {
		applyThinkingLevelImpl(this, level, { persist: true });
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

	setExitHandler(handler: () => void): void {
		this.exitHandler = handler;
	}

	requestExit(): void {
		if (this.exitHandler) {
			this.exitHandler();
			return;
		}
		void this.stop().then(() => process.exit(0));
	}

	stop(): Promise<void> {
		if (!this.stopPromise) {
			this.stopPromise = (async () => {
				this.researchManager.shutdown();
				this.tui.stop();
				await this.bridge.stop();
			})();
		}
		return this.stopPromise;
	}

	getSessionRecoveryTip(): string | null {
		return this.currentSessionId
			? `run \`logician --session ${this.currentSessionId}\` to recover this session\n`
			: null;
	}

	// ── Memory forwarding ────────────────────────────────────────────────

	getMemoryStore() {
		return this.bridge.getMemoryStore();
	}

	getMemoryStats() {
		return this.bridge.getMemoryStats();
	}

	// ── Goal evaluation ──────────────────────────────────────────────────

	async evaluateGoal(goalState: Readonly<GoalState>): Promise<void> {
		await evaluateGoalImpl(this, goalState);
	}
}
