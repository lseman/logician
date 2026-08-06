// ── Main TUI ──────────────────────────────────────────────────────────────────
// Wires agent-core, transcript, and components together.

import {
	AgentCoreBridge,
	GoalManager,
	type GoalState,
	LoopManager,
} from "@logician/coding-agent/application";
import { createSlashCommands } from "@logician/coding-agent/commands";
import { resolveRuntimeConfig } from "@logician/coding-agent/runtime";
import {
	SessionStore,
	Transcript,
	type Turn,
} from "@logician/coding-agent/sessions";
import { InputBar } from "../input/input-bar.ts";
import { KillRing } from "../input/kill-ring.ts";
import { UndoStack } from "../input/undo-stack.ts";
import { ChoicePopup } from "../overlays/choice-popup.ts";
import { FileMentionPopup } from "../overlays/file-mention-popup.ts";
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
import { StatusBar } from "../status/status-bar.ts";
import { SteerQueue } from "../status/steer-queue.ts";
import { TodoBar } from "../status/todo-bar.ts";
import { WorkSurface } from "../status/work-surface.ts";
import { Container, TUI } from "../terminal/core.ts";
import { TuiMainScreen } from "../terminal/main-screen.ts";
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
} from "./inference-settings.ts";
import { setupInputHandler as setupInputHandlerImpl } from "./input-controller.ts";
import {
	handleMcpManagerAction as handleMcpManagerActionImpl,
	handleModelSelectorAction as handleModelSelectorActionImpl,
	handlePluginManagerAction as handlePluginManagerActionImpl,
	handleReasonerSelectorAction as handleReasonerSelectorActionImpl,
	handleSettingsSelectorAction as handleSettingsSelectorActionImpl,
	handleThemeSelectorAction as handleThemeSelectorActionImpl,
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

/**
 * "fullscreen" enters the alternate screen buffer and owns a fixed-height,
 * self-managed scrollable viewport (TUI/core.ts). "regular" (the CLI's
 * default, see index.ts) writes append-only into the main screen and leaves
 * history to the terminal's own scrollback (TuiMainScreen) — no fixed
 * viewport, no app-owned mouse-wheel scrolling. This constructor's own
 * default stays "fullscreen" for any other caller that doesn't pass a mode
 * explicitly.
 */
export type TuiUiMode = "fullscreen" | "regular";

export class LogicianTUI {
	private exitHandler: (() => void) | null = null;
	private stopPromise: Promise<void> | null = null;
	private readonly uiMode: TuiUiMode;
	// Not private: read/written by the extracted app/*.ts functions through
	// their Ctx interfaces, which LogicianTUI satisfies structurally. Typed as
	// the narrow surface both TUI and TuiMainScreen implement — see each
	// class's method list for what's actually called across app/*.ts.
	tui: TUI | TuiMainScreen;
	bridge: AgentCoreBridge;
	transcript: Transcript;
	statusPanel: StatusBar;
	todoBar: TodoBar;
	workSurface: WorkSurface;
	notifications: NotificationCenter;
	steerQueue: SteerQueue;
	inputBar: InputBar;
	slashPopup: SlashPopup;
	fileMentionPopup: FileMentionPopup;
	choicePopup: ChoicePopup;
	choicePopupPreview = false;
	permissionPopup: PermissionPopup;
	pluginManager: PluginManagerOverlay;
	mcpManager: McpManagerOverlay;
	reasonerSelector: ReasonerSelectorOverlay;
	modelSelector: ModelSelectorOverlay;
	themeSelector: ThemeSelectorOverlay;
	settingsSelector: SettingsSelectorOverlay;
	transcriptDisplay: TranscriptDisplay;
	sessionManager: SessionBrowserOverlay;
	sessionStore: SessionStore;
	private killRing: KillRing;
	private undoStack: UndoStack<{ value: string; cursor: number }>;
	loopManager: LoopManager;
	goalManager: GoalManager;
	turnState: TurnState = INITIAL_TURN_STATE;
	loopActive = false;
	goalActive = false;
	goalEvaluationPending = false;
	private cancellationPending = false;
	configPath?: string;
	thinkingLevel = "off";
	inferenceMode: InferenceMode = "instruct-general";
	thinkingDisplayMode: "collapsed" | "summary" | "expanded" = "expanded";
	currentSessionId: string | null = null;
	// Tool call awaiting an interactive allow/deny answer in the input bar.
	pendingPermission: { toolCallId: string; toolName: string } | null = null;

	// Inference mode helper — used by the keyboard shortcut and /settings.
	setInferenceMode(mode: string): void {
		setInferenceModeImpl(this, mode);
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

	constructor(
		runtimeConfig = resolveRuntimeConfig(process.cwd(), process.env, {
			loadProjectConfig: false,
		}),
		uiMode: TuiUiMode = "fullscreen",
	) {
		this.uiMode = uiMode;
		this.configPath = runtimeConfig.configPath;
		this.bridge = new AgentCoreBridge(runtimeConfig.bridge);
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
		this.mcpManager = new McpManagerOverlay();
		this.reasonerSelector = new ReasonerSelectorOverlay();
		this.modelSelector = new ModelSelectorOverlay();
		this.themeSelector = new ThemeSelectorOverlay();
		this.settingsSelector = new SettingsSelectorOverlay();
		this.transcriptDisplay = new TranscriptDisplay({
			thinkingMode: this.thinkingDisplayMode,
			maxMessageLength:
				runtimeConfig.source.truncation?.transcriptMessageMaxChars,
			// Both caps below exist to bound rendering cost against a fixed-height
			// viewport that only ever shows its last screenful (fullscreen mode).
			// In "regular" mode, printed lines are handed off to the terminal's
			// own scrollback and never re-rendered or re-walked, so dropping old
			// turns/lines only fights that model — the terminal already holds the
			// durable record, unbounded, for free. Forced unconditionally in
			// "regular" mode — a user's transcriptMaxTurns/transcriptMaxRenderedLines
			// setting in settings.json is a fullscreen-mode tuning knob and must
			// not re-enable this behavior there.
			maxTurns:
				this.uiMode === "regular"
					? Number.POSITIVE_INFINITY
					: runtimeConfig.source.transcriptMaxTurns,
			// maxRenderedLines exists to bound the cost of re-rendering the whole
			// transcript every frame against a fixed-height viewport (fullscreen
			// mode only ever shows the last viewportHeight rows anyway). In
			// "regular" mode there is no fixed viewport — printed lines are handed
			// off to the terminal's own scrollback and never re-rendered, so the
			// truncation banner ("N older turns not shown") only fights that model
			// instead of protecting anything. Forced unconditionally — see maxTurns
			// above for why a configured value must not override this in "regular"
			// mode.
			maxRenderedLines:
				this.uiMode === "regular"
					? Number.POSITIVE_INFINITY
					: runtimeConfig.source.transcriptMaxRenderedLines,
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
		this.loopManager.setOnStateChange(state => {
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
		this.tui =
			this.uiMode === "regular"
				? new TuiMainScreen(true)
				: new TUI(process.stdout, true);
		this.statusPanel.setOnInvalidate(() => this.tui.requestRender());
		this.todoBar.setOnInvalidate(() => this.tui.requestRender());
		this.workSurface.setOnInvalidate(() => this.tui.requestRender());

		// ── Session store ────────────────────────────────────────────────────
		this.sessionStore = new SessionStore(process.cwd());
		this.sessionManager = new SessionBrowserOverlay();
		this.sessionManager.setStore(this.sessionStore);
		this.sessionManager.setActionCallback(action =>
			this.handleSessionAction(action),
		);
		// Only create initial session — never auto-resume. Sessions are loaded
		// explicitly via the --session CLI flag in index.ts.
		this.currentSessionId = this.sessionStore.createSession({
			title: "New Session",
		});
		this.bridge.useConversationSession(this.currentSessionId);
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
			branch: gitStatus.branch,
			gitModified: gitStatus.modified,
			gitStaged: gitStatus.staged,
			gitUntracked: gitStatus.untracked,
			contextTokens: 0,
			reasoner: this.bridge.getReasonerStatus(),
			contextMaxTokens: runtimeConfig.bridge.contextWindowTokens,
			executionProfile: runtimeConfig.bridge.executionProfile ?? "autonomous",
			rtkProxyEnabled: runtimeConfig.bridge.rtkProxyEnabled ?? false,
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
		return this.sessionStore.loadTurns(sessionId);
	}

	/** Restore turns into transcript and bridge (public for CLI --session usage). */
	restoreSession(turns: Turn[]): void {
		restoreSessionImpl(this, turns);
	}

	// ── Layout ─────────────────────────────────────────────────────────────

	private buildLayout(): void {
		if (this.uiMode === "regular") {
			this.buildFlatLayout();
			return;
		}

		// Transcript scrolls independently and follows newly streamed output
		// while positioned at the end; scrolling away disables follow until the
		// user returns to the bottom (Home/End/PageDown or the new-output
		// indicator's click-to-catch-up).
		const transcriptScroll = new ScrollView(this.transcriptDisplay, {
			follow: "end",
			primary: true,
			overscroll: "chain",
			// The legacy TranscriptDisplay scrollbar was always visible whenever
			// content overflowed the viewport, with no fade — match that rather
			// than ScrollView's "auto" mode, which only appears after the first
			// scroll activity (via markScrollbarActivity) and then fades.
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

		// Stack todo bar + steer queue + question handler above the input bar
		// (both render empty when there's nothing to show, so they only take
		// space when active).
		const pinnedContainer = new Container();
		this.notifications.setOnInvalidate(() => this.tui.requestRender());
		pinnedContainer.addChild(this.notifications);
		pinnedContainer.addChild(this.todoBar);
		pinnedContainer.addChild(this.workSurface);
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

		const dock = new Flex([
			{ component: new Separator(), basis: 1 },
			{ component: pinnedContainer, basis: "auto", shrink: 1, minSize: 0 },
			{
				component: this.tui.getAboveInputOverlaysComponent(),
				basis: "auto",
				shrink: 1,
				minSize: 0,
			},
			{ component: this.inputBar, basis: "auto", shrink: 1, minSize: 1 },
			{ component: new Separator(), basis: 1 },
			{ component: this.statusPanel, basis: "auto", shrink: 1, minSize: 1 },
		]);
		const root = new Flex([
			{ component: transcriptScroll, basis: 0, grow: 1, shrink: 1, minSize: 1 },
			{ component: dock, basis: "auto", grow: 0, shrink: 1, minSize: 1 },
		]);
		this.tui.setLayoutRoot(root);
		this.tui.setInputBarComponent(this.inputBar);
	}

	/**
	 * "regular" (main-screen) layout: no fixed viewport to clip a ScrollView
	 * against, so the transcript and dock are mounted flat, in document order,
	 * directly onto the TuiMainScreen container — every frame just prints
	 * whatever's new at the tail and lets the terminal's own scrollback hold
	 * everything that scrolled off. aboveInput overlays (slash popup, file
	 * mention, plugin/MCP manager) still work via
	 * getAboveInputOverlaysComponent(); center/bottom-anchored overlays
	 * (currently only the "new output below" indicator, which has no
	 * meaning without a clipped viewport to scroll within) are skipped.
	 */
	private buildFlatLayout(): void {
		const pinnedContainer = new Container();
		this.notifications.setOnInvalidate(() => this.tui.requestRender());
		pinnedContainer.addChild(this.notifications);
		pinnedContainer.addChild(this.todoBar);
		pinnedContainer.addChild(this.workSurface);
		pinnedContainer.addChild(this.steerQueue);

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

		this.tui.addChild(this.transcriptDisplay);
		this.tui.addChild(new Separator());
		this.tui.addChild(pinnedContainer);
		this.tui.addChild(this.tui.getAboveInputOverlaysComponent());
		this.tui.addChild(this.inputBar);
		this.tui.addChild(new Separator());
		this.tui.addChild(this.statusPanel);
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
		applyThinkingLevelImpl(this, level);
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
