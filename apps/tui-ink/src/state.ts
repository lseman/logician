// ── Ink TUI State Management ──────────────────────────────────────────────────

import { EventEmitter } from "node:events";
import type {
	AgentRuntime,
} from "@logician/log-runtime/application";
import type {
	Transcript,
	Turn,
} from "@logician/log-runtime/sessions";
import type {
	TuiSessionService,
} from "@logician/log-runtime/sessions";
import type {
	GitStatus,
	Notification,
	NotificationLevel,
	OverlayKind,
	OverlayState,
	SessionInfo,
	ThinkingDisplayMode,
	ThinkingLevel,
	TuiPhase,
	WorkflowMode,
	InferenceMode,
	ReasonerStatus,
	SteerMessage,
	TodoItem,
} from "./types.ts";

export class TuiState extends EventEmitter {
	// Core state
	phase: TuiPhase = "ready";
	model = "local";
	contextTokens = 0;
	contextMaxTokens = 128_000;
	cacheReadTokens = 0;

	// Session
	currentSessionId: string | null = null;
	sessionTitle = "New Session";
	sessions: SessionInfo[] = [];
	thinkingDisplayMode: ThinkingDisplayMode = "expanded";

	// Inference
	thinkingLevel: ThinkingLevel = "off";
	inferenceMode: InferenceMode = "none";
	workflowMode: WorkflowMode = "act";
	planMode = false;
	executionProfile: "autonomous" | "minimal" = "minimal";
	permissionMode: "acceptAll" | "acceptEdits" | "ask" = "acceptEdits";
	reasoner: ReasonerStatus = { name: "default", active: false };

	// Git
	git: GitStatus = { staged: 0, modified: 0, untracked: 0 };

	// Runtime flags
	rtkProxyEnabled = false;
	legroomEnabled = false;
	memoriamEnabled = false;
	graphicianEnabled = true;
	fffgrepEnabled = true;
	mcpLoading = false;

	// Input
	inputValue = "";
	inputCursor = 0;
	inputFocused = true;

	// Notifications
	notifications: Notification[] = [];

	// Overlays
	overlay: OverlayState = { kind: null };

	// TODO
	todos: TodoItem[] = [];

	// Steer queue
	steerMessages: SteerMessage[] = [];

	// Loop / goal state
	loopIteration = 0;
	goalCondition?: string;
	goalTurnCount?: number;
	goalElapsed?: number;

	// Research
	researchActive = false;
	researchIteration = 0;
	researchStatus?: string;

	// CWD
	cwd = process.cwd();
	virtualEnv?: string;
	branch?: string;

	// Input bar autocomplete
	slashQuery = "";
	fileMentionQuery = "";
	fileSuggestions: string[] = [];

	// Transcript turns (synced from transcript object)
	transcriptTurns: Turn[] = [];

	// Bridge reference (for direct calls)
	bridge: AgentRuntime | null = null;

	// Rendering
	private renderScheduled = false;
	// Monotonic version — bumped on every mutation so React's
	// useSyncExternalStore snapshot changes identity and re-renders.
	version = 0;

	constructor() {
		super();
		// Bound so it can be passed directly to useSyncExternalStore.
		this.subscribe = this.subscribe.bind(this);
		this.getSnapshot = this.getSnapshot.bind(this);
	}

	// ── External store contract (React) ──────────────────────────────────────

	subscribe(callback: () => void): () => void {
		this.on("render", callback);
		return () => this.off("render", callback);
	}

	getSnapshot(): number {
		return this.version;
	}

	/** Force a React re-render after mutating a plain field directly. */
	touch(): void {
		this.scheduleRender();
	}

	// ── Mutations ─────────────────────────────────────────────────────────────

	updatePhase(phase: TuiPhase): void {
		this.phase = phase;
		this.scheduleRender();
	}

	addTurn(turn: Turn): void {
		this.scheduleRender();
	}

	removeTurn(id: string): void {
		this.scheduleRender();
	}

	setInputValue(value: string, cursor?: number): void {
		this.inputValue = value;
		if (cursor !== undefined) this.inputCursor = cursor;
		this.scheduleRender();
	}

	focusInput(focused: boolean): void {
		this.inputFocused = focused;
		this.scheduleRender();
	}

	setNotifications(notifications: Notification[]): void {
		this.notifications = notifications;
		this.scheduleRender();
	}

	showNotification(message: string, level: NotificationLevel = "info"): void {
		const id = `n-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
		const notification: Notification = { id, message, level, createdAt: Date.now() };
		this.notifications = [...this.notifications.slice(-4), notification];
		// Auto-dismiss after 5 seconds
		setTimeout(() => {
			this.notifications = this.notifications.filter(n => n.id !== id);
			this.scheduleRender();
		}, 5000);
		this.scheduleRender();
	}

	dismissNotification(id: string): void {
		this.notifications = this.notifications.filter(n => n.id !== id);
		this.scheduleRender();
	}

	setOverlay(kind: OverlayKind | null, data?: Record<string, unknown>): void {
		this.overlay = { kind, data };
		this.scheduleRender();
	}

	setTodos(todos: TodoItem[]): void {
		this.todos = todos;
		this.scheduleRender();
	}

	addTodo(text: string): string {
		const id = `todo-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
		this.todos = [...this.todos, { id, text, done: false }];
		this.scheduleRender();
		return id;
	}

	toggleTodo(id: string): void {
		this.todos = this.todos.map(t => (t.id === id ? { ...t, done: !t.done } : t));
		this.scheduleRender();
	}

	deleteTodo(id: string): void {
		this.todos = this.todos.filter(t => t.id !== id);
		this.scheduleRender();
	}

	setSteerMessages(messages: SteerMessage[]): void {
		this.steerMessages = messages;
		this.scheduleRender();
	}

	addSteerMessage(message: string): void {
		this.steerMessages = [
			...this.steerMessages,
			{ id: `s-${Date.now()}`, message, createdAt: Date.now() },
		];
		this.scheduleRender();
	}

	clearSteerMessages(): void {
		this.steerMessages = [];
		this.scheduleRender();
	}

	setSessions(sessions: SessionInfo[]): void {
		this.sessions = sessions;
		this.scheduleRender();
	}

	updateSession(sessionId: string, updates: Partial<SessionInfo>): void {
		this.sessions = this.sessions.map(s =>
			s.id === sessionId ? { ...s, ...updates } : s,
		);
		this.scheduleRender();
	}

	// Status bar fields
	setModel(model: string): void {
		this.model = model;
		this.scheduleRender();
	}

	setContextTokens(tokens: number): void {
		this.contextTokens = tokens;
		this.scheduleRender();
	}

	setCacheReadTokens(tokens: number): void {
		this.cacheReadTokens = tokens;
		this.scheduleRender();
	}

	setGitStatus(git: GitStatus): void {
		this.git = git;
		this.branch = git.branch;
		this.scheduleRender();
	}

	setReasoner(status: ReasonerStatus): void {
		this.reasoner = status;
		this.scheduleRender();
	}

	setVirtualEnv(env?: string): void {
		this.virtualEnv = env;
		this.scheduleRender();
	}

	setMcpLoading(loading: boolean): void {
		this.mcpLoading = loading;
		this.scheduleRender();
	}

	// Research
	setResearchActive(active: boolean): void {
		this.researchActive = active;
		this.scheduleRender();
	}

	setResearchStatus(status: string, iteration?: number): void {
		this.researchStatus = status;
		if (iteration !== undefined) this.researchIteration = iteration;
		this.scheduleRender();
	}

	// Loop/Goal
	setLoopIteration(iteration: number): void {
		this.loopIteration = iteration;
		this.scheduleRender();
	}

	setGoalState(condition?: string, turnCount?: number, elapsed?: number): void {
		this.goalCondition = condition;
		this.goalTurnCount = turnCount;
		this.goalElapsed = elapsed;
		this.scheduleRender();
	}

	// Inference settings
	setThinkingLevel(level: ThinkingLevel): void {
		this.thinkingLevel = level;
		this.scheduleRender();
	}

	setInferenceMode(mode: InferenceMode): void {
		this.inferenceMode = mode;
		this.scheduleRender();
	}

	setWorkflowMode(mode: WorkflowMode): void {
		this.workflowMode = mode;
		this.scheduleRender();
	}

	setPlanMode(planMode: boolean): void {
		this.planMode = planMode;
		this.scheduleRender();
	}

	setExecutionProfile(profile: "autonomous" | "minimal"): void {
		this.executionProfile = profile;
		this.scheduleRender();
	}

	setPermissionMode(mode: "acceptAll" | "acceptEdits" | "ask"): void {
		this.permissionMode = mode;
		this.scheduleRender();
	}

	setThinkingDisplayMode(mode: ThinkingDisplayMode): void {
		this.thinkingDisplayMode = mode;
		this.scheduleRender();
	}

	// Session
	setCurrentSession(id: string, title: string): void {
		this.currentSessionId = id;
		this.sessionTitle = title;
		this.scheduleRender();
	}

	// Input autocomplete
	setSlashQuery(query: string): void {
		this.slashQuery = query;
		this.scheduleRender();
	}

	setFileSuggestions(suggestions: string[]): void {
		this.fileSuggestions = suggestions;
		this.scheduleRender();
	}

	// ── Bridge integration ──────────────────────────────────────────────────

	setBridge(bridge: AgentRuntime): void {
		this.bridge = bridge;
	}

	setTranscriptTurns(turns: Turn[]): void {
		this.transcriptTurns = turns;
		this.scheduleRender();
	}

	addTranscriptTurn(turn: Turn): void {
		this.transcriptTurns = [...this.transcriptTurns, turn];
		this.scheduleRender();
	}

	// ── Rendering ─────────────────────────────────────────────────────────────

	private scheduleRender(): void {
		this.version++;
		if (this.renderScheduled) return;
		this.renderScheduled = true;
		queueMicrotask(() => {
			this.renderScheduled = false;
			this.emit("render");
		});
	}

	onRender(callback: () => void): () => void {
		this.on("render", callback);
		return () => this.off("render", callback);
	}

	destroy(): void {
		this.removeAllListeners();
	}
}
