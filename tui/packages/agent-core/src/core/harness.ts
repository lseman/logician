// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//

import { withTimeout } from "../tools/shared/async-utils.ts";
import { runSessionStartHooks, runHookEvent } from "../tools/shared/plugins.ts";
import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "./file-checkpoints.ts";
import type { LLMBackend } from "./backend.ts";
import {
	runAgentLoop,
	runAgentLoopContinue,
	type RunAgentLoopConfig,
} from "./agent-loop-runner.ts";
import { Session, SessionManager } from "./session.ts";
import {
	compactMessages,
	convertToChatFormat,
	createUserMessage,
	estimateChatPayloadTokens,
} from "./messages.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import {
	MessageDeliveryManager,
	type DeliveryMode,
} from "../message-queue/manager.ts";
import type { ExtensionRunner, RegisteredTool } from "../extensions/index.ts";
import type { ExtensionEventBus } from "../hooks/extension-event-bus.ts";
import { composeHooks, buildBuiltinHooks, type HookLayer } from "../hooks/builtin-hooks.ts";
import { LoopDetector } from "./loop-detector.ts";
import {
	collectMessagesForBranchSummary,
	extractFileOpsFromMessages,
	computeFileLists,
	formatFileOperations,
	parseBranchSummary,
	serializeMessages,
} from "./branch-summarization.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHarnessStreamOptions,
	AgentHooks,
	AgentModelConfig,
	BeforeCompactContext,
	BeforeCompactResult,
	EventHandler,
	Message,
	QueueMode,
	ThinkingLevel,
	Tool,
} from "./types.ts";
import type { CompactionSettings } from "../compaction/index.ts";
import { OutputGuard } from "./output-guard.ts";
import { validateConfig, throwOnValidationErrors } from "./config-validator.ts";

// Explicit harness phases
export type HarnessPhase = "idle" | "turn" | "compaction" | "branch_summary";

export class HarnessBusyError extends Error {
	constructor(op: string, phase: HarnessPhase, required: HarnessPhase) {
		super(
			`AgentHarness cannot ${op}: phase is "${phase}", requires "${required}"`,
		);
		this.name = "HarnessBusyError";
	}
}

// Legal phase transitions
const PHASE_TRANSITIONS: Record<HarnessPhase, readonly HarnessPhase[]> = {
	idle: ["turn", "compaction", "branch_summary"],
	turn: ["idle"],
	compaction: ["idle"],
	branch_summary: ["idle"],
};

export interface AgentHarnessOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
	extensionRunner?: ExtensionRunner;
}

interface HarnessTurnSnapshot {
	promptText: string;
	initialMessages: Message[];
	config: AgentConfig;
	streamOptions: AgentHarnessStreamOptions;
	signal: AbortSignal;
}

/** Snapshot of the three harness queues, for UI display. */
export interface HarnessQueues {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}

/** Contents of queues cleared by abort(). */
export interface AbortResult {
	clearedSteering: string[];
	clearedFollowUp: string[];
	clearedNextTurn: string[];
}

/** Structured branch summary data. */
export interface BranchSummaryData {
	/** Goal of the branch. */
	goal: string;
	/** Constraints and preferences. */
	constraints: string[];
	/** Progress tracking. */
	progress: BranchProgress;
	/** Key decisions with rationale. */
	keyDecisions: Array<{ decision: string; rationale: string }>;
	/** Next steps to continue work. */
	nextSteps: string[];
	/** Full human-readable summary. */
	full: string;
}

/** Progress tracking for branch summaries. */
export interface BranchProgress {
	done: string[];
	inProgress: string[];
	blocked: string[];
}

/** Public branch info for UI / callers. */
export interface BranchInfo {
	id: string;
	depth: number;
	/** Summary of this branch's work (null until branchSummary is called). */
	summary: BranchSummaryData | null;
	/** Message index where this branch forked. */
	forkedAt: number;
}

// Conversation checkpoints: a snapshot of history is pushed before each
// prompt so a bad turn can be rewound. Bounded ring (newest last).
const MAX_CHECKPOINTS = 20;

export class AgentHarness {
	private config: AgentConfig;
	private backend: LLMBackend;
	private cwd?: string;
	private maxIterations?: number;

	private _phase: HarnessPhase = "idle";
	private idleTools: ToolRegistry;
	private abortController: AbortController | null = null;
	private loopConfig: AgentConfig | null = null;
	private history: Message[] = [];
	private branches: Array<{
		id: string;
		parent: Message[];
		forkedAt: number;
		summary: BranchSummaryData | null;
	}> = [];
	private branchSeq = 0;
	private checkpoints: Message[][] = [];
	private _nextTurnQueue: string[] = [];
	private msgManager: MessageDeliveryManager;
	private loopDetector: LoopDetector;
	private onQueueChange?: (queues: HarnessQueues) => void;
	private onPhaseChange?: (phase: HarnessPhase, prev: HarnessPhase) => void;
	private onSettled?: (nextTurnCount: number) => void;
	private onSavePoint?: () => void;
	private onCompaction?: (
		reason: "auto" | "manual",
		tokensBefore: number,
		tokensAfter: number,
	) => void;
	private autoCompactionSettings: CompactionSettings = {
		enabled: false,
		reserveTokens: 16_384,
		keepRecentTokens: 20_000,
		contextWindow: 128_000,
	};

	// ── Output Guard ─────────────────────────────────────────────────────
	private outputGuard: OutputGuard | null = null;
	private _streamOptions: AgentHarnessStreamOptions = {};
	private _hooksEnabled = true;
	private _session?: Session;
	private _sessionBaseDir?: string;
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
	private _runPromise?: Promise<void>;
	private _runResolve?: () => void;
	private _subscribers: Set<EventHandler> = new Set();
	private _extensionRunner?: ExtensionRunner;
	/** Typed extension event bus for structured lifecycle events */
	private _extensionBus: ExtensionEventBus | undefined;
	private _beforeAgentStart?: (
		promptText: string,
	) =>
		| Promise<{ messages?: Message[]; systemPrompt?: string } | undefined>
		| { messages?: Message[]; systemPrompt?: string }
		| undefined;

	constructor(options: AgentHarnessOptions) {
		const errors = validateConfig(options.config);
		throwOnValidationErrors(errors);

		this.config = options.config;
		this.backend = options.backend;
		this.cwd = options.cwd;
		this.maxIterations = options.maxIterations;
		this._extensionRunner = options.extensionRunner;
		this.loopDetector = new LoopDetector({
			maxHistory: options.config.loopDetectionWindow,
			exactRepeatWindow: options.config.loopDetectionWindow,
			degenerateWindow: options.config.degenerateLoopThreshold,
			stagnationWindow: options.config.stagnationThreshold,
			duplicateThreshold: options.config.duplicateToolThreshold,
			failureThreshold: options.config.toolFailureLoopThreshold,
		});
		this.idleTools = this.createToolRegistry(this.config.tools ?? []);
		this.msgManager = new MessageDeliveryManager({
			steeringMode: (options.config.steeringQueueMode ??
				"one-at-a-time") as DeliveryMode,
			followUpMode: (options.config.followUpQueueMode ??
				"one-at-a-time") as DeliveryMode,
		});
	}

	get phase(): HarnessPhase {
		return this._phase;
	}

	setOnPhaseChange(
		cb: (phase: HarnessPhase, prev: HarnessPhase) => void,
	): void {
		this.onPhaseChange = cb;
	}

	setOnSettled(cb: (nextTurnCount: number) => void): void {
		this.onSettled = cb;
	}

	setOnSavePoint(cb: () => void): void {
		this.onSavePoint = cb;
	}

	subscribe(handler: EventHandler): () => void {
		this._subscribers.add(handler);
		return () => this._subscribers.delete(handler);
	}

	async waitForIdle(): Promise<void> {
		if (this._phase === "idle") return;
		await this._runPromise;
	}

	setBeforeAgentStart(
		cb: (
			promptText: string,
		) =>
			| Promise<{ messages?: Message[]; systemPrompt?: string } | undefined>
			| { messages?: Message[]; systemPrompt?: string }
			| undefined,
	): void {
		this._beforeAgentStart = cb;
	}

	/** Set or replace the typed extension event bus. */
	setExtensionBus(bus: ExtensionEventBus | undefined): void {
		this._extensionBus = bus;
	}

	setExtensionRunner(runner: ExtensionRunner | undefined): void {
		this._extensionRunner = runner;
	}

	// ── Stream options management ──────────────────────────────────────────

	getStreamOptions(): AgentHarnessStreamOptions {
		return { ...this._streamOptions };
	}

	setStreamOptions(options: Partial<AgentHarnessStreamOptions>): void {
		this._streamOptions = { ...this._streamOptions, ...options };
		this.config.streamOptions = this._streamOptions;
	}

	getTimeoutMs(): number | undefined {
		return this._streamOptions.timeoutMs;
	}

	setTimeoutMs(ms: number | undefined): void {
		this._streamOptions = { ...this._streamOptions, timeoutMs: ms };
		this.config.streamOptions = this._streamOptions;
	}

	getMaxRetries(): number | undefined {
		return this._streamOptions.maxRetries;
	}

	setMaxRetries(count: number | undefined): void {
		this._streamOptions = { ...this._streamOptions, maxRetries: count };
		this.config.streamOptions = this._streamOptions;
	}

	getCacheRetention(): string | undefined {
		return this._streamOptions.cacheRetention;
	}

	setCacheRetention(retention: string | undefined): void {
		this._streamOptions = { ...this._streamOptions, cacheRetention: retention };
		this.config.streamOptions = this._streamOptions;
	}

	private transition(to: HarnessPhase, op: string): void {
		if (!PHASE_TRANSITIONS[this._phase].includes(to)) {
			throw new HarnessBusyError(op, this._phase, "idle");
		}
		const prev = this._phase;
		this._phase = to;
		if (prev !== to) this.onPhaseChange?.(prev, to);
	}

	private async runInPhase<T>(
		phase: HarnessPhase,
		op: string,
		fn: () => Promise<T>,
	): Promise<T> {
		this.transition(phase, op);
		try {
			return await fn();
		} finally {
			this.transition("idle", op);
		}
	}

	private assertIdle(op: string): void {
		if (this._phase !== "idle")
			throw new HarnessBusyError(op, this._phase, "idle");
	}

	// ── Structural operation: prompt ───────────────────────────────────────

	async prompt(userMessage: string): Promise<Message[]> {
		this._runPromise = new Promise<void>((resolve) => {
			this._runResolve = resolve;
		});

		if (this.autoCompactionSettings.enabled) {
			const compacted = await this.runAutoCompaction("auto");
			if (compacted) {
				return this.prompt(userMessage);
			}
		}

		// Initialize output guard if not already done
		if (!this.outputGuard && this.config.contextWindowTokens) {
			this.setOutputGuardConfig({
				maxRetries: this.config.streamOptions?.maxRetries ?? 3,
				retryBaseDelayMs: 500,
				maxRetryDelayMs: this.config.streamOptions?.maxRetryDelayMs ?? 15000,
			});
		}

		return this.runInPhase("turn", "prompt", async () => {
			this.abortController = new AbortController();

			if (!this._hasStartedSession) {
				await this.emitSessionStart("startup");
			}

			this.checkpoints.push([...this.history]);
			if (this.checkpoints.length > MAX_CHECKPOINTS) {
				this.checkpoints.shift();
			}
			beginFileFrame();

			const promptText = userMessage;
			const snapshot = await this.createTurnSnapshot(
				promptText,
				this.abortController.signal,
			);

			try {
				this.loopConfig = snapshot.config;
				const newMessages = await runAgentLoop(
					{
						systemPrompt: snapshot.config.systemPrompt,
						messages: snapshot.initialMessages,
						tools: snapshot.config.tools,
						cwd: this.cwd,
					},
					[createUserMessage(snapshot.promptText)],
					{
						...snapshot.config,
						backend: this.backend,
						signal: snapshot.signal,
						maxIterations: this.maxIterations,
						outputGuard: this.outputGuard,
						extensionBus: this._extensionBus,
					} satisfies RunAgentLoopConfig,
					async (event) => {
						await this.handleAgentEvent(event);
					},
				);
				const result = [
					{
						role: "system" as const,
						content:
							snapshot.config.systemPrompt ?? "You are a helpful assistant.",
					},
					...snapshot.initialMessages.filter(
						(message) => message.role !== "system",
					),
					...newMessages,
				];
				this.history = result;
				return result;
			} finally {
				this.abortController = null;
				this.msgManager.queue.clear();
				this.emitQueueChange();
				this._runResolve?.();
				this._runPromise = undefined;
				this._runResolve = undefined;
				const nextTurnCount = this._nextTurnQueue.length;
				this.onSettled?.(nextTurnCount);
				this.emitToSubscribers({ type: "settled", nextTurnCount });
			}
		});
	}

	/**
	 * Resume the agent loop from existing history without injecting a new user message.
	 * The last message in history must be a user or tool-result message (not assistant).
	 * Mirrors pi's agent.continue() — used when the agent stopped prematurely and
	 * the caller wants to re-enter the loop without fabricating a follow-up prompt.
	 */
	async continue(): Promise<Message[]> {
		const nonSystem = this.history.filter((m) => m.role !== "system");
		if (nonSystem.length === 0) {
			throw new Error("Cannot continue: no messages in history");
		}
		const last = nonSystem[nonSystem.length - 1];
		if (last?.role === "assistant") {
			throw new Error("Cannot continue from message role: assistant");
		}

		this._runPromise = new Promise<void>((resolve) => {
			this._runResolve = resolve;
		});

		if (!this.outputGuard && this.config.contextWindowTokens) {
			this.setOutputGuardConfig({
				maxRetries: this.config.streamOptions?.maxRetries ?? 3,
				retryBaseDelayMs: 500,
				maxRetryDelayMs: this.config.streamOptions?.maxRetryDelayMs ?? 15000,
			});
		}

		return this.runInPhase("turn", "continue", async () => {
			this.abortController = new AbortController();

			if (!this._hasStartedSession) {
				await this.emitSessionStart("startup");
			}

			this.checkpoints.push([...this.history]);
			if (this.checkpoints.length > MAX_CHECKPOINTS) {
				this.checkpoints.shift();
			}
			beginFileFrame();

			const snapshot = await this.createContinueSnapshot(
				this.abortController.signal,
			);

			try {
				this.loopConfig = snapshot.config;
				const newMessages = await runAgentLoopContinue(
					{
						systemPrompt: snapshot.config.systemPrompt,
						messages: snapshot.initialMessages,
						tools: snapshot.config.tools,
						cwd: this.cwd,
					},
					{
						...snapshot.config,
						backend: this.backend,
						signal: snapshot.signal,
						maxIterations: this.maxIterations,
						outputGuard: this.outputGuard,
						extensionBus: this._extensionBus,
					} satisfies RunAgentLoopConfig,
					async (event) => {
						await this.handleAgentEvent(event);
					},
				);
				const result = [
					{
						role: "system" as const,
						content:
							snapshot.config.systemPrompt ?? "You are a helpful assistant.",
					},
					...snapshot.initialMessages.filter((m) => m.role !== "system"),
					...newMessages,
				];
				this.history = result;
				return result;
			} finally {
				this.abortController = null;
				this.msgManager.queue.clear();
				this.emitQueueChange();
				this._runResolve?.();
				this._runPromise = undefined;
				this._runResolve = undefined;
				const nextTurnCount = this._nextTurnQueue.length;
				this.onSettled?.(nextTurnCount);
				this.emitToSubscribers({ type: "settled", nextTurnCount });
			}
		});
	}

	private async createContinueSnapshot(
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		const config = this.withDrainHook(this.withExtensionRuntime(this.config));
		const streamOptions = { ...this._streamOptions };
		config.streamOptions = streamOptions;
		return {
			promptText: "",
			initialMessages: [...this.history],
			config,
			streamOptions,
			signal,
		};
	}

	private async createTurnSnapshot(
		promptText: string,
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		const extensionBeforeStart =
			await this.runExtensionBeforeAgentStart(promptText);
		const beforeStart = await this._beforeAgentStart?.(promptText);

		const initialMessages: Message[] = [...this.history];

		const injectedMessages = [
			...(extensionBeforeStart?.messages ?? []),
			...(beforeStart?.messages ?? []),
		];
		if (injectedMessages.length) {
			initialMessages = [...injectedMessages, ...initialMessages];
		}

		const systemPrompt =
			beforeStart?.systemPrompt ?? extensionBeforeStart?.systemPrompt;
		const baseConfig = systemPrompt
			? { ...this.config, systemPrompt }
			: this.config;
		const config = this.withDrainHook(this.withExtensionRuntime(baseConfig));
		const streamOptions = { ...this._streamOptions };
		config.streamOptions = streamOptions;

		return {
			promptText,
			initialMessages,
			config,
			streamOptions,
			signal,
		};
	}

	private async runExtensionBeforeAgentStart(
		promptText: string,
	): Promise<{ messages?: Message[]; systemPrompt?: string } | undefined> {
		if (!this._extensionRunner?.hasHandlers("user_prompt_submit")) {
			return undefined;
		}
		const result = await this._extensionRunner.emit({
			type: "user_prompt_submit",
			context: {
				sessionId: this._sessionId || "",
				cwd: this.cwd || "",
				prompt: promptText,
			},
		});
		if (!result || typeof result !== "object") return undefined;
		const value = result as { messages?: Message[]; systemPrompt?: string };
		return {
			messages: Array.isArray(value.messages) ? value.messages : undefined,
			systemPrompt:
				typeof value.systemPrompt === "string" ? value.systemPrompt : undefined,
		};
	}

	private withExtensionRuntime(config: AgentConfig): AgentConfig {
		const runner = this._extensionRunner;
		if (!runner) return config;

		const extensionTools = runner
			.getTools()
			.map((tool) => this.wrapExtensionTool(tool));
		const tools = [...(config.tools ?? []), ...extensionTools];
		const extensionHooks = runner.getHooks();

		const builtinHooks = buildBuiltinHooks({
			config,
			contextWindowTokens: () => config.contextWindowTokens,
			toolDefs: () => tools,
			loopDetector: this.loopDetector,
		});

		const layers: HookLayer[] = [
			{ source: "builtin", hooks: builtinHooks },
			{ source: "extensions", hooks: extensionHooks },
			{ source: "user", hooks: config.hooks },
		];

		return {
			...config,
			tools,
			hooks: composeHooks(layers, undefined, config.onHookEvent),
		};
	}

	private wrapExtensionTool(tool: RegisteredTool): Tool {
		return {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters as unknown as Record<string, unknown>,
			execute: async (args, ctx) => {
				const result = await tool.execute(
					`extension_${tool.name}_${Date.now()}`,
					args,
					{
						toolCall: {
							id: `extension_${tool.name}`,
							name: tool.name,
							arguments: JSON.stringify(args),
						},
						cwd: ctx.cwd ?? this.cwd ?? "",
						sessionId: this._sessionId || "",
					},
				);
				return { content: result.content, details: result.details };
			},
		};
	}

	private persistTurnMessages(messages: Message[]): void {
		if (!this._session) return;
		for (const message of messages) {
			try {
				this._session.append({
					role: message.role,
					content: message.content,
					tool_call_id: message.tool_call_id,
					tool_calls: message.tool_calls,
					name: message.name,
					timestamp: message.timestamp ?? Date.now(),
				});
			} catch {
				// Session persistence must never break a completed turn.
			}
		}
	}

	private async handleAgentEvent(event: AgentEvent): Promise<void> {
		if (event.type === "message_end" && event.message) {
			this.persistTurnMessages([event.message]);
		}
		await this.emitExtensionAgentEvent(event);
		this.loopConfig?.onEvent?.(event);
	}

	private async emitExtensionAgentEvent(event: AgentEvent): Promise<void> {
		const runner = this._extensionRunner;
		if (!runner) return;
		const context = {
			sessionId: this._sessionId || "",
			cwd: this.cwd || "",
			...event,
		};
		switch (event.type) {
			case "agent_start":
			case "agent_end":
			case "turn_start":
			case "turn_end":
			case "message_start":
			case "message_update":
			case "message_end":
			case "tool_call_start":
			case "tool_call_end":
				await runner.emit({ type: event.type, context });
				break;
		}
	}

	// ── Output guard setup ───────────────────────────────────────────────

	setOutputGuardConfig(config: {
		maxRetries?: number;
		retryBaseDelayMs?: number;
		maxRetryDelayMs?: number;
		autoCompactOnContextFull?: boolean;
		maxEmptyResponses?: number;
	}): void {
		this.outputGuard = new OutputGuard({
			...config,
			onCompact: async () => {
				const result = await this.compact();
				return result ?? null;
			},
			onEvent: (event) => {
				this.emitToSubscribers(event);
			},
		});
	}

	getOutputGuard(): OutputGuard | null {
		return this.outputGuard;
	}

	getLoopDetector(): LoopDetector {
		return this.loopDetector;
	}

	// ── Runtime config setters ─────────────────────────────────────────────

	setSystemPrompt(systemPrompt: string): void {
		this.config.systemPrompt = systemPrompt;
	}

	setTemperature(temperature: number): void {
		this.config.temperature = temperature;
	}

	setMaxTokens(maxTokens: number): void {
		this.config.maxTokens = maxTokens;
	}

	setTools(tools: Tool[]): void {
		this.config.tools = tools;
		this.idleTools = this.createToolRegistry(tools);
		this._session?.appendActiveToolsChange(tools.map((t) => t.name));
		this.emitToSubscribers({
			type: "tools_update",
			toolNames: tools.map((t) => t.name),
		});
	}

	private createToolRegistry(tools: Tool[]): ToolRegistry {
		const registry = new ToolRegistry({
			cwd: this.cwd,
			onQuestionRequest: this.config.onQuestionRequest,
		});
		registry.registerMany(tools);
		return registry;
	}

	// ── Queue operations ───────────────────────────────────────────────────

	steer(text: string): void {
		if (this._phase !== "turn")
			throw new HarnessBusyError("steer", this._phase, "turn");
		this.msgManager.queue.steering(text);
		this.emitQueueChange();
		if (this.config.steeringInterrupt) {
			this.abortController?.abort();
		}
	}

	followUp(text: string): void {
		this.msgManager.queue.followUp(text);
		this.emitQueueChange();
	}

	nextTurn(text: string): void {
		this._nextTurnQueue.push(text);
		this.emitQueueChange();
	}

	abort(): AbortResult {
		const q = this.msgManager.queue;
		const clearedSteering = q.getSteering().map((m) => m.content);
		const clearedFollowUp = q.getFollowUp().map((m) => m.content);
		const clearedNextTurn = [...this._nextTurnQueue];
		this.abortController?.abort();
		this.msgManager.queue.clear();
		this._nextTurnQueue = [];
		this.emitQueueChange();
		this.emitToSubscribers({
			type: "abort",
			clearedSteering,
			clearedFollowUp,
			clearedNextTurn,
		});
		this.emitSessionEnd("abort").catch(() => {});
		return { clearedSteering, clearedFollowUp, clearedNextTurn };
	}

	// ── Queue state ────────────────────────────────────────────────────────

	getQueues(): HarnessQueues {
		const q = this.msgManager.queue;
		return {
			steering: q.getSteering().map((m) => m.content),
			followUp: q.getFollowUp().map((m) => m.content),
			nextTurn: [...this._nextTurnQueue],
		};
	}

	setOnQueueChange(cb: (queues: HarnessQueues) => void): void {
		this.onQueueChange = cb;
	}

	clearQueues(): HarnessQueues {
		const cleared = this.getQueues();
		this.msgManager.queue.clear();
		this._nextTurnQueue = [];
		this.emitQueueChange();
		return cleared;
	}

	private emitQueueChange(): void {
		const queues = this.getQueues();
		this.onQueueChange?.(queues);
		this.emitToSubscribers({
			type: "queue_update",
			steering: queues.steering,
			followUp: queues.followUp,
			nextTurn: queues.nextTurn,
		});
		void this._extensionRunner?.emit({
			type: "queue_update",
			context: {
				sessionId: this._sessionId || "",
				cwd: this.cwd || "",
				...queues,
			},
		});
	}

	setSteeringMode(mode: QueueMode): void {
		this.msgManager.setMode("steering", mode as DeliveryMode);
	}

	getSteeringMode(): QueueMode {
		return this.msgManager.getMode("steering") as QueueMode;
	}

	setFollowUpMode(mode: QueueMode): void {
		this.msgManager.setMode("followUp", mode as DeliveryMode);
	}

	getFollowUpMode(): QueueMode {
		return this.msgManager.getMode("followUp") as QueueMode;
	}

	// ── Plugin lifecycle hooks ─────────────────────────────────────────────

	setHooksEnabled(enabled: boolean): void {
		this._hooksEnabled = enabled;
		this.config.runtimeHooksEnabled = enabled;
	}

	setSessionId(id: string): void {
		this._sessionId = id;
		this.config.hookSessionId = id;
	}

	setTranscriptPath(path: string): void {
		this._transcriptPath = path;
		this.config.hookTranscriptPath = path;
	}

	private async emitSessionStart(source: string = "startup"): Promise<void> {
		if (!this._hooksEnabled) return;
		try {
			await runSessionStartHooks({
				source,
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
			});
			this._hasStartedSession = true;
		} catch {
			// must not block the session
		}
	}

	private async emitSessionEnd(reason: string = "other"): Promise<void> {
		if (!this._hooksEnabled) return;
		try {
			await runHookEvent("SessionEnd", {
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
				reason,
			});
		} catch {
			// must not block cleanup
		}
	}

	private async emitPreCompact(
		ctx?: BeforeCompactContext,
	): Promise<BeforeCompactResult | undefined> {
		let hookResult: BeforeCompactResult | undefined;
		if (ctx) {
			try {
				hookResult =
					(await this.config.internalHooks?.beforeCompact?.(ctx)) ??
					(await this.config.hooks?.beforeCompact?.(ctx)) ??
					undefined;
			} catch {
				// must not block compaction
			}
		}
		if (!this._hooksEnabled) return hookResult;
		try {
			await runHookEvent("PreCompact", {
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
			});
		} catch {
			// must not block compaction
		}
		return hookResult;
	}

	private async emitPostCompact(): Promise<void> {
		if (!this._hooksEnabled) return;
		try {
			await runHookEvent("PostCompact", {
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
			});
		} catch {
			// must not block compaction
		}
	}

	// ── Auto-compaction ────────────────────────────────────────────────────

	setAutoCompactionSettings(settings: Partial<CompactionSettings>): void {
		this.autoCompactionSettings = {
			...this.autoCompactionSettings,
			...settings,
		};
	}

	enableAutoCompaction(enabled: boolean): void {
		this.autoCompactionSettings.enabled = enabled;
	}

	setOnCompaction(
		cb: (
			reason: "auto" | "manual",
			tokensBefore: number,
			tokensAfter: number,
		) => void,
	): void {
		this.onCompaction = cb;
	}

	// ── Session & history ──────────────────────────────────────────────────

	async enableSession(baseDir?: string): Promise<void> {
		const sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		this._sessionBaseDir = baseDir;
		this._session = new Session(sessionId, { baseDir, enabled: true });
		this._sessionId = sessionId;
		await this.emitSessionStart("startup");
	}

	async resumeSession(sessionId: string, baseDir?: string): Promise<boolean> {
		try {
			const sessionBaseDir = baseDir ?? this._sessionBaseDir;
			const session = new Session(sessionId, {
				baseDir: sessionBaseDir,
				enabled: true,
			});
			const persisted = session.load();
			if (persisted.length > 0) {
				const messages: Message[] = persisted.map((m) => ({
					role: m.role as Message["role"],
					content: m.content,
					tool_call_id: m.tool_call_id,
					tool_calls: m.tool_calls,
					name: m.name,
					timestamp: m.timestamp,
				}));
				this.setActiveHistory(messages.filter((m) => m.role !== "system"));
			}
			this._sessionBaseDir = sessionBaseDir;
			this._session = session;
			this._sessionId = sessionId;
			await this.emitSessionStart("resume");
			return true;
		} catch {
			return false;
		}
	}

	listSessions(): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}> {
		try {
			const baseDir =
				this._sessionBaseDir ??
				(this._session ? `${this._session.dirPath}/../..` : undefined);
			const manager = new SessionManager({ baseDir });
			return manager
				.listSessions()
				.map(
					(m: {
						id: string;
						name?: string;
						messageCount: number;
						lastActivity: number;
					}) => ({
						id: m.id,
						name: m.name,
						messageCount: m.messageCount,
						lastActivity: m.lastActivity,
					}),
				);
		} catch {
			return [];
		}
	}

	get messages(): Message[] {
		return this.history;
	}

	clearHistory(): void {
		this.emitSessionEnd("reset").catch(() => {});
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.setActiveHistory([]);
		this.emitSessionStart("clear").catch(() => {});
		this._hasStartedSession = false;
	}

	setHistory(messages: Message[]): void {
		this.assertIdle("setHistory");
		this.emitSessionEnd("switch").catch(() => {});
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.setActiveHistory(messages.filter((m) => m.role !== "system"));
		this.emitSessionStart("resume").catch(() => {});
		this._hasStartedSession = false;
	}

	rewind(): { messages: number; filesRestored: number } | null {
		this.assertIdle("rewind");
		const snapshot = this.checkpoints.pop();
		if (!snapshot) return null;
		this.branches = [];
		this.setActiveHistory(snapshot);
		const filesRestored = restoreFileFrame() ?? 0;
		return { messages: snapshot.length, filesRestored };
	}

	// ── Branching ──────────────────────────────────────────────────────────

	/**
	 * Fork the current conversation into a new branch.
	 * The parent history is saved; the branch continues from the fork point.
	 * @param options.customSummary Optional pre-computed summary (e.g. from extension).
	 * @returns The new branch ID.
	 */
	fork(customSummary?: BranchSummaryData): string {
		this.assertIdle("fork");
		const current = this.activeHistory();
		const branch: (typeof this.branches)[number] = {
			id: `branch_${++this.branchSeq}`,
			parent: current,
			forkedAt: current.length,
			summary: customSummary ?? null,
		};
		this.branches.push(branch);
		this.setActiveHistory([...current]);
		return branch.id;
	}

	/**
	 * Generate a structured summary of the active branch and merge it back.
	 * Collects file operations and produces a structured summary with goal,
	 * progress, key decisions, and next steps.
	 * @param options.customInstructions Optional LLM prompt instructions.
	 * @returns The full structured summary, or null if branch was empty.
	 */
	async branchSummary(options?: {
		customInstructions?: string;
	}): Promise<string | null> {
		this.assertIdle("branchSummary");
		const branch = this.branches.at(-1);
		if (!branch) return null;

		const current = this.activeHistory();
		const diverged = current.slice(branch.forkedAt);
		if (!diverged.length) {
			this.branches.pop();
			this.setActiveHistory(branch.parent);
			return null;
		}

		return this.runInPhase("branch_summary", "branchSummary", async () => {
			// Collect messages with file ops and token budget
			const collection = collectMessagesForBranchSummary(
				current,
				branch.parent,
				branch.forkedAt,
				this.config.contextWindowTokens
					? Math.floor(this.config.contextWindowTokens * 0.5)
					: 0,
			);

			const { fileOps } = collection;

			// Generate structured summary via LLM
			const summary = await this.generateBranchSummaryText(
				collection.messages,
				options?.customInstructions,
				fileOps,
			);

			// Update branch with summary
			branch.summary = summary;
			this.branches.pop();

			if (!summary) {
				return null;
			}

			// Append summary message to history
			const summaryEntry: Message = {
				role: "assistant",
				content: summary.full,
				tool_calls: [],
			};

			this.setActiveHistory([...branch.parent, summaryEntry]);
			return summary.full;
		});
	}

	/**
	 * Discard the active branch without merging. Restores parent history.
	 */
	discardBranch(): boolean {
		this.assertIdle("discardBranch");
		const branch = this.branches.pop();
		if (!branch) return false;
		this.setActiveHistory(branch.parent);
		return true;
	}

	/**
	 * Navigate to a previous checkpoint with optional branch summary merge.
	 * Like rewind but accepts a checkpoint index and optionally summarizes the
	 * abandoned path before navigating back.
	 * @param checkpointIndex Index in the checkpoints array (0 = oldest).
	 * @param options.summarize Whether to summarize the abandoned path.
	 * @param options.customInstructions Optional LLM instructions for summarization.
	 * @returns Summary of the abandoned path, or null.
	 */
	async navigateToCheckpoint(
		checkpointIndex: number,
		options?: { summarize?: boolean; customInstructions?: string },
	): Promise<BranchSummaryData | null> {
		this.assertIdle("navigateToCheckpoint");
		if (checkpointIndex < 0 || checkpointIndex >= this.checkpoints.length) {
			return null;
		}

		const target = this.checkpoints[checkpointIndex];
		const abandoned = this.history;

		// Summarize abandoned path if requested
		let summaryText: BranchSummaryData | null = null;
		if (options?.summarize && abandoned.length > target.length) {
			const abandonedMessages = abandoned.slice(target.length);
			const fileOps = extractFileOpsFromMessages(abandonedMessages);
			summaryText = await this.generateBranchSummaryText(
				abandonedMessages,
				options.customInstructions,
				fileOps,
			);
			if (summaryText) {
				this.history.push({
					role: "assistant",
					content: summaryText.full,
					tool_calls: [],
				});
			}
		}

		this.history = [...target];
		this.checkpoints = this.checkpoints.slice(0, checkpointIndex);
		this.branches = [];
		return summaryText;
	}

	/**
	 * Visualize the branch tree as an ASCII art string.
	 * Shows parent/child relationships with depth indicators.
	 */
	branchTree(): string {
		if (this.branches.length === 0) {
			return "No active branches.";
		}

		const lines: string[] = [];
		lines.push(`Branches (${this.branches.length}):`);

		for (const branch of this.branches) {
			const depth = this.branches.indexOf(branch);
			const prefix = "  ".repeat(depth) + (depth > 0 ? "└─ " : "");
			lines.push(
				`${prefix}[${branch.id}] forked at message ${branch.forkedAt}`,
			);

			if (branch.summary) {
				// Show summary preview (first line of goal)
				const goal = branch.summary.goal;
				const preview = goal.length > 60 ? goal.slice(0, 60) + "..." : goal;
				lines.push(`${"  ".repeat(depth + 1)}Goal: ${preview}`);

				if (branch.summary.progress.done.length > 0) {
					const doneCount = branch.summary.progress.done.length;
					lines.push(`${"  ".repeat(depth + 1)}Done: ${doneCount} items`);
				}
				if (branch.summary.progress.inProgress.length > 0) {
					lines.push(
						`${"  ".repeat(depth + 1)}In Progress: ${branch.summary.progress.inProgress.length} items`,
					);
				}
				if (branch.summary.progress.blocked.length > 0) {
					lines.push(
						`${"  ".repeat(depth + 1)}Blocked: ${branch.summary.progress.blocked.length} items`,
					);
				}
			}
		}

		return lines.join("\n");
	}

	listBranches(): BranchInfo[] {
		return this.branches.map((b, i) => ({
			id: b.id,
			depth: i + 1,
			summary: b.summary,
			forkedAt: b.forkedAt,
		}));
	}

	// ── Conversation management ────────────────────────────────────────────

	private activeHistory(): Message[] {
		return this.history;
	}

	private setActiveHistory(messages: Message[]): void {
		this.history = messages;
	}

	// ── Compaction ─────────────────────────────────────────────────────────

	async compact(): Promise<number | null> {
		this.assertIdle("compact");
		const messages = this.history;
		if (!messages.length) return null;
		const before = this.estimatePayloadTokens();

		return this.runInPhase("compaction", "compact", async () => {
			this.emitToSubscribers({ type: "compaction_start", reason: "manual" });
			await this._extensionRunner?.emit({
				type: "before_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason: "manual",
					tokensBefore: before,
				},
			});
			const preResult = await this.emitPreCompact({
				messages,
				tokensBefore: before,
				reason: "manual",
			});
			if (preResult?.cancel) {
				this.emitToSubscribers({
					type: "compaction_end",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: before,
					changed: false,
				});
				return 0;
			}

			const result = await compactMessages(messages, {
				reason: "manual",
				summarize: preResult?.summary
					? async () => preResult.summary!
					: (older, system) =>
							this.generateSummary(
								older.map((m) => ({
									role: m.role as Message["role"],
									content: m.content ?? "",
									tool_call_id: m.tool_call_id,
									tool_calls: m.tool_calls,
									name: m.name,
									timestamp: m.timestamp,
								})),
								system.map((m) => ({
									role: m.role as Message["role"],
									content: m.content ?? "",
									tool_call_id: m.tool_call_id,
									tool_calls: m.tool_calls,
									name: m.name,
									timestamp: m.timestamp,
								})),
							),
			});

			if (!result.changed) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction_end",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: before,
					changed: false,
				});
				return 0;
			}
			this.history = result.messages;
			const after = this.estimatePayloadTokens();
			this.onCompaction?.("manual", before, after);
			await this.emitPostCompact();
			await this._extensionRunner?.emit({
				type: "after_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: after,
					changed: true,
				},
			});
			this.emitToSubscribers({
				type: "compaction_end",
				reason: "manual",
				tokensBefore: before,
				tokensAfter: after,
				changed: true,
			});
			return before - after;
		});
	}

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.history;
		if (!messages.length || !this.autoCompactionSettings.enabled) return false;

		return this.runInPhase("compaction", "autoCompact", async () => {
			this.emitToSubscribers({ type: "compaction_start", reason });

			if (!this.shouldCompact(messages)) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction_end",
					reason,
					tokensBefore: this.estimateContextTokens(),
					tokensAfter: this.estimateContextTokens(),
					changed: false,
				});
				return false;
			}

			const before = this.estimateContextTokens();
			const preResult = await this.emitPreCompact({
				messages,
				tokensBefore: before,
				reason,
			});
			if (preResult?.cancel) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction_end",
					reason,
					tokensBefore: before,
					tokensAfter: before,
					changed: false,
				});
				return false;
			}

			await this._extensionRunner?.emit({
				type: "before_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason,
					tokensBefore: before,
				},
			});

			const result = await compactMessages(messages, {
				reason,
				summarize: preResult?.summary
					? async () => preResult.summary!
					: (older, system) =>
							this.generateSummary(
								older.map((m) => ({
									role: m.role as Message["role"],
									content: m.content ?? "",
									tool_call_id: m.tool_call_id,
									tool_calls: m.tool_calls,
									name: m.name,
									timestamp: m.timestamp,
								})),
								system.map((m) => ({
									role: m.role as Message["role"],
									content: m.content ?? "",
									tool_call_id: m.tool_call_id,
									tool_calls: m.tool_calls,
									name: m.name,
									timestamp: m.timestamp,
								})),
							),
			});

			if (!result.changed) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction_end",
					reason,
					tokensBefore: before,
					tokensAfter: before,
					changed: false,
				});
				return false;
			}

			this.history = result.messages;
			const after = this.estimatePayloadTokens();
			this.onCompaction?.(reason, before, after);
			await this.emitPostCompact();
			await this._extensionRunner?.emit({
				type: "after_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason,
					tokensBefore: before,
					tokensAfter: after,
					changed: true,
				},
			});
			this.emitToSubscribers({
				type: "compaction_end",
				reason,
				tokensBefore: before,
				tokensAfter: after,
				changed: true,
			});
			return true;
		});
	}

	// ── Summary generation ─────────────────────────────────────────────────

	private async generateSummary(
		messages: Message[],
		systemMessages: Message[],
	): Promise<string | null> {
		try {
			const chatMessages = [
				...systemMessages,
				{
					role: "user" as const,
					content:
						"Summarize the following conversation history concisely. " +
						"Focus on key decisions, actions taken, files modified, " +
						"and any important context that should be retained. " +
						"Be brief but preserve all actionable information.",
				},
				...messages,
			];

			const response = await this.backend.generate(
				convertToChatFormat(chatMessages),
				{
					temperature: this.config.temperature ?? 0.3,
					maxTokens: Math.min(2048, (this.config.maxTokens ?? 4096) / 2),
					thinkingLevel: this.config.thinkingLevel,
				},
			);

			return response.content?.trim() || null;
		} catch {
			return null;
		}
	}

	/**
	 * Generate a structured branch summary via LLM.
	 * Collects file ops, builds the branch summary prompt, calls the LLM,
	 * and returns a BranchSummaryData object.
	 */
	private async generateBranchSummaryText(
		messages: Message[],
		customInstructions?: string,
		fileOps?: { read: Set<string>; modified: Set<string> },
	): Promise<BranchSummaryData | null> {
		if (messages.length === 0) return null;

		const extractedOps = fileOps ?? extractFileOpsFromMessages(messages);
		const { readFiles, modifiedFiles } = computeFileLists(extractedOps);

		try {
			// Build the branch summary prompt
			const conversationText = serializeMessages(messages);

			let instructions = `Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[One sentence: what was the user trying to accomplish]

## Constraints & Preferences
- [constraint 1]
- [constraint 2]
- (none) if none were mentioned

## Progress
### Done
- [x] [completed task/change]
### In Progress
- [ ] [started but unfinished]
### Blocked
- [issue preventing progress, if any]

## Key Decisions
- **[decision]**: [brief rationale]

## Next Steps
1. [first step to continue]
2. [second step, if any]

Preserve exact file paths, function names, and error messages. Be concise.`;

			if (customInstructions) {
				instructions += `\n\nAdditional focus: ${customInstructions}`;
			}

			const response = await this.backend.generate(
				[
					{
						role: "user",
						content: `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`,
					},
				] as { role: string; content: string }[],
				{
					temperature: 0.3,
					maxTokens: Math.min(2048, (this.config.maxTokens ?? 4096) / 2),
					thinkingLevel: this.config.thinkingLevel,
				},
			);

			const summaryText = response.content?.trim();
			if (!summaryText) return null;

			// Parse the structured fields from LLM output
			const parsed = parseBranchSummary(summaryText);

			// Append file operations to full text
			const fileOpsText = formatFileOperations(readFiles, modifiedFiles);
			const full = `${summaryText}\n${fileOpsText}`;

			return {
				goal: parsed.goal || "Branch conversation",
				constraints: parsed.constraints || [],
				progress: parsed.progress || { done: [], inProgress: [], blocked: [] },
				keyDecisions: parsed.keyDecisions || [],
				nextSteps: parsed.nextSteps || [],
				full,
			};
		} catch {
			// Fallback: return a basic summary from raw messages
			const fallback = `Branch ${messages.length} messages explored.`;
			return {
				goal: "Branch exploration",
				constraints: [],
				progress: { done: [], inProgress: [], blocked: [] },
				keyDecisions: [],
				nextSteps: [],
				full: fallback,
			};
		}
	}

	// ── Tool registry ──────────────────────────────────────────────────────

	get tools(): ToolRegistry {
		return this.idleTools;
	}

	// ── Config getters ─────────────────────────────────────────────────────

	getTemperature(): number {
		return this.config.temperature ?? 0.7;
	}

	getToolCount(): number {
		return this.config.tools?.length ?? 0;
	}

	// ── Model cycling with thinking level clamping ────────────────────────

	getModel(): string {
		return this.config.model;
	}

	getBaseUrl(): string {
		return this.config.baseUrl;
	}

	getModels(): string[] {
		const currentName = this.config.model;
		if (this.config.models) {
			const names = this.config.models.map((m) => m.name);
			return [currentName, ...names];
		}
		return [currentName];
	}

	/** Resolve the baseUrl for a given model identifier. */
	private getModelUrl(modelName: string): string {
		const models = this.config.models;
		if (models) {
			const found = models.find((m) => m.model === modelName);
			if (found?.url) {
				return found.url;
			}
		}
		return this.config.baseUrl;
	}

	/** Set the models array for cycling. */
	setModels(models: AgentModelConfig[]): void {
		this.config.models = models;
	}

	/** Clamp thinking level to what the given model supports. */
	private clampThinkingLevel(level: string): ThinkingLevel {
		const levels: ThinkingLevel[] = [
			"off",
			"minimal",
			"low",
			"medium",
			"high",
			"xhigh",
		];
		const idx = levels.indexOf(level as ThinkingLevel);
		if (idx >= 0) return level as ThinkingLevel;

		// Unknown level — default to medium
		return "medium";
	}

	cycleModel(direction: "forward" | "backward" = "forward"): string {
		// Build cycling ring: [currentModel, ...configuredModels].
		const configured = this.config.models ?? [];
		if (configured.length === 0) {
			return this.config.model;
		}

		// Build array of {name, model} for cycling.
		const cycleModels: Array<{ name: string; model: string }> = configured.map(
			(m) => ({ name: m.name, model: m.model }),
		);

		// Check if current model is already in the list.
		const currentInList = cycleModels.some(
			(m) => m.model === this.config.model,
		);
		if (!currentInList) {
			cycleModels.unshift({
				name: this.config.model,
				model: this.config.model,
			});
		}

		if (cycleModels.length <= 1) {
			return this.config.model;
		}

		const currentIndex = cycleModels.findIndex(
			(m) => m.model === this.config.model,
		);
		const nextIndex =
			direction === "forward"
				? (currentIndex + 1) % cycleModels.length
				: (currentIndex - 1 + cycleModels.length) % cycleModels.length;
		const next = cycleModels[nextIndex];
		const model = next?.model ?? this.config.model;
		const fromModel = this.config.model;

		// Switch baseUrl if target model has a custom url.
		const targetUrl = this.getModelUrl(model);
		if (targetUrl !== this.config.baseUrl) {
			this.config.baseUrl = targetUrl;
		}

		// Preserve thinking level, clamped to model capabilities
		const currentLevel = this.config.thinkingLevel ?? "off";
		const clampedLevel = this.clampThinkingLevel(currentLevel);

		if (clampedLevel !== currentLevel) {
			this.config.thinkingLevel = clampedLevel;
			this.emitToSubscribers({
				type: "thinking_level_clamped",
				level: clampedLevel,
				reason: `Model ${model} does not support ${currentLevel} thinking level`,
			});
		}

		this.config.model = model;
		this._session?.appendModelChange(model);
		this.emitToSubscribers({
			type: "model_cycle",
			model,
			fromModel,
			thinkingLevel: clampedLevel,
		});
		return model;
	}

	// ── Thinking level ─────────────────────────────────────────────────────

	getThinkingLevel(): string {
		return this.config.thinkingLevel ?? "off";
	}

	setThinkingLevel(level: string): void {
		const currentLevel = this.config.thinkingLevel;
		this.config.thinkingLevel = level as
			| "off"
			| "minimal"
			| "low"
			| "medium"
			| "high"
			| "xhigh";
		this._session?.appendThinkingLevelChange(this.config.thinkingLevel);
		this.emitToSubscribers({ type: "thinking_level_changed", level });
		// If level changed, emit model_cycle with updated thinking level
		if (level !== currentLevel) {
			this.emitToSubscribers({
				type: "model_cycle",
				model: this.config.model,
				fromModel: this.config.model,
				thinkingLevel: level,
			});
		}
	}

	// ── Model & provider ──────────────────────────────────────────────────

	setModel(model: string): void {
		const targetUrl = this.getModelUrl(model);
		if (targetUrl !== this.config.baseUrl) {
			this.config.baseUrl = targetUrl;
		}
		this.config.model = model;
		this._session?.appendModelChange(model);
		this.emitToSubscribers({ type: "model_update", model });
	}

	setBackend(backend: LLMBackend): void {
		this.backend = backend;
	}

	// ── Internals ──────────────────────────────────────────────────────────

	private emitToSubscribers(event: AgentEvent): void {
		for (const handler of this._subscribers) handler(event);
	}

	// ── Improved token estimation using serialized payload ─────────────────

	private estimatePayloadTokens(): number {
		return estimateChatPayloadTokens(
			this.messages,
			this.idleTools?.toToolDefinitions?.(),
		);
	}

	private estimateContextTokens(): number {
		return this.estimatePayloadTokens();
	}

	private shouldCompact(messages: Message[]): boolean {
		const settings = this.autoCompactionSettings;
		if (!settings.enabled) return false;
		const contextWindow = settings.contextWindow ?? 128000;
		const threshold = contextWindow - (settings.reserveTokens ?? 16384);
		const currentTokens = estimateChatPayloadTokens(messages);
		return currentTokens > threshold;
	}

	private withDrainHook(config: AgentConfig): AgentConfig {
		const internalHooks: AgentHooks = {};

		internalHooks.transformContext = ({ messages }) => {
			const pending = this._nextTurnQueue.splice(0);
			if (!pending.length) return undefined;
			this.emitQueueChange();
			const injected = pending.map((text) => createUserMessage(text));
			const last = messages.length - 1;
			const at = last >= 0 ? last : 0;
			return {
				messages: [
					...messages.slice(0, at),
					...injected,
					...messages.slice(at),
				],
			};
		};

		internalHooks.getSteeringMessages = async (): Promise<
			Message[] | undefined
		> => {
			const drained = this.msgManager.afterTurn();
			if (drained.length === 0) return undefined;
			return this.toInjectedMessages(drained.map((m) => m.content));
		};

		internalHooks.getFollowUpMessages = async (): Promise<
			Message[] | undefined
		> => {
			const drained = this.msgManager.onIdle();
			if (drained.length === 0) return undefined;
			return this.toInjectedMessages(drained.map((m) => m.content));
		};

		const originalOnEvent = config.onEvent;
		const wrappedOnEvent = (event: AgentEvent) => {
			originalOnEvent?.(event);
			if (event.type === "turn_end") this.onSavePoint?.();
			for (const handler of this._subscribers) handler(event);
		};

		return { ...config, internalHooks, onEvent: wrappedOnEvent };
	}

	private toInjectedMessages(texts: string[]): Message[] | undefined {
		if (!texts.length) return undefined;
		this.emitQueueChange();
		return texts.map((text) => createUserMessage(text));
	}
}
