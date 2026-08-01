// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//

import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "./file-checkpoints.ts";
import { randomUUID } from "node:crypto";
import type { LLMBackend } from "./backend.ts";
import {
	runAgentLoop,
	runAgentLoopContinue,
	type RunAgentLoopConfig,
} from "./agent-loop-runner.ts";
import { Session } from "./session.ts";
import { createUserMessage, estimateChatPayloadTokens } from "./messages.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import {
	MessageDeliveryManager,
	type DeliveryMode,
} from "../queue/manager.ts";
import type { ExtensionRunner, RegisteredTool } from "../extensions/index.ts";
import type { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";
import {
	composeHooks,
	buildBuiltinHooks,
	type HookLayer,
} from "../hooks/builtin/builtin-hooks.ts";
import {
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
	type ClaudeCodeHookLayer,
} from "../plugins/claude-code/hook-layer.ts";
import { LoopDetector } from "./guards/loop-detector.ts";
import { ThinkingLoopDetector } from "./guards/thinking-loop-detector.ts";
import {
	forkBranch,
	summarizeAndMergeBranch,
	navigateToCheckpoint as navigateToCheckpointHelper,
	renderBranchTree,
	listBranches as listBranchesHelper,
	type Branch,
} from "./harness/branching.ts";
import { runCompaction, shouldAutoCompact } from "./harness/compaction.ts";
import { cycleModel as cycleModelHelper, resolveModelUrl } from "./harness/model.ts";
import * as queueOps from "./harness/queue-ops.ts";
import type { QueueOpsDeps } from "./harness/queue-ops.ts";
import {
	emitSessionStart as emitSessionStartHelper,
	emitSessionEnd as emitSessionEndHelper,
	emitPreCompact as emitPreCompactHelper,
	emitPostCompact as emitPostCompactHelper,
	loadSessionMessages,
	listSessions as listSessionsHelper,
} from "./harness/session-lifecycle.ts";
import type { BranchSummaryData, BranchInfo } from "./summaries/types.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHarnessStreamOptions,
	AgentModelConfig,
	BeforeCompactContext,
	BeforeCompactResult,
	EventHandler,
	Message,
	QueueMode,
	Tool,
} from "./types.ts";
import type { CompactionSettings } from "../compaction/index.ts";
import { OutputGuard } from "./guards/output-guard.ts";
import { validateConfig, throwOnValidationErrors } from "./configuration/config-validator.ts";
import {
	createMemoryStore,
	formatMemoryPrompt,
	type MemoryStore,
} from "./memory.ts";
import {
	createRuntimeState,
	reduceRuntimeState,
	type AgentRuntimeState,
	type HarnessPhase,
} from "./runtime-state.ts";
import { withHarnessQueueHooks } from "../runtime/harness-queue-hooks.ts";
import { assertIdlePhase, assertPhaseTransition } from "./harness/phase.ts";
import type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
	HarnessTurnSnapshot,
} from "./harness/contracts.ts";

export type { BranchSummaryData, BranchInfo } from "./summaries/types.ts";

export type { AgentRuntimeState, HarnessPhase } from "./runtime-state.ts";
export { HarnessBusyError } from "./harness/phase.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./harness/contracts.ts";

// Conversation checkpoints: a snapshot of history is pushed before each
// prompt so a bad turn can be rewound. Bounded ring (newest last).
const MAX_CHECKPOINTS = 20;

export class AgentHarness {
	private config: AgentConfig;
	private backend: LLMBackend;
	private cwd?: string;
	private maxIterations?: number;

	private _phase: HarnessPhase = "idle";
	private runtime: AgentRuntimeState = createRuntimeState();
	private idleTools: ToolRegistry;
	private abortController: AbortController | null = null;
	private loopConfig: AgentConfig | null = null;
	private history: Message[] = [];
	private branches: Branch[] = [];
	private branchSeq = 0;
	private checkpoints: Message[][] = [];
	private _nextTurnQueue: string[] = [];
	private msgManager: MessageDeliveryManager;
	private loopDetector: LoopDetector;
	private thinkingLoopDetector: ThinkingLoopDetector | null;
	private onQueueChange?: (queues: HarnessQueues) => void;
	private onPhaseChange?: (phase: HarnessPhase, prev: HarnessPhase) => void;
	private onSettled?: (nextTurnCount: number) => void;
	private onSavePoint?: () => void;
	private onShutdown?: () => void;
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

	// ── Structured memory ────────────────────────────────────────────────
	private memoryStore: MemoryStore;
	private memoryEnabled = true;
	private memoryTurnCount = 0;

	// ── Output Guard ─────────────────────────────────────────────────────
	private outputGuard: OutputGuard | null = null;
	private _streamOptions: AgentHarnessStreamOptions = {};
	private _hooksEnabled: boolean;
	private _session?: Session;
	private _sessionBaseDir?: string;
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
	private _activeOperationId?: string;
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
		this._streamOptions = {
			...options.config.streamOptions,
			...(options.config.streamOptions?.timeoutMs === undefined &&
			options.config.turnTimeoutMs !== undefined
				? { timeoutMs: options.config.turnTimeoutMs }
				: {}),
		};
		this.config.streamOptions = this._streamOptions;
		this._hooksEnabled = options.config.runtimeHooksEnabled ?? true;
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
		this.thinkingLoopDetector =
			options.config.thinkingLoopDetectionEnabled !== false
				? new ThinkingLoopDetector({
						minThinkingLength: options.config.thinkingLoopMinThinkingLength,
						thinkingOnlyThreshold:
							options.config.thinkingLoopThinkingOnlyThreshold,
						escalationRatio: options.config.thinkingLoopEscalationRatio,
						maxTotalThinkingTokens:
							options.config.thinkingLoopMaxTotalThinkingTokens,
						metaReasoningThreshold:
							options.config.thinkingLoopMetaReasoningThreshold,
					})
				: null;
		this.idleTools = this.createToolRegistry(this.config.tools ?? []);
		this.msgManager = new MessageDeliveryManager({
			steeringMode: (options.config.steeringQueueMode ??
				"one-at-a-time") as DeliveryMode,
			followUpMode: (options.config.followUpQueueMode ??
				"one-at-a-time") as DeliveryMode,
		});
		this.memoryStore = createMemoryStore();
	}

	get phase(): HarnessPhase {
		return this._phase;
	}

	get runtimeState(): AgentRuntimeState {
		return {
			...this.runtime,
			streamingMessage: this.runtime.streamingMessage
				? { ...this.runtime.streamingMessage }
				: undefined,
			pendingToolCalls: [...this.runtime.pendingToolCalls],
			retry: this.runtime.retry ? { ...this.runtime.retry } : undefined,
			outcome: this.runtime.outcome ? { ...this.runtime.outcome } : undefined,
		};
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

	setOnShutdown(cb: () => void): void {
		this.onShutdown = cb;
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
		assertPhaseTransition(this._phase, to, op);
		const prev = this._phase;
		this._phase = to;
		this.runtime = { ...this.runtime, phase: to };
		if (prev !== to) this.onPhaseChange?.(to, prev);
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
		assertIdlePhase(this._phase, op);
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
					maxRetries:
						this.config.autoRetryEnabled === false
							? 0
							: (this.config.streamOptions?.maxRetries ??
								this.config.maxRetries ??
								3),
					retryBaseDelayMs: this.config.retryBaseDelayMs ?? 500,
					maxRetryDelayMs: this.config.streamOptions?.maxRetryDelayMs ?? 15000,
					autoCompactOnContextFull:
						this.config.autoRetryEnabled !== false,
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
			const operationId = randomUUID();
			this._activeOperationId = operationId;
			this._session?.appendJournalEvent({
				type: "operation_start",
				operationId,
				status: "prompt",
			});

			try {
				this.loopConfig = snapshot.config;
				let compactedContext: Message[] | undefined;
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
						refreshNextTurnConfig: () =>
							this.withExtensionRuntime(this.snapshotConfig()),
						onContextCompacted: (messages) => {
							compactedContext = messages;
						},
					} satisfies RunAgentLoopConfig,
					async (event) => {
						await this.handleAgentEvent(event);
					},
				);
				const result = compactedContext ?? [
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
				this._session?.appendJournalEvent({
					type: "operation_finish",
					operationId,
					status: this.runtime.outcome?.status ?? "unknown",
				});
				this._activeOperationId = undefined;
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
		const nonSystem = this.history.filter((m): m is Message => m != null && m.role !== "system");
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
				maxRetries:
					this.config.autoRetryEnabled === false
						? 0
						: (this.config.streamOptions?.maxRetries ??
							this.config.maxRetries ??
							3),
				retryBaseDelayMs: this.config.retryBaseDelayMs ?? 500,
				maxRetryDelayMs: this.config.streamOptions?.maxRetryDelayMs ?? 15000,
				autoCompactOnContextFull:
					this.config.autoRetryEnabled !== false,
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
			const operationId = randomUUID();
			this._activeOperationId = operationId;
			this._session?.appendJournalEvent({
				type: "operation_start",
				operationId,
				status: "continue",
			});

			try {
				this.loopConfig = snapshot.config;
				let compactedContext: Message[] | undefined;
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
						refreshNextTurnConfig: () =>
							this.withExtensionRuntime(this.snapshotConfig()),
						onContextCompacted: (messages) => {
							compactedContext = messages;
						},
					} satisfies RunAgentLoopConfig,
					async (event) => {
						await this.handleAgentEvent(event);
					},
				);
				const result = compactedContext ?? [
					{
						role: "system" as const,
						content:
							snapshot.config.systemPrompt ?? "You are a helpful assistant.",
					},
					...snapshot.initialMessages.filter((m): m is Message => m != null && m.role !== "system"),
					...newMessages,
				];
				this.history = result;
				return result;
			} finally {
				this._session?.appendJournalEvent({
					type: "operation_finish",
					operationId,
					status: this.runtime.outcome?.status ?? "unknown",
				});
				this._activeOperationId = undefined;
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
		const config = this.withDrainHook(
			this.withExtensionRuntime(this.snapshotConfig()),
		);
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
		const pluginHookLayer = this.createClaudeCodeHookLayer();
		const pluginPromptMessages =
			await pluginHookLayer.userPromptMessages(promptText);
		const extensionBeforeStart =
			await this.runExtensionBeforeAgentStart(promptText);
		const beforeStart = await this._beforeAgentStart?.(promptText);

		let initialMessages: Message[] = [...this.history];

		const injectedMessages = [
			...pluginPromptMessages,
			...(extensionBeforeStart?.messages ?? []),
			...(beforeStart?.messages ?? []),
		];
		if (injectedMessages.length) {
			// These messages were produced for this prompt, so keep them at the
			// current turn boundary. Prepending them to the complete history made
			// each new hook message appear before every older conversation turn.
			initialMessages = [...initialMessages, ...injectedMessages];
		}

		// nextTurn guidance belongs to the next user-initiated prompt. Consume it
		// exactly once here, never from an iteration of the currently active run.
		const nextTurnMessages = this._nextTurnQueue.splice(0).map(createUserMessage);
		if (nextTurnMessages.length > 0) {
			initialMessages = [...initialMessages, ...nextTurnMessages];
			this.emitQueueChange();
		}

		const systemPrompt =
			beforeStart?.systemPrompt ?? extensionBeforeStart?.systemPrompt;
		const baseConfig = {
			...this.snapshotConfig(),
			...(systemPrompt ? { systemPrompt } : {}),
		};
		const config = this.withDrainHook(
			this.withExtensionRuntime(baseConfig, pluginHookLayer),
		);
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

	/** Capture mutable runtime config at a turn boundary. */
	private snapshotConfig(): AgentConfig {
		return {
			...this.config,
			tools: this.config.tools ? [...this.config.tools] : undefined,
			models: this.config.models ? [...this.config.models] : undefined,
			streamOptions: this.config.streamOptions
				? { ...this.config.streamOptions }
				: undefined,
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

	private createClaudeCodeHookLayer(): ClaudeCodeHookLayer {
		return createClaudeCodeHookLayer({
			enabled: this._hooksEnabled,
			sessionId: this._sessionId || "",
			transcriptPath: this._transcriptPath || "",
			cwd: this.cwd || process.cwd(),
				getMatcherValue: (toolName) => {
					const tool = this.config.tools?.find(
						(candidate) => candidate.name === toolName,
					);
					return tool?.hookAliases?.join("|") ||
						claudeToolMatcherName(toolName);
				},
		});
	}

	private withExtensionRuntime(
		config: AgentConfig,
		pluginHookLayer?: ClaudeCodeHookLayer,
	): AgentConfig {
		const runner = this._extensionRunner;
		const extensionTools = runner
			? runner.getTools().map((tool) => this.wrapExtensionTool(tool))
			: [];
		const tools = [...(config.tools ?? []), ...extensionTools];
		const extensionHooks = runner?.getHooks();

		const builtinHooks = buildBuiltinHooks({
			config,
			contextWindowTokens: () => config.contextWindowTokens,
			toolDefs: () => tools as unknown as Record<string, unknown>[],
			loopDetector: this.loopDetector,
			eventBus: this._extensionBus
				? {
						// Builtin hooks emit ad-hoc diagnostic shapes (e.g. thinking_loop_detected)
						// that aren't part of the typed ExtensionEvent contract, so this bridges
						// through the legacy untyped emit rather than the strict typed bus.
						emit: (event: { type: string; [key: string]: unknown }) => {
							void this._extensionBus?.emitLegacy(event);
						},
					}
				: undefined,
		});

		const layers: HookLayer[] = [
			{ source: "builtin", hooks: builtinHooks },
			{ source: "extensions", hooks: extensionHooks },
			{ source: "claude-code-compat", hooks: (pluginHookLayer ?? this.createClaudeCodeHookLayer()).hooks },
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
			} catch (_e: unknown) {
				// Session persistence must never break a completed turn.
				console.error("[harness] session append failed:", _e);
			}
		}
	}

	private async handleAgentEvent(event: AgentEvent): Promise<void> {
		this.reduceRuntimeEvent(event);
		// The primary application event path is synchronous and latency-sensitive;
		// extension delivery must not hold streaming deltas behind an await.
		this.loopConfig?.onEvent?.(event);
		if (event.type === "message_end" && event.message) {
			this.persistTurnMessages([event.message]);
		}
		this.persistJournalEvent(event);
		await this.emitExtensionAgentEvent(event);
	}

	private persistJournalEvent(event: AgentEvent): void {
		if (!this._session || !this._activeOperationId) return;
		const operationId = this._activeOperationId;
		if (event.type === "turn_start" || event.type === "turn_end") {
			this._session.appendJournalEvent({
				type: event.type,
				operationId,
				turnId: event.turnId,
				status: event.type === "turn_end" ? event.stopReason : undefined,
			});
		} else if (event.type === "tool_execution_start") {
			this._session.appendJournalEvent({
				type: "tool_start",
				operationId,
				toolCallId: event.toolCallId,
				toolName: event.toolName,
			});
		} else if (event.type === "tool_execution_end") {
			this._session.appendJournalEvent({
				type: "tool_end",
				operationId,
				toolCallId: event.toolCallId,
				toolName: event.toolName,
				status: event.isError ? "error" : "completed",
			});
		}
	}

	private reduceRuntimeEvent(event: AgentEvent): void {
		this.runtime = reduceRuntimeState(this.runtime, event, this._phase);
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
			loopDetector: this.loopDetector,
			onCompact: async () => {
				const result = await this.compact();
				return result ?? null;
			},
			onEvent: (event) => {
				if (
					event.type === "auto_retry_start" ||
					event.type === "auto_retry_end" ||
					event.type === "error" ||
					event.type === "context_update"
				) {
					return;
				}
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

	getThinkingLoopDetector(): ThinkingLoopDetector | null {
		return this.thinkingLoopDetector;
	}

	setThinkingLoopDetectorEnabled(enabled: boolean): void {
		if (enabled && !this.thinkingLoopDetector) {
			this.thinkingLoopDetector = new ThinkingLoopDetector({
				minThinkingLength: this.config.thinkingLoopMinThinkingLength,
				thinkingOnlyThreshold: this.config.thinkingLoopThinkingOnlyThreshold,
				escalationRatio: this.config.thinkingLoopEscalationRatio,
				maxTotalThinkingTokens: this.config.thinkingLoopMaxTotalThinkingTokens,
				metaReasoningThreshold: this.config.thinkingLoopMetaReasoningThreshold,
			});
		} else if (!enabled) {
			this.thinkingLoopDetector = null;
		}
	}

	// ── Runtime config setters ─────────────────────────────────────────────

	setSystemPrompt(systemPrompt: string): void {
		this.config.systemPrompt = systemPrompt;
	}

	setTemperature(temperature: number): void {
		this.config.temperature = temperature;
	}

	setInferenceMode(mode: string): void {
		// Validate mode name before accepting it.
		const valid = ["thinking-general", "thinking-coding", "instruct-general", "instruct-reasoning"];
		if (!valid.includes(mode)) {
			// Silently ignore invalid mode — the caller (TUI) should handle this.
			return;
		}
		this.config.inferenceMode = mode as typeof this.config.inferenceMode;
	}

	setMaxTokens(maxTokens: number): void {
		this.config.maxTokens = maxTokens;
	}

	setMaxIterations(maxIterations: number): void {
		this.maxIterations = maxIterations;
	}

	setExecutionProfile(
		profile: NonNullable<AgentConfig["executionProfile"]>,
	): void {
		this.config.executionProfile = profile;
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
			allowedPaths: this.config.allowedPaths,
			allowAllPaths: this.config.allowAllPaths,
			cacheSize: this.config.cacheSize,
			cacheTtlMs: this.config.cacheTtlMs,
			onQuestionRequest: this.config.onQuestionRequest,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
		registry.registerMany(tools);
		return registry;
	}

	// ── Queue operations (see harness/queue-ops.ts) ─────────────────────────

	private get queueOpsDeps(): QueueOpsDeps {
		return {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			getNextTurnQueue: () => this._nextTurnQueue,
			setNextTurnQueue: (queue) => {
				this._nextTurnQueue = queue;
			},
			emitQueueChange: () => this.emitQueueChange(),
		};
	}

	steer(text: string): void {
		queueOps.steer(
			this.queueOpsDeps,
			text,
			this.config.steeringInterrupt,
			this.abortController,
		);
	}

	/** Promote queued steering into the immediate next turn and interrupt the current step. */
	flushSteeringNow(): number {
		return queueOps.flushSteeringNow(this.queueOpsDeps, this.abortController);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return queueOps.dropQueuedMessage(this.queueOpsDeps, displayIndex);
	}

	followUp(text: string): void {
		queueOps.followUp(this.queueOpsDeps, text);
	}

	nextTurn(text: string): void {
		queueOps.nextTurn(this.queueOpsDeps, text);
	}

	async abort(): Promise<AbortResult> {
		return queueOps.abort({
			...this.queueOpsDeps,
			abortController: this.abortController,
			activeOperationId: this._activeOperationId,
			session: this._session,
			setAbortRequested: () => {
				this.runtime = { ...this.runtime, abortRequested: true };
			},
			waitForIdle: () => this.waitForIdle(),
			emitAbortEvent: (result) =>
				this.emitToSubscribers({ type: "abort", ...result }),
			emitSessionEnd: (reason) => this.emitSessionEnd(reason),
		});
	}

	// ── Queue state ────────────────────────────────────────────────────────

	getQueues(): HarnessQueues {
		return queueOps.getQueues(this.queueOpsDeps);
	}

	setOnQueueChange(cb: (queues: HarnessQueues) => void): void {
		this.onQueueChange = cb;
	}

	clearQueues(): HarnessQueues {
		return queueOps.clearQueues(this.queueOpsDeps);
	}

	private emitQueueChange(): void {
		const queues = queueOps.getQueues(this.queueOpsDeps);
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
		queueOps.setSteeringMode(this.queueOpsDeps, mode);
	}

	getSteeringMode(): QueueMode {
		return queueOps.getSteeringMode(this.queueOpsDeps);
	}

	setFollowUpMode(mode: QueueMode): void {
		queueOps.setFollowUpMode(this.queueOpsDeps, mode);
	}

	getFollowUpMode(): QueueMode {
		return queueOps.getFollowUpMode(this.queueOpsDeps);
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

	private hookContext(): { sessionId: string; transcriptPath: string; cwd: string } {
		return {
			sessionId: this._sessionId || "",
			transcriptPath: this._transcriptPath || "",
			cwd: this.cwd || process.cwd(),
		};
	}

	private async emitSessionStart(source: string = "startup"): Promise<void> {
		const started = await emitSessionStartHelper(this._hooksEnabled, this.hookContext(), source);
		if (started) this._hasStartedSession = true;
	}

	private async emitSessionEnd(reason: string = "other"): Promise<void> {
		await emitSessionEndHelper(this._hooksEnabled, this.hookContext(), reason);
	}

	private async emitPreCompact(
		ctx?: BeforeCompactContext,
	): Promise<BeforeCompactResult | undefined> {
		return emitPreCompactHelper(
			this._hooksEnabled,
			this.hookContext(),
			this.config.internalHooks?.beforeCompact,
			this.config.hooks?.beforeCompact,
			ctx,
		);
	}

	private async emitPostCompact(): Promise<void> {
		await emitPostCompactHelper(this._hooksEnabled, this.hookContext());
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

	// ── Memory management ────────────────────────────────────────────────

	setMemoryEnabled(enabled: boolean): void {
		this.memoryEnabled = enabled;
	}

	getMemoryStore(): MemoryStore {
		return this.memoryStore;
	}

	getMemoryPrompt(maxEntries?: number): string {
		return formatMemoryPrompt(this.memoryStore, maxEntries ?? 10);
	}

	/** Persist memory to a JSON file for cross-session recall. */
	saveMemory(path: string): void {
		try {
			const fs = require("node:fs");
			fs.writeFileSync(path, this.memoryStore.serialize());
		} catch (_e: unknown) {
			// Ignore persistence errors
			console.error("[harness] saveMemory failed:", _e);
		}
	}

	/** Load memory from a JSON file. */
	loadMemory(path: string): void {
		try {
			const fs = require("node:fs");
			const data = fs.readFileSync(path, "utf-8");
			this.memoryStore.deserialize(data);
		} catch (_e: unknown) {
			// Ignore load errors
			console.error("[harness] loadMemory failed:", _e);
		}
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
		const sessionBaseDir = baseDir ?? this._sessionBaseDir;
		const resumed = loadSessionMessages(sessionId, sessionBaseDir);
		if (!resumed) return false;

		if (resumed.messages.length > 0) {
			this.setActiveHistory(resumed.messages.filter((m): m is Message => m != null && m.role !== "system"));
		}
		this._sessionBaseDir = sessionBaseDir;
		this._session = resumed.session;
		this._sessionId = sessionId;
		await this.emitSessionStart("resume");
		return true;
	}

	listSessions(): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}> {
		const baseDir =
			this._sessionBaseDir ??
			(this._session ? `${this._session.dirPath}/../..` : undefined);
		return listSessionsHelper(baseDir);
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
		this.setActiveHistory(messages.filter((m): m is Message => m != null && m.role !== "system"));
		this.emitSessionStart("resume").catch(() => {});
		this._hasStartedSession = false;
	}

	/**
	 * Append messages to the live conversation without resetting session
	 * lifecycle. Used by direct-mode /spawn to record the spawn request and
	 * subagent result so later turns can see them.
	 */
	appendMessages(messages: Message[]): void {
		this.assertIdle("appendMessages");
		const toAdd = messages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
		if (!toAdd.length) return;
		this.history = [...this.history, ...toAdd];
		this.persistTurnMessages(toAdd);
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
		const { branch, nextBranchSeq } = forkBranch(this.branches, this.branchSeq, current, customSummary);
		this.branchSeq = nextBranchSeq;
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
			const outcome = await summarizeAndMergeBranch(this.backend, branch, current, {
				customInstructions: options?.customInstructions,
				contextWindowTokens: this.config.contextWindowTokens,
				maxTokens: this.config.maxTokens,
				thinkingLevel: this.config.thinkingLevel,
			});
			this.branches.pop();
			this.setActiveHistory(outcome.history);
			return outcome.summaryText;
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
		const outcome = await navigateToCheckpointHelper(
			this.backend,
			this.checkpoints,
			checkpointIndex,
			this.history,
			{
				summarize: options?.summarize,
				customInstructions: options?.customInstructions,
				maxTokens: this.config.maxTokens,
				thinkingLevel: this.config.thinkingLevel,
			},
		);
		if (!outcome) return null;

		this.history = outcome.history;
		this.checkpoints = this.checkpoints.slice(0, checkpointIndex);
		this.branches = [];
		return outcome.summary;
	}

	/**
	 * Visualize the branch tree as an ASCII art string.
	 * Shows parent/child relationships with depth indicators.
	 */
	branchTree(): string {
		return renderBranchTree(this.branches);
	}

	listBranches(): BranchInfo[] {
		return listBranchesHelper(this.branches);
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

			const result = await runCompaction(this.backend, messages, before, {
				reason: "manual",
				presetSummary: preResult?.summary,
				temperature: this.config.temperature,
				maxTokens: this.config.maxTokens,
				thinkingLevel: this.config.thinkingLevel,
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
			const after = result.tokensAfter;
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

			const result = await runCompaction(this.backend, messages, before, {
				reason,
				presetSummary: preResult?.summary,
				temperature: this.config.temperature,
				maxTokens: this.config.maxTokens,
				thinkingLevel: this.config.thinkingLevel,
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

	// ── Tool registry ──────────────────────────────────────────────────────

	get tools(): ToolRegistry {
		return this.idleTools;
	}

	// ── Config getters ─────────────────────────────────────────────────────

	/** Immutable snapshot of current config. Prefer to individual getters. */
	getCurrentConfig(): Readonly<AgentConfig> {
		return Object.freeze({ ...this.config });
	}

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
		const configured = this.config.models ?? [];
		return [
			...(configured.some((option) => option.model === this.config.model)
				? []
				: [this.config.model]),
			...configured.map((option) => option.model),
		];
	}

	setModelEndpoint(model: string, baseUrl: string): void {
		this.config.model = model;
		this.config.baseUrl = baseUrl;
		this.backend = this.backend.withEndpoint?.(model, baseUrl) ?? this.backend.withModel(model);
	}

	/** Resolve the baseUrl for a given model identifier. */
	private getModelUrl(modelName: string): string {
		return resolveModelUrl(this.config.models, modelName, this.config.baseUrl);
	}

	/** Set the models array for cycling. */
	setModels(models: AgentModelConfig[]): void {
		this.config.models = models;
	}

	cycleModel(direction: "forward" | "backward" = "forward"): string {
		const currentLevel = this.config.thinkingLevel ?? "off";
		const result = cycleModelHelper(
			this.config.model,
			this.config.baseUrl,
			this.config.thinkingLevel,
			this.config.models ?? [],
			direction,
		);
		if (!result.didCycle) {
			return result.model;
		}

		if (result.baseUrl !== this.config.baseUrl) {
			this.config.baseUrl = result.baseUrl;
		}

		if (result.thinkingLevelClamped) {
			this.config.thinkingLevel = result.thinkingLevel;
			this.emitToSubscribers({
				type: "thinking_level_clamped",
				level: result.thinkingLevel,
				reason: `Model ${result.model} does not support ${currentLevel} thinking level`,
			});
		}

		this.config.model = result.model;
		this._session?.appendModelChange(result.model);
		this.emitToSubscribers({
			type: "model_cycle",
			model: result.model,
			fromModel: result.fromModel,
			thinkingLevel: result.thinkingLevel,
		});
		return result.model;
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
		return shouldAutoCompact(this.autoCompactionSettings, messages);
	}

	private withDrainHook(config: AgentConfig): AgentConfig {
		return withHarnessQueueHooks(config, {
			messageDelivery: this.msgManager,
			onQueueChange: () => this.emitQueueChange(),
			onSavePoint: this.onSavePoint,
			subscribers: this._subscribers,
		});
	}
}
