// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//

import type { CompactionSettings } from "../../infrastructure/compaction/compaction.ts";
import type { ExtensionRunner } from "./index";
import type { ClaudeCodeHookLayer } from "../../extension/adapters/claude-code/hook-layer.ts";
import {
	type DeliveryMode,
	MessageDeliveryManager,
} from "../../runtime/queue/manager.ts";
import { withQueueEventForwarding } from "../../core/harness-queue-hooks.ts";
import { ToolRegistry } from "../../infrastructure/tools/registry.ts";
import {
	throwOnValidationErrors,
	validateConfig,
} from "../../infrastructure/config/config-validator.ts";
import {
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "../../core/agent-loop-runner.ts";
import type { LLMBackend } from "../../core/backend.ts";
import {
	type ContinuationDecision,
	ContinuationTracker,
	type RunBudgetStatus,
} from "../../core/continuation-tracker.ts";
import { HarnessInterventionController } from "../../core/intervention-controller.ts";
import {
	createUserMessage,
	estimateChatPayloadTokens,
} from "./messages";
import {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
	reduceRuntimeState,
} from "../../core/runtime-state.ts";
import { Session } from "../../core/session.ts";
import { LoopDetector } from "../../infrastructure/guards/loop-detector.ts";
import { OutputGuard } from "../../infrastructure/guards/output-guard.ts";
import type { BranchInfo, BranchSummaryData } from "../../runtime/summaries/types.ts";
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
	Tool,
} from "../../types/index.ts";
import { summarizeAndMergeBranch } from "./branching.ts";
import { runCompaction, shouldAutoCompact } from "./compaction.ts";
import type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
	HarnessTurnSnapshot,
} from "./contracts.ts";
import { ConversationState } from "./conversation-state.ts";
import type { ExtensionRuntimeDeps } from "./extension-runtime.ts";
import {
	createClaudeCodeHookLayerFor,
	createExtensionRuntimeState,
	runExtensionBeforeAgentStart as runExtensionBeforeAgentStartHelper,
	withExtensionRuntime as withExtensionRuntimeHelper,
} from "./extension-runtime.ts";
import type { ModelOpsDeps } from "./model-ops.ts";
import * as modelOps from "./model-ops.ts";
import { assertIdlePhase, assertPhaseTransition } from "./phase.ts";
import type { QueueOpsDeps } from "./queue-ops.ts";
import * as queueOps from "./queue-ops.ts";
import {
	emitPostCompact as emitPostCompactHelper,
	emitPreCompact as emitPreCompactHelper,
	emitSessionEnd as emitSessionEndHelper,
	emitSessionStart as emitSessionStartHelper,
	listSessions as listSessionsHelper,
	loadSessionMessages,
} from "./session-lifecycle.ts";

export type {
	AgentRuntimeState,
	HarnessPhase,
} from "../../core/runtime-state.ts";
export type { BranchInfo, BranchSummaryData } from "../../runtime/summaries/types.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./contracts.ts";
export { HarnessBusyError } from "./phase.ts";

type TurnRequest = { kind: "prompt"; text: string } | { kind: "continue" };

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
	private conversation = new ConversationState();
	private msgManager: MessageDeliveryManager;
	private loopDetector: LoopDetector;
	// Escalation/cooldown state for the built-in safeguard hooks. Owned here
	// (not inside buildBuiltinHooks) because withExtensionRuntime() rebuilds
	// the hooks object on every loop iteration (via refreshNextTurnConfig) —
	// state that must persist across iterations (intervention escalation,
	// budget-stop's consecutive-turn comparison, the compaction cooldown)
	// would otherwise silently reset every time and never do its job.
	private interventions: HarnessInterventionController =
		new HarnessInterventionController();
	private extensionRuntimeState = createExtensionRuntimeState();
	private onQueueChange?: (queues: HarnessQueues) => void;
	private onPhaseChange?: (phase: HarnessPhase, prev: HarnessPhase) => void;
	private onSettled?: (
		nextTurnCount: number,
		reason: "steering_interrupt" | "normal",
	) => void;
	private autoCompactionSettings: CompactionSettings = {
		enabled: false,
		reserveTokens: 16_384,
		keepRecentTokens: 20_000,
		contextWindow: 128_000,
	};

	// ── Output Guard ─────────────────────────────────────────────────────
	private outputGuard: OutputGuard;
	private _hooksEnabled: boolean;
	private _session?: Session;
	private _sessionBaseDir?: string;
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
	private continuationTracker = new ContinuationTracker();
	private durableBudgetState = {
		providerCalls: 0,
		toolCalls: 0,
		tokens: 0,
		startedAt: undefined as number | undefined,
	};
	private _runPromise?: Promise<void>;
	private _runResolve?: () => void;
	private _subscribers: Set<EventHandler> = new Set();
	private _extensionRunner?: ExtensionRunner;
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
		this.config.streamOptions = {
			...options.config.streamOptions,
			...(options.config.streamOptions?.timeoutMs === undefined &&
			options.config.turnTimeoutMs !== undefined
				? { timeoutMs: options.config.turnTimeoutMs }
				: {}),
		};
		this._hooksEnabled = options.config.runtimeHooksEnabled ?? true;
		this.backend = options.backend;
		this.cwd = options.cwd;
		this.maxIterations = options.maxIterations;
		this._extensionRunner = options.extensionRunner;
		this.loopDetector = new LoopDetector({
			duplicateThreshold: options.config.duplicateToolThreshold,
			failureThreshold: options.config.toolFailureLoopThreshold,
		});
		this.outputGuard = new OutputGuard({
			maxRetries:
				options.config.streamOptions?.maxRetries ??
				options.config.maxRetries ??
				3,
			retryBaseDelayMs: options.config.retryBaseDelayMs ?? 500,
			maxRetryDelayMs: options.config.streamOptions?.maxRetryDelayMs ?? 15_000,
			autoCompactOnContextFull: options.config.autoRetryEnabled !== false,
			maxEmptyResponses: 3,
			maxNonCommittalResponses: 3,
			budgetThreshold: 0.95,
			maxConsecutiveCompactions: 3,
			onEvent: event => {
				this.emitToSubscribers(event);
			},
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

	get continuationStatus() {
		const state = this.continuationTracker.snapshot();
		return state.taskId
			? {
					taskId: state.taskId,
					status: state.status,
					compactionGeneration: state.compactionGeneration,
				}
			: undefined;
	}

	get continuationBudget(): RunBudgetStatus | undefined {
		return this.continuationTracker.budgetStatus();
	}

	/** Carries cumulative budget counters across turns in this session. */
	private durableExecutionConfig(): Pick<
		RunAgentLoopConfig,
		"durableBudgetState" | "onBudgetConsumed"
	> {
		return {
			durableBudgetState: { ...this.durableBudgetState },
			onBudgetConsumed: (resource, amount) => {
				if (resource === "provider_call")
					this.durableBudgetState.providerCalls += amount;
				else if (resource === "tool_call")
					this.durableBudgetState.toolCalls += amount;
				else if (resource === "token") this.durableBudgetState.tokens += amount;
			},
		};
	}

	requestContinuation(
		_cause: string,
		progressFingerprint: string,
	): ContinuationDecision {
		return this.continuationTracker.requestContinuation(progressFingerprint);
	}

	failRun(reason?: string): void {
		this.continuationTracker.finish("failed", reason);
	}

	setOnPhaseChange(
		cb: (phase: HarnessPhase, prev: HarnessPhase) => void,
	): void {
		this.onPhaseChange = cb;
	}

	setOnSettled(
		cb: (
			nextTurnCount: number,
			reason: "steering_interrupt" | "normal",
		) => void,
	): void {
		this.onSettled = cb;
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

	setExtensionRunner(runner: ExtensionRunner | undefined): void {
		this._extensionRunner = runner;
	}

	// ── Stream options management ──────────────────────────────────────────

	getStreamOptions(): AgentHarnessStreamOptions {
		return { ...this.config.streamOptions };
	}

	setStreamOptions(options: Partial<AgentHarnessStreamOptions>): void {
		this.config.streamOptions = { ...this.config.streamOptions, ...options };
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

	private endTurn(): void {
		this.abortController = null;
		this.msgManager.queue.clearCurrentTurn();
		this.emitQueueChange();
		this._runResolve?.();
		this._runPromise = undefined;
		this._runResolve = undefined;
		const nextTurnCount = this.msgManager.queue.getNextTurn().length;
		// A steering interrupt should resume as a plain next turn, not go through
		// the autonomous continuation budget/task-restart policy meant for the
		// model queueing its own follow-up work.
		const reason: "steering_interrupt" | "normal" =
			this.runtime.outcome?.status === "cancelled" &&
			this.runtime.outcome.summary === STEERING_INTERRUPT_SUMMARY
				? "steering_interrupt"
				: "normal";
		this.onSettled?.(nextTurnCount, reason);
		this.emitToSubscribers({ type: "agent_settled", nextTurnCount });
	}

	// ── Structural operation: turns (prompt / continue) ─────────────────────

	async prompt(userMessage: string): Promise<Message[]> {
		this.assertIdle("prompt");
		if (this.autoCompactionSettings.enabled) {
			const compacted = await this.runAutoCompaction("auto");
			if (compacted) {
				return this.prompt(userMessage);
			}
		}
		return this.runTurn({ kind: "prompt", text: userMessage });
	}

	/**
	 * Resume the agent loop from existing history without injecting a new user message.
	 * The last message in history must be a user or tool-result message (not assistant).
	 * Mirrors pi's agent.continue() — used when the agent stopped prematurely and
	 * the caller wants to re-enter the loop without fabricating a follow-up prompt.
	 */
	async continue(): Promise<Message[]> {
		this.assertIdle("continue");
		const nonSystem = this.conversation.history.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
		if (nonSystem.length === 0) {
			throw new Error("Cannot continue: no messages in history");
		}
		const last = nonSystem[nonSystem.length - 1];
		if (last?.role === "assistant") {
			throw new Error("Cannot continue from message role: assistant");
		}
		return this.runTurn({ kind: "continue" });
	}

	/**
	 * Consume queued next-turn guidance and resume without fabricating a user
	 * prompt whose content is merely "continue".
	 */
	async continueWithNextTurn(): Promise<Message[]> {
		this.assertIdle("continue with next-turn guidance");
		const guidance = this.msgManager.queue
			.dequeueNextTurn()
			.map(message => createUserMessage(message.content));
		if (guidance.length === 0) {
			throw new Error("Cannot continue: no next-turn guidance queued");
		}
		this.conversation.append(guidance);
		this.emitQueueChange();
		return this.continue();
	}

	/**
	 * Shared turn transaction for both entry points. Callers run kind-specific
	 * validation before invoking this — `prompt` and `continue` differ only in
	 * snapshot creation (prompt injects hook/extension messages and drains
	 * next-turn guidance; continue just wraps current history), in whether a
	 * new task starts, and in whether a user message is threaded into the
	 * loop. Everything else — idle bookkeeping, session start, checkpointing,
	 * and history reconstruction — is identical, so it lives here once.
	 */
	private async runTurn(request: TurnRequest): Promise<Message[]> {
		this._runPromise = new Promise<void>(resolve => {
			this._runResolve = resolve;
		});

		return this.runInPhase("turn", request.kind, async () => {
			try {
				const signal = await this.beginTurn(request);
				const snapshot = await this.createSnapshot(request, signal);
				let compactedContext: Message[] | undefined;
				const newMessages = await this.runLoop(request, snapshot, messages => {
					compactedContext = messages;
				});
				return this.commitResult(snapshot, newMessages, compactedContext);
			} finally {
				this.endTurn();
			}
		});
	}

	/** Task/checkpoint setup that must happen before a snapshot is created. */
	private async beginTurn(request: TurnRequest): Promise<AbortSignal> {
		if (request.kind === "prompt") {
			this.continuationTracker.startTask();
		}
		this.abortController = new AbortController();
		this.conversation.checkpoint();

		if (!this._hasStartedSession) {
			await this.emitSessionStart("startup");
		}
		return this.abortController.signal;
	}

	/** Run the loop against the prepared snapshot. */
	private async runLoop(
		request: TurnRequest,
		snapshot: HarnessTurnSnapshot,
		onContextCompacted: (messages: Message[]) => void,
	): Promise<Message[]> {
		this.loopConfig = snapshot.config;
		const prompts =
			request.kind === "prompt" ? [createUserMessage(snapshot.promptText)] : [];
		return runAgentLoop(
			{
				systemPrompt: snapshot.config.systemPrompt,
				messages: snapshot.initialMessages,
				tools: snapshot.config.tools,
				cwd: this.cwd,
			},
			prompts,
			{
				...snapshot.config,
				...this.durableExecutionConfig(),
				backend: this.backend,
				signal: snapshot.signal,
				maxIterations: this.maxIterations,
				outputGuard: this.outputGuard,
				interventionController: this.interventions,
				refreshNextTurnConfig: () =>
					this.withExtensionRuntime(this.snapshotConfig()),
				onContextCompacted: messages => {
					onContextCompacted(messages);
					this.persistCompactedContext(messages, this.estimatePayloadTokens());
				},
			} satisfies RunAgentLoopConfig,
			async event => {
				await this.handleAgentEvent(event);
			},
		);
	}

	/** Fold the loop's result into history, preferring in-loop compacted context. */
	private commitResult(
		snapshot: HarnessTurnSnapshot,
		newMessages: Message[],
		compactedContext?: Message[],
	): Message[] {
		const result = compactedContext ?? [
			{
				role: "system" as const,
				content: snapshot.config.systemPrompt ?? "You are a helpful assistant.",
			},
			...snapshot.initialMessages.filter(
				(message): message is Message =>
					message != null && message.role !== "system",
			),
			...newMessages,
		];
		this.conversation.history = result;
		return result;
	}

	/**
	 * Snapshot the turn's context. `prompt` and `continue` differ only in
	 * whether a new user message is threaded in: a prompt injects plugin/
	 * extension before-agent-start messages and drains queued next-turn
	 * guidance; continue just wraps current history as-is.
	 */
	private async createSnapshot(
		request: TurnRequest,
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		let initialMessages: Message[] = [...this.conversation.history];
		let systemPrompt: string | undefined;
		let pluginHookLayer: ClaudeCodeHookLayer | undefined;

		if (request.kind === "prompt") {
			pluginHookLayer = this.createClaudeCodeHookLayer();
			const pluginPromptMessages = await pluginHookLayer.userPromptMessages(
				request.text,
			);
			const extensionBeforeStart = await this.runExtensionBeforeAgentStart(
				request.text,
			);
			const beforeStart = await this._beforeAgentStart?.(request.text);

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
			const nextTurnMessages = this.msgManager.queue
				.dequeueNextTurn()
				.map(message => createUserMessage(message.content));
			if (nextTurnMessages.length > 0) {
				initialMessages = [...initialMessages, ...nextTurnMessages];
				this.emitQueueChange();
			}

			systemPrompt =
				beforeStart?.systemPrompt ?? extensionBeforeStart?.systemPrompt;
		}

		const baseConfig = {
			...this.snapshotConfig(),
			...(systemPrompt ? { systemPrompt } : {}),
		};
		const config = this.forwardQueueEvents(
			this.withExtensionRuntime(baseConfig, pluginHookLayer),
		);

		return {
			promptText: request.kind === "prompt" ? request.text : "",
			initialMessages,
			config,
			streamOptions: config.streamOptions ?? {},
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

	private get extensionRuntimeDeps(): ExtensionRuntimeDeps {
		return {
			getExtensionRunner: () => this._extensionRunner,
			getHooksEnabled: () => this._hooksEnabled,
			getSessionId: () => this._sessionId || "",
			getTranscriptPath: () => this._transcriptPath || "",
			getCwd: () => this.cwd || process.cwd(),
			getConfigTools: () => this.config.tools,
			loopDetector: this.loopDetector,
			interventions: this.interventions,
			emit: event => this.emitToSubscribers(event),
			drainHooks: () => this.drainHooks(),
		};
	}

	private async runExtensionBeforeAgentStart(
		promptText: string,
	): Promise<{ messages?: Message[]; systemPrompt?: string } | undefined> {
		return runExtensionBeforeAgentStartHelper(
			this.extensionRuntimeDeps,
			promptText,
			this.config.systemPrompt ?? "",
			this.conversation.history,
		);
	}

	private createClaudeCodeHookLayer(): ClaudeCodeHookLayer {
		return createClaudeCodeHookLayerFor(this.extensionRuntimeDeps);
	}

	private withExtensionRuntime(
		config: AgentConfig,
		pluginHookLayer?: ClaudeCodeHookLayer,
	): AgentConfig {
		return withExtensionRuntimeHelper(
			this.extensionRuntimeDeps,
			this.extensionRuntimeState,
			config,
			pluginHookLayer,
		);
	}

	/** Steering/follow-up messages drained from the harness's own queues. */
	private drainHooks(): AgentHooks {
		const inject = (texts: string[]): Message[] | undefined => {
			if (!texts.length) return undefined;
			this.emitQueueChange();
			return texts.map(createUserMessage);
		};
		return {
			getSteeringMessages: async () =>
				inject(this.msgManager.afterTurn().map(message => message.content)),
			getFollowUpMessages: async () =>
				inject(this.msgManager.onIdle().map(message => message.content)),
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
		if (event.type === "compaction")
			this.continuationTracker.recordCompaction();
		if (event.type === "run_outcome") {
			this.continuationTracker.finish(event.status, event.summary);
		}
		// The primary application event path is synchronous and latency-sensitive;
		// extension delivery must not hold streaming deltas behind an await.
		this.loopConfig?.onEvent?.(event);
		if (event.type === "message_end" && event.message) {
			this.persistTurnMessages([event.message]);
		}
		await this.emitExtensionAgentEvent(event);
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
			case "tool_execution_start":
			case "tool_execution_update":
			case "tool_execution_end":
			case "agent_retry_start":
			case "agent_retry_end":
			case "agent_error":
			case "agent_settled":
			case "session_delete":
			case "model_select":
				await runner.emitToAll({ type: event.type, context });
				break;
		}
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

	setSteeringInterrupt(enabled: boolean): void {
		this.config.steeringInterrupt = enabled;
	}

	setInferenceMode(mode: string): void {
		// Validate mode name before accepting it.
		const valid = [
			"auto",
			"none",
			"thinking-general",
			"thinking-coding",
			"instruct-general",
			"instruct-reasoning",
			"instruct-coding",
			"deterministic",
			"creative",
			"analytical",
		];
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

	setRuntimeOptions(
		options: Partial<
			Pick<
				AgentConfig,
				| "guardsEnabled"
				| "duplicateGuardEnabled"
				| "failureGuardEnabled"
				| "budgetStopEnabled"
				| "continuationEnabled"
				| "autoRetryEnabled"
				| "reflectionConfig"
			>
		>,
	): void {
		Object.assign(this.config, options);
	}

	setTools(tools: Tool[]): void {
		this.config.tools = tools;
		this.idleTools = this.createToolRegistry(tools);
		this._session?.appendActiveToolsChange(tools.map(t => t.name));
		this.emitToSubscribers({
			type: "tools_update",
			toolNames: tools.map(t => t.name),
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
			setAbortRequested: () => {
				this.runtime = { ...this.runtime, abortRequested: true };
			},
			waitForIdle: () => this.waitForIdle(),
			emitAbortEvent: result =>
				this.emitToSubscribers({ type: "abort", ...result }),
			emitSessionEnd: reason => this.emitSessionEnd(reason),
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

	private hookContext(): {
		sessionId: string;
		transcriptPath: string;
		cwd: string;
	} {
		return {
			sessionId: this._sessionId || "",
			transcriptPath: this._transcriptPath || "",
			cwd: this.cwd || process.cwd(),
		};
	}

	private async emitSessionStart(source: string = "startup"): Promise<void> {
		const started = await emitSessionStartHelper(
			this._hooksEnabled,
			this.hookContext(),
			source,
		);
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

	// ── Session & history ──────────────────────────────────────────────────

	async enableSession(baseDir?: string): Promise<void> {
		const sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		this._sessionBaseDir = baseDir;
		this._session = new Session(sessionId, { baseDir, enabled: true });
		this._sessionId = sessionId;
		await this.emitSessionStart("startup");
	}

	/**
	 * Attach an already-constructed Session (e.g. one a caller's own session
	 * manager already owns) as this harness's durable branch/compaction/model
	 * journal, instead of minting a new one via enableSession/resumeSession.
	 * Conversation history itself is not touched — the caller is responsible
	 * for loading it (their Session may hold a different persisted shape than
	 * plain Message[], as the TUI's Turn-based sessions do).
	 */
	attachSession(session: Session): void {
		this._session = session;
		this._sessionId = session.getMeta().id;
	}

	async resumeSession(sessionId: string, baseDir?: string): Promise<boolean> {
		const sessionBaseDir = baseDir ?? this._sessionBaseDir;
		const resumed = loadSessionMessages(sessionId, sessionBaseDir);
		if (!resumed) return false;

		if (resumed.messages.length > 0) {
			this.conversation.history = resumed.messages.filter(
				(m): m is Message => m != null && m.role !== "system",
			);
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
		return this.conversation.history;
	}

	clearHistory(): void {
		this.emitSessionEnd("reset").catch(() => {});
		this.conversation.clear();
		this.emitSessionStart("clear").catch(() => {});
		this._hasStartedSession = false;
	}

	setHistory(messages: Message[]): void {
		this.assertIdle("setHistory");
		this.emitSessionEnd("switch").catch(() => {});
		this.conversation.replace(messages);
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
		const toAdd = this.conversation.append(messages);
		if (toAdd.length) this.persistTurnMessages(toAdd);
	}

	rewind(): { messages: number; filesRestored: number } | null {
		this.assertIdle("rewind");
		return this.conversation.rewind();
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
		return this.conversation.fork(
			customSummary,
			this._session?.getLeafEntryId(),
		);
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
		const branch = this.conversation.activeBranch();
		if (!branch) return null;

		const current = this.conversation.history;
		const diverged = current.slice(branch.forkedAt);
		if (!diverged.length) {
			this.conversation.popBranch();
			this._session?.checkout(branch.sessionLeafId);
			this.conversation.history = branch.parent;
			return null;
		}

		return this.runInPhase("branch_summary", "branchSummary", async () => {
			const outcome = await summarizeAndMergeBranch(
				this.backend,
				branch,
				current,
				{
					customInstructions: options?.customInstructions,
					contextWindowTokens: this.config.contextWindowTokens,
					maxTokens: this.config.maxTokens,
					thinkingLevel: this.config.thinkingLevel,
				},
			);
			this.conversation.popBranch();
			this._session?.checkout(branch.sessionLeafId);
			if (outcome.summaryText)
				this._session?.appendBranchSummary(
					outcome.summaryText,
					branch.sessionLeafId,
				);
			this.conversation.history = outcome.history;
			return outcome.summaryText;
		});
	}

	/**
	 * Discard the active branch without merging. Restores parent history.
	 */
	discardBranch(): boolean {
		this.assertIdle("discardBranch");
		const branch = this.conversation.discardBranch();
		if (!branch) return false;
		this._session?.checkout(branch.sessionLeafId);
		return true;
	}

	/**
	 * Visualize the branch tree as an ASCII art string.
	 * Shows parent/child relationships with depth indicators.
	 */
	branchTree(): string {
		return this.conversation.branchTree();
	}

	listBranches(): BranchInfo[] {
		return this.conversation.listBranches();
	}

	// ── Conversation management ────────────────────────────────────────────

	private persistCompactedContext(
		messages: Message[],
		tokensBefore: number,
	): void {
		const summary = messages.find(
			message => String(message.role) === "compactionSummary",
		)?.content;
		if (typeof summary !== "string" || !summary.trim()) return;
		const firstKeptEntryId = messages
			.map(message => message as Message & { entryId?: string })
			.find(
				message =>
					String(message.role) !== "compactionSummary" && message.entryId,
			)?.entryId;
		this._session?.appendCompaction(summary, tokensBefore, firstKeptEntryId);
	}

	// ── Compaction ─────────────────────────────────────────────────────────

	async compact(): Promise<number | null> {
		this.assertIdle("compact");
		if (!this.conversation.history.length) return null;
		return this.runInPhase("compaction", "compact", () =>
			this.performCompaction("manual", /* force */ true),
		);
	}

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.conversation.history;
		if (!messages.length || !this.autoCompactionSettings.enabled) return false;
		return this.runInPhase("compaction", "autoCompact", () =>
			this.performCompaction(reason, /* force */ false).then(
				saved => saved !== 0,
			),
		);
	}

	/**
	 * Shared compaction transaction used by both the manual `compact()` API
	 * and threshold-triggered auto-compaction. `force` skips the
	 * `shouldCompact` threshold check (manual compaction always attempts it);
	 * both paths otherwise run the identical pre-compact/run/commit/post-compact
	 * sequence. Returns tokens saved (0 if nothing changed).
	 */
	private async performCompaction(
		reason: "auto" | "manual",
		force: boolean,
	): Promise<number> {
		const messages = this.conversation.history;
		this.emitToSubscribers({ type: "compaction", reason });

		if (!force && !this.shouldCompact(messages)) {
			await this.emitPostCompact();
			this.emitToSubscribers({
				type: "compaction",
				reason,
				tokensBefore: this.estimatePayloadTokens(),
				tokensAfter: this.estimatePayloadTokens(),
			});
			return 0;
		}

		const before = this.estimatePayloadTokens();
		const preResult = await this.emitPreCompact({
			messages,
			tokensBefore: before,
			reason,
		});
		if (preResult?.cancel) {
			await this.emitPostCompact();
			this.emitToSubscribers({
				type: "compaction",
				reason,
				tokensBefore: before,
				tokensAfter: before,
			});
			return 0;
		}

		await this._extensionRunner?.emit({
			type: "session_before_compact",
			context: {
				sessionId: this._sessionId || "",
				cwd: this.cwd || "",
				reason,
				tokensBefore: before,
				messages: [...messages],
			},
		});

		const result = await runCompaction(this.backend, messages, before, {
			reason,
			presetSummary: preResult?.summary,
			temperature: this.config.temperature,
			maxTokens: this.config.maxTokens,
			thinkingLevel: this.config.thinkingLevel,
		});

		if (
			this.conversation.history !== messages ||
			!result.changed ||
			result.tokensAfter >= before
		) {
			await this.emitPostCompact();
			this.emitToSubscribers({
				type: "compaction",
				reason,
				tokensBefore: before,
				tokensAfter: before,
			});
			return 0;
		}

		this.conversation.history = result.messages;
		const after = result.tokensAfter;
		this.persistCompactedContext(result.messages, before);
		this.continuationTracker.recordCompaction();
		await this.emitPostCompact();
		await this._extensionRunner?.emit({
			type: "session_compact",
			context: {
				sessionId: this._sessionId || "",
				cwd: this.cwd || "",
				reason,
				tokensBefore: before,
				tokensAfter: after,
				changed: true,
				messages: [...result.messages],
			},
		});
		this.emitToSubscribers({
			type: "compaction",
			reason,
			tokensBefore: before,
			tokensAfter: after,
		});
		return before - after;
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

	// ── Model / thinking-level operations (see harness/model-ops.ts) ───────

	private get modelOpsDeps(): ModelOpsDeps {
		return {
			getConfig: () => this.config,
			setModel: model => {
				this.config.model = model;
			},
			setBaseUrl: baseUrl => {
				this.config.baseUrl = baseUrl;
			},
			setModels: models => {
				this.config.models = models;
			},
			setThinkingLevel: level => {
				this.config.thinkingLevel = level;
			},
			getBackend: () => this.backend,
			setBackend: backend => {
				this.backend = backend;
			},
			appendModelChange: model => this._session?.appendModelChange(model),
			appendThinkingLevelChange: level =>
				this._session?.appendThinkingLevelChange(level),
			emit: event => this.emitToSubscribers(event),
		};
	}

	getModel(): string {
		return modelOps.getModel(this.modelOpsDeps);
	}

	getBaseUrl(): string {
		return modelOps.getBaseUrl(this.modelOpsDeps);
	}

	getModels(): string[] {
		return modelOps.getModels(this.modelOpsDeps);
	}

	setModelEndpoint(model: string, baseUrl: string): void {
		modelOps.setModelEndpoint(this.modelOpsDeps, model, baseUrl);
	}

	/** Set the models array for cycling. */
	setModels(models: AgentModelConfig[]): void {
		modelOps.setModels(this.modelOpsDeps, models);
	}

	cycleModel(direction: "forward" | "backward" = "forward"): string {
		return modelOps.cycleModel(this.modelOpsDeps, direction);
	}

	// ── Thinking level ─────────────────────────────────────────────────────

	getThinkingLevel(): string {
		return modelOps.getThinkingLevel(this.modelOpsDeps);
	}

	setThinkingLevel(level: string): void {
		modelOps.setThinkingLevel(this.modelOpsDeps, level);
	}

	// ── Model & provider ──────────────────────────────────────────────────

	setModel(model: string): void {
		modelOps.setModel(this.modelOpsDeps, model);
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

	private shouldCompact(messages: Message[]): boolean {
		return shouldAutoCompact(this.autoCompactionSettings, messages);
	}

	private forwardQueueEvents(config: AgentConfig): AgentConfig {
		return withQueueEventForwarding(config, {
			subscribers: this._subscribers,
		});
	}
}
