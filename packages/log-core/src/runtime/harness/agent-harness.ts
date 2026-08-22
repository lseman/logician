// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//

import type { CompactionSettings } from "../compaction/engine.ts";
import {
	runCompaction,
	shouldAutoCompact,
} from "../compaction/orchestration.ts";
import { validateAgentConfig } from "../../control/configuration/config-validator.ts";
import { ConfigurationStore } from "../../control/configuration/configuration-store.ts";
import { ContextEngine } from "../../system/context/context-engine.ts";
import {
	createSteeringInterruptReason,
	type RunAgentLoopConfig,
	runAgentLoop,
} from "../execution/agent-loop-runner.ts";
import type { ExtensionRunner } from "../../system/extension/runner.ts";
import { LoopDetector } from "../../control/guards/loop-detector.ts";
import { OutputGuard } from "../../control/guards/output-guard.ts";
import { HarnessInterventionController } from "../../control/policy/intervention-controller.ts";
import { AgentRunController } from "../../control/policy/run-controller.ts";
import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import {
	createUserMessage,
	estimateChatPayloadTokens,
} from "../../capabilities/provider/messages.ts";
import type { Session } from "../../capabilities/session/session.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../capabilities/session/summaries/types.ts";
import type { ThreadItem } from "../../capabilities/session/thread-ledger.ts";
import {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
	reduceRuntimeState,
} from "../state/runtime-state.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import type { AgentConfig, QueueMode } from "../../system/types/types-config.ts";
import type {
	AgentEvent,
	AgentHooks,
	BeforeCompactContext,
	BeforeCompactResult,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import {
	composeHarnessConfig,
	HarnessConfigurationError,
} from "./internal/configuration.ts";
import { HarnessEventRouter } from "./internal/event-router.ts";
import { HarnessModelController } from "./internal/model-controller.ts";
import { HarnessObservation } from "./internal/observation.ts";
import { HarnessQueueController } from "./internal/queue-controller.ts";
import { HarnessTurnController } from "./internal/turn-controller.ts";
import type { ExtensionRuntimeDeps } from "./runtime/extension-runtime.ts";
import {
	resolveRuntimeTools,
	runExtensionBeforeAgentStart as runExtensionBeforeAgentStartHelper,
	withExtensionRuntime as withExtensionRuntimeHelper,
} from "./runtime/extension-runtime.ts";
import {
	assertIdlePhase,
	assertPhaseTransition,
	HarnessBusyError,
} from "./runtime/phase.ts";
import { summarizeAndMergeBranch } from "./session/branching.ts";
import { ConversationState } from "./session/conversation-state.ts";
import {
	emitPostCompact as emitPostCompactHelper,
	emitPreCompact as emitPreCompactHelper,
	emitSessionEnd as emitSessionEndHelper,
	emitSessionStart as emitSessionStartHelper,
} from "./session/lifecycle.ts";
import type {
	AbortResult,
	AgentHarnessOptions,
	HarnessObserver,
	HarnessPluginHookFactory,
	HarnessPluginHookLayer,
	HarnessPluginLifecycle,
	HarnessQueues,
	HarnessTurnSnapshot,
} from "./types.ts";

export type {
	BranchInfo,
	BranchSummaryData,
} from "../../capabilities/session/summaries/types.ts";
export type { ThreadItem } from "../../capabilities/session/thread-ledger.ts";
export type {
	AgentRuntimeState,
	HarnessPhase,
} from "../state/runtime-state.ts";
export { HarnessBusyError } from "./runtime/phase.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessModule,
	HarnessObserver,
	HarnessQueues,
} from "./types.ts";
export { defineHarnessModule } from "./types.ts";

export { HarnessConfigurationError };

type TurnRequest = { kind: "prompt"; text: string } | { kind: "continue" };

export class AgentHarness {
	private readonly configuration: ConfigurationStore<AgentConfig>;
	readonly models: HarnessModelController;
	private cwd?: string;
	private maxIterations?: number;

	private _phase: HarnessPhase = "idle";
	private runtime: AgentRuntimeState = createRuntimeState();
	private idleTools: ToolRegistry;
	private readonly turn = new HarnessTurnController();
	private loopConfig: AgentConfig | null = null;
	private conversation = new ConversationState();
	private readonly contextEngine = new ContextEngine(messages =>
		estimateChatPayloadTokens([...messages]),
	);
	private queue: HarnessQueueController;
	private loopDetector: LoopDetector;
	// Durable intervention history spans turns; run policy is reset per prompt.
	private interventions: HarnessInterventionController =
		new HarnessInterventionController();
	private runController = new AgentRunController();
	private observation: HarnessObservation;
	private readonly eventRouter: HarnessEventRouter;
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
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
	private durableBudgetState = {
		providerCalls: 0,
		toolCalls: 0,
		tokens: 0,
		startedAt: undefined as number | undefined,
	};
	private _extensionRunner?: ExtensionRunner;
	private readonly pluginHookFactory?: HarnessPluginHookFactory;
	private readonly pluginLifecycle?: HarnessPluginLifecycle;
	private _beforeAgentStart?: (
		promptText: string,
	) =>
		| Promise<{ messages?: Message[]; systemPrompt?: string } | undefined>
		| { messages?: Message[]; systemPrompt?: string }
		| undefined;

	constructor(options: AgentHarnessOptions) {
		const config = composeHarnessConfig(options.modules ?? [], options.config);
		this.configuration = new ConfigurationStore(config, {
			clone: AgentHarness.cloneConfig,
			validate: candidate =>
				validateAgentConfig(candidate).map(
					error => `Invalid config: ${error.field}: ${error.message}`,
				),
		});
		this.configuration.update({
			streamOptions: {
				...options.config.streamOptions,
				...(options.config.streamOptions?.timeoutMs === undefined &&
				options.config.turnTimeoutMs !== undefined
					? { timeoutMs: options.config.turnTimeoutMs }
					: {}),
			},
		});
		this._hooksEnabled = options.config.runtimeHooksEnabled ?? true;
		this.models = new HarnessModelController({
			backend: options.backend,
			configuration: this.configuration,
			emit: event => this.emitToSubscribers(event),
			persistModel: model => this._session?.appendModelChange(model),
			persistThinking: level => this._session?.appendThinkingLevelChange(level),
		});
		this.cwd = options.cwd;
		this.maxIterations = options.maxIterations;
		this._extensionRunner = options.extensionRunner;
		this.pluginHookFactory = options.pluginHookFactory;
		this.pluginLifecycle = options.pluginLifecycle;
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
		this.queue = new HarnessQueueController(
			{
				steeringMode: options.config.steeringQueueMode ?? "one-at-a-time",
				followUpMode: options.config.followUpQueueMode ?? "one-at-a-time",
			},
			queues => this.emitQueueChange(queues),
		);
		this.observation = new HarnessObservation(
			(options.modules ?? []).flatMap(module => module.observers ?? []),
		);
		this.eventRouter = new HarnessEventRouter({
			reduce: event => {
				this.runtime = reduceRuntimeState(this.runtime, event, this._phase);
			},
			notifyApplication: event => this.loopConfig?.onEvent?.(event),
			persistCompletedMessage: event => {
				if (event.type === "message_end" && event.message) {
					this.persistTurnMessages([event.message]);
				}
			},
			getExtensionRunner: () => this._extensionRunner,
			getExtensionContext: () => ({
				sessionId: this._sessionId ?? "",
				cwd: this.cwd ?? "",
			}),
		});
	}

	private static cloneConfig(config: AgentConfig): AgentConfig {
		return {
			...config,
			tools: config.tools ? [...config.tools] : undefined,
			models: config.models ? [...config.models] : undefined,
			streamOptions: config.streamOptions
				? { ...config.streamOptions }
				: undefined,
			allowedPaths: config.allowedPaths ? [...config.allowedPaths] : undefined,
		};
	}

	private get config(): Readonly<AgentConfig> {
		return this.configuration.current;
	}

	private get backend(): LLMBackend {
		return this.models.backend;
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

	/** Append-only conversation provenance for persistence, replay, and clients. */
	get threadItems(): readonly ThreadItem[] {
		return this.conversation.items;
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

	observe(observer: HarnessObserver): () => void {
		return this.observation.observe(observer);
	}

	async waitForIdle(): Promise<void> {
		if (this._phase === "idle") return;
		await this.turn.wait();
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

	private transition(to: HarnessPhase, op: string): void {
		assertPhaseTransition(this._phase, to, op);
		const prev = this._phase;
		this._phase = to;
		this.runtime = { ...this.runtime, phase: to };
		this.observation.phase(to, prev);
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
		this.queue.clearCurrentTurn();
		const nextTurnCount = this.queue.snapshot().nextTurn.length;
		this.observation.settled(nextTurnCount);
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
		const guidance = this.queue
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
		return this.turn.run(
			signal =>
				this.runInPhase("turn", request.kind, async () => {
					await this.beginTurn(request);
					const snapshot = await this.createSnapshot(request, signal);
					let compactedContext: Message[] | undefined;
					const newMessages = await this.runLoop(
						request,
						snapshot,
						messages => {
							compactedContext = messages;
						},
					);
					return this.commitResult(snapshot, newMessages, compactedContext);
				}),
			() => this.endTurn(),
		);
	}

	/** Task/checkpoint setup that must happen before a snapshot is created. */
	private async beginTurn(_request: TurnRequest): Promise<void> {
		this.runController = new AgentRunController();
		this.conversation.checkpoint();

		if (!this._hasStartedSession) {
			await this.emitSessionStart("startup");
		}
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
				runController: this.runController,
				refreshNextTurnConfig: () => {
					const refreshed = this.snapshotConfig();
					return {
						...refreshed,
						tools: resolveRuntimeTools(this.extensionRuntimeDeps, refreshed),
						// Extension and policy composition has run lifetime. Provider and
						// model settings refresh without rebuilding hook state.
						hooks: snapshot.config.hooks,
					};
				},
				onContextCompacted: messages => {
					onContextCompacted(messages);
					this.persistCompactedContext(messages, this.estimatePayloadTokens());
				},
			} satisfies RunAgentLoopConfig,
			async event => {
				await this.eventRouter.route(event);
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
		let pluginHookLayer: HarnessPluginHookLayer | undefined;

		if (request.kind === "prompt") {
			pluginHookLayer = this.createPluginHookLayer();
			const pluginPromptMessages = await pluginHookLayer.userPromptMessages(
				request.text,
			);
			const extensionBeforeStart = await this.runExtensionBeforeAgentStart(
				request.text,
			);
			const beforeStart = await this._beforeAgentStart?.(request.text);

			// nextTurn guidance belongs to the next user-initiated prompt. Consume it
			// exactly once here, never from an iteration of the currently active run.
			const nextTurnMessages = this.queue
				.dequeueNextTurn()
				.map(message => createUserMessage(message.content));
			if (nextTurnMessages.length > 0) this.emitQueueChange();

			const assembled = this.contextEngine.assemble({
				history: initialMessages,
				baseSystemPrompt: this.config.systemPrompt,
				contributions: [
					{ source: "plugins", messages: pluginPromptMessages },
					{
						source: "extension",
						messages: extensionBeforeStart?.messages,
						systemPrompt: extensionBeforeStart?.systemPrompt,
					},
					{
						source: "application",
						messages: beforeStart?.messages,
						systemPrompt: beforeStart?.systemPrompt,
						priority: 1,
					},
					{ source: "next-turn", messages: nextTurnMessages },
				],
			});
			initialMessages = assembled.messages;
			systemPrompt = assembled.systemPrompt;
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

	/** Capture an immutable runtime config revision at a turn boundary. */
	private snapshotConfig(): AgentConfig {
		return this.configuration.snapshot().value as AgentConfig;
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

	private createPluginHookLayer(): HarnessPluginHookLayer {
		return (
			this.pluginHookFactory?.({
				enabled: this._hooksEnabled,
				sessionId: this._sessionId ?? "",
				transcriptPath: this._transcriptPath ?? "",
				cwd: this.cwd ?? process.cwd(),
				tools: this.config.tools ?? [],
			}) ?? {
				userPromptMessages: async () => [],
			}
		);
	}

	private withExtensionRuntime(
		config: AgentConfig,
		pluginHookLayer?: HarnessPluginHookLayer,
	): AgentConfig {
		return withExtensionRuntimeHelper(
			this.extensionRuntimeDeps,
			this.runController,
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
				inject(this.queue.afterTurn().map(message => message.content)),
			getFollowUpMessages: async () =>
				inject(this.queue.onIdle().map(message => message.content)),
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

	// Configuration changes become visible at the next turn snapshot.
	configure(patch: Partial<AgentConfig>): void {
		this.configuration.update(patch);
		if (patch.maxIterations !== undefined) {
			this.maxIterations = patch.maxIterations;
		}
		if (patch.tools === undefined) return;
		this.idleTools = this.createToolRegistry(patch.tools);
		this._session?.appendActiveToolsChange(patch.tools.map(t => t.name));
		this.emitToSubscribers({
			type: "tools_update",
			toolNames: patch.tools.map(t => t.name),
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

	// ── Queue operations ──────────────────────────────────────────────────

	steer(text: string): void {
		if (this._phase !== "turn") {
			throw new HarnessBusyError("steer", this._phase, "turn");
		}
		this.queue.steer(text, !!this.config.steeringInterrupt, () =>
			this.turn.abort(createSteeringInterruptReason()),
		);
	}

	/** Queue steering for after the current turn (never interrupts).
	 * Adds to the steering queue so the agent loop's drainSteering()
	 * picks it up after tool calls finish but before the next provider
	 * request. The harness publishes the queue update so the TUI shows
	 * the message. */
	steerQueue(text: string): void {
		if (this._phase !== "turn") {
			throw new HarnessBusyError("steer", this._phase, "turn");
		}
		this.queue.steer(text, false, () => {});
	}

	/** Immediately interrupt and apply steering (always forces abort).
	 * Adds to nextTurn so the message survives the abort (clearCurrentTurn()
	 * only wipes steering/followUp), then aborts the provider call. The
	 * auto-continue path in runMessage() detects nextTurn > 0 and starts a
	 * new turn that consumes the message. */
	steerNow(text: string): void {
		if (this._phase !== "turn") {
			throw new HarnessBusyError("steerNow", this._phase, "turn");
		}
		this.queue.nextTurn(text);
		this.turn.abort(createSteeringInterruptReason());
	}

	/** Promote queued steering into the immediate next turn and interrupt the current step. */
	flushSteeringNow(): number {
		if (this._phase !== "turn") {
			throw new HarnessBusyError("flush steering", this._phase, "turn");
		}
		return this.queue.flushSteering(() =>
			this.turn.abort(createSteeringInterruptReason()),
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.queue.drop(displayIndex);
	}

	followUp(text: string): void {
		this.queue.followUp(text);
	}

	nextTurn(text: string): void {
		this.queue.nextTurn(text);
	}

	async abort(): Promise<AbortResult> {
		const result = this.queue.abortSnapshot();
		this.runtime = { ...this.runtime, abortRequested: true };
		this.turn.abort();
		await this.waitForIdle();
		this.emitToSubscribers({ type: "abort", ...result });
		await this.emitSessionEnd("abort");
		return result;
	}

	// ── Queue state ────────────────────────────────────────────────────────

	getQueues(): HarnessQueues {
		return this.queue.snapshot();
	}

	clearQueues(): HarnessQueues {
		return this.queue.clear();
	}

	private emitQueueChange(queues: HarnessQueues = this.getQueues()): void {
		this.observation.queue(queues);
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
		this.queue.setMode("steering", mode);
	}

	getSteeringMode(): QueueMode {
		return this.queue.getMode("steering");
	}

	setFollowUpMode(mode: QueueMode): void {
		this.queue.setMode("followUp", mode);
	}

	getFollowUpMode(): QueueMode {
		return this.queue.getMode("followUp");
	}

	// ── Plugin lifecycle hooks ─────────────────────────────────────────────

	setSessionId(id: string): void {
		this._sessionId = id;
		this.configuration.update({ hookSessionId: id });
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
			this.pluginLifecycle,
		);
		if (started) this._hasStartedSession = true;
	}

	private async emitSessionEnd(reason: string = "other"): Promise<void> {
		await emitSessionEndHelper(
			this._hooksEnabled,
			this.hookContext(),
			reason,
			this.pluginLifecycle,
		);
	}

	private async emitPreCompact(
		ctx?: BeforeCompactContext,
	): Promise<BeforeCompactResult | undefined> {
		return emitPreCompactHelper(
			this._hooksEnabled,
			this.hookContext(),
			this.config.hooks?.beforeCompact,
			ctx,
			this.pluginLifecycle,
		);
	}

	private async emitPostCompact(): Promise<void> {
		await emitPostCompactHelper(
			this._hooksEnabled,
			this.hookContext(),
			this.pluginLifecycle,
		);
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

	/**
	 * Attach a caller-owned durable branch/compaction/model journal.
	 * Conversation history itself is not touched — the caller is responsible
	 * for loading it.
	 */
	attachSession(session: Session): void {
		this._session = session;
		this._sessionId = session.getMeta().id;
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

	/** Immutable snapshot of the configuration used for the next turn. */
	get currentConfig(): Readonly<AgentConfig> {
		return Object.freeze({ ...this.config });
	}

	// ── Internals ──────────────────────────────────────────────────────────

	private emitToSubscribers(event: AgentEvent): void {
		this.observation.event(event);
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
		const originalOnEvent = config.onEvent;
		return {
			...config,
			onEvent: event => {
				originalOnEvent?.(event);
				this.emitToSubscribers(event);
			},
		};
	}
}
