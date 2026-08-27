// ── AgentSession ────────────────────────────────────────────────────────────
// Interactive coordination: conversation, queues, persistence, branching,
// continuation, and lifecycle around a composed execution harness.
//
// This is the single entry point for the agent's interactive session:
//   prompt(text)    — start a turn with a user message
//   continue()      — resume from existing history
//   steer(text)     — inject guidance into the running turn
//   followUp(text)  — queue for after the turn completes
//   nextTurn(text)  — queue for the next user prompt
//   abort()         — stop the current turn
//   compact()       — manually compact conversation
//   fork()          — branch the conversation
//
// Optional `onEvent` callback emits UI-level events (queue updates, phase
// changes) so the host (AgentRuntime) can map them to RuntimeEvent.

import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import {
	createUserMessage,
	estimateChatPayloadTokens,
} from "../../capabilities/provider/messages.ts";
import type { SessionStore } from "../../capabilities/session/session-store.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../capabilities/session/summaries/types.ts";
import type { ThreadItem } from "../../capabilities/session/thread-ledger.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import { validateAgentConfig } from "../../control/configuration/config-validator.ts";
import { ConfigurationStore } from "../../control/configuration/configuration-store.ts";
import { LoopDetector } from "../../control/guards/loop-detector.ts";
import { OutputGuard } from "../../control/guards/output-guard.ts";
import { HarnessInterventionController } from "../../control/policy/intervention-controller.ts";
import { AgentRunController } from "../../control/policy/run-controller.ts";
import { AdaptiveContextController } from "../../system/context/adaptive-context-controller.ts";
import type { ContextContribution } from "../../system/context/context-engine.ts";
import type { ExtensionRunner } from "../../system/extension/runner.ts";
import type {
	AgentConfig,
	QueueMode,
} from "../../system/types/types-config.ts";
import type {
	AgentEvent,
	AgentHooks,
	BeforeCompactContext,
	BeforeCompactResult,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import type { CompactionSettings } from "../compaction/engine.ts";
import {
	runCompaction,
	shouldAutoCompact,
} from "../compaction/orchestration.ts";
import {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
	reduceRuntimeState,
} from "../state/runtime-state.ts";
import {
	createSteeringInterruptReason,
	type RunAgentLoopConfig,
	runAgentLoop,
} from "./agent-harness.ts";
import {
	composeHarnessConfig,
	HarnessConfigurationError,
} from "./internal/configuration.ts";
import { HarnessEventRouter } from "./internal/event-router.ts";
import { HarnessModelController } from "./internal/model-controller.ts";
import { HarnessObservation } from "./internal/observation.ts";
import { SessionState } from "./internal/session-state.ts";
import { HarnessTurnController } from "./internal/turn-controller.ts";
import type { ExtensionRuntimeDeps } from "./live/extension-runtime.ts";
import {
	resolveRuntimeTools,
	runExtensionBeforeAgentStart as runExtensionBeforeAgentStartHelper,
	withExtensionRuntime as withExtensionRuntimeHelper,
} from "./live/extension-runtime.ts";
import {
	assertIdlePhase,
	assertPhaseTransition,
	HarnessBusyError,
} from "./live/phase.ts";
import { summarizeAndMergeBranch } from "./session/branching.ts";
import {
	emitPostCompact as emitPostCompactHelper,
	emitPreCompact as emitPreCompactHelper,
	emitSessionEnd as emitSessionEndHelper,
	emitSessionStart as emitSessionStartHelper,
} from "./session/lifecycle.ts";
import type {
	AbortResult,
	AgentSessionOptions,
	HarnessObserver,
	HarnessPluginHookFactory,
	HarnessPluginHookLayer,
	HarnessPluginLifecycle,
	HarnessPromptOptions,
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
export { HarnessBusyError } from "./live/phase.ts";
export type {
	AbortResult,
	AgentSessionOptions,
	HarnessModule,
	HarnessObserver,
	HarnessPromptOptions,
	HarnessQueues,
} from "./types.ts";
export { defineHarnessModule } from "./types.ts";

export { HarnessConfigurationError };

// ── UI Event Callback ───────────────────────────────────────────────────────

/** Minimal event shape for the host (AgentRuntime) to map to RuntimeEvent. */
export interface AgentSessionEvent {
	type: string;
	[key: string]: unknown;
}

// ── Types ───────────────────────────────────────────────────────────────────

type TurnRequest = { kind: "prompt"; text: string } | { kind: "continue" };

// ── AgentSession ────────────────────────────────────────────────────────────

export class AgentSession {
	private readonly configuration: ConfigurationStore<AgentConfig>;
	readonly models: HarnessModelController;
	private cwd?: string;
	private maxIterations?: number;

	private _phase: HarnessPhase = "idle";
	private runtime: AgentRuntimeState = createRuntimeState();
	private idleTools: ToolRegistry;
	private readonly turn = new HarnessTurnController();
	private loopConfig: AgentConfig | null = null;
	private readonly session: SessionState;
	private readonly contextController = new AdaptiveContextController(messages =>
		estimateChatPayloadTokens([...messages]),
	);
	private activeContextPlanId?: string;
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

	// ── UI event callback  ─────────────────────
	private onEvent?: (event: AgentSessionEvent) => void;

	constructor(
		options: AgentSessionOptions & {
			onEvent?: (event: AgentSessionEvent) => void;
		},
	) {
		this.onEvent = options.onEvent;
		this.session = new SessionState({
			steeringMode: options.config.steeringQueueMode ?? "one-at-a-time",
			followUpMode: options.config.followUpQueueMode ?? "one-at-a-time",
			onQueueChange: queues => this.emitQueueChange(queues),
		});
		const config = composeHarnessConfig(options.modules ?? [], options.config);
		this.configuration = new ConfigurationStore(config, {
			clone: AgentSession.cloneConfig,
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
			persistModel: model => this.session.store?.appendModelChange(model),
			persistThinking: level =>
				this.session.store?.appendThinkingLevelChange(level),
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
				sessionId: this.session.id ?? "",
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
		return this.session.threadItems;
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
		this.session.queue.clearCurrentTurn();
		const nextTurnCount = this.session.queue.snapshot().nextTurn.length;
		this.observation.settled(nextTurnCount);
		this.emitToSubscribers({ type: "agent_settled", nextTurnCount });
		// If continuation was pending, trigger it.
		if (this.session.takePendingContinuation()) {
			void this.runQueuedContinuation(this.session.repositoryQuery).catch(
				() => {},
			);
		}
	}

	// ── Structural operation: turns (prompt / continue) ─────────────────────

	async prompt(
		userMessage: string,
		options: HarnessPromptOptions = {},
	): Promise<Message[]> {
		this.assertIdle("prompt");
		if (this.autoCompactionSettings.enabled) {
			const compacted = await this.runAutoCompaction("auto");
			if (compacted) {
				return this.prompt(userMessage, options);
			}
		}
		return this.runTurn({ kind: "prompt", text: userMessage }, options);
	}

	/**
	 * Resume the agent loop from existing history without injecting a new user message.
	 * The last message in history must be a user or tool-result message (not assistant).
	 * Mirrors pi's agent.continue() — used when the agent stopped prematurely and
	 * the caller wants to re-enter the loop without fabricating a follow-up prompt.
	 */
	async continue(): Promise<Message[]> {
		this.assertIdle("continue");
		const nonSystem = this.session.conversation.history.filter(
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
	 * @param context Optional additional context to inject (e.g. formatted skill activations).
	 */
	async continueWithNextTurn(
		context?: string,
		repositoryQuery?: string,
		options: HarnessPromptOptions = {},
	): Promise<Message[]> {
		this.assertIdle("continue with next-turn guidance");
		return this.runTurn(
			{ kind: "prompt", text: "" },
			{
				continuationContext: context,
				repositoryQuery,
				contextContributions: options.contextContributions,
			},
		);
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
	private async runTurn(
		request: TurnRequest,
		options?: {
			continuationContext?: string;
			repositoryQuery?: string;
			contextContributions?: readonly ContextContribution[];
		},
	): Promise<Message[]> {
		return this.turn.run(
			signal =>
				this.runInPhase("turn", request.kind, async () => {
					await this.beginTurn(request);
					const snapshot = await this.createSnapshot(request, signal, options);
					let compactedContext: Message[] | undefined;
					const newMessages = await this.runLoop(
						request,
						snapshot,
						messages => {
							compactedContext = messages;
						},
						options,
					);
					return this.commitResult(snapshot, newMessages, compactedContext);
				}),
			() => this.endTurn(),
		);
	}

	/** Task/checkpoint setup that must happen before a snapshot is created. */
	private async beginTurn(_request: TurnRequest): Promise<void> {
		this.runController = new AgentRunController();
		this.session.conversation.checkpoint();

		if (!this.session.hasStarted) {
			await this.emitSessionStart("startup");
		}
	}

	/** Run the loop against the prepared snapshot. */
	private async runLoop(
		request: TurnRequest,
		snapshot: HarnessTurnSnapshot,
		onContextCompacted: (messages: Message[]) => void,
		_options?: {
			continuationContext?: string;
			repositoryQuery?: string;
		},
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
						// Extension and policy composition has lifetime. Provider and
						// model settings refresh without rebuilding hook state.
						hooks: snapshot.config.hooks,
					};
				},
				onContextCompacted: messages => {
					onContextCompacted(messages);
					this.persistCompactedContext(messages, this.estimatePayloadTokens());
				},
			},
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
		this.session.conversation.history = result;
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
		options?: {
			continuationContext?: string;
			repositoryQuery?: string;
			contextContributions?: readonly ContextContribution[];
		},
	): Promise<HarnessTurnSnapshot> {
		let initialMessages: Message[] = [...this.session.conversation.history];
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
			const nextTurnMessages = this.session.queue
				.dequeueNextTurn()
				.map(message => createUserMessage(message.content));
			if (nextTurnMessages.length > 0) this.emitQueueChange();

			const assembled = this.contextController.buildContext({
				// Queued user guidance is control-plane input, not optional retrieved
				// context. Keep it outside the adaptive budget and learning policy.
				history: [...initialMessages, ...nextTurnMessages],
				baseSystemPrompt: this.config.systemPrompt,
				objective: request.text,
				maxInjectedTokens: Math.max(
					2_048,
					Math.min(
						16_384,
						Math.floor(
							(this.autoCompactionSettings.contextWindow ?? 128_000) / 8,
						),
					),
				),
				contributions: [
					...(options?.contextContributions ?? []),
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
				],
			});
			this.activeContextPlanId = assembled.id;
			initialMessages = assembled.messages;
			systemPrompt = assembled.systemPrompt;
		}

		// Apply continuation context (skill activations, repository query).
		let finalSystemPrompt = systemPrompt;
		const dynamicContext: string[] = [];
		const repoQuery = options?.repositoryQuery;
		if (repoQuery) {
			dynamicContext.push(repoQuery);
		}
		const context = options?.continuationContext;
		if (context) {
			dynamicContext.push(context);
		}
		if (dynamicContext.length) {
			finalSystemPrompt = `${systemPrompt ?? this.config.systemPrompt ?? ""}\n\n${dynamicContext.join("\n\n")}`;
		}

		const baseConfig = {
			...this.snapshotConfig(),
			...(finalSystemPrompt ? { systemPrompt: finalSystemPrompt } : {}),
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
			getSessionId: () => this.session.id || "",
			getTranscriptPath: () => this.session.transcriptPath || "",
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
			this.session.conversation.history,
		);
	}

	private createPluginHookLayer(): HarnessPluginHookLayer {
		return (
			this.pluginHookFactory?.({
				enabled: this._hooksEnabled,
				sessionId: this.session.id ?? "",
				transcriptPath: this.session.transcriptPath ?? "",
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
				inject(this.session.queue.afterTurn().map(message => message.content)),
			getFollowUpMessages: async () =>
				inject(this.session.queue.onIdle().map(message => message.content)),
		};
	}

	private persistTurnMessages(messages: Message[]): void {
		if (!this.session.store) return;
		for (const message of messages) {
			try {
				this.session.store.append({
					role: message.role,
					content: message.content,
					tool_call_id: message.tool_call_id,
					tool_calls: message.tool_calls,
					name: message.name,
					timestamp: message.timestamp ?? Date.now(),
				});
			} catch (_e: unknown) {
				// Session persistence must never break a completed turn.
				console.error("[session] session append failed:", _e);
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
		this.session.store?.appendActiveToolsChange(patch.tools.map(t => t.name));
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
		this.session.steer(text, !!this.config.steeringInterrupt, () =>
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
		this.session.steer(text, false, () => {});
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
		this.session.nextTurn(text);
		this.turn.abort(createSteeringInterruptReason());
	}

	/** Promote queued steering into the immediate next turn and interrupt the current step. */
	flushSteeringNow(): number {
		if (this._phase !== "turn") {
			throw new HarnessBusyError("flush steering", this._phase, "turn");
		}
		return this.session.flushSteering(() =>
			this.turn.abort(createSteeringInterruptReason()),
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.session.dropQueuedMessage(displayIndex);
	}

	followUp(text: string): void {
		this.session.followUp(text);
	}

	nextTurn(text: string): void {
		this.session.nextTurn(text);
	}

	async abort(): Promise<AbortResult> {
		const result = this.session.abortQueues();
		this.runtime = { ...this.runtime, abortRequested: true };
		this.turn.abort();
		await this.waitForIdle();
		this.emitToSubscribers({ type: "abort", ...result });
		await this.emitSessionEnd("abort");
		return result;
	}

	// ── Queue state ────────────────────────────────────────────────────────

	getQueues(): HarnessQueues {
		return this.session.getQueues();
	}

	clearQueues(): HarnessQueues {
		return this.session.clearQueues();
	}

	private emitQueueChange(queues: HarnessQueues = this.getQueues()): void {
		this.observation.queue(queues);
		this.emitToSubscribers({
			type: "queue_update",
			steering: queues.steering,
			followUp: queues.followUp,
			nextTurn: queues.nextTurn,
		});
		this.onEvent?.({
			type: "queue_update",
			steering: queues.steering,
			followUp: queues.followUp,
			nextTurn: queues.nextTurn,
		});
		void this._extensionRunner?.emit({
			type: "queue_update",
			context: {
				sessionId: this.session.id || "",
				cwd: this.cwd || "",
				...queues,
			},
		});
	}

	setSteeringMode(mode: QueueMode): void {
		this.session.setQueueMode("steering", mode);
	}

	getSteeringMode(): QueueMode {
		return this.session.getQueueMode("steering");
	}

	setFollowUpMode(mode: QueueMode): void {
		this.session.setQueueMode("followUp", mode);
	}

	getFollowUpMode(): QueueMode {
		return this.session.getQueueMode("followUp");
	}

	// ── Plugin lifecycle hooks ─────────────────────────────────────────────

	setSessionId(id: string): void {
		this.session.setId(id);
		this.configuration.update({ hookSessionId: id });
	}

	private hookContext(): {
		sessionId: string;
		transcriptPath: string;
		cwd: string;
	} {
		return {
			sessionId: this.session.id || "",
			transcriptPath: this.session.transcriptPath || "",
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
		if (started) this.session.hasStarted = true;
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
	attachSession(session: SessionStore): void {
		this.session.attachStore(session);
	}

	get messages(): Message[] {
		return this.session.messages;
	}

	clearHistory(): void {
		this.emitSessionEnd("reset").catch(() => {});
		this.session.clearHistory();
		this.emitSessionStart("clear").catch(() => {});
	}

	setHistory(messages: Message[]): void {
		this.assertIdle("setHistory");
		this.emitSessionEnd("switch").catch(() => {});
		this.session.replaceHistory(messages);
		this.emitSessionStart("resume").catch(() => {});
	}

	/**
	 * Append messages to the live conversation without resetting session
	 * lifecycle. Used by direct-mode /spawn to record the spawn request and
	 * subagent result so later turns can see them.
	 */
	appendMessages(messages: Message[]): void {
		this.assertIdle("appendMessages");
		const toAdd = this.session.appendMessages(messages);
		if (toAdd.length) this.persistTurnMessages(toAdd);
	}

	rewind(): { messages: number; filesRestored: number } | null {
		this.assertIdle("rewind");
		return this.session.rewind();
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
		return this.session.fork(customSummary);
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
		const branch = this.session.conversation.activeBranch();
		if (!branch) return null;

		const current = this.session.conversation.history;
		const diverged = current.slice(branch.forkedAt);
		if (!diverged.length) {
			this.session.conversation.popBranch();
			this.session.store?.checkout(branch.sessionLeafId);
			this.session.conversation.history = branch.parent;
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
			this.session.conversation.popBranch();
			this.session.store?.checkout(branch.sessionLeafId);
			if (outcome.summaryText)
				this.session.store?.appendBranchSummary(
					outcome.summaryText,
					branch.sessionLeafId,
				);
			this.session.conversation.history = outcome.history;
			return outcome.summaryText;
		});
	}

	/**
	 * Discard the active branch without merging. Restores parent history.
	 */
	discardBranch(): boolean {
		this.assertIdle("discardBranch");
		return this.session.discardBranch();
	}

	listBranches(): BranchInfo[] {
		return this.session.listBranches();
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
		this.session.store?.appendCompaction(
			summary,
			tokensBefore,
			firstKeptEntryId,
		);
	}

	// ── Compaction ─────────────────────────────────────────────────────────

	async compact(): Promise<number | null> {
		this.assertIdle("compact");
		if (!this.session.conversation.history.length) return null;
		return this.runInPhase("compaction", "compact", () =>
			this.performCompaction("manual", /* force */ true),
		);
	}

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.session.conversation.history;
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
		const messages = this.session.conversation.history;
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
				sessionId: this.session.id || "",
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
			this.session.conversation.history !== messages ||
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

		this.session.conversation.history = result.messages;
		const after = result.tokensAfter;
		this.persistCompactedContext(result.messages, before);
		await this.emitPostCompact();
		await this._extensionRunner?.emit({
			type: "session_compact",
			context: {
				sessionId: this.session.id || "",
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

	// ── Continuation logic  ─────────────────────

	/**
	 * Set a pending continuation to fire after the current turn settles.
	 */
	setPendingContinuation(value: boolean): void {
		this.session.setPendingContinuation(value);
	}

	/**
	 * Execute queued continuation with the provided context.
	 * Used by AgentRuntime after a turn settles to auto-continue.
	 * @param context Optional formatted context string (e.g. skill activations).
	 */
	async runQueuedContinuation(
		context?: string,
		repositoryQuery?: string,
		options: HarnessPromptOptions = {},
	): Promise<boolean> {
		const originalPrompt = this.config.systemPrompt;
		const turnId = `turn_${Date.now()}`;
		try {
			const dynamicContext: string[] = [];
			if (repositoryQuery) {
				dynamicContext.push(repositoryQuery);
			}
			if (context) {
				dynamicContext.push(context);
			}
			if (dynamicContext.length) {
				this.configure({
					systemPrompt: `${originalPrompt}\n\n${dynamicContext.join("\n\n")}`,
				});
			}
			this.onEvent?.({ type: "turn_start", turnId });
			await this.continueWithNextTurn(context, repositoryQuery, options);
			return true;
		} finally {
			if (context || repositoryQuery) {
				this.configure({ systemPrompt: originalPrompt });
			}
			this.onEvent?.({ type: "turn_end", turnId });
			// A pending continuation is still active work. Keep READY terminal by
			// waiting for the final queued continuation instead of briefly exposing
			// an idle phase between streams.
			if (this.session.takePendingContinuation()) {
				await this.runQueuedContinuation(context, repositoryQuery, options);
			} else {
				this.onEvent?.({ type: "phase", state: "ready" });
			}
		}
	}

	/**
	 * Set the active repository query for continuation context injection.
	 */
	setRepositoryQuery(query: string | undefined): void {
		this.session.setRepositoryQuery(query);
	}

	/** Get the current repository query. */
	getRepositoryQuery(): string | undefined {
		return this.session.getRepositoryQuery();
	}

	// ── Phase mapping  ──────────────────────────

	/**
	 * Map harness structural phases to UI phase states.
	 * Called by the observer callback to emit phase events.
	 */
	emitHarnessPhase(phase: HarnessPhase): void {
		if (phase === "turn") return;
		const state =
			phase === "compaction"
				? "compacting"
				: phase === "branch_summary"
					? "branching"
					: "ready";
		this.onEvent?.({ type: "phase", state });
	}

	// ── Internals ──────────────────────────────────────────────────────────

	private emitToSubscribers(event: AgentEvent): void {
		if (this.activeContextPlanId && event.type === "acceptance_complete") {
			this.contextController.recordOutcome(this.activeContextPlanId, {
				success: event.status === "passed",
			});
			this.activeContextPlanId = undefined;
		} else if (this.activeContextPlanId && event.type === "agent_end") {
			this.contextController.recordOutcome(this.activeContextPlanId, {
				success: event.status === "completed",
			});
			this.activeContextPlanId = undefined;
		}
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
