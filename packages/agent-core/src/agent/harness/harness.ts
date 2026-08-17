// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//

import type { CompactionSettings } from "../../compaction/index.ts";
import type {
	ExtensionRunner,
	RegisteredTool,
} from "../../extensions/index.ts";
import { BudgetTracker } from "../../hooks/builtin/budget.ts";
import {
	buildBuiltinHooks,
	COMPACTION_COOLDOWN_TURNS,
} from "../../hooks/builtin/builtin-hooks.ts";
import { HookBus } from "../../hooks/native/hook-bus.ts";
import {
	type ClaudeCodeHookLayer,
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
} from "../../plugins/claude-code/hook-layer.ts";
import {
	type DeliveryMode,
	MessageDeliveryManager,
} from "../../queue/manager.ts";
import { withQueueEventForwarding } from "../../runtime/harness-queue-hooks.ts";
import { ToolRegistry } from "../../tools/shared/registry.ts";
import {
	throwOnValidationErrors,
	validateConfig,
} from "../config/config-validator.ts";
import {
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "../core/agent-loop-runner.ts";
import type { LLMBackend } from "../core/backend.ts";
import {
	type ContinuationDecision,
	ContinuationTracker,
	type RunBudgetStatus,
} from "../core/continuation-tracker.ts";
import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "../core/file-checkpoints.ts";
import { HarnessInterventionController } from "../core/intervention-controller.ts";
import {
	createUserMessage,
	estimateChatPayloadTokens,
} from "../core/messages.ts";
import {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
	reduceRuntimeState,
} from "../core/runtime-state.ts";
import { Session } from "../core/session.ts";
import { LoopDetector } from "../guards/loop-detector.ts";
import { OutputGuard } from "../guards/output-guard.ts";
import type { BranchInfo, BranchSummaryData } from "../summaries/types.ts";
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
} from "../types/index.ts";
import {
	type Branch,
	forkBranch,
	listBranches as listBranchesHelper,
	renderBranchTree,
	summarizeAndMergeBranch,
} from "./branching.ts";
import { runCompaction, shouldAutoCompact } from "./compaction.ts";
import type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
	HarnessTurnSnapshot,
} from "./contracts.ts";
import { cycleModel as cycleModelHelper, resolveModelUrl } from "./model.ts";
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
} from "../core/runtime-state.ts";
export type { BranchInfo, BranchSummaryData } from "../summaries/types.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./contracts.ts";
export { HarnessBusyError } from "./phase.ts";

// Conversation checkpoints: a snapshot of history is pushed before each
// prompt so a bad turn can be rewound. Bounded ring (newest last).
const MAX_CHECKPOINTS = 20;

type TurnRequest = { kind: "prompt"; text: string } | { kind: "continue" };

interface PreparedTurn {
	snapshot: HarnessTurnSnapshot;
}

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
	private budgetTracker: BudgetTracker | null = null;
	private compactionCooldown = { lastTurn: -COMPACTION_COOLDOWN_TURNS };
	private onQueueChange?: (queues: HarnessQueues) => void;
	private onPhaseChange?: (phase: HarnessPhase, prev: HarnessPhase) => void;
	private onSettled?: (
		nextTurnCount: number,
		reason: "steering_interrupt" | "normal",
	) => void;
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
	private outputGuard: OutputGuard;
	private _streamOptions: AgentHarnessStreamOptions = {};
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
			onCompact: async () => {
				const result = await this.compact();
				return result ?? null;
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
			? { taskId: state.taskId, status: state.status, compactionGeneration: state.compactionGeneration }
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
				if (resource === "provider_call") this.durableBudgetState.providerCalls += amount;
				else if (resource === "tool_call") this.durableBudgetState.toolCalls += amount;
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

	private settleTurn(): void {
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
		const nonSystem = this.history.filter(
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
		this.history = [...this.history, ...guidance];
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
				const signal = await this.preflight(request);
				const turn = await this.prepareTurn(request, signal);
				return await this.executeTurn(request, turn);
			} finally {
				this.settleTurn();
			}
		});
	}

	/** Task setup that must happen before a turn is prepared. */
	private async preflight(request: TurnRequest): Promise<AbortSignal> {
		if (request.kind === "prompt") {
			this.continuationTracker.startTask();
		}
		this.abortController = new AbortController();

		if (!this._hasStartedSession) {
			await this.emitSessionStart("startup");
		}
		return this.abortController.signal;
	}

	/** Snapshot the turn's context before execution. */
	private async prepareTurn(
		request: TurnRequest,
		signal: AbortSignal,
	): Promise<PreparedTurn> {
		this.checkpoints.push([...this.history]);
		if (this.checkpoints.length > MAX_CHECKPOINTS) {
			this.checkpoints.shift();
		}
		beginFileFrame();

		const snapshot =
			request.kind === "prompt"
				? await this.createTurnSnapshot(request.text, signal)
				: await this.createContinueSnapshot(signal);

		return { snapshot };
	}

	/** Run the loop against the prepared snapshot and fold the result into history. */
	private async executeTurn(
		request: TurnRequest,
		turn: PreparedTurn,
	): Promise<Message[]> {
		const { snapshot } = turn;
		this.loopConfig = snapshot.config;
		let compactedContext: Message[] | undefined;
		const prompts =
			request.kind === "prompt" ? [createUserMessage(snapshot.promptText)] : [];
		const newMessages = await runAgentLoop(
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
					compactedContext = messages;
					this.persistCompactedContext(messages, this.estimatePayloadTokens());
				},
			} satisfies RunAgentLoopConfig,
			async event => {
				await this.handleAgentEvent(event);
			},
		);
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
		this.history = result;
		return result;
	}

	private async createContinueSnapshot(
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		const config = this.forwardQueueEvents(
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
		const nextTurnMessages = this.msgManager.queue
			.dequeueNextTurn()
			.map(message => createUserMessage(message.content));
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
		const config = this.forwardQueueEvents(
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
		if (!this._extensionRunner) return undefined;

		const ctx = {
			sessionId: this._sessionId || "",
			cwd: this.cwd || "",
			prompt: promptText,
			systemPrompt: this.config.systemPrompt ?? "",
			messages: [...this.history],
		};

		let nativeMessages: Message[] | undefined;
		let nativeSystemPrompt: string | undefined;

		// Emit before_agent_start (→ native extensions + Pi's before_agent_start)
		if (this._extensionRunner.hasHandlers("before_agent_start")) {
			const result = await this._extensionRunner.emitToAll({
				type: "before_agent_start",
				context: ctx,
			});
			// Native extensions return { messages, systemPrompt } directly
			if (result && typeof result === "object") {
				const value = result as { messages?: Message[]; systemPrompt?: string };
				nativeMessages = Array.isArray(value.messages)
					? value.messages
					: undefined;
				nativeSystemPrompt =
					typeof value.systemPrompt === "string"
						? value.systemPrompt
						: undefined;
			}
		}

		return {
			messages: nativeMessages,
			systemPrompt: nativeSystemPrompt,
		};
	}

	private createClaudeCodeHookLayer(): ClaudeCodeHookLayer {
		return createClaudeCodeHookLayer({
			enabled: this._hooksEnabled,
			sessionId: this._sessionId || "",
			transcriptPath: this._transcriptPath || "",
			cwd: this.cwd || process.cwd(),
			getMatcherValue: toolName => {
				const tool = this.config.tools?.find(
					candidate => candidate.name === toolName,
				);
				return tool?.hookAliases?.join("|") || claudeToolMatcherName(toolName);
			},
		});
	}

	private withExtensionRuntime(
		config: AgentConfig,
		pluginHookLayer?: ClaudeCodeHookLayer,
	): AgentConfig {
		const runner = this._extensionRunner;
		const extensionTools = runner
			? runner.getTools().map(tool => this.wrapExtensionTool(tool))
			: [];
		const tools = [...(config.tools ?? []), ...extensionTools];

		// Rebuild HookBus layers each turn (extensions may add/remove tools/hooks).
		// Using a fresh bus avoids stale registrations between turns.
		const hookBus = new HookBus();

		// Budget-stop's tracker must survive across rebuilds to compare
		// consecutive turns (buildBuiltinHooks decides whether it's actually
		// enabled — this just ensures the same instance is reused if it is).
		this.budgetTracker ??= new BudgetTracker();

		// Layers run in registration order: builtin safeguards, then the
		// harness's own queue-draining, then extensions, then claude-code
		// compat, then caller-supplied hooks last so callers can override.
		const builtinHooks = buildBuiltinHooks({
			config,
			contextWindowTokens: () => config.contextWindowTokens,
			toolDefs: () => tools as unknown as Record<string, unknown>[],
			loopDetector: this.loopDetector,
			emitEvent: (event: { type: string; [key: string]: unknown }) => {
				this.emitToSubscribers(event as AgentEvent);
			},
			interventions: this.interventions,
			budget: this.budgetTracker,
			compactionCooldown: this.compactionCooldown,
		});
		hookBus.register(builtinHooks, { source: "builtin" });
		hookBus.register(this.drainHooks(), { source: "queue-drain" });

		const extensionHooks = runner?.getHooks();
		if (extensionHooks) {
			hookBus.register(extensionHooks, { source: "extensions" });
		}

		const claudeHooks = (pluginHookLayer ?? this.createClaudeCodeHookLayer())
			.hooks;
		if (claudeHooks) {
			hookBus.register(claudeHooks, { source: "claude-code-compat" });
		}

		if (config.hooks) {
			hookBus.register(config.hooks, { source: "user" });
		}

		return {
			...config,
			tools,
			hooks: hookBus.toHooks(),
		};
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
		if (event.type === "compaction") this.continuationTracker.recordCompaction();
		if (event.type === "run_outcome") {
			this.continuationTracker.finish(event.status, event.summary);
		}
		// The primary application event path is synchronous and latency-sensitive;
		// extension delivery must not hold streaming deltas behind an await.
		this.loopConfig?.onEvent?.(event);
		if (event.type === "message_end" && event.message) {
			this.loopConfig?.onEvent?.(event);
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
		const sessionBaseDir = baseDir ?? this._sessionBaseDir;
		const resumed = loadSessionMessages(sessionId, sessionBaseDir);
		if (!resumed) return false;

		if (resumed.messages.length > 0) {
			this.setActiveHistory(
				resumed.messages.filter(
					(m): m is Message => m != null && m.role !== "system",
				),
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
		this.setActiveHistory(
			messages.filter((m): m is Message => m != null && m.role !== "system"),
		);
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
		const { branch, nextBranchSeq } = forkBranch(
			this.branches,
			this.branchSeq,
			current,
			customSummary,
			this._session?.getLeafEntryId(),
		);
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
			this._session?.checkout(branch.sessionLeafId);
			this.setActiveHistory(branch.parent);
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
			this.branches.pop();
			this._session?.checkout(branch.sessionLeafId);
			if (outcome.summaryText)
				this._session?.appendBranchSummary(
					outcome.summaryText,
					branch.sessionLeafId,
				);
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
		this._session?.checkout(branch.sessionLeafId);
		this.setActiveHistory(branch.parent);
		return true;
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
		const messages = this.history;
		if (!messages.length) return null;
		const before = this.estimatePayloadTokens();

		return this.runInPhase("compaction", "compact", async () => {
			this.emitToSubscribers({ type: "compaction", reason: "manual" });
			await this._extensionRunner?.emit({
				type: "session_before_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason: "manual",
					tokensBefore: before,
					messages: [...messages],
				},
			});
			const preResult = await this.emitPreCompact({
				messages,
				tokensBefore: before,
				reason: "manual",
			});
			if (preResult?.cancel) {
				this.emitToSubscribers({
					type: "compaction",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: before,
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

			if (
				this.history !== messages ||
				!result.changed ||
				result.tokensAfter >= before
			) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: before,
				});
				return 0;
			}
			this.history = result.messages;
			const after = result.tokensAfter;
			this.persistCompactedContext(result.messages, before);
			this.continuationTracker.recordCompaction();
			this.onCompaction?.("manual", before, after);
			await this.emitPostCompact();
			await this._extensionRunner?.emit({
				type: "session_compact",
				context: {
					sessionId: this._sessionId || "",
					cwd: this.cwd || "",
					reason: "manual",
					tokensBefore: before,
					tokensAfter: after,
					changed: true,
					messages: [...result.messages],
				},
			});
			this.emitToSubscribers({
				type: "compaction",
				reason: "manual",
				tokensBefore: before,
				tokensAfter: after,
			});
			return before - after;
		});
	}

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.history;
		if (!messages.length || !this.autoCompactionSettings.enabled) return false;

		return this.runInPhase("compaction", "autoCompact", async () => {
			this.emitToSubscribers({ type: "compaction", reason });

			if (!this.shouldCompact(messages)) {
				await this.emitPostCompact();
				this.emitToSubscribers({
					type: "compaction",
					reason,
					tokensBefore: this.estimateContextTokens(),
					tokensAfter: this.estimateContextTokens(),
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
					type: "compaction",
					reason,
					tokensBefore: before,
					tokensAfter: before,
				});
				return false;
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
				this.history !== messages ||
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
				return false;
			}

			this.history = result.messages;
			const after = this.estimatePayloadTokens();
			this.persistCompactedContext(result.messages, before);
			this.continuationTracker.recordCompaction();
			this.onCompaction?.(reason, before, after);
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
			...(configured.some(option => option.model === this.config.model)
				? []
				: [this.config.model]),
			...configured.map(option => option.model),
		];
	}

	setModelEndpoint(model: string, baseUrl: string): void {
		this.config.model = model;
		this.config.baseUrl = baseUrl;
		this.backend =
			this.backend.withEndpoint?.(model, baseUrl) ??
			this.backend.withModel(model);
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
		const oldModel = this.config.model;
		const targetUrl = this.getModelUrl(model);
		if (targetUrl !== this.config.baseUrl) {
			this.config.baseUrl = targetUrl;
		}
		this.config.model = model;
		this._session?.appendModelChange(model);
		this.emitToSubscribers({
			type: "model_cycle",
			model: this.config.model,
			fromModel: oldModel,
			thinkingLevel: this.config.thinkingLevel,
		});
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

	private forwardQueueEvents(config: AgentConfig): AgentConfig {
		return withQueueEventForwarding(config, {
			onSavePoint: this.onSavePoint,
			subscribers: this._subscribers,
		});
	}
}
