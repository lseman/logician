// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above the functional agent runner. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.

import type { CompactionSettings } from "./compaction/index.ts";
import {
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./utils/agent-loop.ts";
import type { LLMBackend } from "./utils/backend.ts";
import {
	createUserMessage,
	estimateChatPayloadTokens,
} from "./messages.ts";
import {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
} from "./result.ts";
import type { ExtensionRunner } from "./utils/extension/index.ts";
import type { ClaudeCodeHookLayer } from "./utils/extension/adapters/claude-code/hook-layer.ts";
import { LoopDetector } from "./utils/guards/loop-detector.ts";
import { ToolRegistry } from "./tools/registry.ts";
import type { AgentConfig, AgentModelConfig, QueueMode } from "../types/types-config.ts";
import type {
	AgentEvent,
	AgentHooks,
	EventHandler,
	Message,
	Tool,
	BeforeCompactContext,
	BeforeCompactResult,
} from "../types/types-messages.ts";
import { summarizeAndMergeBranch } from "./summaries/branch-summarization.ts";
import type { BranchInfo, BranchSummaryData } from "./summaries/types.ts";
import { runCompaction, shouldAutoCompact } from "./session/compaction.ts";
import {
	throwOnValidationErrors,
	validateConfig,
} from "./utils/config-validator.ts";
import type { AbortResult, HarnessQueues } from "./result.ts";
import type { AgentHarnessOptions, HarnessTurnSnapshot } from "./types.ts";
import { ConversationState } from "./session/conversation-state.ts";
import type { ExtensionRuntimeDeps } from "./env/extension-runtime.ts";
import {
	createClaudeCodeHookLayerFor,
	createExtensionRuntimeState,
	runExtensionBeforeAgentStart as runExtensionBeforeAgentStartHelper,
	withExtensionRuntime as withExtensionRuntimeHelper,
} from "./env/extension-runtime.ts";
import type { ModelOpsDeps } from "./env/model/ops.ts";
import * as modelOps from "./env/model/ops.ts";
import { assertIdlePhase } from "./result.ts";
import { withQueueEventForwarding } from "./env/queue/hooks.ts";
import {
	type DeliveryMode,
	MessageDeliveryManager,
} from "./env/queue/manager.ts";
import type { QueueOpsDeps } from "./env/queue/ops.ts";
import * as queueOps from "./env/queue/ops.ts";
import { Session } from "./session/index.ts";
import {
	emitPostCompact as emitPostCompactHelper,
	emitPreCompact as emitPreCompactHelper,
	emitSessionEnd as emitSessionEndHelper,
	emitSessionStart as emitSessionStartHelper,
	listSessions as listSessionsHelper,
	loadSessionMessages,
} from "./session/lifecycle.ts";

export type {
	AgentRuntimeState,
	HarnessPhase,
} from "./types.ts";
export type { BranchInfo, BranchSummaryData } from "./summaries/types.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./types.ts";

type TurnRequest = { kind: "prompt"; text: string } | { kind: "continue" };

export class AgentHarness implements AgentHarnessApi {
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
	// Cooldown state for the built-in safeguard hooks. Owned here (not inside
	// buildBuiltinHooks) because withExtensionRuntime() rebuilds the hooks
	// object on every loop iteration (via refreshNextTurnConfig) — state that
	// must persist across iterations (budget-stop's consecutive-turn
	// comparison, the compaction cooldown) would otherwise silently reset
	// every time and never do its job.
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

	private _hooksEnabled: boolean;
	private _session?: Session;
	private _sessionBaseDir?: string;
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
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
			outcome: this.runtime.outcome ? { ...this.runtime.outcome } : undefined,
		};
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
		if (to !== "idle") assertIdlePhase(this._phase, op);
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
		this.msgManager.clearCurrentTurn();
		this.emitQueueChange();
		this._runResolve?.();
		this._runPromise = undefined;
		this._runResolve = undefined;
		const nextTurnCount = this.msgManager.nextTurn.size;
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
		const guidance = this.msgManager.nextTurn
			.drainAll()
			.map(message => createUserMessage(message.content));
		if (guidance.length === 0) {
			throw new Error("Cannot continue: no next-turn guidance queued");
		}
		this.conversation.append(guidance);
		this.emitQueueChange();
		return this.continue();
	}

	// ── Shared turn transaction ─────────────────────────────────────────────

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
			} catch (err) {
				this.runtime = {
					...this.runtime,
					phase: "idle",
					isStreaming: false,
					lastError: err instanceof Error ? err.message : String(err),
				};
				throw err;
			}
		});
	}

	private async beginTurn(request: TurnRequest): Promise<AbortSignal> {
		this.abortController = new AbortController();
		const signal = this.abortController.signal;

		// Start session if enabled and not yet started
		if (this._sessionBaseDir && !this._hasStartedSession) {
			await this.startSession();
		}

		// Create conversation checkpoint
		this.conversation.checkpoint();

		if (request.kind === "prompt") {
			// Drain next-turn guidance for new prompt
			const nextTurnMessages = this.msgManager.nextTurn.drainAll();
			const guidance = nextTurnMessages.map(m => createUserMessage(m.content));
			if (guidance.length > 0) {
				this.conversation.append(guidance);
			}
			// Add user message
			this.conversation.append([createUserMessage(request.text)]);
		}

		return signal;
	}

	private async createSnapshot(
		request: TurnRequest,
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		const initialMessages = [...this.conversation.history];
		const promptText = request.kind === "prompt" ? request.text : "";

		return {
			promptText,
			initialMessages,
			config: { ...this.config },
			streamOptions: this.getStreamOptions(),
			signal,
		};
	}

	private async runLoop(
		request: TurnRequest,
		snapshot: HarnessTurnSnapshot,
		onContextCompacted?: (messages: Message[]) => void,
	): Promise<Message[]> {
		const { config, signal } = snapshot;

		// Build hooks with extension runtime
		const deps: ExtensionRuntimeDeps = {
			getExtensionRunner: () => this._extensionRunner,
			getHooksEnabled: () => this._hooksEnabled,
			getSessionId: () => this._sessionId ?? "",
			getTranscriptPath: () => this._transcriptPath ?? "",
			getCwd: () => this.cwd ?? process.cwd(),
			getConfigTools: () => config.tools,
			loopDetector: this.loopDetector,
			emit: (event: AgentEvent) => this.emitToSubscribers(event),
			drainHooks: () => this.getDrainHooks(),
		};
		const extState = this.extensionRuntimeState;
		const enrichedConfig = withExtensionRuntimeHelper(
			deps,
			extState,
			config,
		);

		const runConfig: RunAgentLoopConfig = {
			...enrichedConfig,
			backend: this.backend,
			signal,
			maxIterations: this.maxIterations,
			onContextCompacted,
			refreshNextTurnConfig: () => {
				return this.buildNextTurnConfig();
			},
		};

		const systemPrompt = enrichedConfig.systemPrompt ?? "You are a helpful assistant.";

		return runAgentLoop(
			{ systemPrompt, messages: this.conversation.history },
			snapshot.initialMessages,
			runConfig,
			(event) => this.emitToSubscribers(event),
		);
	}

	private async commitResult(
		snapshot: HarnessTurnSnapshot,
		newMessages: Message[],
		compactedContext: Message[] | undefined,
	): Promise<Message[]> {
		if (compactedContext) {
			this.conversation.replace(compactedContext);
		}

		// Update conversation history with new messages
		const nonSystem = newMessages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
		this.conversation.history = [...this.conversation.history, ...nonSystem];

		// Write to session
		if (this._session) {
			for (const msg of newMessages) {
				if (msg.role !== "system") {
					this._session.append({
						role: msg.role,
						content: msg.content ?? null,
						tool_call_id: msg.tool_call_id,
						tool_calls: msg.tool_calls,
						name: msg.name,
						timestamp: msg.timestamp ?? Date.now(),
					});
				}
			}
		}

		this.endTurn();
		return newMessages;
	}

	// ── Config helpers ──────────────────────────────────────────────────────

	private buildNextTurnConfig(): AgentConfig | undefined {
		// Build fresh hooks for next turn
		const deps: ExtensionRuntimeDeps = {
			getExtensionRunner: () => this._extensionRunner,
			getHooksEnabled: () => this._hooksEnabled,
			getSessionId: () => this._sessionId ?? "",
			getTranscriptPath: () => this._transcriptPath ?? "",
			getCwd: () => this.cwd ?? process.cwd(),
			getConfigTools: () => this.config.tools,
			loopDetector: this.loopDetector,
			emit: (event: AgentEvent) => this.emitToSubscribers(event),
			drainHooks: () => this.getDrainHooks(),
		};
		const enriched = withExtensionRuntimeHelper(
			deps,
			this.extensionRuntimeState,
			this.config,
		);
		return enriched;
	}

	private getDrainHooks(): AgentHooks {
		const steering = this.msgManager.steering.drain();
		const followUp = this.msgManager.followUp.drain();

		return {
			getSteeringMessages: async ({ messages }) => {
				if (steering.length === 0) return undefined;
				return steering.map(msg => ({ role: "user" as const, content: msg.content }));
			},
			getFollowUpMessages: async () => {
				if (followUp.length === 0) return undefined;
				return followUp.map(msg => ({ role: "user" as const, content: msg.content }));
			},
		};
	}

	// ── Tool registry ───────────────────────────────────────────────────────

	private createToolRegistry(tools: Tool[]): ToolRegistry {
		const registry = new ToolRegistry({
			cwd: this.cwd,
			allowedPaths: this.config.allowedPaths,
			allowAllPaths: this.config.allowAllPaths,
			signal: undefined,
			onQuestionRequest: this.config.onQuestionRequest,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
		registry.registerMany(tools);
		return registry;
	}

	// ── Session management ──────────────────────────────────────────────────

	private async startSession(): Promise<void> {
		if (!this._sessionBaseDir || this._sessionId) return;

		try {
			await emitSessionStart(this._hooksEnabled, {
				sessionId: this._sessionId ?? "",
				transcriptPath: this._transcriptPath ?? "",
				cwd: this.cwd ?? process.cwd(),
			});
			this._hasStartedSession = true;
		} catch {
			// Session start failures must not block turns
		}
	}

	// ── Auto-compaction ─────────────────────────────────────────────────────

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const contextWindow = this.autoCompactionSettings.contextWindow;
		const reserveTokens = this.autoCompactionSettings.reserveTokens;
		const keepRecentTokens = this.autoCompactionSettings.keepRecentTokens;

		const currentTokens = estimateChatPayloadTokens(this.conversation.history);
		const threshold = contextWindow - reserveTokens;

		if (currentTokens <= threshold) return false;

		const tools = this.idleTools.toToolDefinitions();
		const tokensBefore = estimateChatPayloadTokens(
			this.conversation.history,
			tools,
		);

		const outcome = await runCompaction(this.backend, this.conversation.history, tokensBefore, {
			reason,
			temperature: this.config.temperature,
			maxTokens: this.config.maxTokens,
			thinkingLevel: this.config.thinkingLevel,
		});

		if (!outcome.changed) return false;

		// Apply compaction to conversation
		this.conversation.replace(outcome.messages);

		// Emit compaction event
		this.emitToSubscribers({
			type: "compaction",
			tokensBefore: outcome.tokensBefore,
			tokensAfter: outcome.tokensAfter,
		});

		return true;
	}

	// ── Queue operations ────────────────────────────────────────────────────

	private emitQueueChange(): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		this.onQueueChange?.(queueOps.getQueues(ops));
	}

	emitToSubscribers(event: AgentEvent): void {
		const now = event.ts ?? Date.now();
		for (const handler of this._subscribers) {
			handler({ ...event, ts: now });
		}
	}

	// ── Public API: config setters ──────────────────────────────────────────

	setSystemPrompt(systemPrompt: string): void {
		this.config.systemPrompt = systemPrompt;
	}

	setTemperature(temperature: number): void {
		this.config.temperature = temperature;
	}

	setSteeringInterrupt(enabled: boolean): void {
		this.config.steeringInterruptEnabled = enabled;
	}

	setInferenceMode(mode: string): void {
		this.config.inferenceMode = mode;
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
			>
		>,
	): void {
		if (options.guardsEnabled !== undefined) this.config.guardsEnabled = options.guardsEnabled;
		if (options.duplicateGuardEnabled !== undefined) this.config.duplicateGuardEnabled = options.duplicateGuardEnabled;
		if (options.failureGuardEnabled !== undefined) this.config.failureGuardEnabled = options.failureGuardEnabled;
		if (options.budgetStopEnabled !== undefined) this.config.budgetStopEnabled = options.budgetStopEnabled;
		if (options.continuationEnabled !== undefined) this.config.continuationEnabled = options.continuationEnabled;
	}

	setTools(tools: Tool[]): void {
		this.config.tools = tools;
	}

	// ── Public API: steering / follow-up / next-turn ────────────────────────

	steer(text: string): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		queueOps.steer(ops, text, this.config.steeringInterruptEnabled, this.abortController);
	}

	flushSteeringNow(): number {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.flushSteeringNow(ops, this.abortController);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.dropQueuedMessage(ops, displayIndex);
	}

	followUp(text: string): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		queueOps.followUp(ops, text);
	}

	nextTurn(text: string): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		queueOps.nextTurn(ops, text);
	}

	// ── Public API: abort ───────────────────────────────────────────────────

	async abort(): Promise<AbortResult> {
		const ops: queueOps.AbortDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
			abortController: this.abortController,
			setAbortRequested: () => {
				this.runtime.abortRequested = true;
			},
			waitForIdle: () => this.waitForIdle(),
			emitAbortEvent: (result) => {
				this.emitToSubscribers({
					type: "agent_abort",
					result,
				});
			},
			emitSessionEnd: (reason) => emitSessionEndHelper(this._hooksEnabled, {
				sessionId: this._sessionId ?? "",
				transcriptPath: this._transcriptPath ?? "",
				cwd: this.cwd ?? process.cwd(),
			}, reason),
		};
		return queueOps.abort(ops);
	}

	// ── Public API: queue management ────────────────────────────────────────

	getQueues(): HarnessQueues {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.getQueues(ops);
	}

	setOnQueueChange(cb: (queues: HarnessQueues) => void): void {
		this.onQueueChange = cb;
	}

	clearQueues(): HarnessQueues {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.clearQueues(ops);
	}

	setSteeringMode(mode: QueueMode): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		queueOps.setSteeringMode(ops, mode);
	}

	getSteeringMode(): QueueMode {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.getSteeringMode(ops);
	}

	setFollowUpMode(mode: QueueMode): void {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		queueOps.setFollowUpMode(ops, mode);
	}

	getFollowUpMode(): QueueMode {
		const ops: QueueOpsDeps = {
			msgManager: this.msgManager,
			getPhase: () => this._phase,
			emitQueueChange: () => this.emitQueueChange(),
		};
		return queueOps.getFollowUpMode(ops);
	}

	setHooksEnabled(enabled: boolean): void {
		this._hooksEnabled = enabled;
	}

	// ── Public API: session management ──────────────────────────────────────

	setSessionId(id: string): void {
		this._sessionId = id;
	}

	setTranscriptPath(path: string): void {
		this._transcriptPath = path;
	}

	async enableSession(baseDir?: string): Promise<void> {
		this._sessionBaseDir = baseDir;
		if (!this._sessionId) {
			// Session will be started on first prompt
		}
	}

	attachSession(session: Session): void {
		this._session = session;
		this._hasStartedSession = true;
	}

	async resumeSession(sessionId: string, baseDir?: string): Promise<boolean> {
		this._sessionBaseDir = baseDir;
		const result = loadSessionMessages(sessionId, baseDir);
		if (!result) return false;
		this._session = result.session;
		this.conversation.replace(result.messages);
		this._hasStartedSession = true;
		return true;
	}

	listSessions(): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}> {
		return listSessionsHelper(this._sessionBaseDir);
	}

	// ── Public API: conversation management ─────────────────────────────────

	clearHistory(): void {
		this.conversation.clear();
	}

	setHistory(messages: Message[]): void {
		this.conversation.replace(messages);
	}

	appendMessages(messages: Message[]): void {
		this.conversation.append(messages);
	}

	rewind(): { messages: number; filesRestored: number } | null {
		return this.conversation.rewind();
	}

	// ── Public API: branching ───────────────────────────────────────────────

	fork(customSummary?: BranchSummaryData): string {
		return this.conversation.fork(customSummary);
	}

	async branchSummary(options?: {
		customInstructions?: string;
	}): Promise<string | null> {
		const branch = this.conversation.activeBranch();
		if (!branch) return null;
		return summarizeAndMergeBranch(this.backend, this.conversation.history, {
			customInstructions: options?.customInstructions,
		});
	}

	discardBranch(): boolean {
		return this.conversation.discardBranch() !== undefined;
	}

	branchTree(): string {
		return this.conversation.branchTree();
	}

	listBranches(): BranchInfo[] {
		return this.conversation.listBranches();
	}

	// ── Public API: compaction ──────────────────────────────────────────────

	setAutoCompactionSettings(settings: Partial<CompactionSettings>): void {
		this.autoCompactionSettings = {
			...this.autoCompactionSettings,
			...settings,
		};
	}

	enableAutoCompaction(enabled: boolean): void {
		this.autoCompactionSettings.enabled = enabled;
	}

	async compact(): Promise<number | null> {
		const currentTokens = estimateChatPayloadTokens(this.conversation.history);
		const outcome = await runCompaction(this.backend, this.conversation.history, currentTokens, {
			reason: "manual",
		});
		if (!outcome.changed) return null;
		this.conversation.replace(outcome.messages);
		return outcome.tokensAfter;
	}

	// ── Public API: config getters ──────────────────────────────────────────

	getCurrentConfig(): Readonly<AgentConfig> {
		return this.config;
	}

	getTemperature(): number {
		return this.config.temperature ?? 0.7;
	}

	getToolCount(): number {
		return this.idleTools.size;
	}

	// ── Public API: model operations ────────────────────────────────────────

	getModel(): string {
		return this.backend.model;
	}

	getBaseUrl(): string {
		return (this.backend as any).baseUrl ?? "";
	}

	getModels(): string[] {
		return this.config.models?.map(m => m.model) ?? [];
	}

	setModelEndpoint(model: string, baseUrl: string): void {
		const deps: ModelOpsDeps = {
			getConfig: () => ({
				model: this.backend.model,
				baseUrl: (this.backend as any).baseUrl ?? "",
				models: this.config.models,
			}),
			setModel: (m: string) => { }, // handled by backend.withModel
			setBaseUrl: (url: string) => { }, // handled by backend
			setModels: (models: AgentModelConfig[]) => { this.config.models = models; },
			setThinkingLevel: (level: string) => { this.config.thinkingLevel = level; },
			getBackend: () => this.backend,
			setBackend: (backend: LLMBackend) => { this.backend = backend; },
			appendModelChange: () => { },
			appendThinkingLevelChange: () => { },
			emit: (event: AgentEvent) => this.emitToSubscribers(event),
		};
		modelOps.setModelEndpoint(deps, model, baseUrl);
		this.backend = this.backend.withEndpoint?.(model, baseUrl) ?? this.backend.withModel(model);
	}

	setModels(models: AgentModelConfig[]): void {
		this.config.models = models;
	}

	cycleModel(direction?: "forward" | "backward"): string {
		const deps: ModelOpsDeps = {
			getConfig: () => ({
				model: this.backend.model,
				baseUrl: (this.backend as any).baseUrl ?? "",
				models: this.config.models,
				thinkingLevel: this.config.thinkingLevel,
			}),
			setModel: (m: string) => { },
			setBaseUrl: (url: string) => { },
			setModels: (models: AgentModelConfig[]) => { this.config.models = models; },
			setThinkingLevel: (level: string) => { this.config.thinkingLevel = level; },
			getBackend: () => this.backend,
			setBackend: (backend: LLMBackend) => { this.backend = backend; },
			appendModelChange: () => { },
			appendThinkingLevelChange: () => { },
			emit: (event: AgentEvent) => this.emitToSubscribers(event),
		};
		const result = modelOps.cycleModel(deps, direction);
		this.backend = this.backend.withModel(result);
		return result;
	}

	getThinkingLevel(): string {
		return this.config.thinkingLevel ?? "off";
	}

	setThinkingLevel(level: string): void {
		this.config.thinkingLevel = level;
	}

	setModel(model: string): void {
		this.backend = this.backend.withModel(model);
	}

	setBackend(backend: LLMBackend): void {
		this.backend = backend;
	}

	// ── Public API: tools ───────────────────────────────────────────────────

	get tools(): ToolRegistry {
		return this.idleTools;
	}

	getLoopDetector(): LoopDetector {
		return this.loopDetector;
	}

	get messages(): Message[] {
		return this.conversation.history;
	}

	// ── Public API: session helpers ─────────────────────────────────────────

	get sessions(): {
		list: typeof listSessionsHelper;
		resume: typeof loadSessionMessages;
	} {
		return {
			list: (baseDir?: string) => listSessionsHelper(baseDir ?? this._sessionBaseDir),
			resume: (sessionId: string, baseDir?: string) =>
				loadSessionMessages(sessionId, baseDir ?? this._sessionBaseDir),
		};
	}

	// ── Public API: compaction helpers ──────────────────────────────────────

	get compaction(): {
		shouldAutoCompact: typeof shouldAutoCompact;
	} {
		return {
			shouldAutoCompact: (settings: CompactionSettings, messages: Message[]) =>
				shouldAutoCompact(settings, messages),
		};
	}
}

// ── AgentHarness public contract ────────────────────────────────────────────

export interface AgentHarnessApi {
	get phase(): HarnessPhase;
	get runtimeState(): AgentRuntimeState;
	get messages(): Message[];
	get tools(): ToolRegistry;

	setOnPhaseChange(cb: (phase: HarnessPhase, prev: HarnessPhase) => void): void;
	setOnSettled(
		cb: (
			nextTurnCount: number,
			reason: "steering_interrupt" | "normal",
		) => void,
	): void;
	subscribe(handler: EventHandler): () => void;
	waitForIdle(): Promise<void>;
	setBeforeAgentStart(
		cb: (
			promptText: string,
		) =>
			| Promise<{ messages?: Message[]; systemPrompt?: string } | undefined>
			| { messages?: Message[]; systemPrompt?: string }
			| undefined,
	): void;
	setExtensionRunner(runner: ExtensionRunner | undefined): void;

	getStreamOptions(): AgentHarnessStreamOptions;
	setStreamOptions(options: Partial<AgentHarnessStreamOptions>): void;

	prompt(userMessage: string): Promise<Message[]>;
	continue(): Promise<Message[]>;
	continueWithNextTurn(): Promise<Message[]>;

	getLoopDetector(): LoopDetector;

	setSystemPrompt(systemPrompt: string): void;
	setTemperature(temperature: number): void;
	setSteeringInterrupt(enabled: boolean): void;
	setInferenceMode(mode: string): void;
	setMaxTokens(maxTokens: number): void;
	setMaxIterations(maxIterations: number): void;
	setExecutionProfile(
		profile: NonNullable<AgentConfig["executionProfile"]>,
	): void;
	setRuntimeOptions(
		options: Partial<
			Pick<
				AgentConfig,
				| "guardsEnabled"
				| "duplicateGuardEnabled"
				| "failureGuardEnabled"
				| "budgetStopEnabled"
				| "continuationEnabled"
			>
		>,
	): void;
	setTools(tools: Tool[]): void;

	steer(text: string): void;
	flushSteeringNow(): number;
	dropQueuedMessage(displayIndex: number): string | undefined;
	followUp(text: string): void;
	nextTurn(text: string): void;
	abort(): Promise<AbortResult>;

	getQueues(): HarnessQueues;
	setOnQueueChange(cb: (queues: HarnessQueues) => void): void;
	clearQueues(): HarnessQueues;
	setSteeringMode(mode: QueueMode): void;
	getSteeringMode(): QueueMode;
	setFollowUpMode(mode: QueueMode): void;
	getFollowUpMode(): QueueMode;

	setHooksEnabled(enabled: boolean): void;
	setSessionId(id: string): void;
	setTranscriptPath(path: string): void;

	setAutoCompactionSettings(settings: Partial<CompactionSettings>): void;
	enableAutoCompaction(enabled: boolean): void;

	enableSession(baseDir?: string): Promise<void>;
	attachSession(session: Session): void;
	resumeSession(sessionId: string, baseDir?: string): Promise<boolean>;
	listSessions(): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}>;
	clearHistory(): void;
	setHistory(messages: Message[]): void;
	appendMessages(messages: Message[]): void;
	rewind(): { messages: number; filesRestored: number } | null;

	fork(customSummary?: BranchSummaryData): string;
	branchSummary(options?: {
		customInstructions?: string;
	}): Promise<string | null>;
	discardBranch(): boolean;
	branchTree(): string;
	listBranches(): BranchInfo[];

	compact(): Promise<number | null>;

	getCurrentConfig(): Readonly<AgentConfig>;
	getTemperature(): number;
	getToolCount(): number;

	getModel(): string;
	getBaseUrl(): string;
	getModels(): string[];
	setModelEndpoint(model: string, baseUrl: string): void;
	setModels(models: AgentModelConfig[]): void;
	cycleModel(direction?: "forward" | "backward"): string;

	getThinkingLevel(): string;
	setThinkingLevel(level: string): void;

	setModel(model: string): void;
	setBackend(backend: LLMBackend): void;
}
