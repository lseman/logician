// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above AgentLoop. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn, and steering / follow-up /
// nextTurn queues drained at save points.
//
// Phase types extracted to harness-phase.ts for reference.
// Compaction logic extracted to harness-compaction.ts for reference.

import { get_reasoner, getReasonerMeta } from "../../reasoners/registry.ts";
import { withTimeout } from "../tools/shared/async-utils.ts";
import { runSessionStartHooks, runHookEvent } from "../tools/shared/plugins.ts";
import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "./file-checkpoints.ts";
import type { LLMBackend } from "./backend.ts";
import { AgentLoop } from "./loop.ts";
import { runStatelessAgentLoop } from "./stateless-loop.ts";
import { Session, SessionManager } from "./session.ts";
import { compactMessages, createUserMessage } from "./messages.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import { MessageDeliveryManager, type DeliveryMode } from "../message-queue/manager.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHarnessStreamOptions,
	AgentLoopHooks,
	EventHandler,
	Message,
	QueueMode,
	Tool,
} from "./types.ts";
import type { CompactionSettings } from "../compaction/index.ts";

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

/** Public branch info for UI / callers. */
export interface BranchInfo {
	id: string;
	depth: number;
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
	private loop: AgentLoop | null = null;
	private idleTools: ToolRegistry;
	private abortController: AbortController | null = null;
	private loopConfig: AgentConfig | null = null;
	private history: Message[] = [];
	private branches: Array<{ id: string; parent: Message[]; forkedAt: number }> =
		[];
	private branchSeq = 0;
	private checkpoints: Message[][] = [];
	private _nextTurnQueue: string[] = [];
	private msgManager: MessageDeliveryManager;
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
		contextWindowTokens: 128_000,
	};
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
	private _beforeAgentStart?: (
		promptText: string,
	) =>
		| Promise<{ messages?: Message[]; systemPrompt?: string } | undefined>
		| { messages?: Message[]; systemPrompt?: string }
		| undefined;

	constructor(options: AgentHarnessOptions) {
		this.config = options.config;
		this.backend = options.backend;
		this.cwd = options.cwd;
		this.maxIterations = options.maxIterations;
		this.idleTools = this.createToolRegistry(this.config.tools ?? []);
		this.msgManager = new MessageDeliveryManager({
			steeringMode: (options.config.steeringQueueMode ?? "one-at-a-time") as DeliveryMode,
			followUpMode: (options.config.followUpQueueMode ?? "one-at-a-time") as DeliveryMode,
		});
	}

	get phase(): HarnessPhase {
		return this._phase;
	}

	setOnPhaseChange(cb: (phase: HarnessPhase, prev: HarnessPhase) => void): void {
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
			const snapshot = await this.createTurnSnapshot(promptText, this.abortController.signal);

			try {
				this.loopConfig = snapshot.config;
				const result = await runStatelessAgentLoop({
					config: snapshot.config,
					backend: this.backend,
					prompt: snapshot.promptText,
					cwd: this.cwd,
					maxIterations: this.maxIterations,
					signal: snapshot.signal,
					initialMessages: snapshot.initialMessages,
					onLoopReady: (loop) => {
						this.loop = loop;
					},
				});
				this.persistTurnMessages(result.newMessages);
				this.history = result.messages;
				return result.messages;
			} finally {
				this.abortController = null;
				this.msgManager.queue.clear();
				this.emitQueueChange();
				this._runResolve?.();
				this._runPromise = undefined;
				this._runResolve = undefined;
				this.onSettled?.(this._nextTurnQueue.length);
			}
		});
	}

	private async createTurnSnapshot(
		promptText: string,
		signal: AbortSignal,
	): Promise<HarnessTurnSnapshot> {
		const beforeStart = await this._beforeAgentStart?.(promptText);
		const preReasoning = await this.runPreReasoner(promptText);

		let initialMessages: Message[] = preReasoning
			? [...this.history, { role: "assistant", content: preReasoning, tool_calls: [] }]
			: [...this.history];

		if (beforeStart?.messages?.length) {
			initialMessages = [...beforeStart.messages, ...initialMessages];
		}

		const baseConfig = beforeStart?.systemPrompt
			? { ...this.config, systemPrompt: beforeStart.systemPrompt }
			: this.config;
		const config = this.withDrainHook(baseConfig);
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

	private async runPreReasoner(
		promptText: string,
	): Promise<string | undefined> {
		const reasonerId = this.config.reasonerId;
		if (!reasonerId || reasonerId === "none") return undefined;
		const meta = getReasonerMeta(reasonerId);
		if (!meta) return undefined;
		try {
			const reasoner = get_reasoner(
				reasonerId,
				this.backend,
				meta.defaultConfig,
			);
			const trace = await withTimeout(reasoner.solve(promptText), 60_000);
			return trace.reasoning || undefined;
		} catch {
			return undefined;
		}
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
		if (this.loop) {
			const registry = this.loop.tools;
			for (const existing of registry.list()) {
				registry.unregister(existing.name);
			}
			registry.registerMany(tools);
		}
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
			this.loop?.interruptTurn();
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

	abort(): void {
		this.abortController?.abort();
		this.msgManager.queue.clear();
		this._nextTurnQueue = [];
		this.emitQueueChange();
		this.emitSessionEnd("abort").catch(() => {});
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
		this.onQueueChange?.(this.getQueues());
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

	private async emitPreCompact(): Promise<void> {
		if (!this._hooksEnabled) return;
		try {
			await runHookEvent("PreCompact", {
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
			});
		} catch {
			// must not block compaction
		}
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
			const session = new Session(sessionId, { baseDir: sessionBaseDir, enabled: true });
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
			const baseDir = this._sessionBaseDir ?? (this._session ? `${this._session.dirPath}/../..` : undefined);
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
		return this.loop?.messages ?? this.history;
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

	fork(): string {
		this.assertIdle("fork");
		const current = this.activeHistory();
		const branch: typeof this.branches[number] = {
			id: `branch_${++this.branchSeq}`,
			parent: current,
			forkedAt: current.length,
		};
		this.branches.push(branch);
		this.setActiveHistory([...current]);
		return branch.id;
	}

	async branchSummary(): Promise<string | null> {
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
			const summary =
				(await this.generateSummary(
					diverged.map((m) => ({
						role: m.role as Message["role"],
						content: m.content ?? "",
						tool_call_id: m.tool_call_id,
						tool_calls: m.tool_calls,
						name: m.name,
						timestamp: m.timestamp,
					})),
					[],
				)) ??
				`[branch ${branch.id}: ${diverged.length} messages explored]`;

			this.branches.pop();
			this.setActiveHistory([
				...branch.parent,
				{ role: "assistant", content: `Branch summary: ${summary}`, tool_calls: [] },
			]);
			return summary;
		});
	}

	discardBranch(): boolean {
		this.assertIdle("discardBranch");
		const branch = this.branches.pop();
		if (!branch) return false;
		this.setActiveHistory(branch.parent);
		return true;
	}

	listBranches(): BranchInfo[] {
		return this.branches.map((b, i) => ({ id: b.id, depth: i + 1 }));
	}

	// ── Conversation management ────────────────────────────────────────────

	private activeHistory(): Message[] {
		return this.loop?.messages ?? this.history;
	}

	private setActiveHistory(messages: Message[]): void {
		this.history = messages;
		this.loop?.setMessages(messages);
	}

	// ── Compaction ─────────────────────────────────────────────────────────

	async compact(): Promise<number | null> {
		this.assertIdle("compact");
		const messages = this.loop?.messages ?? this.history;
		if (!messages.length) return null;
		const before = this.estimatePayloadTokens();

		return this.runInPhase("compaction", "compact", async () => {
			await this.emitPreCompact();

			const result = await compactMessages(messages, {
				reason: "manual",
				summarize: (older, system) =>
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
				return 0;
			}
			if (this.loop) this.loop.setMessages(result.messages);
			else this.history = result.messages;
			const after = this.estimatePayloadTokens();
			this.onCompaction?.("manual", before, after);
			await this.emitPostCompact();
			return before - after;
		});
	}

	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.loop?.messages ?? this.history;
		if (!messages.length || !this.autoCompactionSettings.enabled) return false;

		return this.runInPhase("compaction", "autoCompact", async () => {
			await this.emitPreCompact();

			const agentMessages = messages as unknown as Array<{
				role: string;
				content?: unknown[];
			}>;

			if (!this.shouldCompact(agentMessages)) {
				await this.emitPostCompact();
				return false;
			}

			const before = this.estimateContextTokens();

			const compactionResult = await (async () => {
				const compactionMessages = agentMessages as unknown as Message[];
				const summary = await this.generateSummary(
					compactionMessages,
					[],
				);
				return {
					summary:
						summary ||
						"[Auto-compaction summary failed — context preserved but no summary generated]",
					messagesToKeep: compactionMessages.slice(-Math.floor(compactionMessages.length / 2)),
				};
			})();

			const summaryMessage: Message = {
				role: "system",
				content: `\n<compaction_summary>${compactionResult.summary}</compaction_summary>\n`,
			};

			const newHistory = [
				summaryMessage,
				...compactionResult.messagesToKeep.filter((m) => m.content),
			];
			this.history = newHistory;
			if (this.loop) this.loop.setMessages(newHistory);

			const after = this.estimatePayloadTokens();
			this.onCompaction?.(reason, before, after);
			await this.emitPostCompact();
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
				chatMessages,
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

	// ── Tool registry ──────────────────────────────────────────────────────

	get tools(): ToolRegistry {
		return this.loop?.tools ?? this.idleTools;
	}

	// ── Config getters ─────────────────────────────────────────────────────

	getTemperature(): number {
		return this.config.temperature ?? 0.7;
	}

	getToolCount(): number {
		return this.config.tools?.length ?? 0;
	}

	// ── Model cycling ──────────────────────────────────────────────────────

	getModel(): string {
		return this.loop?.getModel() ?? this.config.model;
	}

	getModels(): string[] {
		return (
			this.loop?.getModels() ??
			(this.config.models
				? [this.config.model, ...this.config.models]
				: [this.config.model])
		);
	}

	cycleModel(direction: "forward" | "backward" = "forward"): string {
		const model = this.loop?.cycleModel(direction) ?? this.config.model;
		this.emitToSubscribers({ type: "model_update", model });
		return model;
	}

	// ── Thinking level ─────────────────────────────────────────────────────

	getThinkingLevel(): string {
		return this.config.thinkingLevel ?? "medium";
	}

	setThinkingLevel(level: string): void {
		this.config.thinkingLevel = level as
			| "off"
			| "minimal"
			| "low"
			| "medium"
			| "high"
			| "xhigh";
	}

	// ── Model & provider ──────────────────────────────────────────────────

	setModel(model: string): void {
		this.config.model = model;
		this.emitToSubscribers({ type: "model_update", model });
	}

	setBackend(backend: LLMBackend): void {
		this.backend = backend;
	}

	// ── Internals ──────────────────────────────────────────────────────────

	private emitToSubscribers(event: AgentEvent): void {
		for (const handler of this._subscribers) handler(event);
	}

	private estimatePayloadTokens(): number {
		const msgs = this.messages;
		return msgs.reduce((acc, m) => acc + (m.content?.length ?? 0) / 4, 0) + msgs.length * 4;
	}

	private estimateContextTokens(): number {
		return this.estimatePayloadTokens();
	}

	private shouldCompact(_messages: Array<{ role: string }>): boolean {
		const settings = this.autoCompactionSettings;
		if (!settings.enabled) return false;
		const contextWindow = settings.contextWindow ?? settings.contextWindowTokens ?? 128000;
		const threshold = contextWindow - (settings.reserveTokens ?? 16384);
		const currentTokens = this.estimateContextTokens();
		return currentTokens > threshold;
	}

	private withDrainHook(config: AgentConfig): AgentConfig {
		const internalHooks: AgentLoopHooks = {};

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
