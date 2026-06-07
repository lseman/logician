// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above AgentLoop. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn (never mutating an in-flight
// provider request), and steering / follow-up / nextTurn queues drained at
// save points. Mirrors pi's AgentHarness (packages/agent/docs/agent-harness.md),
// scoped to what logician-tui's loop exposes.
//
// The loop already exposes a save point: the `prepareNextTurn` contract hook
// fires after each turn and can rewrite the working messages. The harness
// installs a drain hook there to inject queued messages and apply config
// changes between turns.

import type { LLMBackend } from "./backend.ts";
import { AgentLoop } from "./loop.ts";
import {
	compactMessages,
	convertToChatFormat,
	createAssistantMessage,
	createUserMessage,
	estimateChatPayloadTokens,
	microCompactMessages,
} from "./messages.ts";
import type { ToolRegistry } from "./tools/registry.ts";
import {
	AgentError,
	AgentErrorType,
	type AgentConfig,
	type AgentLoopHooks,
	type Message,
	type QueueMode,
	type Tool,
} from "./types.ts";
import { get_reasoner, getReasonerMeta } from "../reasoners/registry.ts";

export type HarnessPhase = "idle" | "turn";

export class HarnessBusyError extends Error {
	constructor(op: string) {
		super(`AgentHarness is busy; ${op} requires idle phase`);
		this.name = "HarnessBusyError";
	}
}

export interface AgentHarnessOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
}

/** Snapshot of the three harness queues, for UI display. */
export interface HarnessQueues {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}

export class AgentHarness {
	private config: AgentConfig;
	private backend: LLMBackend;
	private cwd?: string;
	private maxIterations?: number;

	private _phase: HarnessPhase = "idle";
	private loop: AgentLoop | null = null;
	private abortController: AbortController | null = null;
	// Config snapshot for loop reuse. Updated when the harness config changes.
	private loopConfig: AgentConfig | null = null;
	// Conversation persisted across prompts so follow-ups ("continue", "go on")
	// retain context.
	private history: Message[] = [];

	// Queues drained at save points. The harness owns the single source of
	// truth for all three queues; the bridge/UI read snapshots via getQueues()
	// and subscribe to onQueueChange rather than keeping their own copies.
	private steeringQueue: string[] = [];
	private followUpQueue: string[] = [];
	// nextTurn messages survive across runs and abort; injected before the next
	// user prompt.
	private nextTurnQueue: string[] = [];
	// Queue drain modes: "all" = drain everything, "one-at-a-time" = drain one.
	private steeringQueueMode: QueueMode;
	private followUpQueueMode: QueueMode;
	// Fired whenever any queue changes (enqueue, drain, clear) so the UI can
	// reflect the live queue state without polling or mirroring.
	private onQueueChange?: (queues: HarnessQueues) => void;

	constructor(options: AgentHarnessOptions) {
		this.config = options.config;
		this.backend = options.backend;
		this.cwd = options.cwd;
		this.maxIterations = options.maxIterations;
		this.steeringQueueMode =
			options.config.steeringQueueMode ?? "one-at-a-time";
		this.followUpQueueMode =
			options.config.followUpQueueMode ?? "one-at-a-time";
	}

	get phase(): HarnessPhase {
		return this._phase;
	}

	// ── Structural operation: prompt ───────────────────────────────────────
	// Rejected while busy. Drains nextTurn messages before the user prompt.
	async prompt(userMessage: string): Promise<Message[]> {
		if (this._phase !== "idle") throw new HarnessBusyError("prompt");
		this._phase = "turn";
		this.abortController = new AbortController();

		const pending = this.nextTurnQueue.splice(0);
		if (pending.length) this.emitQueueChange();
		const promptText = pending.length
			? `${pending.join("\n\n")}\n\n${userMessage}`
			: userMessage;

		// Optional reasoner pre-phase: run structured reasoning before ReAct.
		// Timeout prevents a slow reasoner from blocking the entire prompt.
		const reasonerId = this.config.reasonerId;
		let preReasoning: string | undefined;
		if (reasonerId && reasonerId !== "none") {
			const meta = getReasonerMeta(reasonerId);
			if (meta) {
				try {
					const reasoner = get_reasoner(
						reasonerId,
						this.backend,
						meta.defaultConfig,
					);
					const trace = await this.withTimeout(
						reasoner.solve(promptText),
						60_000, // 60s reasoner timeout
					);
					if (trace.reasoning) {
						preReasoning = trace.reasoning;
					}
				} catch {
					// Reasoner failure or timeout → fall back to plain ReAct.
				}
			}
		}

		try {
			// Build or update the loop config (new on config change).
			if (!this.loopConfig) {
				this.loopConfig = this.withDrainHook(this.config);
				this.loop = new AgentLoop({
					config: this.loopConfig,
					backend: this.backend,
					cwd: this.cwd,
					maxIterations: this.maxIterations,
					signal: this.abortController.signal,
					initialMessages: this.history.length ? this.history : undefined,
				});
			} else {
				// Reuse existing loop but update signal (abort) and config changes.
				this.loopConfig = this.withDrainHook(this.config);
				this.loop!.updateSignal(this.abortController.signal);
				// Config mutations (systemPrompt, temperature, tools, etc.)
				// take effect live on the next turn since the loop reads
				// this.loopConfig each iteration.
			}

			// Local reference so TS knows loop is non-null in this scope.
			const loop = this.loop!;

			// Inject synthetic assistant message with reasoner output.
			if (preReasoning) {
				loop.setMessages([
					createAssistantMessage(preReasoning),
					...this.history,
				]);
			}

			const result = await loop.run(promptText);
			// Persist the full conversation for the next prompt.
			this.history = result;
			return result;
		} finally {
			this._phase = "idle";
			this.abortController = null;
			// Steering/follow-up are turn-scoped; clear leftovers.
			this.steeringQueue = [];
			this.followUpQueue = [];
			this.emitQueueChange();
		}
	}

	// ── Runtime config setters (take effect next turn) ─────────────────────
	// The loop reads these config fields live each turn, so updating them here
	// affects the next turn snapshot, not the in-flight provider request.

	setSystemPrompt(systemPrompt: string): void {
		this.config.systemPrompt = systemPrompt;
	}

	setTemperature(temperature: number): void {
		this.config.temperature = temperature;
	}

	setMaxTokens(maxTokens: number): void {
		this.config.maxTokens = maxTokens;
	}

	// Tools are held by the loop's registry, built per run. Update config so the
	// next run picks them up; if a run is active, patch the live registry too
	// (applies to the next turn since defs are read each turn).
	setTools(tools: Tool[]): void {
		this.config.tools = tools;
		if (this.loop) {
			const registry = this.loop.tools;
			for (const existing of registry.list()) {
				registry.unregister(existing.name);
			}
			registry.registerMany(tools);
		}
	}

	// ── Queue operations (allowed during a turn) ───────────────────────────

	// Inject guidance into the running turn (drained at the next save point).
	steer(text: string): void {
		this.steeringQueue.push(text);
		this.emitQueueChange();
	}

	// Queue a message for after the current turn completes (same drain point).
	followUp(text: string): void {
		this.followUpQueue.push(text);
		this.emitQueueChange();
	}

	// Queue a message inserted before the next user prompt. Survives abort.
	nextTurn(text: string): void {
		this.nextTurnQueue.push(text);
		this.emitQueueChange();
	}

	// Abort the running turn. Clears steering/follow-up; preserves nextTurn.
	abort(): void {
		this.abortController?.abort();
		this.steeringQueue = [];
		this.followUpQueue = [];
		this.emitQueueChange();
	}

	// ── Queue state (single source of truth for the UI) ────────────────────

	/** Snapshot of all pending queues. */
	getQueues(): HarnessQueues {
		return {
			steering: [...this.steeringQueue],
			followUp: [...this.followUpQueue],
			nextTurn: [...this.nextTurnQueue],
		};
	}

	/** Subscribe to queue changes (enqueue, drain, clear). */
	setOnQueueChange(cb: (queues: HarnessQueues) => void): void {
		this.onQueueChange = cb;
	}

	/** Clear all queues. Returns what was cleared. */
	clearQueues(): HarnessQueues {
		const cleared = this.getQueues();
		this.steeringQueue = [];
		this.followUpQueue = [];
		this.nextTurnQueue = [];
		this.emitQueueChange();
		return cleared;
	}

	private emitQueueChange(): void {
		this.onQueueChange?.(this.getQueues());
	}

	/** Controls how queued steering messages are drained. */
	setSteeringMode(mode: QueueMode): void {
		this.steeringQueueMode = mode;
	}

	/** Controls how queued follow-up messages are drained. */
	setFollowUpMode(mode: QueueMode): void {
		this.followUpQueueMode = mode;
	}

	get messages(): Message[] {
		return this.loop?.messages ?? this.history;
	}

	// Clear persisted conversation (new session / context reset).
	clearHistory(): void {
		this.history = [];
	}

	/** Compact messages using an LLM-generated summary. Returns tokens saved, or null if nothing to compact. */
	async compact(): Promise<number | null> {
		const messages = this.loop?.messages ?? this.history;
		if (!messages.length) return null;
		const before = estimateChatPayloadTokens(messages);

		// Shared compaction skeleton; the summarizer micro-truncates the older
		// block first (sending full messages to the LLM defeats the purpose),
		// then asks the LLM for a summary. On LLM failure it returns null and
		// compactMessages falls back to a local micro-compaction.
		const result = await compactMessages(messages, {
			reason: "manual",
			summarize: (older, system) =>
				this.generateSummary(microCompactMessages(older).messages, system),
		});

		if (!result.changed) return 0;
		if (this.loop) this.loop.setMessages(result.messages);
		else this.history = result.messages;
		const after = estimateChatPayloadTokens(result.messages);
		return before - after;
	}

	/** Ask LLM to summarize older messages for compaction. */
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

	// Live tool registry of the running loop, or null when idle.
	get tools(): ToolRegistry | null {
		return this.loop?.tools ?? null;
	}

	// ── Model cycling ──────────────────────────────────────────────────

	/** Get current model name. */
	getModel(): string {
		return this.loop?.getModel() ?? this.config.model;
	}

	/** Get all available models. */
	getModels(): string[] {
		return (
			this.loop?.getModels() ??
			(this.config.models
				? [this.config.model, ...this.config.models]
				: [this.config.model])
		);
	}

	/** Cycle to the next model. Returns the new model name. */
	cycleModel(direction: "forward" | "backward" = "forward"): string {
		return this.loop?.cycleModel(direction) ?? this.config.model;
	}

	// ── Thinking level ─────────────────────────────────────────────────

	/** Get current thinking/reasoning level. */
	getThinkingLevel(): string {
		return this.config.thinkingLevel ?? "medium";
	}

	/** Set thinking/reasoning level. Takes effect on the next turn. */
	setThinkingLevel(level: string): void {
		this.config.thinkingLevel = level as
			| "off"
			| "minimal"
			| "low"
			| "medium"
			| "high"
			| "xhigh";
	}

	// ── Internals ──────────────────────────────────────────────────────────

	// Provide the queue-drain handlers to the loop as `internalHooks`. The loop
	// composes built-ins → these → user hooks into a single HookBus, so we no
	// longer build a second wrapping bus or touch the user's `config.hooks`.
	// Steering drains before the next assistant response; follow-up drains only
	// when the loop would otherwise stop.
	private withDrainHook(config: AgentConfig): AgentConfig {
		const internalHooks: AgentLoopHooks = {};

		// Drain steering before each assistant response (preemptive).
		internalHooks.getSteeringMessages = async (): Promise<Message[] | undefined> => {
			let out: string[];
			if (this.steeringQueueMode === "all") {
				out = this.steeringQueue.splice(0);
			} else {
				const first = this.steeringQueue[0];
				out = first ? [first] : [];
				if (first) this.steeringQueue.shift();
			}
			if (!out.length) return undefined;
			this.emitQueueChange();
			return out.map((text) => createUserMessage(text));
		};

		// Drain steering + follow-up when the loop would stop (stacked continuation).
		// Steering has priority — user guidance overrides follow-up and todo nudges.
		internalHooks.getFollowUpMessages = async (): Promise<Message[] | undefined> => {
			const out: string[] = [];

			// Drain steering first (user guidance takes priority).
			let steeringOut: string[];
			if (this.steeringQueueMode === "all") {
				steeringOut = this.steeringQueue.splice(0);
			} else {
				const first = this.steeringQueue[0];
				steeringOut = first ? [first] : [];
				if (first) this.steeringQueue.shift();
			}
			out.push(...steeringOut);

			// Drain follow-up queue.
			let followUpOut: string[];
			if (this.followUpQueueMode === "all") {
				followUpOut = this.followUpQueue.splice(0);
			} else {
				const first = this.followUpQueue[0];
				followUpOut = first ? [first] : [];
				if (first) this.followUpQueue.shift();
			}
			out.push(...followUpOut);

			if (!out.length) return undefined;
			this.emitQueueChange();
			return out.map((text) => createUserMessage(text));
		};

		return { ...config, internalHooks };
	}

	/**
	 * Run `fn` with a timeout. Rejects with AgentError if the timeout fires.
	 */
	private async withTimeout<T>(
		fn: Promise<T>,
		timeoutMs: number,
	): Promise<T> {
		return new Promise<T>((resolve, reject) => {
			const timer = setTimeout(() => {
				reject(new AgentError({
					type: AgentErrorType.TURN_TIMEOUT,
					message: `Operation timed out after ${timeoutMs}ms`,
				}));
			}, timeoutMs);
			fn.then(
				(value) => {
					clearTimeout(timer);
					resolve(value);
				},
				(reason) => {
					clearTimeout(timer);
					reject(reason);
				},
			);
		});
	}
}
