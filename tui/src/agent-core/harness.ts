// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above AgentLoop. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn (never mutating an in-flight
// provider request), and steering / follow-up / nextTurn queues drained at
// save points. Mirrors pi's AgentHarness (packages/agent/docs/agent-harness.md),
// scoped to what tui's loop exposes.
//
// The loop already exposes a save point: the `prepareNextTurn` contract hook
// fires after each turn and can rewrite the working messages. The harness
// installs a drain hook there to inject queued messages and apply config
// changes between turns.

import { get_reasoner, getReasonerMeta } from "../reasoners/registry.ts";
import { withTimeout } from "./async-utils.ts";
import { runSessionStartHooks, runHookEvent } from "./plugins.ts";
import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "./file-checkpoints.ts";
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
import {
	compact,
	estimateContextTokens,
	shouldCompact,
	type CompactionSettings,
	DEFAULT_COMPACTION_SETTINGS,
} from "./compaction/index.ts";
import type { ToolRegistry } from "./tools/registry.ts";
import type {
	AgentConfig,
	AgentLoopHooks,
	Message,
	QueueMode,
	Tool,
} from "./types.ts";

// Explicit harness phases (mirrors pi's AgentHarnessPhase). Operations are
// gated on phase: structural ops (prompt, compact) require "idle"; steering
// requires an active "turn". "compaction" and "branch_summary" are transient
// phases held while those background operations run, blocking concurrent
// structural ops.
export type HarnessPhase = "idle" | "turn" | "compaction" | "branch_summary";

export class HarnessBusyError extends Error {
	constructor(op: string, phase: HarnessPhase, required: HarnessPhase) {
		super(
			`AgentHarness cannot ${op}: phase is "${phase}", requires "${required}"`,
		);
		this.name = "HarnessBusyError";
	}
}

// Legal phase transitions. Every working phase is entered only from "idle" and
// exits only back to "idle" — the harness runs one structural operation at a
// time. Encoding this as a table (rather than scattered `if (phase !== ...)`
// checks) makes the state machine the single source of truth and lets every
// operation share one guarded `transition` helper.
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

/** Snapshot of the three harness queues, for UI display. */
export interface HarnessQueues {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}

/**
 * A forked conversation branch. `parent` is the history at fork time (restored
 * on merge/discard); `forkedAt` marks how many messages were shared with the
 * parent so branchSummary() can summarize only the diverged tail.
 */
interface Branch {
	id: string;
	parent: Message[];
	forkedAt: number;
}

/** Public branch info for UI / callers. */
export interface BranchInfo {
	id: string;
	depth: number;
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
	// retain context. When a branch is active this holds the branch's messages;
	// the parent is preserved on the branch stack until merged or discarded.
	private history: Message[] = [];

	// Conversation branches. fork() snapshots the current history into a parent
	// frame and starts a fresh branch that shares the parent's messages; turns
	// run on the branch. branchSummary() collapses the branch's diverged tail
	// into a single summary message merged back into the parent (pi semantics).
	// Single-level by design — nesting can stack frames later if needed.
	private branches: Branch[] = [];
	private branchSeq = 0;

	// Conversation checkpoints: a snapshot of history is pushed before each
	// prompt so a bad turn can be rewound. Bounded ring (newest last).
	private checkpoints: Message[][] = [];
	private static readonly MAX_CHECKPOINTS = 20;

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
	// Fired on every phase transition so the UI / bridge can surface the harness
	// state (idle / turn / compaction / branch_summary) — not just the loop's
	// finer-grained activity sub-states.
	private onPhaseChange?: (phase: HarnessPhase, prev: HarnessPhase) => void;
	// Fired when the harness goes idle after a prompt(), carrying the count of
	// pending nextTurn messages. A non-zero count means the caller can auto-
	// trigger another prompt() to drain them (Pi-style autonomous continuation).
	private onSettled?: (nextTurnCount: number) => void;
	// Fired after every turn_end (each save point). Pi analogue: lets the UI
	// show autosave status and know a safe rewind point exists.
	private onSavePoint?: () => void;
	// Fired when compaction completes (auto or manual).
	private onCompaction?: (reason: "auto" | "manual", tokensBefore: number, tokensAfter: number) => void;
	// Auto-compaction configuration (merged with defaults).
	private autoCompactionSettings: CompactionSettings = DEFAULT_COMPACTION_SETTINGS;
	// Plugin hooks enabled flag — mirrors the loop's hooksEnabled pattern.
	private _hooksEnabled = true;
	// Session state for plugin lifecycle payloads.
	private _sessionId?: string;
	private _transcriptPath?: string;
	private _hasStartedSession = false;
	// Resolved when the current prompt() completes; undefined when idle.
	// waitForIdle() awaits this so callers don't need to poll this.running.
	private _runPromise?: Promise<void>;
	private _runResolve?: () => void;
	// Called once before each prompt() run, after nextTurn drain, before the
	// loop starts. Returns optional extra messages to prepend to the context
	// and/or a system-prompt override for this run only.
	private _beforeAgentStart?: (
		promptText: string,
	) => Promise<{ messages?: Message[]; systemPrompt?: string } | undefined> | { messages?: Message[]; systemPrompt?: string } | undefined;

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

	/** Subscribe to phase transitions (idle ↔ turn / compaction / branch_summary). */
	setOnPhaseChange(
		cb: (phase: HarnessPhase, prev: HarnessPhase) => void,
	): void {
		this.onPhaseChange = cb;
	}

	/**
	 * Subscribe to the settled event. Fired when the harness returns to idle
	 * after a prompt(), with the count of pending nextTurn messages. When
	 * nextTurnCount > 0 the caller can auto-trigger another prompt() to drain
	 * them without user input (Pi-style autonomous continuation).
	 */
	setOnSettled(cb: (nextTurnCount: number) => void): void {
		this.onSettled = cb;
	}

	/**
	 * Subscribe to save-point events. Fired after every turn completes (each
	 * point where the conversation is safely persisted). Use to update autosave
	 * indicators or enable rewind UI.
	 */
	setOnSavePoint(cb: () => void): void {
		this.onSavePoint = cb;
	}

	/**
	 * Resolves when the harness is next idle. If already idle, resolves
	 * immediately. Use before structural ops (compact, setHistory, fork) when
	 * the caller cannot guarantee the harness is idle.
	 */
	async waitForIdle(): Promise<void> {
		if (this._phase === "idle") return;
		await this._runPromise;
	}

	/**
	 * Register a hook called once before each prompt() run. Receives the
	 * prompt text; may return extra messages to prepend to the context and/or
	 * a system-prompt override scoped to this run only. Returning undefined
	 * leaves both unchanged.
	 */
	setBeforeAgentStart(
		cb: (
			promptText: string,
		) => Promise<{ messages?: Message[]; systemPrompt?: string } | undefined> | { messages?: Message[]; systemPrompt?: string } | undefined,
	): void {
		this._beforeAgentStart = cb;
	}

	// Guarded phase transition. Throws HarnessBusyError if `to` is not reachable
	// from the current phase, otherwise applies it and notifies subscribers. The
	// single mutation point for `_phase` — all ops go through here.
	private transition(to: HarnessPhase, op: string): void {
		if (!PHASE_TRANSITIONS[this._phase].includes(to)) {
			// The required phase for any working op is "idle"; surface that.
			throw new HarnessBusyError(op, this._phase, "idle");
		}
		const prev = this._phase;
		this._phase = to;
		if (prev !== to) this.onPhaseChange?.(to, prev);
	}

	// Run `fn` inside a working phase: transition into it (gated), always return
	// to idle afterwards. Shared by prompt / compact / branchSummary so the
	// enter-guard and the idle-restore live in one place.
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

	// Guard for instantaneous idle-only operations (fork / discardBranch) that
	// complete synchronously and so never hold a working phase.
	private assertIdle(op: string): void {
		if (this._phase !== "idle")
			throw new HarnessBusyError(op, this._phase, "idle");
	}

	// ── Structural operation: prompt ───────────────────────────────────────
	// Rejected while busy. Drains nextTurn messages before the user prompt.
	async prompt(userMessage: string): Promise<Message[]> {
		// Start the run promise so waitForIdle() can await turn completion.
		this._runPromise = new Promise<void>((resolve) => {
			this._runResolve = resolve;
		});

		// Pi-style auto-compaction: run BEFORE the turn phase so it can
		// enter its own "compaction" phase. Must fire before entering the
		// turn phase because phase transitions only allow idle → working.
		if (this.autoCompactionSettings.enabled) {
			const compacted = await this.runAutoCompaction("auto");
			if (compacted) {
				// Context was compacted; retry the whole prompt with the new history.
				return this.prompt(userMessage);
			}
		}

		return this.runInPhase("turn", "prompt", async () => {
			this.abortController = new AbortController();

			// Fire SessionStart on first prompt (session init).
			if (!this._hasStartedSession) {
				await this.emitSessionStart("startup");
			}

			// Checkpoint the pre-turn conversation + open a file frame so rewind()
			// restores both the messages and the files this turn writes.
			this.checkpoints.push([...this.history]);
			if (this.checkpoints.length > AgentHarness.MAX_CHECKPOINTS) {
				this.checkpoints.shift();
			}
			beginFileFrame();

			// nextTurn messages are drained inside the loop via the transformContext
			// hook (see withDrainHook), so they enter the context as real user
			// messages rather than being concatenated onto the prompt string.
			const promptText = userMessage;

			// before_agent_start: let callers inject extra context or override the
			// system prompt for this run only (Pi analogue).
			const beforeStart = await this._beforeAgentStart?.(promptText);

			const preReasoning = await this.runPreReasoner(promptText);

			// Continuation history for this prompt: prior conversation plus, when a
			// reasoner ran, its output as a synthetic assistant message so the loop
			// sees the reasoning. run() rebuilds _messages from initialMessages, so
			// this must flow through initialMessages — not a post-construction
			// setMessages, which run() would discard.
			let continuation: Message[] = preReasoning
				? [...this.history, createAssistantMessage(preReasoning)]
				: this.history;

			// Prepend any extra messages injected by beforeAgentStart.
			if (beforeStart?.messages?.length) {
				continuation = [...beforeStart.messages, ...continuation];
			}

			// System-prompt override scoped to this run: patch the config clone
			// before withDrainHook so the loop picks it up.
			const runConfig = beforeStart?.systemPrompt
				? { ...this.config, systemPrompt: beforeStart.systemPrompt }
				: this.config;

			try {
				// Build or update the loop config (new on config change).
				this.loopConfig = this.withDrainHook(runConfig);
				if (!this.loop) {
					this.loop = new AgentLoop({
						config: this.loopConfig,
						backend: this.backend,
						cwd: this.cwd,
						maxIterations: this.maxIterations,
						signal: this.abortController.signal,
						initialMessages: continuation.length ? continuation : undefined,
					});
				} else {
					// Reuse existing loop: refresh its config (system prompt,
					// temperature, tools, freshly-built internalHooks), the abort signal,
					// and the conversation to continue from. Without updateConfig the
					// loop would keep its construction-time config; without
					// updateInitialMessages it would replay only the first prompt's
					// history and drop every turn since.
					this.loop.updateConfig(this.loopConfig);
					this.loop.updateSignal(this.abortController.signal);
					this.loop.updateInitialMessages(
						continuation.length ? continuation : undefined,
					);
				}
				const loop = this.loop;

				const result = await loop.run(promptText);
				// Persist the full conversation for the next prompt.
				this.history = result;
				return result;
			} finally {
				this.abortController = null;
				// Steering/follow-up are turn-scoped; clear leftovers.
				this.steeringQueue = [];
				this.followUpQueue = [];
				this.emitQueueChange();
				// Resolve waitForIdle() waiters before settled fires so callers
				// that await idle inside the settled handler don't deadlock.
				this._runResolve?.();
				this._runPromise = undefined;
				this._runResolve = undefined;
				// Notify subscribers that the harness has settled. nextTurnQueue
				// survives abort and is the signal for autonomous continuation.
				this.onSettled?.(this.nextTurnQueue.length);
			}
		});
	}

	// Optional reasoner pre-phase: run structured reasoning before ReAct. A
	// timeout prevents a slow reasoner from blocking the whole prompt; any
	// failure falls back to plain ReAct (returns undefined).
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
	// Only meaningful during a turn — steering an idle harness has nothing to
	// drain into.
	steer(text: string): void {
		if (this._phase !== "turn")
			throw new HarnessBusyError("steer", this._phase, "turn");
		this.steeringQueue.push(text);
		this.emitQueueChange();
		// Interrupt mode: cut the in-flight stream (partial output is kept) so
		// the steering message takes effect immediately instead of waiting for
		// the current LLM call to finish.
		if (this.config.steeringInterrupt) {
			this.loop?.interruptTurn();
		}
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

	// Abort the running turn. Fires SessionEnd plugin event.
	abort(): void {
		this.abortController?.abort();
		this.steeringQueue = [];
		this.followUpQueue = [];
		this.emitQueueChange();
		this.emitSessionEnd("abort").catch(() => {});
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

	// ── Plugin lifecycle hooks ─────────────────────────────────────────────

	/** Enable or disable plugin hooks entirely. Also propagates to the loop config. */
	setHooksEnabled(enabled: boolean): void {
		this._hooksEnabled = enabled;
		// Propagate to loop config so the loop's hooks also see it.
		this.config.runtimeHooksEnabled = enabled;
	}

	/** Set the session ID for plugin event payloads. Also propagates to the loop config. */
	setSessionId(id: string): void {
		this._sessionId = id;
		// Propagate to loop config so the loop's hooks also see it.
		this.config.hookSessionId = id;
	}

	/** Set the transcript path for plugin event payloads. Also propagates to the loop config. */
	setTranscriptPath(path: string): void {
		this._transcriptPath = path;
		// Propagate to loop config so the loop's hooks also see it.
		this.config.hookTranscriptPath = path;
	}

	/** Fire SessionStart plugin event (called on first prompt / after /clear). */
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
			// SessionStart failure must not block the session.
		}
	}

	/** Fire SessionEnd plugin event (called on clear/abort/quit). */
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
			// SessionEnd failure must not block cleanup.
		}
	}

	/** Fire PreCompact / PostCompact plugin events around compaction. */
	private async emitPreCompact(): Promise<void> {
		if (!this._hooksEnabled) return;
		try {
			await runHookEvent("PreCompact", {
				session_id: this._sessionId || "",
				transcript_path: this._transcriptPath || "",
				cwd: this.cwd || process.cwd(),
			});
		} catch {
			// PreCompact failure must not block compaction.
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
			// PostCompact failure must not block compaction.
		}
	}

	// ── Auto-compaction ────────────────────────────────────────────────────

	/** Configure auto-compaction settings. Takes effect on the next prompt(). */
	setAutoCompactionSettings(settings: Partial<CompactionSettings>): void {
		this.autoCompactionSettings = { ...this.autoCompactionSettings, ...settings };
	}

	/** Enable or disable auto-compaction. */
	enableAutoCompaction(enabled: boolean): void {
		this.autoCompactionSettings.enabled = enabled;
	}

	/** Subscribe to compaction events. */
	setOnCompaction(cb: (reason: "auto" | "manual", tokensBefore: number, tokensAfter: number) => void): void {
		this.onCompaction = cb;
	}

	get messages(): Message[] {
		return this.loop?.messages ?? this.history;
	}

	// Clear persisted conversation (new session / context reset). Fires SessionEnd then SessionStart.
	clearHistory(): void {
		// SessionEnd before clearing (session reset).
		this.emitSessionEnd("reset").catch(() => {});
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.setActiveHistory([]);
		// SessionStart after clearing (fresh session).
		this.emitSessionStart("clear").catch(() => {});
		this._hasStartedSession = false;
	}

	/**
	 * Replace the persisted conversation with externally restored messages
	 * (session resume / switch), so the model continues with the restored
	 * context rather than starting cold. Idle-only. Drops open branches (their
	 * parent snapshots no longer apply) and any system messages — the loop
	 * re-injects the current system prompt each run.
	 * Fires SessionEnd then SessionStart.
	 */
	setHistory(messages: Message[]): void {
		this.assertIdle("setHistory");
		// SessionEnd before switching (old session).
		this.emitSessionEnd("switch").catch(() => {});
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.setActiveHistory(messages.filter((m) => m.role !== "system"));
		// SessionStart after switching (new session context).
		this.emitSessionStart("resume").catch(() => {});
		this._hasStartedSession = false;
	}

	/**
	 * Restore the conversation to the snapshot taken before the most recent
	 * prompt (undo the last turn). Idle-only. Returns the restored message
	 * count, or null when no checkpoint exists.
	 */
	rewind(): { messages: number; filesRestored: number } | null {
		this.assertIdle("rewind");
		const snapshot = this.checkpoints.pop();
		if (!snapshot) return null;
		this.branches = [];
		this.setActiveHistory(snapshot);
		// Restore files the rewound turn wrote (write tools only; bash is not
		// captured — see file-checkpoints.ts).
		const filesRestored = restoreFileFrame() ?? 0;
		return { messages: snapshot.length, filesRestored };
	}

	// ── Conversation branching ─────────────────────────────────────────────
	// fork() snapshots the live conversation and starts a branch that shares the
	// parent's messages; subsequent turns run on the branch. branchSummary()
	// summarizes the branch's diverged tail and merges it back into the parent.

	/**
	 * Fork the current conversation into a new branch. The branch starts as a
	 * copy of the live history; turns run on it until merged (branchSummary) or
	 * discarded (discardBranch). Idle-only. Returns the new branch id.
	 */
	fork(): string {
		this.assertIdle("fork");
		const current = this.activeHistory();
		const branch: Branch = {
			id: `branch_${++this.branchSeq}`,
			parent: current,
			forkedAt: current.length,
		};
		this.branches.push(branch);
		// Branch begins as a copy of the parent so divergence is detectable.
		this.setActiveHistory([...current]);
		return branch.id;
	}

	/**
	 * Collapse the active branch's diverged tail into a single summary message
	 * and merge it back into the parent, then make the parent active again.
	 * Idle-only; holds the "branch_summary" phase while the LLM summarizes.
	 * Returns the summary text, or null if there was nothing to summarize.
	 */
	async branchSummary(): Promise<string | null> {
		this.assertIdle("branchSummary");
		const branch = this.branches.at(-1);
		if (!branch) return null;

		const current = this.activeHistory();
		const diverged = current.slice(branch.forkedAt);
		if (!diverged.length) {
			// Nothing explored on the branch; just restore the parent.
			this.branches.pop();
			this.setActiveHistory(branch.parent);
			return null;
		}

		return this.runInPhase("branch_summary", "branchSummary", async () => {
			const summary =
				(await this.generateSummary(
					microCompactMessages(diverged).messages,
					[],
				)) ??
				// LLM failure: fall back to a terse local note so the merge still
				// records that a branch happened.
				`[branch ${branch.id}: ${diverged.length} messages explored]`;

			this.branches.pop();
			this.setActiveHistory([
				...branch.parent,
				createAssistantMessage(`Branch summary: ${summary}`),
			]);
			return summary;
		});
	}

	/**
	 * Discard the active branch without merging; restore the parent. Idle-only.
	 * Returns true if a branch was discarded.
	 */
	discardBranch(): boolean {
		this.assertIdle("discardBranch");
		const branch = this.branches.pop();
		if (!branch) return false;
		this.setActiveHistory(branch.parent);
		return true;
	}

	/** Stack of active branches (innermost last). */
	listBranches(): BranchInfo[] {
		return this.branches.map((b, i) => ({ id: b.id, depth: i + 1 }));
	}

	// Live conversation: the loop's messages while running, else harness history.
	private activeHistory(): Message[] {
		return this.loop?.messages ?? this.history;
	}

	// Set the live conversation on both the harness and (if present) the loop,
	// keeping the two in sync after a branch switch.
	private setActiveHistory(messages: Message[]): void {
		this.history = messages;
		this.loop?.setMessages(messages);
	}

	/** Compact messages using an LLM-generated summary. Returns tokens saved, or null if nothing to compact. */
	async compact(): Promise<number | null> {
		this.assertIdle("compact");
		const messages = this.loop?.messages ?? this.history;
		if (!messages.length) return null;
		const before = estimateChatPayloadTokens(messages);

		return this.runInPhase("compaction", "compact", async () => {
			// Fire PreCompact before compaction.
			await this.emitPreCompact();
			// Shared compaction skeleton; the summarizer micro-truncates the older
			// block first (sending full messages to the LLM defeats the purpose),
			// then asks the LLM for a summary. On LLM failure it returns null and
			// compactMessages falls back to a local micro-compaction.
			const result = await compactMessages(messages, {
				reason: "manual",
				summarize: (older, system) =>
					this.generateSummary(microCompactMessages(older).messages, system),
			});

			if (!result.changed) {
				await this.emitPostCompact();
				return 0;
			}
			if (this.loop) this.loop.setMessages(result.messages);
			else this.history = result.messages;
			const after = estimateChatPayloadTokens(result.messages);
			this.onCompaction?.("manual", before, after);
			await this.emitPostCompact();
			return before - after;
		});
	}

	/**
	 * Pi-style auto-compaction: compact using the compaction module when
	 * context exceeds the configured threshold. Integrates file operation
	 * tracking into the summary output.
	 */
	private async runAutoCompaction(reason: "auto" | "manual"): Promise<boolean> {
		const messages = this.loop?.messages ?? this.history;
		if (!messages.length || !this.autoCompactionSettings.enabled) return false;

		return this.runInPhase("compaction", "autoCompact", async () => {
			// Fire PreCompact before auto-compaction.
			await this.emitPreCompact();

			const agentMessages = messages as unknown as Array<{ role: string; content?: unknown[] }>;

			// Check if compaction is needed
			if (!shouldCompact(agentMessages, this.autoCompactionSettings)) {
				await this.emitPostCompact();
				return false;
			}

			const before = estimateContextTokens(agentMessages).tokens;

			// Use Pi-style compaction with file operation tracking,
			// passing our LLM-backed summarizer for real summaries.
			const compactionResult = await compact(
				agentMessages,
				this.autoCompactionSettings,
				undefined, // prevSummary
				undefined, // customInstructions
				// Adapt harness's generateSummary to the SummaryFn signature.
				// We need to extract system messages from history for the summarizer.
				async (compactionMessages) => {
					// Convert CompactableMessage[] to Message[] for the harness summarizer
					const messages = compactionMessages as unknown as Message[];
					// System messages are already in history (at the start),
					// so we pass an empty array — the harness summarizer
					// receives the conversation without the system prompt.
					const summary = await this.generateSummary(messages, []);
					return summary || `[Auto-compaction summary failed — context preserved but no summary generated]`;
				},
			);

			// Build new history: summary + kept messages converted back to Message[]
			const summaryMessage = { role: "system" as const, content: `\n<compaction_summary>${compactionResult.summary}</compaction_summary>\n` };

			const keptMessages: Message[] = compactionResult.messagesToKeep as unknown as Message[];
			const newHistory = [summaryMessage, ...keptMessages.filter(m => m.content)];
			this.history = newHistory;
			if (this.loop) this.loop.setMessages(newHistory);

			const after = estimateChatPayloadTokens(newHistory);
			this.onCompaction?.(reason, before, after);
			await this.emitPostCompact();
			return true;
		});
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
	//
	// Phase invariant: these drains run *only* inside loop.run(), which executes
	// inside the harness "turn" phase (see prompt()/runInPhase). The phase state
	// machine forbids compaction / branch_summary while a turn is active and
	// vice-versa (PHASE_TRANSITIONS: every working phase ↔ idle), so a steering
	// drain can never interleave with a compaction or branch-summary rewrite of
	// the same message list. steer() itself is gated to the "turn" phase, so the
	// queue is only ever fed while a turn owns the conversation.
	private withDrainHook(config: AgentConfig): AgentConfig {
		const internalHooks: AgentLoopHooks = {};

		// Drain nextTurn messages into the context before the first LLM call.
		// They enter as real user messages inserted just before the user's
		// prompt (the trailing message), preserving "before the next user
		// prompt" semantics. The splice empties the queue, so subsequent LLM
		// calls in the same turn see nothing to drain.
		internalHooks.transformContext = ({ messages }) => {
			const pending = this.nextTurnQueue.splice(0);
			if (!pending.length) return undefined;
			this.emitQueueChange();
			const injected = pending.map((text) => createUserMessage(text));
			// Insert before the trailing message (the user prompt); if there is
			// no trailing message, append.
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

		// Drain steering before each assistant response (preemptive).
		internalHooks.getSteeringMessages = async (): Promise<
			Message[] | undefined
		> => {
			const out = this.drainQueue(this.steeringQueue, this.steeringQueueMode);
			return this.toInjectedMessages(out);
		};

		// Drain steering + follow-up when the loop would stop (stacked continuation).
		// Steering has priority — user guidance overrides follow-up and todo nudges.
		internalHooks.getFollowUpMessages = async (): Promise<
			Message[] | undefined
		> => {
			const out = [
				...this.drainQueue(this.steeringQueue, this.steeringQueueMode),
				...this.drainQueue(this.followUpQueue, this.followUpQueueMode),
			];
			return this.toInjectedMessages(out);
		};

		// Wrap onEvent to emit save_point after every completed turn. The
		// original handler (set by the bridge) is called first so events reach
		// the UI before the save_point notification fires.
		const originalOnEvent = config.onEvent;
		const wrappedOnEvent = this.onSavePoint
			? (event: import("./types.ts").AgentEvent) => {
					originalOnEvent?.(event);
					if (event.type === "turn_end") this.onSavePoint?.();
			  }
			: originalOnEvent;

		return { ...config, internalHooks, onEvent: wrappedOnEvent };
	}

	// Drain a queue per its mode: "all" empties it, "one-at-a-time" takes the
	// oldest. Mutates the queue in place. Shared by all three drain points.
	private drainQueue(queue: string[], mode: QueueMode): string[] {
		if (mode === "all") return queue.splice(0);
		const first = queue.shift();
		return first ? [first] : [];
	}

	// Convert drained queue text into user messages, emitting a queue change
	// when anything was drained. Returns undefined for an empty drain so the
	// hook reports "no messages".
	private toInjectedMessages(texts: string[]): Message[] | undefined {
		if (!texts.length) return undefined;
		this.emitQueueChange();
		return texts.map((text) => createUserMessage(text));
	}
}
