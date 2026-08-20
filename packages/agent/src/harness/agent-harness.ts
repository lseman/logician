// ── AgentHarness ──────────────────────────────────────────────────────────
// Multi-lane, durable-session-log-backed orchestration surface. Ported from
// pi coding agent's harness/agent-harness.ts.
//
// Upstream pi's own AgentHarness is, as of this port, itself mostly
// HarnessNotImplemented stubs — this is pi's in-progress next-generation
// design (lanes, crash recovery via the reducer, deferred/suspended
// operations), not a finished reference implementation. This file ports the
// full type surface (AgentLane, RunOutcome/CompactionOutcome/NavigationOutcome,
// tagged errors, options) faithfully since that surface is the actual forward
// design worth adopting, but leaves the operational methods unimplemented
// (unavailable(), matching upstream) rather than inventing behavior pi
// hasn't specified yet. A separate, already-working orchestrator
// (agent/agent.ts, ported last) covers single-lane prompt/continue today.

import { runAgentLoop } from "../agent/agent-loop.ts";
import { getDefaultStreamFn } from "../agent/stream-fn.ts";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	QueueMode,
	StreamFn,
	ThinkingLevel,
} from "../agent/types.ts";
import type {
	Api,
	AssistantMessage,
	ImageContent,
	JsonValue,
	Message,
	Model,
	Usage,
} from "../ai/types.ts";
import { err, ok } from "../core/result.ts";
import type { RetryPolicy } from "../core/retry.ts";
import { uuidv7 } from "../core/uuid.ts";
import { generateBranchSummary } from "./compaction/branch-summarization.ts";
import {
	type CompactionSettings,
	DEFAULT_COMPACTION_SETTINGS,
	prepareCompaction,
	compact as runCompactionSummary,
} from "./compaction/compaction.ts";
import { HarnessEventBus } from "./events.ts";
import { HookRegistry } from "./hooks.ts";
import { convertToLlm as harnessConvertToLlm } from "./messages.ts";
import { type Result as ResultValue, TaggedError } from "./result.ts";
import type {
	BranchSummaryEntry,
	CompactionEntry,
	Entry,
	MessageEntry,
	OperationStartedRecord,
	ProvisionedEntry,
	Session,
	SessionTree,
} from "./session/index.ts";
import type { TelemetryContext } from "./telemetry.ts";
import type { AgentHarnessResources, PromptTemplate, Skill } from "./types.ts";

export class LaneBusy extends TaggedError("LaneBusy")<{
	lane: string;
	operationId: string;
	operationKind: "run" | "compaction" | "navigation";
	message: string;
}> {}
export class MissingIdentities extends TaggedError("MissingIdentities")<{
	lane: string;
	tools: string[];
	models: string[];
	message: string;
}> {}
export class NoActiveRun extends TaggedError("NoActiveRun")<{
	lane: string;
	message: string;
}> {}
export class NoActiveOperation extends TaggedError("NoActiveOperation")<{
	lane: string;
	message: string;
}> {}
export class NothingToResume extends TaggedError("NothingToResume")<{
	lane: string;
	message: string;
}> {}
export class InvalidMessage extends TaggedError("InvalidMessage")<{
	lane: string;
	reason: string;
	message: string;
}> {}
export class UnknownSkill extends TaggedError("UnknownSkill")<{
	name: string;
	message: string;
}> {}
export class UnknownTemplate extends TaggedError("UnknownTemplate")<{
	name: string;
	message: string;
}> {}
export class UnknownTarget extends TaggedError("UnknownTarget")<{
	targetId: string;
	message: string;
}> {}
export class UnknownQueueItem extends TaggedError("UnknownQueueItem")<{
	lane: string;
	entryId: string;
	message: string;
}> {}
export class LaneExists extends TaggedError("LaneExists")<{
	lane: string;
	message: string;
}> {}
export class InvalidLane extends TaggedError("InvalidLane")<{
	lane: string;
	reason: string;
	message: string;
}> {}
export class NothingToCompact extends TaggedError("NothingToCompact")<{
	lane: string;
	message: string;
}> {}
export class Closed extends TaggedError("Closed")<{ message: string }> {}

export class HarnessFault extends Error {
	readonly cause: unknown;

	constructor(message: string, cause: unknown) {
		super(message);
		this.name = "HarnessFault";
		this.cause = cause;
	}
}

export class HarnessClosed extends Error {
	constructor() {
		super("AgentHarness was closed while the operation was active");
		this.name = "HarnessClosed";
	}
}

export class HarnessNotImplemented extends Error {
	readonly operation: string;

	constructor(operation: string) {
		super(`AgentHarness.${operation} is not implemented yet`);
		this.name = "HarnessNotImplemented";
		this.operation = operation;
	}
}

export interface OperationError {
	code: string;
	message: string;
}

export type RunOutcome =
	| {
			kind: "completed";
			leafId: string;
			finalEntryId: string;
			finalMessage: import("../ai/types.ts").AssistantMessage;
	  }
	| {
			kind: "aborted";
			leafId: string;
			finalEntryId: string;
			finalMessage: import("../ai/types.ts").AssistantMessage;
	  }
	| {
			kind: "failed";
			leafId: string;
			error: OperationError;
			finalEntryId?: string;
			finalMessage?: import("../ai/types.ts").AssistantMessage;
	  }
	| {
			kind: "suspended";
			leafId: string;
			finalEntryId: string;
			deferred: import("../ai/types.ts").DeferredHandle;
	  };

export type CompactionOutcome =
	| { kind: "completed"; leafId: string; entry: CompactionEntry }
	| { kind: "declined" | "aborted"; leafId: string }
	| { kind: "failed"; leafId: string; error: OperationError };

export type NavigationOutcome =
	| {
			kind: "completed";
			newLeafId: string | null;
			summaryEntry?: BranchSummaryEntry;
	  }
	| { kind: "declined" | "aborted"; leafId: string | null }
	| { kind: "failed"; leafId: string | null; error: OperationError };

export type RunRejected =
	| LaneBusy
	| InvalidMessage
	| UnknownSkill
	| UnknownTemplate
	| Closed;
export type CompactionRejected = LaneBusy | NothingToCompact | Closed;
export type NavigationRejected = LaneBusy | UnknownTarget | Closed;
export type ResumeRejected =
	| LaneBusy
	| NothingToResume
	| MissingIdentities
	| Closed;
export type QueueRejected = NoActiveRun | InvalidMessage | Closed;
export type CancelQueuedRejected = UnknownQueueItem | Closed;
export type AbortRejected = NoActiveOperation | Closed;

export type RunResult = ResultValue<
	{ runId: string } & RunOutcome,
	RunRejected
>;
export type CompactionResult = ResultValue<
	{ runId: string } & CompactionOutcome,
	CompactionRejected
>;
export type NavigationResult = ResultValue<
	{ runId: string } & NavigationOutcome,
	NavigationRejected
>;
export type QueueResult = ResultValue<{ entryId: string }, QueueRejected>;
export type CancelQueuedResult = ResultValue<
	{ outcome: "cancelled" | "already_consumed" | "already_cleared" },
	CancelQueuedRejected
>;
export type RecordUsageResult = ResultValue<void, Closed>;
export type AbortResult = ResultValue<
	{ runId: string; steer: AgentMessage[]; followUp: AgentMessage[] },
	AbortRejected
>;

export type ResumeOutcome =
	| ({ operation: "run"; runId: string } & RunOutcome)
	| ({ operation: "compaction"; runId: string } & CompactionOutcome)
	| ({ operation: "navigation"; runId: string } & NavigationOutcome);
export type ResumeResult = ResultValue<ResumeOutcome, ResumeRejected>;
export type CreateLaneResult = ResultValue<
	AgentLane,
	LaneExists | InvalidLane | UnknownTarget | Closed
>;

export interface NavigateOptions {
	summarize?: boolean;
	customInstructions?: string;
	label?: string;
}

export interface SuspendedOperation {
	lane: string;
	kind: "run" | "compaction" | "navigation";
	id: string;
	startedAt: number;
	reason: "crash" | "deferred";
	prompt?: AgentMessage[];
	deferred?: import("../ai/types.ts").DeferredHandle;
	aborting?: { steer: AgentMessage[]; followUp: AgentMessage[] };
	missing: { tools: string[]; models: string[] };
}

export interface LaneInfo {
	name: string;
	leafId: string | null;
	operation: null | {
		id: string;
		kind: "run" | "compaction" | "navigation";
		status: "running" | "suspended" | "aborting";
	};
}

export interface QueuedItem {
	entryId: string;
	message: AgentMessage;
}

export interface LaneSnapshot {
	lane: string;
	transcript: Entry[];
	leafId: string | null;
	operation: LaneInfo["operation"];
	queues: {
		steer: QueuedItem[];
		followUp: QueuedItem[];
		nextRun: QueuedItem[];
	};
	pendingWrites: { id: string; entry: ProvisionedEntry }[];
	faulted: boolean;
}

export interface SessionSnapshot {
	lanes: (LaneInfo & { suspended?: SuspendedOperation })[];
	faulted: boolean;
}

export type ActionInfo =
	| { kind: "append_entry"; entryType: Entry["type"]; entryId: string }
	| { kind: "append_record"; recordType: string }
	| { kind: "move_lane"; to: string | null }
	| { kind: "set_fact"; fact: "name" | "label" }
	| { kind: "try_finish_run"; outcome: "completed" | "failed" }
	| {
			kind: "finish_operation";
			outcome: "completed" | "declined" | "failed" | "aborted";
	  }
	| { kind: "commit_follow_up" }
	| { kind: "consume_queue_item"; queue: "steer" | "followUp"; entryId: string }
	| { kind: "apply_pending_write"; entryId: string }
	| {
			kind: "stream_assistant";
			step: "assistant" | "compaction" | "branch_summary";
			attempt: number;
	  }
	| { kind: "execute_tool"; toolCallId: string; toolName: string }
	| { kind: "fetch_deferred" | "cancel_deferred"; provider: string; id: string }
	| { kind: "hook"; name: HookName }
	| { kind: "sleep"; delayMs: number };

export type HookName =
	| "before_run"
	| "before_resume"
	| "before_run_end"
	| "transform_context"
	| "before_request"
	| "before_payload"
	| "after_response"
	| "before_tool"
	| "after_tool"
	| "before_compaction"
	| "before_navigation";

export interface Hooks {
	on(
		name: HookName,
		handler: (event: unknown) => unknown | Promise<unknown>,
		options?: { id?: string },
	): () => void;
}

export interface Events {
	on(
		type: string,
		listener: (event: unknown) => void | Promise<void>,
	): () => void;
}

/**
 * Deep-clone a value, omitting any object key whose value is `undefined`. Tool results and
 * compaction summaries legitimately carry `details: undefined` / `usage: undefined` when they
 * have no structured payload; assertJsonSerializable (session/session.ts) rejects `undefined`
 * anywhere in a durable payload, so anything headed for appendEntry must be sanitized first.
 */
function stripUndefinedDeep<T>(value: T): T {
	if (Array.isArray(value)) {
		return value.map(item => stripUndefinedDeep(item)) as unknown as T;
	}
	if (value !== null && typeof value === "object") {
		const result: Record<string, unknown> = {};
		for (const [key, val] of Object.entries(value)) {
			if (val === undefined) continue;
			result[key] = stripUndefinedDeep(val);
		}
		return result as T;
	}
	return value;
}

/**
 * In-memory mirror of a lane's steer/followUp/nextRun queue. Durability lives in the session
 * log (queue_enqueued/queue_cancelled records, per QueueEnqueuedRecord in session/types.ts) —
 * this class only tracks live-process drain order and entry-id lookups for cancelQueued.
 */
class PendingQueue {
	private items: QueuedItem[] = [];
	mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	enqueue(item: QueuedItem): void {
		this.items.push(item);
	}

	hasItems(): boolean {
		return this.items.length > 0;
	}

	peekAll(): QueuedItem[] {
		return [...this.items];
	}

	drain(): QueuedItem[] {
		if (this.mode === "all") {
			const drained = this.items.slice();
			this.items = [];
			return drained;
		}
		const first = this.items[0];
		if (!first) return [];
		this.items = this.items.slice(1);
		return [first];
	}

	/** Remove one queued item by entry id. Returns true if it was present. */
	remove(entryId: string): boolean {
		const index = this.items.findIndex(item => item.entryId === entryId);
		if (index === -1) return false;
		this.items.splice(index, 1);
		return true;
	}

	clear(): void {
		this.items = [];
	}
}

export type HarnessTool = AgentTool & { replay?: "never" | "safe" };
export type Resources = AgentHarnessResources<Skill, PromptTemplate>;
export type StreamOptions = import("../ai/types.ts").SimpleStreamOptions;
export type StreamOptionsPatch = Partial<StreamOptions>;
export type EntryProjector = (
	entry: Entry,
) => AgentMessage[] | Promise<AgentMessage[]>;

export interface AgentHarnessOptions {
	session: Session;
	model: Model<Api>;
	/** Stream function used to call the model. Defaults to the process-wide default set via setDefaultStreamFn(). */
	streamFn?: StreamFn;
	thinkingLevel?: ThinkingLevel;
	activeToolNames?: string[];
	tools?: HarnessTool[];
	toolContext?: object | (() => object | Promise<object>);
	systemPrompt?: string | (() => string | Promise<string>);
	resources?: Resources;
	streamOptions?: StreamOptions;
	retry?: RetryPolicy;
	compaction?: CompactionSettings;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	toolExecution?: "sequential" | "parallel";
	drive?: "automatic" | "manual";
	toProviderMessages?: (
		messages: AgentMessage[],
	) => Message[] | Promise<Message[]>;
	entryProjectors?: Record<string, EntryProjector>;
	context?: TelemetryContext;
	/**
	 * Raw per-token/per-tool-call agent-loop event stream (text_delta, tool_execution_start,
	 * streaming message_update, etc.) — the same events driveLoop's internal emit() already
	 * receives for entry persistence, forwarded here for live UI rendering (transcript
	 * streaming, incremental tool-call display). Errors thrown by this callback are swallowed;
	 * a bad listener must not break a run. Coarser lane-level lifecycle (run/tool/phase/queue
	 * changes) is available separately via `.events`.
	 */
	onLoopEvent?: (event: AgentEvent) => void | Promise<void>;
}

export interface WatchHandle<TSnapshot> {
	snapshot: TSnapshot;
	start(listener: (event: unknown) => void): void;
	unsubscribe(): void;
}

export interface AgentLane {
	readonly name: string;
	getLeafId(): Promise<string | null>;
	prompt(text: string, images?: ImageContent[]): Promise<RunResult>;
	prompt(message: AgentMessage | AgentMessage[]): Promise<RunResult>;
	skill(name: string, additionalInstructions?: string): Promise<RunResult>;
	promptFromTemplate(name: string, args?: string[]): Promise<RunResult>;
	compact(options?: { customInstructions?: string }): Promise<CompactionResult>;
	navigateTree(
		targetId: string | null,
		options?: NavigateOptions,
	): Promise<NavigationResult>;
	resume(): Promise<ResumeResult>;
	abort(): Promise<AbortResult>;
	steer(text: string, images?: ImageContent[]): Promise<QueueResult>;
	steer(message: AgentMessage): Promise<QueueResult>;
	followUp(text: string, images?: ImageContent[]): Promise<QueueResult>;
	followUp(message: AgentMessage): Promise<QueueResult>;
	nextRun(text: string, images?: ImageContent[]): Promise<QueueResult>;
	nextRun(message: AgentMessage): Promise<QueueResult>;
	cancelQueued(entryId: string): Promise<CancelQueuedResult>;
	recordUsage(
		usage: Usage,
		options?: { entryId?: string; details?: JsonValue },
	): Promise<RecordUsageResult>;
	waitForIdle(): Promise<void>;
	runWhenIdle(callback: () => void | Promise<void>): Promise<void>;
	peekAction(): Promise<ActionInfo | undefined>;
	executeAction(): Promise<ActionInfo | undefined>;
	runToCompletion(): Promise<void>;
	getModel(): Promise<Model<Api>>;
	setModel(model: Model<Api>): Promise<void>;
	/**
	 * Current system prompt source (static string, or the live result of the configured function).
	 * Reflects the last `setSystemPrompt()` override, if any — not necessarily the constructor value.
	 */
	getSystemPrompt(): Promise<string>;
	/** Override the system prompt used by future runs on this lane, replacing the constructor value. */
	setSystemPrompt(
		prompt: string | (() => string | Promise<string>),
	): Promise<void>;
	getThinkingLevel(): Promise<ThinkingLevel>;
	setThinkingLevel(level: ThinkingLevel): Promise<void>;
	getActiveTools(): Promise<string[]>;
	setActiveTools(names: string[]): Promise<void>;
	readonly session: SessionTree;
	watch(): Promise<WatchHandle<LaneSnapshot>>;
}

export class AgentHarness implements AgentLane {
	readonly name = "main";
	readonly session: SessionTree;
	readonly hooks: HookRegistry;
	readonly events: HarnessEventBus;
	private readonly durableSession: Session;
	private model: Model<Api>;
	private readonly streamFunction: StreamFn;
	private systemPromptSource: string | (() => string | Promise<string>);
	private readonly toProviderMessages: (
		messages: AgentMessage[],
	) => Message[] | Promise<Message[]>;
	private readonly onLoopEvent?: (event: AgentEvent) => void | Promise<void>;
	private thinkingLevel: ThinkingLevel;
	private activeToolNames: string[];
	private tools: HarnessTool[];
	private resources: Resources;
	private streamOptions: StreamOptions;
	private retryPolicy: RetryPolicy;
	private compactionSettings: CompactionSettings;
	private steeringMode: QueueMode;
	private followUpMode: QueueMode;
	private readonly toolExecutionMode: "sequential" | "parallel";
	private closed = false;
	private activeAbortController: AbortController | null = null;
	private activeRunId: string | null = null;
	private activeOperationKind: "run" | "compaction" | "navigation" | null =
		null;
	private readonly steerQueue: PendingQueue;
	private readonly followUpQueue: PendingQueue;
	private readonly nextRunQueue: PendingQueue;
	private idleWaiters: Array<() => void> = [];
	private idleCallbacks: Array<() => void | Promise<void>> = [];

	private constructor(options: AgentHarnessOptions) {
		this.durableSession = options.session;
		this.session = options.session;
		this.hooks = new HookRegistry(() => this.closed);
		this.events = new HarnessEventBus();
		this.model = options.model;
		this.streamFunction = options.streamFn ?? getDefaultStreamFn();
		this.systemPromptSource = options.systemPrompt ?? "";
		this.toProviderMessages = options.toProviderMessages ?? harnessConvertToLlm;
		this.onLoopEvent = options.onLoopEvent;
		this.thinkingLevel = options.thinkingLevel ?? "off";
		this.activeToolNames = [
			...(options.activeToolNames ??
				options.tools?.map(tool => tool.name) ??
				[]),
		];
		this.tools = [...(options.tools ?? [])];
		this.resources = {
			skills: options.resources?.skills
				? [...options.resources.skills]
				: undefined,
			promptTemplates: options.resources?.promptTemplates
				? [...options.resources.promptTemplates]
				: undefined,
		};
		this.streamOptions = { ...(options.streamOptions ?? {}) };
		this.retryPolicy = options.retry ?? {
			enabled: false,
			maxRetries: 0,
			baseDelayMs: 1000,
		};
		this.compactionSettings = options.compaction ?? DEFAULT_COMPACTION_SETTINGS;
		this.steeringMode = options.steeringMode ?? "one-at-a-time";
		this.followUpMode = options.followUpMode ?? "one-at-a-time";
		this.toolExecutionMode = options.toolExecution ?? "parallel";
		this.steerQueue = new PendingQueue(this.steeringMode);
		this.followUpQueue = new PendingQueue(this.followUpMode);
		// nextRun has no configurable mode upstream — it always drains in full, since it's
		// consumed once per prompt() call rather than mid-run.
		this.nextRunQueue = new PendingQueue("all");
	}

	/**
	 * Opens a harness against a session. Sessions with unfinished (open) operations from a prior process are
	 * reported as `suspended` rather than automatically resumed — full crash-recovery replay via reduceLaneState
	 * (harness/reducer.ts) is not wired up yet, so a session left mid-run must be explicitly `.resume()`d
	 * (not yet implemented either) or navigated away from before this harness will prompt() again on it.
	 */
	static async create(
		options: AgentHarnessOptions,
	): Promise<{ harness: AgentHarness; suspended: SuspendedOperation[] }> {
		const openOperations = await options.session.findOpenOperations("main", {
			limit: 2,
		});
		const harness = new AgentHarness(options);
		if (openOperations.length === 0) return { harness, suspended: [] };
		const suspended: SuspendedOperation[] = openOperations.map(record => ({
			lane: "main",
			kind: record.intent.kind,
			id: record.id,
			startedAt: record.timestamp,
			reason: "crash",
			missing: { tools: [], models: [] },
		}));
		return { harness, suspended };
	}

	private unavailable<T>(operation: string): Promise<T> {
		return Promise.reject(
			this.closed ? new HarnessClosed() : new HarnessNotImplemented(operation),
		);
	}

	async getLeafId(): Promise<string | null> {
		return this.durableSession.getLeafId();
	}

	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		if (Array.isArray(input)) return input;
		if (typeof input !== "string") return [input];
		const content: (import("../ai/types.ts").TextContent | ImageContent)[] = [
			{ type: "text", text: input },
		];
		if (images && images.length > 0) content.push(...images);
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	/**
	 * Shared operation lifecycle: claims the lane, runs `body` with an abort signal wired up, records
	 * operation_finished with the resulting outcome, and always releases the lane before returning.
	 */
	private async runOperation<TOutcome extends { kind: string }>(
		kind: "run" | "compaction" | "navigation",
		body: (runId: string) => Promise<TOutcome>,
	): Promise<ResultValue<{ runId: string } & TOutcome, never>> {
		const runId = uuidv7();
		const controller = new AbortController();
		this.activeAbortController = controller;
		this.activeRunId = runId;
		this.activeOperationKind = kind;
		this.events.emit({
			type: "phase_change",
			lane: this.name,
			phase: kind,
			previousPhase: "idle",
		});
		try {
			const outcome = await body(runId);
			return ok({ runId, ...outcome });
		} finally {
			this.activeAbortController = null;
			this.activeRunId = null;
			this.activeOperationKind = null;
			this.events.emit({
				type: "phase_change",
				lane: this.name,
				phase: "idle",
				previousPhase: kind,
			});
			this.notifyIdle();
		}
	}

	/** Resolve waitForIdle() callers and fire runWhenIdle() callbacks queued while a run was active. */
	private notifyIdle(): void {
		const waiters = this.idleWaiters;
		this.idleWaiters = [];
		for (const resolve of waiters) resolve();

		const callbacks = this.idleCallbacks;
		this.idleCallbacks = [];
		for (const callback of callbacks) void callback();
	}

	/**
	 * Drain a pending queue for injection into the running agent-loop. The drained items become
	 * real message entries via driveLoop's emit() once the loop actually injects them — no
	 * separate queue_cancelled record is needed here, unlike explicit cancelQueued().
	 */
	private drainQueueForLoop(queue: PendingQueue): AgentMessage[] {
		const drained = queue.drain().map(item => item.message);
		if (drained.length > 0) this.emitQueueChange();
		return drained;
	}

	/**
	 * Append a message entry, stripping `undefined`-valued fields first. Tools legitimately
	 * return `details: undefined` / `usage: undefined` when they have no structured payload,
	 * and agent-loop.ts's createToolResultMessage copies those through as present-but-undefined
	 * keys — assertJsonSerializable correctly rejects that, so every message must be sanitized
	 * before it reaches the durable log.
	 */
	private async appendMessageEntry(
		message: AgentMessage,
	): Promise<MessageEntry> {
		return this.durableSession.appendEntry(
			{ id: uuidv7(), type: "message", message: stripUndefinedDeep(message) },
			this.name,
		);
	}

	private async finishOperation(
		runId: string,
		outcome: "completed" | "aborted" | "failed" | "declined",
		error?: { code: string; message: string },
	): Promise<void> {
		await this.durableSession.appendRecord({
			id: uuidv7(),
			lane: this.name,
			type: "operation_finished",
			runId,
			outcome,
			...(error ? { error } : {}),
		});
	}

	private buildSystemPrompt(): string | Promise<string> {
		return typeof this.systemPromptSource === "function"
			? this.systemPromptSource()
			: this.systemPromptSource;
	}

	private toAgentLoopConfig(): AgentLoopConfig {
		return {
			model: this.model,
			convertToLlm: this.toProviderMessages,
			reasoning: this.thinkingLevel === "off" ? undefined : this.thinkingLevel,
			toolExecution: this.toolExecutionMode,
			...this.streamOptions,
			transformContext: this.hooks.has("transform_context")
				? messages => this.hooks.transformContext(messages)
				: undefined,
			beforeToolCall: this.hooks.has("before_tool")
				? context => this.hooks.beforeToolCall(context)
				: undefined,
			afterToolCall: this.hooks.has("after_tool")
				? context => this.hooks.afterToolCall(context)
				: undefined,
		};
	}

	/** Drive the agent-loop for a run, persisting each finalized message and tool start as durable entries/records. */
	private async driveLoop(
		runId: string,
		prompts: AgentMessage[],
	): Promise<RunOutcome> {
		const controller = this.activeAbortController;
		const systemPrompt = await this.buildSystemPrompt();
		// Load prior session history before this run's prompt entries are appended below, so the
		// model sees the full conversation rather than starting fresh on every prompt() call.
		const priorEntries = await this.durableSession.findEntriesOnBranch({
			order: "oldestFirst",
		});
		const { buildSessionContext } = await import("./session/context.ts");
		const priorMessages = buildSessionContext(priorEntries).messages;
		const context: AgentContext = {
			systemPrompt,
			messages: priorMessages,
			tools: this.tools.filter(tool =>
				this.activeToolNames.includes(tool.name),
			),
		};
		const config: AgentLoopConfig = {
			...this.toAgentLoopConfig(),
			getSteeringMessages: async () => this.drainQueueForLoop(this.steerQueue),
			getFollowUpMessages: async () =>
				this.drainQueueForLoop(this.followUpQueue),
		};

		let assistantSeen = 0;
		const toolCallIndexByAssistant = new Map<string, number>();

		const emit = async (event: AgentEvent): Promise<void> => {
			if (this.onLoopEvent) {
				try {
					await this.onLoopEvent(event);
				} catch {
					// A bad listener must not break a run.
				}
			}
			if (event.type !== "message_end") return;
			const message = event.message;
			if (message.role === "user" && assistantSeen === 0) {
				// Initial prompt messages are already persisted by prompt() as provisioned entries before the loop starts.
				return;
			}
			if (message.role === "assistant") {
				assistantSeen++;
				const entry = await this.appendMessageEntry(message);
				this.events.emit({
					type: "entry_added",
					lane: this.name,
					entryId: entry.id,
				});
				const toolCalls = (message as AssistantMessage).content.filter(
					(c): c is AgentToolCall => c.type === "toolCall",
				);
				toolCallIndexByAssistant.set(entry.id, 0);
				for (let index = 0; index < toolCalls.length; index++) {
					const toolCall = toolCalls[index];
					if (!toolCall) continue;
					await this.durableSession.appendRecord({
						id: uuidv7(),
						lane: this.name,
						type: "tool_started",
						runId,
						assistantEntryId: entry.id,
						toolIndex: index,
						toolCallId: toolCall.id,
						toolName: toolCall.name,
						effectiveArgs: toolCall.arguments,
						resultEntryId: uuidv7(),
						replay: "safe",
					});
					this.events.emit({
						type: "tool_start",
						lane: this.name,
						toolCallId: toolCall.id,
						toolName: toolCall.name,
					});
				}
				return;
			}
			if (message.role === "toolResult") {
				const entry = await this.appendMessageEntry(message);
				this.events.emit({
					type: "entry_added",
					lane: this.name,
					entryId: entry.id,
				});
				this.events.emit({
					type: "tool_end",
					lane: this.name,
					toolCallId: message.toolCallId,
					toolName: message.toolName,
					isError: message.isError ?? false,
				});
				return;
			}
			// steering / follow-up user messages injected mid-run
			const entry = await this.appendMessageEntry(message);
			this.events.emit({
				type: "entry_added",
				lane: this.name,
				entryId: entry.id,
			});
		};

		try {
			this.events.emit({ type: "run_start", lane: this.name, runId });
			// Persist the initial prompt entries first so the operation_started intent's initialMessages resolve.
			for (const message of prompts) {
				const entry = await this.appendMessageEntry(message);
				this.events.emit({
					type: "entry_added",
					lane: this.name,
					entryId: entry.id,
				});
			}
			const newMessages = await runAgentLoop(
				prompts,
				context,
				config,
				emit,
				controller?.signal,
				this.streamFunction,
			);
			const finalMessage = [...newMessages]
				.reverse()
				.find((m): m is AssistantMessage => m.role === "assistant");
			await this.finishOperation(runId, "completed");
			// Guaranteed non-null: the initial prompt entries are appended above before this point is reachable.
			const leafId = (await this.durableSession.getLeafId()) as string;
			this.events.emit({
				type: "run_end",
				lane: this.name,
				runId,
				outcome: "completed",
				leafId,
			});
			if (!finalMessage) {
				return {
					kind: "completed",
					leafId,
					finalEntryId: leafId,
					finalMessage: undefined as never,
				};
			}
			return { kind: "completed", leafId, finalEntryId: leafId, finalMessage };
		} catch (error) {
			const aborted = controller?.signal.aborted ?? false;
			const message = error instanceof Error ? error.message : String(error);
			await this.finishOperation(
				runId,
				aborted ? "aborted" : "failed",
				aborted ? undefined : { code: "unknown", message },
			);
			const leafId = (await this.durableSession.getLeafId()) as string;
			this.events.emit({
				type: "run_end",
				lane: this.name,
				runId,
				outcome: aborted ? "aborted" : "failed",
				leafId,
			});
			if (aborted)
				return {
					kind: "aborted",
					leafId,
					finalEntryId: leafId,
					finalMessage: undefined as never,
				};
			return { kind: "failed", leafId, error: { code: "unknown", message } };
		}
	}

	async prompt(text: string, images?: ImageContent[]): Promise<RunResult>;
	async prompt(message: AgentMessage | AgentMessage[]): Promise<RunResult>;
	async prompt(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): Promise<RunResult> {
		if (this.closed)
			return err(new Closed({ message: "AgentHarness is closed" }));
		if (this.activeAbortController) {
			return err(
				new LaneBusy({
					lane: this.name,
					operationId: this.activeRunId ?? "",
					operationKind: this.activeOperationKind ?? "run",
					message: "A run is already active on this lane",
				}),
			);
		}

		// A caller-provided prompt starts after any nextRun() guidance queued since the last run.
		const queuedNextRun = this.nextRunQueue.drain().map(item => item.message);
		if (queuedNextRun.length > 0) this.emitQueueChange();
		const prompts = [
			...queuedNextRun,
			...this.normalizePromptInput(input, images),
		];
		return this.runOperation("run", async runId => {
			const initialEntries: ProvisionedEntry[] = [];
			for (const message of prompts) {
				initialEntries.push({ id: uuidv7(), type: "message", message });
			}
			await this.durableSession.appendRecord({
				id: runId,
				lane: this.name,
				type: "operation_started",
				sourceLeafId: await this.durableSession.getLeafId(),
				intent: {
					kind: "run",
					originalPrompt: prompts,
					initialMessages: initialEntries,
				},
			} satisfies Omit<OperationStartedRecord, "seq" | "timestamp">);

			return this.driveLoop(runId, prompts);
		});
	}

	async skill(
		name: string,
		additionalInstructions?: string,
	): Promise<RunResult> {
		const skill = this.resources.skills?.find(
			candidate => candidate.name === name,
		);
		if (!skill) {
			return err(new UnknownSkill({ name, message: `Unknown skill: ${name}` }));
		}
		const { formatSkillInvocation } = await import("./skills.ts");
		return this.prompt(formatSkillInvocation(skill, additionalInstructions));
	}

	async promptFromTemplate(name: string, args?: string[]): Promise<RunResult> {
		const template = this.resources.promptTemplates?.find(
			candidate => candidate.name === name,
		);
		if (!template) {
			return err(
				new UnknownTemplate({
					name,
					message: `Unknown prompt template: ${name}`,
				}),
			);
		}
		const { formatPromptTemplateInvocation } = await import(
			"./prompt-templates.ts"
		);
		return this.prompt(formatPromptTemplateInvocation(template, args ?? []));
	}

	async compact(options?: {
		customInstructions?: string;
	}): Promise<CompactionResult> {
		if (this.closed)
			return err(new Closed({ message: "AgentHarness is closed" }));
		if (this.activeAbortController) {
			return err(
				new LaneBusy({
					lane: this.name,
					operationId: this.activeRunId ?? "",
					operationKind: this.activeOperationKind ?? "compaction",
					message: "A run is already active on this lane",
				}),
			);
		}

		const pathEntries = await this.durableSession.findEntriesOnBranch({
			order: "oldestFirst",
		});
		const preparationResult = prepareCompaction(
			pathEntries,
			this.compactionSettings,
		);
		if (!preparationResult.ok) {
			return err(
				new NothingToCompact({
					lane: this.name,
					message: preparationResult.error.message,
				}),
			);
		}
		const preparation = preparationResult.value;
		if (!preparation) {
			return err(
				new NothingToCompact({
					lane: this.name,
					message: "Nothing to compact",
				}),
			);
		}

		return this.runOperation("compaction", async runId => {
			const controller = this.activeAbortController;
			const summaryResult = await runCompactionSummary(
				preparation,
				this.model,
				options?.customInstructions,
				controller?.signal,
				this.thinkingLevel,
				this.retryPolicy,
			);
			// Guaranteed non-null: prepareCompaction above only returns a preparation when pathEntries is non-empty.
			if (!summaryResult.ok) {
				await this.finishOperation(runId, "failed", {
					code: summaryResult.error.code,
					message: summaryResult.error.message,
				});
				const leafId = (await this.durableSession.getLeafId()) as string;
				return {
					kind: "failed",
					leafId,
					error: {
						code: summaryResult.error.code,
						message: summaryResult.error.message,
					},
				} satisfies CompactionOutcome;
			}
			const summary = summaryResult.value;
			const compactionEntry = await this.durableSession.appendEntry(
				stripUndefinedDeep({
					id: uuidv7(),
					type: "compaction" as const,
					summary: summary.summary,
					retainedTail: summary.retainedTail,
					tokensBefore: summary.tokensBefore,
					details: summary.details,
					usage: summary.usage,
				}),
				this.name,
			);
			await this.finishOperation(runId, "completed");
			const leafId = (await this.durableSession.getLeafId()) as string;
			return {
				kind: "completed",
				leafId,
				entry: compactionEntry as CompactionEntry,
			} satisfies CompactionOutcome;
		});
	}

	/**
	 * Summarize the current lane's history so far and append it as a branch_summary entry —
	 * a standalone checkpoint, not tied to navigating away (see navigateTree for that case).
	 * Returns the summary text, or null if there was nothing to summarize.
	 */
	async branchSummary(options?: {
		customInstructions?: string;
	}): Promise<string | null> {
		if (this.closed) throw new HarnessClosed();
		const entries = await this.durableSession.findEntriesOnBranch({
			order: "oldestFirst",
		});
		if (entries.length === 0) return null;
		const fromId = (await this.durableSession.getLeafId()) as string;
		const controller = new AbortController();
		const result = await generateBranchSummary(entries, {
			model: this.model,
			signal: controller.signal,
			customInstructions: options?.customInstructions,
			thinkingLevel:
				this.thinkingLevel === "off" ? undefined : this.thinkingLevel,
			retry: this.retryPolicy,
		});
		if (!result.ok) return null;
		const summary = result.value;
		await this.durableSession.appendEntry(
			stripUndefinedDeep({
				id: uuidv7(),
				type: "branch_summary" as const,
				fromId,
				summary: summary.summary,
				details: {
					readFiles: summary.readFiles,
					modifiedFiles: summary.modifiedFiles,
				},
				usage: summary.usage,
			}),
			this.name,
		);
		return summary.summary;
	}

	async navigateTree(
		_targetId: string | null,
		_options?: NavigateOptions,
	): Promise<NavigationResult> {
		return this.unavailable("navigateTree");
	}
	async resume(): Promise<ResumeResult> {
		return this.unavailable("resume");
	}

	async abort(): Promise<AbortResult> {
		if (this.closed)
			return err(new Closed({ message: "AgentHarness is closed" }));
		const controller = this.activeAbortController;
		const runId = this.activeRunId;
		if (!controller || !runId) {
			return err(
				new NoActiveOperation({
					lane: this.name,
					message: "No active operation to abort",
				}),
			);
		}
		controller.abort();
		// Return queued-but-unconsumed steer/followUp messages to the caller rather than
		// silently dropping them — the aborted run will never drain these itself.
		const steer = this.steerQueue.drain().map(item => item.message);
		const followUp = this.followUpQueue.drain().map(item => item.message);
		this.emitQueueChange();
		return ok({ runId, steer, followUp });
	}
	/** Text preview of a queued message, for queue_change's display-only string arrays. */
	private static queuedMessageText(item: QueuedItem): string {
		const message = item.message;
		if ("content" in message) {
			if (typeof message.content === "string") return message.content;
			if (Array.isArray(message.content)) {
				const text = message.content.find(
					(block): block is { type: "text"; text: string } =>
						typeof block === "object" &&
						block !== null &&
						"type" in block &&
						block.type === "text",
				);
				if (text) return text.text;
			}
		}
		return item.entryId;
	}

	/** Emit the current contents of all three queues as a single queue_change event. */
	private emitQueueChange(): void {
		this.events.emit({
			type: "queue_change",
			lane: this.name,
			steering: this.steerQueue.peekAll().map(AgentHarness.queuedMessageText),
			followUp: this.followUpQueue
				.peekAll()
				.map(AgentHarness.queuedMessageText),
			nextRun: this.nextRunQueue.peekAll().map(AgentHarness.queuedMessageText),
		});
	}

	private async enqueue(
		queue: PendingQueue,
		queueName: "steer" | "followUp" | "nextRun",
		input: string | AgentMessage,
		images?: ImageContent[],
	): Promise<QueueResult> {
		if (this.closed)
			return err(new Closed({ message: "AgentHarness is closed" }));
		const [message] = this.normalizePromptInput(input, images);
		if (!message)
			return err(
				new InvalidMessage({
					lane: this.name,
					reason: "empty",
					message: "No message provided",
				}),
			);

		const entryId = uuidv7();
		const target: ProvisionedEntry = { id: entryId, type: "message", message };
		if (queueName === "nextRun") {
			await this.durableSession.appendRecord({
				id: uuidv7(),
				lane: this.name,
				type: "queue_enqueued",
				queue: "nextRun",
				target,
			});
		} else {
			await this.durableSession.appendRecord({
				id: uuidv7(),
				lane: this.name,
				type: "queue_enqueued",
				queue: queueName,
				runId: this.activeRunId ?? "",
				target,
			});
		}
		queue.enqueue({ entryId, message });
		this.emitQueueChange();
		return ok({ entryId });
	}

	async steer(text: string, images?: ImageContent[]): Promise<QueueResult>;
	async steer(message: AgentMessage): Promise<QueueResult>;
	async steer(
		input: string | AgentMessage,
		images?: ImageContent[],
	): Promise<QueueResult> {
		if (!this.activeAbortController) {
			return err(
				new NoActiveRun({ lane: this.name, message: "No active run to steer" }),
			);
		}
		return this.enqueue(this.steerQueue, "steer", input, images);
	}
	async followUp(text: string, images?: ImageContent[]): Promise<QueueResult>;
	async followUp(message: AgentMessage): Promise<QueueResult>;
	async followUp(
		input: string | AgentMessage,
		images?: ImageContent[],
	): Promise<QueueResult> {
		if (!this.activeAbortController) {
			return err(
				new NoActiveRun({
					lane: this.name,
					message: "No active run to follow up on",
				}),
			);
		}
		return this.enqueue(this.followUpQueue, "followUp", input, images);
	}
	async nextRun(text: string, images?: ImageContent[]): Promise<QueueResult>;
	async nextRun(message: AgentMessage): Promise<QueueResult>;
	async nextRun(
		input: string | AgentMessage,
		images?: ImageContent[],
	): Promise<QueueResult> {
		return this.enqueue(this.nextRunQueue, "nextRun", input, images);
	}
	async cancelQueued(entryId: string): Promise<CancelQueuedResult> {
		if (this.closed)
			return err(new Closed({ message: "AgentHarness is closed" }));
		const removed =
			this.steerQueue.remove(entryId) ||
			this.followUpQueue.remove(entryId) ||
			this.nextRunQueue.remove(entryId);
		if (!removed) {
			return err(
				new UnknownQueueItem({
					lane: this.name,
					entryId,
					message: `No queued item ${entryId}`,
				}),
			);
		}
		await this.durableSession.appendRecord({
			id: uuidv7(),
			lane: this.name,
			type: "queue_cancelled",
			entryId,
		});
		this.emitQueueChange();
		return ok({ outcome: "cancelled" });
	}
	async recordUsage(
		_usage: Usage,
		_options?: { entryId?: string; details?: JsonValue },
	): Promise<RecordUsageResult> {
		return this.unavailable("recordUsage");
	}
	async waitForIdle(): Promise<void> {
		if (!this.activeAbortController) return;
		await new Promise<void>(resolve => {
			this.idleWaiters.push(resolve);
		});
	}
	async runWhenIdle(callback: () => void | Promise<void>): Promise<void> {
		if (!this.activeAbortController) {
			await callback();
			return;
		}
		this.idleCallbacks.push(callback);
	}
	async peekAction(): Promise<ActionInfo | undefined> {
		return this.unavailable("peekAction");
	}
	async executeAction(): Promise<ActionInfo | undefined> {
		return this.unavailable("executeAction");
	}
	async runToCompletion(): Promise<void> {
		return this.unavailable("runToCompletion");
	}
	async getModel(): Promise<Model<Api>> {
		return this.model;
	}
	async setModel(model: Model<Api>): Promise<void> {
		this.model = model;
	}
	async getSystemPrompt(): Promise<string> {
		return this.buildSystemPrompt();
	}
	async setSystemPrompt(
		prompt: string | (() => string | Promise<string>),
	): Promise<void> {
		this.systemPromptSource = prompt;
	}
	async getThinkingLevel(): Promise<ThinkingLevel> {
		return this.thinkingLevel;
	}
	async setThinkingLevel(level: ThinkingLevel): Promise<void> {
		this.thinkingLevel = level;
	}
	async getActiveTools(): Promise<string[]> {
		return [...this.activeToolNames];
	}
	async setActiveTools(names: string[]): Promise<void> {
		this.activeToolNames = [...names];
	}
	async watch(): Promise<WatchHandle<LaneSnapshot>> {
		return this.unavailable("watch");
	}

	async lane(_name: string): Promise<AgentLane | undefined> {
		return this.unavailable("lane");
	}
	async createLane(
		_name: string,
		_at: string | null,
	): Promise<CreateLaneResult> {
		return this.unavailable("createLane");
	}
	async lanes(): Promise<LaneInfo[]> {
		return this.unavailable("lanes");
	}
	async getTools(): Promise<HarnessTool[]> {
		return [...this.tools];
	}
	async setTools(tools: HarnessTool[], activeNames?: string[]): Promise<void> {
		this.tools = [...tools];
		this.activeToolNames = [...(activeNames ?? tools.map(tool => tool.name))];
	}
	async getResources(): Promise<Resources> {
		return {
			skills: this.resources.skills ? [...this.resources.skills] : undefined,
			promptTemplates: this.resources.promptTemplates
				? [...this.resources.promptTemplates]
				: undefined,
		};
	}
	async setResources(resources: Resources): Promise<void> {
		this.resources = {
			skills: resources.skills ? [...resources.skills] : undefined,
			promptTemplates: resources.promptTemplates
				? [...resources.promptTemplates]
				: undefined,
		};
	}
	async getStreamOptions(): Promise<StreamOptions> {
		return { ...this.streamOptions };
	}
	async setStreamOptions(options: StreamOptions): Promise<void> {
		this.streamOptions = { ...options };
	}
	async getRetryPolicy(): Promise<RetryPolicy> {
		return { ...this.retryPolicy };
	}
	async setRetryPolicy(policy: RetryPolicy): Promise<void> {
		this.retryPolicy = { ...policy };
	}
	async getCompactionSettings(): Promise<CompactionSettings> {
		return { ...this.compactionSettings };
	}
	async setCompactionSettings(settings: CompactionSettings): Promise<void> {
		this.compactionSettings = { ...settings };
	}
	async getSteeringMode(): Promise<QueueMode> {
		return this.steeringMode;
	}
	async setSteeringMode(mode: QueueMode): Promise<void> {
		this.steeringMode = mode;
	}
	async getFollowUpMode(): Promise<QueueMode> {
		return this.followUpMode;
	}
	async setFollowUpMode(mode: QueueMode): Promise<void> {
		this.followUpMode = mode;
	}
	async watchSession(): Promise<WatchHandle<SessionSnapshot>> {
		return this.unavailable("watchSession");
	}
	async close(): Promise<void> {
		this.closed = true;
	}
}
