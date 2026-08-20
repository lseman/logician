// ── AgentCoreBridgeV2 ────────────────────────────────────────────────────────
// Core-slice port of AgentCoreBridge (agent-bridge.ts) onto @logician/agent.
//
// Scope: construction, prompt/compact/abort/steer/followUp/nextRun, default
// tool registration, permission gating, and event translation to the same
// RuntimeEvent shapes the transcript already renders. Deferred subsystems —
// MCP, memory, LSP, reasoner, repository-map, extensions, plugins, skills,
// prompts, sandbox — are stubbed as documented no-ops below; each returns a
// value/shape consistent with "feature unavailable" rather than throwing, so
// callers written against the full AgentCoreBridge surface don't crash, but
// none of those features actually do anything yet.
//
// New file alongside the original — apps/tui does not import this yet.

import { randomUUID } from "node:crypto";
import {
	AgentHarness,
	type AgentMessage,
	buildSessionContext,
	type Context,
	createBashTool,
	createEditTool,
	createPermissionHook,
	createReadTool,
	createWriteTool,
	InMemorySessionRepo,
	type AgentEvent as LoopEvent,
	type Model,
	NodeExecutionEnv,
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
	type RunResult,
	type Session,
	type SimpleStreamOptions,
	streamSimple,
	ToolRegistry,
} from "@logician/agent/node";
import type { RuntimeEvent } from "../runtime/events.ts";

export type EventCallback = (event: RuntimeEvent) => void;
export type ErrorCallback = (err: Error) => void;

// ── Bridge options ───────────────────────────────────────────────────────────
// Deliberately a subset of AgentBridgeOptions: only the fields the core
// slice actually consumes. Fields belonging to deferred subsystems are
// omitted rather than accepted-and-ignored, so a caller notices at the type
// level that (e.g.) memory/MCP/reasoner options have no effect here yet.

export interface AgentBridgeV2Options {
	baseUrl: string;
	model: string;
	apiKey?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	thinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
	contextWindowTokens?: number;
	permissionMode?: PermissionMode;
	permissionRules?: PermissionRules;
	cwd?: string;
	systemPrompt?: string;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
}

// A "flat" message shape matching what the TUI's RuntimeEvent/transcript code
// already expects from the old agent-core Message type — role/content/tool_calls.
export interface FlatMessage {
	role: string;
	content: string | null;
	tool_call_id?: string;
	tool_calls?: Array<{ id: string; name: string; arguments: string }>;
	timestamp?: number;
}

/** Convert one AgentMessage (this package's message model) to the old flat shape. */
function toFlatMessage(message: AgentMessage): FlatMessage | null {
	if (!("role" in message)) return null; // custom app message types — not renderable here
	if (message.role === "user") {
		const text =
			typeof message.content === "string"
				? message.content
				: message.content
						.filter(
							(block): block is { type: "text"; text: string } =>
								block.type === "text",
						)
						.map(block => block.text)
						.join("");
		return { role: "user", content: text, timestamp: message.timestamp };
	}
	if (message.role === "assistant") {
		const text = message.content
			.filter(
				(block): block is { type: "text"; text: string } =>
					block.type === "text",
			)
			.map(block => block.text)
			.join("");
		const toolCalls = message.content
			.filter(
				(block): block is Extract<typeof block, { type: "toolCall" }> =>
					block.type === "toolCall",
			)
			.map(call => ({
				id: call.id,
				name: call.name,
				arguments: JSON.stringify(call.arguments),
			}));
		return {
			role: "assistant",
			content: text || null,
			tool_calls: toolCalls.length ? toolCalls : undefined,
			timestamp: message.timestamp,
		};
	}
	if (message.role === "toolResult") {
		const text = message.content
			.filter(
				(block): block is { type: "text"; text: string } =>
					block.type === "text",
			)
			.map(block => block.text)
			.join("");
		return {
			role: "tool",
			content: text,
			tool_call_id: message.toolCallId,
			timestamp: message.timestamp,
		};
	}
	return null;
}

export class AgentCoreBridgeV2 {
	private readonly opts: AgentBridgeV2Options;
	private readonly cwd: string;
	private readonly model: Model;
	private readonly permissionManager: PermissionManager;
	private readonly toolRegistry: ToolRegistry;
	private readonly env: NodeExecutionEnv;
	private readonly sessionRepo = new InMemorySessionRepo();
	private durableSession: Session | undefined;
	private harness: AgentHarness | null = null;
	private harnessPromise: Promise<AgentHarness> | null = null;
	private sessionId: string;

	private callbacks: EventCallback[] = [];
	private errorCb: ErrorCallback | null = null;
	private running = false;
	private sendTail: Promise<void> = Promise.resolve();
	private permissionResolvers = new Map<
		string,
		(decision: "allow" | "deny" | "always") => void
	>();
	private questionResolvers = new Map<
		string,
		{ allow: (answer: string) => void; deny: () => void }
	>();
	private contextTokens = 0;
	private contextMaxTokens: number | undefined;
	private systemPromptOverride: string | undefined;

	constructor(opts: AgentBridgeV2Options) {
		this.opts = opts;
		this.cwd = opts.cwd || process.cwd();
		this.sessionId = randomUUID();
		this.model = {
			id: opts.model,
			name: opts.model,
			api: "openai-completions",
			provider: "openai-compatible",
			baseUrl: opts.baseUrl,
			reasoning: false,
			contextWindow: opts.contextWindowTokens ?? 128_000,
			maxTokens: opts.maxTokens ?? 4096,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		};
		this.env = new NodeExecutionEnv({ cwd: this.cwd });
		this.permissionManager = new PermissionManager({
			mode: opts.permissionMode ?? "acceptEdits",
			rules: opts.permissionRules,
		});
		this.toolRegistry = new ToolRegistry();
		this.toolRegistry.registerMany([
			createBashTool({ env: this.env }),
			createReadTool({
				env: this.env,
				allowedPaths: opts.allowedPaths,
				allowAllPaths: opts.allowAllPaths,
			}),
			createWriteTool({
				env: this.env,
				allowedPaths: opts.allowedPaths,
				allowAllPaths: opts.allowAllPaths,
			}),
			createEditTool({
				env: this.env,
				allowedPaths: opts.allowedPaths,
				allowAllPaths: opts.allowAllPaths,
			}),
		]);
	}

	// ── Harness lifecycle ───────────────────────────────────────────────────

	/** Lazily build the singleton harness. Session-first construction, unlike the old post-hoc attachSession(). */
	private async ensureHarness(): Promise<AgentHarness> {
		if (this.harness) return this.harness;
		if (!this.harnessPromise) {
			this.harnessPromise = this.buildHarness();
		}
		this.harness = await this.harnessPromise;
		return this.harness;
	}

	private async buildHarness(): Promise<AgentHarness> {
		if (!this.durableSession) {
			this.durableSession = await this.sessionRepo.create({
				id: this.sessionId,
			});
		}
		const streamFn = (
			model: Model,
			context: Context,
			options?: SimpleStreamOptions,
		) => streamSimple(model, context, { ...options, apiKey: this.opts.apiKey });

		const { harness } = await AgentHarness.create({
			session: this.durableSession,
			model: this.model,
			streamFn,
			thinkingLevel: this.opts.thinkingLevel,
			systemPrompt: () =>
				this.systemPromptOverride ??
				this.opts.systemPrompt ??
				"You are a helpful assistant.",
			retry: undefined,
			toolExecution: "parallel",
			...this.toolRegistry.toHarnessOptions(),
			onLoopEvent: event => this.handleLoopEvent(event),
		});

		harness.hooks.on(
			"before_tool",
			createPermissionHook(this.permissionManager, (call, args, reason) =>
				this.askPermission(call.id, call.name, args, reason),
			),
		);

		harness.events.on("phase_change", e => {
			if (e.phase === "run") return; // matches old emitHarnessPhase: "turn" phase is a no-op
			this.emit({
				type: "phase",
				state: e.phase === "compaction" ? "compacting" : "ready",
			});
		});
		harness.events.on("queue_change", e => {
			this.emit({
				type: "queue_update",
				steering: [...e.steering],
				followUp: [...e.followUp],
				nextTurn: [...e.nextRun],
			});
		});

		return harness;
	}

	private askPermission(
		toolCallId: string,
		toolName: string,
		args: Record<string, unknown>,
		_reason: string | undefined,
	): Promise<"allow" | "deny" | "always"> {
		return new Promise(resolve => {
			this.permissionResolvers.set(toolCallId, resolve);
			this.emit({
				type: "permission_request",
				toolName,
				toolCallId,
				args,
			});
		});
	}

	/** Translate the raw per-token agent-loop event stream into RuntimeEvent for the transcript. */
	private handleLoopEvent(event: LoopEvent): void {
		switch (event.type) {
			case "turn_start":
				this.emit({ type: "agent_iteration_start", iteration: 1 });
				return;
			case "message_update": {
				const ame = event.assistantMessageEvent;
				if (ame.type === "text_delta") {
					this.emit({ type: "token", token: ame.delta });
				} else if (ame.type === "thinking_delta") {
					this.emit({ type: "thinking_token", token: ame.delta });
				} else if (ame.type === "toolcall_end") {
					this.emit({
						type: "tool_call_start",
						toolName: ame.toolCall.name,
						args: ame.toolCall.arguments as Record<string, unknown>,
						toolCallId: ame.toolCall.id,
					});
				}
				return;
			}
			case "tool_execution_start":
				this.emit({
					type: "tool_execution_start",
					toolName: event.toolName,
					args: (event.args as Record<string, unknown>) ?? {},
					toolCallId: event.toolCallId,
				});
				return;
			case "tool_execution_update":
				this.emit({
					type: "tool_execution_update",
					toolName: event.toolName,
					partialResult: String(event.partialResult ?? ""),
					toolCallId: event.toolCallId,
				});
				return;
			case "tool_execution_end": {
				const toolResult = event.result as {
					content?: Array<{ type: string; text?: string }>;
					details?: Record<string, unknown>;
				};
				const text = (toolResult?.content ?? [])
					.filter(block => block.type === "text")
					.map(block => block.text ?? "")
					.join("");
				this.emit({
					type: "tool_execution_end",
					toolName: event.toolName,
					result: text,
					isError: event.isError,
					toolCallId: event.toolCallId,
					details: toolResult?.details,
				});
				return;
			}
			default:
				return;
		}
	}

	private emit(event: RuntimeEvent): void {
		for (const cb of this.callbacks) {
			try {
				cb(event);
			} catch {
				// Don't let a bad handler kill the bridge.
			}
		}
	}

	// ── Event registration ──────────────────────────────────────────────────

	on(callback: EventCallback): () => void {
		this.callbacks.push(callback);
		return () => {
			this.callbacks = this.callbacks.filter(cb => cb !== callback);
		};
	}

	onError(callback: ErrorCallback): void {
		this.errorCb = callback;
	}

	reportError(error: unknown): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.emit({
			type: "notice",
			level: "error",
			label: "Error",
			text: normalized.message,
		});
	}

	// ── High-level commands ──────────────────────────────────────────────────

	async sendMessage(message: string): Promise<void> {
		if (this.running && this.harness) {
			await this.steer(message);
			this.emit({ type: "steered", message });
			return;
		}
		const run = this.sendTail.then(() => this.runMessage(message));
		this.sendTail = run.catch(() => {});
		return run;
	}

	sendSlash(raw: string): void {
		this.sendMessage(raw).catch(err => this.errorCb?.(err));
	}

	private async runMessage(message: string): Promise<void> {
		this.running = true;
		const turnId = `turn_${Date.now()}`;
		try {
			const harness = await this.ensureHarness();
			this.emit({ type: "turn_start", turnId });
			const result = await harness.prompt(message);
			this.reportRunResult(result);
		} catch (err: unknown) {
			const error = err as Error;
			this.emit({
				type: "notice",
				level: "error",
				label: "Error",
				text: error.message,
			});
			this.errorCb?.(error);
			throw error;
		} finally {
			this.running = false;
			this.publishContextUsage();
			this.emit({ type: "turn_end", turnId });
			this.emit({ type: "phase", state: "ready" });
		}
	}

	private reportRunResult(result: RunResult): void {
		if (result.ok) return;
		this.emit({
			type: "notice",
			level: "error",
			label: "Run rejected",
			text: result.error.message,
		});
	}

	// ── Queue operations ─────────────────────────────────────────────────────

	async steer(message: string): Promise<void> {
		const harness = await this.ensureHarness();
		await harness.steer(message);
	}

	async followUp(message: string): Promise<void> {
		const harness = await this.ensureHarness();
		await harness.followUp(message);
	}

	async nextTurn(message: string): Promise<void> {
		const harness = await this.ensureHarness();
		await harness.nextRun(message);
	}

	/** No mode-toggle equivalent yet — queue drain mode is fixed at harness construction. */
	setSteeringMode(_mode: unknown): void {}
	setFollowUpMode(_mode: unknown): void {}
	setSteeringInterrupt(_enabled: boolean): void {}
	getSteeringInterrupt(): boolean {
		return false;
	}
	getSteeringMessages(): string[] {
		return [];
	}
	getFollowUpMessages(): string[] {
		return [];
	}
	getNextTurnMessages(): string[] {
		return [];
	}
	/** No cancelQueued-by-position equivalent surfaced yet (harness cancelQueued() takes an entryId, not a display index). */
	dropQueuedMessage(_displayIndex: number): string | undefined {
		return undefined;
	}
	clearQueue(): { steering: string[]; followUp: string[]; nextTurn: string[] } {
		return { steering: [], followUp: [], nextTurn: [] };
	}
	flushSteeringNow(): number {
		return 0;
	}

	async abort(): Promise<{ runId: string } | null> {
		if (!this.harness) return null;
		const result = await this.harness.abort();
		return result.ok ? { runId: result.value.runId } : null;
	}

	async cancel(): Promise<{ runId: string } | null> {
		this.denyPendingPermissions();
		return this.abort();
	}

	private denyPendingPermissions(): void {
		for (const [id, resolve] of this.permissionResolvers) {
			resolve("deny");
			this.permissionResolvers.delete(id);
		}
	}

	// ── Compaction & branching ───────────────────────────────────────────────

	async compact(): Promise<{
		tokensSaved: number;
		tokensBefore: number;
		tokensAfter: number;
	} | null> {
		const harness = await this.ensureHarness();
		const result = await harness.compact();
		if (!result.ok || result.value.kind !== "completed") return null;
		const before = result.value.entry.tokensBefore;
		const after = this.contextTokens;
		this.emit({
			type: "compaction",
			reason: "manual",
			tokensBefore: before,
			tokensAfter: after,
		});
		return {
			tokensSaved: before - after,
			tokensBefore: before,
			tokensAfter: after,
		};
	}

	async branchSummary(): Promise<string | null> {
		const harness = await this.ensureHarness();
		return harness.branchSummary();
	}

	/** No fork/rewind/discardBranch equivalent in the new session-tree model (see session gap notes). */
	fork(): string | null {
		return null;
	}
	rewind(): { messages: number; filesRestored: number } | null {
		return null;
	}
	discardBranch(): boolean {
		return false;
	}

	// ── Permissions & questions ──────────────────────────────────────────────

	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean {
		const resolve = this.permissionResolvers.get(toolCallId);
		if (!resolve) return false;
		this.permissionResolvers.delete(toolCallId);
		resolve(decision);
		return true;
	}

	hasPendingPermission(): boolean {
		return this.permissionResolvers.size > 0;
	}

	askQuestion(
		question: string,
		choices: Array<{ value: string; label: string }>,
	): string {
		const qid = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
		this.questionResolvers.set(qid, { allow: () => {}, deny: () => {} });
		this.emit({
			type: "question_request",
			questionId: qid,
			questions: [{ id: "answer", question, choices }],
		});
		return qid;
	}

	respondToQuestion(questionId: string, answer: string): boolean {
		const resolver = this.questionResolvers.get(questionId);
		if (!resolver) return false;
		this.questionResolvers.delete(questionId);
		resolver.allow(answer);
		return true;
	}

	hasPendingQuestion(): boolean {
		return this.questionResolvers.size > 0;
	}

	setPermissionMode(mode: PermissionMode): void {
		this.permissionManager.setMode(mode);
		this.emit({
			type: "notice",
			level: "info",
			label: "Permissions",
			text: `mode: ${mode}`,
		});
	}

	getPermissionMode(): PermissionMode {
		return this.permissionManager.getMode();
	}

	// ── Model ─────────────────────────────────────────────────────────────────

	getCurrentModel(): string {
		return this.model.id;
	}

	getCurrentBaseUrl(): string {
		return this.model.baseUrl;
	}

	getModels(): string[] {
		return [this.getCurrentModel()];
	}

	getModelOptions(): Array<{
		key: string;
		name: string;
		model: string;
		url: string;
		active: boolean;
	}> {
		return [
			{
				key: this.getCurrentModel(),
				name: this.getCurrentModel(),
				model: this.getCurrentModel(),
				url: this.getCurrentBaseUrl(),
				active: true,
			},
		];
	}

	/** No multi-model cycling list yet — single fixed model per bridge instance. */
	setModelOption(_key: string): { model: string; url: string } | null {
		return null;
	}
	cycleModel(_direction?: "forward" | "backward"): string | null {
		return null;
	}
	setModels(_models: unknown[]): void {}

	async setModel(modelId: string): Promise<void> {
		if (!this.harness) return;
		await this.harness.setModel({ ...this.model, id: modelId, name: modelId });
	}

	async setThinkingLevel(level: string): Promise<void> {
		if (!this.harness) return;
		await this.harness.setThinkingLevel(
			level as "off" | "minimal" | "low" | "medium" | "high" | "xhigh",
		);
	}

	setTemperature(_temperature: number): void {
		// No live temperature setter on AgentHarness yet — samplingParams is set at model construction.
	}

	setMaxTokens(_maxTokens: number): void {}
	setMaxIterations(_maxIterations: number): void {}
	setExecutionProfile(_profile: string): void {}
	setInferenceMode(_mode: string): void {}
	setRuntimeToggle(_key: string, _enabled: boolean): void {}
	setGuardMode(_mode: "auto" | "on" | "off"): void {}

	// ── Config & state ───────────────────────────────────────────────────────

	getConfig(): { baseUrl: string; model: string } {
		return { baseUrl: this.model.baseUrl, model: this.model.id };
	}

	getSettingsData(): Record<string, unknown> {
		return {
			model: this.model.id,
			temperature: this.opts.temperature ?? 0.5,
			maxTokens: this.model.maxTokens,
			maxIterations: this.opts.maxIterations ?? 30,
			thinkingLevel: this.opts.thinkingLevel ?? "off",
			permissionMode: this.getPermissionMode(),
		};
	}

	async getState(): Promise<Record<string, unknown>> {
		return {
			agent_name: "logician",
			model: this.model.id,
			base_url: this.model.baseUrl,
			tools: this.toolRegistry.list().map(t => t.name),
			context_tokens: this.contextTokens,
			context_max_tokens: this.contextMaxTokens,
			connected: true,
		};
	}

	isActive(): boolean {
		return this.running;
	}

	async getMessages(): Promise<FlatMessage[]> {
		if (!this.durableSession) return [];
		const entries = await this.durableSession.findEntriesOnBranch({
			order: "oldestFirst",
		});
		const context = buildSessionContext(entries);
		return context.messages
			.map(toFlatMessage)
			.filter((m): m is FlatMessage => m !== null);
	}

	async getContext(): Promise<string> {
		const messages = await this.getMessages();
		const lines = messages.map(
			m => `[${m.role.toUpperCase()}]\n${m.content ?? ""}`,
		);
		return `## Context (${messages.length} messages)\n\n${lines.join("\n\n")}`;
	}

	private publishContextUsage(): void {
		// No token-accounting accessor is exposed by AgentHarness yet; context_update
		// stays at its last known value until that gap is closed.
		this.emit({
			type: "context_update",
			tokens: this.contextTokens,
			maxTokens: this.contextMaxTokens,
			compacted: false,
		});
	}

	getTools(): ToolRegistry {
		return this.toolRegistry;
	}

	async useConversationSession(
		sessionId: string,
		durableSession?: Session,
	): Promise<void> {
		if (!sessionId.trim()) return;
		this.sessionId = sessionId;
		if (durableSession) {
			this.durableSession = durableSession;
			this.harness = null;
			this.harnessPromise = null;
		}
	}

	renameConversationSession(_sessionId: string, _name: string): void {}

	reset(): void {
		this.harness = null;
		this.harnessPromise = null;
		this.durableSession = undefined;
		this.sessionId = randomUUID();
		this.contextTokens = 0;
		this.emit({ type: "turn_end", turnId: "reset" });
	}

	async init(): Promise<Record<string, unknown>> {
		await this.ensureHarness();
		this.emit({ type: "phase", state: "ready" });
		return this.getState();
	}

	async stop(): Promise<void> {
		await this.cancel();
		this.running = false;
	}

	// ── Deferred subsystems: honest no-ops ───────────────────────────────────
	// Each below belongs to MCP / memory / LSP / reasoner / repository-map /
	// extensions / plugins / skills / prompts / sandbox — none built yet against
	// @logician/agent. Returns are shaped as "nothing available" rather than
	// throwing, so a caller written against the full bridge surface degrades
	// gracefully instead of crashing.

	getSandboxMode(): "none" | "code" | "full" {
		return "none";
	}
	setSandboxMode(_mode: "none" | "code" | "full"): void {}
	cycleSandboxMode(): "none" | "code" | "full" {
		return "none";
	}
	getReasonerStatus(): string {
		return "none";
	}
	setReasonerId(_reasonerId: string): void {}
	spawnAgentDirectly(_task: string, _agent?: string): void {}
	eohCommand(_raw: string): string {
		return "";
	}
	isMcpLoading(): boolean {
		return false;
	}
	async getMcpSnapshot(): Promise<{ servers: unknown[] }> {
		return { servers: [] };
	}
	async setMcpServerEnabled(
		_serverName: string,
		_enabled: boolean,
	): Promise<{ ok: boolean }> {
		return { ok: false };
	}
	async getPluginSnapshot(): Promise<{ status: string; plugins: unknown[] }> {
		return { status: "unavailable", plugins: [] };
	}
	async runPluginCommand(_input: string): Promise<string> {
		return "Plugins are not available in this bridge yet.";
	}
	getSkills(): unknown[] {
		return [];
	}
	invokeSkill(_name: string, _args: string): boolean {
		return false;
	}
	getPrompts(): unknown[] {
		return [];
	}
	invokePrompt(_name: string, _args: string): boolean {
		return false;
	}
	getMemoryStore(): null {
		return null;
	}
	getMemoryStats(): {
		memoryEnabled: boolean;
		memoryCount: number;
		sessionCount: number;
		observationCount: number;
	} {
		return {
			memoryEnabled: false,
			memoryCount: 0,
			sessionCount: 0,
			observationCount: 0,
		};
	}
	getExtensionCommands(): unknown[] {
		return [];
	}
	async invokeExtensionCommand(
		_name: string,
		_args: string,
	): Promise<string | undefined> {
		return undefined;
	}
	async emitUserBashEvent(
		_command: string,
		_excludeFromContext?: boolean,
	): Promise<{ action: "continue" | "intercept" | "replace" } | null> {
		return { action: "continue" };
	}
	async emitProjectTrustEvent(
		_cwd: string,
	): Promise<{ trusted: "yes" | "no" | "undecided" } | null> {
		return { trusted: "undecided" };
	}
	async executeBashCommand(
		command: string,
	): Promise<{ output: string; exitCode: number }> {
		const result = await this.env.exec(command);
		if (!result.ok) return { output: result.error.message, exitCode: 1 };
		return {
			output: result.value.stdout + result.value.stderr,
			exitCode: result.value.exitCode,
		};
	}
}
