// ── AgentCoreBridge ──────────────────────────────────────────────────────────────
import { envNumber } from "../tui-utils.ts";
// Replaces the Python bridge with direct TypeScript agent-core integration.
// Translates agent-core events to the same shapes the transcript expects.

import { readFileSync } from "node:fs";
import {
	readdir as readdirAsync,
	readFile as readFileAsync,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
	type AgentDefinition,
	loadAgentDefinitions,
} from "@logician/agent-capabilities/delegation/definitions.ts";
import {
	getBuiltInSubagentTools,
	type SubagentToolDeps,
} from "@logician/agent-capabilities/tools";
import {
	type AgentConfig,
	type AgentEvent,
	AgentHarness,
	type AgentModelConfig,
	type HarnessPhase,
	type Message,
	type Tool,
	type TruncationConfig,
	type WebSearchConfig,
} from "@logician/agent-core";
import { OpenAIBackend } from "@logician/agent-core/agent/backend.ts";
import {
	estimateChatPayloadTokens,
	estimateTokens,
} from "@logician/agent-core/agent/messages.ts";
import { onTodosChanged } from "@logician/agent-core/agent/tasks/todo-state.ts";
import { parseFrontmatter } from "@logician/agent-core/tools/shared/frontmatter.ts";
import {
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
} from "@logician/agent-core/tools/shared/permissions.ts";
import {
	configurePluginRuntimeEnv,
	type PluginCommandResult,
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
} from "@logician/agent-core/tools/shared/plugins.ts";
import { ToolRegistry } from "@logician/agent-core/tools/shared/registry.ts";
import {
	findLogicianConfig,
	loadLogicianConfig,
} from "../configuration/config.ts";
import {
	McpManager,
	type McpSnapshotResult,
	type McpToggleResult,
} from "../mcp/index.ts";
import {
	formatActivatedSkills,
	formatSkillActivationNotice,
	SkillActivationSession,
	type selectSkillsForPrompt,
} from "../skills/activation.ts";
import {
	findSkillByName,
	formatSkillCatalog,
	formatSkillInvocation,
	loadSkills,
	type Skill,
} from "../skills/index.ts";
import {
	findPromptByName,
	loadPrompts,
	type Prompt,
} from "../prompts/index.ts";
import { buildDefaultSystemPrompt } from "../context/system-prompt.ts";
import { createDefaultTools } from "../tools/default-tools.ts";
import { createReadSkillTool } from "../tools/read-skill.ts";
import {
	getDefaultSandboxProfile,
	type SandboxProfile,
	setDefaultSandboxProfile,
} from "../tools/sandbox.ts";
import { killAllTrackedChildren } from "../tools/shell.ts";
import { EohController } from "./eoh/controller.ts";
import { mapAgentEvent } from "../runtime/event-mapping.ts";
import type { ParsedBridgeEvent } from "../runtime/events.ts";
import { LspManager } from "../developer-tools/lsp-manager.ts";
import { formatPluginResult } from "../runtime/plugin-result-formatter.ts";
import { createPostEditDiagnosticHooks } from "../developer-tools/post-edit-diagnostics.ts";
import {
	buildPluginRuntimeEnv,
	createHookTranscriptPath,
	eventLogPathFor,
	resolveWebSearchConfig,
} from "./bridge-environment.ts";
import {
	applyCompactionSettings,
	loadUserSettings,
} from "./bridge-settings.ts";
import {
	getProjectPromptDirs,
	getProjectSkillDirs,
} from "./resource-directories.ts";
export type EventCallback = (event: ParsedBridgeEvent) => void;
export type ErrorCallback = (err: Error) => void;

export function findJbPrompt(cwd: string): string | null {
	for (const candidate of [
		path.join(cwd, "jb.md"),
		path.join(cwd, "tui", "jb.md"),
	]) {
		try {
			return readFileSync(candidate, "utf8");
		} catch (error: unknown) {
			const code = (error as { code?: string }).code;
			if (code !== "ENOENT") throw error;
		}
	}
	return null;
}

// ── Bridge options ──────────────────────────────────────────────────────────────

export interface AgentBridgeOptions {
	baseUrl: string;
	model: string;
	models?: AgentModelConfig[];
	chatTemplate?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	executionProfile?: AgentConfig["executionProfile"];
	contextWindowTokens?: number;
	toolExecution?: AgentConfig["toolExecution"];
	runtimeHooksEnabled?: boolean;
	permissionMode?: PermissionMode;
	permissionRules?: PermissionRules;
	steeringInterrupt?: boolean;
	maxTotalTokens?: number;
	mcpEager?: boolean;
	tools?: Tool[];
	cwd?: string;
	systemPrompt?: string;
	webSearch?: Partial<WebSearchConfig>;
	// Safeguard options: default OFF (match pi's trust-model approach).
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	continuationEnabled?: boolean;
	postEditDiagnostics?: boolean;
	rtkProxyEnabled?: boolean;
	autoRetryEnabled?: boolean;
	maxRetries?: number;
	retryBaseDelayMs?: number;
	turnTimeoutMs?: number;
	cacheSize?: number;
	cacheTtlMs?: number;
	streamOptions?: AgentConfig["streamOptions"];
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	truncation?: TruncationConfig;
	/** Whether project-local configuration, skills, hooks, and agents may load. */
	projectTrusted?: boolean;
}

// ── AgentCoreBridge ─────────────────────────────────────────────────────────────

export class AgentCoreBridge {
	private config: AgentConfig;
	private backend: OpenAIBackend;
	private harness: AgentHarness | null = null;
	private callbacks: EventCallback[] = [];
	private errorCb: ErrorCallback | null = null;
	private running = false;
	private sendTail: Promise<void> = Promise.resolve();
	private pendingAutoContinue = false;
	private skillActivation = new SkillActivationSession();
	private cwd: string;
	private defaultTools: Tool[];
	private mcpManager = new McpManager();
	private mcpLoaded = false;
	private mcpLoadPromise: Promise<void> | null = null;
	private mcpServerCount = 0;
	private mcpErrors: string[] = [];
	private mcpToolNames = new Set<string>();
	private mcpSystemContext = "";
	private baseSystemPrompt: string;
	private additionalSystemPrompt?: string;
	private pluginSystemContext = "";
	private skillsContext: string | null = null;
	private skillsInjected: boolean = false;
	private promptsInjected: boolean = false;
	private sessionId =
		`tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
	private transcriptPath = "";
	private startupHooksRan = false;
	private startupHookResult: PluginCommandResult | null = null;
	private startupPluginCount = 0;
	private contextTokens = 0;
	private contextMaxTokens?: number;
	private configPath: string | null;
	private mcpEager: boolean;
	private postEditDiagnosticsEnabled: boolean;
	private lspManagerEnabled: boolean;
	private lspManager: LspManager;
	private agentDefs: AgentDefinition[] = [];
	private loadedSkills: Skill[] = [];
	private loadedPrompts: Prompt[] = [];
	private readonly projectTrusted: boolean;
	private enabledPluginRoots: Array<{ name: string; installPath: string }> = [];
	private permissionManager: PermissionManager;
	// Pending interactive permission requests, keyed by tool_call_id; resolved
	// by respondToPermission() from the UI.
	private permissionResolvers = new Map<
		string,
		(decision: "allow" | "deny" | "always") => void
	>();

	// Pending interactive question requests, keyed by question_id; resolved
	// by respondToQuestion() from the UI.
	private questionResolvers = new Map<
		string,
		{ allow: (answer: string) => void; deny: () => void }
	>();

	// ── EoH (Evolution of Heuristics) ─────────────────────────────────
	private eohController!: EohController;

	/** EoH command: /eoh <file.py> [generations] | stop | status | best | reset */
	eohCommand(raw: string): string {
		return this.eohController.command(raw);
	}

	constructor(
		opts: AgentBridgeOptions = {
			baseUrl: "http://localhost:8080",
			model: "",
		},
	) {
		this.cwd = opts.cwd || process.cwd();
		this.eohController = new EohController({
			cwd: this.cwd,
			emit: (event) => this.emit(event),
			getBaseUrl: () => this.config.baseUrl,
			getCurrentModel: () => this.getCurrentModel(),
		});
		this.projectTrusted = opts.projectTrusted === true;
		this.configPath = this.projectTrusted ? findLogicianConfig(this.cwd) : null;
		configurePluginRuntimeEnv(buildPluginRuntimeEnv(opts));
		this.mcpEager =
			process.env.LOGICIAN_MCP === "0" ? false : opts.mcpEager !== false;
		this.postEditDiagnosticsEnabled =
			process.env.LOGICIAN_POST_EDIT_DIAGNOSTICS === "0"
				? false
				: opts.postEditDiagnostics !== false;
		// LSP config from settings.json.
		let lspEnabled = true;
		let lspTimeoutMs = 2_000;
		const serverOverrides: Record<
			string,
			{ command: string; args: string[]; languageId: string }
		> = {};
		if (this.projectTrusted) {
			try {
				const resolved = loadLogicianConfig(this.cwd);
				const lspCfg = resolved.config.lsp;
				if (lspCfg !== undefined) {
					if (lspCfg.enabled === false) lspEnabled = false;
					if (lspCfg.timeoutMs !== undefined && lspCfg.timeoutMs > 0)
						lspTimeoutMs = lspCfg.timeoutMs;
					if (lspCfg.serverOverrides) {
						Object.assign(serverOverrides, lspCfg.serverOverrides);
					}
				}
			} catch {
				// Config load failure is non-fatal; LSP stays on with defaults.
			}
		}
		this.lspManager = new LspManager(this.cwd, {
			timeoutMs: lspTimeoutMs,
			servers:
				Object.keys(serverOverrides).length > 0 ? serverOverrides : undefined,
		});
		this.lspManagerEnabled = lspEnabled;
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		const defaultWebSearch = resolveWebSearchConfig();
		const webSearch = {
			baseUrl: opts.webSearch?.baseUrl || defaultWebSearch.baseUrl,
			maxResults: opts.webSearch?.maxResults ?? defaultWebSearch.maxResults,
		};
		this.defaultTools = opts.tools?.length
			? opts.tools
			: createDefaultTools({ webSearch });
		this.backend = new OpenAIBackend({
			baseUrl: opts.baseUrl,
			model: opts.model,
			chatTemplate: opts.chatTemplate,
		});

		this.additionalSystemPrompt = opts.systemPrompt;
		this.baseSystemPrompt = this.buildBaseSystemPrompt();

		this.permissionManager = new PermissionManager({
			mode: opts.permissionMode ?? "acceptAll",
			rules: opts.permissionRules,
		});

		this.config = {
			baseUrl: opts.baseUrl,
			model: opts.model,
			models: opts.models,
			systemPrompt: this.baseSystemPrompt,
			tools: this.defaultTools,
			webSearch,
			cwd: this.cwd,
			maxIterations: opts.maxIterations || 30,
			executionProfile: opts.executionProfile,
			temperature: opts.temperature,
			maxTokens: opts.maxTokens,
			// Parallel scheduling is transparent to the model. Tools that require
			// exclusivity declare executionMode: "sequential" and become barriers.
			toolExecution: opts.toolExecution ?? "parallel",
			contextWindowTokens:
				envNumber("LOGICIAN_CONTEXT_WINDOW") ||
				envNumber("LOGICIAN_CTX_SIZE") ||
				opts.contextWindowTokens,
			runtimeHooksEnabled:
				opts.runtimeHooksEnabled ?? process.env.LOGICIAN_HOOKS !== "0",
			hookSessionId: this.sessionId,
			hookTranscriptPath: this.transcriptPath,
			eventLogPath: eventLogPathFor(this.transcriptPath),
			steeringInterrupt: opts.steeringInterrupt,
			maxTotalTokens: opts.maxTotalTokens,
			permissions: this.permissionManager,
			guardsEnabled: opts.guardsEnabled,
			duplicateGuardEnabled: opts.duplicateGuardEnabled,
			failureGuardEnabled: opts.failureGuardEnabled,
			continuationEnabled: opts.continuationEnabled,
			rtkProxyEnabled: opts.rtkProxyEnabled,
			autoRetryEnabled: opts.autoRetryEnabled,
			maxRetries: opts.maxRetries,
			retryBaseDelayMs: opts.retryBaseDelayMs,
			turnTimeoutMs: opts.turnTimeoutMs,
			cacheSize: opts.cacheSize,
			cacheTtlMs: opts.cacheTtlMs,
			streamOptions: opts.streamOptions,
			allowedPaths: opts.allowedPaths,
			allowAllPaths: opts.allowAllPaths,
			truncation: opts.truncation,
			onPermissionRequest: (ctx) =>
				new Promise((resolve) => {
					this.permissionResolvers.set(ctx.toolCallId, resolve);
					this.emit({
						type: "permission_request",
						tool_name: ctx.toolName,
						tool_call_id: ctx.toolCallId,
						args: ctx.args,
					});
				}),
			onQuestionRequest: (ctx) =>
				new Promise<string>((resolve) => {
					const questionId = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
					this.questionResolvers.set(questionId, {
						allow: resolve,
						deny: () => resolve("__dismissed__"),
					});
					this.emit({
						type: "question_request",
						question_id: questionId,
						questions: ctx.questions,
					});
				}),
			hooks: createPostEditDiagnosticHooks(
				this.cwd,
				() => this.postEditDiagnosticsEnabled,
				this.lspManager,
				{
					allowedPaths: opts.allowedPaths,
					allowAllPaths: opts.allowAllPaths,
				},
			),
			turnEndCallback: (turnId: string) => {
				this.emit({ type: "turn_end", turn_id: turnId, message: "" });
			},
			onEvent: (event: AgentEvent) => {
				if (event.type === "context_update") {
					this.contextTokens = event.tokens;
					this.contextMaxTokens = event.maxTokens;
				}
				const mapped = mapAgentEvent(event);
				if (mapped) {
					this.emit(mapped);
				}
			},
		};

		onTodosChanged((todos) => {
			this.emit({ type: "todos", todos });
		});
	}

	// ── Event registration ─────────────────────────────────────────────────

	on(callback: EventCallback): () => void {
		this.callbacks.push(callback);
		return () => {
			this.callbacks = this.callbacks.filter((cb) => cb !== callback);
		};
	}

	onError(callback: ErrorCallback): void {
		this.errorCb = callback;
	}

	/** Surface an asynchronous caller-side failure through the normal UI path. */
	reportError(error: unknown): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.emit({
			type: "notice",
			level: "error",
			label: "Error",
			text: normalized.message,
		});
		this.errorCb?.(normalized);
	}

	private emit(event: ParsedBridgeEvent): void {
		for (const cb of this.callbacks) {
			try {
				cb(event);
			} catch (_e: unknown) {
				// Don't let a bad handler kill the bridge
			}
		}
	}

	// ── High-level commands ──────────────────────────────────────────────

	async sendMessage(message: string): Promise<void> {
		// A message submitted while a turn is in flight steers the running
		// turn instead of starting a second concurrent run. Route through
		// steer() so the queue update reaches the UI.
		if (this.running && this.harness) {
			this.steer(message);
			this.emit({ type: "steered", message });
			return;
		}
		const run = this.sendTail.then(() => this.runMessage(message));
		// Keep the queue usable after a failed startup/provider boundary while
		// returning the original rejection to this caller.
		this.sendTail = run.catch(() => {});
		return run;
	}

	private async runMessage(message: string): Promise<void> {
		this.running = true;
		const turnId = `turn_${Date.now()}`;
		let persistentSystemPrompt: string | undefined;
		let turnActivations: ReturnType<typeof selectSkillsForPrompt> = [];
		try {
			await this.runStartupHooksOnce();
			// Eager discovery is a real first-turn barrier so the provider's tool
			// snapshot includes MCP capabilities. Deferred mode remains non-blocking.
			if (!this.mcpLoaded) {
				const mcpLoad = this.loadMcpToolsOnce();
				if (this.mcpEager) {
					await mcpLoad;
				} else {
					void mcpLoad.catch((error) => this.reportError(error));
				}
			}
			// Reuse one harness across messages so conversation history (and thus
			// "continue" / "go on" follow-ups) persists. Created lazily once.
			const harness = this.ensureHarness();
			const activations = this.skillActivation.select(
				this.loadedSkills,
				message,
			);
			turnActivations = activations;
			if (activations.length) {
				persistentSystemPrompt = this.config.systemPrompt;
				harness.setSystemPrompt(
					`${persistentSystemPrompt}\n\n${formatActivatedSkills(activations)}`,
				);
				this.emit({
					type: "notice",
					level: "info",
					label: "Skills",
					text: formatSkillActivationNotice(activations),
				});
			}

			this.emit({ type: "turn_start", turn_id: turnId });
			await harness.prompt(message);
		} catch (err: unknown) {
			const error = err as Error;
			// Emit a visible error notice so the user sees connection/server
			// failures in the transcript rather than only in the console.
			this.emit({
				type: "notice",
				level: "error",
				label: "Error",
				text: error.message,
			});
			this.errorCb?.(error);
			throw error;
		} finally {
			if (persistentSystemPrompt !== undefined) {
				this.harness?.setSystemPrompt(persistentSystemPrompt);
			}
			this.running = false;
			this.publishContextUsage();
			this.emit({ type: "turn_end", turn_id: turnId, message: "" });
			// Keep the harness alive to retain history across turns.
			this.emit({ type: "phase", state: "ready" });
			if (this.pendingAutoContinue) {
				this.pendingAutoContinue = false;
				this.skillActivation.continueWith(turnActivations);
				void this.sendMessage("continue");
			}
		}
	}

	// Lazily build the singleton harness and wire its UI callbacks.
	private ensureHarness(): AgentHarness {
		if (!this.harness) {
			this.harness = new AgentHarness({
				config: this.config,
				backend: this.backend,
				cwd: this.config.cwd,
				maxIterations: this.config.maxIterations,
			});
			// Harness owns the queue state; mirror every change to the UI.
			this.harness.setOnQueueChange(() => this._emitQueueUpdate());
			// Surface harness phase transitions the loop can't see — compaction
			// and branch_summary. turn/idle are already covered by the
			// streaming/ready phase emits around prompt().
			this.harness.setOnPhaseChange((phase) => this._emitHarnessPhase(phase));
			// Autonomous continuation: when the harness settles with pending
			// nextTurn messages, auto-trigger the next prompt so the agent
			// continues without requiring user input. The nextTurn items are
			// injected before the trigger message by the transformContext hook.
			this.harness.setOnSettled((nextTurnCount) => {
				if (nextTurnCount > 0) this.pendingAutoContinue = true;
			});
			// Emit a save_point event after every completed turn so the UI can
			// show autosave status and know a rewind point exists.
			this.harness.setOnSavePoint(() => {
				this.emit({ type: "save_point" });
			});
			// Apply compaction settings from user settings (~/.logician/settings.json).
			const userSettings = loadUserSettings();
			applyCompactionSettings(this.harness, userSettings);
		}
		return this.harness;
	}

	/**
	 * Replace the harness conversation with restored session history (resume /
	 * session switch), so the model continues with the restored context instead
	 * of starting cold. Pass [] to clear (new session). No-op while a turn is
	 * running (the harness rejects structural ops mid-turn).
	 */
	restoreHistory(messages: Message[]): boolean {
		try {
			this.ensureHarness().setHistory(messages);
			this.publishContextUsage();
			return true;
		} catch (_e: unknown) {
			return false;
		}
	}

	// ── Session-level steering queue (Pi-style) ────────────────────────
	// Tracks pending steering/follow-up messages for UI display.
	// Items are removed when consumed by the loop (detected via
	// message_start events emitted before assistant responses).

	/** Inject guidance into the running turn (drained at the next save point). */
	steer(message: string): void {
		// Harness emits onQueueChange → _emitQueueUpdate, so no local mirror.
		this.harness?.steer(message);
	}

	/** Queue a message for after the current turn completes. */
	followUp(message: string): void {
		this.harness?.followUp(message);
	}

	/** Queue a message before the next user prompt; survives abort. */
	nextTurn(message: string): void {
		this.harness?.nextTurn(message);
	}

	/** Controls how queued steering messages are drained. */
	setSteeringMode(mode: "all" | "one-at-a-time"): void {
		this.config.steeringQueueMode = mode;
		this.harness?.setSteeringMode(mode);
	}

	/** Toggle mid-stream steering interrupt (cut the stream vs. queue). */
	setSteeringInterrupt(enabled: boolean): void {
		this.config.steeringInterrupt = enabled;
	}

	getSteeringInterrupt(): boolean {
		return this.config.steeringInterrupt === true;
	}

	/** Return config snapshot for external LLM calls (goal evaluator, etc.). */
	getConfig(): { baseUrl: string; model: string; rtkProxyEnabled?: boolean } {
		return {
			baseUrl: this.config.baseUrl,
			model: this.config.model,
			rtkProxyEnabled: this.config.rtkProxyEnabled,
		};
	}

	/** Controls how queued follow-up messages are drained. */
	setFollowUpMode(mode: "all" | "one-at-a-time"): void {
		this.config.followUpQueueMode = mode;
		this.harness?.setFollowUpMode(mode);
	}

	private _emitQueueUpdate(): void {
		const q = this.harness?.getQueues() ?? {
			steering: [],
			followUp: [],
			nextTurn: [],
		};
		this.emit({
			type: "queue_update",
			steering: q.steering,
			followUp: q.followUp,
			nextTurn: q.nextTurn,
		});
	}

	// Map harness structural phases to UI phase states. The "turn" phase is
	// already covered by the streaming/ready emits around prompt() (and is
	// skipped here to avoid clobbering the loop's finer-grained states). The
	// background phases — compaction, branch_summary — are otherwise invisible
	// to the UI, so surface them and restore "ready" when they return to idle.
	private _emitHarnessPhase(phase: HarnessPhase): void {
		// Don't touch UI phase while a turn drives its own streaming/ready cycle.
		if (phase === "turn" || this.running) return;
		const state =
			phase === "compaction"
				? "compacting"
				: phase === "branch_summary"
					? "branching"
					: "ready";
		this.emit({ type: "phase", state });
	}

	/** Get current steering messages (read-only). */
	getSteeringMessages(): string[] {
		return this.harness?.getQueues().steering ?? [];
	}

	/** Interrupt the current provider step and process queued steering immediately. */
	flushSteeringNow(): number {
		if (!this.running || !this.harness) return 0;
		return this.harness.flushSteeringNow();
	}

	/** Get current follow-up messages (read-only). */
	getFollowUpMessages(): string[] {
		return this.harness?.getQueues().followUp ?? [];
	}

	/** Get current next-turn messages (read-only). */
	getNextTurnMessages(): string[] {
		return this.harness?.getQueues().nextTurn ?? [];
	}

	/** Clear all pending messages, returns the messages that were cleared. */
	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return (
			this.harness?.clearQueues() ?? {
				steering: [],
				followUp: [],
				nextTurn: [],
			}
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.harness?.dropQueuedMessage(displayIndex);
	}

	/** Abort: clear steering/follow-up queues (preserves nextTurn). */
	async abort(): Promise<void> {
		// harness.abort() clears steering/follow-up and emits onQueueChange.
		await this.harness?.abort();
	}

	/** Execute a slash command (sends as chat message to the agent). */
	sendSlash(raw: string): void {
		const trimmed = raw.trim();
		if (trimmed === "/steer-now") {
			const count = this.flushSteeringNow();
			this.emit({
				type: "notice",
				level: count > 0 ? "info" : "warn",
				label: "Steering",
				text:
					count > 0
						? `Processing ${count} queued steering message${count === 1 ? "" : "s"} now.`
						: "No queued steering messages to process.",
			});
			return;
		}
		if (trimmed === "/queue") {
			const steering = this.getSteeringMessages();
			const followUp = this.getFollowUpMessages();
			const rows = [
				...steering.map((message) => `▸ ${message}`),
				...followUp.map((message) => `↳ ${message}`),
			];
			this.emit({
				type: "notice",
				level: "info",
				label: "Queue",
				text: rows.length
					? rows.map((row, index) => `${index + 1}. ${row}`).join("\n")
					: "Queue is empty.",
			});
			return;
		}
		if (trimmed === "/queue-clear") {
			const cleared = this.clearQueue();
			const count =
				cleared.steering.length +
				cleared.followUp.length +
				cleared.nextTurn.length;
			this.emit({
				type: "notice",
				level: "info",
				label: "Queue",
				text: `Cleared ${count} queued message${count === 1 ? "" : "s"}.`,
			});
			return;
		}
		if (trimmed === "/queue-drop" || trimmed.startsWith("/queue-drop ")) {
			const value = Number.parseInt(
				trimmed.slice("/queue-drop".length).trim(),
				10,
			);
			const removed =
				Number.isInteger(value) && value > 0
					? this.dropQueuedMessage(value - 1)
					: undefined;
			this.emit({
				type: "notice",
				level: removed ? "info" : "warn",
				label: "Queue",
				text: removed ? `Removed: ${removed}` : "Usage: /queue-drop <number>",
			});
			return;
		}
		// /reload — reload settings, skills, extensions, and MCP config
		if (trimmed === "/reload") {
			this.reload().catch((err) => this.errorCb?.(err));
			return;
		}
		this.sendMessage(raw).catch((err) => this.errorCb?.(err));
	}

	// ── Reload ────────────────────────────────────────────────────────────

	/** Reload: restart the session (like Pi's /reload). */
	private async reload(): Promise<void> {
		// Stop any running turn
		this.cancel();
		this.running = false;

		// Drop the old harness — conversation starts fresh
		this.harness = null;

		// Generate a new session ID
		this.sessionId = `tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);

		// Reset state that is per-session
		this.loadedSkills = [];
		this.skillsContext = null;
		this.skillsInjected = false;
		this.loadedPrompts = [];
		this.promptsInjected = false;
		this.startupHooksRan = false;
		this.startupHookResult = null;
		this.pluginSystemContext = "";
		this.skillActivation.reset();

		// Send reload confirmation (not via sendMessage to avoid starting a turn)
		this.emit({
			type: "turn_end",
			turn_id: "reload",
			message: "**Session reloaded.**",
		});
	}

	// ── Skill invocation ───────────────────────────────────────────────

	/** Skills discovered at startup (for /<skill-name> completion). */
	getSkills(): Skill[] {
		return this.loadedSkills;
	}

	/**
	 * Invoke a skill by name as a user prompt: sends the skill's full body
	 * (plus any arguments) to the agent. Returns false for unknown names so the
	 * caller can fall back to normal slash handling.
	 */
	invokeSkill(name: string, args: string): boolean {
		const skill = findSkillByName(this.loadedSkills, name);
		if (!skill) return false;
		const trimmedArgs = args.trim();
		// Claude Code command convention: $ARGUMENTS in the body is replaced with
		// the user's arguments instead of appending an instructions line.
		const substitutes = skill.content.includes("$ARGUMENTS");
		const effective = substitutes
			? {
					...skill,
					content: skill.content.replaceAll("$ARGUMENTS", trimmedArgs),
				}
			: skill;
		const message = formatSkillInvocation(
			effective,
			trimmedArgs && !substitutes
				? `User arguments for this skill invocation: ${trimmedArgs}`
				: undefined,
		);
		this.sendMessage(message).catch((err) => this.errorCb?.(err));
		return true;
	}

	/** Prompts discovered at startup (for /<prompt-name> completion). */
	getPrompts(): Prompt[] {
		return this.loadedPrompts;
	}

	/**
	 * Invoke a prompt by name as a user message: sends the prompt's body
	 * (with $ARGUMENTS substituted, or arguments appended) directly — no XML
	 * wrapping, unlike invokeSkill, since a prompt is meant to read exactly as
	 * if the user had typed it. Returns false for unknown names so the caller
	 * can fall back to normal slash handling.
	 */
	invokePrompt(name: string, args: string): boolean {
		const prompt = findPromptByName(this.loadedPrompts, name);
		if (!prompt) return false;
		const trimmedArgs = args.trim();
		const substitutes = prompt.content.includes("$ARGUMENTS");
		const message = substitutes
			? prompt.content.replaceAll("$ARGUMENTS", trimmedArgs)
			: trimmedArgs
				? `${prompt.content}\n\n${trimmedArgs}`
				: prompt.content;
		this.sendMessage(message).catch((err) => this.errorCb?.(err));
		return true;
	}

	// ── Permissions ────────────────────────────────────────────────────

	/** Answer a pending permission_request. Returns false for unknown ids. */
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

	/** True while a permission_request awaits a decision. */
	hasPendingPermission(): boolean {
		return this.permissionResolvers.size > 0;
	}

	// ── Interactive questions ────────────────────────────────────────────

	/**
	 * Register a pending question and emit it to the UI. Returns the question id
	 * so the agent can track which question it asked. Call respondToQuestion() to
	 * resolve it.
	 */
	askQuestion(
		question: string,
		choices: Array<{ value: string; label: string }>,
	): string {
		const questionId = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
		this.questionResolvers.set(questionId, {
			allow: (_ans: string) => {},
			deny: () => {},
		});
		this.emit({
			type: "question_request",
			question_id: questionId,
			questions: [{ id: "answer", question, choices }],
		});
		return questionId;
	}

	/**
	 * Answer a pending question by id. The answer is forwarded to the agent's
	 * resolver. Returns false if the question id is unknown.
	 */
	respondToQuestion(questionId: string, answer: string): boolean {
		const resolver = this.questionResolvers.get(questionId);
		if (!resolver) return false;
		this.questionResolvers.delete(questionId);
		resolver.allow(answer);
		return true;
	}

	/** True while a question_request awaits an answer. */
	hasPendingQuestion(): boolean {
		return this.questionResolvers.size > 0;
	}

	/** Deny every pending permission request (abort / shutdown). */
	private denyPendingPermissions(): void {
		for (const [id, resolve] of this.permissionResolvers) {
			this.permissionResolvers.delete(id);
			resolve("deny");
		}
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

	// ── Sandbox mode ─────────────────────────────────────────────────────
	// Default profile applied by the sandbox tool when a call omits one.
	// Cycled by the UI (Ctrl+K); "none" is exposed to the user as "off".

	private static readonly SANDBOX_CYCLE: SandboxProfile[] = [
		"none",
		"code",
		"full",
	];

	getSandboxMode(): SandboxProfile {
		return getDefaultSandboxProfile();
	}

	setSandboxMode(mode: SandboxProfile): void {
		setDefaultSandboxProfile(mode);
		this.emit({
			type: "notice",
			level: "info",
			label: "Sandbox",
			text: `mode: ${mode === "none" ? "off" : mode}`,
		});
	}

	cycleSandboxMode(): SandboxProfile {
		const cycle = AgentCoreBridge.SANDBOX_CYCLE;
		const currentIndex = cycle.indexOf(this.getSandboxMode());
		const next = cycle[(currentIndex + 1) % cycle.length];
		this.setSandboxMode(next);
		return next;
	}

	// ── Model cycling ──────────────────────────────────────────────────

	/** Get current model name. */
	getCurrentModel(): string {
		return this.harness?.getModel() ?? this.config.model ?? "";
	}

	/** Get current base URL. */
	getCurrentBaseUrl(): string {
		return this.config.baseUrl;
	}

	/** Resolve the URL for a given model name. */
	getModelUrl(modelName: string): string {
		const models = this.config.models;
		if (models) {
			const found = models.find((m) => m.model === modelName);
			if (found?.url) {
				return found.url;
			}
		}
		return this.config.baseUrl;
	}

	/** Get all available models. */
	getModels(): string[] {
		return this.config.models?.length
			? this.config.models.map((model) => model.model)
			: [this.getCurrentModel()];
	}

	getModelOptions(): Array<{
		key: string;
		name: string;
		model: string;
		url: string;
		active: boolean;
	}> {
		const configured = this.config.models ?? [];
		if (configured.length === 0) {
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
		return configured.map((option, index) => {
			const url = option.url || this.config.baseUrl;
			return {
				key: `${index}:${option.name}`,
				name: option.name,
				model: option.model,
				url,
				active:
					option.model === this.getCurrentModel() &&
					url === this.getCurrentBaseUrl(),
			};
		});
	}

	setModelOption(key: string): { model: string; url: string } | null {
		const option = this.getModelOptions().find(
			(candidate) => candidate.key === key,
		);
		if (!option) return null;
		this.config.model = option.model;
		this.config.baseUrl = option.url;
		this.harness?.setModelEndpoint(option.model, option.url);
		return { model: option.model, url: option.url };
	}

	/** Cycle to the next model. Returns the new model name. */
	cycleModel(direction: "forward" | "backward" = "forward"): string | null {
		return this.harness?.cycleModel(direction) ?? null;
	}

	/** Set the model list for cycling. */
	setModels(models: AgentModelConfig[]): void {
		this.config.models = models;
		// If the harness is already built, update its config directly.
		if (this.harness) {
			this.harness.setModels(models);
		}
	}

	/** Change the current model. */
	setModel(modelId: string): void {
		this.config.model = modelId;
		if (this.harness) {
			this.harness.setModel(modelId);
		}
	}

	async getState(): Promise<Record<string, unknown>> {
		// Status is a snapshot, not a synchronization barrier for external MCP
		// transports. The manager UI provides explicit awaited refresh operations.
		if (!this.mcpLoaded && !this.mcpLoadPromise) {
			void this.loadMcpToolsOnce().catch((error) => this.reportError(error));
		}
		this.contextTokens = this.measureContextTokens();
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this.defaultTools.map((t) => t.name);
		const state = {
			agent_name: "logician",
			model: this.config.model,
			base_url: this.config.baseUrl,
			web_search_url: this.config.webSearch?.baseUrl || "",
			web_search_enabled: toolNames.includes("web_search"),
			tools: toolNames,
			mcp_servers: this.mcpServerCount,
			mcp_tools: this.mcpToolNames.size,
			mcp_errors: this.mcpErrors,
			context_tokens: this.contextTokens,
			context_max_tokens: this.contextMaxTokens,
			runtime_state: this.harness?.runtimeState ?? {
				phase: "idle",
				isStreaming: false,
				pendingToolCalls: [],
				abortRequested: false,
			},
			config_path: this.configPath || "",
			connected: true,
		};
		return state;
	}

	async getPlugins(): Promise<Record<string, unknown>[]> {
		const result = await runPluginBackend("list", []);
		return result.plugins || [];
	}

	async getPluginSnapshot(): Promise<PluginCommandResult> {
		return runPluginBackend("list", []);
	}

	async getMcpSnapshot(): Promise<McpSnapshotResult> {
		const snapshot = await this.mcpManager.getSnapshot(this.cwd);
		// MCP config handled by snapshot
		return snapshot;
	}

	async setMcpServerEnabled(
		serverName: string,
		enabled: boolean,
	): Promise<McpToggleResult> {
		const result = await this.mcpManager.setServerEnabled(
			serverName,
			enabled,
			this.cwd,
		);
		// MCP config handled by result
		return result;
	}

	async setPluginEnabled(
		pluginId: string,
		enabled: boolean,
	): Promise<PluginCommandResult> {
		const result = await runPluginBackend(enabled ? "enable" : "disable", [
			pluginId,
		]);
		if (result.status !== "error") {
			this.startupHooksRan = false;
			await this.runStartupHooksOnce();
		}
		return result;
	}

	async runPluginCommand(input: string): Promise<string> {
		const parts = splitPluginArgs(input);
		const action = (parts.shift() || "list").toLowerCase();

		if (action === "help" || action === "-h" || action === "--help") {
			return [
				"# Plugins",
				"Usage: /plugins [list|enable|disable|install|remove|update|deps|info|hooks|run-hooks]",
				"",
				"- /plugins list",
				"- /plugins enable <plugin>",
				"- /plugins disable <plugin>",
				"- /plugins hooks [startup|clear|compact|Stop|PreToolUse|PostToolUse|SessionEnd]",
				"- /plugins run-hooks [startup|clear|compact]",
			].join("\n");
		}

		const backendAction = action === "refresh" ? "run-hooks" : action;
		const result = await runPluginBackend(backendAction, parts);

		if (backendAction === "run-hooks" && result.status !== "error") {
			this.applyPluginHookContext(result);
		}

		return formatPluginResult(backendAction, result);
	}

	setThinkingLevel(level: string): void {
		this.config.thinkingLevel = level as
			| "off"
			| "minimal"
			| "low"
			| "medium"
			| "high"
			| "xhigh";
		// Also update the backend's default so future turns pick it up.
		(this.backend as OpenAIBackend).setDefaultThinkingLevel(
			level as "off" | "minimal" | "low" | "medium" | "high" | "xhigh",
		);
	}

	setTemperature(temperature: number): void {
		this.config.temperature = temperature;
		this.harness?.setTemperature(temperature);
	}

	setInferenceMode(mode: string): void {
		this.config.inferenceMode = mode as typeof this.config.inferenceMode;
		this.harness?.setInferenceMode(mode);
	}

	setMaxTokens(maxTokens: number): void {
		this.config.maxTokens = maxTokens;
		this.harness?.setMaxTokens(maxTokens);
	}

	setMaxIterations(maxIterations: number): void {
		this.config.maxIterations = maxIterations;
		this.harness?.setMaxIterations(maxIterations);
	}

	setExecutionProfile(profile: "autonomous" | "minimal"): void {
		this.config.executionProfile = profile;
		this.harness?.setExecutionProfile(profile);
	}

	setRuntimeToggle(
		key:
			| "guardsEnabled"
			| "proactiveCompactionEnabled"
			| "postEditDiagnostics"
			| "rtkProxyEnabled",
		enabled: boolean,
	): void {
		if (key === "postEditDiagnostics") {
			this.postEditDiagnosticsEnabled = enabled;
			return;
		}
		this.config[key] = enabled;
		if (key === "proactiveCompactionEnabled") {
			this.harness?.enableAutoCompaction(enabled);
		}
	}

	getSettingsText(): string {
		return [
			"Runtime settings",
			`  Model: ${this.config.model}`,
			`  Temperature: ${this.config.temperature ?? 0.5}`,
			`  Max tokens: ${this.config.maxTokens ?? 4096}`,
			`  Max iterations: ${this.config.maxIterations ?? 30}`,
			`  Context window: ${this.config.contextWindowTokens ?? "unset"}`,
			`  Thinking: ${this.config.thinkingLevel ?? "off"}`,
			`  Permission mode: ${this.getPermissionMode()}`,
			`  Guards: ${this.config.guardsEnabled ? "on" : "off"}`,
			`  Compaction: ${this.config.proactiveCompactionEnabled ? "on" : "off"}`,
			`  Post-edit diagnostics: ${this.postEditDiagnosticsEnabled ? "on" : "off"}`,
			`  RTK proxy: ${this.config.rtkProxyEnabled ? "on" : "off"}`,
	].join("\n");
	}

	/** Return structured settings data for the overlay UI. */
	getSettingsData(): {
		model: string;
		temperature: number;
		maxTokens: number;
		maxIterations: number;
		thinkingLevel: string;
		inferenceMode: string;
		permissionMode: string;
		executionProfile: string;
		guardsEnabled: boolean;
		proactiveCompactionEnabled: boolean;
		postEditDiagnostics: boolean;
		rtkProxyEnabled: boolean;
	} {
		return {
			model: this.config.model,
			temperature: this.config.temperature ?? 0.5,
			maxTokens: this.config.maxTokens ?? 4096,
			maxIterations: this.config.maxIterations ?? 30,
			thinkingLevel: this.config.thinkingLevel ?? "off",
			inferenceMode: this.config.inferenceMode ?? "instruct-general",
			permissionMode: this.getPermissionMode(),
			executionProfile: this.config.executionProfile ?? "autonomous",
			guardsEnabled: this.config.guardsEnabled ?? false,
			proactiveCompactionEnabled:
				this.config.proactiveCompactionEnabled ?? false,
			postEditDiagnostics: this.postEditDiagnosticsEnabled,
			rtkProxyEnabled: this.config.rtkProxyEnabled ?? false,
		};
	}

	reset(): void {
		// Reset tool state and conversation
		void this.fireSessionEnd("reset");
		// Drop the persisted harness so history starts fresh.
		this.harness?.clearHistory();
		this.harness = null;
		this.sessionId = `tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Reset skill/prompt injection state
		this.skillsContext = null;
		this.skillsInjected = false;
		this.promptsInjected = false;
		this.startupHooksRan = false;
		this.pluginSystemContext = "";
		this.skillActivation.reset();
		this.rebuildBaseSystemPrompt();
		this.contextTokens = 0;
		this.publishContextUsage();
		this.emit({
			type: "turn_end",
			turn_id: "reset",
			message: "Tool state reset.",
		});
	}

	cancel(): void {
		// A turn blocked on an approval must unblock to abort cleanly.
		this.denyPendingPermissions();
		void this.harness?.abort().catch((error) => this.errorCb?.(error));
	}

	/** Manual context compaction. Returns { tokensSaved, tokensBefore, tokensAfter } or null if nothing to compact. */
	async compact(): Promise<{
		tokensSaved: number;
		tokensBefore: number;
		tokensAfter: number;
	} | null> {
		if (!this.harness) return null;
		const saved = await this.harness.compact();
		if (saved === null) return null;
		// Re-emit context update with new token count
		const messages = this.harness.messages;
		const after = estimateChatPayloadTokens(messages);
		const before = after + saved;
		this.contextTokens = after;
		this.emit({
			type: "compaction",
			reason: "manual",
			tokens_before: before,
			tokens_after: after,
		} as ParsedBridgeEvent);
		return { tokensSaved: saved, tokensBefore: before, tokensAfter: after };
	}

	// ── Conversation branching ─────────────────────────────────────────────

	/** Fork the conversation; returns the new branch id, or null if no harness. */
	fork(): string | null {
		return this.harness?.fork() ?? null;
	}

	/**
	 * Summarize the active branch and merge it back into the parent. Returns the
	 * summary text, or null if nothing to summarize / no harness.
	 */
	async branchSummary(): Promise<string | null> {
		if (!this.harness) return null;
		const summary = await this.harness.branchSummary();
		// Token count changed (branch tail collapsed into one message).
		this.publishContextUsage();
		return summary;
	}

	/**
	 * Rewind to the checkpoint taken before the last prompt: restores the
	 * conversation AND the files that turn wrote via the write tools. Returns
	 * what was restored, or null when there is nothing to rewind / a turn is
	 * running.
	 */
	rewind(): { messages: number; filesRestored: number } | null {
		try {
			const restored = this.harness?.rewind() ?? null;
			if (restored !== null && this.harness) {
				this.publishContextUsage();
			}
			return restored;
		} catch (_e: unknown) {
			return null;
		}
	}

	/** Discard the active branch without merging. Returns true if one was discarded. */
	discardBranch(): boolean {
		const discarded = this.harness?.discardBranch() ?? false;
		if (discarded && this.harness) this.publishContextUsage();
		return discarded;
	}

	// ── State management ─────────────────────────────────────────────────

	async init(): Promise<Record<string, unknown>> {
		await this.runStartupHooksOnce();
		this.ensureHarness();
		if (this.mcpEager) {
			// Start discovery during initialization. The first user turn awaits
			// this same promise before taking its tool snapshot.
			void this.loadMcpToolsOnce().then(
				() => {
					this.emit({
						type: "notice",
						level: this.mcpErrors.length ? "warn" : "info",
						label: "MCP",
						text: `Loaded ${this.mcpServerCount} server(s).`,
					});
				},
				(error) => this.reportError(error),
			);
		}
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this.defaultTools.map((t) => t.name);
		const info: Record<string, unknown> = {
			agent_name: "logician",
			model: this.config.model,
			base_url: this.config.baseUrl,
			web_search_url: this.config.webSearch?.baseUrl || "",
			web_search_enabled: toolNames.includes("web_search"),
			mcp_deferred: !this.mcpLoaded && process.env.LOGICIAN_MCP !== "0",
			mcp_loading: this.mcpLoadPromise !== null && !this.mcpLoaded,
			tools: toolNames,
			mcp_servers_loaded: this.mcpServerCount,
			mcp_tools_loaded: this.mcpToolNames.size,
			mcp_errors: this.mcpErrors,
			context_tokens: this.contextTokens,
			context_max_tokens:
				this.contextMaxTokens || this.config.contextWindowTokens,
			runtime_state: this.harness?.runtimeState ?? {
				phase: "idle",
				isStreaming: false,
				pendingToolCalls: [],
				abortRequested: false,
			},
			config_path: this.configPath || "",
			hooks_enabled: this.config.runtimeHooksEnabled !== false,
			hook_transcript_path: this.config.hookTranscriptPath || "",
			startup_plugins_loaded: this.startupPluginCount,
			startup_plugins: this.enabledPluginRoots.map((plugin) => plugin.name),
			startup_hooks_loaded: this.startupHookResult?.hook_count || 0,
			startup_hook_contexts: this.startupHookResult?.additional_contexts || [],
			startup_hook_messages: this.startupHookResult?.context_messages || [],
			startup_hook_initial_message:
				this.startupHookResult?.initial_user_message || "",
			startup_hook_errors: this.startupHookResult?.errors || [],
			skills_injected: this.skillsInjected
				? this.loadedSkills.filter((skill) => !skill.disableModelInvocation)
						.length
				: 0,
			skills_visible: !!this.skillsContext,
			loaded_skills: this.loadedSkills.map((skill) => ({
				name: skill.name,
				slash_name: skill.slashName,
				description: skill.description,
				model_visible: !skill.disableModelInvocation,
			})),
		};
		// Explicitly signal ready so the TUI status bar doesn't get stuck in
		// streaming after init.
		this.emit({ type: "phase", state: "ready" });
		return info;
	}

	async stop(): Promise<void> {
		this.cancel();
		await this.fireSessionEnd("shutdown");
		this.lspManager.close();
		await this.mcpManager.close();
		killAllTrackedChildren();
		this.running = false;
	}

	isActive(): boolean {
		return this.running;
	}

	getMessages(): Message[] {
		return this.harness?.messages || [];
	}

	/** Return full context as formatted text for /context command. */
	getContext(): string {
		const msgs = this.getMessages();
		const contextTokens = this.measureContextTokens();
		this.contextTokens = contextTokens;

		const sourceMap = this.getContextSourceMap();
		const sourceLines = sourceMap.map(
			(zone) =>
				`- ${zone.name}: ~${zone.tokens} tokens${zone.detail ? ` — ${zone.detail}` : ""}`,
		);
		const lines: string[] = [
			"## Prompt source map",
			"",
			...sourceLines,
			"",
			"## Conversation",
			"",
		];
		if (!msgs.length) lines.push("No messages yet.");

		for (const msg of msgs) {
			const role = msg.role.toUpperCase();
			const ts = msg.timestamp ? new Date(msg.timestamp).toISOString() : "";
			const header = `[${role}]${ts ? ` ${ts}` : ""}`;

			if (msg.role === "tool" && msg.content) {
				// Tool result: show the originating tool name alongside its content.
				const callId = msg.tool_call_id || "";
				const name =
					msgs
						.find(
							(m) =>
								m.role === "assistant" &&
								m.tool_calls?.some((tc) => tc.id === callId),
						)
						?.tool_calls?.find((tc) => tc.id === callId)?.name || "tool";
				lines.push(`${header} (${name})\n${msg.content}`);
			} else if (msg.role === "assistant" && msg.tool_calls?.length) {
				// Assistant with tool calls
				lines.push(
					`${header}\n${msg.content || "(no content)"}\n\nTool calls:`,
				);
				for (const tc of msg.tool_calls) {
					lines.push(`  - ${tc.name}(${tc.arguments || ""})`);
				}
			} else {
				lines.push(`${header}\n${msg.content || ""}`);
			}
			lines.push("");
		}

		return `## Context (${msgs.length} messages, ~${contextTokens} tokens)\n\n${lines.join("\n")}`;
	}

	getContextSourceMap(): Array<{
		name: string;
		tokens: number;
		detail: string;
	}> {
		const messages = this.getMessages();
		const conversation = messages.filter((message) => message.role !== "tool");
		const toolEvidence = messages.filter((message) => message.role === "tool");
		const memory = this.harness?.getMemoryPrompt() ?? "";
		return [
			{
				name: "Base instructions",
				tokens: estimateTokens(this.baseSystemPrompt),
				detail: "system zone",
			},
			{
				name: "Plugin context",
				tokens: estimateTokens(this.pluginSystemContext),
				detail: "startup hooks",
			},
			{
				name: "Skill catalog",
				tokens: estimateTokens(this.skillsContext ?? ""),
				detail: `${this.loadedSkills.length} loaded`,
			},
			{
				name: "Memory",
				tokens: estimateTokens(memory),
				detail: memory ? "active" : "empty",
			},
			{
				name: "Conversation",
				tokens: conversation.length
					? estimateChatPayloadTokens(conversation)
					: 0,
				detail: `${conversation.length} messages`,
			},
			{
				name: "Tool evidence",
				tokens: toolEvidence.length
					? estimateChatPayloadTokens(toolEvidence)
					: 0,
				detail: `${toolEvidence.length} results`,
			},
		].filter((zone) => zone.tokens > 0 || zone.name === "Conversation");
	}

	/** Canonical size used by /context, /status, and the status bar. */
	private measureContextTokens(): number {
		const messages = this.getMessages();
		return messages.length > 0 ? estimateChatPayloadTokens(messages) : 0;
	}

	private publishContextUsage(): void {
		this.contextTokens = this.measureContextTokens();
		this.contextMaxTokens =
			this.contextMaxTokens || this.config.contextWindowTokens;
		this.emit({
			type: "context_update",
			tokens: this.contextTokens,
			max_tokens: this.contextMaxTokens,
			compacted: false,
		});
	}

	getTools(): ToolRegistry {
		const live = this.harness?.tools;
		if (live) return live;
		const registry = new ToolRegistry({
			cwd: this.config.cwd,
			allowedPaths: this.config.allowedPaths,
			allowAllPaths: this.config.allowAllPaths,
			cacheSize: this.config.cacheSize,
			cacheTtlMs: this.config.cacheTtlMs,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
		registry.registerMany(this.defaultTools);
		return registry;
	}

	private async loadMcpToolsOnce(): Promise<void> {
		if (this.mcpLoaded || process.env.LOGICIAN_MCP === "0") return;
		if (!this.mcpLoadPromise) {
			this.mcpLoadPromise = (async () => {
				const result = await this.mcpManager.load(
					this.config.cwd || process.cwd(),
					this.defaultTools.map((tool) => tool.name),
				);
				this.mcpServerCount = result.servers;
				this.mcpErrors = result.errors;
				this.mcpToolNames = new Set(
					result.tools.map((tool) => tool.name),
				);
				// Tool presence alone doesn't tell the model whether a missing
				// capability was never configured or failed to connect — surface
				// connection failures in the system prompt so it can explain a gap
				// instead of silently working around it or guessing.
				this.mcpSystemContext = result.errors.length
					? `<mcp-status>\n${result.errors.length} MCP server(s) failed to load:\n${result.errors.map((e) => `- ${e}`).join("\n")}\n` +
						"Tools from these servers are unavailable this session.\n</mcp-status>"
					: "";
				if (result.tools.length || this.mcpSystemContext) {
					const existing = new Set(this.defaultTools.map((tool) => tool.name));
					const newTools = result.tools.filter(
						(tool) => !existing.has(tool.name),
					);
					this.defaultTools = [...this.defaultTools, ...newTools];
					this.config.tools = this.defaultTools;
					this.harness?.setTools(this.defaultTools);
					this.rebuildBaseSystemPrompt();
				}
				this.mcpLoaded = true;
			})();
		}
		await this.mcpLoadPromise;
	}

	private rebuildBaseSystemPrompt(): void {
		this.baseSystemPrompt = this.buildBaseSystemPrompt();
		this.applyContextLayers();
	}

	/** Recombine base prompt + plugin/MCP/skills context layers into config.systemPrompt. */
	private applyContextLayers(): void {
		const contexts: string[] = [];
		if (this.pluginSystemContext) contexts.push(this.pluginSystemContext);
		if (this.mcpSystemContext) contexts.push(this.mcpSystemContext);
		if (this.skillsContext) contexts.push(this.skillsContext);
		this.config.systemPrompt = contexts.length
			? `${this.baseSystemPrompt}\n\n${contexts.join("\n\n")}`
			: this.baseSystemPrompt;
	}

	private buildBaseSystemPrompt(): string {
		const defaultPrompt = buildDefaultSystemPrompt(this.cwd, this.defaultTools, {
			loadProjectContext: this.projectTrusted,
		});
		return this.additionalSystemPrompt
			? `${defaultPrompt}\n\nAdditional user/system instructions:\n${this.additionalSystemPrompt}`
			: defaultPrompt;
	}

	private applyPluginHookContext(result: PluginCommandResult): void {
		// Hook results cross a process/plugin boundary and can be malformed
		// despite PluginCommandResult's static type. One bad plugin entry must
		// not prevent the TUI from starting.
		const messageContexts = Array.isArray(result.context_messages)
			? result.context_messages.flatMap((message) => {
					if (
						!message ||
						typeof message !== "object" ||
						typeof message.content !== "string"
					) {
						return [];
					}
					return [message.content];
				})
			: [];
		const contexts = [
			...(result.additional_contexts || []),
			...messageContexts,
			result.initial_user_message || "",
		]
			.map((item) => String(item || "").trim())
			.filter(
				(item, index, all) => Boolean(item) && all.indexOf(item) === index,
			);
		this.pluginSystemContext = contexts.length
			? `<startup-hook-context>\n${contexts.join("\n\n")}\n</startup-hook-context>`
			: "";
		// Recombine via applyContextLayers (not a standalone reset) so
		// mcpSystemContext/skillsContext survive regardless of whether MCP load
		// or skill discovery finished before or after this hook —
		// loadMcpToolsOnce runs fire-and-forget and can resolve on either side.
		this.applyContextLayers();
	}

	/**
	 * Discover SKILL.md files from installed plugins and inject them into
	 * the system prompt so the agent can see available skills.
	 * Runs after startup hooks as a fallback when hooks fail to produce context.
	 */
	private async injectSkillsFromPlugins(): Promise<void> {
		if (this.skillsInjected) return;
		this.skillsInjected = true;

		const registry = await runPluginBackend("list", []);
		const plugins = registry.plugins || [];

		// Collect skills directories from all enabled, on-disk plugins.
		const skillsDirs: string[] = [];
		const enabledPlugins: Array<{ name: string; installPath: string }> = [];
		for (const plugin of plugins) {
			const enabled = plugin.enabled !== false;
			const onDisk = plugin.on_disk !== false;
			const installPath = String(plugin.install_path || "");
			const pluginName = String(plugin.name || plugin.plugin_id || "");
			if (!enabled || !onDisk || !installPath) continue;
			enabledPlugins.push({ name: pluginName, installPath });
			skillsDirs.push(path.join(installPath, "skills"));
		}
		this.enabledPluginRoots = enabledPlugins;

		// Load user-global skills independently of installed plugins.
		// This is the shared agents convention used by Codex and other harnesses.
		skillsDirs.push(path.join(os.homedir(), ".agents", "skills"));

		// Also discover project-local skills by walking cwd ancestors.
		// Missing directories are skipped silently by loadSkills.
		const cwd = this.config.cwd || process.cwd();
		if (this.projectTrusted) {
			skillsDirs.push(...getProjectSkillDirs(cwd));
		}

		if (!skillsDirs.length) return;

		const { skills: rawSkills, diagnostics } = await loadSkills(skillsDirs);

		// Namespace plugin skills as plugin:skill (Claude Code convention); the
		// bare name stays available as an alias when unambiguous.
		const skills = rawSkills.map((skill) => {
			const owner = enabledPlugins.find((p) =>
				skill.filePath.startsWith(p.installPath + path.sep),
			);
			if (!owner || !owner.name || skill.name.startsWith(`${owner.name}:`)) {
				return skill;
			}
			return {
				...skill,
				name: `${owner.name}:${skill.name}`,
				slashName: `${owner.name}:${skill.slashName}`,
				aliases: [...(skill.aliases ?? []), skill.name],
			};
		});

		// Claude Code plugin commands (commands/*.md) become user-invocable
		// skills: /plugin:command or /command, never advertised to the model.
		skills.push(...(await this.loadPluginCommands(enabledPlugins)));

		// Log diagnostics to transcript for visibility.
		for (const diag of diagnostics) {
			this.emit({
				type: "token",
				token: `[Skill warning] ${diag.code}: ${diag.message}`,
			});
		}

		// All loaded skills are user-invocable via /<skill-name>; only the ones
		// not flagged disable-model-invocation are advertised to the model.
		this.loadedSkills = skills;
		const visible = skills.filter((s) => !s.disableModelInvocation);
		// Only skip catalog injection when there are no skills at all.
		// Plugin commands (disableModelInvocation) still need read_skill.

		// Inject a compact catalog (name + description), not full bodies. The
		// model loads a skill's full instructions on demand via read_skill.
		this.skillsContext = formatSkillCatalog(visible);

		// Register the read_skill tool bound to ALL loaded skills so the model can
		// pull full bodies for any skill, including plugin commands (disableModelInvocation).
		// Append to the tool set (next loop turn picks it up) and
		// patch the live harness registry if a run is already active.
		const readSkill = createReadSkillTool(skills);
		if (readSkill && !this.defaultTools.some((t) => t.name === "read_skill")) {
			this.defaultTools = [...this.defaultTools, readSkill];
			this.config.tools = this.defaultTools;
			this.harness?.setTools(this.defaultTools);
		}

		this.rebuildBaseSystemPrompt();
	}

	/**
	 * Discover prompts/.logician/prompts markdown files and register them as
	 * direct, user-typed /<name> slash commands. Unlike skills, prompts are
	 * never surfaced to the model — they exist only to be typed by the user.
	 */
	private async injectPrompts(): Promise<void> {
		if (this.promptsInjected) return;
		this.promptsInjected = true;

		if (!this.projectTrusted) return;
		const cwd = this.config.cwd || process.cwd();
		const promptDirs = getProjectPromptDirs(cwd);
		this.loadedPrompts = await loadPrompts(promptDirs);
	}

	/**
	 * Load Claude Code plugin commands (commands/*.md) as user-invocable
	 * skill entries. Command bodies are prompt templates; $ARGUMENTS is
	 * substituted at invocation time by invokeSkill.
	 */
	private async loadPluginCommands(
		plugins: Array<{ name: string; installPath: string }>,
	): Promise<Skill[]> {
		const out: Skill[] = [];
		for (const { name: pluginName, installPath } of plugins) {
			const dir = path.join(installPath, "commands");
			let entries: string[];
			try {
				entries = await readdirAsync(dir);
			} catch (err: unknown) {
				// Most plugins have no commands/ dir at all — only a real error
				// (permissions, etc.) is worth surfacing.
				if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") {
					console.error(
						`[plugins] failed to read commands dir for "${pluginName}":`,
						err,
					);
				}
				continue;
			}
			for (const entry of entries) {
				if (!entry.endsWith(".md")) continue;
				const filePath = path.join(dir, entry);
				let raw: string;
				try {
					raw = await readFileAsync(filePath, "utf8");
				} catch (err: unknown) {
					// The file was just listed by readdir, so a read failure here
					// is a genuine anomaly (permissions, race), not "expected".
					console.error(
						`[plugins] failed to read command file "${filePath}":`,
						err,
					);
					continue;
				}
				const parsed = parseFrontmatter<Record<string, unknown>>(raw);
				const frontmatter = parsed.ok ? parsed.value.frontmatter : {};
				const body = parsed.ok ? parsed.value.body : raw;
				const cmdName = entry.slice(0, -3);
				const description =
					typeof frontmatter.description === "string" &&
					frontmatter.description.trim()
						? frontmatter.description
						: `Command from the ${pluginName} plugin.`;
				out.push({
					name: `${pluginName}:${cmdName}`,
					displayName: cmdName,
					description,
					content: body,
					filePath,
					baseDir: dir,
					slashName: `${pluginName}:${cmdName}`,
					disableModelInvocation: true,
					aliases: [cmdName],
					source: "path",
				});
			}
		}
		return out;
	}

	private async runStartupHooksOnce(source = "startup"): Promise<void> {
		if (this.startupHooksRan) return;
		this.startupHooksRan = true;
		const snapshot = await runPluginBackend("list", []);
		this.startupPluginCount = (snapshot.plugins || []).filter((plugin) => {
			return plugin.enabled !== false && plugin.on_disk !== false;
		}).length;
		if (this.config.runtimeHooksEnabled !== false) {
			const result = await runSessionStartHooks({
				source,
				session_id: this.sessionId,
				transcript_path: this.config.hookTranscriptPath,
				cwd: this.config.cwd || process.cwd(),
			});
			this.startupHookResult = result;
			if (result.status !== "error") {
				this.applyPluginHookContext(result);
			}
		}
		// Skills, prompts, and agents are runtime resources, independent of
		// whether command hooks are enabled.
		await this.injectSkillsFromPlugins();
		await this.injectPrompts();
		await this.injectSubagents();
	}

	/**
	 * Register the spawn_agent and spawn_agents tools bound to discovered definitions
	 * (.logician/agents/*.md + built-ins). Subagent events are forwarded into
	 * the normal event stream as subagent_* envelopes.
	 */
	private async injectSubagents(): Promise<void> {
		const cwd = this.config.cwd || process.cwd();
		this.agentDefs = await loadAgentDefinitions([
			...(this.projectTrusted ? [path.join(cwd, ".logician", "agents")] : []),
			// Claude Code plugin agents (agents/*.md in each enabled plugin).
			...this.enabledPluginRoots.map((p) => path.join(p.installPath, "agents")),
		]);

		// Inject subagent tools
		const userSettings = loadUserSettings();
		const maxParallelAgents =
			userSettings.subagents?.maxParallelAgents ??
			(typeof userSettings.maxParallelAgents === "number"
				? userSettings.maxParallelAgents
				: undefined);
		const subagentDeps: SubagentToolDeps = {
			config: () => this.config,
			backend: this.backend,
			cwd,
			agents: () => this.agentDefs,
			emit: (event) => this.config.onEvent?.(event),
			maxParallelAgents,
		};
		const subagentTools = getBuiltInSubagentTools(subagentDeps);
		for (const tool of subagentTools) {
			if (!this.defaultTools.some((t) => t.name === tool.name)) {
				this.defaultTools = [...this.defaultTools, tool];
			}
		}

		this.config.tools = this.defaultTools;
		this.harness?.setTools(this.defaultTools);
		this.rebuildBaseSystemPrompt();
	}

	private async fireSessionEnd(reason: string): Promise<void> {
		if (this.config.runtimeHooksEnabled === false) return;
		try {
			await runHookEvent("SessionEnd", {
				session_id: this.sessionId,
				transcript_path: this.config.hookTranscriptPath || "",
				cwd: this.config.cwd || process.cwd(),
				reason,
			});
		} catch (_e: unknown) {
			// SessionEnd hooks are best-effort during shutdown/reset.
		}
	}
}

export { getSkillsDirs } from "./resource-directories.ts";
