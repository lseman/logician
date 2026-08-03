// ── AgentCoreBridge ──────────────────────────────────────────────────────────────
import { envNumber } from "../tui-utils.ts";
// Replaces the Python bridge with direct TypeScript agent-core integration.
// Translates agent-core events to the same shapes the transcript expects.

import { readFileSync } from "node:fs";
import path from "node:path";
import {
	type AgentConfig,
	type AgentEvent,
	AgentHarness,
	type AgentModelConfig,
	type AbortResult,
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
import {
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
import type { McpSnapshotResult, McpToggleResult } from "../mcp/index.ts";
import {
	formatActivatedSkills,
	formatSkillActivationNotice,
	SkillActivationSession,
	type selectSkillsForPrompt,
} from "../skills/activation.ts";
import {
	findSkillByName,
	formatSkillInvocation,
	type Skill,
} from "../skills/index.ts";
import { findPromptByName, type Prompt } from "../prompts/index.ts";
import { buildDefaultSystemPrompt } from "../context/system-prompt.ts";
import type { SandboxProfile } from "../tools/sandbox.ts";
import { killAllTrackedChildren } from "../tools/shell.ts";
import { EohController } from "./eoh/controller.ts";
import { InteractionCoordinator } from "./interaction-coordinator.ts";
import { SubagentCoordinator } from "./subagent-coordinator.ts";
import { ToolRouter } from "./tool-router.ts";
import { mapAgentEvent } from "../runtime/event-mapping.ts";
import type { ParsedBridgeEvent } from "../runtime/events.ts";
import { LspManager } from "../developer-tools/lsp-manager.ts";
import { formatPluginResult } from "../runtime/plugin-result-formatter.ts";
import { createMemoryStore, createMemoryHooks, setSessionId } from "@logician/memory";
import { startViewerServer, getBoundViewerPort } from "@logician/memory-viewer";
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
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	budgetStopEnabled?: boolean;
	thinkingLoopDetectionEnabled?: boolean;
	continuationEnabled?: boolean;
	reflectionConfig?: AgentConfig["reflectionConfig"];
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
	/** Whether to auto-resume the most recent session on startup (default: true). */
	autoResumeSession?: boolean;
	// ── Memory ────────────────────────────────────────────────────────────
	/** Whether to enable memory hooks. Default: false (opt-in). */
	memoryEnabled?: boolean;
	/** Path to the memory SQLite database. Default: ~/.logician/memory.db. */
	memoryDbPath?: string;
	/** Whether to capture tool observations. Default: true. */
	memoryCaptureTools?: boolean;
	/** Whether to inject context into agent messages. Default: true. */
	memoryInjectContext?: boolean;
	/** Token budget for memory context injection. Default: 4000. */
	memoryContextBudget?: number;
	/** Whether to start the memory viewer web dashboard. Default: true when memory enabled. */
	memoryViewerEnabled?: boolean;
	/** Port for the memory viewer dashboard. Default: 3200. */
	memoryViewerPort?: number;
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
	private toolRouter: ToolRouter;
	private baseSystemPrompt: string;
	private additionalSystemPrompt?: string;
	private pluginSystemContext = "";
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
	private readonly projectTrusted: boolean;
	private interaction: InteractionCoordinator;
	private memoryStore: ReturnType<typeof createMemoryStore> | null = null;
	private memoryCaptureTools: boolean;
	private memoryInjectContext: boolean;
	private memoryContextBudget: number;
	private memoryDbPath: string;
	private memoryViewerServer: ReturnType<typeof startViewerServer> | null = null;
	private memoryViewerPort: number = 3200;
	private memoryViewerEnabled: boolean = true;
	private memoryViewerPortConfig: number = 3200;
	private subagents: SubagentCoordinator;

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
		this.toolRouter = new ToolRouter({
			cwd: this.cwd,
			projectTrusted: this.projectTrusted,
			tools: opts.tools,
			webSearch: opts.webSearch,
			emit: (event) => this.emit(event),
			onToolAdded: () => this.addDefaultTool(),
			onContextChanged: () => this.rebuildBaseSystemPrompt(),
		});
		this.backend = new OpenAIBackend({
			baseUrl: opts.baseUrl,
			model: opts.model,
			chatTemplate: opts.chatTemplate,
		});

		this.additionalSystemPrompt = opts.systemPrompt;
		this.baseSystemPrompt = this.buildBaseSystemPrompt();

		this.interaction = new InteractionCoordinator({
			emit: (event) => this.emit(event),
			permissionMode: opts.permissionMode ?? "acceptAll",
			permissionRules: opts.permissionRules,
		});

		// Initialize memory store if enabled
		if (opts.memoryEnabled !== false) {
			this.memoryDbPath = opts.memoryDbPath || "~/.logician/memory.db";
			this.memoryCaptureTools = opts.memoryCaptureTools ?? true;
			this.memoryInjectContext = opts.memoryInjectContext ?? true;
			this.memoryContextBudget = opts.memoryContextBudget ?? 4000;
			this.memoryStore = createMemoryStore(this.memoryDbPath);
			// Set initial session and derive workspace from cwd
			if (this.memoryStore) {
				const workspace = this.cwd || "";
				this.memoryStore.setCurrentWorkspace(workspace);
				setSessionId(this.memoryStore, this.sessionId);
				// Create session with workspace
				this.memoryStore.createSession(this.sessionId, { project: "", cwd: this.cwd, workspace });
			}
			// Start the memory viewer dashboard
			if (opts.memoryViewerEnabled !== false) {
				this.memoryViewerEnabled = true;
				this.memoryViewerPortConfig = opts.memoryViewerPort || 3200;
				this.memoryViewerPort = this.memoryViewerPortConfig;
				try {
					this.memoryViewerServer = startViewerServer({
						port: this.memoryViewerPort,
						host: "0.0.0.0",
						store: this.memoryStore,
					});
					const bound = getBoundViewerPort();
					if (bound) this.memoryViewerPort = bound;
					console.log(`[bridge] Memory viewer: http://localhost:${this.memoryViewerPort}`);
				} catch (e) {
					console.error("[bridge] Failed to start memory viewer:", e);
				}
			}
		} else {
			this.memoryDbPath = "~/.logician/memory.db";
			this.memoryCaptureTools = true;
			this.memoryInjectContext = true;
			this.memoryContextBudget = 4000;
		}

		this.config = {
			baseUrl: opts.baseUrl,
			model: opts.model,
			models: opts.models,
			systemPrompt: this.baseSystemPrompt,
			tools: this.toolRouter.getDefaultTools(),
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
			permissions: this.interaction.getPermissionManager(),
			guardsEnabled: opts.guardsEnabled,
			duplicateGuardEnabled: opts.duplicateGuardEnabled,
			failureGuardEnabled: opts.failureGuardEnabled,
			duplicateToolThreshold: opts.duplicateToolThreshold,
			toolFailureLoopThreshold: opts.toolFailureLoopThreshold,
			budgetStopEnabled: opts.budgetStopEnabled,
			thinkingLoopDetectionEnabled: opts.thinkingLoopDetectionEnabled,
			continuationEnabled: opts.continuationEnabled,
			reflectionConfig: opts.reflectionConfig,
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
			...this.interaction.buildConfigCallbacks(),
			hooks: this.buildMemoryHooks(
				createPostEditDiagnosticHooks(
					this.cwd,
					() => this.postEditDiagnosticsEnabled,
					this.lspManager,
					{
						allowedPaths: opts.allowedPaths,
						allowAllPaths: opts.allowAllPaths,
					},
				),
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

		this.subagents = new SubagentCoordinator({
			config: () => this.config,
			backend: this.backend,
			cwd: this.cwd,
			projectTrusted: this.projectTrusted,
			getEnabledPluginRoots: () => this.toolRouter.getEnabledPluginRoots(),
			getDefaultTools: () => this.toolRouter.getDefaultTools(),
			onToolAdded: () => this.addDefaultTool(),
			ensureHarness: () => this.ensureHarness(),
			emit: (event) => this.emit(event),
			reportError: (error) => this.reportError(error),
		});
	}

	/**
	 * Merge memory hooks with existing hooks. Memory hooks capture observations
	 * and inject context. Returns the combined hooks object.
	 */
	private buildMemoryHooks(existingHooks: AgentConfig["hooks"]): AgentConfig["hooks"] {
		if (!this.memoryStore) return existingHooks;

		const memoryHooks = createMemoryHooks(this.memoryStore, {
			captureTools: this.memoryCaptureTools,
			injectContext: this.memoryInjectContext,
			contextBudget: this.memoryContextBudget,
			onMemoriesSaved: (memories) => {
				this.emit({
					type: "memory_update",
					kind: "reflections_added",
					count: memories.length,
					items: memories.map((memory) => ({ id: memory.id, content: memory.title })),
				});
			},
		});

		// Merge hooks: existing hooks run first, then memory hooks
		const merged: Record<string, any> = {};

		for (const [key, value] of Object.entries(existingHooks || {})) {
			merged[key as keyof AgentConfig["hooks"]] = value;
		}

		for (const [key, value] of Object.entries(memoryHooks || {})) {
			const existing = merged[key as keyof AgentConfig["hooks"]];
			if (existing) {
				// Chain: existing hook runs first, then memory hook
				merged[key as keyof AgentConfig["hooks"]] = async (ctx: any, signal: any) => {
					const existingResult = await existing(ctx, signal);
					const memoryResult = await (value as Function)(ctx, signal);
					// Return whichever has a non-undefined result
					if (memoryResult !== undefined) return memoryResult;
					return existingResult;
				};
			} else {
				merged[key as keyof AgentConfig["hooks"]] = value;
			}
		}

		return merged as AgentConfig["hooks"];
	}

	/** Add a tool to the default set and propagate it into live config/harness/system prompt. */
	/** Propagate a tool the router just registered into live config/harness/system prompt. */
	private addDefaultTool(): void {
		const tools = this.toolRouter.getDefaultTools();
		this.config.tools = tools;
		this.harness?.setTools(tools);
		this.rebuildBaseSystemPrompt();
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
			if (!this.toolRouter.isMcpLoaded()) {
				const mcpLoad = this.toolRouter.loadMcpToolsOnce();
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
				this.toolRouter.getLoadedSkills(),
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
	async abort(): Promise<AbortResult | null> {
		// harness.abort() clears steering/follow-up and emits onQueueChange.
		return (await this.harness?.abort()) ?? null;
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
		void this.cancel();
		this.running = false;

		// Drop the old harness — conversation starts fresh
		this.harness = null;

		// Reload the runtime without splitting memory from the active
		// user-facing conversation session.
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		if (this.memoryStore) {
			setSessionId(this.memoryStore, this.sessionId);
			this.memoryStore.createSession(this.sessionId, {
				project: "",
				cwd: this.cwd,
				workspace: this.cwd || "",
			});
		}

		// Reset state that is per-session
		this.toolRouter.resetSkillsAndPrompts();
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
		return this.toolRouter.getLoadedSkills();
	}

	/**
	 * Invoke a skill by name as a user prompt: sends the skill's full body
	 * (plus any arguments) to the agent. Returns false for unknown names so the
	 * caller can fall back to normal slash handling.
	 */
	invokeSkill(name: string, args: string): boolean {
		const skill = findSkillByName(this.toolRouter.getLoadedSkills(), name);
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
		return this.toolRouter.getLoadedPrompts();
	}

	/**
	 * Invoke a prompt by name as a user message: sends the prompt's body
	 * (with $ARGUMENTS substituted, or arguments appended) directly — no XML
	 * wrapping, unlike invokeSkill, since a prompt is meant to read exactly as
	 * if the user had typed it. Returns false for unknown names so the caller
	 * can fall back to normal slash handling.
	 */
	invokePrompt(name: string, args: string): boolean {
		const prompt = findPromptByName(this.toolRouter.getLoadedPrompts(), name);
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

	// ── Permissions & interactive questions (see interaction-coordinator.ts) ─

	/** Answer a pending permission_request. Returns false for unknown ids. */
	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean {
		return this.interaction.respondToPermission(toolCallId, decision);
	}

	/** True while a permission_request awaits a decision. */
	hasPendingPermission(): boolean {
		return this.interaction.hasPendingPermission();
	}

	/**
	 * Register a pending question and emit it to the UI. Returns the question id
	 * so the agent can track which question it asked. Call respondToQuestion() to
	 * resolve it.
	 */
	askQuestion(
		question: string,
		choices: Array<{ value: string; label: string }>,
	): string {
		return this.interaction.askQuestion(question, choices);
	}

	/**
	 * Answer a pending question by id. The answer is forwarded to the agent's
	 * resolver. Returns false if the question id is unknown.
	 */
	respondToQuestion(questionId: string, answer: string): boolean {
		return this.interaction.respondToQuestion(questionId, answer);
	}

	/** True while a question_request awaits an answer. */
	hasPendingQuestion(): boolean {
		return this.interaction.hasPendingQuestion();
	}

	/** Deny every pending permission request (abort / shutdown). */
	private denyPendingPermissions(): void {
		this.interaction.denyPendingPermissions();
	}

	setPermissionMode(mode: PermissionMode): void {
		this.interaction.setPermissionMode(mode);
	}

	getPermissionMode(): PermissionMode {
		return this.interaction.getPermissionMode();
	}

	// ── Sandbox mode (see tool-router.ts) ───────────────────────────────

	getSandboxMode(): SandboxProfile {
		return this.toolRouter.getSandboxMode();
	}

	setSandboxMode(mode: SandboxProfile): void {
		this.toolRouter.setSandboxMode(mode);
	}

	cycleSandboxMode(): SandboxProfile {
		return this.toolRouter.cycleSandboxMode();
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
		if (!this.toolRouter.isMcpLoaded() && !this.toolRouter.isMcpLoading()) {
			void this.toolRouter
				.loadMcpToolsOnce()
				.catch((error) => this.reportError(error));
		}
		this.contextTokens = this.measureContextTokens();
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this.toolRouter.getDefaultTools().map((t) => t.name);
		const state = {
			agent_name: "logician",
			model: this.config.model,
			base_url: this.config.baseUrl,
			web_search_url: this.config.webSearch?.baseUrl || "",
			web_search_enabled: toolNames.includes("web_search"),
			tools: toolNames,
			mcp_servers: this.toolRouter.getMcpServerCount(),
			mcp_tools: this.toolRouter.getMcpToolCount(),
			mcp_errors: this.toolRouter.getMcpErrors(),
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
		return this.toolRouter.getMcpSnapshot();
	}

	async setMcpServerEnabled(
		serverName: string,
		enabled: boolean,
	): Promise<McpToggleResult> {
		return this.toolRouter.setMcpServerEnabled(serverName, enabled);
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
			| "rtkProxyEnabled"
			| "memoryEnabled",
		enabled: boolean,
	): void {
		if (key === "memoryEnabled") {
			if (enabled && !this.memoryStore) {
				// Enable memory on the fly
				const dbPath = this.memoryDbPath;
				this.memoryStore = createMemoryStore(dbPath);
				setSessionId(this.memoryStore, this.sessionId);
				const workspace = this.cwd || "";
				this.memoryStore.setCurrentWorkspace(workspace);
				this.memoryStore.createSession(this.sessionId, {
					project: "",
					cwd: this.cwd,
					workspace,
				});
				// Start viewer if enabled
				if (this.memoryViewerEnabled) {
					try {
						this.memoryViewerServer = startViewerServer({
							port: this.memoryViewerPortConfig,
							host: "127.0.0.1",
							store: this.memoryStore,
						});
						const bound = getBoundViewerPort();
						if (bound) this.memoryViewerPort = bound;
						console.log(`[bridge] Memory viewer started on port ${this.memoryViewerPort}`);
					} catch (e) {
						console.error("[bridge] Failed to start memory viewer:", e);
					}
				}
			} else if (!enabled) {
				if (this.memoryStore) this.memoryStore.close();
				this.memoryStore = null;
				if (this.memoryViewerServer) {
					this.memoryViewerServer.stop();
					this.memoryViewerServer = null;
				}
			}
			this.emit({
				type: "notice",
				level: "info",
				label: "Memory",
				text: enabled
					? "Memory enabled"
					: "Memory disabled",
			});
			return;
		}
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
			`  Memory: ${this.memoryStore ? "on" : "off"}`,
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
		memoryEnabled: boolean;
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
			memoryEnabled: this.memoryStore !== null,
		};
	}

	getMemoryStore(): ReturnType<typeof createMemoryStore> | null {
		return this.memoryStore;
	}

	getMemoryStats(): {
		memoryEnabled: boolean;
		memoryCount: number;
		sessionCount: number;
		observationCount: number;
		viewerPort?: number;
	} {
		if (!this.memoryStore) {
			return { memoryEnabled: false, memoryCount: 0, sessionCount: 0, observationCount: 0 };
		}
		const memories = this.memoryStore.list({ limit: 1000 });
		const sessions = this.memoryStore.listSessions();
		const observations = this.memoryStore.listObservations(this.sessionId, 1000);
		return {
			memoryEnabled: true,
			memoryCount: memories.length,
			sessionCount: sessions.length,
			observationCount: observations.length,
			viewerPort: this.memoryViewerPort,
		};
	}

	/** Use the user-facing conversation session as the hook and memory session. */
	useConversationSession(sessionId: string): void {
		if (!sessionId.trim()) return;
		const provisionalSessionId = this.sessionId;
		if (this.memoryStore && provisionalSessionId !== sessionId) {
			this.memoryStore.discardEmptySession(provisionalSessionId);
		}
		this.sessionId = sessionId;
		this.transcriptPath = createHookTranscriptPath(this.cwd, sessionId);
		this.config.hookSessionId = sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		if (this.memoryStore) {
			setSessionId(this.memoryStore, sessionId);
			this.memoryStore.createSession(sessionId, {
				project: "",
				cwd: this.cwd,
				workspace: this.cwd || "",
			});
		}
	}

	renameConversationSession(sessionId: string, name: string): void {
		this.memoryStore?.updateSession(sessionId, { name: name.trim() });
	}

	reset(): void {
		// Reset tool state and conversation
		void this.fireSessionEnd("reset");
		// Drop the persisted harness so history starts fresh.
		this.harness?.clearHistory();
		this.harness = null;
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Update memory session ID
		if (this.memoryStore) {
			setSessionId(this.memoryStore, this.sessionId);
			this.memoryStore.createSession(this.sessionId, {
				project: "",
				cwd: this.cwd,
				workspace: this.cwd || "",
			});
		}
		// Reset skill/prompt injection state
		this.toolRouter.resetInjectedContext();
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

	async cancel(): Promise<AbortResult | null> {
		// A turn blocked on an approval must unblock to abort cleanly.
		this.denyPendingPermissions();
		try {
			return (await this.harness?.abort()) ?? null;
		} catch (error) {
			const normalized =
				error instanceof Error ? error : new Error(String(error));
			this.errorCb?.(normalized);
			throw normalized;
		}
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
			void this.toolRouter.loadMcpToolsOnce().then(
				() => {
					this.emit({
						type: "notice",
						level: this.toolRouter.getMcpErrors().length ? "warn" : "info",
						label: "MCP",
						text: `Loaded ${this.toolRouter.getMcpServerCount()} server(s).`,
					});
				},
				(error) => this.reportError(error),
			);
		}
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this.toolRouter.getDefaultTools().map((t) => t.name);
		const status = this.toolRouter.getStatus();
		const info: Record<string, unknown> = {
			agent_name: "logician",
			model: this.config.model,
			base_url: this.config.baseUrl,
			web_search_url: this.config.webSearch?.baseUrl || "",
			web_search_enabled: toolNames.includes("web_search"),
			mcp_deferred: !status.mcpLoaded && process.env.LOGICIAN_MCP !== "0",
			mcp_loading: status.mcpLoading,
			tools: toolNames,
			mcp_servers_loaded: status.mcpServerCount,
			mcp_tools_loaded: status.mcpToolCount,
			mcp_errors: status.mcpErrors,
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
			startup_plugins: status.enabledPluginRoots.map((plugin) => plugin.name),
			startup_hooks_loaded: this.startupHookResult?.hook_count || 0,
			startup_hook_contexts: this.startupHookResult?.additional_contexts || [],
			startup_hook_messages: this.startupHookResult?.context_messages || [],
			startup_hook_initial_message:
				this.startupHookResult?.initial_user_message || "",
			startup_hook_errors: this.startupHookResult?.errors || [],
			skills_injected: status.skillsInjected
				? status.loadedSkills.filter((skill) => !skill.disableModelInvocation)
						.length
				: 0,
			skills_visible: status.skillsVisible,
			loaded_skills: status.loadedSkills.map((skill) => ({
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
		void this.cancel();
		await this.fireSessionEnd("shutdown");
		this.lspManager.close();
		await this.toolRouter.closeMcp();
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
			if (!msg) continue;
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
		const toolDefinitions = this.getTools().toToolDefinitions();
		const conversation = messages.filter((message) => message.role !== "tool");
		const toolEvidence = messages.filter((message) => message.role === "tool");
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
				name: "Tool definitions",
				tokens: estimateChatPayloadTokens([], toolDefinitions),
				detail: `${toolDefinitions.length} tools`,
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
		const toolDefinitions = this.getTools().toToolDefinitions();
		return estimateChatPayloadTokens(messages, toolDefinitions);
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
		return this.toolRouter.buildRegistry({
			cwd: this.config.cwd,
			allowedPaths: this.config.allowedPaths,
			allowAllPaths: this.config.allowAllPaths,
			cacheSize: this.config.cacheSize,
			cacheTtlMs: this.config.cacheTtlMs,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
	}

	private rebuildBaseSystemPrompt(): void {
		this.baseSystemPrompt = this.buildBaseSystemPrompt();
		this.applyContextLayers();
	}

	/** Recombine base prompt + plugin/MCP/skills context layers into config.systemPrompt. */
	private applyContextLayers(): void {
		const contexts: string[] = [];
		if (this.pluginSystemContext) contexts.push(this.pluginSystemContext);
		const mcpSystemContext = this.toolRouter.getMcpSystemContext();
		if (mcpSystemContext) contexts.push(mcpSystemContext);
		const skillsContext = this.toolRouter.getSkillsContext();
		if (skillsContext) contexts.push(skillsContext);
		this.config.systemPrompt = contexts.length
			? `${this.baseSystemPrompt}\n\n${contexts.join("\n\n")}`
			: this.baseSystemPrompt;
	}

	private buildBaseSystemPrompt(): string {
		const defaultPrompt = buildDefaultSystemPrompt(
			this.cwd,
			this.toolRouter.getDefaultTools(),
			{ loadProjectContext: this.projectTrusted },
		);
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
		await this.toolRouter.injectSkillsFromPlugins();
		this.rebuildBaseSystemPrompt();
		await this.toolRouter.injectPrompts();
		await this.injectSubagents();
	}

	/**
	 * Register the spawn_agent and spawn_agents tools bound to discovered
	 * definitions (.logician/agents/*.md + built-ins). See subagent-coordinator.ts.
	 */
	private async injectSubagents(): Promise<void> {
		await this.subagents.inject();
	}

	/**
	 * Directly invoke the spawn_agent tool without going through the LLM.
	 * See subagent-coordinator.ts for the full lifecycle/event wiring.
	 */
	spawnAgentDirectly(task: string, agent?: string): void {
		this.subagents.spawnDirectly(task, agent);
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
