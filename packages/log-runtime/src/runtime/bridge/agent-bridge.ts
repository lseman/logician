/** Coordinates one interactive agent session and its runtime integrations. */

import type {
	AgentConfig,
	AgentEvent,
	Message,
	QueueMode,
	Tool,
} from "@logician/log-core";
import { OpenAIBackend } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import { AgentSession } from "@logician/log-core/harness";
import type { PermissionMode } from "@logician/log-core/permissions";
import type { AbortResult, Session } from "@logician/log-core/runtime";
import {
	estimateChatPayloadTokens,
	estimateTokens,
	type ToolRegistry,
} from "@logician/log-core/runtime";
import {
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
} from "../../adapters/claude-code/hook-layer.ts";
import type { PluginCommandResult } from "../../adapters/claude-code/plugin-runtime.ts";
import {
	configurePluginRuntimeEnv,
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
} from "../../adapters/claude-code/plugin-runtime.ts";
import { ExtensionRegistry } from "../../capabilities/extensions/extensions.ts";
import { InteractionGateway } from "../../capabilities/interactions/interaction-gateway.ts";
import { LspClientPool } from "../../capabilities/lsp/lsp-client-pool.ts";
import { createPostEditDiagnosticHooks } from "../../capabilities/lsp/post-edit-diagnostics.ts";
import type {
	McpSnapshotResult,
	McpToggleResult,
} from "../../capabilities/mcp/mcp-server-registry.ts";
import { MemoryHost } from "../../capabilities/memory/memory.ts";
import {
	createMemoryGetTool,
	createMemorySearchTool,
} from "../../capabilities/memory/memory-tools.ts";
import {
	findPromptByName,
	type Prompt,
} from "../../capabilities/prompts/loader.ts";
import { RepositoryMap } from "../../capabilities/repository-map/repository-map.ts";
import {
	formatActivatedSkills,
	formatSkillActivationNotice,
	type selectSkillsForPrompt,
} from "../../capabilities/skills/activation.ts";
import {
	findSkillByName,
	formatSkillInvocation,
	type Skill,
} from "../../capabilities/skills/loader.ts";
import { getTasks, onTodosChanged } from "../../capabilities/tasks/todo.ts";
import type { SandboxProfile } from "../../capabilities/tools/sandbox.ts";
import { killAllTrackedChildren } from "../../capabilities/tools/support/utils/shell.ts";
import { buildDefaultSystemPrompt } from "../context/system-prompt.ts";
import { mapAgentEvent } from "../events/event-mapping.ts";
import { RuntimeEventBus } from "../events/runtime-event-bus.ts";
import { ModelSelector } from "../model-selector.ts";
import { buildToolRegistry } from "../runtime-context.ts";
import { type RuntimeToggleKey, SettingsGateway } from "../settings-gateway.ts";
import { ToolRouter } from "../tool-router.ts";
import { AgentCoordinator } from "./agent-coordinator.ts";
import { RuntimeContext } from "./capability-context.ts";
import {
	buildPluginRuntimeEnv,
	createHookTranscriptPath,
	envNumber,
	eventLogPathFor,
	resolveWebSearchConfig,
} from "./environment.ts";
import { formatPluginResult } from "./plugin-result-formatter.ts";

export { findJbPrompt } from "./project-prompt.ts";

export type {
	AgentBridgeOptions,
	ErrorCallback,
	ProtocolCallback,
	RuntimeSettingsPatch,
} from "./types.ts";

import type { AgentBridgeOptions, RuntimeSettingsPatch } from "./types.ts";

// ── AgentRuntime ─────────────────────────────────────────────────────────────

export class AgentRuntime {
	private config: AgentConfig;
	private backend: OpenAIBackend;
	private session: AgentSession | null = null;
	private durableSession: Session | undefined;
	readonly events = new RuntimeEventBus();
	readonly models: ModelSelector;
	private running = false;
	private sendTail: Promise<void> = Promise.resolve();

	private cwd: string;
	private readonly toolRouter: ToolRouter;
	private baseSystemPrompt = "";
	private get _defaultTools(): Tool[] {
		return this.toolRouter.getDefaultTools();
	}
	private get _loadedSkills(): Skill[] {
		return this.toolRouter.getLoadedSkills();
	}
	private get _loadedPrompts(): Prompt[] {
		return this.toolRouter.getLoadedPrompts();
	}
	private get _enabledPluginRoots(): Array<{
		name: string;
		installPath: string;
	}> {
		return this.toolRouter.getEnabledPluginRoots();
	}

	async loadMcpToolsOnce(): Promise<void> {
		await this.toolRouter.loadMcpToolsOnce();
		this.refreshInjectedContext();
	}

	async injectSkillsFromPlugins(): Promise<void> {
		await this.toolRouter.injectSkillsFromPlugins();
		this.refreshInjectedContext();
	}

	async injectPrompts(): Promise<void> {
		await this.toolRouter.injectPrompts();
	}

	private refreshInjectedContext(): void {
		this.applyPluginHookContext(
			this.startupHookResult || {
				additional_contexts: [],
				context_messages: [],
				initial_user_message: "",
			},
		);
	}

	applyPluginHookContext(result: PluginCommandResult): void {
		const messageContexts = Array.isArray(result.context_messages)
			? result.context_messages.flatMap(message => {
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
			.map(item => String(item || "").trim())
			.filter(
				(item, index, all) => Boolean(item) && all.indexOf(item) === index,
			);

		// Recombine base + plugin context + MCP context + skills context
		const allContexts: string[] = [];
		if (contexts.length) {
			allContexts.push(
				`<startup-hook-context>\n${contexts.join("\n\n")}\n</startup-hook-context>`,
			);
		}
		const mcpContext = this.toolRouter.getMcpSystemContext();
		const skillsContext = this.toolRouter.getSkillsContext();
		if (mcpContext) allContexts.push(mcpContext);
		if (skillsContext) allContexts.push(skillsContext);

		this.config.systemPrompt = allContexts.length
			? `${this.baseSystemPrompt}\n\n${allContexts.join("\n\n")}`
			: this.baseSystemPrompt;
	}
	private readonly runtimeCtx: RuntimeContext;
	private get extensions(): ExtensionRegistry {
		return this.runtimeCtx.extensions;
	}

	private sessionId =
		`tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
	private transcriptPath = "";
	private startupHooksRan = false;
	private startupHooksPromise: Promise<void> | null = null;
	private startupHookResult: PluginCommandResult | null = null;
	private startupPluginCount = 0;
	private contextTokens = 0;
	private contextMaxTokens?: number;
	private configPath: string | null;
	private postEditDiagnosticsEnabled: boolean;
	private get lsp(): LspClientPool {
		return this.runtimeCtx.lsp;
	}
	private readonly projectTrusted: boolean;
	private get interactions(): InteractionGateway {
		return this.runtimeCtx.interactions;
	}
	private readonly settings: SettingsGateway;
	private get memory(): MemoryHost {
		return this.runtimeCtx.memory;
	}
	private agentCoordinator: AgentCoordinator | null = null;

	private get repositoryMap(): RepositoryMap | undefined {
		return this.runtimeCtx.repositoryMap;
	}
	private runtimeRetry?: string;
	private runtimeRepair?: string;
	private readonly activeRuntimeSubagents = new Set<string>();
	private readonly compactionSettings?: AgentBridgeOptions["compaction"];

	// ── EoH (Evolution of Heuristics) ─────────────────────────────────

	/** EoH command: /eoh <file.py> [generations] | stop | status | best | reset */
	eohCommand(raw: string): string {
		return this.agentCoordinator?.eohCommand(raw) ?? "";
	}

	constructor(
		opts: AgentBridgeOptions = {
			baseUrl: "http://localhost:8080",
			model: "",
		},
	) {
		this.compactionSettings = opts.compaction;
		this.cwd = opts.cwd || process.cwd();
		this.projectTrusted = opts.projectTrusted === true;
		this.configPath = opts.configPath ?? null;
		configurePluginRuntimeEnv(buildPluginRuntimeEnv(opts));
		this.postEditDiagnosticsEnabled =
			process.env.LOGICIAN_POST_EDIT_DIAGNOSTICS === "0"
				? false
				: opts.postEditDiagnostics !== false;

		// Independent capabilities — none reads another slot during
		// construction (RuntimeContext's own contract) — mount together so
		// downstream code (toolRouter's extraTools closures, the AgentConfig
		// build below) can rely on every slot existing.
		this.runtimeCtx = RuntimeContext.mount([
			{
				id: "repositoryMap",
				register: () =>
					opts.repositoryMap?.enabled !== false
						? new RepositoryMap(this.cwd, {
								maxTokens: opts.repositoryMap?.maxTokens,
							})
						: undefined,
			},
			{
				id: "lsp",
				register: () => {
					const serverOverrides = opts.lsp?.serverOverrides;
					return new LspClientPool(this.cwd, {
						timeoutMs: opts.lsp?.timeoutMs ?? 2_000,
						servers:
							serverOverrides && Object.keys(serverOverrides).length > 0
								? serverOverrides
								: undefined,
					});
				},
			},
			{
				id: "extensions",
				register: () =>
					new ExtensionRegistry({
						sessionId: this.sessionId,
						cwd: this.cwd,
						extensionDirs: opts.extensions?.dirs,
						projectTrusted: this.projectTrusted,
					}),
			},
			{
				id: "interactions",
				register: () =>
					new InteractionGateway({
						mode: opts.permissions?.mode ?? "acceptEdits",
						rules: opts.permissions?.rules,
						emit: event => this.emit(event),
					}),
			},
			{
				id: "memory",
				register: () =>
					new MemoryHost(this.cwd, this.sessionId, {
						memoryEnabled: opts.memory?.enabled,
						memoryDbPath: opts.memory?.dbPath,
						memoryExtractorModel: opts.memory?.extractorModel,
						memoryExtractorBaseUrl: opts.memory?.extractorBaseUrl,
						memoryCaptureTools: opts.memory?.captureTools,
						memoryInjectContext: opts.memory?.injectContext,
						memoryContextBudget: opts.memory?.contextBudget,
						memoryViewerEnabled: opts.memory?.viewerEnabled,
						memoryViewerPort: opts.memory?.viewerPort,
						memoryEmbeddingsEnabled: opts.memory?.embeddingsEnabled,
						memoryEmbeddingModel: opts.memory?.embeddingModel,
						model: opts.model,
					}),
			},
		]);
		// Extension loading is a real side effect (reads disk, may run init
		// hooks) — kept as an explicit statement rather than folded into
		// register(), which the context treats as pure construction.
		void this.extensions.initialize();

		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		const defaultWebSearch = resolveWebSearchConfig();
		const webSearch = {
			baseUrl: opts.webSearch?.baseUrl || defaultWebSearch.baseUrl,
			maxResults: opts.webSearch?.maxResults ?? defaultWebSearch.maxResults,
		};
		const extraTools = opts.extraTools?.length
			? [
					...(opts.memory?.enabled !== false
						? [
								createMemorySearchTool(() => this.memory?.getStore() ?? null),
								createMemoryGetTool(() => this.memory?.getStore() ?? null),
							]
						: []),
					...opts.extraTools,
				]
			: undefined;
		this.toolRouter = new ToolRouter({
			cwd: this.cwd,
			projectTrusted: this.projectTrusted,
			tools: opts.tools,
			extraTools,
			webSearch,
			ariadneEnabled: opts.ariadneEnabled,
			fffgrepEnabled: opts.fffgrepEnabled,
			autoStartMcp: false,
			emit: event => this.emit(event),
			onToolAdded: _tool => {
				if (!this.config) return;
				this.config.tools = this.toolRouter.getDefaultTools();
				this.session?.configure({ tools: this.config.tools });
			},
			onContextChanged: () => {
				if (this.baseSystemPrompt) this.refreshInjectedContext();
			},
		});
		this.baseSystemPrompt = buildDefaultSystemPrompt(
			this.cwd,
			this._defaultTools,
			{ loadProjectContext: this.projectTrusted },
		);
		this.backend = new OpenAIBackend({
			baseUrl: opts.baseUrl,
			model: opts.model,
			chatTemplate: opts.chatTemplate,
		});
		if (opts.thinkingLevel) {
			this.backend.setDefaultThinkingLevel(opts.thinkingLevel);
		}

		this.config = {
			baseUrl: opts.baseUrl,
			model: opts.model,
			models: opts.models,
			systemPrompt: this.baseSystemPrompt,
			tools: this._defaultTools,
			webSearch,
			cwd: this.cwd,
			maxIterations: opts.maxIterations || 30,
			executionProfile: opts.executionProfile,
			temperature: opts.temperature,
			maxTokens: opts.maxTokens,
			thinkingLevel: opts.thinkingLevel ?? "off",
			inferenceMode: opts.inferenceMode ?? "none",
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
			permissions: this.interactions.permissions,
			guardsEnabled: opts.guardsEnabled,
			duplicateGuardEnabled: opts.duplicateGuardEnabled,
			failureGuardEnabled: opts.failureGuardEnabled,
			duplicateToolThreshold: opts.duplicateToolThreshold,
			toolFailureLoopThreshold: opts.toolFailureLoopThreshold,
			progressStopEnabled: opts.progressStopEnabled,
			proactiveCompactionEnabled: opts.proactiveCompactionEnabled,
			continuationEnabled: opts.continuationEnabled,
			rtkProxyEnabled: opts.rtkProxyEnabled,
			ariadneEnabled: opts.ariadneEnabled ?? true,
			fffgrepEnabled: opts.fffgrepEnabled ?? true,
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
			onPermissionRequest: context =>
				this.interactions.requestPermission(context),
			onQuestionRequest: context => this.interactions.requestQuestion(context),
			hooks: this.buildMemoryHooks(
				createPostEditDiagnosticHooks(
					this.cwd,
					() => this.postEditDiagnosticsEnabled,
					opts.lsp?.enabled === false ? undefined : this.lsp,
					{
						allowedPaths: opts.allowedPaths,
						allowAllPaths: opts.allowAllPaths,
					},
				),
			),
			turnEndCallback: (turnId: string) => {
				this.emit({ type: "turn_end", turnId });
			},
			onEvent: (event: AgentEvent) => {
				if (event.type === "context_update") {
					this.contextTokens = event.tokens;
					this.contextMaxTokens = event.maxTokens;
				}

				if (event.type === "agent_retry_start") {
					this.runtimeRetry = `${event.attempt}/${event.maxRetries}`;
				} else if (event.type === "agent_retry_end")
					this.runtimeRetry = undefined;
				if (event.type === "repair_nudge")
					this.runtimeRepair = event.repairStage;
				if (event.type === "turn_start") this.runtimeRepair = undefined;
				if (event.type === "subagent_start") {
					this.activeRuntimeSubagents.add(event.agentId);
				} else if (event.type === "subagent_end") {
					this.activeRuntimeSubagents.delete(event.agentId);
				}
				const mapped = mapAgentEvent(event);
				if (mapped) {
					this.emit(mapped);
				}
				this.emitRuntimeStatus();
			},
		};
		this.models = new ModelSelector(
			() => this.config,
			() => this.session,
		);
		this.settings = new SettingsGateway({
			config: () => this.config,
			patchCore: patch => {
				Object.assign(this.config, patch);
				this.session?.configure(patch);
			},
			setThinkingLevel: level => {
				if (level !== undefined) this.setThinkingLevel(level);
			},
			setTemperature: value => this.setTemperature(value),
			setReasoner: id => this.setReasonerId(id),
			setSteeringInterrupt: enabled => this.setSteeringInterrupt(enabled),
			setToggle: (key, enabled) => this.applyRuntimeToggle(key, enabled),
			permissionMode: () => this.getPermissionMode(),
			postEditDiagnostics: () => this.postEditDiagnosticsEnabled,
			memoryEnabled: () => Boolean(this.memory?.getStore()),
		});

		onTodosChanged(todos => {
			this.emit({ type: "todos", todos });
		});

		// Create agent coordinator for reasoner, EoH, and subagents
		this.agentCoordinator = new AgentCoordinator(
			{
				emit: event => this.emit(event),
				getBackend: () => this.backend,
				getBaseUrl: () => this.config.baseUrl,
				getCurrentModel: () => this.models.current(),
				harness: null, // set below via ensureSession
				cwd: this.cwd,
				projectTrusted: this.projectTrusted,
				maxParallelAgents: opts.maxParallelAgents,
				getEnabledPluginRoots: () => this._enabledPluginRoots,
				getDefaultTools: () => this._defaultTools,
				ensureSession: () => this.ensureSession(),
				reportError: error => this.events.reportError(error),
			},
			opts.reasoner,
			opts.reasonerConfig,
		);
	}

	/**
	 * Build memory hooks by delegating to the MemoryHost.
	 */
	private buildMemoryHooks(
		existingHooks: AgentConfig["hooks"],
	): AgentConfig["hooks"] {
		if (!this.memory) return existingHooks;
		return this.memory.createHooks(existingHooks, {
			isRunning: () => this.running,
			getBackend: () => this.backend,
			emit: event => this.emit(event),
		});
	}

	/** Add a tool to the default set and propagate it into live config/harness/system prompt. */
	/** Propagate a tool the router just registered into live config/harness/system prompt. */
	// ── Event registration ─────────────────────────────────────────────────

	private emit(event: RuntimeEvent): void {
		this.events.emit(event);
	}

	// ── High-level commands ──────────────────────────────────────────────

	async sendMessage(message: string): Promise<void> {
		await this.extensions?.getLoadPromise();
		// A message submitted while a turn is in flight steers the running
		// turn instead of starting a second concurrent run. Route through
		// steer() so the queue update reaches the UI.
		if (this.running && this.session) {
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
		// Local extractor models can saturate CPU/GPU and make both the UI and
		// primary provider sluggish. Prefer the interactive turn; extraction's
		// deterministic fallback still records the completed prior turn.
		this.memory?.abortExtractors();
		this.running = true;
		const turnId = `turn_${Date.now()}`;
		let persistentSystemPrompt: string | undefined;
		let turnSystemPrompt = this.config.systemPrompt;
		let turnActivations: ReturnType<typeof selectSkillsForPrompt> = [];
		let turnSucceeded = false;
		try {
			await this.runStartupHooksOnce();
			// MCP loads in the background from the moment the bridge is
			// constructed (see ToolRouter's constructor) — never block turn
			// submission on it. Whatever has finished connecting by the time the
			// prompt actually goes out is what the model sees; a load still in
			// flight keeps running and its tools become available on the next
			// turn once it settles.
			if (!this.toolRouter.isMcpLoaded()) {
				void this.toolRouter
					.loadMcpToolsOnce()
					.catch(error => this.events.reportError(error));
			}
			// Reuse one session across messages so conversation history (and thus
			// "continue" / "go on" follow-ups) persists. Created lazily once.
			const session = this.ensureSession();
			const repositoryContext = this.repositoryMap?.render(message);
			if (repositoryContext) {
				persistentSystemPrompt = this.config.systemPrompt;
				turnSystemPrompt = `${persistentSystemPrompt}\n\n${repositoryContext}`;
				session.configure({ systemPrompt: turnSystemPrompt });
			}
			if (this.agentCoordinator) {
				const advisory = await this.agentCoordinator.runReasoner(
					message,
					this.backend,
				);
				if (advisory) {
					persistentSystemPrompt ??= this.config.systemPrompt;
					turnSystemPrompt = `${turnSystemPrompt}\n\nA structured reasoner produced the following advisory analysis for this turn. Verify it, use tools as needed, and do not mention this internal advisory unless useful:\n\n${advisory}`;
					session.configure({ systemPrompt: turnSystemPrompt });
				}
			}
			const activations: ReturnType<typeof selectSkillsForPrompt> = [];
			// Skills stay discoverable through read_skill; none are injected unless
			// a caller explicitly activates them for this turn.
			turnActivations = activations;
			if (activations.length) {
				persistentSystemPrompt ??= this.config.systemPrompt;
				session.configure({
					systemPrompt: `${turnSystemPrompt}\n\n${formatActivatedSkills(activations)}`,
				});
				this.emit({
					type: "notice",
					level: "info",
					label: "Skills",
					text: formatSkillActivationNotice(activations),
				});
			}

			this.emit({ type: "turn_start", turnId: turnId });
			await session.prompt(message);
			turnSucceeded = true;
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
			this.events.notifyError(error);
			throw error;
		} finally {
			if (persistentSystemPrompt !== undefined) {
				this.session?.configure({ systemPrompt: persistentSystemPrompt });
			}
			this.running = false;
			this.publishContextUsage();
			this.emit({ type: "turn_end", turnId });
			// Keep the harness alive to retain history across turns.
			this.emit({ type: "phase", state: "ready" });
			if (
				turnSucceeded &&
				(this.session?.getQueues().nextTurn.length ?? 0) > 0
			) {
				this.running = true;
				const repoQuery = this.repositoryMap
					? await this.repositoryMap.render("")
					: undefined;
				this.session?.setRepositoryQuery(repoQuery);
				await this.runContinuation(turnActivations);
			}
		}
	}

	// ── Continuation ───────────────────────────────────────────────────────

	private async runContinuation(
		activations: ReturnType<typeof selectSkillsForPrompt>,
	): Promise<void> {
		try {
			const repoQuery = this.repositoryMap?.render("");
			this.session?.setRepositoryQuery(repoQuery);
			const context = activations.length
				? formatActivatedSkills(activations)
				: undefined;
			await this.session?.runQueuedContinuation(context, repoQuery);
		} finally {
			this.running = false;
		}
	}

	// Lazily build the singleton harness and wire its UI callbacks.
	private ensureSession(): AgentSession {
		if (!this.session) {
			this.session = new AgentSession({
				config: {
					...this.config,
					taskLedger: { snapshot: getTasks },
				},
				backend: this.backend,
				cwd: this.config.cwd,
				maxIterations: this.config.maxIterations,
				extensionRunner: this.extensions?.runner ?? undefined,
				pluginHookFactory: context =>
					createClaudeCodeHookLayer({
						enabled: context.enabled,
						sessionId: context.sessionId,
						transcriptPath: context.transcriptPath,
						cwd: context.cwd,
						getMatcherValue: toolName => {
							const tool = context.tools.find(item => item.name === toolName);
							return (
								tool?.hookAliases?.join("|") || claudeToolMatcherName(toolName)
							);
						},
					}),
				pluginLifecycle: {
					sessionStart: async (context, source) => {
						await runSessionStartHooks({
							source,
							session_id: context.sessionId,
							transcript_path: context.transcriptPath,
							cwd: context.cwd,
						});
					},
					sessionEnd: async (context, reason) => {
						await runHookEvent("SessionEnd", {
							session_id: context.sessionId,
							transcript_path: context.transcriptPath,
							cwd: context.cwd,
							reason,
						});
					},
					preCompact: async context => {
						await runHookEvent("PreCompact", {
							session_id: context.sessionId,
							transcript_path: context.transcriptPath,
							cwd: context.cwd,
						});
					},
					postCompact: async context => {
						await runHookEvent("PostCompact", {
							session_id: context.sessionId,
							transcript_path: context.transcriptPath,
							cwd: context.cwd,
						});
					},
				},
			});
			this.session.setSessionId(this.sessionId);
			if (this.durableSession) this.session.attachSession(this.durableSession);
			if (this.compactionSettings)
				this.session.setAutoCompactionSettings(this.compactionSettings);
			// Session handles queue/phase events internally via onEvent callback.
			// Observe settled to show continuation notice.
			this.session.observe({
				settled: nextTurnCount => {
					if (nextTurnCount === 0) return;
					this.emit({
						type: "notice",
						level: "info",
						label: "Continuation",
						text: `${nextTurnCount} next-turn message(s) queued; continuation will start after settlement.`,
					});
				},
			});
		}
		return this.session;
	}

	private emitRuntimeStatus(): void {
		if (!this.session) return;
		this.emit({
			type: "runtime_status",
			runPhase: this.session.phase,
			retry: this.runtimeRetry,
			repair: this.runtimeRepair,
			activeSubagents: this.activeRuntimeSubagents.size,
		});
	}

	/**
	 * Replace the harness conversation with restored session history (resume /
	 * session switch), so the model continues with the restored context instead
	 * of starting cold. Pass [] to clear (new session). No-op while a turn is
	 * running (the harness rejects structural ops mid-turn).
	 */
	restoreHistory(messages: Message[]): boolean {
		try {
			this.ensureSession().setHistory(messages);
			this.publishContextUsage();
			return true;
		} catch (_e: unknown) {
			return false;
		}
	}

	// ── Queue operations (delegated to AgentSession) ─────────────────

	steer(message: string): void {
		this.session?.steer(message);
	}

	/** Queue steering for after the current turn (never interrupts). */
	steerQueue(message: string): void {
		this.session?.steerQueue(message);
	}

	/** Immediately interrupt and apply steering (always forces abort). */
	steerNow(message: string): void {
		this.session?.steerNow(message);
	}

	followUp(message: string): void {
		this.session?.followUp(message);
	}

	nextTurn(message: string): void {
		this.session?.nextTurn(message);
	}

	setSteeringMode(mode: QueueMode): void {
		this.session?.setSteeringMode(mode);
	}

	private setSteeringInterrupt(enabled: boolean): void {
		this.session?.configure({ steeringInterrupt: enabled });
	}

	getSteeringInterrupt(): boolean {
		return this.config.steeringInterrupt === true;
	}

	/** Return config snapshot for external LLM calls (goal evaluator, etc.). */
	getConfig(): {
		baseUrl: string;
		model: string;
		rtkProxyEnabled?: boolean;
		ariadneEnabled?: boolean;
		fffgrepEnabled?: boolean;
	} {
		return {
			baseUrl: this.config.baseUrl,
			model: this.config.model,
			rtkProxyEnabled: this.config.rtkProxyEnabled,
			ariadneEnabled: this.config.ariadneEnabled,
			fffgrepEnabled: this.config.fffgrepEnabled,
		};
	}

	setFollowUpMode(mode: QueueMode): void {
		this.session?.setFollowUpMode(mode);
	}

	getSteeringMessages(): string[] {
		return this.session?.getQueues().steering ?? [];
	}

	flushSteeringNow(): number {
		return this.session?.flushSteeringNow() ?? 0;
	}

	getFollowUpMessages(): string[] {
		return this.session?.getQueues().followUp ?? [];
	}

	getNextTurnMessages(): string[] {
		return this.session?.getQueues().nextTurn ?? [];
	}

	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		const q = this.session?.clearQueues() ?? {
			steering: [],
			followUp: [],
			nextTurn: [],
		};
		return q;
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.session?.dropQueuedMessage(displayIndex);
	}

	/** Abort: clear steering/follow-up queues (preserves nextTurn). */
	async abort(): Promise<AbortResult | null> {
		// harness.abort() clears steering/follow-up and emits onQueueChange.
		return (await this.session?.abort()) ?? null;
	}

	/** Execute a slash command (sends as chat message to the agent). */
	sendSlash(raw: string): void {
		const trimmed = raw.trim();
		const result = this.session?.handleQueueSlashCommand(trimmed);
		if (result?.handled) {
			this.emit({
				type: "notice",
				level: result.level ?? "info",
				label: "Queue",
				text: result.text ?? "",
			});
			return;
		}
		// /reload — reload settings, skills, extensions, and MCP config
		if (trimmed === "/reload") {
			this.reload().catch(err => this.events.notifyError(err));
			return;
		}
		this.sendMessage(raw).catch(err => this.events.notifyError(err));
	}

	// ── Reload ────────────────────────────────────────────────────────────

	/** Reload: restart the session (like Pi's /reload). */
	private async reload(): Promise<void> {
		// Stop any running turn
		void this.cancel();
		this.running = false;

		// Drop the old harness — conversation starts fresh
		this.session = null;

		// Reload the runtime without splitting memory from the active
		// user-facing conversation session.
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Sync memory session
		this.memory?.resetSession(this.sessionId);

		// Reset state that is per-session
		this.toolRouter.resetSkillsAndPrompts();
		this.startupHooksRan = false;
		this.startupHooksPromise = null;
		this.startupHookResult = null;

		// ── Re-discover skills and prompts ────────────────────────────────
		await this.injectSkillsFromPlugins();
		await this.injectPrompts();

		// ── Re-load extensions ────────────────────────────────────────────
		await this.extensions
			?.reload()
			.catch(err => console.error("[logician] extension reload error:", err));

		// ── Re-discover MCP servers ───────────────────────────────────────
		await this.loadMcpToolsOnce();

		// Send reload confirmation (not via sendMessage to avoid starting a turn)
		this.emit({
			type: "turn_end",
			turnId: "reload",
		});
	}

	// ── Skill invocation ───────────────────────────────────────────────

	/** Skills discovered at startup (for /<skill-name> completion). */
	getSkills(): Skill[] {
		return this._loadedSkills;
	}

	/**
	 * Invoke a skill by name as a user prompt: sends the skill's full body
	 * (plus any arguments) to the agent. Returns false for unknown names so the
	 * caller can fall back to normal slash handling.
	 */
	invokeSkill(name: string, args: string): boolean {
		const skill = findSkillByName(this._loadedSkills, name);
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
		this.sendMessage(message).catch(err => this.events.notifyError(err));
		return true;
	}

	/** Prompts discovered at startup (for /<prompt-name> completion). */
	getPrompts(): Prompt[] {
		return this._loadedPrompts;
	}

	/**
	 * Invoke a prompt by name as a user message: sends the prompt's body
	 * (with $ARGUMENTS substituted, or arguments appended) directly — no XML
	 * wrapping, unlike invokeSkill, since a prompt is meant to read exactly as
	 * if the user had typed it. Returns false for unknown names so the caller
	 * can fall back to normal slash handling.
	 */
	invokePrompt(name: string, args: string): boolean {
		const prompt = findPromptByName(this._loadedPrompts, name);
		if (!prompt) return false;
		const trimmedArgs = args.trim();
		const substitutes = prompt.content.includes("$ARGUMENTS");
		const message = substitutes
			? prompt.content.replaceAll("$ARGUMENTS", trimmedArgs)
			: trimmedArgs
				? `${prompt.content}\n\n${trimmedArgs}`
				: prompt.content;
		this.sendMessage(message).catch(err => this.events.notifyError(err));
		return true;
	}

	// ── Permissions & interactive questions (inlined from InteractionCoordinator) ─

	/** Answer a pending permission_request. Returns false for unknown ids. */
	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean {
		return this.interactions.respondToPermission(toolCallId, decision);
	}

	/**
	 * Answer a pending question by id. The answer is forwarded to the agent's
	 * resolver. Returns false if the question id is unknown.
	 */
	respondToQuestion(questionId: string, answer: string): boolean {
		return this.interactions.respondToQuestion(questionId, answer);
	}

	/** Deny every pending permission request (abort / shutdown). */
	private denyPendingPermissions(): void {
		this.interactions.denyPending();
	}

	setPermissionMode(mode: PermissionMode): void {
		this.interactions.setMode(mode);
	}

	getPermissionMode(): PermissionMode {
		return this.interactions.mode;
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

	async getState(): Promise<Record<string, unknown>> {
		// Status is a snapshot, not a synchronization barrier for external MCP
		// transports. The manager UI provides explicit awaited refresh operations.
		if (!this.toolRouter.isMcpLoaded() && !this.toolRouter.isMcpLoading()) {
			void this.toolRouter
				.loadMcpToolsOnce()
				.catch(error => this.events.reportError(error));
		}
		this.contextTokens = this.measureContextTokens();
		const toolNames =
			this.session?.tools?.list().map((t: Tool) => t.name) ||
			this._defaultTools.map(t => t.name);
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
			runtime_state: this.session?.runtimeState ?? {
				phase: "idle",
				isStreaming: false,
				pendingToolCalls: [],
				abortRequested: false,
			},
			config_path: this.configPath || "",
			connected: true,
			reasoner: this.agentCoordinator?.getReasonerStatus() ?? "none",
		};
		return state;
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
			this.startupHooksPromise = null;
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

	private setThinkingLevel(level: string): void {
		this.config.thinkingLevel = level as
			| "off"
			| "minimal"
			| "low"
			| "medium"
			| "high"
			| "xhigh";
		this.session?.models.setThinkingLevel(level);
		// Also update the backend's default so future turns pick it up.
		(this.backend as OpenAIBackend).setDefaultThinkingLevel(
			level as "off" | "minimal" | "low" | "medium" | "high" | "xhigh",
		);
	}

	private setTemperature(temperature: number): void {
		this.config.temperature = temperature;
		this.session?.configure({ temperature });
	}

	private setReasonerId(reasonerId: string): void {
		this.agentCoordinator?.setReasonerId(reasonerId);
	}

	getReasonerStatus(): string {
		return this.agentCoordinator?.getReasonerStatus() ?? "none";
	}

	/** Directly invoke the spawn_agent tool without going through the LLM. */
	spawnAgentDirectly(task: string, agent?: string): void {
		this.agentCoordinator?.spawnAgentDirectly(task, agent);
	}

	/** Whether MCP discovery has started but not finished — lets the TUI show
	 * a "loading" status while background discovery is in flight. */
	isMcpLoading(): boolean {
		return this.toolRouter.isMcpLoading();
	}

	private applyRuntimeToggle(key: RuntimeToggleKey, enabled: boolean): void {
		if (key === "memoryEnabled") {
			this.memory?.setEnabled(enabled, this.sessionId);
			this.emit({
				type: "notice",
				level: "info",
				label: "Memory",
				text: enabled ? "Memory enabled" : "Memory disabled",
			});
			return;
		}
		if (key === "postEditDiagnostics") {
			this.postEditDiagnosticsEnabled = enabled;
			return;
		}
		if (key === "ariadneEnabled") {
			this.config.ariadneEnabled = enabled;
			this.toolRouter.setAriadneEnabled(enabled);
			this.config.tools = this._defaultTools;
			this.session?.configure({ tools: this.config.tools });
			return;
		}
		if (key === "fffgrepEnabled") {
			this.config.fffgrepEnabled = enabled;
			this.toolRouter.setFffgrepEnabled(enabled);
			this.config.tools = this._defaultTools;
			this.session?.configure({ tools: this.config.tools });
			return;
		}
		this.config[key] = enabled;
		if (
			key === "guardsEnabled" ||
			key === "duplicateGuardEnabled" ||
			key === "failureGuardEnabled" ||
			key === "progressStopEnabled" ||
			key === "continuationEnabled" ||
			key === "autoRetryEnabled"
		) {
			this.session?.configure({ [key]: enabled });
		}
		if (key === "proactiveCompactionEnabled") {
			this.session?.enableAutoCompaction(enabled);
		}
	}

	updateSettings(patch: RuntimeSettingsPatch): void {
		this.settings.update(patch);
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
		ariadneEnabled: boolean;
		fffgrepEnabled: boolean;
		memoryEnabled: boolean;
		duplicateGuardEnabled: boolean;
		failureGuardEnabled: boolean;
		continuationEnabled: boolean;
		autoRetryEnabled: boolean;
		progressStopEnabled: boolean;
		guardMode: "auto" | "on" | "off";
	} {
		return this.settings.read();
	}

	getMemoryStore(): ReturnType<
		typeof import("@logician/log-memory").createMemoryStore
	> | null {
		return this.memory?.getStore() ?? null;
	}

	getMemoryStats(): {
		memoryEnabled: boolean;
		memoryCount: number;
		sessionCount: number;
		observationCount: number;
		viewerPort?: number;
	} {
		if (!this.memory) {
			return {
				memoryEnabled: false,
				memoryCount: 0,
				sessionCount: 0,
				observationCount: 0,
			};
		}
		return this.memory.getStats(this.sessionId);
	}

	/** Use the user-facing conversation session as the hook and memory session. */
	useConversationSession(sessionId: string, durableSession?: Session): void {
		if (!sessionId.trim()) return;
		const provisionalSessionId = this.sessionId;
		this.sessionId = sessionId;
		this.durableSession = durableSession;
		this.session?.setSessionId(sessionId);
		if (durableSession) this.session?.attachSession(durableSession);
		this.transcriptPath = createHookTranscriptPath(this.cwd, sessionId);
		this.config.hookSessionId = sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Sync memory session
		this.memory?.onSessionChanged(sessionId, provisionalSessionId);
	}

	renameConversationSession(sessionId: string, name: string): void {
		this.memory?.renameSession(sessionId, name);
	}

	reset(): void {
		// Reset tool state and conversation
		void this.fireSessionEnd("reset");
		// Drop the persisted harness so history starts fresh.
		this.session?.clearHistory();
		this.session = null;
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		this.config.hookSessionId = this.sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Update memory session ID
		this.memory?.resetSession(this.sessionId);
		// Reset skill/prompt injection state
		this.toolRouter.resetInjectedContext();
		this.startupHooksRan = false;
		this.startupHooksPromise = null;
		this.applyPluginHookContext(
			this.startupHookResult || {
				additional_contexts: [],
				context_messages: [],
				initial_user_message: "",
			},
		);
		this.contextTokens = 0;
		this.publishContextUsage();
		this.emit({
			type: "turn_end",
			turnId: "reset",
		});
	}

	async cancel(): Promise<AbortResult | null> {
		// A turn blocked on an approval must unblock to abort cleanly.
		this.denyPendingPermissions();
		try {
			return (await this.session?.abort()) ?? null;
		} catch (error) {
			const normalized =
				error instanceof Error ? error : new Error(String(error));
			this.events.notifyError(normalized);
			throw normalized;
		}
	}

	/** Manual context compaction. Returns { tokensSaved, tokensBefore, tokensAfter } or null if nothing to compact. */
	async compact(): Promise<{
		tokensSaved: number;
		tokensBefore: number;
		tokensAfter: number;
	} | null> {
		if (!this.session) return null;
		const saved = await this.session.compact();
		if (saved === null) return null;
		// Re-emit context update with new token count
		const messages = this.session.messages;
		const after = estimateChatPayloadTokens(messages);
		const before = after + saved;
		this.contextTokens = after;
		this.emit({
			type: "compaction",
			reason: "manual",
			tokensBefore: before,
			tokensAfter: after,
		} as RuntimeEvent);
		return { tokensSaved: saved, tokensBefore: before, tokensAfter: after };
	}

	// ── Conversation branching ─────────────────────────────────────────────

	/** Fork the conversation; returns the new branch id, or null if no harness. */
	fork(): string | null {
		return this.session?.fork() ?? null;
	}

	/**
	 * Summarize the active branch and merge it back into the parent. Returns the
	 * summary text, or null if nothing to summarize / no harness.
	 */
	async branchSummary(): Promise<string | null> {
		if (!this.session) return null;
		const summary = await this.session.branchSummary();
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
			const restored = this.session?.rewind() ?? null;
			if (restored !== null && this.session) {
				this.publishContextUsage();
			}
			return restored;
		} catch (_e: unknown) {
			return null;
		}
	}

	/** Discard the active branch without merging. Returns true if one was discarded. */
	discardBranch(): boolean {
		const discarded = this.session?.discardBranch() ?? false;
		if (discarded && this.session) this.publishContextUsage();
		return discarded;
	}

	// ── State management ─────────────────────────────────────────────────

	async init(): Promise<Record<string, unknown>> {
		await this.extensions?.getLoadPromise();
		await this.runStartupHooksOnce();
		this.ensureSession();
		// MCP discovery already started in ToolRouter's constructor — fire-
		// and-forget from the moment the bridge exists, not gated behind
		// init() or the first message. loadMcpToolsOnce() is memoized, so this
		// just observes the same in-flight/settled load instead of starting a
		// second one; the "Loaded N server(s)" notice fires once, from inside
		// loadMcpToolsOnce() itself, whenever that load actually finishes.
		void this.loadMcpToolsOnce().catch(error => this.events.reportError(error));
		const toolNames =
			this.session?.tools?.list().map((t: Tool) => t.name) ||
			this._defaultTools.map(t => t.name);
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
			runtime_state: this.session?.runtimeState ?? {
				phase: "idle",
				isStreaming: false,
				pendingToolCalls: [],
				abortRequested: false,
			},
			config_path: this.configPath || "",
			hooks_enabled: this.config.runtimeHooksEnabled !== false,
			hook_transcript_path: this.config.hookTranscriptPath || "",
			startup_plugins_loaded: this.startupPluginCount,
			startup_plugins: status.enabledPluginRoots.map(plugin => plugin.name),
			startup_hooks_loaded: this.startupHookResult?.hook_count || 0,
			startup_hook_contexts: this.startupHookResult?.additional_contexts || [],
			startup_hook_messages: this.startupHookResult?.context_messages || [],
			startup_hook_initial_message:
				this.startupHookResult?.initial_user_message || "",
			startup_hook_errors: this.startupHookResult?.errors || [],
			skills_injected: status.skillsInjected
				? status.loadedSkills.filter(skill => !skill.disableModelInvocation)
						.length
				: 0,
			skills_visible: status.skillsVisible,
			loaded_skills: status.loadedSkills.map(skill => ({
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

	getExtensionCommands(): Array<{
		name: string;
		description: string;
		usage?: string;
		acceptsArgs?: boolean;
	}> {
		return this.extensions?.getCommands() ?? [];
	}

	invokeExtensionCommand(
		name: string,
		args: string,
	): Promise<string | undefined> {
		return (
			this.extensions?.executeCommand(name, args) ?? Promise.resolve(undefined)
		);
	}

	async stop(): Promise<void> {
		void this.cancel();
		// Abort extraction and wait for background tasks to complete.
		await this.memory?.waitForBackgroundTasks();
		await this.fireSessionEnd("shutdown");
		this.lsp.close();
		await this.toolRouter.closeMcp();
		killAllTrackedChildren();
		this.running = false;
	}

	isActive(): boolean {
		return this.running;
	}

	getMessages(): Message[] {
		return this.session?.messages || [];
	}

	/** Return full context as formatted text for /context command. */
	getContext(): string {
		const msgs = this.getMessages();
		const memoryContext = this.getMemoryContextForInspection(msgs);
		const contextTokens =
			this.measureContextTokens() + estimateTokens(memoryContext);
		this.contextTokens = contextTokens;

		const sourceMap = this.getContextSourceMap(memoryContext);
		const sourceLines = sourceMap.map(
			zone =>
				`- ${zone.name}: ~${zone.tokens} tokens${zone.detail ? ` — ${zone.detail}` : ""}`,
		);
		const lines: string[] = ["## Prompt source map", "", ...sourceLines, ""];
		lines.push("## Effective context", "");

		// System prompt zone (the static identity + tool instructions).
		// Session history filters out system messages, so we pull the current
		// config system prompt and show it at the top of the context dump.
		const systemPrompt = this.config.systemPrompt || "";
		if (!msgs.length && !systemPrompt && !memoryContext)
			lines.push("No messages yet.");
		if (systemPrompt) {
			lines.push("[SYSTEM] system prompt");
			lines.push(systemPrompt);
			lines.push("");
		}

		// Memory retrieval is a request-time context block, injected between
		// the system prompt and conversation turns. Show it in the correct
		// position so /context reflects the effective provider payload.
		if (memoryContext) {
			lines.push("[SYSTEM] Memory Context");
			lines.push(memoryContext);
			lines.push("");
		}

		// Conversation turns (user / assistant / tool messages).
		// System messages are already shown separately above, so skip them here.
		for (const msg of msgs) {
			if (!msg) continue;
			if (msg.role === "system") continue;
			const role = msg.role.toUpperCase();
			const ts = msg.timestamp ? new Date(msg.timestamp).toISOString() : "";
			const header = `[${role}]${ts ? ` ${ts}` : ""}`;

			if (msg.role === "tool" && msg.content) {
				// Tool result: show the originating tool name alongside its content.
				const callId = msg.tool_call_id || "";
				const name =
					msgs
						.find(
							m =>
								m.role === "assistant" &&
								m.tool_calls?.some(tc => tc.id === callId),
						)
						?.tool_calls?.find(tc => tc.id === callId)?.name || "tool";
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

	private getMemoryContextForInspection(messages: Message[]): string {
		if (!this.memory) return "";
		// Cast: Message content is string | null but manager expects string | undefined
		const msgArray = messages as Array<{ role: string; content?: string }>;
		return this.memory.getContextForInspection(msgArray);
	}

	getContextSourceMap(memoryContext: string = ""): Array<{
		name: string;
		tokens: number;
		detail: string;
	}> {
		const messages = this.getMessages();
		const toolDefinitions = this.getTools().toToolDefinitions();
		const conversation = messages.filter(message => message.role !== "tool");
		const toolEvidence = messages.filter(message => message.role === "tool");
		return [
			{
				name: "Base instructions",
				tokens: estimateTokens(this.config.systemPrompt || ""),
				detail: "system zone",
			},
			{
				name: "Plugin context",
				tokens: estimateTokens(""),
				detail: "startup hooks",
			},
			{
				name: "Tool definitions",
				tokens: estimateChatPayloadTokens([], toolDefinitions),
				detail: `${toolDefinitions.length} tools`,
			},
			{
				name: "Retrieved memory",
				tokens: estimateTokens(memoryContext),
				detail: memoryContext ? "request-time compact index" : "none retrieved",
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
		].filter(zone => zone.tokens > 0 || zone.name === "Conversation");
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
			maxTokens: this.contextMaxTokens,
			compacted: false,
		});
	}

	getTools(): ToolRegistry {
		const live = this.session?.tools;
		if (live) return live;
		return buildToolRegistry(this.toolRouter, {
			cwd: this.config.cwd,
			allowedPaths: this.config.allowedPaths,
			allowAllPaths: this.config.allowAllPaths,
			cacheSize: this.config.cacheSize,
			cacheTtlMs: this.config.cacheTtlMs,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
	}

	/**
	 * Runs plugin listing, session-start hooks, and skill/prompt/subagent
	 * injection exactly once per (re)set of `startupHooksRan`. Memoized as a
	 * promise — not just the boolean flag — so a caller that starts this
	 * eagerly (e.g. right after construction, to prewarm before the user's
	 * first message) and a concurrent `runMessage()` call both await the same
	 * in-flight work instead of the second caller seeing the flag flip early
	 * and racing ahead of hook/skill injection.
	 */
	private async runStartupHooksOnce(source = "startup"): Promise<void> {
		if (this.startupHooksRan) return;
		if (!this.startupHooksPromise) {
			this.startupHooksPromise = this.runStartupHooksNow(source).then(() => {
				this.startupHooksRan = true;
			});
		}
		await this.startupHooksPromise;
	}

	private async runStartupHooksNow(source: string): Promise<void> {
		const snapshot = await runPluginBackend("list", []);
		this.startupPluginCount = (snapshot.plugins || []).filter(plugin => {
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
		await this.agentCoordinator?.injectSubagents();
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

	/**
	 * Execute a bash command directly (for user_bash / !command in the input bar).
	 * Returns the command output and exit code.
	 */
	async executeBashCommand(command: string): Promise<{
		output: string;
		exitCode: number;
	}> {
		const { spawn } = await import("node:child_process");
		const { getShellConfig } = await import(
			"../../capabilities/tools/support/utils/shell.ts"
		);
		const { shell, args: shellArgs } = getShellConfig();

		return new Promise((resolve, reject) => {
			const child = spawn(shell, [...shellArgs, command], {
				cwd: this.cwd,
				stdio: ["ignore", "pipe", "pipe"],
			});

			let output = "";
			child.stdout?.on("data", (data: Buffer) => {
				output += data.toString();
			});
			child.stderr?.on("data", (data: Buffer) => {
				output += data.toString();
			});

			child.on("close", (code: number | null) => {
				resolve({
					output: output || "(no output)",
					exitCode: code ?? 1,
				});
			});

			child.on("error", (err: Error) => {
				reject(err);
			});
		});
	}
}

export { getSkillsDirs } from "../resource-directories.ts";

/** @deprecated Import and use {@link AgentRuntime}. */
export { AgentRuntime as AgentCoreBridge };
