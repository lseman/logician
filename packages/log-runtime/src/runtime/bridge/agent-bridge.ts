/** Coordinates one interactive agent session and its runtime integrations. */

import type { AgentConfig, Message, QueueMode, Tool } from "@logician/log-core";
import { OpenAIBackend } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import type { PermissionMode } from "@logician/log-core/permissions";
import type { AbortResult, SessionStore } from "@logician/log-core/runtime";
import {
	estimateChatPayloadTokens,
	type ToolRegistry,
} from "@logician/log-core/runtime";
import type { AgentSession } from "@logician/log-core/session";
import {
	configurePluginRuntimeEnv,
	type PluginCommandResult,
} from "../../adapters/claude-code/plugin-runtime.ts";
import type { ExtensionRegistry } from "../../capabilities/extensions/extensions.ts";
import type { InteractionGateway } from "../../capabilities/interactions/interaction-gateway.ts";
import { LegroomGateway } from "../../capabilities/legroom/legroom-gateway.ts";
import type {
	CalibrationStatus,
	CompressResult,
	LegroomWorker,
	StoreStats,
	WorkerHistory,
	WorkerStats,
} from "../../capabilities/legroom/worker.ts";
import type { LspClientPool } from "../../capabilities/lsp/lsp-client-pool.ts";
import { createPostEditDiagnosticHooks } from "../../capabilities/lsp/post-edit-diagnostics.ts";
import type {
	McpSnapshotResult,
	McpToggleResult,
} from "../../capabilities/mcp/mcp-server-registry.ts";
import { MemoriamGateway } from "../../capabilities/memoriam/memoriam-gateway.ts";
import type { MemoriamWorker } from "../../capabilities/memoriam/worker.ts";
import type { Prompt } from "../../capabilities/prompts/loader.ts";
import type { RepositoryMap } from "../../capabilities/repository-map/repository-map.ts";
import type { Skill } from "../../capabilities/skills/loader.ts";
import { onTodosChanged } from "../../capabilities/tasks/todo.ts";
import type { SandboxProfile } from "../../capabilities/tools/sandbox.ts";
import { killAllTrackedChildren } from "../../capabilities/tools/support/utils/shell.ts";
import {
	contextSources,
	inspectContext,
} from "../context/context-inspector.ts";
import { buildDefaultSystemPrompt } from "../context/system-prompt.ts";
import { RuntimeEventBus } from "../events/runtime-event-bus.ts";
import { createAgentConfig } from "./application/agent-config-factory.ts";
import { AgentCoordinator } from "./application/agent-coordinator.ts";
import { CommandDispatcher } from "./application/command-dispatcher.ts";
import { ConversationIdentity } from "./application/conversation-identity.ts";
import { ConversationSession } from "./application/conversation-session.ts";
import { PluginLifecycle } from "./application/plugin-lifecycle.ts";
import { executeProcessCommand } from "./application/process-command.ts";
import { RuntimeActivity } from "./application/runtime-activity.ts";
import { RuntimeConfiguration } from "./application/runtime-configuration.ts";
import { RuntimeLifecycle } from "./application/runtime-lifecycle.ts";
import {
	projectInitializationStatus,
	projectRuntimeStatus,
	runtimeToolNames,
} from "./application/runtime-status.ts";
import { TurnOrchestrator } from "./application/turn-orchestrator.ts";
import {
	createRuntimeContext,
	type RuntimeContext,
} from "./capability-context.ts";
import {
	buildPluginRuntimeEnv,
	resolveWebSearchConfig,
} from "./environment.ts";
import { SessionRunner } from "./session-runner.ts";
import { ModelSelector } from "./support/model-selector.ts";
import { buildToolRegistry } from "./support/runtime-context.ts";
import { ToolRouter } from "./support/tool-router.ts";

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
	private readonly sessions: ConversationSession;
	private readonly commands: CommandDispatcher;
	private get session(): AgentSession | null {
		return this.sessions?.current ?? null;
	}
	readonly events: RuntimeEventBus;
	readonly models: ModelSelector;
	private readonly turns: TurnOrchestrator;
	private readonly lifecycle: RuntimeLifecycle;

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
		this.plugins.refreshContext();
	}

	async injectSkillsFromPlugins(): Promise<void> {
		await this.toolRouter.injectSkillsFromPlugins();
		this.plugins.refreshContext();
	}

	async injectPrompts(): Promise<void> {
		await this.toolRouter.injectPrompts();
	}

	private readonly runtimeCtx: RuntimeContext;
	private get extensions(): ExtensionRegistry {
		return this.runtimeCtx.extensions;
	}

	private readonly identity: ConversationIdentity;
	private get sessionId(): string {
		return this.identity.id;
	}
	private get transcriptPath(): string {
		return this.identity.transcript;
	}
	private readonly plugins: PluginLifecycle;
	private readonly activity: RuntimeActivity;
	private configPath: string | null;
	private readonly runtimeConfiguration: RuntimeConfiguration;
	private get lsp(): LspClientPool {
		return this.runtimeCtx.lsp;
	}
	private readonly projectTrusted: boolean;
	private get interactions(): InteractionGateway {
		return this.runtimeCtx.interactions;
	}
	private agentCoordinator: AgentCoordinator | null = null;
	private readonly sessionRunner: SessionRunner;
	private readonly legroom: LegroomGateway;
	private get legroomEnabled(): boolean {
		return this.legroom.isEnabled();
	}

	private readonly memoriam: MemoriamGateway;
	private get memoriamEnabled(): boolean {
		return this.memoriam.isEnabled();
	}

	private get repositoryMap(): RepositoryMap | undefined {
		return this.runtimeCtx.repositoryMap;
	}
	private readonly compactionSettings?: AgentBridgeOptions["compaction"];
	private readonly unsubscribeTodos: () => void;

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
		this.events = new RuntimeEventBus({
			historyCapacity: opts.eventStream?.historyCapacity,
		});
		this.compactionSettings = opts.compaction;
		this.cwd = opts.cwd || process.cwd();
		this.identity = new ConversationIdentity(
			`tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
			{
				cwd: this.cwd,
				config: () => this.config,
				sessions: () => this.sessions,
				events: () => this.events,
			},
		);
		this.projectTrusted = opts.projectTrusted === true;
		this.configPath = opts.configPath ?? null;
		configurePluginRuntimeEnv(buildPluginRuntimeEnv(opts));
		const postEditDiagnosticsEnabled =
			process.env.LOGICIAN_POST_EDIT_DIAGNOSTICS === "0"
				? false
				: opts.postEditDiagnostics !== false;
		this.legroom = new LegroomGateway(opts.legroom ?? {});
		this.memoriam = new MemoriamGateway(opts.memoriam ?? {});

		this.runtimeCtx = createRuntimeContext({
			opts,
			cwd: this.cwd,
			sessionId: this.sessionId,
			projectTrusted: this.projectTrusted,
			emit: event => this.emit(event),
		});
		// Extension loading is a real side effect (reads disk, may run init
		// hooks) — kept as an explicit statement rather than folded into
		// register(), which the context treats as pure construction.
		void this.extensions.initialize();

		const defaultWebSearch = resolveWebSearchConfig();
		const webSearch = {
			baseUrl: opts.webSearch?.baseUrl || defaultWebSearch.baseUrl,
			maxResults: opts.webSearch?.maxResults ?? defaultWebSearch.maxResults,
		};
		const extraTools = opts.extraTools?.length
			? [...opts.extraTools]
			: undefined;
		this.toolRouter = new ToolRouter({
			cwd: this.cwd,
			projectTrusted: this.projectTrusted,
			tools: opts.tools,
			extraTools,
			webSearch,
			graphicianEnabled: opts.graphicianEnabled,
			fffgrepEnabled: opts.fffgrepEnabled,
			autoStartMcp: false,
			emit: event => this.emit(event),
			onToolAdded: _tool => {
				if (!this.config) return;
				this.config.tools = this.toolRouter.getDefaultTools();
				this.session?.configure({ tools: this.config.tools });
			},
			onContextChanged: () => {
				if (this.baseSystemPrompt) this.plugins?.refreshContext();
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
			thinkingFormat: opts.thinkingFormat,
		});
		if (opts.thinkingLevel) {
			this.backend.setDefaultThinkingLevel(opts.thinkingLevel);
		}

		this.config = createAgentConfig({
			bridge: opts,
			cwd: this.cwd,
			sessionId: this.sessionId,
			transcriptPath: this.transcriptPath,
			systemPrompt: this.baseSystemPrompt,
			tools: this._defaultTools,
			webSearch,
			permissions: this.interactions.permissions,
			onPermissionRequest: context =>
				this.interactions.requestPermission(context),
			onQuestionRequest: context => this.interactions.requestQuestion(context),
			hooks: this.buildMemoriamHooks(
				this.buildLegroomHooks(
					createPostEditDiagnosticHooks(
						this.cwd,
						() => this.runtimeConfiguration.postEditDiagnostics,
						opts.lsp?.enabled === false ? undefined : this.lsp,
						{
							allowedPaths: opts.allowedPaths,
							allowAllPaths: opts.allowAllPaths,
						},
					),
				),
			),
			onTurnEnd: turnId => this.emit({ type: "turn_end", turnId }),
			onEvent: event => this.activity.handle(event),
		});
		this.activity = new RuntimeActivity({
			emit: event => this.emit(event),
			runPhase: () => this.session?.phase,
		});
		this.sessions = new ConversationSession(
			{
				config: () => this.config,
				backend: this.backend,
				extensions: () => this.extensions,
				emit: event => this.emit(event),
				contextChanged: () => this.publishContextUsage(),
				contextCompacted: tokens => {
					this.activity.setContext(tokens);
				},
				compaction: this.compactionSettings,
			},
			this.sessionId,
		);
		this.commands = new CommandDispatcher({
			session: () => this.session,
			skills: () => this._loadedSkills,
			prompts: () => this._loadedPrompts,
			sendMessage: message => this.sendMessage(message),
			reload: () => this.reload(),
			emit: event => this.emit(event),
			reportError: error => this.events.notifyError(error),
		});
		this.models = new ModelSelector(
			() => this.config,
			() => this.session,
		);
		this.unsubscribeTodos = onTodosChanged(todos => {
			this.emit({ type: "todos", todos });
		});

		// Create agent coordinator for reasoner, EoH, and subagents
		this.agentCoordinator = new AgentCoordinator(
			{
				emit: event => this.emit(event),
				getBackend: () => this.backend,
				getConfig: () => this.config,
				getBaseUrl: () => this.config.baseUrl,
				getCurrentModel: () => this.models.current(),
				cwd: this.cwd,
				projectTrusted: this.projectTrusted,
				maxParallelAgents: opts.maxParallelAgents,
				getEnabledPluginRoots: () => this._enabledPluginRoots,
				getDefaultTools: () => this._defaultTools,
				ensureSession: () => this.ensureSession(),
				reportError: error =>
					this.events.reportError(error, {
						component: "agent-coordinator",
						operation: "capability-run",
						recoverable: true,
					}),
			},
			opts.reasoner,
			opts.reasonerConfig,
		);
		this.runtimeConfiguration = new RuntimeConfiguration({
			config: this.config,
			backend: this.backend,
			session: () => this.session,
			sessionId: () => this.sessionId,
			tools: this.toolRouter,
			interactions: this.interactions,
			legroom: this.legroom,
			memoriam: this.memoriam,
			defaultTools: () => this._defaultTools,
			setReasoner: id => this.agentCoordinator?.setReasonerId(id),
			emit: event => this.emit(event),
			postEditDiagnostics: postEditDiagnosticsEnabled,
		});
		this.plugins = new PluginLifecycle({
			config: () => this.config,
			baseSystemPrompt: () => this.baseSystemPrompt,
			sessionId: () => this.sessionId,
			tools: this.toolRouter,
			injectSubagents: async () => {
				await this.agentCoordinator?.injectSubagents();
			},
		});

		this.sessionRunner = new SessionRunner({
			callbacks: {
				emit: event => this.emit(event),
				reportError: (error, context) =>
					this.events.reportError(error, context),
				getSession: () => this.session,
				ensureSession: () => this.ensureSession(),
				getSessionId: () => this.sessionId,
				getSystemPrompt: () => this.config.systemPrompt,
				getSkills: () => this._loadedSkills,
				renderRepositoryContext: message => this.repositoryMap?.render(message),
				publishUsage: () => this.publishContextUsage(),
			},
			events: this.events,
			backend: this.backend,
			getAgentCoordinator: () => this.agentCoordinator,
			getRepositoryMap: () => this.repositoryMap,
		});
		this.turns = new TurnOrchestrator({
			extensionsReady: () => this.extensions.getLoadPromise(),
			hasSession: () => this.session !== null,
			steer: message => this.sessions.queues.steer(message),
			emit: event => this.emit(event),
			ensureStartup: () => this.plugins.ensureStarted(),
			isMcpLoaded: () => this.toolRouter.isMcpLoaded(),
			loadMcp: () => this.toolRouter.loadMcpToolsOnce(),
			reportMcpError: error =>
				this.events.reportError(error, {
					component: "mcp",
					operation: "background-discovery",
					recoverable: true,
				}),
			runTurn: message => this.sessionRunner.submit(message),
		});
		this.lifecycle = new RuntimeLifecycle({
			cancel: () => this.cancel(),
			resetTurns: () => this.turns.reset(),
			dropSession: () => this.sessions.drop(),
			clearSession: () => this.sessions.clearAndDrop(),
			resetIdentity: () => this.identity.reset(),
			endPluginSession: reason => this.plugins.endSession(reason),
			resetPlugin: options => this.plugins.reset(options),
			refreshPluginContext: () => this.plugins.refreshContext(),
			resetInjectedContext: () => this.toolRouter.resetInjectedContext(),
			resetDiscoveredResources: () => this.toolRouter.resetSkillsAndPrompts(),
			injectSkills: () => this.injectSkillsFromPlugins(),
			injectPrompts: () => this.injectPrompts(),
			reloadExtensions: () => this.extensions.reload(),
			reportExtensionError: error =>
				this.events.reportError(error, {
					component: "extensions",
					operation: "reload",
					recoverable: true,
				}),
			extensionsReady: () => this.extensions.getLoadPromise(),
			ensurePluginsStarted: () => this.plugins.ensureStarted(),
			ensureSession: () => {
				this.ensureSession();
			},
			loadMcp: () => this.loadMcpToolsOnce(),
			reportMcpError: error =>
				this.events.reportError(error, {
					component: "mcp",
					operation: "reload-tools",
					recoverable: true,
				}),
			closeResources: async () => {
				this.lsp.close();
				this.legroom.close();
				this.memoriam.close();
				await this.toolRouter.closeMcp();
				killAllTrackedChildren();
			},
			resetActivity: () => this.activity.resetContext(),
			publishUsage: () => this.publishContextUsage(),
			emitTurnEnd: turnId => this.emit({ type: "turn_end", turnId }),
		});
	}

	private buildLegroomHooks(
		existingHooks: AgentConfig["hooks"],
	): AgentConfig["hooks"] {
		return this.legroom.createHooks(existingHooks);
	}

	/** Build Memoriam hooks by delegating to the MemoriamGateway. */
	private buildMemoriamHooks(
		existingHooks: AgentConfig["hooks"],
	): AgentConfig["hooks"] {
		return this.memoriam.createHooks(existingHooks);
	}

	/** Add a tool to the default set and propagate it into live config/harness/system prompt. */
	/** Propagate a tool the router just registered into live config/harness/system prompt. */
	// ── Event registration ─────────────────────────────────────────────────

	private emit(event: RuntimeEvent): void {
		this.events.emit(event);
	}

	// ── High-level commands ──────────────────────────────────────────────

	async sendMessage(message: string): Promise<void> {
		return this.turns.submit(message);
	}

	// Lazily build the singleton harness and wire its UI callbacks.
	private ensureSession(): AgentSession {
		return this.sessions.ensure();
	}

	/**
	 * Replace the harness conversation with restored session history (resume /
	 * session switch), so the model continues with the restored context instead
	 * of starting cold. Pass [] to clear (new session). No-op while a turn is
	 * running (the harness rejects structural ops mid-turn).
	 */
	restoreHistory(messages: Message[]): boolean {
		return this.sessions.restoreHistory(messages);
	}

	// ── Queue operations (delegated to AgentSession) ─────────────────

	steer(message: string): void {
		this.sessions.queues.steer(message);
	}

	/** Queue steering for after the current turn (never interrupts). */
	steerQueue(message: string): void {
		this.sessions.queues.steerQueue(message);
	}

	/** Immediately interrupt and apply steering (always forces abort). */
	steerNow(message: string): void {
		this.sessions.queues.steerNow(message);
	}

	followUp(message: string): void {
		this.sessions.queues.followUp(message);
	}

	nextTurn(message: string): void {
		this.sessions.queues.nextTurn(message);
	}

	setSteeringMode(mode: QueueMode): void {
		this.sessions.queues.setSteeringMode(mode);
	}

	getSteeringInterrupt(): boolean {
		return this.config.steeringInterrupt === true;
	}

	/** Return config snapshot for external LLM calls (goal evaluator, etc.). */
	getConfig(): {
		baseUrl: string;
		model: string;
		rtkProxyEnabled?: boolean;
		graphicianEnabled?: boolean;
		fffgrepEnabled?: boolean;
		legroomEnabled?: boolean;
		memoriamEnabled?: boolean;
	} {
		return {
			baseUrl: this.config.baseUrl,
			model: this.config.model,
			rtkProxyEnabled: this.config.rtkProxyEnabled,
			graphicianEnabled: this.config.graphicianEnabled,
			fffgrepEnabled: this.config.fffgrepEnabled,
			legroomEnabled: this.legroomEnabled,
			memoriamEnabled: this.memoriamEnabled,
		};
	}

	setFollowUpMode(mode: QueueMode): void {
		this.sessions.queues.setFollowUpMode(mode);
	}

	getSteeringMessages(): string[] {
		return this.sessions.queues.snapshot().steering;
	}

	flushSteeringNow(): number {
		return this.sessions.queues.flushSteeringNow();
	}

	getFollowUpMessages(): string[] {
		return this.sessions.queues.snapshot().followUp;
	}

	getNextTurnMessages(): string[] {
		return this.sessions.queues.snapshot().nextTurn;
	}

	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return this.sessions.queues.clear();
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.sessions.queues.drop(displayIndex);
	}

	/** Abort: clear steering/follow-up queues (preserves nextTurn). */
	async abort(): Promise<AbortResult | null> {
		// harness.abort() clears steering/follow-up and emits onQueueChange.
		return this.sessions.abort();
	}

	/** Execute a slash command (sends as chat message to the agent). */
	sendSlash(raw: string): void {
		this.commands.dispatchSlash(raw);
	}

	// ── Reload ────────────────────────────────────────────────────────────

	/** Reload: restart the session (like Pi's /reload). */
	private async reload(): Promise<void> {
		await this.lifecycle.reload();
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
		return this.commands.invokeSkill(name, args);
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
		return this.commands.invokePrompt(name, args);
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
			void this.toolRouter.loadMcpToolsOnce().catch(error =>
				this.events.reportError(error, {
					component: "mcp",
					operation: "load-tools",
					recoverable: true,
				}),
			);
		}
		const context = this.activity.setContext(this.measureContextTokens());
		return projectRuntimeStatus({
			config: this.config,
			toolNames: runtimeToolNames(this.session?.tools, this._defaultTools),
			mcpServerCount: this.toolRouter.getMcpServerCount(),
			mcpToolCount: this.toolRouter.getMcpToolCount(),
			mcpErrors: this.toolRouter.getMcpErrors(),
			contextTokens: context.tokens,
			contextMaxTokens: context.maxTokens,
			runtimeState: this.session?.runtimeState,
			configPath: this.configPath,
			reasoner: this.getReasonerStatus(),
		});
	}

	async getPluginSnapshot(): Promise<PluginCommandResult> {
		return this.plugins.snapshot();
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
		return this.plugins.setEnabled(pluginId, enabled);
	}

	async runPluginCommand(input: string): Promise<string> {
		return this.plugins.runCommand(input);
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

	updateSettings(patch: RuntimeSettingsPatch): void {
		this.runtimeConfiguration.update(patch);
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
		graphicianEnabled: boolean;
		fffgrepEnabled: boolean;
		legroomEnabled: boolean;
		memoriamEnabled: boolean;
		duplicateGuardEnabled: boolean;
		failureGuardEnabled: boolean;
		continuationEnabled: boolean;
		autoRetryEnabled: boolean;
		progressStopEnabled: boolean;
		guardMode: "auto" | "on" | "off";
	} {
		return this.runtimeConfiguration.read();
	}

	/** Get the underlying LegroomWorker for advanced CCR store operations. */
	getLegroomWorker(): LegroomWorker | null {
		return this.legroom.worker;
	}

	// ── Legroom CCR Store ──────────────────────────────────────────────────

	/** Compress messages using a named CCR store (enables CCR automatically). */
	async compressWithStore(
		storeId: string,
		messages: Record<string, unknown>[],
		model: string,
	): Promise<CompressResult> {
		return this.legroom.compressWithStore(storeId, messages, model);
	}

	/** Retrieve original content from a CCR store by hash. */
	async storeRetrieve(storeId: string, hash: string): Promise<string> {
		return this.legroom.storeRetrieve(storeId, hash);
	}

	/** Get CCR store statistics. */
	async storeStats(storeId: string): Promise<StoreStats> {
		return this.legroom.storeStats(storeId);
	}

	/** Get aggregate worker statistics (includes CCR store metrics). */
	async getLegroomStats(): Promise<WorkerStats> {
		return this.legroom.workerStats();
	}

	/** Get recent compression request history. */
	async getLegroomHistory(limit = 50, offset = 0): Promise<WorkerHistory> {
		return this.legroom.workerHistory(limit, offset);
	}

	/** Query current calibration state. */
	async getCalibrationStatus(): Promise<CalibrationStatus> {
		return this.legroom.calibrationStatus();
	}

	/** Record quality feedback for phase calibration. */
	async calibrationRecord(
		phaseReports: Record<string, unknown>[],
		quality: number,
	): Promise<CalibrationStatus> {
		return this.legroom.calibrationRecord(phaseReports, quality);
	}

	// ── Memoriam ──────────────────────────────────────────────────────────

	/** Get the underlying MemoriamWorker for advanced memory-store operations. */
	getMemoriamWorker(): MemoriamWorker | null {
		return this.memoriamEnabled ? this.memoriam.worker : null;
	}

	/** Observe a tool interaction or conversation turn. */
	async memoriamObserve(
		sessionId: string,
		hookType: string,
		opts?: {
			toolName?: string;
			toolInput?: unknown;
			toolOutput?: unknown;
			userPrompt?: string;
			raw?: unknown;
		},
	): Promise<unknown> {
		return this.memoriam.observe(sessionId, hookType, opts);
	}

	/** Get memory context as plain text for a query. */
	async memoriamGetContext(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<string> {
		return this.memoriam.getContext(sessionId, query, budget);
	}

	/** Recall memories in a formatted string (markdown/plain). */
	async memoriamRecall(
		query: Record<string, unknown>,
		format: string,
	): Promise<string> {
		return this.memoriam.recall(query, format);
	}

	/** List memories, optionally filtered by a query. */
	async memoriamListMemories(
		query?: Record<string, unknown>,
	): Promise<unknown[]> {
		return this.memoriam.listMemories(query);
	}

	/** Remove a single memory by id. */
	async memoriamRemoveMemory(id: string): Promise<boolean> {
		return this.memoriam.removeMemory(id);
	}

	/** Consolidate a session's observations into memories. */
	async memoriamConsolidate(sessionId: string): Promise<unknown[]> {
		return this.memoriam.consolidate(sessionId);
	}

	/** List observations for a session. */
	async memoriamListObservations(
		sessionId: string,
		limit: number,
	): Promise<unknown[]> {
		return this.memoriam.listObservations(sessionId, limit);
	}

	/** Search observations across sessions. */
	async memoriamSearchObservations(
		query: string,
		limit: number,
	): Promise<unknown[]> {
		return this.memoriam.searchObservations(query, limit);
	}

	/** Clear all observations. Returns the number removed. */
	async memoriamClearObservations(): Promise<number> {
		return this.memoriam.clearObservations();
	}

	/** List memory sessions. */
	async memoriamListSessions(
		query?: Record<string, unknown>,
	): Promise<unknown[]> {
		return this.memoriam.listSessions(query);
	}

	/** Clear stored sessions, optionally keeping one. */
	async memoriamClearSessions(keepSessionId?: string | null): Promise<void> {
		return this.memoriam.clearSessions(keepSessionId ?? null);
	}

	/** Aggregate Memoriam worker statistics. */
	async getMemoriamStats(): Promise<Record<string, unknown>> {
		return this.memoriam.workerStats();
	}

	/** Use the user-facing conversation session as the hook and memory session. */
	useConversationSession(
		sessionId: string,
		durableSession?: SessionStore,
	): void {
		this.identity.use(sessionId, durableSession);
	}

	/** Rename the Memoriam session metadata (best-effort). */
	async renameConversationSession(
		sessionId: string,
		name: string,
	): Promise<void> {
		if (!this.memoriamEnabled) return;
		try {
			await this.memoriam.updateSession(sessionId, { name: name.trim() });
		} catch {
			// Fail open — memory metadata rename is not load-bearing.
		}
	}

	reset(): void {
		this.lifecycle.reset();
	}

	async cancel(): Promise<AbortResult | null> {
		// A turn blocked on an approval must unblock to abort cleanly.
		this.denyPendingPermissions();
		try {
			return await this.sessions.abort();
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
		return this.sessions.compact();
	}

	// ── Conversation branching ─────────────────────────────────────────────

	/** Fork the conversation; returns the new branch id, or null if no harness. */
	fork(): string | null {
		return this.sessions.fork();
	}

	/**
	 * Summarize the active branch and merge it back into the parent. Returns the
	 * summary text, or null if nothing to summarize / no harness.
	 */
	async branchSummary(): Promise<string | null> {
		return this.sessions.branchSummary();
	}

	/**
	 * Rewind to the checkpoint taken before the last prompt: restores the
	 * conversation AND the files that turn wrote via the write tools. Returns
	 * what was restored, or null when there is nothing to rewind / a turn is
	 * running.
	 */
	rewind(): { messages: number; filesRestored: number } | null {
		return this.sessions.rewind();
	}

	/** Discard the active branch without merging. Returns true if one was discarded. */
	discardBranch(): boolean {
		return this.sessions.discardBranch();
	}

	// ── State management ─────────────────────────────────────────────────

	async init(): Promise<Record<string, unknown>> {
		await this.lifecycle.initialize();
		const toolNames = runtimeToolNames(this.session?.tools, this._defaultTools);
		const status = this.toolRouter.getStatus();
		const pluginStatus = this.plugins.status();
		const context = this.activity.context();
		const info = projectInitializationStatus({
			config: this.config,
			toolNames,
			mcpServerCount: status.mcpServerCount,
			mcpToolCount: status.mcpToolCount,
			mcpErrors: status.mcpErrors,
			contextTokens: context.tokens,
			contextMaxTokens: context.maxTokens || this.config.contextWindowTokens,
			runtimeState: this.session?.runtimeState,
			configPath: this.configPath,
			reasoner: this.getReasonerStatus(),
			mcpLoaded: status.mcpLoaded,
			mcpLoading: status.mcpLoading,
			enabledPluginRoots: status.enabledPluginRoots,
			loadedSkills: status.loadedSkills,
			skillsInjected: status.skillsInjected,
			skillsVisible: status.skillsVisible,
			pluginCount: pluginStatus.pluginCount,
			hookResult: pluginStatus.hookResult,
		});
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
		return this.extensions.getCommands();
	}

	invokeExtensionCommand(
		name: string,
		args: string,
	): Promise<string | undefined> {
		return this.extensions.executeCommand(name, args);
	}

	async stop(): Promise<void> {
		try {
			await this.lifecycle.stop();
		} finally {
			this.unsubscribeTodos();
		}
	}

	isActive(): boolean {
		return this.turns.isActive();
	}

	getMessages(): Message[] {
		return this.session?.messages || [];
	}

	/** Return full context as formatted text for /context command.
	 *
	 * Memoriam memory context is injected into the provider payload by the
	 * beforeProviderPayload hook rather than composed here, so it does not
	 * appear in this synchronous inspection view.
	 */
	getContext(): string {
		const msgs = this.getMessages();
		const inspection = inspectContext({
			messages: msgs,
			systemPrompt: this.config.systemPrompt || "",
			memoryContext: "",
			toolDefinitions: this.getTools().toToolDefinitions(),
		});
		this.activity.setContext(inspection.tokens);
		return inspection.text;
	}

	getContextSourceMap(memoryContext: string = ""): Array<{
		name: string;
		tokens: number;
		detail: string;
	}> {
		return contextSources({
			messages: this.getMessages(),
			systemPrompt: this.config.systemPrompt || "",
			memoryContext,
			toolDefinitions: this.getTools().toToolDefinitions(),
		});
	}

	/** Canonical size used by /context, /status, and the status bar. */
	private measureContextTokens(): number {
		const messages = this.getMessages();
		const toolDefinitions = this.getTools().toToolDefinitions();
		return estimateChatPayloadTokens(messages, toolDefinitions);
	}

	private publishContextUsage(): void {
		const current = this.activity.context();
		const context = this.activity.setContext(
			this.measureContextTokens(),
			current.maxTokens || this.config.contextWindowTokens,
		);
		this.emit({
			type: "context_update",
			tokens: context.tokens,
			maxTokens: context.maxTokens,
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
	 * Execute a bash command directly (for user_bash / !command in the input bar).
	 * Returns the command output and exit code.
	 */
	async executeBashCommand(command: string): Promise<{
		output: string;
		exitCode: number;
	}> {
		return executeProcessCommand(this.cwd, command);
	}
}

export { getSkillsDirs } from "./support/resource-directories.ts";
