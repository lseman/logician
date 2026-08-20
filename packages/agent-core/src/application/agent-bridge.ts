// ── AgentCoreBridge ──────────────────────────────────────────────────────────────
import { envNumber } from "../tui-utils.ts";
// Replaces the Python bridge with direct TypeScript agent-core integration.
// Translates agent-core events to the same shapes the transcript expects.

import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { type ReasonerConfig } from "@logician/agent-blocks/reasoning";
import type {
	AgentConfig,
	AgentModelConfig,
	QueueMode,
	TruncationConfig,
} from "../core/types/index.ts";
import type {
	AgentEvent,
	Message,
	Tool,
	StopReason,
} from "../core/types/types-messages.ts";
import type { WebSearchConfig } from "../core/types/types-config.ts";
import {
	AgentHarness,
	type HarnessPhase,
} from "../core/harness/agent-harness.ts";
import type { AbortResult } from "../core/harness/types.ts";
import type { AskUserContext } from "../core/types/types-messages.ts";
import {
	estimateChatPayloadTokens,
	estimateTokens,
} from "../core/provider/messages.ts";
import { OpenAIBackend, type LLMBackend } from "../core/provider/backend.ts";
import type { PluginCommandResult } from "../infrastructure/tools/utils/plugins.ts";
import type { Session } from "../core/session/session.ts";
import { resolveAgentSettings } from "../core/configuration/agent-settings.ts";
import { onTodosChanged } from "@logician/agent-blocks/tasks/todo.ts";
import {
	configurePluginRuntimeEnv,
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
	ToolRegistry,
} from "../infrastructure/tools/index.ts";
import {
	ExtensionManager,
	loadPluginCommands,
} from "./manager/extension-manager.ts";
import { BridgeSessionManager } from "./manager/bridge-session-manager.ts";
import { AgentCoordinator } from "./manager/agent-coordinator.ts";
import {
	MemoryManager,
	type MemoryManagerOptions,
	type MemoryManagerRuntime,
} from "./manager/memory-manager.ts";
import { buildDefaultSystemPrompt } from "../infrastructure/context/system-prompt.ts";
import { LspManager } from "../infrastructure/developer-tools/lsp-manager.ts";
import { createPostEditDiagnosticHooks } from "../infrastructure/developer-tools/post-edit-diagnostics.ts";
import { McpManager } from "../features/mcp/manager.ts";
import type {
	McpSnapshotResult,
	McpToggleResult,
} from "../features/mcp/manager.ts";
import {
	findPromptByName,
	loadPrompts,
	type Prompt,
} from "../features/prompts/loader.ts";
import { mapAgentEvent } from "../runtime/event-mapping.ts";
import type { RuntimeEvent } from "../runtime/events.ts";
import { formatPluginResult } from "../runtime/plugin-result-formatter.ts";
import {
	findSkillByName,
	formatActivatedSkills,
	formatSkillActivationNotice,
	formatSkillCatalog,
	formatSkillInvocation,
	loadSkills,
	type selectSkillsForPrompt,
	type Skill,
} from "../features/skills/index.ts";
import {
	createMemoryGetTool,
	createMemorySearchTool,
} from "../infrastructure/tools/memory-tools.ts";
import { createDefaultTools } from "../infrastructure/tools/default-tools.ts";
import { createReadSkillTool } from "../infrastructure/tools/read-skill.ts";
import {
	getDefaultSandboxProfile,
	type SandboxProfile,
	setDefaultSandboxProfile,
} from "../infrastructure/tools/sandbox.ts";
import { killAllTrackedChildren } from "../infrastructure/tools/utils/shell.ts";
import {
	buildPluginRuntimeEnv,
	createHookTranscriptPath,
	eventLogPathFor,
	resolveWebSearchConfig,
} from "./bridge-environment.ts";
import { RepositoryMap } from "./repository-map.ts";
import {
	getProjectSkillDirs,
	getProjectPromptDirs,
} from "./resource-directories.ts";
export type EventCallback = (event: RuntimeEvent) => void;
export type ErrorCallback = (err: Error) => void;

export type RuntimeSettingsPatch = Partial<
	Pick<
		AgentConfig,
		| "thinkingLevel"
		| "temperature"
		| "inferenceMode"
		| "maxTokens"
		| "maxIterations"
		| "executionProfile"
		| "guardsEnabled"
		| "duplicateGuardEnabled"
		| "failureGuardEnabled"
		| "budgetStopEnabled"
		| "continuationEnabled"
		| "autoRetryEnabled"
		| "proactiveCompactionEnabled"
		| "rtkProxyEnabled"
		| "ariadneEnabled"
		| "fffgrepEnabled"
	>
> & {
	reasonerId?: string;
	steeringInterrupt?: boolean;
	postEditDiagnostics?: boolean;
	memoryEnabled?: boolean;
	guardMode?: "auto" | "on" | "off";
};

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
	/** Authoritative resolved config source; never recomputed inside the bridge. */
	configPath?: string;
	baseUrl: string;
	model: string;
	models?: AgentModelConfig[];
	chatTemplate?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	thinkingLevel?: AgentConfig["thinkingLevel"];
	inferenceMode?: AgentConfig["inferenceMode"];
	executionProfile?: AgentConfig["executionProfile"];
	contextWindowTokens?: number;
	toolExecution?: AgentConfig["toolExecution"];
	runtimeHooksEnabled?: boolean;
	permissionMode?: PermissionMode;
	permissionRules?: PermissionRules;
	steeringInterrupt?: boolean;
	maxTotalTokens?: number;
	/** Test-only: suppress the construction-time MCP auto-start so unit tests
	 * can stub McpManager/loadMcpToolsOnce before anything real fires.
	 * Real app startup never sets this — MCP always auto-starts on open. */
	autoStartMcp?: boolean;
	tools?: Tool[];
	/** Additional tools merged in alongside the default set and any
	 * memory tools, deduped by name (see ToolRouterDeps.extraTools). */
	extraTools?: Tool[];
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
	proactiveCompactionEnabled?: boolean;
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	maxParallelAgents?: number;
	lsp?: {
		enabled?: boolean;
		timeoutMs?: number;
		serverOverrides?: Record<
			string,
			{ command: string; args?: string[]; languageId: string }
		>;
	};
	continuationEnabled?: boolean;
	postEditDiagnostics?: boolean;
	rtkProxyEnabled?: boolean;
	ariadneEnabled?: boolean;
	fffgrepEnabled?: boolean;
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
	// ── Extensions ──────────────────────────────────────────────────────────
	extensionDirs?: { user?: string; paths?: string[] };
	// ── Memory ────────────────────────────────────────────────────────────
	/** Whether to enable memory hooks. Default: false (opt-in). */
	memoryEnabled?: boolean;
	/** Path to the memory SQLite database. Default: <cwd>/.logician/memory.db. */
	memoryDbPath?: string;
	/** Smaller model for semantic memory extraction; defaults to the active model. */
	memoryExtractorModel?: string;
	/** Dedicated OpenAI-compatible endpoint for semantic memory extraction. */
	memoryExtractorBaseUrl?: string;
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
	/** Host for the memory viewer dashboard. Default: "0.0.0.0". */
	memoryViewerHost?: string;
	/** Enable optional local MiniLM semantic retrieval. Default: false. */
	memoryEmbeddingsEnabled?: boolean;
	/** Hugging Face model ID used for local embeddings. */
	memoryEmbeddingModel?: string;
	/** Structured pre-reasoning mode. Default: "none" (disabled). */
	reasoner?: string;
	/** Overrides merged over the selected reasoner's defaults. */
	reasonerConfig?: ReasonerConfig;
	/** Inject a change-refreshed symbol/import map into each user turn. Default: true. */
	repositoryMapEnabled?: boolean;
	/** Maximum approximate tokens for repository-map context. Default: 2000. */
	repositoryMapMaxTokens?: number;
}

// ── AgentCoreBridge ─────────────────────────────────────────────────────────────

export class AgentCoreBridge {
	private config: AgentConfig;
	private backend: OpenAIBackend;
	private harness: AgentHarness | null = null;
	private durableSession: Session | undefined;
	private callbacks: EventCallback[] = [];
	private errorCb: ErrorCallback | null = null;
	private running = false;
	private sendTail: Promise<void> = Promise.resolve();
	private sessionManager: BridgeSessionManager | null = null;
	private cwd: string;
	private _defaultTools: Tool[];
	private _mcpLoaded = false;
	private _mcpLoadPromise: Promise<void> | null = null;
	private _mcpServerCount = 0;
	private _mcpErrors: string[] = [];
	private _mcpToolNames = new Set<string>();
	private _mcpSystemContext = "";
	private _skillsInjected = false;
	private _skillsContext: string | null = null;
	private _loadedSkills: Skill[] = [];
	private _enabledPluginRoots: Array<{ name: string; installPath: string }> =
		[];
	private _promptsInjected = false;
	private _loadedPrompts: Prompt[] = [];
	private baseSystemPrompt: string;
	private static readonly SANDBOX_CYCLE: SandboxProfile[] = [
		"none",
		"code",
		"full",
	];
	private readonly mcpManager = new McpManager();
	// toolRouter facade for backward compatibility with tests
	private readonly toolRouter = {
		getDefaultTools: () => this._defaultTools,
		getLoadedSkills: () => this._loadedSkills,
		getLoadedPrompts: () => this._loadedPrompts,
		getEnabledPluginRoots: () => this._enabledPluginRoots,
		getMcpSnapshot: () => this.mcpManager.getSnapshot(this.cwd),
		setMcpServerEnabled: (name: string, enabled: boolean) =>
			this.mcpManager.setServerEnabled(name, enabled, this.cwd),
		closeMcp: () => this.mcpManager.close(),
		getMcpServerCount: () => this._mcpServerCount,
		getMcpToolCount: () => this._mcpToolNames.size,
		getMcpErrors: () => this._mcpErrors,
		isMcpLoaded: () => this._mcpLoaded,
		isMcpLoading: () => this._mcpLoadPromise !== null && !this._mcpLoaded,
		getMcpSystemContext: () => this._mcpSystemContext,
		getSkillsContext: () => this._skillsContext,
		getSandboxMode: () => getDefaultSandboxProfile(),
		setSandboxMode: (mode: SandboxProfile) => setDefaultSandboxProfile(mode),
		cycleSandboxMode: () => this.cycleSandboxMode(),
		injectSkillsFromPlugins: () => this.injectSkillsFromPlugins(),
		injectPrompts: () => this.injectPrompts(),
		loadMcpToolsOnce: () => this.loadMcpToolsOnce(),
		getStatus: () => ({
			mcpServerCount: this._mcpServerCount,
			mcpToolCount: this._mcpToolNames.size,
			mcpErrors: this._mcpErrors,
			mcpLoaded: this._mcpLoaded,
			mcpLoading: this._mcpLoadPromise !== null && !this._mcpLoaded,
			skillsInjected: this._skillsInjected,
			skillsVisible: !!this._skillsContext,
			loadedSkills: this._loadedSkills,
			enabledPluginRoots: this._enabledPluginRoots,
		}),
		resetInjectedContext: () => {
			this._skillsContext = null;
			this._skillsInjected = false;
			this._promptsInjected = false;
		},
		resetSkillsAndPrompts: () => {
			this._loadedSkills = [];
			this._skillsContext = null;
			this._skillsInjected = false;
			this._loadedPrompts = [];
			this._promptsInjected = false;
		},
		setAriadneEnabled: (enabled: boolean) => {
			const hasAriadne = this._defaultTools.some(
				tool => tool.name === "ariadne",
			);
			if (enabled === hasAriadne) return;
			this._defaultTools = enabled
				? [
						{
							name: "ariadne",
							description: "Ariadne code graph tool",
							execute: async () => "",
							parameters: {},
						},
						...this._defaultTools,
					]
				: this._defaultTools.filter(tool => tool.name !== "ariadne");
		},
		setFffgrepEnabled: (enabled: boolean) => {
			// fffgrep tools are handled by the tool registry
		},
		buildRegistry: (config: any) => {
			const registry = new ToolRegistry(config);
			registry.registerMany(this._defaultTools);
			return registry;
		},
	} as const;

	// ── Inline MCP loading ───────────────────────────────────────────────

	async loadMcpToolsOnce(): Promise<void> {
		if (this._mcpLoaded || process.env.LOGICIAN_MCP === "0") return;
		if (!this._mcpLoadPromise) {
			this._mcpLoadPromise = (async () => {
				const result = await this.mcpManager.load(
					this.cwd,
					this._defaultTools.map(tool => tool.name),
				);
				this._mcpServerCount = result.servers;
				this._mcpErrors = result.errors;
				this._mcpToolNames = new Set(result.tools.map(tool => tool.name));
				this._mcpSystemContext = result.errors.length
					? `<mcp-status>\n${result.errors.length} MCP server(s) failed to load:\n${result.errors.map(e => `- ${e}`).join("\n")}\n` +
						"Tools from these servers are unavailable this session.\n</mcp-status>"
					: "";
				if (result.tools.length || this._mcpSystemContext) {
					const existing = new Set(this._defaultTools.map(tool => tool.name));
					const newTools = result.tools.filter(
						tool => !existing.has(tool.name),
					);
					for (const tool of newTools) {
						if (!this._defaultTools.some(t => t.name === tool.name)) {
							this._defaultTools = [...this._defaultTools, tool];
						}
					}
				}
				this._mcpLoaded = true;
				this.applyPluginHookContext(
					this.startupHookResult || {
						additional_contexts: [],
						context_messages: [],
						initial_user_message: "",
					},
				);
				this.emit({
					type: "notice",
					level: result.errors.length ? "warn" : "info",
					label: "MCP",
					text: `Loaded ${result.servers} server(s).`,
				});
			})();
		}
		await this._mcpLoadPromise;
	}

	// ── Inline skills loading ────────────────────────────────────────────

	async injectSkillsFromPlugins(): Promise<void> {
		if (this._skillsInjected) return;
		this._skillsInjected = true;

		const registry = await runPluginBackend("list", []);
		const plugins = registry.plugins || [];

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
		this._enabledPluginRoots = enabledPlugins;

		skillsDirs.push(path.join(os.homedir(), ".agents", "skills"));

		if (this.projectTrusted) {
			skillsDirs.push(...getProjectSkillDirs(this.cwd));
		}

		if (!skillsDirs.length) return;

		const { skills: rawSkills, diagnostics } = await loadSkills(skillsDirs);

		const skills = rawSkills.map(skill => {
			const owner = enabledPlugins.find(p =>
				skill.filePath.startsWith(p.installPath + path.sep),
			);
			if (!owner?.name || skill.name.startsWith(`${owner.name}:`)) {
				return skill;
			}
			return {
				...skill,
				name: `${owner.name}:${skill.name}`,
				slashName: `${owner.name}:${skill.slashName}`,
				aliases: [...(skill.aliases ?? []), skill.name],
			};
		});

		skills.push(...(await loadPluginCommands(enabledPlugins)));

		for (const diag of diagnostics) {
			this.emit({
				type: "token",
				token: `[Skill warning] ${diag.code}: ${diag.message}`,
			});
		}

		this._loadedSkills = skills;
		const visible = skills.filter(s => !s.disableModelInvocation);
		this._skillsContext = formatSkillCatalog(visible);

		const readSkill = createReadSkillTool(skills);
		if (readSkill) {
			if (!this._defaultTools.some(t => t.name === readSkill.name)) {
				this._defaultTools = [...this._defaultTools, readSkill];
			}
		}
		this.applyPluginHookContext(
			this.startupHookResult || {
				additional_contexts: [],
				context_messages: [],
				initial_user_message: "",
			},
		);
	}

	// ── Inline prompts loading ───────────────────────────────────────────

	async injectPrompts(): Promise<void> {
		if (this._promptsInjected) return;
		this._promptsInjected = true;

		if (!this.projectTrusted) return;
		const promptDirs = getProjectPromptDirs(this.cwd);
		if (!promptDirs.length) return;
		this._loadedPrompts = await loadPrompts(promptDirs);
	}

	// ── Apply plugin hook context ────────────────────────────────────────

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
		if (this._mcpSystemContext) allContexts.push(this._mcpSystemContext);
		if (this._skillsContext) allContexts.push(this._skillsContext);

		this.config.systemPrompt = allContexts.length
			? `${this.baseSystemPrompt}\n\n${allContexts.join("\n\n")}`
			: this.baseSystemPrompt;
	}
	private extensionManager: ExtensionManager | null = null;

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
	private lspManager: LspManager;
	private readonly projectTrusted: boolean;
	// Permission/question resolvers (inlined from InteractionCoordinator)
	private readonly permissionResolvers = new Map<
		string,
		(d: "allow" | "deny" | "always") => void
	>();
	private readonly questionResolvers = new Map<
		string,
		{ allow: (a: string) => void; deny: () => void }
	>();
	private permissionManager!: PermissionManager;
	private memoryManager: MemoryManager | null = null;
	private agentCoordinator: AgentCoordinator | null = null;

	private repositoryMap?: RepositoryMap;
	private activeRepositoryQuery?: string;
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
		if (opts.repositoryMapEnabled !== false) {
			this.repositoryMap = new RepositoryMap(this.cwd, {
				maxTokens: opts.repositoryMapMaxTokens,
			});
		}

		this.projectTrusted = opts.projectTrusted === true;
		this.configPath = opts.configPath ?? null;
		configurePluginRuntimeEnv(buildPluginRuntimeEnv(opts));
		this.postEditDiagnosticsEnabled =
			process.env.LOGICIAN_POST_EDIT_DIAGNOSTICS === "0"
				? false
				: opts.postEditDiagnostics !== false;
		const lspTimeoutMs = opts.lsp?.timeoutMs ?? 2_000;
		const serverOverrides = opts.lsp?.serverOverrides;
		this.lspManager = new LspManager(this.cwd, {
			timeoutMs: lspTimeoutMs,
			servers:
				serverOverrides && Object.keys(serverOverrides).length > 0
					? serverOverrides
					: undefined,
		});
		this.transcriptPath = createHookTranscriptPath(this.cwd, this.sessionId);
		const defaultWebSearch = resolveWebSearchConfig();
		const webSearch = {
			baseUrl: opts.webSearch?.baseUrl || defaultWebSearch.baseUrl,
			maxResults: opts.webSearch?.maxResults ?? defaultWebSearch.maxResults,
		};
		this._defaultTools = opts.tools?.length
			? opts.tools
			: createDefaultTools({ webSearch, ariadneEnabled: opts.ariadneEnabled });
		if (opts.extraTools?.length) {
			this._defaultTools = [
				...this._defaultTools,
				...(opts.memoryEnabled !== false
					? [
							createMemorySearchTool(
								() => this.memoryManager?.getStore() ?? null,
							),
							createMemoryGetTool(() => this.memoryManager?.getStore() ?? null),
						]
					: []),
				...(opts.extraTools ?? []),
			].filter((t, i, arr) => arr.findIndex(x => x.name === t.name) === i);
		}
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

		// Create extension manager and load extensions
		this.extensionManager = new ExtensionManager({
			sessionId: this.sessionId,
			cwd: this.cwd,
			extensionDirs: opts.extensionDirs,
			projectTrusted: this.projectTrusted,
			piRuntime: {
				isIdle: () => !this.running,
				hasPendingMessages: () => {
					const queues = this.harness?.getQueues();
					return (
						!!queues &&
						queues.steering.length +
							queues.followUp.length +
							queues.nextTurn.length >
							0
					);
				},
				abort: () => void this.cancel(),
				shutdown: () => void this.stop(),
				compact: () => void this.compact(),
				getSystemPrompt: () =>
					this.config.systemPrompt ?? this.baseSystemPrompt,
				sendUserMessage: content => void this.sendMessage(content),
				getActiveTools: () =>
					this.harness?.tools?.list().map((tool: Tool) => tool.name) ??
					this._defaultTools.map(tool => tool.name),
				getAllTools: () =>
					(this.harness?.tools?.list() ?? this._defaultTools).map(tool => ({
						name: tool.name,
						description: tool.description,
					})),
				setModel: async model => {
					const id =
						typeof model === "string"
							? model
							: typeof model === "object" && model !== null && "id" in model
								? String((model as { id: unknown }).id)
								: "";
					if (!id) return false;
					this.setModel(id);
					return true;
				},
				getThinkingLevel: () => this.config.thinkingLevel,
				setThinkingLevel: level => this.setThinkingLevel(String(level)),
			},
		});
		void this.extensionManager.initialize();

		this.permissionManager = new PermissionManager({
			mode: opts.permissionMode ?? "acceptEdits",
			rules: opts.permissionRules,
		});

		// Initialize memory manager
		this.memoryManager = new MemoryManager(this.cwd, this.sessionId, {
			memoryEnabled: opts.memoryEnabled,
			memoryDbPath: opts.memoryDbPath,
			memoryExtractorModel: opts.memoryExtractorModel,
			memoryExtractorBaseUrl: opts.memoryExtractorBaseUrl,
			memoryCaptureTools: opts.memoryCaptureTools,
			memoryInjectContext: opts.memoryInjectContext,
			memoryContextBudget: opts.memoryContextBudget,
			memoryViewerEnabled: opts.memoryViewerEnabled,
			memoryViewerPort: opts.memoryViewerPort,
			memoryEmbeddingsEnabled: opts.memoryEmbeddingsEnabled,
			memoryEmbeddingModel: opts.memoryEmbeddingModel,
			model: opts.model,
		});

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
			permissions: this.permissionManager,
			guardsEnabled: opts.guardsEnabled,
			duplicateGuardEnabled: opts.duplicateGuardEnabled,
			failureGuardEnabled: opts.failureGuardEnabled,
			duplicateToolThreshold: opts.duplicateToolThreshold,
			toolFailureLoopThreshold: opts.toolFailureLoopThreshold,
			budgetStopEnabled: opts.budgetStopEnabled,
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
			// Permission & question callbacks (inlined from InteractionCoordinator)
			onPermissionRequest: ctx =>
				new Promise(resolve => {
					this.permissionResolvers.set(ctx.toolCallId, resolve);
					this.emit({
						type: "permission_request",
						toolName: ctx.toolName,
						toolCallId: ctx.toolCallId,
						args: ctx.args,
					});
				}),
			onQuestionRequest: (ctx: AskUserContext) =>
				new Promise<string>(resolve => {
					const qid = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
					this.questionResolvers.set(qid, {
						allow: resolve,
						deny: () => resolve("__dismissed__"),
					});
					this.emit({
						type: "question_request",
						questionId: qid,
						questions: ctx.questions,
					});
				}),
			hooks: this.buildMemoryHooks(
				createPostEditDiagnosticHooks(
					this.cwd,
					() => this.postEditDiagnosticsEnabled,
					opts.lsp?.enabled === false ? undefined : this.lspManager,
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

		onTodosChanged(todos => {
			this.emit({ type: "todos", todos });
		});

		// Create agent coordinator for reasoner, EoH, and subagents
		this.agentCoordinator = new AgentCoordinator(
			{
				emit: event => this.emit(event),
				getBackend: () => this.backend,
				getBaseUrl: () => this.config.baseUrl,
				getCurrentModel: () => this.getCurrentModel(),
				harness: null, // set below via ensureHarness
				cwd: this.cwd,
				projectTrusted: this.projectTrusted,
				maxParallelAgents: opts.maxParallelAgents,
				getEnabledPluginRoots: () => this._enabledPluginRoots,
				getDefaultTools: () => this._defaultTools,
				ensureHarness: () => this.ensureHarness(),
				reportError: error => this.reportError(error),
			},
			opts.reasoner,
			opts.reasonerConfig,
		);

		// Create session manager for queue/session/continuation management
		this.sessionManager = new BridgeSessionManager({
			harness: null, // set below via setHarness
			emit: event => this.emit(event),
			getSystemPrompt: () => this.config.systemPrompt ?? this.baseSystemPrompt,
			setConfigSteeringMode: mode => {
				this.config.steeringQueueMode = mode;
			},
			setConfigSteeringInterrupt: enabled => {
				this.config.steeringInterrupt = enabled;
			},
			setConfigFollowUpMode: mode => {
				this.config.followUpQueueMode = mode;
			},
			getSteeringInterrupt: () => this.config.steeringInterrupt === true,
		});
	}

	/**
	 * Build memory hooks by delegating to the MemoryManager.
	 */
	private buildMemoryHooks(
		existingHooks: AgentConfig["hooks"],
	): AgentConfig["hooks"] {
		if (!this.memoryManager) return existingHooks;
		return this.memoryManager.createHooks(existingHooks, {
			isRunning: () => this.running,
			getBackend: () => this.backend,
			emit: event => this.emit(event),
		});
	}

	/** Add a tool to the default set and propagate it into live config/harness/system prompt. */
	/** Propagate a tool the router just registered into live config/harness/system prompt. */
	// ── Event registration ─────────────────────────────────────────────────

	on(callback: EventCallback): () => void {
		this.callbacks.push(callback);
		return () => {
			this.callbacks = this.callbacks.filter(cb => cb !== callback);
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

	private emit(event: RuntimeEvent): void {
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
		await this.extensionManager?.getLoadPromise();
		// Emit Pi input event for Pi extension interception.
		// If a Pi extension handles the input (returns 'handled'), skip processing.
		// If it transforms, use the transformed text.
		const inputResult = await this.extensionManager?.emitInputEvent(
			message,
			[],
			"interactive",
		);
		if (inputResult) {
			if (inputResult.action === "handled") return;
			if (
				inputResult.action === "transform" &&
				inputResult.text !== undefined
			) {
				message = inputResult.text;
			}
		}

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
		// Local extractor models can saturate CPU/GPU and make both the UI and
		// primary provider sluggish. Prefer the interactive turn; extraction's
		// deterministic fallback still records the completed prior turn.
		this.memoryManager?.abortExtractors();
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
			if (!this._mcpLoaded) {
				void this.toolRouter
					.loadMcpToolsOnce()
					.catch(error => this.reportError(error));
			}
			// Reuse one harness across messages so conversation history (and thus
			// "continue" / "go on" follow-ups) persists. Created lazily once.
			const harness = this.ensureHarness();
			this.activeRepositoryQuery = undefined;
			const repositoryContext = this.repositoryMap?.render(message);
			if (repositoryContext) {
				persistentSystemPrompt = this.config.systemPrompt;
				this.activeRepositoryQuery = message;
				turnSystemPrompt = `${persistentSystemPrompt}\n\n${repositoryContext}`;
				harness.updateConfig({ systemPrompt: turnSystemPrompt });
			}
			if (this.agentCoordinator) {
				try {
					const advisory = await this.agentCoordinator.runReasoner(
						message,
						this.backend,
					);
					if (advisory) {
						persistentSystemPrompt ??= this.config.systemPrompt;
						turnSystemPrompt = `${turnSystemPrompt}\n\nA structured reasoner produced the following advisory analysis for this turn. Verify it, use tools as needed, and do not mention this internal advisory unless useful:\n\n${advisory}`;
						harness.updateConfig({ systemPrompt: turnSystemPrompt });
					}
				} catch (error) {
					// Reasoner errors are already emitted by the coordinator
					throw error;
				}
			}
			const activations: ReturnType<typeof selectSkillsForPrompt> = [];
			// Simplified: no scoring, just use all visible skills
			// The model will read skills as needed via read_skill
			turnActivations = activations;
			if (activations.length) {
				persistentSystemPrompt ??= this.config.systemPrompt;
				harness.updateConfig({
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
			await harness.prompt(message);
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
			this.errorCb?.(error);
			throw error;
		} finally {
			if (persistentSystemPrompt !== undefined) {
				this.harness?.updateConfig({ systemPrompt: persistentSystemPrompt });
			}
			this.running = false;
			this.publishContextUsage();
			this.emit({ type: "turn_end", turnId });
			// Keep the harness alive to retain history across turns.
			this.emit({ type: "phase", state: "ready" });
			if (turnSucceeded)
				this.sessionManager?.checkPendingContinuation(turnActivations);
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
				extensionRunner: this.extensionManager?.runner ?? undefined,
			});
			this.harness.setSessionId(this.sessionId);
			if (this.durableSession) this.harness.attachSession(this.durableSession);
			if (this.compactionSettings)
				this.harness.setAutoCompactionSettings(this.compactionSettings);
			// Wire session manager to harness
			this.sessionManager?.setHarness(this.harness);
			// Harness owns the queue state; mirror every change to the UI.
			this.harness.setOnQueueChange(() =>
				this.sessionManager?.emitQueueUpdate(),
			);
			// Surface harness phase transitions the loop can't see.
			this.harness.setOnPhaseChange(phase =>
				this.sessionManager?.emitHarnessPhase(phase),
			);
			// Autonomous continuation: when the harness settles with pending
			// nextTurn messages, auto-trigger the next prompt.
			this.harness.setOnSettled(nextTurnCount => {
				if (nextTurnCount === 0) return;
				this.sessionManager?.setPendingContinuation(true);
				this.emit({
					type: "notice",
					level: "info",
					label: "Continuation",
					text: `${nextTurnCount} next-turn message(s) queued; continuation will start after settlement.`,
				});
			});
		}
		return this.harness;
	}

	private emitRuntimeStatus(): void {
		if (!this.harness) return;
		this.emit({
			type: "runtime_status",
			runPhase: this.harness.phase,
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
			this.ensureHarness().setHistory(messages);
			this.publishContextUsage();
			return true;
		} catch (_e: unknown) {
			return false;
		}
	}

	// ── Queue operations (delegated to SessionManager) ─────────────────

	steer(message: string): void {
		this.sessionManager?.steer(message);
	}

	followUp(message: string): void {
		this.sessionManager?.followUp(message);
	}

	nextTurn(message: string): void {
		this.sessionManager?.nextTurn(message);
	}

	setSteeringMode(mode: QueueMode): void {
		this.sessionManager?.setSteeringMode(mode);
	}

	private setSteeringInterrupt(enabled: boolean): void {
		this.sessionManager?.setSteeringInterrupt(enabled);
	}

	getSteeringInterrupt(): boolean {
		return this.sessionManager?.getSteeringInterrupt() ?? false;
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
		this.sessionManager?.setFollowUpMode(mode);
	}

	getSteeringMessages(): string[] {
		return this.sessionManager?.getSteeringMessages() ?? [];
	}

	flushSteeringNow(): number {
		return this.sessionManager?.flushSteeringNow() ?? 0;
	}

	getFollowUpMessages(): string[] {
		return this.sessionManager?.getFollowUpMessages() ?? [];
	}

	getNextTurnMessages(): string[] {
		return this.sessionManager?.getNextTurnMessages() ?? [];
	}

	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return (
			this.sessionManager?.clearQueue() ?? {
				steering: [],
				followUp: [],
				nextTurn: [],
			}
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.sessionManager?.dropQueuedMessage(displayIndex);
	}

	/** Abort: clear steering/follow-up queues (preserves nextTurn). */
	async abort(): Promise<AbortResult | null> {
		// harness.abort() clears steering/follow-up and emits onQueueChange.
		return (await this.harness?.abort()) ?? null;
	}

	/** Execute a slash command (sends as chat message to the agent). */
	sendSlash(raw: string): void {
		const trimmed = raw.trim();
		if (this.sessionManager?.handleSlashCommand(trimmed)) return;
		// /reload — reload settings, skills, extensions, and MCP config
		if (trimmed === "/reload") {
			this.reload().catch(err => this.errorCb?.(err));
			return;
		}
		this.sendMessage(raw).catch(err => this.errorCb?.(err));
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
		// Sync memory session
		this.memoryManager?.resetSession(this.sessionId);

		// Reset state that is per-session
		this._loadedSkills = [];
		this._skillsContext = null;
		this._skillsInjected = false;
		this._loadedPrompts = [];
		this._promptsInjected = false;
		this.startupHooksRan = false;
		this.startupHooksPromise = null;
		this.startupHookResult = null;

		// ── Re-discover skills and prompts ────────────────────────────────
		await this.injectSkillsFromPlugins();
		await this.injectPrompts();

		// ── Re-load extensions ────────────────────────────────────────────
		await this.extensionManager
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
		this.sendMessage(message).catch(err => this.errorCb?.(err));
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
		this.sendMessage(message).catch(err => this.errorCb?.(err));
		return true;
	}

	// ── Permissions & interactive questions (inlined from InteractionCoordinator) ─

	/** Answer a pending permission_request. Returns false for unknown ids. */
	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean {
		const r = this.permissionResolvers.get(toolCallId);
		if (!r) return false;
		this.permissionResolvers.delete(toolCallId);
		r(decision);
		return true;
	}

	/**
	 * Answer a pending question by id. The answer is forwarded to the agent's
	 * resolver. Returns false if the question id is unknown.
	 */
	respondToQuestion(questionId: string, answer: string): boolean {
		const r = this.questionResolvers.get(questionId);
		if (!r) return false;
		this.questionResolvers.delete(questionId);
		r.allow(answer);
		return true;
	}

	/** Deny every pending permission request (abort / shutdown). */
	private denyPendingPermissions(): void {
		const ids = [...this.permissionResolvers.keys()];
		for (const id of ids) {
			const r = this.permissionResolvers.get(id);
			if (r) {
				r("deny");
				this.permissionResolvers.delete(id);
			}
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

	// ── Sandbox mode (see tool-router.ts) ───────────────────────────────

	getSandboxMode(): SandboxProfile {
		return getDefaultSandboxProfile();
	}

	setSandboxMode(mode: SandboxProfile): void {
		setDefaultSandboxProfile(mode);
	}

	cycleSandboxMode(): SandboxProfile {
		const cycle = AgentCoreBridge.SANDBOX_CYCLE;
		const currentIndex = cycle.indexOf(getDefaultSandboxProfile());
		const next = cycle[(currentIndex + 1) % cycle.length];
		setDefaultSandboxProfile(next);
		this.emit({
			type: "notice",
			level: "info",
			label: "Sandbox",
			text: `mode: ${next === "none" ? "off" : next}`,
		});
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

	/** Get all available models. */
	getModels(): string[] {
		return this.config.models?.length
			? this.config.models.map(model => model.model)
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
			candidate => candidate.key === key,
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
		if (
			!this._mcpLoaded &&
			!this._mcpLoadPromise !== null &&
			!this._mcpLoaded
		) {
			void this.toolRouter
				.loadMcpToolsOnce()
				.catch(error => this.reportError(error));
		}
		this.contextTokens = this.measureContextTokens();
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this._defaultTools.map(t => t.name);
		const state = {
			agent_name: "logician",
			model: this.config.model,
			base_url: this.config.baseUrl,
			web_search_url: this.config.webSearch?.baseUrl || "",
			web_search_enabled: toolNames.includes("web_search"),
			tools: toolNames,
			mcp_servers: this._mcpServerCount,
			mcp_tools: this._mcpToolNames.size,
			mcp_errors: this._mcpErrors,
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
			reasoner: this.agentCoordinator?.getReasonerStatus() ?? "none",
		};
		return state;
	}

	async getPluginSnapshot(): Promise<PluginCommandResult> {
		return runPluginBackend("list", []);
	}

	async getMcpSnapshot(): Promise<McpSnapshotResult> {
		return this.mcpManager.getSnapshot(this.cwd);
	}

	async setMcpServerEnabled(
		serverName: string,
		enabled: boolean,
	): Promise<McpToggleResult> {
		return this.mcpManager.setServerEnabled(serverName, enabled, this.cwd);
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
		this.harness?.setThinkingLevel(level);
		// Also update the backend's default so future turns pick it up.
		(this.backend as OpenAIBackend).setDefaultThinkingLevel(
			level as "off" | "minimal" | "low" | "medium" | "high" | "xhigh",
		);
	}

	private setTemperature(temperature: number): void {
		this.config.temperature = temperature;
		this.harness?.updateConfig({ temperature });
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
		return this._mcpLoadPromise !== null && !this._mcpLoaded;
	}

	private setInferenceMode(mode: string): void {
		this.config.inferenceMode = mode as typeof this.config.inferenceMode;
		this.harness?.updateConfig({
			inferenceMode: mode as AgentConfig["inferenceMode"],
		});
	}

	private setMaxTokens(maxTokens: number): void {
		this.config.maxTokens = maxTokens;
		this.harness?.updateConfig({ maxTokens });
	}

	private setMaxIterations(maxIterations: number): void {
		this.config.maxIterations = maxIterations;
		this.harness?.updateConfig({ maxIterations });
	}

	private setExecutionProfile(profile: "autonomous" | "minimal"): void {
		this.config.executionProfile = profile;
		this.harness?.updateConfig({ executionProfile: profile });
	}

	private setRuntimeToggle(
		key:
			| "guardsEnabled"
			| "duplicateGuardEnabled"
			| "failureGuardEnabled"
			| "budgetStopEnabled"
			| "continuationEnabled"
			| "autoRetryEnabled"
			| "proactiveCompactionEnabled"
			| "postEditDiagnostics"
			| "rtkProxyEnabled"
			| "ariadneEnabled"
			| "fffgrepEnabled"
			| "memoryEnabled",
		enabled: boolean,
	): void {
		if (key === "memoryEnabled") {
			this.memoryManager?.setEnabled(enabled, this.sessionId);
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
			this._defaultTools = enabled
				? [
						...this._defaultTools,
						{
							name: "ariadne",
							description: "Ariadne code graph tool",
							execute: async () => "",
							parameters: {},
						},
					]
				: this._defaultTools.filter(t => t.name !== "ariadne");
			// Tool added to _defaultTools directly;
			return;
		}
		if (key === "fffgrepEnabled") {
			this.config.fffgrepEnabled = enabled;
			// Tool added to _defaultTools directly;
			void this.setMcpServerEnabled("fff", enabled).catch(error =>
				this.reportError(error),
			);
			return;
		}
		this.config[key] = enabled;
		if (
			key === "guardsEnabled" ||
			key === "duplicateGuardEnabled" ||
			key === "failureGuardEnabled" ||
			key === "budgetStopEnabled" ||
			key === "continuationEnabled" ||
			key === "autoRetryEnabled"
		) {
			this.harness?.updateConfig({ [key]: enabled });
		}
		if (key === "proactiveCompactionEnabled") {
			this.harness?.enableAutoCompaction(enabled);
		}
	}

	private setGuardMode(mode: "auto" | "on" | "off"): void {
		this.config.guardsEnabled = mode === "auto" ? undefined : mode === "on";
		this.harness?.updateConfig({
			guardsEnabled: this.config.guardsEnabled,
		});
	}

	updateSettings(patch: RuntimeSettingsPatch): void {
		if ("thinkingLevel" in patch && patch.thinkingLevel !== undefined)
			this.setThinkingLevel(patch.thinkingLevel);
		if ("temperature" in patch && patch.temperature !== undefined)
			this.setTemperature(patch.temperature);
		if ("reasonerId" in patch && patch.reasonerId !== undefined)
			this.setReasonerId(patch.reasonerId);
		if ("inferenceMode" in patch && patch.inferenceMode !== undefined)
			this.setInferenceMode(patch.inferenceMode);
		if ("maxTokens" in patch && patch.maxTokens !== undefined)
			this.setMaxTokens(patch.maxTokens);
		if ("maxIterations" in patch && patch.maxIterations !== undefined)
			this.setMaxIterations(patch.maxIterations);
		if ("executionProfile" in patch && patch.executionProfile !== undefined)
			this.setExecutionProfile(patch.executionProfile);
		if ("steeringInterrupt" in patch && patch.steeringInterrupt !== undefined)
			this.setSteeringInterrupt(patch.steeringInterrupt);
		if ("guardMode" in patch && patch.guardMode !== undefined)
			this.setGuardMode(patch.guardMode);

		for (const key of [
			"guardsEnabled",
			"duplicateGuardEnabled",
			"failureGuardEnabled",
			"budgetStopEnabled",
			"continuationEnabled",
			"autoRetryEnabled",
			"proactiveCompactionEnabled",
			"postEditDiagnostics",
			"rtkProxyEnabled",
			"ariadneEnabled",
			"fffgrepEnabled",
			"memoryEnabled",
		] as const) {
			const enabled = patch[key];
			if (enabled !== undefined) this.setRuntimeToggle(key, enabled);
		}
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
		budgetStopEnabled: boolean;
		guardMode: "auto" | "on" | "off";
	} {
		const settings = resolveAgentSettings(this.config);
		return {
			model: this.config.model,
			temperature: this.config.temperature ?? 0.5,
			maxTokens: this.config.maxTokens ?? 4096,
			maxIterations: settings.maxIterations,
			thinkingLevel: settings.thinkingLevel,
			inferenceMode: settings.inferenceMode,
			permissionMode: this.getPermissionMode(),
			executionProfile: settings.executionProfile,
			guardsEnabled: this.config.guardsEnabled ?? false,
			proactiveCompactionEnabled:
				this.config.proactiveCompactionEnabled ?? true,
			postEditDiagnostics: this.postEditDiagnosticsEnabled,
			rtkProxyEnabled: this.config.rtkProxyEnabled ?? false,
			ariadneEnabled: this.config.ariadneEnabled ?? true,
			fffgrepEnabled: this.config.fffgrepEnabled ?? true,
			memoryEnabled: !!this.memoryManager?.getStore(),
			duplicateGuardEnabled: this.config.duplicateGuardEnabled ?? true,
			failureGuardEnabled: this.config.failureGuardEnabled ?? false,
			continuationEnabled: this.config.continuationEnabled ?? true,
			autoRetryEnabled: this.config.autoRetryEnabled ?? true,
			budgetStopEnabled: this.config.budgetStopEnabled ?? false,
			guardMode:
				this.config.guardsEnabled === undefined
					? "auto"
					: this.config.guardsEnabled
						? "on"
						: "off",
		};
	}

	getMemoryStore(): ReturnType<
		typeof import("@logician/memory").createMemoryStore
	> | null {
		return this.memoryManager?.getStore() ?? null;
	}

	getMemoryStats(): {
		memoryEnabled: boolean;
		memoryCount: number;
		sessionCount: number;
		observationCount: number;
		viewerPort?: number;
	} {
		if (!this.memoryManager) {
			return {
				memoryEnabled: false,
				memoryCount: 0,
				sessionCount: 0,
				observationCount: 0,
			};
		}
		return this.memoryManager.getStats(this.sessionId);
	}

	/** Use the user-facing conversation session as the hook and memory session. */
	useConversationSession(sessionId: string, durableSession?: Session): void {
		if (!sessionId.trim()) return;
		const provisionalSessionId = this.sessionId;
		this.sessionId = sessionId;
		this.durableSession = durableSession;
		this.harness?.setSessionId(sessionId);
		if (durableSession) this.harness?.attachSession(durableSession);
		this.transcriptPath = createHookTranscriptPath(this.cwd, sessionId);
		this.config.hookSessionId = sessionId;
		this.config.hookTranscriptPath = this.transcriptPath;
		this.config.eventLogPath = eventLogPathFor(this.transcriptPath);
		// Sync memory session
		this.memoryManager?.onSessionChanged(sessionId, provisionalSessionId);
	}

	renameConversationSession(sessionId: string, name: string): void {
		this.memoryManager?.renameSession(sessionId, name);
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
		this.memoryManager?.resetSession(this.sessionId);
		// Reset skill/prompt injection state
		this._skillsContext = null;
		this._skillsInjected = false;
		this._promptsInjected = false;
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
			tokensBefore: before,
			tokensAfter: after,
		} as RuntimeEvent);
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
		await this.extensionManager?.getLoadPromise();
		await this.runStartupHooksOnce();
		this.ensureHarness();
		// MCP discovery already started in ToolRouter's constructor — fire-
		// and-forget from the moment the bridge exists, not gated behind
		// init() or the first message. loadMcpToolsOnce() is memoized, so this
		// just observes the same in-flight/settled load instead of starting a
		// second one; the "Loaded N server(s)" notice fires once, from inside
		// loadMcpToolsOnce() itself, whenever that load actually finishes.
		void this.loadMcpToolsOnce().catch(error => this.reportError(error));
		const toolNames =
			this.harness?.tools?.list().map((t: Tool) => t.name) ||
			this._defaultTools.map(t => t.name);
		const status = {
			mcpServerCount: this._mcpServerCount,
			mcpToolCount: this._mcpToolNames.size,
			mcpErrors: this._mcpErrors,
			mcpLoaded: this._mcpLoaded,
			mcpLoading: this._mcpLoadPromise !== null && !this._mcpLoaded,
			skillsInjected: this._skillsInjected,
			skillsVisible: !!this._skillsContext,
			loadedSkills: this._loadedSkills,
			enabledPluginRoots: this._enabledPluginRoots,
		};
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
		return this.extensionManager?.getCommands() ?? [];
	}

	invokeExtensionCommand(
		name: string,
		args: string,
	): Promise<string | undefined> {
		return (
			this.extensionManager?.executeCommand(name, args) ??
			Promise.resolve(undefined)
		);
	}

	async stop(): Promise<void> {
		void this.cancel();
		// Abort extraction and wait for background tasks to complete.
		await this.memoryManager?.waitForBackgroundTasks();
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
		lines.push("## Conversation", "");
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

		// Memory retrieval is a request-time context block, not persistent
		// conversation history. Include the same synthetic block here so /context
		// displays the effective provider payload instead of only the harness log.
		// The backend re-roles these trailing system messages to `user` for chat
		// templates that require a system message at position 0.
		if (memoryContext) {
			lines.push("[USER]", memoryContext, "");
		}

		return `## Context (${msgs.length} messages, ~${contextTokens} tokens)\n\n${lines.join("\n")}`;
	}

	private getMemoryContextForInspection(messages: Message[]): string {
		if (!this.memoryManager) return "";
		// Cast: Message content is string | null but manager expects string | undefined
		const msgArray = messages as Array<{ role: string; content?: string }>;
		return this.memoryManager.getContextForInspection(msgArray);
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
		registry.registerMany(this._defaultTools);
		return registry;
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
	 * Emit a Pi user_bash event for Pi extension interception.
	 * Call from bash execution before running the command.
	 * @returns {action: 'continue'|'intercept'|'replace', result?, operations?} from the first non-null handler.
	 */
	async emitUserBashEvent(
		command: string,
		excludeFromContext: boolean = false,
	): Promise<{
		action: "continue" | "intercept" | "replace";
		result?: { output: string; exitCode: number; cancelled: boolean };
		operations?: unknown;
	} | null> {
		return (
			this.extensionManager?.emitUserBashEvent(command, excludeFromContext) ??
			null
		);
	}

	/**
	 * Emit a Pi project_trust event for Pi extension interception.
	 * Call before making a trust decision.
	 * @returns {trusted: 'yes'|'no'|'undecided', remember?} from the first non-null handler.
	 */
	async emitProjectTrustEvent(cwd: string): Promise<{
		trusted: "yes" | "no" | "undecided";
		remember?: boolean;
	} | null> {
		return this.extensionManager?.emitProjectTrustEvent(cwd) ?? null;
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
			"../infrastructure/tools/utils/shell.ts"
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

export { getSkillsDirs } from "./resource-directories.ts";
