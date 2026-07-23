// ── AgentCoreBridge ──────────────────────────────────────────────────────────────
import { envNumber, tableRow } from "../tui-utils.ts";
// Replaces the Python bridge with direct TypeScript agent-core integration.
// Translates agent-core events to the same shapes the transcript expects.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import {
	readFile as readFileAsync,
	readdir as readdirAsync,
} from "node:fs/promises";
import { parseFrontmatter } from "@logician/agent-core/tools/shared/frontmatter.ts";
import os from "node:os";
import path from "node:path";
import { OpenAIBackend } from "@logician/agent-core/core/backend.ts";
import {
	createDefaultTools,
	DEFAULT_SEARXNG_URL,
} from "../tools/default-tools.ts";
import {
	type AgentConfig,
	type AgentModelConfig,
	type AgentEvent,
	AgentHarness,
	type HarnessPhase,
	type Message,
	type Tool,
	type WebSearchConfig,
	type TruncationConfig,
} from "@logician/agent-core";
import {
	type McpSnapshotResult,
	type McpToggleResult,
	McpManager,
} from "../mcp/index.ts";
import {
	type PermissionMode,
	PermissionManager,
	type PermissionRules,
} from "@logician/agent-core/tools/shared/permissions.ts";
import {
	estimateChatPayloadTokens,
	estimateTokens,
} from "@logician/agent-core/core/messages.ts";
import {
	configurePluginRuntimeEnv,
	type PluginCommandResult,
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
} from "@logician/agent-core/tools/shared/plugins.ts";
import {
	formatSkillCatalog,
	findSkillByName,
	formatSkillInvocation,
	loadSkills,
	type Skill,
} from "../skills.ts";
import {
	type AgentDefinition,
	loadAgentDefinitions,
} from "@logician/agent-capabilities/subagents/subagent.ts";
import {
	getBuiltInSubagentTools,
	type SubagentToolDeps,
} from "@logician/agent-capabilities/tools";
import { buildDefaultSystemPrompt } from "../system-prompt.ts";
import { createReadSkillTool } from "../tools/read-skill.ts";
import { ToolRegistry } from "@logician/agent-core/tools/shared/registry.ts";
import { onTodosChanged } from "@logician/agent-core/core/todo-state.ts";
import { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import { findLogicianConfig, loadLogicianConfig } from "../configuration/config.ts";
import type { ParsedBridgeEvent } from "./events.ts";
import {
	createMemorySystem,
	type MemoryStore,
	createRecallTool,
	RECALL_TOOL_NAME,
	hashId,
} from "@logician/observational-memory";
import { createPostEditDiagnosticHooks } from "./post-edit-diagnostics.ts";
import { LspManager } from "./lsp-manager.ts";

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

// ── Event shape mapping ─────────────────────────────────────────────────────────

function mapAgentEvent(event: AgentEvent): ParsedBridgeEvent | null {
	switch (event.type) {
		case "message_start":
			return {
				type: "message_start",
				turnId: event.turnId,
				role: event.role,
			} as ParsedBridgeEvent;
		case "text_start":
			return { type: "text_start", turnId: event.turnId };
		case "text_delta":
			return { type: "token", token: event.delta };
		case "text_end":
			return { type: "text_end", turnId: event.turnId };
		case "message_update":
			return {
				type: "message_update",
				turnId: event.turnId,
				message: event.message,
			} as ParsedBridgeEvent;
		case "thinking_delta":
			return { type: "thinking_token", token: event.delta };
		case "tool_call_start":
			return {
				type: "tool_execution_start",
				tool: event.toolName,
				tool_name: event.toolName,
				tool_args: parseToolArgs(event.args),
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_call_end":
			return {
				type: "tool_execution_end",
				tool: event.toolName,
				tool_name: event.toolName,
				result: event.result,
				is_error: event.isError,
				tool_call_id: event.toolCallId,
				details: event.details,
			} as ParsedBridgeEvent;
		case "tool_call_delta":
			return {
				type: "tool_execution_update",
				tool: "",
				tool_name: "",
				partial_result: event.delta,
				update_kind: "arguments",
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_call_update":
			return {
				type: "tool_execution_update",
				tool: event.toolName,
				tool_name: event.toolName,
				partial_result: event.partialResult,
				update_kind: "output",
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "repair_nudge":
			return {
				type: "repair_nudge",
				turn_id: event.turnId,
				repair_stage: event.repairStage,
				tool_name: event.toolName,
				message: event.message,
			};
		case "turn_start":
		case "turn_end":
		case "agent_start":
		case "agent_end":
		case "phase":
			return null; // Handled separately
		case "context_update":
			return {
				type: "context_update",
				tokens: event.tokens,
				max_tokens: event.maxTokens,
				compacted: event.compacted,
			};
		case "compaction":
			return {
				type: "compaction",
				reason: event.reason,
				tokens_before: event.tokensBefore,
				tokens_after: event.tokensAfter,
			};
		case "error":
			return {
				type: "notice",
				level: "error",
				label: "Error",
				text: event.message,
			};
		case "auto_retry_start":
			return {
				type: "notice",
				level: "warn",
				label: `Retry ${event.attempt}/${event.maxRetries}`,
				text: `${event.error} — retrying in ${formatDelay(event.delayMs)}`,
			};
		case "auto_retry_end":
			return {
				type: "notice",
				level: event.success ? "success" : "warn",
				label: `Retry ${event.attempt}`,
				text: event.success ? "succeeded" : "failed",
			};
		case "run_outcome":
			if (event.status === "completed" && event.source === "heuristic") {
				return null;
			}
			return {
				type: "notice",
				level:
					event.status === "completed"
						? "success"
						: event.status === "failed"
							? "error"
							: "warn",
				label: `Run ${event.status.replace("_", " ")}`,
				text: event.summary || `Run ended with status: ${event.status}`,
			};
		case "model_select":
			return {
				type: "notice",
				level: "info",
				label: "Model",
				text: event.model,
			};
		case "subagent_start":
			return {
				type: "notice",
				level: "info",
				label: `Subagent ${event.agent}`,
				text: `started: ${event.task.slice(0, 120)}`,
			};
		case "subagent_end":
			return {
				type: "notice",
				level: event.isError ? "warn" : "success",
				label: `Subagent ${event.agent}`,
				text: event.isError
					? event.result
					: `done${event.turns ? ` in ${event.turns} turn(s)` : ""}`,
			};
		case "subagent_event": {
			// Render the child's tool activity as compact notice lines so the user
			// can follow what a subagent is doing; its streamed text already flows
			// through the parent spawn_agent tool cell via onUpdate.
			const child = event.event;
			if (child.type === "tool_call_start") {
				return {
					type: "notice",
					level: "info",
					label: `↳ ${event.agentId}`,
					text: `▶ ${child.toolName}${truncateArgs(child.args)}`,
				};
			}
			if (child.type === "tool_call_end") {
				return {
					type: "notice",
					level: child.isError ? "warn" : "success",
					label: `↳ ${event.agentId}`,
					text: `${child.isError ? "✗" : "✓"} ${child.toolName} ${child.result.slice(0, 240)}`,
				};
			}
			if (child.type === "error") {
				return {
					type: "notice",
					level: "warn",
					label: `↳ ${event.agentId}`,
					text: child.message,
				};
			}
			return null;
		}
		case "tool_permission_request":
			return {
				type: "notice",
				level: "warn",
				label: "Permission",
				text: `${event.toolName} awaiting approval`,
			};
		case "tool_permission_decision":
			return {
				type: "notice",
				level: event.decision === "deny" ? "warn" : "info",
				label: "Permission",
				text: `${event.toolName}: ${event.decision} (${event.source})`,
			};
		case "budget_exhausted":
			return {
				type: "notice",
				level: "warn",
				label: "Budget",
				text: `token budget exhausted (${event.usedTokens}/${event.limitTokens}) — run stopped.`,
			};
		case "max_iterations":
			return {
				type: "notice",
				level: "warn",
				label: "Stopped",
				text: `reached the ${event.limit}-turn safety limit without finishing (${event.iterations} turns).`,
			};
		default:
			return null;
	}
}

// One-line argument preview for subagent tool notices.
// Extracts the most meaningful key (path, pattern, command) from JSON args
// so subagent tool lines are human-readable instead of raw JSON fragments.
function truncateArgs(args: string): string {
	const flat = (args || "").replace(/\s+/g, " ").trim();
	if (!flat || flat === "{}" || flat === "{") return "";
	// Always try to extract the most meaningful single key for common tools.
	const key = pickArgKey(args, flat);
	return key ? ` ${key}` : ` ${flat}`;
}

/** Pick the most meaningful key-value pair for a tool argument string. */
function pickArgKey(_raw: string, flat: string): string | null {
	const priorities = [
		// File operations
		"path",
		"file_path",
		// Search
		"pattern",
		"glob",
		// Bash
		"command",
		// MCP / generic
	];
	for (const key of priorities) {
		// Match "key":"value" or "key": "value" (handle escaped quotes in value).
		const re = new RegExp(
			`"${key}"\\s*:\\s*(?:"((?:[^"\\\\]|\\\\.)*)"|([^,}\\s]+))`,
		);
		const m = re.exec(flat);
		if (m) {
			const val = m[1] || m[2] || "";
			return `${key}=${val}`;
		}
	}
	// Fallback: first string value.
	const sm = flat.match(/"([^"]{1,80})"/);
	return sm ? sm[1].slice(0, 80) : null;
}

// Humanize a backoff delay for retry notices: "500ms", "1.0s", "4.0s".
function formatDelay(ms: number): string {
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function parseToolArgs(args: string): Record<string, unknown> | undefined {
	try {
		const parsed = JSON.parse(args || "{}");
		return parsed && typeof parsed === "object" ? parsed : undefined;
	} catch (e: unknown) {
		return undefined;
	}
}

function createHookTranscriptPath(cwd: string, sessionId: string): string {
	const safeCwd = cwd
		.replace(/[^a-zA-Z0-9._-]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 96);
	const dir = path.join(
		os.homedir(),
		".logician",
		"tui",
		"sessions",
		safeCwd || "workspace",
	);
	const transcriptPath = path.join(dir, `${sessionId}.jsonl`);
	try {
		mkdirSync(dir, { recursive: true });
		writeFileSync(
			transcriptPath,
			`${JSON.stringify({
				type: "session",
				timestamp: new Date().toISOString(),
				session_id: sessionId,
				cwd,
			})}\n`,
			"utf8",
		);
	} catch (e: unknown) {
		return "";
	}
	return transcriptPath;
}

function buildPluginRuntimeEnv(opts: AgentBridgeOptions): NodeJS.ProcessEnv {
	const model = opts.model?.trim() || "";
	const baseUrl = opts.baseUrl?.trim().replace(/\/+$/, "");
	const env: NodeJS.ProcessEnv = {};
	if (baseUrl) {
		env.CLAUDE_MEM_MODEL = model;
		env.CLAUDE_MEM_OPENROUTER_MODEL = model;
		env.CLAUDE_MEM_TIER_ROUTING_ENABLED = "false";
		env.CLAUDE_MEM_TIER_SIMPLE_MODEL = "";
		env.CLAUDE_MEM_TIER_SUMMARY_MODEL = "";
		env.CLAUDE_MEM_TIER_FAST_MODEL = "";
		env.CLAUDE_MEM_TIER_SMART_MODEL = "";
		env.CLAUDE_MEM_PROVIDER = "openrouter";
		env.CLAUDE_MEM_OPENROUTER_BASE_URL = baseUrl;
		env.OPENROUTER_BASE_URL = baseUrl;
		env.CLAUDE_MEM_OPENROUTER_API_KEY =
			process.env.CLAUDE_MEM_OPENROUTER_API_KEY ||
			process.env.OPENROUTER_API_KEY ||
			"logician-local";
		env.OPENROUTER_API_KEY = env.CLAUDE_MEM_OPENROUTER_API_KEY;
	}
	return env;
}

// SearXNG web search defaults to DEFAULT_SEARXNG_URL; override the instance via
// LOGICIAN_SEARXNG_URL and result count via LOGICIAN_SEARXNG_MAX_RESULTS.
function resolveWebSearchConfig(): WebSearchConfig {
	return {
		baseUrl: process.env.LOGICIAN_SEARXNG_URL?.trim() || DEFAULT_SEARXNG_URL,
		maxResults: envNumber("LOGICIAN_SEARXNG_MAX_RESULTS"),
	};
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
	loopDetectionEnabled?: boolean;
	guardsEnabled?: boolean;
	continuationEnabled?: boolean;
	postEditDiagnostics?: boolean;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	truncation?: TruncationConfig;
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
	private cwd: string;
	private defaultTools: Tool[];
	private mcpManager = new McpManager();
	private mcpLoaded = false;
	private mcpLoadPromise: Promise<void> | null = null;
	private mcpServerCount = 0;
	private mcpErrors: string[] = [];
	private baseSystemPrompt: string;
	private additionalSystemPrompt?: string;
	private pluginSystemContext = "";
	private skillsContext: string | null = null;
	private skillsInjected: boolean = false;
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

	// ── Observational memory (V3) ───────────────────────────────────────
	private memoryStore: MemoryStore | null = null;
	private memoryEventBus = new ExtensionEventBus({ defaultTimeoutMs: 30_000 });

	constructor(
		opts: AgentBridgeOptions = {
			baseUrl: "http://localhost:8080",
			model: "",
		},
	) {
		this.cwd = opts.cwd || process.cwd();
		this.configPath = findLogicianConfig(this.cwd);
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
		const serverOverrides: Record<string, { command: string; args: string[]; languageId: string }> = {};
		try {
			const resolved = loadLogicianConfig(this.cwd);
			const lspCfg = resolved.config.lsp;
			if (lspCfg !== undefined) {
				if (lspCfg.enabled === false) lspEnabled = false;
				if (lspCfg.timeoutMs !== undefined && lspCfg.timeoutMs > 0) lspTimeoutMs = lspCfg.timeoutMs;
				if (lspCfg.serverOverrides) {
					Object.assign(serverOverrides, lspCfg.serverOverrides);
				}
			}
		} catch {
			// Config load failure is non-fatal; LSP stays on with defaults.
		}
		this.lspManager = new LspManager(this.cwd, {
			timeoutMs: lspTimeoutMs,
			servers: Object.keys(serverOverrides).length > 0 ? serverOverrides : undefined,
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
			loopDetectionEnabled: opts.loopDetectionEnabled,
			guardsEnabled: opts.guardsEnabled,
			continuationEnabled: opts.continuationEnabled,
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
						question: ctx.question,
						choices: ctx.choices,
					});
				}),
			hooks: createPostEditDiagnosticHooks(
				this.cwd,
				() => this.postEditDiagnosticsEnabled,
				this.lspManager,
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
			} catch (e: unknown) {
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
		try {
			await this.runStartupHooksOnce();
			// MCP discovery is opportunistic. A slow or broken external server must
			// never hold the user's prompt before it reaches the model. Tools that
			// finish loading are added to the live harness for subsequent turns.
			if (!this.mcpLoaded && !this.mcpLoadPromise) {
				void this.loadMcpToolsOnce().catch((error) => this.reportError(error));
			}
			// Reuse one harness across messages so conversation history (and thus
			// "continue" / "go on" follow-ups) persists. Created lazily once.
			const harness = this.ensureHarness();

			// Emit turn start
			const turnId = `turn_${Date.now()}`;
			// turnId tracked via turn lifecycle, no separate field needed
			this.emit({ type: "turn_start", turn_id: turnId });

			await harness.prompt(message);
		} catch (e: unknown) {
			const error = e as Error;
			// Emit a visible error notice so the user sees connection/server
			// failures in the transcript rather than only in the console.
			this.emit({
				type: "notice",
				level: "error",
				label: "Error",
				text: error.message,
			});
			this.errorCb?.(error);
		} finally {
			this.running = false;
			this.publishContextUsage();
			// Keep the harness alive to retain history across turns.
			// turn lifecycle ended
			this.emit({ type: "phase", state: "ready" });
			if (this.pendingAutoContinue) {
				this.pendingAutoContinue = false;
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
			this.harness.setExtensionBus(this.memoryEventBus);
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

			// ── Observational memory (V3) ────────────────────────────────
			this.initObservationalMemory();
		}
		return this.harness;
	}

	/** Initialize the V3 observational memory system. */
	private initObservationalMemory(): void {
		try {
			const model = this.config.model || "gpt-4o";
			const apiKey = process.env.OPENAI_API_KEY || "";

			const memorySystem = createMemorySystem({
				model,
				apiKey,
				baseUrl: this.config.baseUrl,
				persistencePath: path.join(
					this.cwd,
					".logician/observational-memory",
					"memory.json",
				),
			});

			this.memoryStore = memorySystem.store;

			// Load persisted memory
			this.memoryStore.load();
			const unsubscribeStore = this.memoryStore.subscribe((event) => {
				if (event.type === "observations_added") {
					this.emit({
						type: "memory_update",
						kind: event.type,
						count: event.observations.length,
						items: event.observations.map((observation) => ({
							id: observation.id,
							content: observation.content,
							relevance: observation.relevance,
						})),
					});
				} else if (event.type === "reflections_added") {
					this.emit({
						type: "memory_update",
						kind: event.type,
						count: event.reflections.length,
					});
				} else if (event.type === "observations_dropped") {
					this.emit({
						type: "memory_update",
						kind: event.type,
						count: event.observationIds.length,
					});
				} else {
					this.emit({ type: "memory_update", kind: "cleared", count: 0 });
				}
			});

			// Wire up the consolidation pipeline
			const unsubHooks = memorySystem.registerHooks(this.memoryEventBus, {
				currentTokens: () => this.contextTokens,
				getSourceEntries: () =>
					(this.harness?.messages ?? [])
						.filter(
							(message) =>
								typeof message.content === "string" && message.content.trim(),
						)
						.map((message, index) => ({
							id: hashId(`${index}:${message.role}:${message.content}`),
							role: message.role,
							content: String(message.content),
						})),
			});

			// Add recall tool to agent's tool registry
			const recallHandler = createRecallTool({
				memoryStore: this.memoryStore,
				sourceEntries: [],
			});
			const recallTool: Tool = {
				name: RECALL_TOOL_NAME,
				description: "Recover exact evidence for an observational memory ID",
				parameters: {
					type: "object",
					properties: { id: { type: "string", pattern: "^[a-f0-9]{12}$" } },
				},
				execute: async (args: Record<string, unknown>) => {
					const result = recallHandler(args.id as string);
					return JSON.stringify(result);
				},
			};
			if (!this.defaultTools.some((tool) => tool.name === RECALL_TOOL_NAME)) {
				this.defaultTools = [...this.defaultTools, recallTool];
				this.config.tools = this.defaultTools;
				this.harness?.setTools(this.defaultTools);
			}

			// Cleanup on harness destroy
			this.harness!.setOnShutdown(() => {
				this.memoryStore?.save();
				unsubHooks();
				unsubscribeStore();
			});
		} catch (error) {
			console.error("[observational-memory] init failed:", error);
		}
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
		} catch (e: unknown) {
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
	getConfig(): { baseUrl: string; model: string } {
		return { baseUrl: this.config.baseUrl, model: this.config.model };
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
				text: count > 0
					? `Processing ${count} queued steering message${count === 1 ? "" : "s"} now.`
					: "No queued steering messages to process.",
			});
			return;
		}
		if (trimmed === "/queue") {
			const steering = this.getSteeringMessages();
			const followUp = this.getFollowUpMessages();
			const rows = [...steering.map((message) => `▸ ${message}`), ...followUp.map((message) => `↳ ${message}`)];
			this.emit({ type: "notice", level: "info", label: "Queue", text: rows.length ? rows.map((row, index) => `${index + 1}. ${row}`).join("\n") : "Queue is empty." });
			return;
		}
		if (trimmed === "/queue-clear") {
			const cleared = this.clearQueue();
			const count = cleared.steering.length + cleared.followUp.length + cleared.nextTurn.length;
			this.emit({ type: "notice", level: "info", label: "Queue", text: `Cleared ${count} queued message${count === 1 ? "" : "s"}.` });
			return;
		}
		if (trimmed === "/queue-drop" || trimmed.startsWith("/queue-drop ")) {
			const value = Number.parseInt(trimmed.slice("/queue-drop".length).trim(), 10);
			const removed = Number.isInteger(value) && value > 0 ? this.dropQueuedMessage(value - 1) : undefined;
			this.emit({ type: "notice", level: removed ? "info" : "warn", label: "Queue", text: removed ? `Removed: ${removed}` : "Usage: /queue-drop <number>" });
			return;
		}
		// /om:status — show observational memory status
		if (trimmed === "/om:status") {
			const status =
				this.getMemoryStatus() || "Observational memory not available.";
			this.emit({
				type: "turn_end",
				turn_id: "om:status",
				message: status,
			} as ParsedBridgeEvent);
			return;
		}
		// /jb — inject jb.md content as user prompt
		if (trimmed === "/jb" || trimmed.startsWith("/jb ")) {
			try {
				const content = findJbPrompt(this.cwd);
				if (!content) {
					this.emit({
						type: "notice",
						level: "warn",
						label: "JB prompt",
						text: `jb.md was not found in ${this.cwd} or ${path.join(this.cwd, "tui")}.`,
					});
					return;
				}
				this.sendMessage(content).catch((err) => this.errorCb?.(err));
				this.emit({
					type: "notice",
					level: "info",
					label: "JB prompt",
					text: `Injected jb.md as the user prompt (${content.length.toLocaleString()} characters).`,
				});
			} catch (e: unknown) {
				this.errorCb?.(e as Error);
			}
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
		this.startupHooksRan = false;
		this.startupHookResult = null;
		this.pluginSystemContext = "";

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
			? { ...skill, content: skill.content.replaceAll("$ARGUMENTS", trimmedArgs) }
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
			question,
			choices,
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

	getModelOptions(): Array<{ key: string; name: string; model: string; url: string; active: boolean }> {
		const configured = this.config.models ?? [];
		if (configured.length === 0) {
			return [{ key: this.getCurrentModel(), name: this.getCurrentModel(), model: this.getCurrentModel(), url: this.getCurrentBaseUrl(), active: true }];
		}
		return configured.map((option, index) => {
			const url = option.url || this.config.baseUrl;
			return {
				key: `${index}:${option.name}`,
				name: option.name,
				model: option.model,
				url,
				active: option.model === this.getCurrentModel() && url === this.getCurrentBaseUrl(),
			};
		});
	}

	setModelOption(key: string): { model: string; url: string } | null {
		const option = this.getModelOptions().find((candidate) => candidate.key === key);
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
			mcp_tools: this.defaultTools.filter((tool) =>
				tool.name.startsWith("mcp__"),
			).length,
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

		return this.formatPluginResult(backendAction, result);
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

	setRuntimeToggle(
		key:
			| "loopDetectionEnabled"
			| "guardsEnabled"
			| "proactiveCompactionEnabled"
			| "postEditDiagnostics",
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
			`  Loop detection: ${this.config.loopDetectionEnabled ? "on" : "off"}`,
			`  Guards: ${this.config.guardsEnabled ? "on" : "off"}`,
			`  Compaction: ${this.config.proactiveCompactionEnabled ? "on" : "off"}`,
			`  Post-edit diagnostics: ${this.postEditDiagnosticsEnabled ? "on" : "off"}`,
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
		loopDetectionEnabled: boolean;
		guardsEnabled: boolean;
		proactiveCompactionEnabled: boolean;
		postEditDiagnostics: boolean;
	} {
		return {
			model: this.config.model,
			temperature: this.config.temperature ?? 0.5,
			maxTokens: this.config.maxTokens ?? 4096,
			maxIterations: this.config.maxIterations ?? 30,
			thinkingLevel: this.config.thinkingLevel ?? "off",
			inferenceMode: this.config.inferenceMode ?? "instruct-general",
			permissionMode: this.getPermissionMode(),
			loopDetectionEnabled: this.config.loopDetectionEnabled ?? false,
			guardsEnabled: this.config.guardsEnabled ?? false,
			proactiveCompactionEnabled: this.config.proactiveCompactionEnabled ?? false,
			postEditDiagnostics: this.postEditDiagnosticsEnabled,
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
		// Reset skill injection state
		this.skillsContext = null;
		this.skillsInjected = false;
		this.startupHooksRan = false;
		this.pluginSystemContext = "";
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

	// ── Observational memory ─────────────────────────────────────────────

	/** Get the current memory status (for /om:status). */
	getMemoryStatus(): string | null {
		if (!this.memoryStore) return "Observational memory is not initialized.";
		const status = this.memoryStore.getStatus();
		return [
			`Observational Memory`,
			`  Observations: ${status.observationCount} (${status.droppedCount} dropped)`,
			`  Reflections: ${status.reflectionCount}`,
			`  Active tokens: ${status.activeObservationTokens.toLocaleString()} / ${status.observationPoolTargetTokens.toLocaleString()} target`,
		].join("\n");
	}

	/** Execute the local `/memory` command family. */
	memoryCommand(raw = ""): string {
		if (!this.memoryStore) return "Observational memory is not initialized.";
		const [action = "status", ...rest] = raw
			.trim()
			.split(/\s+/)
			.filter(Boolean);
		const query = rest.join(" ").toLowerCase();
		if (action === "status")
			return this.getMemoryStatus() ?? "Memory unavailable.";
		if (action === "clear") {
			this.memoryStore.clear();
			return "Observational memory cleared.";
		}
		if (action === "add") {
			const content = rest.join(" ").trim();
			if (!content) return "Usage: /memory add <text>";
			const id = hashId(content.trim().replace(/\s+/g, " ").toLowerCase());
			this.memoryStore.recordObservations(
				[
					{
						id,
						content,
						timestamp: new Date().toISOString(),
						relevance: "high",
						sourceEntryIds: ["manual"],
						tokenCount: Math.ceil(content.length / 4),
					},
				],
				"manual",
			);
			return `Memory pinned: [${id}] ${content}`;
		}
		if (action === "drop") {
			const id = rest[0];
			if (!id) return "Usage: /memory drop <id>";
			this.memoryStore.recordDrops([id], "manual");
			return this.memoryStore.isDropped(id)
				? `Memory archived: ${id}`
				: `Active memory not found: ${id}`;
		}
		const observations = this.memoryStore.getActiveObservations();
		const reflections = this.memoryStore.getReflections();
		if (action === "list" || action === "search") {
			const filtered = query
				? observations.filter((item) =>
						item.content.toLowerCase().includes(query),
					)
				: observations;
			if (!filtered.length)
				return query
					? `No memories match "${query}".`
					: "No active observations.";
			return filtered
				.slice(-20)
				.map((item) => `[${item.id}] ${item.relevance}: ${item.content}`)
				.join("\n");
		}
		if (action === "reflections") {
			return reflections.length
				? reflections
						.slice(-20)
						.map((item) => `[${item.id}] ${item.content}`)
						.join("\n")
				: "No reflections.";
		}
		if (action === "show") {
			const id = rest[0];
			const observation = this.memoryStore
				.getAllObservations()
				.find((item) => item.id === id);
			const reflection = reflections.find((item) => item.id === id);
			if (observation)
				return `[${observation.id}] ${observation.relevance}\n${observation.content}\nSources: ${observation.sourceEntryIds.join(", ")}`;
			if (reflection)
				return `[${reflection.id}] reflection\n${reflection.content}\nSupports: ${reflection.supportingObservationIds.join(", ")}`;
			return `Memory not found: ${id ?? "missing id"}`;
		}
		return "Usage: /memory [status|list|search <text>|show <id>|add <text>|drop <id>|reflections|clear]";
	}

	/** Recall a memory item by ID (for the `recall` tool). */
	recall(_memoryId: string): { content: string; status: string } | null {
		if (!this.memoryStore) return null;
		// This is handled by the agent's tool system, not the bridge directly.
		// The bridge just exposes the store to the tool.
		return null;
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
		} catch (e: unknown) {
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
		if (this.mcpEager) {
			// MCP transports can take seconds to connect. Do not hold the opening
			// screen behind external servers; the first turn still awaits this same
			// promise before building its tool snapshot.
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
			mcp_tools_loaded: this.defaultTools.filter((tool) =>
				tool.name.startsWith("mcp__"),
			).length,
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
				? await this.countInstalledSkills()
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

	private async countInstalledSkills(): Promise<number> {
		const registry = await runPluginBackend("list", []);
		const plugins = registry.plugins || [];
		let count = 0;
		for (const plugin of plugins) {
			const enabled = plugin.enabled !== false;
			const onDisk = plugin.on_disk !== false;
			const installPath = String(plugin.install_path || "");
			if (!enabled || !onDisk || !installPath) continue;
			try {
				const entries = await readdirAsync(path.join(installPath, "skills"));
				count += entries.filter(
					(e) => e !== ".git" && !e.startsWith("."),
				).length;
			} catch (e: unknown) {
				// no skills dir
			}
		}
		return count;
	}

	async stop(): Promise<void> {
		this.cancel();
		await this.fireSessionEnd("shutdown");
		this.lspManager.close();
		await this.mcpManager.close();
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
			(zone) => `- ${zone.name}: ~${zone.tokens} tokens${zone.detail ? ` — ${zone.detail}` : ""}`,
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
				// Tool result: show name + truncated result
				const callId = msg.tool_call_id || "";
				const name =
					msgs
						.find(
							(m) =>
								m.role === "assistant" &&
								m.tool_calls?.some((tc) => tc.id === callId),
						)
						?.tool_calls?.find((tc) => tc.id === callId)?.name || "tool";
				const truncated =
					msg.content.length > 2000
						? `${msg.content.slice(0, 2000)}\n... [truncated]`
						: msg.content;
				lines.push(`${header} (${name})\n${truncated}`);
			} else if (msg.role === "assistant" && msg.tool_calls?.length) {
				// Assistant with tool calls
				lines.push(
					`${header}\n${msg.content || "(no content)"}\n\nTool calls:`,
				);
				for (const tc of msg.tool_calls) {
					lines.push(`  - ${tc.name}(${tc.arguments || ""})`);
				}
			} else {
				const truncated =
					msg.content && msg.content.length > 2000
						? `${msg.content.slice(0, 2000)}\n... [truncated]`
						: msg.content || "";
				lines.push(`${header}\n${truncated}`);
			}
			lines.push("");
		}

		return `## Context (${msgs.length} messages, ~${contextTokens} tokens)\n\n${lines.join("\n")}`;
	}

	getContextSourceMap(): Array<{ name: string; tokens: number; detail: string }> {
		const messages = this.getMessages();
		const conversation = messages.filter((message) => message.role !== "tool");
		const toolEvidence = messages.filter((message) => message.role === "tool");
		const memory = this.harness?.getMemoryPrompt() ?? "";
		return [
			{ name: "Base instructions", tokens: estimateTokens(this.baseSystemPrompt), detail: "system zone" },
			{ name: "Plugin context", tokens: estimateTokens(this.pluginSystemContext), detail: "startup hooks" },
			{ name: "Skill catalog", tokens: estimateTokens(this.skillsContext ?? ""), detail: `${this.loadedSkills.length} loaded` },
			{ name: "Memory", tokens: estimateTokens(memory), detail: memory ? "active" : "empty" },
			{ name: "Conversation", tokens: conversation.length ? estimateChatPayloadTokens(conversation) : 0, detail: `${conversation.length} messages` },
			{ name: "Tool evidence", tokens: toolEvidence.length ? estimateChatPayloadTokens(toolEvidence) : 0, detail: `${toolEvidence.length} results` },
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
			allowAllPaths: this.config.allowAllPaths,
			maxResultChars: this.config.truncation?.toolResultMaxChars,
		});
		registry.registerMany(this.defaultTools);
		return registry;
	}

	private async loadMcpToolsOnce(): Promise<void> {
		if (this.mcpLoaded || process.env.LOGICIAN_MCP === "0") return;
		if (!this.mcpLoadPromise) {
			this.mcpLoadPromise = (async () => {
				const result = await this.mcpManager.load(this.config.cwd || process.cwd());
				this.mcpServerCount = result.servers;
				this.mcpErrors = result.errors;
				if (result.tools.length) {
					const existing = new Set(this.defaultTools.map((tool) => tool.name));
					const newTools = result.tools.filter((tool) => !existing.has(tool.name));
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
		const contexts: string[] = [];
		if (this.pluginSystemContext) contexts.push(this.pluginSystemContext);
		if (this.skillsContext) contexts.push(this.skillsContext);
		if (contexts.length) {
			this.config.systemPrompt = `${this.baseSystemPrompt}\n\n${contexts.join("\n\n")}`;
		} else {
			this.config.systemPrompt = this.baseSystemPrompt;
		}
	}

	private buildBaseSystemPrompt(): string {
		const defaultPrompt = buildDefaultSystemPrompt(this.cwd, this.defaultTools);
		return this.additionalSystemPrompt
			? `${defaultPrompt}\n\nAdditional user/system instructions:\n${this.additionalSystemPrompt}`
			: defaultPrompt;
	}

	private applyPluginHookContext(result: PluginCommandResult): void {
		const contexts = (result.additional_contexts || [])
			.map((item) => String(item || "").trim())
			.filter(Boolean);
		if (!contexts.length) {
			this.pluginSystemContext = "";
			this.config.systemPrompt = this.baseSystemPrompt;
			return;
		}

		this.pluginSystemContext = `<startup-hook-context>\n${contexts.join("\n\n")}\n</startup-hook-context>`;
		this.config.systemPrompt = `${this.baseSystemPrompt}\n\n${this.pluginSystemContext}`;
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
		skillsDirs.push(...getProjectSkillDirs(cwd));

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
		if (!visible.length) return;

		// Inject a compact catalog (name + description), not full bodies. The
		// model loads a skill's full instructions on demand via read_skill.
		this.skillsContext = formatSkillCatalog(visible);

		// Register the read_skill tool bound to the loaded skills so the model can
		// pull full bodies. Append to the tool set (next loop turn picks it up) and
		// patch the live harness registry if a run is already active.
		const readSkill = createReadSkillTool(visible);
		if (readSkill && !this.defaultTools.some((t) => t.name === "read_skill")) {
			this.defaultTools = [...this.defaultTools, readSkill];
			this.config.tools = this.defaultTools;
			this.harness?.setTools(this.defaultTools);
		}

		this.rebuildBaseSystemPrompt();
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
			} catch (e: unknown) {
				continue;
			}
			for (const entry of entries) {
				if (!entry.endsWith(".md")) continue;
				const filePath = path.join(dir, entry);
				let raw: string;
				try {
					raw = await readFileAsync(filePath, "utf8");
				} catch (e: unknown) {
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
		// Skills and agents are runtime resources, independent of whether command
		// hooks are enabled.
		await this.injectSkillsFromPlugins();
		await this.injectSubagents();
	}

	/**
	 * Register the spawn_agent tool bound to discovered agent definitions
	 * (.logician/agents/*.md + built-ins). Subagent events are forwarded into
	 * the normal event stream as subagent_* envelopes.
	 */
	private async injectSubagents(): Promise<void> {
		const cwd = this.config.cwd || process.cwd();
		this.agentDefs = await loadAgentDefinitions([
			path.join(cwd, ".logician", "agents"),
			// Claude Code plugin agents (agents/*.md in each enabled plugin).
			...this.enabledPluginRoots.map((p) =>
				path.join(p.installPath, "agents"),
			),
		]);

		// Inject subagent tools
		const subagentDeps: SubagentToolDeps = {
			config: () => this.config,
			backend: this.backend,
			cwd,
			agents: () => this.agentDefs,
			emit: (event) => this.config.onEvent?.(event),
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
		} catch (e: unknown) {
			// SessionEnd hooks are best-effort during shutdown/reset.
		}
	}

	private formatPluginResult(
		action: string,
		result: PluginCommandResult,
	): string {
		if (result.status === "error") {
			return `/plugins failed: ${result.message || "unknown error"}`;
		}

		if (action === "list") {
			const plugins = result.plugins || [];
			const hooks = result.session_start_hooks || {};
			const lines = [
				"# Installed plugins",
				`Registry: ${result.plugins_dir || "unknown"}`,
			];
			if (!plugins.length) {
				lines.push("", "No plugins installed.");
				return lines.join("\n");
			}
			lines.push("", "| Plugin | Version | State | Hooks | Path |");
			lines.push("|--------|---------|-------|-------|------|");
			for (const plugin of plugins) {
				const id = String(plugin.plugin_id || plugin.name || "");
				const hookCount = hooks[id] || 0;
				const state = plugin.enabled ? "enabled" : "disabled";
				const onDisk = plugin.on_disk === false ? " missing" : "";
				lines.push(
					tableRow([
						id,
						String(plugin.version || ""),
						`${state}${onDisk}`,
						hookCount ? `SessionStart x${hookCount}` : "-",
						String(plugin.install_path || ""),
					]),
				);
			}
			return lines.join("\n");
		}

		if (action === "hooks") {
			const hooks = result.hooks || [];
			const source = String(result.source || "startup");
			const lines = [
				"# Plugin SessionStart hooks",
				`Source: ${source}`,
				`Registry: ${result.plugins_dir || "unknown"}`,
			];
			if (!hooks.length) {
				lines.push("", "No enabled SessionStart hooks matched this source.");
				return lines.join("\n");
			}
			lines.push("", "| Plugin | Matcher | Commands |");
			lines.push("|--------|---------|----------|");
			for (const hook of hooks) {
				const commands = Array.isArray(hook.commands)
					? hook.commands
							.map(
								(cmd: { type?: string; command?: string }) =>
									`${cmd.type}${cmd.command ? `: ${cmd.command}` : ""}`,
							)
							.join("<br>")
					: "";
				lines.push(
					tableRow([
						String(hook.plugin_id || hook.plugin_name || ""),
						String(hook.matcher || "*"),
						commands || "-",
					]),
				);
			}
			return lines.join("\n");
		}

		if (action === "run-hooks") {
			const lines = [
				"# Plugin hooks executed",
				`Source: ${result.source || "startup"}`,
				`Hooks: ${result.hook_count || 0}`,
				`Contexts added: ${(result.additional_contexts || []).length}`,
			];
			const errors = result.errors || [];
			if (errors.length) {
				lines.push("", "Errors:");
				lines.push(...errors.map((err) => `- ${err}`));
			}
			if ((result.additional_contexts || []).length) {
				lines.push("", "Hook context has been applied to future agent turns.");
			}
			return lines.join("\n");
		}

		if (action === "update" && Array.isArray(result.updates)) {
			const lines = ["# Plugin updates"];
			for (const update of result.updates) {
				lines.push(
					`- ${update.message || update.status || JSON.stringify(update)}`,
				);
			}
			return lines.join("\n");
		}

		if (action === "deps") {
			const issues = result.issues || [];
			if (!issues.length) return "All plugin dependencies OK.";
			const lines = ["# Plugin dependency issues"];
			for (const issue of issues) {
				lines.push(
					`- ${issue.plugin_id || "plugin"}: ${issue.status || "issue"}`,
				);
				if (Array.isArray(issue.missing) && issue.missing.length) {
					lines.push(`  Missing: ${issue.missing.join(", ")}`);
				}
			}
			return lines.join("\n");
		}

		return String(
			result.message || result.status || JSON.stringify(result, null, 2),
		);
	}
}

// Event-log path derived from the transcript path ("…/<id>.events.jsonl").
function eventLogPathFor(transcriptPath: string): string | undefined {
	if (!transcriptPath) return undefined;
	return transcriptPath.replace(/\.jsonl$/, ".events.jsonl");
}

// ── User settings (local ~/.logician/settings.json) ──────────────────────

interface UserSettings {
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	[key: string]: unknown;
}

/** Load user settings from ~/.logician/settings.json. Returns empty object on failure. */
function loadUserSettings(): UserSettings {
	const settingsPath = path.join(os.homedir(), ".logician", "settings.json");
	try {
		const raw = readFileSync(settingsPath, "utf8");
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		return typeof parsed === "object" && parsed !== null
			? (parsed as UserSettings)
			: {};
	} catch (e: unknown) {
		return {};
	}
}

/** Apply compaction settings from user settings to the harness. */
function applyCompactionSettings(
	harness: AgentHarness,
	settings: UserSettings,
): void {
	const compaction = settings.compaction;
	if (!compaction) return;

	const compactionSettings: {
		reserveTokens?: number;
		keepRecentTokens?: number;
	} = {};

	if (compaction.reserveTokens !== undefined && compaction.reserveTokens > 0) {
		compactionSettings.reserveTokens = compaction.reserveTokens;
	}
	if (
		compaction.keepRecentTokens !== undefined &&
		compaction.keepRecentTokens > 0
	) {
		compactionSettings.keepRecentTokens = compaction.keepRecentTokens;
	}

	if (Object.keys(compactionSettings).length > 0) {
		harness.setAutoCompactionSettings(compactionSettings);
	}

	if (compaction.enabled === true) {
		harness.enableAutoCompaction(true);
	}
}

// ── Directory discovery ──────────────────────────────────────────────────

/** Discover skill directories from plugins and project layout. */
export async function getSkillsDirs(cwd: string): Promise<string[]> {
	const dirs: string[] = [];

	// Plugin skills
	try {
		const registry = await runPluginBackend("list", []);
		const plugins = registry.plugins || [];
		for (const plugin of plugins) {
			const enabled = plugin.enabled !== false;
			const onDisk = plugin.on_disk !== false;
			const installPath = String(plugin.install_path || "");
			if (!enabled || !onDisk || !installPath) continue;
			dirs.push(path.join(installPath, "skills"));
		}
	} catch (e: unknown) {
		// Plugin backend may not be ready during early reload
	}

	dirs.push(path.join(os.homedir(), ".agents", "skills"));
	dirs.push(...getProjectSkillDirs(cwd));

	return Array.from(new Set(dirs));
}

function getProjectSkillDirs(cwd: string): string[] {
	const dirs: string[] = [];
	let current = path.resolve(cwd);
	while (true) {
		dirs.push(path.join(current, ".logician", "skills"));
		dirs.push(path.join(current, "skills"));
		const parent = path.dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return Array.from(new Set(dirs));
}
