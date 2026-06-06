// ── AgentCoreBridge ──────────────────────────────────────────────────────────────
// Replaces the Python bridge with direct TypeScript agent-core integration.
// Translates agent-core events to the same shapes the transcript expects.

import { mkdirSync, writeFileSync } from "node:fs";
import { readdir as readdirAsync, readFile as readFileAsync } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { AgentHarness, type AgentConfig, type AgentEvent, type Message, type Tool, type WebSearchConfig } from "./agent-core/index.ts";
import { OpenAIBackend } from "./agent-core/backend.ts";
import type { ParsedBridgeEvent } from "./events.ts";
import { ToolRegistry } from "./agent-core/tools/registry.ts";
import {
    createDefaultTools,
    DEFAULT_SEARXNG_URL,
} from "./agent-core/default-tools.ts";
import { McpManager } from "./agent-core/mcp.ts";
import { onTodosChanged } from "./agent-core/tools/todo-write.ts";
import { buildDefaultSystemPrompt } from "./agent-core/system-prompt.ts";
import {
    configurePluginRuntimeEnv,
    runHookEvent,
    runPluginBackend,
    runSessionStartHooks,
    splitPluginArgs,
    type PluginCommandResult,
} from "./agent-core/plugins.ts";
import { findLogicianConfig } from "./config.ts";

export type EventCallback = (event: ParsedBridgeEvent) => void;
export type ErrorCallback = (err: Error) => void;

// ── Event shape mapping ─────────────────────────────────────────────────────────

function mapAgentEvent(event: AgentEvent): ParsedBridgeEvent | null {
    switch (event.type) {
        case "message_start":
            return {
                type: "message_start",
                turnId: event.turnId,
                role: event.role,
            } as ParsedBridgeEvent;
        case "message_delta":
            return { type: "token", token: event.delta };
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
            } as ParsedBridgeEvent;
        case "tool_call_update":
            return {
                type: "tool_execution_update",
                tool: event.toolName,
                tool_name: event.toolName,
                partial_result: event.partialResult,
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
            return { type: "token", token: `[Error] ${event.message}` };
        case "auto_retry_start":
            return {
                type: "token",
                token: `[Retry ${event.attempt}/${event.maxRetries}] ${event.error} (wait ${event.delayMs}ms)`,
            };
        case "auto_retry_end":
            return {
                type: "token",
                token: `[Retry ${event.attempt}] succeeded`,
            };
        case "model_select":
            return {
                type: "token",
                token: `[Model] ${event.model}`,
            };
        default:
            return null;
    }
}

function parseToolArgs(args: string): Record<string, unknown> | undefined {
    try {
        const parsed = JSON.parse(args || "{}");
        return parsed && typeof parsed === "object" ? parsed : undefined;
    } catch {
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
    } catch {
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

function envNumber(name: string): number | undefined {
    const raw = process.env[name];
    if (!raw) return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
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
    chatTemplate?: string;
    temperature?: number;
    maxTokens?: number;
    maxIterations?: number;
    contextWindowTokens?: number;
    toolExecution?: AgentConfig["toolExecution"];
    runtimeHooksEnabled?: boolean;
    mcpEager?: boolean;
    tools?: Tool[];
    cwd?: string;
    systemPrompt?: string;
    webSearch?: Partial<WebSearchConfig>;
}

// ── AgentCoreBridge ─────────────────────────────────────────────────────────────

export class AgentCoreBridge {
    private config: AgentConfig;
    private backend: OpenAIBackend;
    private harness: AgentHarness | null = null;
    private callbacks: EventCallback[] = [];
    private errorCb: ErrorCallback | null = null;
    private running = false;
    private currentTurnId: string | null = null;
    private cwd: string;
    private defaultTools: Tool[];
    private mcpManager = new McpManager();
    private mcpLoaded = false;
    private mcpServerCount = 0;
    private mcpErrors: string[] = [];
    private baseSystemPrompt: string;
    private additionalSystemPrompt?: string;
    private pluginSystemContext = "";
    private skillsContext: string | null = null;
    private skillsInjected: boolean = false;
    private sessionId = `tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    private transcriptPath = "";
    private startupHooksRan = false;
    // Session-level steering queue — tracked so the UI can display pending items.
    // Removed when the loop consumes them (detected via message_start).
    private _steeringMessages: string[] = [];
    private _followUpMessages: string[] = [];
    private startupHookResult: PluginCommandResult | null = null;
    private startupPluginCount = 0;
    private contextTokens = 0;
    private contextMaxTokens?: number;
    private configPath: string | null;
    private mcpEager: boolean;

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
        this.transcriptPath = createHookTranscriptPath(
            this.cwd,
            this.sessionId,
        );
        const defaultWebSearch = resolveWebSearchConfig();
        const webSearch = {
            baseUrl: opts.webSearch?.baseUrl || defaultWebSearch.baseUrl,
            maxResults:
                opts.webSearch?.maxResults ?? defaultWebSearch.maxResults,
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

        this.config = {
            baseUrl: opts.baseUrl,
            model: opts.model,
            systemPrompt: this.baseSystemPrompt,
            tools: this.defaultTools,
            webSearch,
            cwd: this.cwd,
            maxIterations: opts.maxIterations || 30,
            temperature: opts.temperature,
            maxTokens: opts.maxTokens,
            toolExecution: opts.toolExecution,
            contextWindowTokens:
                envNumber("LOGICIAN_CONTEXT_WINDOW") ||
                envNumber("LOGICIAN_CTX_SIZE") ||
                opts.contextWindowTokens,
            runtimeHooksEnabled:
                opts.runtimeHooksEnabled ?? process.env.LOGICIAN_HOOKS !== "0",
            hookSessionId: this.sessionId,
            hookTranscriptPath: this.transcriptPath,
            turnEndCallback: (turnId: string) => {
                this.emit({ type: "turn_end", turn_id: turnId, message: "" });
            },
            onEvent: (event: AgentEvent) => {
                if (event.type === "context_update") {
                    this.contextTokens = event.tokens;
                    this.contextMaxTokens = event.maxTokens;
                }
                // Detect consumed steering messages: when the loop emits
                // message_start for an assistant response, the steering messages
                // injected into the turn have been consumed.
                if (event.type === "message_start" && event.role === "assistant") {
                    this._removeConsumedSteeringMessages();
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

    private emit(event: ParsedBridgeEvent): void {
        for (const cb of this.callbacks) {
            try {
                cb(event);
            } catch {
                // Don't let a bad handler kill the bridge
            }
        }
    }

    // ── High-level commands ──────────────────────────────────────────────

    async sendMessage(message: string): Promise<void> {
        // A message submitted while a turn is in flight steers the running
        // turn instead of starting a second concurrent run.
        if (this.running && this.harness) {
            this.harness.steer(message);
            this.emit({ type: "steered", message });
            return;
        }

        await this.runStartupHooksOnce();
        await this.loadMcpToolsOnce();
        this.running = true;
        // Reuse one harness across messages so conversation history (and thus
        // "continue" / "go on" follow-ups) persists. Created lazily once.
        if (!this.harness) {
            this.harness = new AgentHarness({
                config: this.config,
                backend: this.backend,
                cwd: this.config.cwd,
                maxIterations: this.config.maxIterations,
            });
        }

        // Emit turn start
        const turnId = `turn_${Date.now()}`;
        this.currentTurnId = turnId;
        this.emit({ type: "turn_start", turn_id: turnId });
        this.emit({ type: "phase", state: "streaming" });

        try {
            await this.harness.prompt(message);
        } catch (e: unknown) {
            const error = e as Error;
            this.errorCb?.(error);
        } finally {
            this.running = false;
            // Keep the harness alive to retain history across turns.
            this.currentTurnId = null;
            this.emit({ type: "phase", state: "ready" });
        }
    }

    // ── Session-level steering queue (Pi-style) ────────────────────────
    // Tracks pending steering/follow-up messages for UI display.
    // Items are removed when consumed by the loop (detected via
    // message_start events emitted before assistant responses).

    /** Inject guidance into the running turn (drained at the next save point). */
    steer(message: string): void {
        this._steeringMessages.push(message);
        this._emitQueueUpdate();
        this.harness?.steer(message);
    }

    /** Queue a message for after the current turn completes. */
    followUp(message: string): void {
        this._followUpMessages.push(message);
        this._emitQueueUpdate();
        this.harness?.followUp(message);
    }

    /** Queue a message before the next user prompt; survives abort. */
    nextTurn(message: string): void {
        this.harness?.nextTurn(message);
    }

    private _emitQueueUpdate(): void {
        this.emit({
            type: "queue_update",
            steering: [...this._steeringMessages],
            followUp: [...this._followUpMessages],
        });
    }

    /** Remove messages that the loop has consumed (delivered to assistant). */
    private _removeConsumedSteeringMessages(): void {
        if (!this.harness) return;
        const messages = this.harness.messages;
        let changed = false;

        // Remove steering messages whose text appears in the harness messages
        const remaining: string[] = [];
        for (const msg of this._steeringMessages) {
            const consumed = messages.some(
                (m) =>
                    m.role === "user" &&
                    typeof m.content === "string" &&
                    m.content.includes(msg.trim()),
            );
            if (consumed) {
                changed = true;
            } else {
                remaining.push(msg);
            }
        }
        if (changed) {
            this._steeringMessages = remaining;
            this._emitQueueUpdate();
        }
    }

    /** Get current steering messages (read-only). */
    getSteeringMessages(): string[] {
        return [...this._steeringMessages];
    }

    /** Get current follow-up messages (read-only). */
    getFollowUpMessages(): string[] {
        return [...this._followUpMessages];
    }

    /** Clear all pending messages, returns the messages that were cleared. */
    clearQueue(): { steering: string[]; followUp: string[] } {
        const steering = [...this._steeringMessages];
        const followUp = [...this._followUpMessages];
        this._steeringMessages = [];
        this._followUpMessages = [];
        this._emitQueueUpdate();
        return { steering, followUp };
    }

    /** Abort: clear steering/follow-up queues (preserves nextTurn). */
    abort(): void {
        this.harness?.abort();
        this._steeringMessages = [];
        this._followUpMessages = [];
        this._emitQueueUpdate();
    }

    /** Execute a slash command (sends as chat message to the agent). */
    sendSlash(raw: string): void {
        this.sendMessage(raw).catch((err) => this.errorCb?.(err));
    }

    // ── Model cycling ──────────────────────────────────────────────────

    /** Get current model name. */
    getCurrentModel(): string {
        return this.harness?.getModel() ?? this.config.model ?? "";
    }

    /** Get all available models. */
    getModels(): string[] {
        return this.harness?.getModels() ?? (this.config.model ? [this.config.model] : []);
    }

    /** Cycle to the next model. Returns the new model name. */
    cycleModel(direction: "forward" | "backward" = "forward"): string | null {
        return this.harness?.cycleModel(direction) ?? null;
    }

    async getState(): Promise<Record<string, unknown>> {
        await this.loadMcpToolsOnce();
        const state = {
            agent_name: "logician",
            model: this.config.model,
            base_url: this.config.baseUrl,
            tools:
                this.harness?.tools?.list().map((t: Tool) => t.name) ||
                this.defaultTools.map((t) => t.name),
            mcp_servers: this.mcpServerCount,
            mcp_tools: this.defaultTools.filter((tool) =>
                tool.name.startsWith("mcp__"),
            ).length,
            mcp_errors: this.mcpErrors,
            context_tokens: this.contextTokens,
            context_max_tokens: this.contextMaxTokens,
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
        // In a full implementation, this would update the config
        this.config.temperature =
            level === "high" ? 0.8 : level === "medium" ? 0.5 : 0.2;
    }

    reset(): void {
        // Reset tool state and conversation
        void this.fireSessionEnd("reset");
        // Drop the persisted harness so history starts fresh.
        this.harness?.clearHistory();
        this.harness = null;
        this.sessionId = `tui_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
        this.transcriptPath = createHookTranscriptPath(
            this.cwd,
            this.sessionId,
        );
        this.config.hookSessionId = this.sessionId;
        this.config.hookTranscriptPath = this.transcriptPath;
        // Reset skill injection state
        this.skillsContext = null;
        this.skillsInjected = false;
        this.pluginSystemContext = "";
        this.rebuildBaseSystemPrompt();
        this.emit({
            type: "turn_end",
            turn_id: "reset",
            message: "Tool state reset.",
        });
    }

    cancel(): void {
        this.harness?.abort();
    }

    // ── State management ─────────────────────────────────────────────────

    async init(): Promise<Record<string, unknown>> {
        await this.runStartupHooksOnce();
        if (this.mcpEager) {
            await this.loadMcpToolsOnce();
        }
        return {
            agent_name: "logician",
            model: this.config.model,
            base_url: this.config.baseUrl,
            mcp_deferred: !this.mcpLoaded && process.env.LOGICIAN_MCP !== "0",
            tools:
                this.harness?.tools?.list().map((t: Tool) => t.name) ||
                this.defaultTools.map((t) => t.name),
            mcp_servers_loaded: this.mcpServerCount,
            mcp_tools_loaded: this.defaultTools.filter((tool) =>
                tool.name.startsWith("mcp__"),
            ).length,
            mcp_errors: this.mcpErrors,
            context_tokens: this.contextTokens,
            context_max_tokens:
                this.contextMaxTokens || this.config.contextWindowTokens,
            config_path: this.configPath || "",
            hooks_enabled: this.config.runtimeHooksEnabled !== false,
            hook_transcript_path: this.config.hookTranscriptPath || "",
            startup_plugins_loaded: this.startupPluginCount,
            startup_hooks_loaded: this.startupHookResult?.hook_count || 0,
            startup_hook_contexts:
                this.startupHookResult?.additional_contexts || [],
            startup_hook_messages:
                this.startupHookResult?.context_messages || [],
            startup_hook_initial_message:
                this.startupHookResult?.initial_user_message || "",
            startup_hook_errors: this.startupHookResult?.errors || [],
            skills_injected: this.skillsInjected
                ? await this.countInstalledSkills()
                : 0,
            skills_visible: !!this.skillsContext,
        };
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
                const entries = await readdirAsync(
                    path.join(installPath, "skills")
                );
                count += entries.filter(
                    (e) => e !== ".git" && !e.startsWith(".")
                ).length;
            } catch {
                // no skills dir
            }
        }
        return count;
    }

    async stop(): Promise<void> {
        this.cancel();
        await this.fireSessionEnd("shutdown");
        await this.mcpManager.close();
        this.running = false;
    }

    isActive(): boolean {
        return this.running;
    }

    getMessages(): Message[] {
        return this.harness?.messages || [];
    }

    getTools(): ToolRegistry {
        const live = this.harness?.tools;
        if (live) return live;
        const registry = new ToolRegistry({ cwd: this.config.cwd });
        registry.registerMany(this.defaultTools);
        return registry;
    }

    private async loadMcpToolsOnce(): Promise<void> {
        if (this.mcpLoaded || process.env.LOGICIAN_MCP === "0") return;
        this.mcpLoaded = true;
        const result = await this.mcpManager.load(
            this.config.cwd || process.cwd(),
        );
        this.mcpServerCount = result.servers;
        this.mcpErrors = result.errors;
        if (result.tools.length) {
            const existing = new Set(
                this.defaultTools.map((tool) => tool.name),
            );
            const newTools = result.tools.filter(
                (tool) => !existing.has(tool.name),
            );
            this.defaultTools = [...this.defaultTools, ...newTools];
            this.config.tools = this.defaultTools;
            this.rebuildBaseSystemPrompt();
        }
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
        const defaultPrompt = buildDefaultSystemPrompt(
            this.cwd,
            this.defaultTools,
        );
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
        const skillSummaries: string[] = [];

        for (const plugin of plugins) {
            const enabled = plugin.enabled !== false;
            const onDisk = plugin.on_disk !== false;
            const installPath = String(plugin.install_path || "");
            if (!enabled || !onDisk || !installPath) continue;

            const skillsDir = path.join(installPath, "skills");
            try {
                const entries = await readdirAsync(skillsDir);
                const skillDirs = entries.filter(
                    (e) => e !== ".git" && !e.startsWith(".")
                );
                for (const skillDir of skillDirs) {
                    const skillMd = path.join(skillsDir, skillDir, "SKILL.md");
                    try {
                        const content = await readFileAsync(skillMd, "utf8");
                        const firstLines = content.split("\n").slice(0, 40).join("\n");
                        const name = firstLines.includes("name:")
                            ? firstLines.match(/^name:\s*(.+)$/im)?.[1]?.trim()
                            : skillDir;
                        const desc = firstLines.includes("description:")
                            ? firstLines.match(/^description:\s*(.+)$/im)?.[1]?.trim()
                            : "";
                        if (name || desc) {
                            skillSummaries.push(
                                `## Skill: ${name || skillDir} (plugin: ${plugin.plugin_id || "unknown"})\n${desc ? `Description: ${desc}\n` : ""}\n---\n${firstLines}`
                            );
                        }
                    } catch {
                        // Skill has no SKILL.md, skip silently
                    }
                }
            } catch {
                // No skills directory, skip
            }
        }

        if (!skillSummaries.length) return;

        this.skillsContext = `<plugin-skills>\n${skillSummaries.join("\n\n")}\n</plugin-skills>`;
        this.rebuildBaseSystemPrompt();
    }

    private async runStartupHooksOnce(source = "startup"): Promise<void> {
        if (this.startupHooksRan || this.config.runtimeHooksEnabled === false)
            return;
        this.startupHooksRan = true;
        const snapshot = await runPluginBackend("list", []);
        this.startupPluginCount = (snapshot.plugins || []).filter((plugin) => {
            return plugin.enabled !== false && plugin.on_disk !== false;
        }).length;
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
        // Inject skills from plugins as a fallback when hooks produce no context
        await this.injectSkillsFromPlugins();
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
        } catch {
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
                lines.push(
                    "",
                    "No enabled SessionStart hooks matched this source.",
                );
                return lines.join("\n");
            }
            lines.push("", "| Plugin | Matcher | Commands |");
            lines.push("|--------|---------|----------|");
            for (const hook of hooks) {
                const commands = Array.isArray(hook.commands)
                    ? hook.commands
                          .map(
                              (cmd: any) =>
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
                lines.push(
                    "",
                    "Hook context has been applied to future agent turns.",
                );
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

function tableRow(values: string[]): string {
    return `| ${values.map(escapeTable).join(" | ")} |`;
}

function escapeTable(value: string): string {
    return value.replace(/\|/g, "\\|").replace(/\n/g, " ");
}
