// ── Main TUI ──────────────────────────────────────────────────────────────────
// Wires agent-core, transcript, and components together.

import { execSync } from "node:child_process";
import { TUI } from "./tui-core.ts";
import { AgentCoreBridge } from "./agent-bridge.ts";
import { Transcript } from "./transcript.ts";
import { TranscriptDisplay } from "./components/transcript-display.ts";
import { InputBar } from "./components/input-bar.ts";
import { StatusBar } from "./components/status-bar.ts";
import { TodoBar } from "./components/todo-bar.ts";
import { ThinkingPanel } from "./components/thinking-panel.ts";
import { SlashPopup } from "./components/slash-popup.ts";
import {
    PluginManagerOverlay,
    type PluginManagerAction,
} from "./components/plugin-manager.ts";

import type { ParsedBridgeEvent } from "./events.ts";
import { UndoStack } from "./undo-stack.ts";
import { KillRing } from "./kill-ring.ts";
import { createSlashCommands, type SlashCommandDef } from "./slash-commands.ts";

// ── Main TUI ─────────────────────────────────────────────────────────────────

export class LogicianTUI {
    private tui: TUI;
    private bridge: AgentCoreBridge;
    private transcript: Transcript;
    private statusPanel: StatusBar;
    private todoBar: TodoBar;
    private thinkingPanel: ThinkingPanel;
    private inputBar: InputBar;
    private slashPopup: SlashPopup;
    private pluginManager: PluginManagerOverlay;
    private transcriptDisplay: TranscriptDisplay;
    private killRing: KillRing;
    private undoStack: UndoStack<{ value: string; cursor: number }>;
    private streaming = false;

    // Feature flags
    private traceOn = false;
    private thinkingLevel = "medium";
    private cacheEnabled = true;
    private thinkingDisplayMode: "collapsed" | "summary" | "expanded" =
        "expanded";

    constructor() {
        this.bridge = new AgentCoreBridge({
            baseUrl: process.env.LOGICIAN_LLM_URL || "http://127.0.0.1:8080",
            model: process.env.LOGICIAN_MODEL || "",
            systemPrompt: process.env.LOGICIAN_SYSTEM_PROMPT,
            cwd: process.cwd(),
        });
        this.transcript = new Transcript();
        this.statusPanel = new StatusBar();
        this.todoBar = new TodoBar();
        this.thinkingPanel = new ThinkingPanel();
        this.inputBar = new InputBar();
        this.slashPopup = new SlashPopup();
        this.pluginManager = new PluginManagerOverlay();
        this.transcriptDisplay = new TranscriptDisplay({
            thinkingMode: this.thinkingDisplayMode,
        });
        this.killRing = new KillRing();
        this.undoStack = new UndoStack();

        // Create the TUI with hardware cursor support
        this.tui = new TUI(process.stdout, true);
        this.statusPanel.setOnInvalidate(() => this.tui.requestRender());
        this.todoBar.setOnInvalidate(() => this.tui.requestRender());

        // Wire up dependencies
        this.inputBar.setKillRing(this.killRing);
        this.inputBar.setUndoStack(this.undoStack);

        // Setup bridge event handling
        this.setupBridge();

        // Setup transcript change handling
        this.setupTranscript();

        // Wire up scrollable component
        this.tui.setScrollableComponent(this.transcriptDisplay);

        // ── Async helpers (must be defined before setupInputHandler) ─────────

        const setStatusPhase = (phase: string) => {
            this.statusPanel.update({ phase });
        };

        const handleStatus = async () => {
            try {
                const state = await this.bridge.getState();
                const lines = [
                    `Agent: ${state.agent_name || "unknown"}`,
                    `Model: ${state.model || "unknown"}`,
                    `Tools: ${(state.tools as string[])?.length || 0} loaded`,
                    `MCP: ${state.mcp_servers || 0} server(s), ${state.mcp_tools || 0} tool(s)`,
                    `Context: ${formatContextSize(
                        Number(state.context_tokens || 0),
                        Number(state.context_max_tokens || 0) || undefined,
                    )}`,
                    `Hooks: ${state.hooks_enabled === false ? "disabled" : "enabled"}`,
                    `Hook transcript: ${state.hook_transcript_path || "-"}`,
                    `Connected: ${state.connected !== false}`,
                ];
                const mcpErrors = Array.isArray(state.mcp_errors)
                    ? state.mcp_errors
                          .map((item) => String(item || "").trim())
                          .filter(Boolean)
                    : [];
                if (mcpErrors.length) {
                    lines.push(
                        "",
                        "MCP errors:",
                        ...mcpErrors.map((err) => `- ${err}`),
                    );
                }
                this.transcript.addSystemMessage(lines.join("\n"));
                this.transcriptDisplay.setTurns(this.transcript.getTurns());
                this.tui.requestRender();
            } catch (e: unknown) {
                this.transcript.addSystemMessage(
                    `Status error: ${e instanceof Error ? e.message : String(e)}`,
                );
                this.tui.requestRender();
            }
        };

        const handlePlugins = async (args: string) => {
            try {
                const normalized = args.trim().toLowerCase();
                if (!normalized || normalized === "list") {
                    await this.openPluginManager();
                    return;
                }
                const result = await this.bridge.runPluginCommand(args);
                this.transcript.addSystemMessage(result);
                this.transcriptDisplay.setTurns(this.transcript.getTurns());
                this.tui.requestRender();
            } catch (e: unknown) {
                this.transcript.addSystemMessage(
                    `Plugins error: ${e instanceof Error ? e.message : String(e)}`,
                );
                this.tui.requestRender();
            }
        };

        // Setup keyboard shortcuts
        this.setupInputHandler(handleStatus, handlePlugins);

        // Focus input bar by default
        this.tui.setFocus(this.inputBar);

        // Initial state
        this.statusPanel.update({
            thinkingLevel: this.thinkingLevel,
            cacheEnabled: this.cacheEnabled,
            phase: "ready",
            model: process.env.LOGICIAN_MODEL || "local",
            cwd: process.cwd(),
            branch: this.getGitBranch(),
            contextTokens: 0,
            contextMaxTokens:
                envNumber("LOGICIAN_CONTEXT_WINDOW") ||
                envNumber("LOGICIAN_CTX_SIZE"),
        });

        // Setup slash commands
        const localHandlers: Record<string, (...args: any[]) => any> = {
            setThinking: (level: string) => {
                this.thinkingLevel = level;
                this.bridge.setThinkingLevel(level);
                this.statusPanel.update({ thinkingLevel: level });
                setStatusPhase("ready");
            },
            setCache: (enabled: boolean) => {
                this.cacheEnabled = enabled;
                this.transcript.setCacheEnabled(enabled);
                this.bridge.sendSlash(
                    `/cache ${enabled ? "enable" : "disable"}`,
                );
                this.statusPanel.update({ cacheEnabled: enabled });
                setStatusPhase("ready");
            },
            setThinkingMode: (mode: string) => {
                this.thinkingDisplayMode =
                    mode as typeof this.thinkingDisplayMode;
                this.transcript.setThinkingDisplayMode(mode as any);
                setStatusPhase("ready");
            },
            cycleThinking: () => {
                this.transcript.cycleThinkingDisplayMode();
                setStatusPhase("ready");
            },
            setTrace: (on: boolean) => {
                this.traceOn = on;
                setStatusPhase("ready");
            },
            clear: () => {
                this.transcript.clear();
                this.thinkingPanel.clear();
                setStatusPhase("ready");
            },
        };

        const slashCommands = createSlashCommands(this.bridge, localHandlers);
        this.slashPopup.setCommands(slashCommands);

        // Wire up slash popup submit to handle quit dispatch
        this.slashPopup.setOnSubmit(async (result, dispatch, command) => {
            if (dispatch === "quit") {
                await this.stop();
                process.exit(0);
            }
            if (result) {
                this.transcript.addSystemMessage(String(result));
            }
            // Add slash command as user message to transcript
            if (command && command.trim()) {
                this.transcript.addTurn(command.trim());
                const cmdName =
                    command.trim().split(/\s+/)[0]?.toLowerCase() || "";
                const args = command.trim().split(/\s+/).slice(1).join(" ");
                const allCmds = this.slashPopup[
                    "commands"
                ] as SlashCommandDef[];
                const match = allCmds?.find(
                    (c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
                );
                if (match && match.command === "/plugins") {
                    handlePlugins(args);
                }
                if (match && match.dispatch === "bridge") {
                    this.bridge.sendSlash(command.trim());
                }
                if (match && match.dispatch === "state") {
                    handleStatus();
                }
            }
            this.transcriptDisplay.setTurns(this.transcript.getTurns());
            this.tui.requestRender();
        });
    }

    // ── Bridge setup ─────────────────────────────────────────────────────────

    private setupBridge(): void {
        const eventHandler = (event: ParsedBridgeEvent): void => {
            this.handleEvent(event);
        };

        this.bridge.on(eventHandler);
        this.bridge.onError((err) => {
            console.error(`Bridge error: ${err.message}`);
        });

        // Initialize bridge
        this.bridge
            .init()
            .then((state) => {
                this.statusPanel.update({
                    contextTokens: Number(state.context_tokens || 0),
                    contextMaxTokens:
                        Number(state.context_max_tokens || 0) || undefined,
                });
                const message = this.formatStartupMessage(state);
                if (message) {
                    this.transcript.addSystemMessage(message);
                    this.transcriptDisplay.setTurns(this.transcript.getTurns());
                    this.tui.requestRender();
                }
            })
            .catch((err) => {
                console.error(`Bridge init failed: ${err.message}`);
            });
    }

    private handleEvent(event: ParsedBridgeEvent): void {
        // Update transcript state
        this.transcript.handleEvent(event);

        switch (event.type) {
            case "todos":
                this.todoBar.setTodos(event.todos);
                this.tui.requestRender();
                break;
            case "token":
                if (!this.streaming) {
                    this.streaming = true;
                    this.statusPanel.update({ phase: "streaming" });
                    this.statusPanel.startAnimation();
                }
                break;
            case "tool_start":
            case "tool_execution_start":
                this.statusPanel.update({ phase: "tool" });
                this.statusPanel.startAnimation();
                break;
            case "turn_end":
                this.streaming = false;
                this.statusPanel.stopAnimation();
                this.statusPanel.update({
                    phase: "ready",
                    turnCount: this.transcript.getTurns().length,
                    messageCount: this.transcript.getMessageCount(),
                });
                break;
            case "turn_start":
                this.statusPanel.update({ phase: "thinking" });
                this.statusPanel.startAnimation();
                break;
            case "phase":
                this.statusPanel.update({ phase: event.state });
                if (event.state !== "ready") {
                    this.statusPanel.startAnimation();
                } else {
                    this.streaming = false;
                    this.statusPanel.stopAnimation();
                    this.statusPanel.update({
                        turnCount: this.transcript.getTurns().length,
                        messageCount: this.transcript.getMessageCount(),
                    });
                }
                break;
            case "context_update":
                this.statusPanel.update({
                    contextTokens: Number(event.tokens || 0),
                    contextMaxTokens:
                        Number(event.max_tokens || 0) || undefined,
                    contextCompacted: event.compacted === true,
                });
                break;
            case "compaction":
                this.transcript.addSystemMessage(
                    `Context compacted (${formatContextSize(
                        Number(event.tokens_before || 0),
                    )} -> ${formatContextSize(Number(event.tokens_after || 0))}).`,
                );
                this.transcriptDisplay.setTurns(this.transcript.getTurns());
                this.statusPanel.update({
                    phase: "compacted",
                    contextTokens: Number(event.tokens_after || 0),
                    contextCompacted: true,
                });
                break;
        }

        this.tui.requestRender();
    }

    private formatStartupMessage(state: Record<string, unknown>): string {
        const pluginCount = Number(state.startup_plugins_loaded || 0);
        const hookCount = Number(state.startup_hooks_loaded || 0);
        const mcpServerCount = Number(state.mcp_servers_loaded || 0);
        const mcpToolCount = Number(state.mcp_tools_loaded || 0);
        const contexts = Array.isArray(state.startup_hook_contexts)
            ? state.startup_hook_contexts
                  .map((item) => String(item || "").trim())
                  .filter(Boolean)
            : [];
        const hookMessages = Array.isArray(state.startup_hook_messages)
            ? state.startup_hook_messages
                  .map(normalizeStartupHookMessage)
                  .filter((item) => item.content)
            : [];
        const initialMessage = String(
            state.startup_hook_initial_message || "",
        ).trim();
        const errors = Array.isArray(state.startup_hook_errors)
            ? state.startup_hook_errors
                  .map((item) => String(item || "").trim())
                  .filter(Boolean)
            : [];
        const mcpErrors = Array.isArray(state.mcp_errors)
            ? state.mcp_errors
                  .map((item) => String(item || "").trim())
                  .filter(Boolean)
            : [];

        const lines = [
            `Plugins loaded: ${pluginCount}`,
            `Startup hooks: ${hookCount}`,
            state.mcp_deferred
                ? "MCP: deferred until first agent turn or /status"
                : `MCP: ${mcpServerCount} server(s), ${mcpToolCount} tool(s)`,
        ];

        if (initialMessage) {
            lines.push("", "## Startup message", initialMessage);
        }

        if (contexts.length) {
            lines.push("", "## Plugin startup messages");
            if (hookMessages.length) {
                hookMessages.forEach((message) => {
                    lines.push("", `### ${message.title}`, message.content);
                });
            } else {
                contexts.forEach((context, idx) => {
                    lines.push("", `### Startup hook ${idx + 1}`, context);
                });
            }
        }

        if (errors.length) {
            lines.push(
                "",
                "## Startup hook errors",
                ...errors.map((err) => `- ${err}`),
            );
        }

        if (mcpErrors.length) {
            lines.push(
                "",
                "## MCP errors",
                ...mcpErrors.map((err) => `- ${err}`),
            );
        }

        return lines.join("\n");
    }

    // ── Transcript setup ─────────────────────────────────────────────────────

    private setupTranscript(): void {
        this.transcript.onChange(() => {
            this.transcriptDisplay.setTurns(this.transcript.getTurns());
            this.transcriptDisplay.setThinkingMode(
                this.transcript.getThinkingDisplayMode(),
            );
            // Auto-scroll to bottom only when already at bottom
            if (this.tui.isAtBottom) {
                this.transcriptDisplay.scrollToBottom();
                this.tui.scrollToBottom();
            }
            this.tui.requestRender();
        });
    }

    // ── Input handling ─────────────────────────────────────────────────────

    private setupInputHandler(
        handleStatus: () => Promise<void>,
        handlePlugins: (args: string) => Promise<void>,
    ): void {
        // Global input listener
        this.tui.addInputListener((data: string) => {
            if (this.pluginManager.isVisibleOverlay()) {
                const action = this.pluginManager.handleInput(data);
                if (action) {
                    this.handlePluginManagerAction(action);
                }
                this.tui.requestRender();
                return { consume: true };
            }

            // Inline slash autocomplete: while the popup is showing matches, the
            // input bar keeps focus and ordinary typing flows through to it. We only
            // intercept the navigation/accept keys here.
            if (this.slashPopup.isVisibleOverlay()) {
                // Up / Down — move highlight
                if (data === "\x1b[A" || data === "\x1bOA") {
                    this.slashPopup.moveSelection(-1);
                    this.tui.requestRender();
                    return { consume: true };
                }
                if (data === "\x1b[B" || data === "\x1bOB") {
                    this.slashPopup.moveSelection(1);
                    this.tui.requestRender();
                    return { consume: true };
                }
                // Tab — complete input to the highlighted command
                if (data === "\t") {
                    const cmd = this.slashPopup.currentCommand();
                    if (cmd) {
                        this.inputBar.valueText = cmd + " ";
                        this.tui.requestRender();
                    }
                    return { consume: true };
                }
                // Escape — dismiss the menu but keep what was typed
                if (data === "\x1b") {
                    this.slashPopup.hide();
                    this.tui.requestRender();
                    return { consume: true };
                }
                // Enter — accept highlighted command (submit it directly)
                if (data === "\r" || data === "\n") {
                    const cmd = this.slashPopup.currentCommand();
                    if (cmd && this.inputBar.valueText.trim() !== cmd) {
                        // If the typed text isn't already an exact command, accept the
                        // highlighted one (carrying over any args the user typed).
                        const typedArgs = this.inputBar.valueText.replace(
                            /^\/\S*\s*/,
                            "",
                        );
                        this.inputBar.valueText = typedArgs
                            ? `${cmd} ${typedArgs}`
                            : cmd;
                    }
                    this.slashPopup.hide();
                    // Fall through to the input bar so it submits the value.
                    return { consume: false };
                }
                // Everything else (typing, backspace, etc.) goes to the input bar; the
                // onChange hook re-syncs the popup query afterwards.
            }

            // Ctrl+L — clear screen
            if (data === "\x0c") {
                this.tui.requestRender(true);
                return { consume: true };
            }

            // Ctrl+O — expand/collapse tool execution details
            if (data === "\x0f") {
                const expanded = this.transcriptDisplay.toggleToolsExpanded();
                this.statusPanel.update({
                    phase: expanded ? "tools expanded" : "tools collapsed",
                });
                this.tui.requestRender();
                return { consume: true };
            }

            // Ctrl+Shift+T — cycle thinking display mode
            if (data === "\x14") {
                this.transcript.cycleThinkingDisplayMode();
                this.transcriptDisplay.setThinkingMode(
                    this.transcript.getThinkingDisplayMode(),
                );
                this.transcriptDisplay.setTurns(this.transcript.getTurns());
                this.tui.requestRender();
                return { consume: true };
            }

            // Ctrl+Backspace in input bar is handled by InputBar directly
            return { consume: false };
        });

        // Live slash autocomplete: show/hide + filter the popup as the input text
        // changes. The popup only appears while the line begins with "/" and has no
        // space yet (i.e. the user is still picking a command, not typing args).
        this.inputBar.onChange = (text: string) => {
            const isCommandPrefix = text.startsWith("/") && !text.includes(" ");
            if (isCommandPrefix) {
                this.slashPopup.setQuery(text);
                if (this.slashPopup.hasMatches()) {
                    if (!this.slashPopup.isVisibleOverlay())
                        this.slashPopup.show();
                } else {
                    this.slashPopup.hide();
                }
            } else if (this.slashPopup.isVisibleOverlay()) {
                this.slashPopup.hide();
            }
            this.tui.requestRender();
        };

        // Input bar handler
        this.inputBar.onSubmit = (text: string) => {
            // Always push to history (both slash and regular messages)
            this.inputBar.pushHistory(text);

            // Check for slash commands
            if (text.startsWith("/")) {
                const parts = text.trim().split(/\s+/);
                const cmdName = parts[0].toLowerCase();
                const args = parts.slice(1).join(" ");
                const allCmds = this.slashPopup[
                    "commands"
                ] as SlashCommandDef[];
                const match = allCmds?.find(
                    (c: SlashCommandDef) => c.command.toLowerCase() === cmdName,
                );

                if (match) {
                    // Add the slash command as a user message to transcript
                    this.transcript.addTurn(text.trim());

                    if (match.dispatch === "quit") {
                        this.transcriptDisplay.setTurns(
                            this.transcript.getTurns(),
                        );
                        this.tui.requestRender();
                        void this.stop().then(() => process.exit(0));
                        return;
                    }

                    if (match.handler) {
                        const result = match.handler(args);
                        if (result) {
                            this.transcript.addSystemMessage(String(result));
                        }
                    }
                    if (match.bridgeHandler) {
                        match.bridgeHandler(args);
                    }
                    if (match.command === "/plugins") {
                        handlePlugins(args);
                        this.transcriptDisplay.setTurns(
                            this.transcript.getTurns(),
                        );
                        this.tui.requestRender();
                        return;
                    }
                    if (match.dispatch === "bridge") {
                        this.bridge.sendSlash(text.trim());
                    }
                    if (match.dispatch === "state") {
                        handleStatus();
                    }
                    if (
                        match.dispatch === "local" &&
                        !match.handler &&
                        !match.bridgeHandler &&
                        match.command !== "/plugins"
                    ) {
                        // Local command without handler — just show result
                        this.transcript.addSystemMessage(
                            `[${match.command}] executed`,
                        );
                    }
                    this.transcriptDisplay.setTurns(this.transcript.getTurns());
                    this.tui.requestRender();
                    return;
                }

                // Unknown command — send to bridge as slash
                this.transcript.addTurn(text.trim());
                this.bridge.sendSlash(text.trim());
                this.transcriptDisplay.setTurns(this.transcript.getTurns());
                this.tui.requestRender();
                return;
            }

            this.transcript.addTurn(text);
            this.bridge
                .sendMessage(text)
                .catch((err) => this.bridge.onError?.(err));
            this.statusPanel.update({ phase: "streaming" });
            this.statusPanel.startAnimation();
        };

        this.inputBar.onCancel = () => {
            this.bridge.cancel();
            this.statusPanel.update({ phase: "ready" });
        };
    }

    // ── Layout ─────────────────────────────────────────────────────────────

    private buildLayout(): void {
        // Fixed layout: transcript (scrollable, top) + separator + input bar (fixed) + status bar (fixed, bottom)
        this.tui.setInputBarComponent(this.inputBar);
        this.tui.setScrollableComponent(this.transcriptDisplay);
        this.tui.setFixedBottomComponent(this.statusPanel);
        this.tui.setFixedAboveInputComponent(this.todoBar);

        // Slash popup as overlay anchored to the bottom of the transcript area, so
        // the suggestion list sits directly above the input bar like an inline
        // autocomplete menu.
        this.tui.showOverlay(this.slashPopup, {
            anchor: "bottom",
            align: "left",
            maxHeight: 12,
        });
        this.tui.showOverlay(this.pluginManager, {
            anchor: "center",
            maxHeight: 18,
        });
    }

    private async openPluginManager(): Promise<void> {
        this.statusPanel.update({ phase: "plugins" });
        try {
            const snapshot = await this.bridge.getPluginSnapshot();
            this.pluginManager.setSnapshot({
                pluginsDir: String(snapshot.plugins_dir || ""),
                plugins: snapshot.plugins || [],
                sessionStartHooks: snapshot.session_start_hooks || {},
            });
            this.pluginManager.setMessage(
                "Space toggles enabled state in the Claude plugin registry.",
            );
            this.pluginManager.show();
        } catch (e: unknown) {
            this.transcript.addSystemMessage(
                `Plugins error: ${e instanceof Error ? e.message : String(e)}`,
            );
        } finally {
            this.statusPanel.update({ phase: "ready" });
            this.transcriptDisplay.setTurns(this.transcript.getTurns());
            this.tui.requestRender();
        }
    }

    private handlePluginManagerAction(action: PluginManagerAction): void {
        if (action.type === "close") {
            this.pluginManager.hide();
            return;
        }
        if (action.type === "refresh") {
            this.openPluginManager();
            return;
        }

        const plugin = action.plugin;
        const nextEnabled = !plugin.enabled;
        this.pluginManager.setBusy(plugin.pluginId);
        this.pluginManager.setMessage(
            `${nextEnabled ? "Enabling" : "Disabling"} ${plugin.pluginId}...`,
        );
        this.tui.requestRender();
        this.bridge
            .setPluginEnabled(plugin.pluginId, nextEnabled)
            .then(async (result) => {
                this.pluginManager.setMessage(
                    String(result.message || `${plugin.pluginId} updated.`),
                );
                const snapshot = await this.bridge.getPluginSnapshot();
                this.pluginManager.setSnapshot({
                    pluginsDir: String(snapshot.plugins_dir || ""),
                    plugins: snapshot.plugins || [],
                    sessionStartHooks: snapshot.session_start_hooks || {},
                });
            })
            .catch((e: unknown) => {
                this.pluginManager.setMessage(
                    `Plugin update failed: ${e instanceof Error ? e.message : String(e)}`,
                );
            })
            .finally(() => {
                this.pluginManager.setBusy(null);
                this.statusPanel.update({ phase: "ready" });
                this.tui.requestRender();
            });
    }

    private getGitBranch(): string {
        try {
            return execSync("git branch --show-current", {
                cwd: process.cwd(),
                encoding: "utf8",
                stdio: ["ignore", "pipe", "ignore"],
            }).trim();
        } catch {
            return "";
        }
    }

    // ── Start ──────────────────────────────────────────────────────────────

    start(): void {
        this.buildLayout();
        this.tui.enableMouse();
        this.tui.start();
    }

    // ── Public accessors for external integration ──────────────────────────

    getSlashPopup(): SlashPopup {
        return this.slashPopup;
    }

    getInputBar(): InputBar {
        return this.inputBar;
    }

    async stop(): Promise<void> {
        this.tui.stop();
        await this.bridge.stop();
    }
}

function normalizeStartupHookMessage(item: unknown): {
    title: string;
    content: string;
} {
    if (!item || typeof item !== "object") {
        return { title: "Startup hook", content: String(item || "").trim() };
    }
    const raw = item as Record<string, unknown>;
    const pluginName = String(raw.plugin_name || "").trim();
    const pluginId = String(raw.plugin_id || "").trim();
    const matcher = String(raw.matcher || "").trim();
    const label = pluginName || pluginId || "Startup hook";
    const suffix =
        pluginName && pluginId && pluginName !== pluginId
            ? ` (${pluginId})`
            : "";
    const matcherText = matcher && matcher !== "*" ? ` · ${matcher}` : "";
    return {
        title: `${label}${suffix}${matcherText}`,
        content: String(raw.content || "").trim(),
    };
}

function formatContextSize(tokens: number, maxTokens?: number): string {
    const current = formatTokenCount(Math.max(0, Math.round(tokens || 0)));
    if (!maxTokens || maxTokens <= 0) return current;
    return `${current}/${formatTokenCount(Math.round(maxTokens))}`;
}

function formatTokenCount(tokens: number): string {
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}m`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
    return String(tokens);
}

function envNumber(name: string): number | undefined {
    const raw = process.env[name];
    if (!raw) return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
}
