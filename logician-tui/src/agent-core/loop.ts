// ── Agent loop ─────────────────────────────────────────────────────────────────────
// Main ReAct-style loop for the agent. Mirrors Python AgentLoop.

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type {
    AgentConfig,
    AgentEvent,
    AgentLoopHooks,
    EventHandler,
    Message,
    ToolCall,
} from "./types.ts";
import type { LLMBackend } from "./backend.ts";
import { EventEmitter, createEventEmitter } from "./events.ts";
import {
    createUserMessage,
    createSystemMessage,
    createToolResultMessage,
    createAssistantMessage,
    compactMessagesForContext,
    convertToChatFormat,
    estimateChatPayloadTokens,
} from "./messages.ts";
import { parseToolCalls } from "./parser.ts";
import { ToolRegistry } from "./tools/registry.ts";
import { createDefaultTools } from "./default-tools.ts";
import { runHookEvent } from "./plugins.ts";
import { composeHooks, buildBuiltinHooks } from "./builtin-hooks.ts";

// Cap on pi-style continuations within one run, to bound runaway loops when a
// continuation hook keeps resuming the agent.
const DEFAULT_MAX_CONTINUATIONS = 12;

export interface AgentLoopOptions {
    config: AgentConfig;
    backend: LLMBackend;
    cwd?: string;
    maxIterations?: number;
    signal?: AbortSignal;
    // Prior conversation to continue from. When provided, the new user message
    // is appended to this history instead of starting a fresh transcript, so
    // follow-ups like "continue" retain context across turns.
    initialMessages?: Message[];
}

export class AgentLoop {
    private config: AgentConfig;
    private backend: LLMBackend;
    private toolRegistry: ToolRegistry;
    private emitter: EventEmitter;
    private _messages: Message[];
    private cwd: string;
    private maxIterations: number;
    private iterationCount: number = 0;
    private onEvent?: EventHandler;
    private signal?: AbortSignal;
    private hooks: AgentLoopHooks;
    private initialMessages?: Message[];
    private continuationCount = 0;
    private maxContinuations: number;

    constructor(options: AgentLoopOptions) {
        this.config = options.config;
        this.backend = options.backend;
        this.cwd = options.cwd || process.cwd();
        this.maxIterations = options.maxIterations || 30;
        this.signal = options.signal;
        this.initialMessages = options.initialMessages;
        this.maxContinuations =
            this.config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
        this.iterationCount = 0;
        this._messages = [];
        this.hooks = this.config.hooks || {};
        this.emitter = createEventEmitter();

        // Set up tool registry
        this.toolRegistry = new ToolRegistry({ cwd: this.cwd });
        this.toolRegistry.registerMany(
            this.config.tools?.length
                ? this.config.tools
                : createDefaultTools(),
        );

        // Set up event handler
        this.onEvent = this.config.onEvent;
        this.config.onEvent = (event: AgentEvent) => {
            this.emitter.emit(event);
            this.onEvent?.(event);
        };
    }

    get events(): EventEmitter {
        return this.emitter;
    }

    get tools(): ToolRegistry {
        return this.toolRegistry;
    }

    get messages(): Message[] {
        return this._messages;
    }

    async run(userMessage: string): Promise<Message[]> {
        // Compose built-in safeguard hooks (guards, budget stop, proactive
        // compaction) with any user-supplied hooks. Built-in state (failure
        // counts, budget tracker, compaction cooldown) is per-run, so build
        // here rather than in the constructor.
        const builtin = buildBuiltinHooks({
            config: this.config,
            contextWindowTokens: () => this.contextWindowTokens(),
            toolDefs: () => this.toolRegistry.toToolDefinitions(),
        });
        this.hooks = composeHooks(builtin, this.config.hooks || {});

        // Initialize with system prompt
        const systemPrompt =
            this.config.systemPrompt || "You are a helpful assistant.";
        const sessionId = this.config.hookSessionId || `tui_${Date.now()}`;
        const transcriptPath = this.config.hookTranscriptPath || "";
        const hookBasePayload = {
            session_id: sessionId,
            transcript_path: transcriptPath,
            cwd: this.cwd,
        };
        const hookMessages = await this.userPromptHookMessages(
            userMessage,
            hookBasePayload,
        );
        this.appendTranscript(transcriptPath, {
            type: "user",
            timestamp: new Date().toISOString(),
            message: { role: "user", content: userMessage },
        });
        if (this.initialMessages && this.initialMessages.length) {
            // Continue an existing conversation: keep prior history, refresh the
            // system prompt to the current one, append hook context + new turn.
            const priorNonSystem = this.initialMessages.filter(
                (m) => m.role !== "system",
            );
            this._messages = [
                createSystemMessage(systemPrompt),
                ...priorNonSystem,
                ...hookMessages,
                createUserMessage(userMessage),
            ];
        } else {
            this._messages = [
                createSystemMessage(systemPrompt),
                ...hookMessages,
                createUserMessage(userMessage),
            ];
        }
        this.iterationCount = 0;
        this.continuationCount = 0;

        this.emitEvent({ type: "agent_start" });
        this.emitEvent({ type: "phase", phase: "idle" });

        while (this.iterationCount < this.maxIterations) {
            if (this.signal?.aborted) {
                this.emitEvent({ type: "error", message: "Operation aborted" });
                break;
            }
            this.iterationCount++;

            // Start turn
            const turnId = `turn_${this.iterationCount}`;
            this.emitEvent({ type: "turn_start", turnId });
            this.emitEvent({ type: "phase", phase: "thinking" });

            // Estimate context before the model call so the status bar stays live.
            const toolDefs = this.toolRegistry.toToolDefinitions();
            this.emitContextUpdate(toolDefs);

            // Get LLM response
            let assistantContent = "";
            let assistantToolCalls: ToolCall[] = [];

            for (let attempt = 0; attempt < 2; attempt++) {
                try {
                    const activeToolDefs =
                        this.toolRegistry.toToolDefinitions();
                    const activeChatMessages = convertToChatFormat(
                        this._messages,
                    );
                    this.emitContextUpdate(activeToolDefs);
                    const response = await this.backend.generate(
                        activeChatMessages,
                        activeToolDefs.length > 0 ? activeToolDefs : undefined,
                        this.config.temperature || 0.5,
                        this.config.maxTokens || 4096,
                        this.signal,
                        (delta: string) => {
                            assistantContent += delta;
                            this.emitEvent({
                                type: "message_delta",
                                turnId,
                                delta,
                            });
                        },
                        (delta: string) => {
                            this.emitEvent({ type: "thinking_delta", delta });
                        },
                    );

                    assistantContent = response.content || "";
                    assistantToolCalls = response.toolCalls;
                    if (assistantToolCalls.length === 0 && assistantContent) {
                        assistantToolCalls = parseToolCalls(assistantContent);
                    }
                    break;
                } catch (e: unknown) {
                    const error = e as Error;
                    if (attempt === 0 && isContextFullError(error)) {
                        const before = estimateChatPayloadTokens(
                            this._messages,
                            toolDefs,
                        );
                        const compacted = compactMessagesForContext(
                            this._messages,
                            {
                                targetTokens: this.contextWindowTokens()
                                    ? Math.floor(
                                          this.contextWindowTokens()! * 0.65,
                                      )
                                    : undefined,
                            },
                        );
                        if (compacted.changed) {
                            this._messages = compacted.messages;
                            const after = estimateChatPayloadTokens(
                                this._messages,
                                toolDefs,
                            );
                            this.emitEvent({
                                type: "compaction",
                                reason: "context_full",
                                tokensBefore: before,
                                tokensAfter: after,
                            });
                            this.emitContextUpdate(toolDefs, true);
                            continue;
                        }
                    }
                    this.emitEvent({ type: "error", message: error.message });
                    assistantContent = "";
                    assistantToolCalls = [];
                    break;
                }
            }

            if (!assistantContent && assistantToolCalls.length === 0) break;

            // Add assistant message
            this._messages.push(
                createAssistantMessage(assistantContent, assistantToolCalls),
            );
            this.appendTranscript(transcriptPath, {
                type: "assistant",
                timestamp: new Date().toISOString(),
                message: {
                    role: "assistant",
                    content: assistantContent
                        ? [{ type: "text", text: assistantContent }]
                        : [],
                    tool_calls: assistantToolCalls.map((toolCall) => ({
                        id: toolCall.id,
                        name: toolCall.name,
                        input: parseToolInput(toolCall.arguments),
                    })),
                },
            });
            this.emitEvent({ type: "message_end", turnId });

            // Check if we have tool calls
            if (assistantToolCalls.length > 0) {
                this.emitEvent({ type: "phase", phase: "tool" });

                for (const tc of assistantToolCalls) {
                    if (this.signal?.aborted) {
                        this.emitEvent({
                            type: "error",
                            message: "Operation aborted",
                        });
                        break;
                    }
                    // Emit tool call start
                    this.emitEvent({
                        type: "tool_call_start",
                        toolName: tc.name,
                        toolCallId: tc.id,
                        args: tc.arguments,
                    });

                    let toolInput = parseToolInput(tc.arguments);
                    let activeToolCall = tc;

                    // beforeToolCall contract hook: may rewrite args or
                    // short-circuit execution entirely.
                    let result: string;
                    let isError: boolean;
                    const before = await this.runBeforeToolCall(tc, toolInput);
                    if (before?.content !== undefined) {
                        // Short-circuit: tool is NOT executed.
                        result = before.content;
                        isError = before.isError ?? false;
                        this.emitEvent({
                            type: "tool_call_end",
                            toolName: tc.name,
                            toolCallId: tc.id,
                            result,
                            isError,
                        });
                        await this.recordToolResult(
                            transcriptPath,
                            hookBasePayload,
                            activeToolCall,
                            toolInput,
                            result,
                            isError,
                        );
                        continue;
                    }
                    if (before?.args !== undefined) {
                        toolInput = before.args;
                        activeToolCall = {
                            ...tc,
                            arguments: JSON.stringify(before.args),
                        };
                    }

                    await this.runHookSafely("PreToolUse", {
                        ...hookBasePayload,
                        matcher_value: hookMatcherValue(activeToolCall.name),
                        tool_name: activeToolCall.name,
                        tool_input: toolInput,
                    });

                    // Execute tool
                    result = await this.toolRegistry.execute(activeToolCall, {
                        signal: this.signal,
                        onUpdate: (partialResult) => {
                            this.emitEvent({
                                type: "tool_call_update",
                                toolName: activeToolCall.name,
                                toolCallId: activeToolCall.id,
                                partialResult,
                            });
                        },
                    });
                    isError = result.startsWith("Error:");

                    // afterToolCall contract hook: may rewrite the result.
                    const after = await this.runAfterToolCall(
                        activeToolCall,
                        toolInput,
                        result,
                        isError,
                    );
                    if (after) {
                        if (after.content !== undefined) result = after.content;
                        if (after.isError !== undefined) isError = after.isError;
                    }

                    // Emit tool call end
                    this.emitEvent({
                        type: "tool_call_end",
                        toolName: activeToolCall.name,
                        toolCallId: activeToolCall.id,
                        result,
                        isError,
                    });

                    await this.recordToolResult(
                        transcriptPath,
                        hookBasePayload,
                        activeToolCall,
                        toolInput,
                        result,
                        isError,
                    );
                }

                if (this.signal?.aborted) break;
            }

            // prepareNextTurn / shouldStopAfterTurn contract hooks.
            const hadToolCalls = assistantToolCalls.length > 0;
            await this.runPrepareNextTurn(hadToolCalls);
            if (await this.runShouldStopAfterTurn(hadToolCalls)) break;

            // Continue loop with tool results.
            if (hadToolCalls) continue;

            // No tool calls: the turn would end here. Pi-style continuation —
            // let a hook resume the agent (e.g. todos still pending) by
            // injecting a user message, capped to avoid runaway loops.
            if (this.continuationCount < this.maxContinuations) {
                const cont = await this.runContinueAfterTurn(assistantContent);
                if (cont) {
                    this.continuationCount++;
                    this._messages.push(createUserMessage(cont.message));
                    this.appendTranscript(transcriptPath, {
                        type: "user",
                        timestamp: new Date().toISOString(),
                        message: { role: "user", content: cont.message },
                    });
                    continue;
                }
            }
            break;
        }

        this.emitEvent({
            type: "turn_end",
            turnId: `turn_${this.iterationCount}`,
        });
        await this.runHookSafely("Stop", {
            ...hookBasePayload,
            stop_hook_active: false,
        });
        this.emitEvent({ type: "phase", phase: "idle" });
        this.emitEvent({ type: "agent_end" });

        return this._messages;
    }

    private emitEvent(event: AgentEvent): void {
        if (this.config.onEvent) {
            this.config.onEvent(event);
        }
    }

    private emitContextUpdate(
        tools: Record<
            string,
            unknown
        >[] = this.toolRegistry.toToolDefinitions(),
        compacted = false,
    ): void {
        this.emitEvent({
            type: "context_update",
            tokens: estimateChatPayloadTokens(this._messages, tools),
            maxTokens: this.contextWindowTokens(),
            compacted,
        });
    }

    private contextWindowTokens(): number | undefined {
        const configured =
            this.config.contextWindowTokens ||
            envNumber("LOGICIAN_CONTEXT_WINDOW") ||
            envNumber("LOGICIAN_CTX_SIZE");
        return configured && configured > 0 ? configured : undefined;
    }

    private hooksEnabled(): boolean {
        return (
            this.config.runtimeHooksEnabled !== false &&
            process.env.LOGICIAN_HOOKS !== "0"
        );
    }

    private async userPromptHookMessages(
        userMessage: string,
        basePayload: Record<string, unknown>,
    ): Promise<Message[]> {
        if (!this.hooksEnabled()) return [];
        try {
            const result = await runHookEvent("UserPromptSubmit", {
                ...basePayload,
                prompt: userMessage,
                timeout_seconds: 30,
            });
            const context = (result.additional_contexts || [])
                .map((item) => String(item || "").trim())
                .filter(Boolean)
                .join("\n\n");
            if (!context) return [];
            return [
                createUserMessage(
                    `<user-prompt-submit-hook>\n${context}\n</user-prompt-submit-hook>`,
                ),
            ];
        } catch {
            return [];
        }
    }

    private async runBeforeToolCall(
        toolCall: ToolCall,
        args: Record<string, unknown>,
    ): Promise<{
        content?: string;
        isError?: boolean;
        args?: Record<string, unknown>;
    } | undefined> {
        if (!this.hooks.beforeToolCall) return undefined;
        try {
            return (
                (await this.hooks.beforeToolCall({
                    toolCall,
                    args,
                    iteration: this.iterationCount,
                })) || undefined
            );
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `beforeToolCall hook failed: ${(e as Error).message}`,
            });
            return undefined;
        }
    }

    private async runAfterToolCall(
        toolCall: ToolCall,
        args: Record<string, unknown>,
        result: string,
        isError: boolean,
    ): Promise<{ content?: string; isError?: boolean } | undefined> {
        if (!this.hooks.afterToolCall) return undefined;
        try {
            return (
                (await this.hooks.afterToolCall({
                    toolCall,
                    args,
                    result,
                    isError,
                    iteration: this.iterationCount,
                })) || undefined
            );
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `afterToolCall hook failed: ${(e as Error).message}`,
            });
            return undefined;
        }
    }

    private async runPrepareNextTurn(hadToolCalls: boolean): Promise<void> {
        if (!this.hooks.prepareNextTurn) return;
        try {
            const out = await this.hooks.prepareNextTurn({
                messages: this._messages,
                iteration: this.iterationCount,
                hadToolCalls,
            });
            if (out?.messages) this._messages = out.messages;
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `prepareNextTurn hook failed: ${(e as Error).message}`,
            });
        }
    }

    private async runShouldStopAfterTurn(
        hadToolCalls: boolean,
    ): Promise<boolean> {
        if (!this.hooks.shouldStopAfterTurn) return false;
        try {
            return (
                (await this.hooks.shouldStopAfterTurn({
                    messages: this._messages,
                    iteration: this.iterationCount,
                    hadToolCalls,
                })) === true
            );
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `shouldStopAfterTurn hook failed: ${(e as Error).message}`,
            });
            return false;
        }
    }

    private async runContinueAfterTurn(
        assistantText: string,
    ): Promise<{ message: string } | undefined> {
        if (!this.hooks.continueAfterTurn) return undefined;
        try {
            const r = await this.hooks.continueAfterTurn({
                messages: this._messages,
                iteration: this.iterationCount,
                assistantText,
            });
            return r && r.message ? { message: r.message } : undefined;
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `continueAfterTurn hook failed: ${(e as Error).message}`,
            });
            return undefined;
        }
    }

    private async recordToolResult(
        transcriptPath: string,
        hookBasePayload: Record<string, unknown>,
        toolCall: ToolCall,
        toolInput: Record<string, unknown>,
        result: string,
        isError: boolean,
    ): Promise<void> {
        this._messages.push(
            createToolResultMessage(
                toolCall.id,
                toolCall.name,
                result,
                isError,
            ),
        );

        this.appendTranscript(transcriptPath, {
            type: "toolResult",
            timestamp: new Date().toISOString(),
            toolCallId: toolCall.id,
            toolName: toolCall.name,
            content: [{ type: "text", text: result }],
            isError,
        });

        await this.runHookSafely("PostToolUse", {
            ...hookBasePayload,
            matcher_value: hookMatcherValue(toolCall.name),
            tool_name: toolCall.name,
            tool_input: toolInput,
            tool_response: result,
        });
    }

    private async runHookSafely(
        eventType: string,
        payload: Record<string, unknown>,
    ): Promise<void> {
        if (!this.hooksEnabled()) return;
        try {
            await runHookEvent(eventType, payload);
        } catch {
            // Hook failures should not break the agent turn.
        }
    }

    private appendTranscript(
        transcriptPath: string,
        entry: Record<string, unknown>,
    ): void {
        if (!transcriptPath) return;
        try {
            mkdirSync(dirname(transcriptPath), { recursive: true });
            appendFileSync(
                transcriptPath,
                `${JSON.stringify(entry)}\n`,
                "utf8",
            );
        } catch {
            // Transcript persistence is best-effort for hook integrations.
        }
    }
}

function parseToolInput(raw: string): Record<string, unknown> {
    try {
        const parsed = JSON.parse(raw || "{}");
        return parsed && typeof parsed === "object" && !Array.isArray(parsed)
            ? (parsed as Record<string, unknown>)
            : {};
    } catch {
        return {};
    }
}

function hookMatcherValue(toolName: string): string {
    const aliases: Record<string, string[]> = {
        bash: ["Bash"],
        read_file: ["Read"],
        write_file: ["Write"],
        edit_file: ["Edit"],
        list_files: ["LS"],
        rg_search: ["Grep"],
        todo_write: ["TodoWrite"],
        web_search: ["WebSearch"],
        web_fetch: ["WebFetch"],
    };
    return [toolName, ...(aliases[toolName] || [])].join("|");
}

function isContextFullError(error: Error): boolean {
    const message = `${error.name || ""} ${error.message || ""}`.toLowerCase();
    return [
        "context full",
        "context window",
        "exceeds context",
        "exceed context",
        "maximum context",
        "max context",
        "prompt too long",
        "too many tokens",
        "tokens exceed",
        "context size",
        "n_ctx",
    ].some((pattern) => message.includes(pattern));
}

function envNumber(name: string): number | undefined {
    const raw = process.env[name];
    if (!raw) return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
}
