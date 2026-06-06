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
import type { LLMBackend, OpenAIBackend } from "./backend.ts";
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
import { parseToolCalls, parseToolInput } from "./parser.ts";
import { ToolRegistry } from "./tools/registry.ts";
import { createDefaultTools } from "./default-tools.ts";
import { runHookEvent } from "./plugins.ts";
import { composeHooks, buildBuiltinHooks } from "./builtin-hooks.ts";

// Cap on pi-style continuations within one run, to bound runaway loops when a
// continuation hook keeps resuming the agent.
const DEFAULT_MAX_CONTINUATIONS = 12;

interface RunnableToolCall {
    kind: "runnable";
    call: ToolCall;
    args: Record<string, unknown>;
}

interface FinalToolCall {
    kind: "final";
    call: ToolCall;
    args: Record<string, unknown>;
    result: string;
    isError: boolean;
}

type PreparedLoopToolCall = RunnableToolCall | FinalToolCall;

interface ExecutedLoopToolCall {
    call: ToolCall;
    args: Record<string, unknown>;
    result: string;
    isError: boolean;
}

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
    private _retryAttempt = 0;
    private _retryAbort?: () => void;
    private get maxRetries(): number {
        return this.config.maxRetries ?? 3;
    }
    private get retryBaseDelayMs(): number {
        return this.config.retryBaseDelayMs ?? 1000;
    }

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
        this._retryAttempt = 0;

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

            const steeringMessages = await this.runGetSteeringMessages();
            if (steeringMessages.length) {
                this.appendInjectedMessages(transcriptPath, steeringMessages);
            }

            // Estimate context before the model call so the status bar stays live.
            const toolDefs = this.toolRegistry.toToolDefinitions();
            this.emitContextUpdate(toolDefs);

            // Get LLM response — with auto-retry on provider errors.
            let assistantContent = "";
            let assistantToolCalls: ToolCall[] = [];
            let llmSuccess = false;

            // Context-full compaction path (one retry, no backoff).
            let compactionAttempted = false;
            while (!llmSuccess && !compactionAttempted) {
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
                        if (assistantToolCalls.length > 0) {
                            this.emitEvent({
                                type: "repair_nudge",
                                turnId,
                                repairStage: "parse_tool_calls",
                                message:
                                    "Recovered tool call(s) from textual model output.",
                            });
                        }
                    }
                    llmSuccess = true;
                } catch (e: unknown) {
                    const error = e as Error;

                    // 1. Context-full → compact once and retry.
                    if (!compactionAttempted && isContextFullError(error)) {
                        compactionAttempted = true;
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
                        // Compaction didn't help — fall through to error.
                    }

                    // 2. Auto-retry on provider errors (429, 500, 502, 503, 504).
                    if (
                        this.config.autoRetryEnabled !== false &&
                        isProviderError(error)
                    ) {
                        const canRetry = this._retryAttempt < this.maxRetries;
                        if (canRetry) {
                            this._retryAttempt++;
                            const delayMs =
                                this.retryBaseDelayMs *
                                Math.pow(2, this._retryAttempt - 1);
                            this.emitEvent({
                                type: "auto_retry_start",
                                attempt: this._retryAttempt,
                                maxRetries: this.maxRetries,
                                delayMs,
                                error: error.message,
                            });
                            await this._sleep(delayMs, turnId);
                            this.emitEvent({
                                type: "auto_retry_end",
                                attempt: this._retryAttempt,
                                success: true,
                            });
                            continue;
                        }
                    }

                    // 3. Give up.
                    this.emitEvent({ type: "error", message: error.message });
                    assistantContent = "";
                    assistantToolCalls = [];
                    break;
                }
            }

            // Reset retry state after each turn (success or failure).
            this._retryAttempt = 0;

            if (!assistantContent && assistantToolCalls.length === 0) {
                this.emitEvent({ type: "turn_end", turnId });
                break;
            }

            // Emit message_start before assistant response (for steering
            // detection — the bridge can detect when steering messages have
            // been consumed by checking if their text appears in messages).
            this.emitEvent({ type: "message_start", turnId, role: "assistant" });
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
                await this.executeToolCalls(
                    assistantToolCalls,
                    turnId,
                    transcriptPath,
                    hookBasePayload,
                );

                if (this.signal?.aborted) {
                    this.emitEvent({ type: "turn_end", turnId });
                    break;
                }
            }

            // prepareNextTurn / shouldStopAfterTurn contract hooks.
            const hadToolCalls = assistantToolCalls.length > 0;
            const isContinuation = this.continuationCount > 0;
            await this.runPrepareNextTurn(hadToolCalls, isContinuation);
            if (await this.runShouldStopAfterTurn(hadToolCalls, isContinuation)) {
                this.emitEvent({ type: "turn_end", turnId });
                break;
            }

            // Continue loop: model called tools or follow-ups exist.
            // turn_end stays open — the TUI sees one continuous turn.
            if (hadToolCalls) {
                continue;
            }

            // No tool calls: check follow-ups before ending the turn.
            if (this.continuationCount < this.maxContinuations) {
                const followUps =
                    await this.runGetFollowUpMessages(assistantContent);
                if (followUps.length) {
                    this.continuationCount++;
                    this.appendInjectedMessages(transcriptPath, followUps);
                    continue;
                }
            }

            // Truly done — no tools, no follow-ups.
            this.emitEvent({ type: "turn_end", turnId });
            break;
        }

        await this.runHookSafely("Stop", {
            ...hookBasePayload,
            stop_hook_active: false,
        });
        this.emitEvent({ type: "phase", phase: "idle" });
        this.emitEvent({ type: "agent_end" });

        return this._messages;
    }

    private emitEvent(event: AgentEvent): void {
        if (event.type === "turn_end" && this.config.turnEndCallback) {
            this.config.turnEndCallback(event.turnId);
        }
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

    private async runPrepareNextTurn(
        hadToolCalls: boolean,
        isContinuation: boolean,
    ): Promise<void> {
        if (!this.hooks.prepareNextTurn) return;
        try {
            const out = await this.hooks.prepareNextTurn({
                messages: this._messages,
                iteration: this.iterationCount,
                hadToolCalls,
                continuationCount: this.continuationCount,
                isContinuation,
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
        isContinuation: boolean,
    ): Promise<boolean> {
        if (!this.hooks.shouldStopAfterTurn) return false;
        try {
            return (
                (await this.hooks.shouldStopAfterTurn({
                    messages: this._messages,
                    iteration: this.iterationCount,
                    hadToolCalls,
                    continuationCount: this.continuationCount,
                    isContinuation,
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

    private async runGetSteeringMessages(): Promise<Message[]> {
        if (!this.hooks.getSteeringMessages) return [];
        try {
            const r = await this.hooks.getSteeringMessages({
                messages: this._messages,
                iteration: this.iterationCount,
            });
            return r?.length ? r : [];
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `getSteeringMessages hook failed: ${(e as Error).message}`,
            });
            return [];
        }
    }

    private async runGetFollowUpMessages(
        assistantText: string,
    ): Promise<Message[]> {
        if (!this.hooks.getFollowUpMessages) return [];
        try {
            const r = await this.hooks.getFollowUpMessages({
                messages: this._messages,
                iteration: this.iterationCount,
                assistantText,
                continuationCount: this.continuationCount,
                maxContinuations: this.maxContinuations,
            });
            return r?.length ? r : [];
        } catch (e) {
            this.emitEvent({
                type: "error",
                message: `getFollowUpMessages hook failed: ${(e as Error).message}`,
            });
            return [];
        }
    }

    private async executeToolCalls(
        toolCalls: ToolCall[],
        turnId: string,
        transcriptPath: string,
        hookBasePayload: Record<string, unknown>,
    ): Promise<void> {
        const prepared: PreparedLoopToolCall[] = [];
        for (const toolCall of toolCalls) {
            if (this.signal?.aborted) {
                this.emitEvent({
                    type: "error",
                    message: "Operation aborted",
                });
                return;
            }
            prepared.push(
                await this.prepareLoopToolCall(
                    toolCall,
                    turnId,
                    hookBasePayload,
                ),
            );
        }

        const runnable = prepared.filter(
            (item): item is RunnableToolCall => item.kind === "runnable",
        );
        const executedById = new Map<string, ExecutedLoopToolCall>();
        const parallel = this.shouldExecuteParallel(runnable);

        const executed = parallel
            ? await Promise.all(
                  runnable.map((item) => this.executePreparedToolCall(item)),
              )
            : await this.executePreparedToolCallsSequentially(runnable);
        for (const item of executed) executedById.set(item.call.id, item);

        for (const item of prepared) {
            const executedItem =
                item.kind === "final" ? item : executedById.get(item.call.id);
            if (!executedItem) continue;
            await this.finalizeToolCall(
                executedItem,
                transcriptPath,
                hookBasePayload,
            );
        }
    }

    private async prepareLoopToolCall(
        toolCall: ToolCall,
        turnId: string,
        hookBasePayload: Record<string, unknown>,
    ): Promise<PreparedLoopToolCall> {
        const prepared = this.toolRegistry.prepare(toolCall);
        let toolInput = prepared.args;
        let activeToolCall = prepared.call;

        this.emitEvent({
            type: "tool_call_start",
            toolName: activeToolCall.name,
            toolCallId: activeToolCall.id,
            args: activeToolCall.arguments,
        });

        if (prepared.error) {
            this.emitEvent({
                type: "repair_nudge",
                turnId,
                repairStage: "prepare_arguments",
                toolName: toolCall.name,
                message: prepared.error,
            });
            return {
                kind: "final",
                call: activeToolCall,
                args: toolInput,
                result: prepared.error,
                isError: true,
            };
        }

        const before = await this.runBeforeToolCall(activeToolCall, toolInput);
        if (before?.content !== undefined) {
            return {
                kind: "final",
                call: activeToolCall,
                args: toolInput,
                result: before.content,
                isError: before.isError ?? false,
            };
        }
        if (before?.args !== undefined) {
            toolInput = before.args;
            activeToolCall = {
                ...toolCall,
                arguments: JSON.stringify(before.args),
            };
        }

        await this.runHookSafely("PreToolUse", {
            ...hookBasePayload,
            matcher_value: hookMatcherValue(activeToolCall.name),
            tool_name: activeToolCall.name,
            tool_input: toolInput,
        });

        return { kind: "runnable", call: activeToolCall, args: toolInput };
    }

    private async executePreparedToolCallsSequentially(
        calls: RunnableToolCall[],
    ): Promise<ExecutedLoopToolCall[]> {
        const out: ExecutedLoopToolCall[] = [];
        for (const call of calls) {
            if (this.signal?.aborted) break;
            out.push(await this.executePreparedToolCall(call));
        }
        return out;
    }

    private async executePreparedToolCall(
        prepared: RunnableToolCall,
    ): Promise<ExecutedLoopToolCall> {
        const result = await this.toolRegistry.execute(
            prepared.call,
            {
                signal: this.signal,
                onUpdate: (partialResult) => {
                    this.emitEvent({
                        type: "tool_call_update",
                        toolName: prepared.call.name,
                        toolCallId: prepared.call.id,
                        partialResult,
                    });
                },
            },
            prepared.args,
        );
        return {
            call: prepared.call,
            args: prepared.args,
            result,
            isError: result.startsWith("Error:"),
        };
    }

    private async finalizeToolCall(
        executed: ExecutedLoopToolCall,
        transcriptPath: string,
        hookBasePayload: Record<string, unknown>,
    ): Promise<void> {
        let { result, isError } = executed;
        const after = await this.runAfterToolCall(
            executed.call,
            executed.args,
            result,
            isError,
        );
        if (after) {
            if (after.content !== undefined) result = after.content;
            if (after.isError !== undefined) isError = after.isError;
        }

        this.emitEvent({
            type: "tool_call_end",
            toolName: executed.call.name,
            toolCallId: executed.call.id,
            result,
            isError,
        });

        await this.recordToolResult(
            transcriptPath,
            hookBasePayload,
            executed.call,
            executed.args,
            result,
            isError,
        );
    }

    private shouldExecuteParallel(calls: RunnableToolCall[]): boolean {
        if ((this.config.toolExecution ?? "parallel") !== "parallel")
            return false;
        return calls.every(
            (call) =>
                this.toolRegistry.get(call.call.name)?.executionMode ===
                "parallel",
        );
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

    private appendInjectedMessages(
        transcriptPath: string,
        messages: Message[],
    ): void {
        for (const message of messages) {
            this._messages.push(message);
            if (message.role !== "user") continue;
            this.appendTranscript(transcriptPath, {
                type: "user",
                timestamp: new Date().toISOString(),
                message: { role: "user", content: message.content || "" },
            });
        }
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

    private async _sleep(ms: number, _turnId: string): Promise<void> {
        // Respect abort signal during backoff sleep.
        if (this.signal?.aborted) return;
        await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, ms);
            this.signal?.addEventListener(
                "abort",
                () => {
                    clearTimeout(timer);
                    this.emitEvent({
                        type: "error",
                        message: "Retry cancelled by abort",
                    });
                    reject(new Error("Retry cancelled"));
                },
                { once: true },
            );
        });
    }

    // ── Model cycling ─────────────────────────────────────────────────
    // Pi-style: cycle through configured models (forward or backward).
    // Creates a new backend with the selected model. Emits `model_select`.

    /** Build the model list — primary + alternates. */
    private getModelList(): string[] {
        const models = this.config.models;
        if (!models || models.length === 0) return [this.config.model];
        // Deduplicate while preserving order; primary is always first.
        const seen = new Set<string>();
        const list: string[] = [];
        for (const m of [this.config.model, ...models]) {
            if (!seen.has(m)) {
                seen.add(m);
                list.push(m);
            }
        }
        return list;
    }

    /** Current model index in the model list. */
    private _modelIndex = 0;
    private get modelIndex(): number {
        const list = this.getModelList();
        if (this._modelIndex >= list.length) this._modelIndex = 0;
        return this._modelIndex;
    }
    private get currentModel(): string {
        return this.getModelList()[this.modelIndex] ?? this.config.model;
    }

    /** Cycle to the next model (forward). Returns the new model name. */
    cycleModel(direction: "forward" | "backward" = "forward"): string {
        const list = this.getModelList();
        if (list.length <= 1) return this.config.model;

        const step = direction === "forward" ? 1 : -1;
        this._modelIndex =
            (this._modelIndex + step + list.length) % list.length;

        // Swap backend model.
        const newModel = list[this._modelIndex];
        this.backend = new OpenAIBackend({
            baseUrl: this.backend instanceof OpenAIBackend
                ? (this.backend as any).baseUrl
                : this.config.baseUrl,
            model: newModel,
        });

        this.emitEvent({
            type: "model_select",
            model: newModel,
            index: this._modelIndex,
        });
        return newModel;
    }

    /** Get the current model name (for TUI status bar). */
    getModel(): string {
        return this.currentModel;
    }

    /** Get all available models (for TUI status bar). */
    getModels(): string[] {
        return this.getModelList();
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

function isProviderError(error: Error): boolean {
    const message = `${error.name || ""} ${error.message || ""}`.toLowerCase();
    // HTTP status codes from the backend.
    if (
        /\b(429|500|502|503|504)\b/.test(message) ||
        /llm request failed/.test(message)
    ) {
        return true;
    }
    // Network-level errors.
    if (
        message.includes("econnrefused") ||
        message.includes("econnreset") ||
        message.includes("etimedout") ||
        message.includes("eai-again") ||
        message.includes("socket hang up") ||
        message.includes("connection refused") ||
        message.includes("connection reset") ||
        message.includes("connection timeout") ||
        message.includes("network error") ||
        message.includes("fetch failed")
    ) {
        return true;
    }
    return false;
}
