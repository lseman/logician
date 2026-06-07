// ── AgentHarness ───────────────────────────────────────────────────────────
// Orchestration layer above AgentLoop. Adds an explicit phase, runtime config
// setters that take effect on the *next* turn (never mutating an in-flight
// provider request), and steering / follow-up / nextTurn queues drained at
// save points. Mirrors pi's AgentHarness (packages/agent/docs/agent-harness.md),
// scoped to what logician-tui's loop exposes.
//
// The loop already exposes a save point: the `prepareNextTurn` contract hook
// fires after each turn and can rewrite the working messages. The harness
// installs a drain hook there to inject queued messages and apply config
// changes between turns.

import type {
    AgentConfig,
    AgentLoopHooks,
    GetFollowUpMessagesContext,
    GetSteeringMessagesContext,
    Message,
    PrepareNextTurnContext,
    PrepareNextTurnResult,
    QueueMode,
    Tool,
} from "./types.ts";
import type { LLMBackend } from "./backend.ts";
import type { ToolRegistry } from "./tools/registry.ts";
import { AgentLoop } from "./loop.ts";
import {
    createUserMessage,
    compactMessagesForContext,
    microCompactMessages,
    estimateChatPayloadTokens,
    convertToChatFormat,
} from "./messages.ts";

export type HarnessPhase = "idle" | "turn";

export class HarnessBusyError extends Error {
    constructor(op: string) {
        super(`AgentHarness is busy; ${op} requires idle phase`);
        this.name = "HarnessBusyError";
    }
}

export interface AgentHarnessOptions {
    config: AgentConfig;
    backend: LLMBackend;
    cwd?: string;
    maxIterations?: number;
}

export class AgentHarness {
    private config: AgentConfig;
    private backend: LLMBackend;
    private cwd?: string;
    private maxIterations?: number;

    private _phase: HarnessPhase = "idle";
    private loop: AgentLoop | null = null;
    private abortController: AbortController | null = null;
    // Conversation persisted across prompts so follow-ups ("continue", "go on")
    // retain context.
    private history: Message[] = [];

    // Queues drained at save points.
    private steeringQueue: string[] = [];
    private followUpQueue: string[] = [];
    // nextTurn messages survive across runs and abort; injected before the next
    // user prompt. Exposed for bridge UI display.
    nextTurnQueue: string[] = [];
    // Queue drain modes: "all" = drain everything, "one-at-a-time" = drain one.
    private steeringQueueMode: QueueMode;
    private followUpQueueMode: QueueMode;

    constructor(options: AgentHarnessOptions) {
        this.config = options.config;
        this.backend = options.backend;
        this.cwd = options.cwd;
        this.maxIterations = options.maxIterations;
        this.steeringQueueMode =
            options.config.steeringQueueMode ?? "all";
        this.followUpQueueMode =
            options.config.followUpQueueMode ?? "all";
    }

    get phase(): HarnessPhase {
        return this._phase;
    }

    // ── Structural operation: prompt ───────────────────────────────────────
    // Rejected while busy. Drains nextTurn messages before the user prompt.
    async prompt(userMessage: string): Promise<Message[]> {
        if (this._phase !== "idle") throw new HarnessBusyError("prompt");
        this._phase = "turn";
        this.abortController = new AbortController();

        const pending = this.nextTurnQueue.splice(0);
        const promptText = pending.length
            ? `${pending.join("\n\n")}\n\n${userMessage}`
            : userMessage;

        try {
            this.loop = new AgentLoop({
                config: this.withDrainHook(this.config),
                backend: this.backend,
                cwd: this.cwd,
                maxIterations: this.maxIterations,
                signal: this.abortController.signal,
                initialMessages: this.history.length
                    ? this.history
                    : undefined,
            });
            const result = await this.loop.run(promptText);
            // Persist the full conversation for the next prompt.
            this.history = result;
            return result;
        } finally {
            this._phase = "idle";
            this.loop = null;
            this.abortController = null;
            // Steering/follow-up are turn-scoped; clear leftovers.
            this.steeringQueue = [];
            this.followUpQueue = [];
        }
    }

    // ── Runtime config setters (take effect next turn) ─────────────────────
    // The loop reads these config fields live each turn, so updating them here
    // affects the next turn snapshot, not the in-flight provider request.

    setSystemPrompt(systemPrompt: string): void {
        this.config.systemPrompt = systemPrompt;
    }

    setTemperature(temperature: number): void {
        this.config.temperature = temperature;
    }

    setMaxTokens(maxTokens: number): void {
        this.config.maxTokens = maxTokens;
    }

    // Tools are held by the loop's registry, built per run. Update config so the
    // next run picks them up; if a run is active, patch the live registry too
    // (applies to the next turn since defs are read each turn).
    setTools(tools: Tool[]): void {
        this.config.tools = tools;
        if (this.loop) {
            const registry = this.loop.tools;
            for (const existing of registry.list()) {
                registry.unregister(existing.name);
            }
            registry.registerMany(tools);
        }
    }

    // ── Queue operations (allowed during a turn) ───────────────────────────

    // Inject guidance into the running turn (drained at the next save point).
    steer(text: string): void {
        this.steeringQueue.push(text);
    }

    // Queue a message for after the current turn completes (same drain point).
    followUp(text: string): void {
        this.followUpQueue.push(text);
    }

    // Queue a message inserted before the next user prompt. Survives abort.
    nextTurn(text: string): void {
        this.nextTurnQueue.push(text);
    }

    // Abort the running turn. Clears steering/follow-up; preserves nextTurn.
    abort(): void {
        this.abortController?.abort();
        this.steeringQueue = [];
        this.followUpQueue = [];
    }

    // Expose the current steering queue for the TUI bridge.
    getSteerQueue(): string[] {
        return this.steeringQueue;
    }

    /** Controls how queued steering messages are drained. */
    setSteeringMode(mode: QueueMode): void {
        this.steeringQueueMode = mode;
    }

    /** Controls how queued follow-up messages are drained. */
    setFollowUpMode(mode: QueueMode): void {
        this.followUpQueueMode = mode;
    }

    get messages(): Message[] {
        return this.loop?.messages ?? this.history;
    }

   // Clear persisted conversation (new session / context reset).
    clearHistory(): void {
        this.history = [];
    }

    /** Compact messages using LLM-generated summary. Returns tokens saved, or null if nothing to compact. */
    async compact(): Promise<number | null> {
        const messages = this.loop?.messages ?? this.history;
        if (!messages.length) return null;
        const before = estimateChatPayloadTokens(messages);

        const targetTokens = Math.floor(
            (this.config.contextWindowTokens ?? 0) * 0.65,
        );
        const keepRecentMessages = Math.max(2, 8);

        const systemMessages = messages.filter(
            (m) => m.role === "system",
        );
        const nonSystem = messages.filter((m) => m.role !== "system");
        let tailStart = Math.max(0, nonSystem.length - keepRecentMessages);
        // Ensure tool pairs stay together
        while (tailStart > 0 && nonSystem[tailStart]?.role === "tool") {
            tailStart--;
        }

        const older = nonSystem.slice(0, tailStart);
        const recent = nonSystem.slice(tailStart);
        if (!older.length) {
            // Nothing to summarize — micro-compact oversized bodies
            const microResult = microCompactMessages(messages);
            if (microResult.changed) {
                if (this.loop) this.loop.setMessages(microResult.messages);
                else this.history = microResult.messages;
                return before - microResult.tokensAfter;
            }
            return 0;
        }

        // Two-pass: first truncate locally to fit budget, then LLM summarizes.
        // Sending full messages to the LLM defeats the purpose of compaction.
        const truncated = microCompactMessages(older);
        const summary = await this.generateSummary(
            truncated.messages,
            systemMessages,
        );
        if (!summary) {
            // LLM failed — fall back to local truncation
            const fallbackResult = compactMessagesForContext(messages, {
                targetTokens,
            });
            if (fallbackResult.changed) {
                if (this.loop) this.loop.setMessages(fallbackResult.messages);
                else this.history = fallbackResult.messages;
                return before - fallbackResult.tokensAfter;
            }
            return 0;
        }

        const compacted = [
            ...systemMessages,
            createUserMessage(
                `<context-compaction reason="manual">\n${summary}\n</context-compaction>`,
            ),
            ...recent,
        ];

        if (this.loop) {
            this.loop.setMessages(compacted);
        } else {
            this.history = compacted;
        }

        const after = estimateChatPayloadTokens(compacted);
        return before - after;
    }

    /** Ask LLM to summarize older messages for compaction. */
    private async generateSummary(
        messages: Message[],
        systemMessages: Message[],
    ): Promise<string | null> {
        try {
            const chatMessages = [
                ...systemMessages,
                {
                    role: "user" as const,
                    content:
                        "Summarize the following conversation history concisely. " +
                        "Focus on key decisions, actions taken, files modified, " +
                        "and any important context that should be retained. " +
                        "Be brief but preserve all actionable information.",
                },
                ...messages,
            ];

            const response = await this.backend.generate(
                convertToChatFormat(chatMessages),
                undefined,
                this.config.temperature ?? 0.3,
                Math.min(2048, (this.config.maxTokens ?? 4096) / 2),
                undefined,
                undefined,
                undefined,
                this.config.thinkingLevel,
            );

            return response.content?.trim() || null;
        } catch {
            return null;
        }
    }

    // Live tool registry of the running loop, or null when idle.
    get tools(): ToolRegistry | null {
        return this.loop?.tools ?? null;
    }

    // ── Model cycling ──────────────────────────────────────────────────

    /** Get current model name. */
    getModel(): string {
        return this.loop?.getModel() ?? this.config.model;
    }

    /** Get all available models. */
    getModels(): string[] {
        return this.loop?.getModels() ?? (this.config.models ? [this.config.model, ...this.config.models] : [this.config.model]);
    }

    /** Cycle to the next model. Returns the new model name. */
    cycleModel(direction: "forward" | "backward" = "forward"): string {
        return this.loop?.cycleModel(direction) ?? this.config.model;
    }

    // ── Thinking level ─────────────────────────────────────────────────

    /** Get current thinking/reasoning level. */
    getThinkingLevel(): string {
        return this.config.thinkingLevel ?? "medium";
    }

    /** Set thinking/reasoning level. Takes effect on the next turn. */
    setThinkingLevel(level: string): void {
        this.config.thinkingLevel = level as
            | "off"
            | "minimal"
            | "low"
            | "medium"
            | "high"
            | "xhigh";
    }

    // ── Internals ──────────────────────────────────────────────────────────

    // Wrap config hooks with harness queue drains. Steering drains before the
    // next assistant response; follow-up drains only when the loop would stop.
    private withDrainHook(config: AgentConfig): AgentConfig {
        const userHooks: AgentLoopHooks = config.hooks || {};
        const userPrepare = userHooks.prepareNextTurn;
        const userAfterToolCall = userHooks.afterToolCall;
        const userSteering = userHooks.getSteeringMessages;
        const userFollowUp = userHooks.getFollowUpMessages;

        const getSteeringMessages = async (
            ctx: GetSteeringMessagesContext,
        ): Promise<Message[] | undefined> => {
            // Respect queue mode
            let out: string[];
            if (this.steeringQueueMode === "all") {
                out = this.steeringQueue.splice(0);
            } else {
                // one-at-a-time
                const first = this.steeringQueue[0];
                if (!first) out = [];
                else {
                    out = [first];
                    this.steeringQueue.shift();
                }
            }
            const messages = out.map((text) => createUserMessage(text));
            const user = await userSteering?.({
                ...ctx,
                messages: [...ctx.messages, ...messages],
            });
            if (user?.length) messages.push(...user);
            return messages.length ? messages : undefined;
        };

        const getFollowUpMessages = async (
            ctx: GetFollowUpMessagesContext,
        ): Promise<Message[] | undefined> => {
            // Respect queue mode
            let out: string[];
            if (this.followUpQueueMode === "all") {
                out = this.followUpQueue.splice(0);
            } else {
                // one-at-a-time
                const first = this.followUpQueue[0];
                if (!first) out = [];
                else {
                    out = [first];
                    this.followUpQueue.shift();
                }
            }
            const messages = out.map((text) => createUserMessage(text));
            const user = await userFollowUp?.({
                ...ctx,
                messages: [...ctx.messages, ...messages],
            });
            if (user?.length) messages.push(...user);
            return messages.length ? messages : undefined;
        };

        const prepareNextTurn = async (
            ctx: PrepareNextTurnContext,
        ): Promise<PrepareNextTurnResult | undefined> => {
            let messages = ctx.messages;
            if (userPrepare) {
                const r = await userPrepare({ ...ctx, messages });
                if (r?.messages) messages = r.messages;
            }
            return messages === ctx.messages ? undefined : { messages };
        };

        // Pi-style: afterToolCall can set terminate=true to stop the loop
        // after the current tool batch (only when ALL tools in the batch set it).
        const afterToolCall = async (
            ctx: import("./types").AfterToolCallContext,
        ) => userAfterToolCall?.(ctx);

        return {
            ...config,
            hooks: {
                ...userHooks,
                getSteeringMessages,
                getFollowUpMessages,
                prepareNextTurn,
                afterToolCall,
            },
        };
    }
}
