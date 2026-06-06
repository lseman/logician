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
    Message,
    PrepareNextTurnContext,
    PrepareNextTurnResult,
    Tool,
} from "./types.ts";
import type { LLMBackend } from "./backend.ts";
import type { ToolRegistry } from "./tools/registry.ts";
import { AgentLoop } from "./loop.ts";
import { createUserMessage } from "./messages.ts";

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
    // user prompt.
    private nextTurnQueue: string[] = [];

    constructor(options: AgentHarnessOptions) {
        this.config = options.config;
        this.backend = options.backend;
        this.cwd = options.cwd;
        this.maxIterations = options.maxIterations;
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

    get messages(): Message[] {
        return this.loop?.messages ?? this.history;
    }

    // Clear persisted conversation (new session / context reset).
    clearHistory(): void {
        this.history = [];
    }

    // Live tool registry of the running loop, or null when idle.
    get tools(): ToolRegistry | null {
        return this.loop?.tools ?? null;
    }

    // ── Internals ──────────────────────────────────────────────────────────

    // Wrap config.hooks.prepareNextTurn with the harness drain so queued
    // steering/follow-up messages are injected between turns. Any user-supplied
    // prepareNextTurn runs after the drain.
    private withDrainHook(config: AgentConfig): AgentConfig {
        const userHooks: AgentLoopHooks = config.hooks || {};
        const userPrepare = userHooks.prepareNextTurn;

        const prepareNextTurn = async (
            ctx: PrepareNextTurnContext,
        ): Promise<PrepareNextTurnResult | undefined> => {
            let messages = ctx.messages;
            const drained = [
                ...this.steeringQueue.splice(0),
                ...this.followUpQueue.splice(0),
            ];
            if (drained.length) {
                messages = [
                    ...messages,
                    ...drained.map((text) => createUserMessage(text)),
                ];
            }
            if (userPrepare) {
                const r = await userPrepare({ ...ctx, messages });
                if (r?.messages) messages = r.messages;
            }
            return messages === ctx.messages ? undefined : { messages };
        };

        return {
            ...config,
            hooks: { ...userHooks, prepareNextTurn },
        };
    }
}
