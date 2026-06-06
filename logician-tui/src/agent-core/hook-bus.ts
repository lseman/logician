// ── Typed hook bus ─────────────────────────────────────────────────────────
// Unifies the single-handler contract hooks into one multi-handler bus with
// per-event reducer semantics, mirroring pi's hook design
// (packages/agent/docs/hooks.md). Multiple extensions register handlers for the
// same event and compose deterministically:
//
//   beforeToolCall   → early-block: first {content} short-circuits; {args}
//                      rewrites thread to later handlers.
//   afterToolCall    → patch-accumulate: each handler sees the prior patch;
//                      later non-undefined fields win.
//   prepareNextTurn  → transform: messages thread through all handlers.
//   shouldStopAfterTurn → first-true wins.
//
// The bus emits a single `AgentLoopHooks` via `toHooks()`, so the agent loop is
// unchanged — it still calls one handler per event. `observe()` is a read-only
// firehose over every event. Handlers can be scoped with source metadata so a
// failing extension is identifiable, and `errorMode` controls whether a thrown
// handler aborts the chain or is skipped.

import type {
    AgentLoopHooks,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    PrepareNextTurnContext,
    PrepareNextTurnResult,
    ShouldStopAfterTurnContext,
    GetSteeringMessagesContext,
    GetFollowUpMessagesContext,
    Message,
} from "./types.ts";

export type HookEventName = keyof AgentLoopHooks;

export interface HookRegistration {
    source?: string;
}

export type HookErrorMode = "continue" | "throw";

export interface HookBusOptions {
    errorMode?: HookErrorMode;
    onError?: (error: Error, event: HookEventName, source?: string) => void;
}

interface Entry<H> {
    handler: H;
    source?: string;
}

type BeforeHandler = NonNullable<AgentLoopHooks["beforeToolCall"]>;
type AfterHandler = NonNullable<AgentLoopHooks["afterToolCall"]>;
type PrepareHandler = NonNullable<AgentLoopHooks["prepareNextTurn"]>;
type StopHandler = NonNullable<AgentLoopHooks["shouldStopAfterTurn"]>;
type SteeringHandler = NonNullable<AgentLoopHooks["getSteeringMessages"]>;
type FollowUpHandler = NonNullable<AgentLoopHooks["getFollowUpMessages"]>;

// Read-only observer: sees every event with its name, return ignored.
export type HookObserver = (
    event: HookEventName,
    ctx: unknown,
) => void | Promise<void>;

export class HookBus {
    private before: Entry<BeforeHandler>[] = [];
    private after: Entry<AfterHandler>[] = [];
    private prepare: Entry<PrepareHandler>[] = [];
    private stop: Entry<StopHandler>[] = [];
    private steering: Entry<SteeringHandler>[] = [];
    private followUp: Entry<FollowUpHandler>[] = [];
    private observers: HookObserver[] = [];

    private errorMode: HookErrorMode;
    private onError?: HookBusOptions["onError"];

    constructor(options: HookBusOptions = {}) {
        this.errorMode = options.errorMode ?? "continue";
        this.onError = options.onError;
    }

    // Register one handler for an event. Returns an unsubscribe function.
    on<E extends HookEventName>(
        event: E,
        handler: NonNullable<AgentLoopHooks[E]>,
        reg: HookRegistration = {},
    ): () => void {
        const list = this.listFor(event) as Entry<AgentLoopHooks[E]>[];
        const entry = { handler, source: reg.source };
        list.push(entry);
        return () => {
            const i = list.indexOf(entry);
            if (i >= 0) list.splice(i, 1);
        };
    }

    // Register a whole AgentLoopHooks object at once (each present handler).
    register(hooks: AgentLoopHooks, reg: HookRegistration = {}): () => void {
        const offs: Array<() => void> = [];
        if (hooks.beforeToolCall)
            offs.push(this.on("beforeToolCall", hooks.beforeToolCall, reg));
        if (hooks.afterToolCall)
            offs.push(this.on("afterToolCall", hooks.afterToolCall, reg));
        if (hooks.prepareNextTurn)
            offs.push(this.on("prepareNextTurn", hooks.prepareNextTurn, reg));
        if (hooks.shouldStopAfterTurn)
            offs.push(
                this.on("shouldStopAfterTurn", hooks.shouldStopAfterTurn, reg),
            );
        if (hooks.getSteeringMessages)
            offs.push(
                this.on("getSteeringMessages", hooks.getSteeringMessages, reg),
            );
        if (hooks.getFollowUpMessages)
            offs.push(
                this.on("getFollowUpMessages", hooks.getFollowUpMessages, reg),
            );
        return () => offs.forEach((off) => off());
    }

    observe(observer: HookObserver): () => void {
        this.observers.push(observer);
        return () => {
            const i = this.observers.indexOf(observer);
            if (i >= 0) this.observers.splice(i, 1);
        };
    }

    clear(): void {
        this.before = [];
        this.after = [];
        this.prepare = [];
        this.stop = [];
        this.steering = [];
        this.followUp = [];
        this.observers = [];
    }

    // Single composed AgentLoopHooks for the agent loop. Each event runs its
    // reducer over all registered handlers.
    toHooks(): AgentLoopHooks {
        return {
            beforeToolCall: (ctx) => this.runBefore(ctx),
            afterToolCall: (ctx) => this.runAfter(ctx),
            prepareNextTurn: (ctx) => this.runPrepare(ctx),
            shouldStopAfterTurn: (ctx) => this.runStop(ctx),
            getSteeringMessages: (ctx) => this.runSteering(ctx),
            getFollowUpMessages: (ctx) => this.runFollowUp(ctx),
        };
    }

    // ── Reducers ───────────────────────────────────────────────────────────

    private async runBefore(
        ctx: BeforeToolCallContext,
    ): Promise<BeforeToolCallResult | undefined> {
        await this.notify("beforeToolCall", ctx);
        if (!this.before.length) return undefined;
        let current = ctx;
        let rewritten: Record<string, unknown> | undefined;
        for (const { handler, source } of this.before) {
            const r = await this.guard(
                () => handler(current),
                "beforeToolCall",
                source,
            );
            if (!r) continue;
            // A content result short-circuits: tool is not run.
            if (r.content !== undefined) {
                return rewritten ? { ...r, args: r.args ?? rewritten } : r;
            }
            if (r.args !== undefined) {
                rewritten = r.args;
                current = { ...current, args: r.args };
            }
        }
        return rewritten ? { args: rewritten } : undefined;
    }

    private async runAfter(
        ctx: AfterToolCallContext,
    ): Promise<AfterToolCallResult | undefined> {
        await this.notify("afterToolCall", ctx);
        if (!this.after.length) return undefined;
        let current = ctx;
        let modified = false;
        for (const { handler, source } of this.after) {
            const r = await this.guard(
                () => handler(current),
                "afterToolCall",
                source,
            );
            if (!r) continue;
            current = {
                ...current,
                result: r.content ?? current.result,
                isError: r.isError ?? current.isError,
            };
            modified = true;
        }
        return modified
            ? { content: current.result, isError: current.isError }
            : undefined;
    }

    private async runPrepare(
        ctx: PrepareNextTurnContext,
    ): Promise<PrepareNextTurnResult | undefined> {
        await this.notify("prepareNextTurn", ctx);
        if (!this.prepare.length) return undefined;
        let messages = ctx.messages;
        for (const { handler, source } of this.prepare) {
            const r = await this.guard(
                () => handler({ ...ctx, messages }),
                "prepareNextTurn",
                source,
            );
            if (r?.messages) messages = r.messages;
        }
        return messages === ctx.messages ? undefined : { messages };
    }

    private async runStop(
        ctx: ShouldStopAfterTurnContext,
    ): Promise<boolean | undefined> {
        await this.notify("shouldStopAfterTurn", ctx);
        for (const { handler, source } of this.stop) {
            const r = await this.guard(
                () => handler(ctx),
                "shouldStopAfterTurn",
                source,
            );
            if (r === true) return true;
        }
        return undefined;
    }

    private async runSteering(
        ctx: GetSteeringMessagesContext,
    ): Promise<Message[] | undefined> {
        await this.notify("getSteeringMessages", ctx);
        const out: Message[] = [];
        for (const { handler, source } of this.steering) {
            const r = await this.guard(
                () => handler({ ...ctx, messages: [...ctx.messages, ...out] }),
                "getSteeringMessages",
                source,
            );
            if (r?.length) out.push(...r);
        }
        return out.length ? out : undefined;
    }

    private async runFollowUp(
        ctx: GetFollowUpMessagesContext,
    ): Promise<Message[] | undefined> {
        await this.notify("getFollowUpMessages", ctx);
        const out: Message[] = [];
        for (const { handler, source } of this.followUp) {
            const r = await this.guard(
                () =>
                    handler({
                        ...ctx,
                        messages: [...ctx.messages, ...out],
                    }),
                "getFollowUpMessages",
                source,
            );
            if (r?.length) out.push(...r);
        }
        return out.length ? out : undefined;
    }

    // ── Internals ──────────────────────────────────────────────────────────

    private listFor(event: HookEventName): Entry<unknown>[] {
        switch (event) {
            case "beforeToolCall":
                return this.before as Entry<unknown>[];
            case "afterToolCall":
                return this.after as Entry<unknown>[];
            case "prepareNextTurn":
                return this.prepare as Entry<unknown>[];
            case "shouldStopAfterTurn":
                return this.stop as Entry<unknown>[];
            case "getSteeringMessages":
                return this.steering as Entry<unknown>[];
            case "getFollowUpMessages":
                return this.followUp as Entry<unknown>[];
        }
    }

    private async notify(event: HookEventName, ctx: unknown): Promise<void> {
        for (const observer of this.observers) {
            try {
                await observer(event, ctx);
            } catch {
                // Observers are read-only; their failures never affect a turn.
            }
        }
    }

    private async guard<T>(
        fn: () => T | Promise<T>,
        event: HookEventName,
        source?: string,
    ): Promise<T | undefined> {
        try {
            return await fn();
        } catch (e) {
            const error = e as Error;
            this.onError?.(error, event, source);
            if (this.errorMode === "throw") throw error;
            return undefined;
        }
    }
}
