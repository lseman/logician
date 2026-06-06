// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentLoopHooks object, and composes them with any
// user-supplied hooks so both run. The contract is single-handler per event,
// so composition is explicit here rather than a generic bus (see #7 in
// AGENT_IMPROVEMENTS.md for the eventual typed hook bus).

import type { AgentConfig, AgentLoopHooks } from "./types.ts";
import { GuardEngine } from "./guards.ts";
import { BudgetTracker } from "./budget.ts";
import {
    compactMessagesForContext,
    microCompactMessages,
    estimateChatPayloadTokens,
} from "./messages.ts";

const DEFAULT_COMPACTION_FRACTION = 0.8;
// Don't run proactive compaction every turn — cooldown in turns.
const COMPACTION_COOLDOWN_TURNS = 3;

export interface BuiltinHookDeps {
    config: AgentConfig;
    contextWindowTokens: () => number | undefined;
    // Tool definitions for accurate payload token estimates.
    toolDefs: () => Record<string, unknown>[];
}

// Build the default safeguard hooks. Returns undefined per-event when a
// safeguard is disabled so composition can skip it cleanly.
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentLoopHooks {
    const { config } = deps;
    const guardsEnabled = config.guardsEnabled !== false;
    const budgetEnabled = config.budgetStopEnabled !== false;
    const compactionEnabled = config.proactiveCompactionEnabled !== false;

    const guards = guardsEnabled
        ? new GuardEngine({
              duplicateThreshold: config.duplicateToolThreshold,
              failureThreshold: config.toolFailureLoopThreshold,
          })
        : null;
    const budget = budgetEnabled ? new BudgetTracker() : null;

    const fraction =
        config.proactiveCompactionFraction ?? DEFAULT_COMPACTION_FRACTION;
    let lastCompactionTurn = -COMPACTION_COOLDOWN_TURNS;

    const hooks: AgentLoopHooks = {};

    if (guards) {
        hooks.beforeToolCall = ({ toolCall }) => {
            const decision = guards.inspect(toolCall);
            if (decision.block) {
                return { content: decision.message, isError: true };
            }
            return undefined;
        };
        hooks.afterToolCall = ({ toolCall, result, isError }) => {
            guards.record(toolCall, isError, result);
            return undefined;
        };
    }

    if (compactionEnabled) {
        hooks.prepareNextTurn = ({ messages, iteration }) => {
            const max = deps.contextWindowTokens();
            if (!max || max <= 0) return undefined;
            if (iteration - lastCompactionTurn < COMPACTION_COOLDOWN_TURNS) {
                return undefined;
            }
            const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
            if (tokens < max * fraction) return undefined;

            // Cheap pass first: trim oversized bodies only.
            const micro = microCompactMessages(messages);
            const microTokens = estimateChatPayloadTokens(
                micro.messages,
                deps.toolDefs(),
            );
            if (microTokens < max * fraction) {
                lastCompactionTurn = iteration;
                return micro.changed ? { messages: micro.messages } : undefined;
            }
            // Still over: full summarizing compaction.
            const full = compactMessagesForContext(micro.messages, {
                targetTokens: Math.floor(max * 0.65),
            });
            lastCompactionTurn = iteration;
            return full.changed ? { messages: full.messages } : undefined;
        };
    }

    if (budget) {
        hooks.shouldStopAfterTurn = ({ messages }) => {
            const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
            return budget.shouldStop(tokens);
        };
    }

    return hooks;
}

// Compose built-in hooks with user hooks so both run. Built-ins run first;
// user hooks see the built-ins' output and can override.
export function composeHooks(
    builtin: AgentLoopHooks,
    user: AgentLoopHooks,
): AgentLoopHooks {
    return {
        beforeToolCall: async (ctx) => {
            if (builtin.beforeToolCall) {
                const r = await builtin.beforeToolCall(ctx);
                // A built-in block short-circuits; user hook doesn't run.
                if (r?.content !== undefined) return r;
                if (r?.args !== undefined) {
                    ctx = { ...ctx, args: r.args };
                    if (!user.beforeToolCall) return r;
                    const u = await user.beforeToolCall(ctx);
                    return u ?? r;
                }
            }
            return user.beforeToolCall?.(ctx);
        },
        afterToolCall: async (ctx) => {
            let result = builtin.afterToolCall
                ? await builtin.afterToolCall(ctx)
                : undefined;
            if (result) {
                ctx = {
                    ...ctx,
                    result: result.content ?? ctx.result,
                    isError: result.isError ?? ctx.isError,
                };
            }
            if (user.afterToolCall) {
                const u = await user.afterToolCall(ctx);
                if (u) result = { ...result, ...u };
            }
            return result;
        },
        prepareNextTurn: async (ctx) => {
            let messages = ctx.messages;
            if (builtin.prepareNextTurn) {
                const r = await builtin.prepareNextTurn(ctx);
                if (r?.messages) messages = r.messages;
            }
            if (user.prepareNextTurn) {
                const u = await user.prepareNextTurn({ ...ctx, messages });
                if (u?.messages) messages = u.messages;
            }
            return messages === ctx.messages ? undefined : { messages };
        },
        shouldStopAfterTurn: async (ctx) => {
            if (builtin.shouldStopAfterTurn) {
                const stop = await builtin.shouldStopAfterTurn(ctx);
                if (stop === true) return true;
            }
            return user.shouldStopAfterTurn?.(ctx);
        },
    };
}
