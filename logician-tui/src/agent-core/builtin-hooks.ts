// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentLoopHooks object, and composes them with any
// user-supplied hooks so both run. The contract is single-handler per event,
// so composition is explicit here rather than a generic bus (see #7 in
// AGENT_IMPROVEMENTS.md for the eventual typed hook bus).

import type { AgentConfig, AgentLoopHooks } from "./types.ts";
import { HookBus } from "./hook-bus.ts";
import { GuardEngine } from "./guards.ts";
import { BudgetTracker } from "./budget.ts";
import {
    compactMessagesForContext,
    microCompactMessages,
    estimateChatPayloadTokens,
} from "./messages.ts";
import { getTodos } from "./tools/todo-write.ts";

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
    // Budget-based early stop is opt-in: it can cut off a legitimate multi-step
    // run (e.g. one following a todo list) when per-turn token growth is small.
    const budgetEnabled = config.budgetStopEnabled === true;
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

    // Pi-style continuation: if the model stops (no tool calls) while its own
    // todo list still has unfinished items, nudge it to keep working. The loop
    // caps total continuations, so this cannot run away.
    const continuationEnabled = config.continuationEnabled !== false;
    if (continuationEnabled) {
        hooks.continueAfterTurn = () => {
            const todos = getTodos();
            if (!todos.length) return undefined;
            const remaining = todos.filter((t) => t.status !== "completed");
            if (!remaining.length) return undefined;
            const next =
                remaining.find((t) => t.status === "in_progress") ??
                remaining[0];
            return {
                message:
                    `You still have ${remaining.length} unfinished todo item(s). ` +
                    `Continue working — next: ${next.content}. ` +
                    "Use your tools to make progress, and mark items completed as you finish. " +
                    "If you are truly blocked or done, say so explicitly and stop.",
            };
        };
    }

    return hooks;
}

// Compose built-in hooks with user hooks via the typed hook bus. Built-ins
// register first (so they run first within each event's reducer); user hooks
// register after and see the built-ins' output. Returns a single
// AgentLoopHooks the agent loop consumes unchanged.
export function composeHooks(
    builtin: AgentLoopHooks,
    user: AgentLoopHooks,
): AgentLoopHooks {
    const bus = new HookBus();
    bus.register(builtin, { source: "builtin" });
    bus.register(user, { source: "user" });
    return bus.toHooks();
}
