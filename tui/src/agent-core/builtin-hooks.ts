// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentLoopHooks object, and composes them with any
// user-supplied hooks via the typed HookBus so both run.

import { BudgetTracker } from "./budget.ts";
import { GuardEngine } from "./guards.ts";
import { HookBus } from "./hook-bus.ts";
import {
	COMPACTION_TARGET_FRACTION,
	compactMessagesForContext,
	estimateChatPayloadTokens,
	microCompactMessages,
} from "./messages.ts";
import { getTodos } from "./tools/todo-write.ts";
import type { AgentConfig, AgentLoopHooks } from "./types.ts";

// Proactive compaction triggers when the payload exceeds this fraction of the
// context window (higher than the post-compaction target so it fires before the
// window is actually full).
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
				targetTokens: Math.floor(max * COMPACTION_TARGET_FRACTION),
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

	// Blocked patterns: text indicating the model is stuck, not just
	// choosing to stop. When detected, skip the todo nudge so the loop
	// ends cleanly instead of wasting turns.
	const BLOCKED_PATTERNS = [
		"i can't",
		"i'm unable",
		"i don't have",
		"cannot access",
		"i am unable",
		"i do not have",
		"not available",
		"no access",
		"i don't have access",
		"i cannot",
		"unable to",
		"out of my",
		"beyond my",
		"i do not have access",
	];

	function isBlocked(assistantText: string): boolean {
		const lower = assistantText.toLowerCase();
		return BLOCKED_PATTERNS.some((p) => lower.includes(p));
	}

	// Pi-style follow-up: if the model stops (no tool calls) while its own
	// todo list still has unfinished items, nudge it to keep working. The loop
	// caps total continuations, so this cannot run away.
	const continuationEnabled = config.continuationEnabled !== false;
	if (continuationEnabled) {
		hooks.getFollowUpMessages = ({
			messages: _messages,
			assistantText,
			continuationCount,
			maxContinuations,
		}) => {
			// Skip if model appears blocked by external constraints.
			if (isBlocked(assistantText)) return undefined;

			// Don't nudge on the last allowed continuation — let the loop
			// end cleanly instead of wasting the final turn.
			if (continuationCount >= maxContinuations - 1) return undefined;

			const todos = getTodos();
			if (!todos.length) return undefined;
			const remaining = todos.filter((t) => t.status !== "completed");
			if (!remaining.length) return undefined;
			const next =
				remaining.find((t) => t.status === "in_progress") ?? remaining[0];
			return [
				{
					role: "user",
					content:
						`You still have ${remaining.length} unfinished todo item(s). ` +
						`Continue working — next: ${next.content}. ` +
						"Use your tools to make progress, and mark items completed as you finish. " +
						"If you are truly blocked or done, say so explicitly and stop.",
				},
			];
		};
	}

	return hooks;
}

/** A named layer of hooks registered into the shared bus, in order. */
export interface HookLayer {
	source: string;
	hooks: AgentLoopHooks | undefined;
}

// Compose ordered hook layers via one typed HookBus. Earlier layers run first
// within each event's reducer and later layers see their output — so the
// canonical order is built-ins → harness queues → user. Returns a single
// AgentLoopHooks the agent loop consumes unchanged.
//
// The bus owns error isolation: a thrown handler is skipped (errorMode
// "continue") and reported via onError, so the loop's call sites don't need
// their own try/catch.
export function composeHooks(
	layers: HookLayer[],
	onError?: (error: Error, event: string, source?: string) => void,
	onHookEvent?: (event: string, ctx: unknown) => void,
): AgentLoopHooks {
	const bus = new HookBus({ onError });
	if (onHookEvent) bus.observe(onHookEvent);
	for (const layer of layers) {
		if (layer.hooks) bus.register(layer.hooks, { source: layer.source });
	}
	return bus.toHooks();
}
