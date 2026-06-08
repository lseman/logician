// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentLoopHooks object, and composes them with any
// user-supplied hooks via the typed HookBus so both run.

import { BudgetTracker } from "./budget.ts";
import { GuardEngine } from "./guards.ts";
import { HookBus } from "./hook-bus.ts";
import {
	COMPACTION_TARGET_FRACTION,
	compactToFit,
	estimateChatPayloadTokens,
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
			// Shared ladder: estimate → micro → full-if-still-over. Fires at
			// `fraction` of the window, targets COMPACTION_TARGET_FRACTION.
			const result = compactToFit(messages, {
				triggerTokens: max * fraction,
				targetTokens: Math.floor(max * COMPACTION_TARGET_FRACTION),
				toolDefs: deps.toolDefs(),
			});
			lastCompactionTurn = iteration;
			return result.changed ? { messages: result.messages } : undefined;
		};
	}

	if (budget) {
		hooks.shouldStopAfterTurn = ({ messages }) => {
			const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
			return budget.shouldStop(tokens);
		};
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
			// Skip the nudge when the model explicitly declares it is done or
			// blocked. We honour the model's own stop signal (the nudge prompt
			// asks it to "say so explicitly and stop") rather than sniffing prose
			// anywhere in the message — see declaresStop().
			if (declaresStop(assistantText)) return undefined;

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

// Explicit stop declarations the model makes when done or blocked. Matched only
// against the message *tail* (its final statement, where models declare status)
// and anchored to a line/sentence start — so a mid-paragraph "I cannot stress
// enough …" no longer reads as "blocked". Patterns are intentionally narrow:
// the cost of a false negative (one wasted nudge, capped by maxContinuations) is
// far lower than a false positive (silently abandoning unfinished work).
const STOP_DECLARATIONS = [
	// Completion.
	/\b(task|work|everything|all (?:the )?(?:tasks|items|todos))\s+(?:is|are)\s+(?:now\s+)?(?:complete|completed|done|finished)\b/,
	/\b(?:i(?:'ve| have))\s+(?:now\s+)?(?:completed|finished|done)\b/,
	/\ball\s+(?:done|complete|completed|finished)\b/,
	/\bnothing\s+(?:more|else|further)\s+to\s+do\b/,
	// Blocked / lacking capability, stated as a standalone status. "I can't" /
	// "I am unable" must be followed by a stop-relevant object so an incidental
	// "I cannot stress enough …" does not read as blocked.
	/\bi\s+(?:can(?:'t|not)|am unable to)\s+(?:access|proceed|continue|complete|finish|do (?:this|that|so)|help with)\b/,
	/\bi\s+(?:do(?:n't| not)|don't)\s+have\s+(?:access|the ability|permission)\b/,
	/\b(?:unable to proceed|cannot proceed|cannot continue|i'm blocked|i am blocked)\b/,
];

/**
 * Does the assistant's message explicitly declare it is done or blocked? Only
 * the last non-empty line (capped at 240 chars) is inspected, lower-cased, so a
 * status declaration at the end of a turn is honoured while incidental phrasing
 * earlier in the message is ignored.
 */
export function declaresStop(assistantText: string): boolean {
	const lines = assistantText
		.split("\n")
		.map((l) => l.trim())
		.filter(Boolean);
	const tail = lines.at(-1)?.toLowerCase().slice(0, 240) ?? "";
	if (!tail) return false;
	return STOP_DECLARATIONS.some((re) => re.test(tail));
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
