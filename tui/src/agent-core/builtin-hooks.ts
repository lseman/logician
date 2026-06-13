// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentLoopHooks object, and composes them with any
// user-supplied hooks via the typed HookBus so both run.

import { BudgetTracker } from "./budget.ts";
import { recordFileBeforeWrite } from "./file-checkpoints.ts";
import { GuardEngine } from "./guards.ts";
import { HookBus } from "./hook-bus.ts";
import {
	COMPACTION_TARGET_FRACTION,
	compactToFit,
	estimateChatPayloadTokens,
} from "./messages.ts";
import { getTaskStatus } from "./tools/task-status.ts";
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

	hooks.beforeToolCall = ({ toolCall, args }) => {
		// Snapshot a file's pre-write state for /rewind (write tools only; bash
		// mutations are out of scope — see file-checkpoints.ts).
		if (toolCall.name === "write_file" || toolCall.name === "edit_file") {
			const p = args.path ?? args.file_path ?? args.filename;
			if (typeof p === "string" && p) {
				recordFileBeforeWrite(p, config.cwd);
			}
		}
		if (guards) {
			const decision = guards.inspect(toolCall);
			if (decision.block) {
				return { content: decision.message, isError: true };
			}
		}
		return undefined;
	};

	// A task_status call is a structured stop request: terminate after its
	// batch (the loop applies the all-tools-in-batch gate, and task_status is
	// documented as a final, standalone call).
	hooks.afterToolCall = ({ toolCall, result, isError }) => {
		guards?.record(toolCall, isError, result);
		if (toolCall.name === "task_status" && !isError) {
			return { terminate: true };
		}
		return undefined;
	};

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
	// todo list still has unfinished items, nudge it to keep working. Also
	// auto-continues on a truncated response (stopReason="length") regardless
	// of todos — a length-stop is never a real completion. The loop caps total
	// continuations, so this cannot run away.
	const continuationEnabled = config.continuationEnabled !== false;
	if (continuationEnabled) {
		hooks.getFollowUpMessages = ({
			messages,
			assistantText,
			continuationCount,
			maxContinuations,
			stopReason,
		}) => {
			// Structured stop beats everything: a task_status call this run is
			// an explicit done/blocked declaration — never nudge past it.
			if (getTaskStatus()) return undefined;

			// Don't nudge on the last allowed continuation — let the loop
			// end cleanly instead of wasting the final turn.
			if (continuationCount >= maxContinuations - 1) return undefined;

			// Length-stop = provider truncated the response mid-output. The model
			// did not choose to stop; always continue so it can finish its thought.
			// Skip declaresStop check — it makes no sense for a cut-off response.
			if (stopReason === "length") {
				return [
					{
						role: "user",
						content:
							"Your previous response was cut off because it reached the output limit. " +
							"Please continue exactly where you left off — do not repeat what you already wrote.",
					},
				];
			}

			// Skip the nudge when the model explicitly declares it is done or
			// blocked. We honour the model's own stop signal rather than sniffing
			// prose anywhere in the message — see declaresStop().
			if (declaresStop(assistantText)) return undefined;

			const todos = getTodos();
			const remaining = todos.filter((t) => t.status !== "completed");
			if (!remaining.length) return undefined;

			// Detect whether the assistant is circling — repeating the same
			// approach without progress. When detected, the nudge changes tone
			// to force a strategy shift rather than blind continuation.
			const isCircling = detectsCircling(assistantText);

			// Build adaptive nudge text based on continuation count and
			// circling detection.
			const next =
				remaining.find((t) => t.status === "in_progress") ?? remaining[0];

			let content: string;
			if (isCircling) {
				// Circling detected — force a strategy shift. This is the most
				// important case: the model has admitted (explicitly or implicitly)
				// that it's retrying the same thing. The nudge must break the
				// pattern, not reinforce it.
				content =
					`You appear to be circling — retrying the same approach without progress. ` +
					`Stop and assess: what have you actually tried so far? ` +
					`What specifically failed or didn't work? ` +
					`You need to try a different approach, not just repeat. ` +
					`Remaining items: ${remaining.map((t) => t.content).join(", ")}. ` +
					`If you're truly stuck, explain why and stop.`;
			} else if (continuationCount === 0) {
				// First nudge — gentle, context-rich. Include the next item and
				// remind the model it can use any tool.
				content =
					`You still have ${remaining.length} unfinished todo item(s). ` +
					`Continue working — next: ${next.content}. ` +
					"Use your tools to make progress, and mark items completed as you finish. " +
					"If you are truly blocked or done, say so explicitly and stop.";
			} else if (continuationCount <= 2) {
				// Subsequent nudges — more directive. Signal that the model should
				// focus on the specific remaining item rather than going wide.
				content =
					`${remaining.length} todo item(s) remain${remaining.length === 1 ? "s" : ""}. ` +
					`You've been asked to continue ${continuationCount} time(s) already. ` +
					`Focus on completing: ${next.content}. ` +
					"If this item is truly done, mark it completed. " +
					"Otherwise use your tools and finish it.";
			} else {
				// Late nudges — urgent. Signal that the model is approaching the
					// continuation limit and should either finish or declare done.
				content =
					`This is continuation ${continuationCount}/${maxContinuations - 1}. ` +
					`You have ${remaining.length} todo item(s) left: ${remaining.map((t) => t.content).join(", ")}. ` +
					"Finish one and mark it completed, or declare explicitly that you are done/blocked. " +
					"If you cannot proceed, explain why and stop.";
			}

			return [{ role: "user", content }];
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
	// Stuck / circling — the model admits it can't resolve the task.
	/\bi\s*(?:'m\s+|am\s+)?(?:truly|completely|totally)\s+(?:stuck|lost|confused)\b/,
	/\bi\s+(?:don't|do not)\s+(?:know|think)\s+(?:how|what else)\s+to\s+(?:do|try|attempt|proceed|help)\b/,
	/\bi\s+can(?:'t|not)\s+(?:make\s+progress|proceed|go\s+further|resolve\s+this)\b/,
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

// Patterns that suggest the model is circling — retrying the same approach
// without success. These are broader than stop declarations because we want
// to escalate the nudge tone (not skip the nudge) when circling is detected.
const CIRCLING_PATTERNS = [
	// "I'll try again", "I'll try X", "I'll attempt X" — future retry intent.
	/\bi(?:\s+will|'ll|ll)\s+(?:try|try again|attempt|attempt again)\b/i,
	// "I tried X again", "I attempted X yet/another", past retry.
	/\bi\s+(?:tried|attempted|tried to)\s+.*\b(again|another|a different|yet|elsewhere|else|but|however|though|next)\b/i,
	// "Let me try again", "Let me try X", future retry (self-directed).
	/\blet\s+me\s+(?:try|try again|attempt)\b/i,
	// "I've tried X again", "I've attempted X yet".
	/\b(i\'|ve|I\'ve)\s+(?:tried|attempted)\s+.*\b(again|another|a different|yet|elsewhere|else)\b/i,
	// "cannot/can't X but try/attempt" — failed then retrying.
	/\b(cannot|can\'t|unable)\s+.*\b(but\s+|however\s+|instead\s+|though\s+|though\s+I)\b.*\b(try|attempt|do|make|go)\b/i,
	// "I'm going to try again", "I'm going to attempt".
	/\bi\s*'m\s+going\s+to\s+(?:try|try again|attempt)\b/i,
];

/**
 * Does the assistant's message suggest it is circling — repeating or about to
 * repeat the same approach without progress? Inspects the full text (not just
 * the tail) since circling patterns can appear anywhere.
 */
export function detectsCircling(assistantText: string): boolean {
	if (!assistantText || assistantText.trim().length < 10) return false;
	const lower = assistantText.toLowerCase();
	return CIRCLING_PATTERNS.some((re) => re.test(lower));
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
// One stuck handler must not stall every turn: any single hook handler that
// runs longer than this is skipped (and reported) like a thrown one.
const HOOK_HANDLER_TIMEOUT_MS = 60_000;

export function composeHooks(
	layers: HookLayer[],
	onError?: (error: Error, event: string, source?: string) => void,
	onHookEvent?: (event: string, ctx: unknown) => void,
): AgentLoopHooks {
	const bus = new HookBus({
		onError,
		defaultTimeoutMs: HOOK_HANDLER_TIMEOUT_MS,
	});
	if (onHookEvent) bus.observe(onHookEvent);
	for (const layer of layers) {
		if (layer.hooks) bus.register(layer.hooks, { source: layer.source });
	}
	return bus.toHooks();
}
