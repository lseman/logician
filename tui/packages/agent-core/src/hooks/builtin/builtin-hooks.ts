// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentHooks object, and composes them with any
// user-supplied hooks via the typed HookBus so both run.

import { BudgetTracker } from "./budget.ts";
import { ThinkingLoopDetector } from "../../core/thinking-loop-detector.ts";
import {
	recordBashMutations,
	recordFileBeforeWrite,
	snapshotBeforeBash,
	type WorkspaceSnapshot,
} from "../../core/file-checkpoints.ts";
import type { LoopDetector } from "../../core/loop-detector.ts";
import { HookBus } from "../native/hook-bus.ts";
import {
	COMPACTION_TARGET_FRACTION,
	estimateChatPayloadTokens,
} from "../../core/messages.ts";
import { compactToFit } from "../../compaction/compaction.ts";
import { getTaskStatus } from "../../core/task-status-state.ts";
import { getTasks } from "../../core/todo-state.ts";
import type {
	AgentConfig,
	AgentHooks,
	CompactableMessage,
	Message,
} from "../../core/types.ts";

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
	// Loop detector instance for guard integration (merged from GuardEngine).
	loopDetector: LoopDetector;
	// Typed event emitter for structured events (optional).
	eventBus?: { emit: (event: Record<string, unknown>) => void };
}

// Build the default safeguard hooks. Returns undefined per-event when a
// safeguard is disabled so composition can skip it cleanly.
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentHooks {
	const { config, loopDetector } = deps;
	// Tool guards (duplicate + failure-loop, merged from GuardEngine). Default
	// OFF — matching pi's trust-model approach. Guards that block tools after 3
	// failures/dupe calls force the model to switch strategies mid-task, often
	// causing more looping. Enable only when debugging specific failure loops.
	const guardThresholds =
		config.guardsEnabled === true
			? {
					duplicateThreshold: config.duplicateToolThreshold,
					failureThreshold: config.toolFailureLoopThreshold,
				}
			: undefined;
	// Budget-based early stop is opt-in: it can cut off a legitimate multi-step
	// run (e.g. one following a todo list) when per-turn token growth is small.
	const budgetEnabled = config.budgetStopEnabled === true;
	// Thinking loop detection: detects meta-reasoning spirals where the model
	// keeps thinking without taking action. Default ON.
	const thinkingLoopEnabled = config.thinkingLoopDetectionEnabled ?? true;
	// Proactive compaction: default ON but aggressive (80% window). Can lose
	// context mid-task. Consider disabling for long-running tasks.
	const compactionEnabled = config.proactiveCompactionEnabled !== false;

	const budget = budgetEnabled ? new BudgetTracker() : null;

	const fraction =
		config.proactiveCompactionFraction ?? DEFAULT_COMPACTION_FRACTION;
	let lastCompactionTurn = -COMPACTION_COOLDOWN_TURNS;

	const thinkingLoopDetector = thinkingLoopEnabled
		? new ThinkingLoopDetector({
				minThinkingLength: config.thinkingLoopMinThinkingLength,
				thinkingOnlyThreshold: config.thinkingLoopThinkingOnlyThreshold,
				escalationRatio: config.thinkingLoopEscalationRatio,
				maxTotalThinkingTokens: config.thinkingLoopMaxTotalThinkingTokens,
				metaReasoningThreshold: config.thinkingLoopMetaReasoningThreshold,
			})
		: null;

	const hooks: AgentHooks = {};

	// Pre-bash workspace snapshots keyed by tool call id, so the afterToolCall
	// hook can diff and record the paths the command mutated.
	const bashSnapshots = new Map<string, WorkspaceSnapshot | null>();

	hooks.beforeToolCall = ({ toolCall, args }) => {
		// Snapshot pre-write state for /rewind: file tools record the target
		// path directly; bash records a workspace tree to diff afterwards.
		if (toolCall.name === "write_file" || toolCall.name === "edit_file") {
			const p = args.path ?? args.file_path ?? args.filename;
			if (typeof p === "string" && p) {
				recordFileBeforeWrite(p, config.cwd);
			}
		}
		if (toolCall.name === "bash") {
			bashSnapshots.set(toolCall.id, snapshotBeforeBash(config.cwd));
		}
		if (guardThresholds) {
			const decision = loopDetector.checkToolCall(
				toolCall.name,
				JSON.stringify(args),
			);
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
		if (toolCall.name === "bash" && bashSnapshots.has(toolCall.id)) {
			recordBashMutations(bashSnapshots.get(toolCall.id) ?? null);
			bashSnapshots.delete(toolCall.id);
		}
		if (guardThresholds && isError) {
			loopDetector.recordFailure(toolCall.name, toolCall.arguments, result);
		}
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
			const result = compactToFit(messages as CompactableMessage[], {
				triggerTokens: max * fraction,
				settings: {
					contextWindow: Math.round(max * COMPACTION_TARGET_FRACTION * 1.5),
				},
			});
			lastCompactionTurn = iteration;
			return result.changed
				? { messages: result.messages as Message[] }
				: undefined;
		};
	}

	if (budget) {
		hooks.shouldStopAfterTurn = ({ messages }) => {
			const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
			return budget.shouldStop(tokens);
		};
	}

	// Thinking loop detection: record each LLM response via afterProviderResponse,
	// then check for thinking loops on shouldStopAfterTurn.
	if (thinkingLoopDetector) {
		hooks.afterProviderResponse = ({
			content,
			toolCallCount,
			iteration,
			usageTokens,
		}) => {
			const diagnostic = thinkingLoopDetector.recordTurn(
				content ?? "",
				toolCallCount,
				iteration,
				usageTokens,
			);
			if (diagnostic) {
				// Extract strategy from diagnostic for the event
				let strategy:
					| "thinking_only"
					| "escalation"
					| "meta_reasoning"
					| "budget_exhausted" = "thinking_only";
				if (
					diagnostic.includes("escalation") ||
					diagnostic.includes("spiral")
				) {
					strategy = "escalation";
				} else if (diagnostic.includes("meta-reasoning")) {
					strategy = "meta_reasoning";
				} else if (diagnostic.includes("budget")) {
					strategy = "budget_exhausted";
				}
				const event = {
					type: "thinking_loop_detected" as const,
					message: diagnostic,
					strategy,
					iteration,
				};
				deps.eventBus?.emit(event);
			}
		};

		// Merge into shouldStopAfterTurn alongside budget check
		const prevShouldStop = hooks.shouldStopAfterTurn;
		hooks.shouldStopAfterTurn = ({ messages, iteration }) => {
			const budgetResult = prevShouldStop?.({
				messages,
				iteration,
				hadToolCalls: false,
			});
			if (budgetResult === true) return true;
			return thinkingLoopDetector?.getDiagnostic() !== null;
		};
	}

	// Pi-style follow-up: if the model stops (no tool calls) while its own
	// todo list still has unfinished items, nudge it to keep working. Also
	// auto-continues on a truncated response (stopReason="length") regardless
	// of todos — a length-stop is never a real completion. The loop caps total
	// continuations, so this cannot run away.
	// Default OFF — matching pi's trust-model approach. Todo nudges often
	// cause the model to retry the same failing approach.
	const continuationEnabled = config.continuationEnabled === true;
	if (continuationEnabled) {
		hooks.getFollowUpMessages = ({ assistantText, stopReason }) => {
			// Structured stop beats everything: a task_status call this run is
			// an explicit done/blocked declaration — never nudge past it.
			if (getTaskStatus()) return undefined;

			// Length-stop = provider truncated the response mid-output. The model
			// did not choose to stop; always continue so it can finish its thought.
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

			const tasks = getTasks();
			const remaining = tasks.filter(
				(t) => t.status !== "completed" && t.status !== "deleted",
			);
			// Nudge whenever tasks remain — the only clean exits are: no remaining
			// tasks (model used the todo tool correctly) or task_status (checked above).
			if (!remaining.length) return undefined;

			// Detect whether the assistant is circling — repeating the same
			// approach without progress. When detected, the nudge changes tone
			// to force a strategy shift rather than blind continuation.
			const isCircling = detectsCircling(assistantText);

			// Build adaptive nudge text based on circling detection. This hook
			// only decides content — the loop continues until shouldStopAfterTurn
			// returns true.
			const next =
				remaining.find((t) => t.status === "in_progress") ?? remaining[0];

			let content: string;
			if (isCircling) {
				// Circling detected — force a strategy shift. This is the most
				// important case: the model has admitted (explicitly or implicitly)
				// that it's retrying the same thing. The nudge must break the
				// pattern, not reinforce it.
				content =
					"You appear to be circling — retrying the same approach without progress. " +
					"Stop and assess: what have you actually tried so far? " +
					"What specifically failed or didn't work? " +
					"You need to try a different approach, not just repeat. " +
					`Remaining items: ${remaining.map((t) => `#${t.id} ${t.subject}`).join(", ")}. ` +
					"If you're truly stuck, explain why and stop.";
			} else {
				// Standard nudge — context-rich. Include the next task and
				// remind the model it can use any tool.
				content =
					`You still have ${remaining.length} unfinished task(s). ` +
					`Continue working — next: #${next.id} ${next.subject}. ` +
					"Use the todo tool to track progress: create tasks, mark them in_progress before working, and completed when done. " +
					"Do not skip calling the todo tool — the system only knows you finished via that tool call. " +
					"If you are truly blocked or done, say so explicitly and stop.";
			}

			return [{ role: "user", content }];
		};
	}

	return hooks;
}

// Patterns that suggest the model is circling — retrying the same approach
// without success. These are broader than stop declarations because we want
// to escalate the nudge tone (not skip the nudge) when circling is detected.
// Stricter patterns to avoid false positives on legitimate multi-step work.
const CIRCLING_PATTERNS = [
	// Future retry intent without evidence of a changed strategy.
	/\b(?:i\s+will|i'll)\s+(?:try|attempt)(?:\s+to)?\b/i,
	/\b(?:let\s+me|i(?:'m|\s+am)\s+going\s+to)\s+(?:try|attempt)\b/i,
	// A failed attempt followed by an explicit failure clause.
	/\bi\s+(?:tried|attempted)\b.*\b(?:but|however)\b.*\b(?:did(?:n't| not)\s+work|failed|unable)\b/i,
	/\bi\s+(?:tried|attempted)\b.*\b(?:again|next|yet)\b/i,
	// "I'll try again" (no X) — bare retry intent without specifying a new approach.
	/\bi(?:\s+will|'ll|ll)\s+(?:try again|attempt again)\b/i,
	// "I tried X again" — explicit past retry with "again".
	/\bi\s+(?:tried|attempted)\s+.*\b(again|yet)\b/i,
	// "Let me try again" (bare) — retry without new approach.
	/\blet\s+me\s+(?:try again|attempt again)\b/i,
	// "I've tried X again" — past retry with "again".
	/\b(i'|ve|I've)\s+(?:tried|attempted)\s+.*\b(again|yet)\b/i,
	// "cannot/can't X but try/attempt" — failed then retrying.
	/\b(cannot|can't|unable)\s+.*\b(but\s+|however\s+|instead\s+)\b.*\b(try|attempt|do|make|go)\b/i,
];

// Inspects the full text (not just the tail) since circling patterns appear anywhere.
export function detectsCircling(assistantText: string): boolean {
	if (!assistantText || assistantText.trim().length < 10) return false;
	const lower = assistantText.toLowerCase();
	return CIRCLING_PATTERNS.some((re) => re.test(lower));
}

/** A named layer of hooks registered into the shared bus, in order. */
export interface HookLayer {
	source: string;
	hooks: AgentHooks | undefined;
}

// Compose ordered hook layers via one typed HookBus. Earlier layers run first
// within each event's reducer and later layers see their output — so the
// canonical order is built-ins → harness queues → user. Returns a single
// AgentHooks object the runner consumes unchanged.
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
): AgentHooks {
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
