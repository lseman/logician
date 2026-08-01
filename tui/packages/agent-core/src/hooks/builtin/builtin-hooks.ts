// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentHooks object, and composes them with any
// user-supplied hooks via the typed HookBus so both run.

import { spawnSync } from "node:child_process";
import { BudgetTracker } from "./budget.ts";
import { ThinkingLoopDetector } from "../../agent/guards/thinking-loop-detector.ts";
import {
	recordBashMutations,
	recordFileBeforeWrite,
	snapshotBeforeBash,
	type WorkspaceSnapshot,
} from "../../agent/file-checkpoints.ts";
import type { LoopDetector } from "../../agent/guards/loop-detector.ts";
import {
	awaitsUserInput,
	detectsCircling,
} from "../../agent/guards/response-patterns.ts";
import { HookBus } from "../native/hook-bus.ts";
import {
	COMPACTION_TARGET_FRACTION,
	estimateChatPayloadTokens,
} from "../../agent/messages.ts";
import { compactToFit } from "../../compaction/compaction.ts";
import { getTaskStatus } from "../../agent/tasks/task-status-state.ts";
import { getTasks } from "../../agent/tasks/todo-state.ts";
import { resolveExecutionPolicy } from "../../agent/execution-policy.ts";
import type {
	AgentConfig,
	AgentHooks,
	CompactableMessage,
	Message,
} from "../../agent/types.ts";

// Proactive compaction triggers when the payload exceeds this fraction of the
// context window (higher than the post-compaction target so it fires before the
// window is actually full).
const DEFAULT_COMPACTION_FRACTION = 0.8;
// Don't run proactive compaction every turn — cooldown in turns.
const COMPACTION_COOLDOWN_TURNS = 3;
const RTK_REWRITE_TIMEOUT_MS = 2_000;

/**
 * Ask RTK's own command registry whether a shell command should be proxied.
 * Any missing binary, unsupported command, timeout, or rewrite failure falls
 * back to the original command so enabling RTK can never block execution.
 */
export function rewriteCommandWithRtk(command: string): string {
	const result = spawnSync("rtk", ["rewrite", command], {
		encoding: "utf8",
		timeout: RTK_REWRITE_TIMEOUT_MS,
		maxBuffer: 1024 * 1024,
		windowsHide: true,
	});
	if (result.error || (result.status !== 0 && result.status !== 3)) {
		return command;
	}
	const rewritten = result.stdout.trim();
	return rewritten || command;
}

export interface BuiltinHookDeps {
	config: AgentConfig;
	contextWindowTokens: () => number | undefined;
	// Tool definitions for accurate payload token estimates.
	toolDefs: () => Record<string, unknown>[];
	// Loop detector instance for guard integration (merged from GuardEngine).
	loopDetector: LoopDetector;
	// Typed event emitter for structured events (optional).
	eventBus?: { emit: (event: { type: string; [key: string]: unknown }) => void };
}

// Build the default safeguard hooks. Returns undefined per-event when a
// safeguard is disabled so composition can skip it cleanly.
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentHooks {
	const { config, loopDetector } = deps;
	const executionPolicy = resolveExecutionPolicy(config.executionProfile);
	// Tool guards (duplicate + failure-loop, merged from GuardEngine).
	// Duplicate-call detection defaults ON: blocking exact same-args repeats
	// (e.g. re-reading the same file over and over) is safe to force a
	// strategy change. Failure-loop blocking stays default OFF — it can cut
	// off a legitimate retry-with-variation sequence, matching pi's
	// trust-model approach. `guardsEnabled` is the umbrella toggle for both.
	const duplicateGuardOn =
		executionPolicy.embeddedPoliciesEnabled &&
		(config.guardsEnabled === true || config.duplicateGuardEnabled !== false);
	const failureGuardOn =
		executionPolicy.embeddedPoliciesEnabled &&
		(config.guardsEnabled === true || config.failureGuardEnabled === true);
	const guardThresholds =
		duplicateGuardOn || failureGuardOn
			? {
					duplicateThreshold: duplicateGuardOn
						? config.duplicateToolThreshold
						: 0,
					failureThreshold: failureGuardOn
						? config.toolFailureLoopThreshold
						: 0,
				}
			: undefined;
	// Budget-based early stop is opt-in: it can cut off a legitimate multi-step
	// run (e.g. one following a todo list) when per-turn token growth is small.
	const budgetEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		config.budgetStopEnabled === true;
	// Thinking loop detection: detects meta-reasoning spirals where the model
	// keeps thinking without taking action. Default ON.
	const thinkingLoopEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		(config.thinkingLoopDetectionEnabled ?? true);
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

	hooks.beforeToolCall = ({ toolCall, args, iteration }) => {
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
		// RTK proxy: let RTK's registry selectively rewrite supported commands.
		if (toolCall.name === "bash" && config.rtkProxyEnabled === true) {
			if (args.command !== undefined && typeof args.command === "string") {
				args.command = rewriteCommandWithRtk(args.command);
			}
			if (args.commands !== undefined && Array.isArray(args.commands)) {
				args.commands = args.commands.map((entry: unknown) => {
					if (typeof entry === "object" && entry !== null) {
						const obj = entry as Record<string, unknown>;
						if (typeof obj.command === "string") {
							obj.command = rewriteCommandWithRtk(obj.command);
						}
					}
					return entry;
				});
			}
		}
		if (guardThresholds) {
			const decision = loopDetector.checkToolCall(
				toolCall.name,
				JSON.stringify(args),
			);
			if (decision.block) {
				deps.eventBus?.emit({
					type: "guard_triggered",
					guard: decision.guard ?? "duplicate",
					message: decision.message ?? "",
					toolName: toolCall.name,
					iteration,
				});
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
		if (
			executionPolicy.embeddedPoliciesEnabled &&
			toolCall.name === "task_status" &&
			!isError
		) {
			return { terminate: true };
		}
		return undefined;
	};

	if (compactionEnabled) {
		hooks.prepareNextTurn = async ({ messages, iteration }) => {
			const max = deps.contextWindowTokens();
			if (!max || max <= 0) return undefined;
			if (iteration - lastCompactionTurn < COMPACTION_COOLDOWN_TURNS) {
				return undefined;
			}
			// Shared ladder: estimate → micro → full-if-still-over. Fires at
			// `fraction` of the window, targets COMPACTION_TARGET_FRACTION.
			const result = await compactToFit(messages as CompactableMessage[], {
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
	const continuationEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		config.continuationEnabled === true;
	if (continuationEnabled) {
		hooks.getFollowUpMessages = ({ assistantText, stopReason, iteration }) => {
			// Structured stop beats everything: a task_status call this run is
			// an explicit done/blocked declaration — never nudge past it.
			if (getTaskStatus()) return undefined;

			// A final question is an explicit transfer of control to the user.
			// Pending todos do not authorize the harness to answer it itself.
			if (awaitsUserInput(assistantText)) return undefined;

			// Length-stop = provider truncated the response mid-output. The model
			// did not choose to stop; always continue so it can finish its thought.
			if (stopReason === "length") {
				const message =
					"[continuation-nudge:length] Your previous response was cut off because it reached the output limit. " +
					"Please continue exactly where you left off — do not repeat what you already wrote.";
				deps.eventBus?.emit({
					type: "guard_triggered",
					guard: "continuation_nudge",
					message,
					iteration,
				});
				return [{ role: "user", content: message }];
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
					"[continuation-nudge:circling] You appear to be circling — retrying the same approach without progress. " +
					"Stop and assess: what have you actually tried so far? " +
					"What specifically failed or didn't work? " +
					"You need to try a different approach, not just repeat. " +
					`Remaining items: ${remaining.map((t) => `#${t.id} ${t.subject}`).join(", ")}. ` +
					"If you're truly stuck, explain why and stop.";
			} else {
				// Standard nudge — context-rich. Include the next task and
				// remind the model it can use any tool.
				content =
					`[continuation-nudge:todo] You still have ${remaining.length} unfinished task(s). ` +
					`Continue working — next: #${next.id} ${next.subject}. ` +
					"Use the todo tool to track progress: create tasks, mark them in_progress before working, and completed when done. " +
					"Do not skip calling the todo tool — the system only knows you finished via that tool call. " +
					"If you are truly blocked or done, say so explicitly and stop.";
			}

			deps.eventBus?.emit({
				type: "guard_triggered",
				guard: "continuation_nudge",
				message: content,
				iteration,
			});
			return [{ role: "user", content }];
		};
	}

	return hooks;
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
