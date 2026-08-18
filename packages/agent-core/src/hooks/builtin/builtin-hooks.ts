// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentHooks object.
//
// Tool-call guards (duplicate + failure-loop) are powered by LoopDetector,
// the harness's single live instance.

import { spawnSync } from "node:child_process";
import { resolveExecutionPolicy } from "../../core/execution-policy.ts";
import {
	recordBashMutations,
	recordFileBeforeWrite,
	snapshotBeforeBash,
	type WorkspaceSnapshot,
} from "../../core/file-checkpoints.ts";
import { HarnessInterventionController } from "../../core/intervention-controller.ts";
import {
	COMPACTION_TARGET_FRACTION,
	estimateChatPayloadTokens,
} from "../../core/messages.ts";
import type { LoopDetector } from "../../guards/loop-detector.ts";
import { awaitsUserInput } from "../../guards/response-patterns.ts";
import {
	getTaskStatus,
	recordTaskStatus,
} from "../../tasks/task-status-state.ts";
import { getTasks } from "../../tasks/todo-state.ts";
import type {
	AgentConfig,
	AgentHooks,
	CompactableMessage,
	Message,
} from "../../types/index.ts";
import { compactToFit } from "../../compaction/compaction.ts";
import { BudgetTracker } from "./budget.ts";

// Proactive compaction triggers when the payload exceeds this fraction of the
// context window (higher than the post-compaction target so it fires before the
// window is actually full).
const DEFAULT_COMPACTION_FRACTION = 0.8;
// Don't run proactive compaction every turn — cooldown in turns.
export const COMPACTION_COOLDOWN_TURNS = 3;
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
	// LoopDetector instance powering the duplicate/failure-loop tool-call guards.
	loopDetector?: LoopDetector;
	// Sink for structured events this module raises (optional). Distinct from
	// extensions/event-bus.ts's cross-extension pub/sub — this is a single
	// emit callback, forwarded by the harness to its own AgentEvent stream.
	emitEvent?: (event: { type: string; [key: string]: unknown }) => void;
	// Escalation state for guard/continuation/budget interventions. Callers
	// that rebuild hooks mid-run (e.g. a config refresh between loop
	// iterations) MUST pass the same instance across rebuilds — escalation
	// (attempt counts) and `recordProgress()`'s incident-clearing only work
	// across repeated detections if this outlives a single hook build.
	interventions?: HarnessInterventionController;
	// Diminishing-returns budget-stop tracker, reused across rebuilds for the
	// same reason as `interventions` — BudgetTracker compares consecutive
	// turns, so a fresh instance every rebuild can never trigger. Only
	// consulted while the feature is actually enabled (config + execution
	// policy); a fresh instance is created locally when omitted.
	budget?: BudgetTracker;
	// Proactive-compaction cooldown, in loop iterations since the last
	// compaction. Boxed in an object (not a bare number) so callers that
	// rebuild hooks mid-run can share and mutate it across rebuilds.
	compactionCooldown?: { lastTurn: number };
}

// Build the default safeguard hooks. Returns undefined per-event when a
// safeguard is disabled so composition can skip it cleanly.
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentHooks {
	const { config, loopDetector } = deps;
	const executionPolicy = resolveExecutionPolicy(config.executionProfile);
	const interventions =
		deps.interventions ?? new HarnessInterventionController();
	const emitIntervention = (
		input: Parameters<HarnessInterventionController["record"]>[0],
	) => {
		const intervention = interventions.record(input);
		deps.emitEvent?.({ type: "harness_intervention", ...intervention });
		return intervention;
	};
	// Tool guards (duplicate + failure-loop) via LoopDetector.
	// Duplicate-call detection defaults ON: blocking exact same-args repeats
	// (e.g. re-reading the same file over and over) is safe to force a
	// strategy change. Failure-loop blocking stays default OFF — it can cut
	// off a legitimate retry-with-variation sequence, matching pi's
	// trust-model approach. `guardsEnabled` is the umbrella toggle for both.
	const duplicateGuardOn =
		executionPolicy.embeddedPoliciesEnabled &&
		config.guardsEnabled !== false &&
		(config.guardsEnabled === true || config.duplicateGuardEnabled !== false);
	const failureGuardOn =
		executionPolicy.embeddedPoliciesEnabled &&
		config.guardsEnabled !== false &&
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
	// Proactive compaction: default ON but aggressive (80% window). Can lose
	// context mid-task. Consider disabling for long-running tasks.
	const compactionEnabled = config.proactiveCompactionEnabled !== false;

	const budget = budgetEnabled ? (deps.budget ?? new BudgetTracker()) : null;

	const fraction =
		config.proactiveCompactionFraction ?? DEFAULT_COMPACTION_FRACTION;
	const compactionCooldown = deps.compactionCooldown ?? {
		lastTurn: -COMPACTION_COOLDOWN_TURNS,
	};

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
		if (guardThresholds && loopDetector) {
			const decision = loopDetector.checkToolCall(
				toolCall.name,
				JSON.stringify(args),
			);
			if (decision.block) {
				emitIntervention({
					kind: "loop",
					cause: decision.guard ?? "duplicate",
					detector: "tool_call_guard",
					message: decision.message ?? "",
					iteration,
				});
				recordTaskStatus({
					status: "blocked",
					summary: decision.message ?? "Blocked by guard.",
					ts: Date.now(),
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
		// Record failure/success against the duplicate/failure-loop guard's
		// LoopDetector, gated on guardThresholds (the blocking thresholds).
		if (isError && guardThresholds && loopDetector) {
			loopDetector.recordFailure(toolCall.name, toolCall.arguments, result);
		} else if (!isError && guardThresholds && loopDetector) {
			loopDetector.recordSuccess(toolCall.name, toolCall.arguments);
			interventions.recordProgress();
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
			if (iteration - compactionCooldown.lastTurn < COMPACTION_COOLDOWN_TURNS) {
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
			compactionCooldown.lastTurn = iteration;
			if (result.changed) {
				deps.emitEvent?.({
					type: "compaction",
					reason: "threshold",
					tokensBefore: estimateChatPayloadTokens(messages, deps.toolDefs()),
					tokensAfter: result.tokensAfter,
				});
			}
			return result.changed
				? { messages: result.messages as Message[] }
				: undefined;
		};
	}

	if (budget) {
		hooks.shouldStopAfterTurn = ({ messages, iteration }) => {
			const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
			const stopped = budget.shouldStop(tokens);
			if (stopped) {
				const message =
					"Continuation stopped because token growth stayed below the progress threshold for two turns.";
				recordTaskStatus({
					status: "blocked",
					summary: message,
					ts: Date.now(),
				});
				emitIntervention({
					kind: "budget",
					cause: "low_progress",
					detector: "token_budget",
					message,
					iteration,
					action: "stop",
				});
			}
			return stopped;
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
				emitIntervention({
					kind: "continuation",
					cause: "length_truncation",
					detector: "builtin_continuation",
					message,
					iteration,
				});
				return [{ role: "user", content: message }];
			}

			const tasks = getTasks();
			const remaining = tasks.filter(
				t => t.status !== "completed" && t.status !== "deleted",
			);
			// Nudge whenever tasks remain — the only clean exits are: no remaining
			// tasks (model used the todo tool correctly) or task_status (checked above).
			if (!remaining.length) return undefined;

			// Build nudge text. Circling detection (regex-based) has been
			// removed — trust the model's reasoning instead.
			const next =
				remaining.find(t => t.status === "in_progress") ?? remaining[0];

			const content =
				`[continuation-nudge:todo] You still have ${remaining.length} unfinished task(s). ` +
				`Continue working — next: #${next.id} ${next.subject}. ` +
				"Use the todo tool to track progress: create tasks, mark them in_progress before working, and completed when done. " +
				"Do not skip calling the todo tool — the system only knows you finished via that tool call. " +
				"If you are truly blocked or done, say so explicitly and stop.";

			emitIntervention({
				kind: "continuation",
				cause: "unfinished_todos",
				detector: "builtin_continuation",
				message: content,
				iteration,
			});
			return [{ role: "user", content }];
		};
	}

	return hooks;
}
