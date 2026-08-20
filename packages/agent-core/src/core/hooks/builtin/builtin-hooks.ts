// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (guards, budget stop, proactive
// compaction) as a single AgentHooks object.
//
// Tool-call guards (duplicate + failure-loop) are powered by LoopDetector,
// the harness's single live instance.

import { execFile } from "node:child_process";
import { compactToFit } from "../../compaction/engine.ts";
import { resetToRunCheckpoint } from "../../compaction/run-checkpoint.ts";
import type { LoopDetector } from "../../guards/loop-detector.ts";
import { decideAutonomousContinuation } from "../../policy/autonomy-policy.ts";
import { resolveExecutionPolicy } from "../../policy/execution-policy.ts";
import { HarnessInterventionController } from "../../policy/intervention-controller.ts";
import { ProgressTracker } from "../../policy/progress-tracker.ts";
import { EMPTY_TASK_LEDGER } from "../../policy/task-ledger.ts";
import {
	COMPACTION_TARGET_FRACTION,
	estimateChatPayloadTokens,
} from "../../provider/messages.ts";
import {
	recordBashMutations,
	recordFileBeforeWrite,
	snapshotBeforeBash,
	type WorkspaceSnapshot,
} from "../../session/file-checkpoints.ts";
import type { AgentConfig } from "../../types/types-config.ts";
import type {
	AgentHooks,
	CompactableMessage,
	Message,
} from "../../types/types-messages.ts";

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
export function rewriteCommandWithRtk(command: string): Promise<string> {
	return new Promise(resolve => {
		execFile(
			"rtk",
			["rewrite", command],
			{
				encoding: "utf8",
				timeout: RTK_REWRITE_TIMEOUT_MS,
				maxBuffer: 1024 * 1024,
				windowsHide: true,
			},
			(error, stdout) => {
				const exitCode =
					error && "code" in error && typeof error.code === "number"
						? error.code
						: undefined;
				if (error && exitCode !== 3) {
					resolve(command);
					return;
				}
				resolve(stdout.trim() || command);
			},
		);
	});
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
	// Evidence-based progress tracker. Shared because autonomous hooks are
	// rebuilt between turns while progress belongs to the whole run.
	progress?: ProgressTracker;
	// Proactive-compaction cooldown, in loop iterations since the last
	// compaction. Boxed in an object (not a bare number) so callers that
	// rebuild hooks mid-run can share and mutate it across rebuilds.
	compactionCooldown?: {
		lastTurn: number;
		consecutiveCompactions?: number;
	};
}

// Build the default safeguard hooks. Returns undefined per-event when a
// safeguard is disabled so composition can skip it cleanly.
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentHooks {
	const { config, loopDetector } = deps;
	const taskLedger = config.taskLedger ?? EMPTY_TASK_LEDGER;
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
	const progressStopEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		config.progressStopEnabled === true;
	// Proactive compaction: default ON but aggressive (80% window). Can lose
	// context mid-task. Consider disabling for long-running tasks.
	const compactionEnabled = config.proactiveCompactionEnabled !== false;

	const progress = progressStopEnabled
		? (deps.progress ?? new ProgressTracker())
		: null;

	const fraction =
		config.proactiveCompactionFraction ?? DEFAULT_COMPACTION_FRACTION;
	const compactionCooldown = deps.compactionCooldown ?? {
		lastTurn: -COMPACTION_COOLDOWN_TURNS,
	};

	const hooks: AgentHooks = {};

	// Pre-bash workspace snapshots keyed by tool call id, so the afterToolCall
	// hook can diff and record the paths the command mutated.
	const bashSnapshots = new Map<string, WorkspaceSnapshot | null>();

	hooks.beforeToolCall = async ({ toolCall, args, iteration }) => {
		const originalArgs = args;
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
			const rewrittenArgs = { ...args };
			if (typeof args.command === "string") {
				rewrittenArgs.command = await rewriteCommandWithRtk(args.command);
			}
			if (Array.isArray(args.commands)) {
				rewrittenArgs.commands = await Promise.all(
					args.commands.map(async (entry: unknown) => {
						if (!entry || typeof entry !== "object") return entry;
						const command = (entry as Record<string, unknown>).command;
						return typeof command === "string"
							? {
									...(entry as Record<string, unknown>),
									command: await rewriteCommandWithRtk(command),
								}
							: entry;
					}),
				);
			}
			args = rewrittenArgs;
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
				return { content: decision.message, isError: true };
			}
		}
		return args === originalArgs ? undefined : { args };
	};

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
		if (!isError && progress) {
			progress.recordToolResult(
				toolCall.name,
				toolCall.arguments,
				String(result),
			);
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
				compactionCooldown.consecutiveCompactions =
					(compactionCooldown.consecutiveCompactions ?? 0) + 1;
				deps.emitEvent?.({
					type: "compaction",
					reason: "threshold",
					tokensBefore: estimateChatPayloadTokens(messages, deps.toolDefs()),
					tokensAfter: result.tokensAfter,
				});
				const tasks = taskLedger.snapshot();
				const unfinished = tasks.some(
					task => task.status !== "completed" && task.status !== "deleted",
				);
				if (
					executionPolicy.embeddedPoliciesEnabled &&
					unfinished &&
					(compactionCooldown.consecutiveCompactions ?? 0) >= 2
				) {
					const checkpointed = resetToRunCheckpoint(
						result.messages as Message[],
						tasks,
					);
					compactionCooldown.consecutiveCompactions = 0;
					emitIntervention({
						kind: "compaction",
						cause: "structured_context_reset",
						detector: "compaction_checkpoint",
						message:
							"Repeated compaction replaced the transcript with a structured run checkpoint.",
						iteration,
						action: "recover",
					});
					return { messages: checkpointed };
				}
			}
			return result.changed
				? { messages: result.messages as Message[] }
				: undefined;
		};
	}

	if (progress) {
		hooks.shouldStopAfterTurn = ({ iteration }) => {
			const stopped = progress.shouldStop(taskLedger.snapshot());
			if (stopped) {
				const message =
					"Continuation stopped after repeated turns produced no new tool or task-state evidence.";
				emitIntervention({
					kind: "loop",
					cause: "no_progress",
					detector: "progress_tracker",
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
			const decision = decideAutonomousContinuation({
				assistantText,
				stopReason,
				tasks: taskLedger.snapshot(),
			});
			if (!decision) return undefined;

			emitIntervention({
				kind: "continuation",
				cause: decision.reason,
				detector: "builtin_continuation",
				message: decision.message,
				iteration,
			});
			return [{ role: "user", content: decision.message }];
		};
	}

	return hooks;
}
