// ── Built-in loop hooks ────────────────────────────────────────────────────
// Constructs the default safeguard hooks (tool-call guards, budget stop,
// proactive compaction) as a single AgentHooks object.
//
// Scoped port of the pre-restructuring hooks/builtin/builtin-hooks.ts: the
// intervention-controller/task-status/todo/RTK-proxy/continuation-nudge
// features are dropped here since their backing modules (core/*, tasks/*,
// guards/response-patterns.ts) were removed by this restructuring and
// harness/utils/agent-loop.ts's own header notes those features are
// intentionally not ported to the new loop. Tool-call guards, proactive
// compaction, and budget-stop are the subset this module's one caller
// (env/extension-runtime.ts) actually wires up.

import type {
	AgentConfig,
	AgentHooks,
	CompactableMessage,
	Message,
} from "../../types/index.ts";
import { compactToFit } from "../compaction/compaction.ts";
import { resolveExecutionPolicy } from "../env/agent-settings.ts";
import {
	COMPACTION_TARGET_FRACTION,
	estimateChatPayloadTokens,
} from "../messages.ts";
import {
	recordBashMutations,
	recordFileBeforeWrite,
	snapshotBeforeBash,
	type WorkspaceSnapshot,
} from "./file-checkpoints.ts";
import type { LoopDetector } from "./guards/loop-detector.ts";

const DEFAULT_DIMINISHING_FLOOR = 500;
const DEFAULT_MIN_CONTINUATIONS = 3;

export interface BudgetTrackerOptions {
	diminishingFloor?: number;
	minContinuations?: number;
}

/**
 * Diminishing-returns stop: detect when the agent keeps emitting tokens
 * across turns but is no longer making meaningful progress.
 */
export class BudgetTracker {
	private diminishingFloor: number;
	private minContinuations: number;

	private turns = 0;
	private lastTotalTokens = 0;
	private lastDelta = Number.POSITIVE_INFINITY;

	constructor(options: BudgetTrackerOptions = {}) {
		this.diminishingFloor =
			options.diminishingFloor ?? DEFAULT_DIMINISHING_FLOOR;
		this.minContinuations =
			options.minContinuations ?? DEFAULT_MIN_CONTINUATIONS;
	}

	shouldStop(totalTokens: number): boolean {
		this.turns++;
		const delta = totalTokens - this.lastTotalTokens;
		const stalled =
			this.turns > this.minContinuations &&
			delta < this.diminishingFloor &&
			this.lastDelta < this.diminishingFloor;
		this.lastDelta = delta;
		this.lastTotalTokens = totalTokens;
		return stalled;
	}
}

// Proactive compaction triggers when the payload exceeds this fraction of the
// context window (higher than the post-compaction target so it fires before
// the window is actually full).
const DEFAULT_COMPACTION_FRACTION = 0.8;
// Don't run proactive compaction every turn — cooldown in turns.
export const COMPACTION_COOLDOWN_TURNS = 3;

export interface BuiltinHookDeps {
	config: AgentConfig;
	contextWindowTokens: () => number | undefined;
	toolDefs: () => Record<string, unknown>[];
	loopDetector?: LoopDetector;
	emitEvent?: (event: { type: string; [key: string]: unknown }) => void;
	/** Reused across hook rebuilds — BudgetTracker compares consecutive turns. */
	budget?: BudgetTracker;
	/** Boxed so callers that rebuild hooks mid-run can share and mutate it. */
	compactionCooldown?: { lastTurn: number };
}

/** Build the default safeguard hooks. */
export function buildBuiltinHooks(deps: BuiltinHookDeps): AgentHooks {
	const { config, loopDetector } = deps;
	const executionPolicy = resolveExecutionPolicy(
		config.executionProfile ?? "minimal",
	);

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
	const budgetEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		config.budgetStopEnabled === true;
	const compactionEnabled = config.proactiveCompactionEnabled !== false;

	const budget = budgetEnabled ? (deps.budget ?? new BudgetTracker()) : null;

	const fraction =
		config.proactiveCompactionFraction ?? DEFAULT_COMPACTION_FRACTION;
	const compactionCooldown = deps.compactionCooldown ?? {
		lastTurn: -COMPACTION_COOLDOWN_TURNS,
	};

	const hooks: AgentHooks = {};

	// Pre-bash workspace snapshots keyed by tool call id, so afterToolCall can
	// diff and record the paths the command mutated.
	const bashSnapshots = new Map<string, WorkspaceSnapshot | null>();

	hooks.beforeToolCall = ({ toolCall, args, iteration }) => {
		if (toolCall.name === "write_file" || toolCall.name === "edit_file") {
			const p = args.path ?? args.file_path ?? args.filename;
			if (typeof p === "string" && p) {
				recordFileBeforeWrite(p, config.cwd);
			}
		}
		if (toolCall.name === "bash") {
			bashSnapshots.set(toolCall.id, snapshotBeforeBash(config.cwd));
		}
		if (guardThresholds && loopDetector) {
			const decision = loopDetector.checkToolCall(
				toolCall.name,
				JSON.stringify(args),
			);
			if (decision.block) {
				deps.emitEvent?.({
					type: "harness_intervention",
					kind: "loop",
					cause: decision.guard ?? "duplicate",
					detector: "tool_call_guard",
					message: decision.message ?? "",
					iteration,
				});
				return { content: decision.message, isError: true };
			}
		}
		return undefined;
	};

	hooks.afterToolCall = ({ toolCall, result, isError }) => {
		if (toolCall.name === "bash" && bashSnapshots.has(toolCall.id)) {
			recordBashMutations(bashSnapshots.get(toolCall.id) ?? null);
			bashSnapshots.delete(toolCall.id);
		}
		if (isError && guardThresholds && loopDetector) {
			loopDetector.recordFailure(
				toolCall.name,
				JSON.stringify(toolCall.arguments),
				result,
			);
		} else if (!isError && guardThresholds && loopDetector) {
			loopDetector.recordSuccess(
				toolCall.name,
				JSON.stringify(toolCall.arguments),
			);
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
		hooks.shouldStopAfterTurn = ({ messages }) => {
			const tokens = estimateChatPayloadTokens(messages, deps.toolDefs());
			return budget.shouldStop(tokens);
		};
	}

	return hooks;
}
