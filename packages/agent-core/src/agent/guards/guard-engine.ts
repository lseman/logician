// ── GuardEngine ──────────────────────────────────────────────────────────
// Central orchestrator for all agent guard rails, loop detection, and recovery.
//
// Two upgrades over a flat per-detector API:
//   1. Multi-signal fusion — composite scoring from all detectors
//   2. Graduated intervention ladder — severity-based escalation
//
// Only loop_detection, thinking_loop, and output_errors are pattern-based on
// unambiguous repetition — they may escalate the ladder up to abort.
// Everything else (progress_signal, recovery_memory, hypothesis_tracker,
// goal_decomposition) scores from heuristics that can't reliably distinguish
// a stuck agent from healthy work, so they're capped at "nudge": informative,
// never an interrupt. See RELIABLE_ESCALATION_DETECTORS.
//
// Backward compatible: existing API methods work unchanged.

import type { LoopDetector } from "./loop-detector.ts";
import type { OutputGuard } from "./output-guard.ts";
import type { ThinkingLoopDetector } from "./thinking-loop-detector.ts";
import type { ProgressSignalTracker } from "./progress-signal.ts";
import type { HypothesisTracker } from "./hypothesis-tracker.ts";
import type { RecoveryMemory } from "./recovery-memory.ts";
import type { GoalDecomposer } from "./goal-decomposer.ts";
import type { EventHandler } from "../types.ts";

// ── Types ────────────────────────────────────────────────────────────────

/** Graduated intervention severity — escalation ladder. */
export type InterventionSeverity =
	| "info"      // Just logging, no intervention
	| "nudge"     // Soft nudge, agent continues
	| "restrict"  // Force reflection or strategy change
	| "pause"     // Stop the agent, require reassessment
	| "abort";    // Terminate the session

/** Action corresponding to intervention severity. */
export type InterventionAction =
	| "proceed"    // Continue normally
	| "nudge"      // Send a soft nudge
	| "reflect"    // Force reflection before continuing
	| "restrict"   // Restrict available tools/operations
	| "pause"      // Pause and require intervention
	| "abort";     // Abort the session

/** A single detector signal with score and weight. */
export interface GuardSignal {
	/** Which detector produced this signal. */
	detector: string;
	/** Signal name (e.g., "thinking_loop", "low_progress"). */
	signal: string;
	/** 0-100 score indicating severity. */
	score: number;
	/** Configurable weight for fusion (0-1). */
	weight: number;
	/** Human-readable explanation. */
	description: string;
}

/** Composite decision from multi-signal fusion. */
export interface GuardDecision {
	/** Intervention severity level. */
	severity: InterventionSeverity;
	/** Action to take. */
	action: InterventionAction;
	/** Message to show the agent/user. */
	message?: string;
	/** All contributing signals. */
	evidence: GuardSignal[];
	/** Composite risk score (0-100). */
	compositeScore: number;
	/** Whether to proceed or intervene. */
	shouldIntervene: boolean;
}

/** Risk level for the composite risk assessment. */
export type RiskLevel = "green" | "yellow" | "orange" | "red";

// ── Fusion Config ───────────────────────────────────────────────────────────

/** Default detector weights for multi-signal fusion. */
const DEFAULT_FUSION_WEIGHTS: Record<string, number> = {
	// Core safety detectors get higher weight
	loop_detection: 0.20,
	thinking_loop: 0.25, // Thinking loops are most dangerous
	output_errors: 0.15,
	// Progress and recovery detectors
	progress_signal: 0.15,
	recovery_memory: 0.10,
	hypothesis_tracker: 0.05,
	goal_decomposition: 0.05,
};

/** Default severity thresholds for graduated intervention. */
const SEVERITY_THRESHOLDS = {
	nudge: 30,      // Below this: proceed; above this: nudge
	restrict: 55,   // Below this: nudge; above this: restrict
	pause: 75,      // Below this: restrict; above this: pause
	abort: 90,      // Below this: pause; above this: abort
};

/** Detectors whose signal is pattern-based on unambiguous repetition (the
 *  same tool call, the same turn shape, thinking without acting) — these are
 *  trusted to escalate all the way to pause/abort on their own.
 *
 *  Every other detector (progress_signal, recovery_memory, hypothesis_tracker,
 *  goal_decomposition) scores from heuristics — keyword matches, string
 *  similarity, phase labels — against content that varies enormously in
 *  legitimate work (a read_file on a JSON config scores identically to a
 *  stuck agent re-reading the same file, because neither result contains the
 *  "meaningful change" keywords). A single such signal must never be able to
 *  interrupt a healthy run, so composite severity is capped at "nudge"
 *  unless a reliable detector is present in the evidence.
 */
const RELIABLE_ESCALATION_DETECTORS = new Set(["loop_detection", "thinking_loop", "output_errors"]);

// ── Unified Config ──────────────────────────────────────────────────────────

/** Single config interface for all guard rails. */
export interface GuardEngineConfig {
	// ── Loop detection ─────────────────────────────────────────────────────
	/** Rolling history kept for analysis (default 10). */
	loopMaxHistory?: number;
	/** Consecutive identical turns to trigger exact-repeat (default 3). */
	loopExactRepeatWindow?: number;
	/** Consecutive turns with same tool-name sequence to flag (default 4). */
	loopDegenerateWindow?: number;
	/** Consecutive turns with zero new signal to flag (default 5). */
	loopStagnationWindow?: number;
	/** Duplicate call threshold (default 3). */
	loopDuplicateThreshold?: number;
	/** Failure loop threshold (default 3). */
	loopFailureThreshold?: number;

	// ── Tool guards ────────────────────────────────────────────────────────
	/** Whether to enable tool-call guards (duplicate + failure-loop). */
	guardsEnabled?: boolean;
	/** Whether duplicate guard is on by default. */
	duplicateGuardEnabled?: boolean;
	/** Whether failure guard is on by default. */
	failureGuardEnabled?: boolean;

	// ── Thinking loop detection ────────────────────────────────────────────
	/** Whether to enable thinking loop detection (default true). */
	thinkingLoopDetectionEnabled?: boolean;
	/** Min assistant text length to count as a "thinking turn" (default 500). */
	thinkingLoopMinThinkingLength?: number;
	/** Consecutive thinking-only turns to trigger (default 8). */
	thinkingLoopThinkingOnlyThreshold?: number;
	/** Thinking length growth ratio to flag escalation (default 2.0). */
	thinkingLoopEscalationRatio?: number;
	/** Max total thinking tokens across the session (default 120000). */
	thinkingLoopMaxTotalThinkingTokens?: number;
	/** Number of meta-reasoning hits to trigger (default 5). */
	thinkingLoopMetaReasoningThreshold?: number;

	// ── Output guard ───────────────────────────────────────────────────────
	/** Max retry attempts for transient/provider errors (default 3). */
	outputMaxRetries?: number;
	/** Base delay in ms before first retry (default 500). */
	outputRetryBaseDelayMs?: number;
	/** Max delay cap for retries (default 15000). */
	outputMaxRetryDelayMs?: number;
	/** Whether to auto-compact on context_full (default true). */
	outputAutoCompactOnContextFull?: boolean;
	/** Max consecutive empty assistant responses before aborting (default 3). */
	outputMaxEmptyResponses?: number;
	/** Max consecutive non-committal assistant responses before aborting (default 3). */
	outputMaxNonCommittalResponses?: number;
	/** Context-usage fraction that triggers budget_exhausted (default 0.95). */
	outputBudgetThreshold?: number;
	/** Max consecutive context_full→compact_then_retry cycles before aborting (default 3). */
	outputMaxConsecutiveCompactions?: number;

	// ── Progress signal ────────────────────────────────────────────────────
	/** Whether to enable progress signal tracking (default true). */
	progressSignalEnabled?: boolean;
	/** Min score before nudging (default 30). */
	progressSignalMinScore?: number;
	/** Min low-score turns before nudging (default 5). */
	progressSignalMinLowScoreTurns?: number;

	// ── Goal decomposition ─────────────────────────────────────────────────
	/** Whether to enable goal decomposition (default true). */
	goalDecompositionEnabled?: boolean;
	/** Max subgoals (default 10). */
	goalDecomposerMaxSubgoals?: number;

	// ── Recovery memory ────────────────────────────────────────────────────
	/** Whether to enable recovery memory (default true). */
	recoveryMemoryEnabled?: boolean;
	/** Max entries (default 50). */
	recoveryMemoryMaxEntries?: number;

	// ── Hypothesis tracking ────────────────────────────────────────────────
	/** Whether to enable hypothesis tracking (default true). */
	hypothesisTrackingEnabled?: boolean;
	/** Max hypotheses (default 10). */
	hypothesisTrackerMaxHypotheses?: number;

	// ── Fusion / graduated intervention ─────────────────────────────────────
	/** Whether to enable multi-signal fusion (default true). */
	fusionEnabled?: boolean;
	/** Custom detector weights for fusion (overrides defaults). */
	fusionWeights?: Record<string, number>;
	/** Whether to use the graduated intervention ladder (default true). Heuristic
	 *  detectors are capped at "nudge" regardless — see RELIABLE_ESCALATION_DETECTORS. */
	graduatedIntervention?: boolean;

	// ── Events ─────────────────────────────────────────────────────────────
	/** Emit events to the UI/event bus. */
	onEvent?: EventHandler;
	/** Trigger compaction when output guard requests it. */
	onCompact?: () => Promise<number | null>;
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface GuardEngine {
	// ── Expose internal OutputGuard (for loop runner integration) ────────
	/** The internal OutputGuard — used by the loop runner for error/response handling. */
	outputGuard: OutputGuard;

	// ── Legacy API (backward compatible) ─────────────────────────────────
	/** Check a tool call before execution. Returns block=true with a message. */
	checkToolCall(name: string, args: string): { block: boolean; message?: string; guard?: "duplicate" | "failure" };

	/** Record a failed tool call. */
	recordFailure(name: string, args: string, result: string): void;

	/** Record a successful tool call (decays failure state). */
	recordSuccess(name: string, args: string): void;

	/** Record a turn and check for loops. Returns true if a loop is detected. */
	recordTurn(assistantContent: string, toolCalls: Array<{ name: string; args: string; result: string }>): boolean;

	/** Get loop diagnostic message if a loop is detected. */
	getLoopDiagnostic(): string | null;

	/** Record a provider response and check for thinking loops. */
	recordThinkingTurn(content: string, toolCallCount: number, iteration: number, thinkingTokens?: number): string | null;

	/** Get thinking loop diagnostic if detected. */
	getThinkingDiagnostic(): string | null;

	/** Process a backend error. Returns guard decision. */
	handleError(error: unknown): {
		action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
		retryDelayMs?: number;
		attempt?: number;
		message?: string;
		isRetryable?: boolean;
	};

	/** Process a successful model response for empty/degenerate patterns. */
	checkResponse(content: string | null | undefined, toolCallsCount: number): {
		action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
		message?: string;
	};

	/** Process token usage for budget tracking. */
	processResponse(tokensUsed?: number, maxTokens?: number): { action: "proceed" | "budget_exhausted" };

	/** Record tool calls and results, compute progress signal. */
	recordProgress(
		calls: Array<{ name: string; args: string }>,
		results: Array<{ content: string; file?: string }>,
		iteration: number,
		currentPhase: string,
	): { score: number; shouldNudge: boolean; nudgeMessage?: string; stuckReasons: string[] };

	/** Check if the agent should be nudged based on accumulated low progress. */
	shouldNudge(): boolean;

	/** Record a failure/nudge event. Returns similar past entries. */
	recordFailureEntry(
		failureType: string,
		approach: string,
		outcome: string,
		suggestedAlternative?: string,
	): { entryId: string; similarEntries: Array<{ approach: string; outcome: string; repeatCount: number }> };

	/** Get warnings for a new approach. */
	getFailureWarnings(approach: string, failureType: string): string[];

	/** Look up recovery warnings for this failure and record it, atomically —
	 *  so the warning lookup and the recorded entry share one derivation. */
	checkAndRecordFailure(name: string, args: string, result: string): { warnings: string[] };

	/** Add a new hypothesis. */
	addHypothesis(statement: string, test: string, confidence?: number): { id: string };

	/** Falsify a hypothesis. */
	falsifyHypothesis(hypothesisId: string, testResult: string): boolean;

	/** Verify a hypothesis. */
	verifyHypothesis(hypothesisId: string, testResult: string): boolean;

	/** Get active hypotheses. */
	getActiveHypotheses(): Array<{ id: string; statement: string; confidence: number }>;

	/** Build a prompt asking the model to generate hypotheses. */
	buildHypothesisPrompt(stuckReasons: string[]): string;

	/** Parse hypotheses from model output and store them. Returns count parsed. */
	parseHypothesesFromText(text: string): number;

	/** Check active hypotheses against new evidence; falsifies any that are contradicted.
	 *  Returns the IDs of hypotheses falsified by this evidence. */
	checkHypothesesAgainstEvidence(evidence: Array<{ content: string; type?: string }>): string[];

	/** Build a steering message asking the model to decompose the objective. */
	buildDecompositionPrompt(objective: string): string;

	/** Parse and set a breakdown from model output. */
	parseGoalBreakdown(text: string, objective: string): boolean;

	/** Get the current breakdown. */
	getGoalBreakdown(): { completedCount: number; totalCount: number; completionPercentage: number } | null;

	/** Get a status summary for the model. */
	getGoalStatusSummary(): string | null;

	/** Evaluate all guard signals and return a composite decision. */
	evaluate(): GuardDecision;

	/** Get the current composite risk assessment. */
	getCompositeRisk(): { score: number; level: RiskLevel; signals: GuardSignal[] };

	// ── Lifecycle ──────────────────────────────────────────────────────────
	/** Reset all guard state. */
	reset(): void;

	/** Get a summary of all guard stats for diagnostics. */
	getStats(): Record<string, unknown>;
}

// ── Factory ─────────────────────────────────────────────────────────────────

export function createGuardEngine(config: GuardEngineConfig = {}): GuardEngine {
	const {
		onEvent,
		onCompact,
		// Loop detection defaults
		loopMaxHistory = 10,
		loopExactRepeatWindow = 3,
		loopDegenerateWindow = 4,
		loopStagnationWindow = 5,
		loopDuplicateThreshold = 3,
		loopFailureThreshold = 3,
		// Tool guard defaults
		guardsEnabled = true,
		duplicateGuardEnabled = true,
		failureGuardEnabled = false,
		// Thinking loop defaults
		thinkingLoopDetectionEnabled = true,
		thinkingLoopMinThinkingLength = 500,
		thinkingLoopThinkingOnlyThreshold = 8,
		thinkingLoopEscalationRatio = 2.0,
		thinkingLoopMaxTotalThinkingTokens = 120_000,
		thinkingLoopMetaReasoningThreshold = 5,
		// Output guard defaults
		outputMaxRetries = 3,
		outputRetryBaseDelayMs = 500,
		outputMaxRetryDelayMs = 15_000,
		outputAutoCompactOnContextFull = true,
		outputMaxEmptyResponses = 3,
		outputMaxNonCommittalResponses = 3,
		outputBudgetThreshold = 0.95,
		outputMaxConsecutiveCompactions = 3,
		// Progress signal defaults
		progressSignalEnabled = true,
		progressSignalMinScore = 30,
		progressSignalMinLowScoreTurns = 5,
		// Goal decomposition defaults
		goalDecompositionEnabled = true,
		goalDecomposerMaxSubgoals = 10,
		// Recovery memory defaults
		recoveryMemoryEnabled = true,
		recoveryMemoryMaxEntries = 50,
		// Hypothesis tracking defaults
		hypothesisTrackingEnabled = true,
		hypothesisTrackerMaxHypotheses = 10,
		// Fusion / graduated intervention (on by default)
		fusionEnabled = true,
		fusionWeights = {},
		graduatedIntervention = true,
	} = config;

	// ── Merge fusion weights ─────────────────────────────────────────────
	const weights = { ...DEFAULT_FUSION_WEIGHTS, ...fusionWeights };

	// ── Lazy detector instantiation ──────────────────────────────────────
	let _loopDetector: LoopDetector | null = null;
	let _thinkingLoopDetector: ThinkingLoopDetector | null = null;
	let _outputGuard: OutputGuard | null = null;
	let _progressSignal: ProgressSignalTracker | null = null;
	let _recoveryMemory: RecoveryMemory | null = null;
	let _hypothesisTracker: HypothesisTracker | null = null;
	let _goalDecomposer: GoalDecomposer | null = null;

	const getLoopDetector = (): LoopDetector => {
		if (!_loopDetector) {
			const { LoopDetector: LD } = require("./loop-detector.ts");
			_loopDetector = new LD({
				maxHistory: loopMaxHistory,
				exactRepeatWindow: loopExactRepeatWindow,
				degenerateWindow: loopDegenerateWindow,
				stagnationWindow: loopStagnationWindow,
				duplicateThreshold: loopDuplicateThreshold,
				failureThreshold: loopFailureThreshold,
			});
		}
		return _loopDetector!;
	};

	const getThinkingLoopDetector = (): ThinkingLoopDetector | null => {
		if (!thinkingLoopDetectionEnabled) return null;
		if (!_thinkingLoopDetector) {
			const { ThinkingLoopDetector: TLD } = require("./thinking-loop-detector.ts");
			_thinkingLoopDetector = new TLD({
				minThinkingLength: thinkingLoopMinThinkingLength,
				thinkingOnlyThreshold: thinkingLoopThinkingOnlyThreshold,
				escalationRatio: thinkingLoopEscalationRatio,
				maxTotalThinkingTokens: thinkingLoopMaxTotalThinkingTokens,
				metaReasoningThreshold: thinkingLoopMetaReasoningThreshold,
			});
		}
		return _thinkingLoopDetector!;
	};

	const getOutputGuard = (): OutputGuard => {
		if (!_outputGuard) {
			const { OutputGuard: OG } = require("./output-guard.ts");
			_outputGuard = new OG({
				maxRetries: outputMaxRetries,
				retryBaseDelayMs: outputRetryBaseDelayMs,
				maxRetryDelayMs: outputMaxRetryDelayMs,
				autoCompactOnContextFull: outputAutoCompactOnContextFull,
				maxEmptyResponses: outputMaxEmptyResponses,
				maxNonCommittalResponses: outputMaxNonCommittalResponses,
				budgetThreshold: outputBudgetThreshold,
				maxConsecutiveCompactions: outputMaxConsecutiveCompactions,
				onEvent,
				onCompact,
				loopDetector: getLoopDetector(),
			});
		}
		return _outputGuard!;
	};

	const getProgressSignal = (): ProgressSignalTracker => {
		if (!_progressSignal) {
			const { ProgressSignalTracker: PST } = require("./progress-signal.ts");
			_progressSignal = new PST({
				minScoreBeforeNudge: progressSignalMinScore,
				minLowScoreTurns: progressSignalMinLowScoreTurns,
			});
		}
		return _progressSignal!;
	};

	const getRecoveryMemory = (): RecoveryMemory => {
		if (!_recoveryMemory) {
			const { RecoveryMemory: RM } = require("./recovery-memory.ts");
			_recoveryMemory = new RM({ maxEntries: recoveryMemoryMaxEntries });
		}
		return _recoveryMemory!;
	};

	const getHypothesisTracker = (): HypothesisTracker => {
		if (!_hypothesisTracker) {
			const { HypothesisTracker: HT } = require("./hypothesis-tracker.ts");
			_hypothesisTracker = new HT({ maxHypotheses: hypothesisTrackerMaxHypotheses });
		}
		return _hypothesisTracker!;
	};

	const getGoalDecomposer = (): GoalDecomposer => {
		if (!_goalDecomposer) {
			const { GoalDecomposer: GD } = require("./goal-decomposer.ts");
			_goalDecomposer = new GD({ maxSubgoals: goalDecomposerMaxSubgoals });
		}
		return _goalDecomposer!;
	};

	// Turn counter for recordTurn's emitted intervention events — it doesn't
	// receive an iteration number, unlike the other record* methods.
	let turnCounter = 0;

	// ── Map intervention severity to event format ─────────────────────────
	const mapSeverity = (s: InterventionSeverity): string => {
		switch (s) {
			case "info": return "info";
			case "nudge": return "warning";
			case "restrict": return "warning";
			case "pause": return "error";
			case "abort": return "error";
		}
	};

	// ── Unified intervention emitter ─────────────────────────────────────
	const emitIntervention = (
		kind: string,
		cause: string,
		detector: string,
		message: string,
		iteration: number,
		action: string,
		severity: InterventionSeverity = "nudge",
	): void => {
		onEvent?.({
			type: "harness_intervention",
			id: `guard-${Date.now()}`,
			kind: kind as never,
			cause,
			detector,
			evidence: { summary: message },
			iteration,
			action: action as never,
			severity: mapSeverity(severity),
			attempt: 1,
		} as never);
	};

	// ── Tool guard config ────────────────────────────────────────────────
	const duplicateGuardOn = guardsEnabled && duplicateGuardEnabled;
	const failureGuardOn = guardsEnabled && failureGuardEnabled;
	const guardThresholds = duplicateGuardOn || failureGuardOn;

	// ── Composite signal collection ─────────────────────────────────────────
	const collectSignals = (): GuardSignal[] => {
		const signals: GuardSignal[] = [];

		// Loop detection signal
		const loopDet = getLoopDetector();
		const loopDiagnostic = loopDet.getLoopDiagnostic();
		if (loopDiagnostic) {
			signals.push({
				detector: "loop_detection",
				signal: "loop_detected",
				score: 85,
				weight: weights.loop_detection,
				description: loopDiagnostic,
			});
		}

		// Thinking loop signal
		if (thinkingLoopDetectionEnabled) {
			const thinkingDet = getThinkingLoopDetector();
			const thinkingDiag = thinkingDet?.getDiagnostic();
			if (thinkingDiag) {
				signals.push({
					detector: "thinking_loop",
					signal: "thinking_loop_detected",
					score: 90,
					weight: weights.thinking_loop,
					description: thinkingDiag,
				});
			}
		}

		// Progress signal
		if (progressSignalEnabled) {
			const progress = getProgressSignal();
			const lastScore = progress["lastScore"] ?? 100;
			const score = Math.max(0, 100 - lastScore); // Invert: low progress = high score
			if (score > 30) {
				signals.push({
					detector: "progress_signal",
					signal: "low_progress",
					score,
					weight: weights.progress_signal,
					description: `Progress score is ${lastScore}/100`,
				});
			}
		}

		// Recovery memory signal
		if (recoveryMemoryEnabled) {
			const entries = getRecoveryMemory().getEntries();
			const recentFailures = entries.filter((e) => e.repeatCount >= 2);
			if (recentFailures.length > 0) {
				const maxRepeat = Math.max(...recentFailures.map((e) => e.repeatCount));
				const score = Math.min(100, maxRepeat * 25);
				signals.push({
					detector: "recovery_memory",
					signal: "repeated_failure",
					score,
					weight: weights.recovery_memory,
					description: `${recentFailures.length} approaches have failed ${maxRepeat}+ times`,
				});
			}
		}

		// Hypothesis tracker signal
		if (hypothesisTrackingEnabled) {
			const hypotheses = getHypothesisTracker().getActiveHypotheses();
			if (hypotheses.length > 0 && hypotheses.every((h) => h.confidence < 20)) {
				signals.push({
					detector: "hypothesis_tracker",
					signal: "all_hypotheses_low_confidence",
					score: 60,
					weight: weights.hypothesis_tracker,
					description: "All active hypotheses have low confidence",
				});
			}
		}

		// Output error signal
		const outputGuard = getOutputGuard();
		const retryCount = outputGuard.getRetryCount();
		if (retryCount > 2) {
			signals.push({
				detector: "output_errors",
				signal: "excessive_retries",
				score: Math.min(100, retryCount * 20),
				weight: weights.output_errors,
				description: `${retryCount} consecutive retries`,
			});
		}

		return signals;
	};

	// ── Multi-signal fusion ────────────────────────────────────────────────
	const computeCompositeScore = (signals: GuardSignal[]): number => {
		if (signals.length === 0) return 0;

		// Weighted average
		let weightedSum = 0;
		let weightSum = 0;
		for (const signal of signals) {
			weightedSum += signal.score * signal.weight;
			weightSum += signal.weight;
		}

		return Math.round(weightedSum / weightSum);
	};

	// ── Graduated intervention ─────────────────────────────────────────────
	const severityFromScore = (score: number): InterventionSeverity => {
		if (score >= SEVERITY_THRESHOLDS.abort) return "abort";
		if (score >= SEVERITY_THRESHOLDS.pause) return "pause";
		if (score >= SEVERITY_THRESHOLDS.restrict) return "restrict";
		if (score >= SEVERITY_THRESHOLDS.nudge) return "nudge";
		return "info";
	};

	const actionFromSeverity = (severity: InterventionSeverity): InterventionAction => {
		switch (severity) {
			case "info": return "proceed";
			case "nudge": return "nudge";
			case "restrict": return "reflect";
			case "pause": return "pause";
			case "abort": return "abort";
		}
	};

	const messageFromDecision = (decision: GuardDecision): string => {
		switch (decision.severity) {
			case "info":
				return "";
			case "nudge":
				return decision.evidence
					.map((s) => `🔍 ${s.description}`)
					.join("\n");
			case "restrict":
				return `⚠️ ${decision.evidence[0].description}\nPlease reflect on your approach and consider alternatives.`;
			case "pause":
				return `🛑 ${decision.evidence.map((s) => s.description).join(" | ")}\nThe agent has been paused. A reassessment is required.`;
			case "abort":
				return `❌ ${decision.evidence.map((s) => s.description).join(" | ")}\nThe session is being aborted due to persistent issues.`;
		}
	};

	// ── Composite decision ──────────────────────────────────────────────────
	const evaluate = (): GuardDecision => {
		const signals = collectSignals();
		const compositeScore = computeCompositeScore(signals);
		let severity = graduatedIntervention ? severityFromScore(compositeScore) : (compositeScore > 50 ? "nudge" : "info");

		// Cap at "nudge" unless a reliable (pattern-based) detector is among the
		// evidence — heuristic-only signals (progress/recovery/hypothesis/goal)
		// must never interrupt a healthy run on their own. See
		// RELIABLE_ESCALATION_DETECTORS for why.
		const hasReliableSignal = signals.some((s) => RELIABLE_ESCALATION_DETECTORS.has(s.detector));
		if (!hasReliableSignal && (severity === "restrict" || severity === "pause" || severity === "abort")) {
			severity = "nudge";
		}

		const action = actionFromSeverity(severity);

		return {
			severity,
			action,
			message: messageFromDecision({ severity, action, evidence: signals, compositeScore, shouldIntervene: action !== "proceed" }),
			evidence: signals,
			compositeScore,
			shouldIntervene: action !== "proceed",
		};
	};

	// ── Composite risk ───────────────────────────────────────────────────
	const getCompositeRisk = () => {
		const signals = collectSignals();
		const score = computeCompositeScore(signals);
		const level: RiskLevel = score < 30 ? "green" :
			score < 55 ? "yellow" :
				score < 75 ? "orange" : "red";
		return { score, level, signals };
	};

	// ── Shared failure-context derivation ────────────────────────────────
	// Used by both recordFailure and checkAndRecordFailure so the warning
	// lookup and the entry that gets recorded always agree on the same
	// failureType/approach derivation.
	const deriveFailureContext = (
		name: string,
		args: string,
		result: string,
	): { failureType: string; approach: string } => ({
		failureType: result.includes("Error:")
			? result.slice(0, 100).replace(/^Error:\s*/i, "").slice(0, 60)
			: "unknown",
		approach: `${name} ${args.slice(0, 200)}`,
	});

	// ── Legacy: Unified intervention emitter ─────────────────────────────
	const emitInterventionLegacy = (
		kind: string,
		cause: string,
		detector: string,
		message: string,
		iteration: number,
		action: string,
	): void => {
		onEvent?.({
			type: "harness_intervention",
			id: `guard-${Date.now()}`,
			kind: kind as never,
			cause,
			detector,
			evidence: { summary: message },
			iteration,
			action: action as never,
			severity: "warning" as const,
			attempt: 1,
		} as never);
	};

	// ── Public API implementation ────────────────────────────────────────
	return {
		// Expose internal OutputGuard (for loop runner integration)
		get outputGuard() {
			return getOutputGuard();
		},

		// Tool-call guard
		checkToolCall(name: string, args: string) {
			if (!guardThresholds) return { block: false };
			return getLoopDetector().checkToolCall(name, args);
		},

		recordFailure(name: string, args: string, result: string) {
			if (guardThresholds) {
				getLoopDetector().recordFailure(name, args, result);
			}
			if (recoveryMemoryEnabled) {
				const { failureType, approach } = deriveFailureContext(name, args, result);
				const { similarEntries } = getRecoveryMemory().recordFailure(
					failureType,
					approach,
					result.slice(0, 300),
				);
				if (similarEntries.length > 0 && similarEntries[0].repeatCount >= 3) {
					emitInterventionLegacy(
						"loop", "repeated_failure", "recovery_memory",
						`Repeated failure: this approach has failed ${similarEntries[0].repeatCount} times. Last: ${similarEntries[0].outcome.slice(0, 150)}`,
						0, "change_strategy",
					);
				}
			}
		},

		checkAndRecordFailure(name: string, args: string, result: string) {
			if (!recoveryMemoryEnabled) {
				if (guardThresholds) getLoopDetector().recordFailure(name, args, result);
				return { warnings: [] };
			}
			const { failureType, approach } = deriveFailureContext(name, args, result);
			if (guardThresholds) {
				getLoopDetector().recordFailure(name, args, result);
			}
			// Record first, then look up warnings — getWarnings' repeatCount > 1
			// check needs this failure's own occurrence already counted, or a
			// second-time failure would never cross the threshold.
			const { similarEntries } = getRecoveryMemory().recordFailure(
				failureType,
				approach,
				result.slice(0, 300),
			);
			const warnings = getRecoveryMemory().getWarnings(approach, failureType);
			if (similarEntries.length > 0 && similarEntries[0].repeatCount >= 3) {
				emitInterventionLegacy(
					"loop", "repeated_failure", "recovery_memory",
					`Repeated failure: this approach has failed ${similarEntries[0].repeatCount} times. Last: ${similarEntries[0].outcome.slice(0, 150)}`,
					0, "change_strategy",
				);
			}
			return { warnings };
		},

		recordSuccess(name: string, args: string) {
			if (guardThresholds) {
				getLoopDetector().recordSuccess(name, args);
			}
			if (recoveryMemoryEnabled) {
				getRecoveryMemory().recordSuccess(`${name} ${args.slice(0, 200)}`, "success");
			}
		},

		recordTurn(assistantContent: string, toolCalls: Array<{ name: string; args: string; result: string }>) {
			const detected = getLoopDetector().recordAndDetect(assistantContent, toolCalls);
			turnCounter++;

			if (graduatedIntervention) {
				const decision = evaluate();
				if (decision.shouldIntervene && decision.action !== "proceed") {
					emitIntervention(
						decision.evidence[0]?.signal ?? "composite",
						"multi_signal_fusion",
						"guard_engine",
						decision.message ?? "",
						turnCounter,
						decision.action,
						decision.severity,
					);
				}
			}

			return detected;
		},

		getLoopDiagnostic() {
			return getLoopDetector().getLoopDiagnostic();
		},

		recordThinkingTurn(content: string, toolCallCount: number, iteration: number, thinkingTokens?: number) {
			const detector = getThinkingLoopDetector();
			if (!detector) return null;

			const diagnostic = detector.recordTurn(content, toolCallCount, iteration, thinkingTokens);

			if (diagnostic && graduatedIntervention) {
				const decision = evaluate();
				if (decision.shouldIntervene) {
					emitIntervention(
						"thinking_loop", "thinking_loop_detected", "guard_engine",
						diagnostic, iteration, decision.action, decision.severity,
					);
				}
			}

			return diagnostic;
		},

		getThinkingDiagnostic() {
			const detector = getThinkingLoopDetector();
			if (!detector) return null;
			return detector.getDiagnostic();
		},

		handleError(error: unknown) {
			return getOutputGuard().handleError(error);
		},

		checkResponse(content: string | null | undefined, toolCallsCount: number) {
			return getOutputGuard().checkResponse(content, toolCallsCount);
		},

		processResponse(tokensUsed?: number, maxTokens?: number) {
			return getOutputGuard().processResponse(tokensUsed, maxTokens);
		},

		recordProgress(calls, results, iteration, currentPhase) {
			const signal = getProgressSignal().record(calls, results, iteration, currentPhase);

			if (signal.shouldNudge && signal.nudgeMessage) {
				// progress_signal is heuristic-only (see RELIABLE_ESCALATION_DETECTORS)
				// — always emitted as a "nudge", never a higher severity, even when
				// the raw score is very low.
				emitIntervention(
					"progress", "low_progress", "progress_signal",
					signal.nudgeMessage, iteration, "nudge", "nudge",
				);
			}

			return {
				score: signal.score,
				shouldNudge: signal.shouldNudge,
				nudgeMessage: signal.nudgeMessage,
				stuckReasons: signal.stuckReasons,
			};
		},

		shouldNudge() {
			return getProgressSignal().shouldNudge();
		},

		recordFailureEntry(failureType, approach, outcome, suggestedAlternative) {
			const { entryId, similarEntries } = getRecoveryMemory().recordFailure(
				failureType, approach, outcome, suggestedAlternative,
			);
			return {
				entryId,
				similarEntries: similarEntries.map(e => ({
					approach: e.approach,
					outcome: e.outcome,
					repeatCount: e.repeatCount,
				})),
			};
		},

		getFailureWarnings(approach, failureType) {
			return getRecoveryMemory().getWarnings(approach, failureType);
		},

		addHypothesis(statement, test, confidence = 50) {
			return getHypothesisTracker().add(statement, test, confidence);
		},

		falsifyHypothesis(hypothesisId, testResult) {
			return getHypothesisTracker().falsify(hypothesisId, testResult);
		},

		verifyHypothesis(hypothesisId, testResult) {
			return getHypothesisTracker().verify(hypothesisId, testResult);
		},

		getActiveHypotheses() {
			return getHypothesisTracker()
				.getActiveHypotheses()
				.map(h => ({ id: h.id, statement: h.statement, confidence: h.confidence }));
		},

		buildHypothesisPrompt(stuckReasons) {
			return getHypothesisTracker().buildHypothesisPrompt(stuckReasons);
		},

		parseHypothesesFromText(text: string) {
			return getHypothesisTracker().parseFromText(text).length;
		},

		checkHypothesesAgainstEvidence(evidence) {
			return getHypothesisTracker().checkAgainstEvidence(evidence);
		},

		buildDecompositionPrompt(objective) {
			return getGoalDecomposer().buildDecompositionPrompt(objective);
		},

		parseGoalBreakdown(text, objective) {
			return getGoalDecomposer().parseFromText(text, objective);
		},

		getGoalBreakdown() {
			const breakdown = getGoalDecomposer().getBreakdown();
			if (!breakdown) return null;
			return {
				completedCount: breakdown.completedCount,
				totalCount: breakdown.totalCount,
				completionPercentage: breakdown.completionPercentage,
			};
		},

		getGoalStatusSummary() {
			return getGoalDecomposer().getStatusSummary();
		},

		evaluate,
		getCompositeRisk,

		// Lifecycle
		reset() {
			getLoopDetector().reset();
			if (_thinkingLoopDetector) _thinkingLoopDetector.reset();
			getOutputGuard().reset();
			getProgressSignal().reset();
			getRecoveryMemory().clear();
			getHypothesisTracker().reset();
			getGoalDecomposer().reset();
			turnCounter = 0;
		},

		getStats() {
			const stats: Record<string, unknown> = {};
			const loop = getLoopDetector();
			stats.loopHistoryLength = (loop as unknown as Record<string, unknown> & { history?: unknown[] })?.history?.length ?? 0;
			const thinking = getThinkingLoopDetector();
			if (thinking) {
				const tStats = thinking.getStats();
				stats.thinkingConsecutiveOnly = tStats.consecutiveThinkingOnly;
				stats.thinkingTotalTurns = tStats.totalThinkingTurns;
				stats.thinkingTotalTokens = tStats.totalThinkingTokens;
				stats.thinkingMetaReasoningHits = tStats.metaReasoningHits;
			}
			stats.outputRetryCount = getOutputGuard().getRetryCount();
			const progress = getProgressSignal();
			stats.progressLowScoreTurns = progress["lowScoreTurns"] ?? 0;
			stats.progressLastScore = progress["lastScore"] ?? 100;
			stats.recoveryMemoryEntries = getRecoveryMemory().getEntries().length;
			const hypotheses = getHypothesisTracker();
			stats.hypothesisActive = hypotheses.getActiveHypotheses().length;
			stats.hypothesisVerified = hypotheses.getVerifiedHypotheses().length;
			stats.hypothesisFalsified = hypotheses["hypotheses"]
				?.filter((h: { status: string }) => h.status === "falsified")
				.length ?? 0;
			const goal = getGoalDecomposer();
			const gb = goal.getBreakdown();
			if (gb) {
				stats.goalCompleted = gb.completedCount;
				stats.goalTotal = gb.totalCount;
			}

			const signals = collectSignals();
			stats.guardEngine = {
				fusionEnabled,
				graduatedIntervention,
				compositeScore: computeCompositeScore(signals),
				riskLevel: computeCompositeScore(signals) < 30 ? "green" :
					computeCompositeScore(signals) < 55 ? "yellow" :
						computeCompositeScore(signals) < 75 ? "orange" : "red",
				signalCount: signals.length,
			};
			return stats;
		},
	};
}
