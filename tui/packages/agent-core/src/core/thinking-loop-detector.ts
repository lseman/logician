// ── Thinking Loop Detector ─────────────────────────────────────────────────
// Detects when the agent is stuck in meta-thinking loops — long reasoning
// without tool calls, escalating thinking length, or circular meta-reasoning.
//
// Strategies:
//
// 1. **Thinking-only turns** — tracks consecutive turns with no tool calls but
//    long assistant content. A thinking-only turn is one where the model emits
//    text but no tool calls, and that text exceeds a configurable length.
//
// 2. **Thinking escalation** — tracks the length of assistant text across
//    consecutive thinking-only turns. If the text keeps getting longer (more
//    tokens consumed per turn), the model is spiraling.
//
// 3. **Meta-reasoning patterns** — detects text patterns where the model is
//    reasoning about its own reasoning ("I need to think about how to approach
//    this", "Let me reconsider my approach", etc.).
//
// 4. **Total thinking budget** — caps cumulative thinking tokens. If the agent
//    has consumed more than the budget across all turns, stop.
//
// Any single strategy tripping triggers a stop decision.

export interface ThinkingTurnSnapshot {
	iteration: number;
	assistantTextLength: number;
	assistantTextSnippet: string;
	toolCallCount: number;
	thinkingTokens?: number;
}

export interface ThinkingLoopDetectorOptions {
	/** Minimum assistant text length to count as a "thinking turn" (default 500). */
	minThinkingLength?: number;
	/** Consecutive thinking-only turns to trigger (default 5). */
	thinkingOnlyThreshold?: number;
	/** Thinking length growth ratio to flag escalation (default 1.5x). */
	escalationRatio?: number;
	/** Maximum total thinking tokens across the session (default 80000). */
	maxTotalThinkingTokens?: number;
	/** Number of meta-reasoning hits to trigger (default 3). */
	metaReasoningThreshold?: number;
	/** Whether to enable the detector by default. */
	enabled?: boolean;
}

const DEFAULT_MIN_THINKING_LENGTH = 500;
const DEFAULT_THINKING_ONLY_THRESHOLD = 5;
const DEFAULT_ESCALATION_RATIO = 1.5;
const DEFAULT_MAX_TOTAL_THINKING_TOKENS = 80_000;
const DEFAULT_META_REASONING_THRESHOLD = 3;

// Patterns indicating the model is reasoning about its own reasoning rather
// than taking action. These capture the "meta-loop" signature.
const META_REASONING_PATTERNS = [
	// "Let me think about how to approach..." — planning about planning.
	/\blet\s+me\s+(think\s+(about|on|over)|consider)\s+(how\s+)?(?:to\s+)?(?:approach|handle|solve| tackle)\b/i,
	// "I need to think about..." — pure meta-reasoning, no action.
	/\bi\s+(need\s+to|should\s+to|must\s+to)\s+(think\s+about|consider|reflect\s+on|ponder)\b/i,
	// "Before I do X, I need to think about Y..." — procrastination pattern.
	/\b(?:before|while|when)\s+(?:i\s+)?(?:do|proceed|start|begin|move)\s+.*?\b,\s+(?:i\s+)?(?:need\s+to|should|have\s+to)\s+(think|consider|reflect)\b/i,
	// "I'm not sure about X, let me think..." — hesitation loop.
	/\b(?:i'm\s+not\s+sure|not\s+sure\s+about|i\s+don't\s+know\s+how)\b.*?(?:let\s+me\s+think|let\s+me\s+consider)\b/i,
	// "I should first understand... then I can..." — endless analysis.
	/\b(?:i\s+should\s+first|first\s+i\s+need\s+to)\s+(understand|comprehend|grasp|analyze)\b.*?\b(?:then|after\s+that|once\s+i.*)\b/i,
	// "Upon further reflection..." — meta-reasoning escalation.
	/\b(?:upon\s+further\s+reflection|after\s+considering|on\s+second\s+thought)\b/i,
	// "This requires me to think..." — meta-reasoning declaration.
	/\b(?:this\s+(?:requires|demands|needs)\s+(?:me\s+)?to\s+think)\b/i,
	// "I need to step back and think..." — retreat into thinking.
	/\b(?:step\s+back|pause)\s+(?:to\s+)?(?:think|consider|reflect)\b/i,
	// "I realize I need to think..." — meta-realization loop.
	/\b(?:i\s+realize|i\s+see\s+that|i\s+understand\s+now)\s+(?:that\s+)?(?:i\s+need\s+to|i\s+should|i\s+must)\s+(think|consider|rethink)\b/i,
	// "Thinking through this... okay..." — self-talk spiral.
	/\b(?:thinking\s+through\s+this|let\s+me\s+walk\s+through\s+this|working\s+through\s+this)\b.*?\b(?:okay|alright|right)\b/i,
];

export class ThinkingLoopDetector {
	private options: Required<ThinkingLoopDetectorOptions>;

	// ── State ─────────────────────────────────────────────────────────────
	private thinkingTurns: ThinkingTurnSnapshot[] = [];
	private consecutiveThinkingOnly = 0;
	private totalThinkingTokens = 0;
	private metaReasoningHits = 0;
	private lastThinkingLength = 0;

	constructor(options: ThinkingLoopDetectorOptions = {}) {
		this.options = {
			minThinkingLength:
				options.minThinkingLength ?? DEFAULT_MIN_THINKING_LENGTH,
			thinkingOnlyThreshold:
				options.thinkingOnlyThreshold ?? DEFAULT_THINKING_ONLY_THRESHOLD,
			escalationRatio: options.escalationRatio ?? DEFAULT_ESCALATION_RATIO,
			maxTotalThinkingTokens:
				options.maxTotalThinkingTokens ?? DEFAULT_MAX_TOTAL_THINKING_TOKENS,
			metaReasoningThreshold:
				options.metaReasoningThreshold ?? DEFAULT_META_REASONING_THRESHOLD,
			enabled: options.enabled ?? true,
		};
	}

	// ── Record a turn and check for thinking loops ─────────────────────────
	/**
	 * Record a turn and check if the agent is in a thinking loop.
	 * Returns null if no loop detected, or a diagnostic string if a loop is detected.
	 */
	recordTurn(
		assistantText: string,
		toolCallCount: number,
		iteration: number,
		thinkingTokens?: number,
	): string | null {
		const textLength = assistantText.trim().length;

		// Track total thinking tokens if provided
		if (thinkingTokens !== undefined) {
			this.totalThinkingTokens += thinkingTokens;
		}

		const isThinkingTurn =
			toolCallCount === 0 && textLength >= this.options.minThinkingLength;

		if (isThinkingTurn) {
			this.consecutiveThinkingOnly++;

			this.thinkingTurns.push({
				iteration,
				assistantTextLength: textLength,
				assistantTextSnippet: assistantText.slice(0, 200),
				toolCallCount: 0,
				thinkingTokens,
			});

			// Check for escalation (thinking getting longer each thinking turn)
			if (
				this.lastThinkingLength > 0 &&
				textLength > this.lastThinkingLength * this.options.escalationRatio
			) {
				return this.buildEscalationDiagnostic(
					textLength,
					this.lastThinkingLength,
				);
			}
			this.lastThinkingLength = textLength;

			// Check consecutive thinking-only threshold
			if (this.consecutiveThinkingOnly >= this.options.thinkingOnlyThreshold) {
				return this.buildThinkingOnlyDiagnostic(
					this.consecutiveThinkingOnly,
					assistantText,
				);
			}

			// Check for meta-reasoning patterns (after threshold to avoid false positives)
			if (this.countMetaReasoningPatterns(assistantText) > 0) {
				this.metaReasoningHits++;
			}
			if (this.metaReasoningHits >= this.options.metaReasoningThreshold) {
				return this.buildMetaReasoningDiagnostic(
					this.metaReasoningHits,
					assistantText,
				);
			}
		} else {
			// Not a thinking turn — reset counters
			this.consecutiveThinkingOnly = 0;
			this.lastThinkingLength = 0;
		}

		// Check total thinking budget regardless of turn type
		if (this.totalThinkingTokens > this.options.maxTotalThinkingTokens) {
			return this.buildBudgetDiagnostic(
				this.totalThinkingTokens,
				this.options.maxTotalThinkingTokens,
			);
		}

		return null;
	}

	// ── Pattern matching ──────────────────────────────────────────────────
	private countMetaReasoningPatterns(text: string): number {
		return META_REASONING_PATTERNS.filter((re) => re.test(text)).length;
	}

	// ── Diagnostic builders ───────────────────────────────────────────────
	private buildThinkingOnlyDiagnostic(
		consecutive: number,
		lastText: string,
	): string {
		return (
			`Thinking loop detected: ${consecutive} consecutive turns with no tool calls. ` +
			`You kept reasoning (${lastText.slice(0, 120)}...) instead of taking action. ` +
			`Stop thinking and either act on what you know or say you're done.`
		);
	}

	private buildEscalationDiagnostic(
		currentLength: number,
		previousLength: number,
	): string {
		const ratio = (currentLength / previousLength).toFixed(1);
		return (
			`Thinking spiral detected: your response grew ${ratio}x this turn ` +
			`(${previousLength} → ${currentLength} chars) with no tool calls. ` +
			`You're writing longer thoughts but not making progress. Act now.`
		);
	}

	private buildMetaReasoningDiagnostic(hits: number, lastText: string): string {
		return (
			`Meta-reasoning loop detected: ${hits} turns with meta-reasoning patterns ` +
			`("let me think", "I need to consider", etc.). You're reasoning about reasoning ` +
			`instead of acting. ${lastText.slice(0, 100)}... Stop and decide: what's the next action?`
		);
	}

	private buildBudgetDiagnostic(total: number, limit: number): string {
		return (
			`Thinking budget exhausted: ${total} total thinking tokens used (limit: ${limit}). ` +
			`You've consumed too many tokens on reasoning without completing the task. ` +
			`Stop and produce actionable output immediately.`
		);
	}

	// ── Reset ─────────────────────────────────────────────────────────────
	reset(): void {
		this.thinkingTurns = [];
		this.consecutiveThinkingOnly = 0;
		this.totalThinkingTokens = 0;
		this.metaReasoningHits = 0;
		this.lastThinkingLength = 0;
	}

	// ── Diagnostics ───────────────────────────────────────────────────────
	getDiagnostic(): string | null {
		if (this.consecutiveThinkingOnly >= this.options.thinkingOnlyThreshold) {
			return this.buildThinkingOnlyDiagnostic(this.consecutiveThinkingOnly, "");
		}
		if (this.metaReasoningHits >= this.options.metaReasoningThreshold) {
			const last = this.thinkingTurns.at(-1);
			return this.buildMetaReasoningDiagnostic(
				this.metaReasoningHits,
				last?.assistantTextSnippet ?? "",
			);
		}
		if (this.totalThinkingTokens > this.options.maxTotalThinkingTokens) {
			return this.buildBudgetDiagnostic(
				this.totalThinkingTokens,
				this.options.maxTotalThinkingTokens,
			);
		}
		return null;
	}

	getStats(): {
		consecutiveThinkingOnly: number;
		totalThinkingTurns: number;
		totalThinkingTokens: number;
		metaReasoningHits: number;
	} {
		return {
			consecutiveThinkingOnly: this.consecutiveThinkingOnly,
			totalThinkingTurns: this.thinkingTurns.length,
			totalThinkingTokens: this.totalThinkingTokens,
			metaReasoningHits: this.metaReasoningHits,
		};
	}
}
