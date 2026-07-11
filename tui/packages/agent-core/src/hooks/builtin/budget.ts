// ── Token-budget tracker ───────────────────────────────────────────────────
// Diminishing-returns stop: detect when the agent keeps emitting tokens across
// turns but is no longer making meaningful progress, and stop early instead of
// spinning to the iteration cap. Ported from openclaude's query/tokenBudget.ts,
// consulted via the `shouldStopAfterTurn` contract hook.

const DEFAULT_DIMINISHING_FLOOR = 500;
const DEFAULT_MIN_CONTINUATIONS = 3;

export interface BudgetTrackerOptions {
	// Per-turn token delta below which a turn counts as "low progress".
	diminishingFloor?: number;
	// Number of turns before diminishing-returns can trip.
	minContinuations?: number;
}

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

	// Record this turn's cumulative token count and decide whether to stop.
	// Returns true when two consecutive turns both fell below the floor after
	// the minimum number of continuations.
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
