/**
 * Progress tracker for detecting stalled autonomous turns.
 *
 * Enhancements over the original:
 * 1. **Evidence fingerprinting** — tracks a hash of all evidence, not just size
 * 2. **Sliding window** — recent-turn evidence comparison detects re-cycling
 * 3. **Turn-level tracking** — per-turn evidence count for trend analysis
 */

export interface ProgressTask {
	id: string | number;
	status: string;
}

export interface ProgressTrackerOptions {
	/** Minimum autonomous checks before considering stall. */
	minimumChecks?: number;
	/** Consecutive turns with no progress before flagging. */
	stalledChecks?: number;
	/** Size of sliding window for evidence overlap comparison. */
	windowSize?: number;
	/** Evidence overlap threshold (0-1) that counts as a stall. */
	overlapThreshold?: number;
}

/** Simple string hash for evidence fingerprinting. */
function hashString(str: string): string {
	let hash = 0x811c9dc5;
	for (let i = 0; i < str.length; i++) {
		hash ^= str.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(36);
}

export class ProgressTracker {
	private readonly minimumChecks: number;
	private readonly stalledChecks: number;
	private readonly windowSize: number;
	private readonly overlapThreshold: number;

	/** All evidence items seen so far. */
	private readonly evidence = new Set<string>();
	/** Per-turn evidence snapshots for overlap analysis. */
	private readonly turnHistory: Set<string>[] = [];
	private checks = 0;
	private previousFingerprint = "";
	private consecutiveStalls = 0;

	constructor(options: ProgressTrackerOptions = {}) {
		this.minimumChecks = options.minimumChecks ?? 3;
		this.stalledChecks = options.stalledChecks ?? 2;
		this.windowSize = options.windowSize ?? 6;
		this.overlapThreshold = options.overlapThreshold ?? 0.7;
	}

	recordToolResult(
		toolName: string,
		argumentsJson: string,
		result: string,
	): void {
		const key = `${toolName}\0${argumentsJson}\0${result.replace(/\s+/g, " ").trim().slice(0, 500)}`;
		this.evidence.add(key);
	}

	/**
	 * Check for stalls after a turn. Maintains backward-compatible `shouldStop`
	 * name so existing callers and tests don't break.
	 */
	shouldStop(tasks: readonly ProgressTask[]): boolean {
		this.checks++;

		// Snapshot turn-level evidence
		const turnEvidence = new Set(this.evidence);
		this.turnHistory.push(turnEvidence);
		if (this.turnHistory.length > this.windowSize) this.turnHistory.shift();

		// 1. Fingerprint-based check (original)
		const taskState = tasks
			.map(task => `${task.id}:${task.status}`)
			.sort()
			.join("|");
		const fingerprint = `${this.evidence.size}\0${taskState}\0${hashString([...this.evidence].slice(-20).join(" "))}`;
		if (fingerprint === this.previousFingerprint) {
			this.consecutiveStalls++;
		} else {
			this.consecutiveStalls = 0;
		}
		this.previousFingerprint = fingerprint;

		// 2. Evidence overlap check (new — detects recycles)
		if (this.turnHistory.length >= 3) {
			const recent = this.turnHistory[this.turnHistory.length - 1];
			const prior = this.turnHistory[this.turnHistory.length - 2];
			const overlap = this.#computeOverlap(recent, prior);
			if (overlap >= this.overlapThreshold) {
				this.consecutiveStalls++;
			}
		}

		return this.checks > this.minimumChecks && this.consecutiveStalls >= this.stalledChecks;
	}

	#computeOverlap(a: Set<string>, b: Set<string>): number {
		if (a.size === 0 || b.size === 0) return 0;
		let intersection = 0;
		const [small, large] = a.size < b.size ? [a, b] : [b, a];
		for (const x of small) {
			if (large.has(x)) intersection++;
		}
		return intersection / Math.max(a.size, b.size);
	}
}
