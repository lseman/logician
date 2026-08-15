// ── RecoveryMemory ───────────────────────────────────────────────────────────
// The agent learns from past failures across turns, not just one-shot nudges.
//
// Current problem: when the model gets nudged (e.g. "you're stuck in a loop"),
// the nudge is a one-shot message. The model doesn't remember what failed
// before and keeps making the same mistakes.
//
// RecoveryMemory solves this by:
// 1. Recording each failure/nudge as a structured entry with:
//    - What went wrong (failure type)
//    - What the model tried (approach)
//    - What the result was (outcome)
//    - What to try next (suggested alternative)
// 2. When the model is about to repeat a similar approach, checking the memory
//    and warning it: "You tried this 3 turns ago and it failed because X."
// 3. After a successful approach, recording that too so the model knows what
//    works.
//
// This turns the agent's history into a learning system, not just a log.

export interface RecoveryEntry {
	/** Unique ID for this recovery entry. */
	id: string;
	/** When this entry was created. */
	createdAt: number;
	/** When this entry was last used (for freshness scoring). */
	lastUsedAt?: number;
	/** What went wrong — e.g. "file not found", "wrong tool for the job". */
	failureType: string;
	/** What the model tried to do. */
	approach: string;
	/** What actually happened. */
	outcome: string;
	/** What to try instead (empty if unknown). */
	suggestedAlternative?: string;
	/** Whether this approach was later found to work (after a nudge). */
	ultimatelySuccessful?: boolean;
	/** How many times this failure was repeated before being caught. */
	repeatCount: number;
	/** Tags for grouping similar failures. */
	tags: string[];
}

export interface RecoveryMemoryConfig {
	/** Max entries to keep (default 50). */
	maxEntries?: number;
	/** Max entries per failure type (default 10). */
	maxEntriesPerType?: number;
	/** Whether to auto-detect similar approaches (default true). */
	autoDetectSimilar?: boolean;
	/** Whether to clear entries after a successful subgoal (default true). */
	clearOnSuccess?: boolean;
}

const DEFAULT_CONFIG: Required<RecoveryMemoryConfig> = {
	maxEntries: 50,
	maxEntriesPerType: 10,
	autoDetectSimilar: true,
	clearOnSuccess: true,
};

// ── Similarity detection ─────────────────────────────────────────────────────

/**
 * Check if two approaches are similar enough to be the same failure.
 * Uses simple keyword overlap + tool name matching.
 */
function approachesMatch(
	entry: RecoveryEntry,
	newApproach: string,
	newFailureType: string,
): boolean {
	const entryTools = extractTools(entry.approach);
	const newTools = extractTools(newApproach);

	// Same tools + similar failure type
	const sharedTools = entryTools.filter((t) => newTools.includes(t));
	if (sharedTools.length > 0 && failureTypesSimilar(entry.failureType, newFailureType)) {
		return true;
	}

	// Same file + same tool
	const entryFile = extractFile(entry.approach);
	const newFile = extractFile(newApproach);
	if (
		entryFile &&
		newFile &&
		entryFile === newFile &&
		sharedTools.length > 0
	) {
		return true;
	}

	// Keyword overlap (simple)
	const entryWords = new Set(entry.approach.toLowerCase().split(/\s+/));
	const newWords = newApproach.toLowerCase().split(/\s+/);
	const overlap = newWords.filter((w) => entryWords.has(w)).length;
	const coverage = overlap / Math.max(newWords.length, 1);
	if (coverage > 0.6 && sharedTools.length > 0) {
		return true;
	}

	return false;
}

/** Check if two failure types are similar. */
function failureTypesSimilar(a: string, b: string): boolean {
	const lowerA = a.toLowerCase();
	const lowerB = b.toLowerCase();

	// An empty failure type means "unknown", not "matches everything" — every
	// string trivially includes "", so without this guard a substring check
	// below would treat every entry as similar to a blank type.
	if (!lowerA || !lowerB) return lowerA === lowerB;

	// Exact match
	if (lowerA === lowerB) return true;

	// One contains the other
	if (lowerA.includes(lowerB) || lowerB.includes(lowerA)) return true;

	// Common failure synonyms
	const synonyms = [
		["not found", "missing", "does not exist"],
		["permission denied", "access denied", "forbidden"],
		["timeout", "timed out", "deadline exceeded"],
		["parse error", "invalid format", "malformed"],
		["rate limit", "too many requests", "throttled"],
	];

	for (const group of synonyms) {
		if (
			group.some((s) => lowerA.includes(s)) &&
			group.some((s) => lowerB.includes(s))
		) {
			return true;
		}
	}

	return false;
}

/** Extract tool names from an approach string. */
function extractTools(approach: string): string[] {
	const tools: string[] = [];
	const toolPatterns = [
		/edit_file\b/,
		/write_file\b/,
		/read_file\b/,
		/bash\b/,
		/grep\b/,
		/find\b/,
		/list_files\b/,
		/apply_patch\b/,
		/web_search\b/,
		/task_status\b/,
	];
	for (const pattern of toolPatterns) {
		const match = approach.match(pattern);
		if (match) tools.push(match[0]);
	}
	return tools;
}

/** Extract file path from an approach string. */
function extractFile(approach: string): string | null {
	const match = approach.match(/\/[^\s"'`]+/);
	return match ? match[0] : null;
}

// ── RecoveryMemory class ─────────────────────────────────────────────────────

export class RecoveryMemory {
	private config: Required<RecoveryMemoryConfig>;
	private entries: RecoveryEntry[] = [];
	private idCounter = 0;

	constructor(config: RecoveryMemoryConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Record a failure/nudge event.
	 * Returns the entry ID and any similar past entries.
	 */
	recordFailure(
		failureType: string,
		approach: string,
		outcome: string,
		suggestedAlternative?: string,
	): {
		entryId: string;
		similarEntries: RecoveryEntry[];
	} {
		const id = `rec-${++this.idCounter}`;

		// Check for similar past entries
		const similarEntries = this.config.autoDetectSimilar
			? this.entries.filter((e) =>
				approachesMatch(e, approach, failureType),
			)
			: [];

		// Update repeat count for similar entries
		for (const similar of similarEntries) {
			similar.repeatCount++;
		}

		const entry: RecoveryEntry = {
			id,
			createdAt: Date.now(),
			failureType,
			approach,
			outcome,
			suggestedAlternative,
			repeatCount: similarEntries.length > 0
				? similarEntries[0].repeatCount + 1
				: 1,
			tags: generateTags(failureType, approach),
		};

		this.entries.push(entry);

		// Trim if over limit
		if (this.entries.length > this.config.maxEntries) {
			this.entries = this.entries.slice(-this.config.maxEntries);
		}

		return { entryId: id, similarEntries };
	}

	/**
	 * Record that an approach was ultimately successful (after a nudge).
	 */
	recordSuccess(approach: string, outcome: string): void {
		// Find similar past failures and mark them as ultimately successful
		if (this.config.clearOnSuccess) {
			// Clear old failures that are similar to the successful approach
			const similarIndices = this.entries
				.map((e, i) => ({ e, i }))
				.filter(({ e }) => approachesMatch(e, approach, ""))
				.map(({ i }) => i);

			for (const idx of similarIndices) {
				this.entries[idx].ultimatelySuccessful = true;
			}

			// Remove the successful entries after a delay (or immediately if too many)
			if (this.entries.length > this.config.maxEntries * 0.8) {
				this.entries = this.entries.filter(
					(e) => !e.ultimatelySuccessful,
				);
			}
		}
	}

	/**
	 * Get warnings for a new approach — checks if this approach has failed before.
	 * Returns warning messages if similar failures exist.
	 */
	getWarnings(
		approach: string,
		failureType: string,
	): string[] {
		const similarEntries = this.entries.filter((e) =>
			approachesMatch(e, approach, failureType),
		);

		if (similarEntries.length === 0) return [];

		// Sort by most recent
		similarEntries.sort((a, b) => b.createdAt - a.createdAt);

		const warnings: string[] = [];
		const mostRecent = similarEntries[0];

		if (mostRecent.repeatCount > 1) {
			warnings.push(
				`[recovery-memory] You've tried this approach ${mostRecent.repeatCount} times before and it failed. ` +
				`Last failure: ${mostRecent.outcome.slice(0, 200)}.`,
			);
		}

		if (mostRecent.suggestedAlternative) {
			warnings.push(
				`[recovery-memory] Previous attempt suggested: ${mostRecent.suggestedAlternative}.`,
			);
		}

		// Check if this was ultimately successful before (maybe the context changed)
		const ultimatelySuccessful = similarEntries.some(
			(e) => e.ultimatelySuccessful,
		);
		if (ultimatelySuccessful) {
			warnings.push(
				`[recovery-memory] Note: a similar approach was eventually successful in a different context. ` +
				`If this fails again, try a different approach.`,
			);
		}

		return warnings;
	}

	/**
	 * Get all entries for diagnostics.
	 */
	getEntries(): RecoveryEntry[] {
		return [...this.entries];
	}

	/**
	 * Clear all entries (e.g. after a successful subgoal).
	 */
	clear(): void {
		this.entries = [];
	}

	/**
	 * Get a summary of the most common failure types.
	 */
	getFailureSummary(): Array<{ type: string; count: number; lastOutcome: string }> {
		const typeMap = new Map<string, RecoveryEntry>();

		for (const entry of this.entries) {
			const existing = typeMap.get(entry.failureType);
			if (!existing || entry.createdAt > existing.createdAt) {
				typeMap.set(entry.failureType, entry);
			}
		}

		return Array.from(typeMap.entries())
			.map(([type, entry]) => ({
				type,
				count: entry.repeatCount,
				lastOutcome: entry.outcome.slice(0, 200),
			}))
			.sort((a, b) => b.count - a.count)
			.slice(0, 10);
	}
}

/**
 * Generate tags from a failure type and approach.
 */
function generateTags(failureType: string, approach: string): string[] {
	const tags: string[] = [];
	const combined = `${failureType} ${approach}`.toLowerCase();

	if (combined.includes("not found") || combined.includes("missing")) {
		tags.push("not-found");
	}
	if (combined.includes("permission") || combined.includes("access")) {
		tags.push("permission");
	}
	if (combined.includes("timeout") || combined.includes("timed out")) {
		tags.push("timeout");
	}
	if (combined.includes("parse") || combined.includes("format")) {
		tags.push("parse-error");
	}
	if (combined.includes("rate") || combined.includes("limit")) {
		tags.push("rate-limit");
	}
	if (combined.includes("tool") || combined.includes("command")) {
		tags.push("tool-error");
	}
	if (combined.includes("file") || combined.includes("path")) {
		tags.push("file-path");
	}

	return tags;
}
