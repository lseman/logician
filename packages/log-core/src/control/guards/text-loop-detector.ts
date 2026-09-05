/**
 * Text-level loop detector for assistant messages.
 *
 * Detects three categories of stagnation in assistant text:
 * 1. **Exact suffix cycles** — byte-for-byte repetition (Z-algorithm)
 * 2. **Near-duplicate paragraphs** — high word-trigram overlap (Jaccard similarity)
 * 3. **Vocabulary stall** — low novelty + no new concrete references
 *
 * Inspired by OMP's ThinkingLoopDetector from packages/ai/src/utils/thinking-loop.ts.
 * Adapted for assistant-message-level detection (per-message, not per-stream).
 */

// ── Exact suffix cycle detection ──────────────────────────────────────────────

/** Minimum repeated characters for short cycle detection. */
const EXACT_SHORT_MIN_REPEATED_CHARS = 180;
/** Maximum unit length for short cycle detection. */
const EXACT_SHORT_MAX_UNIT = 60;
/** Minimum characters for long cycle detection. */
const EXACT_LONG_MIN_REPEATED_CHARS = 1024;
/** Maximum unit length for long cycle detection. */
const EXACT_LONG_MAX_UNIT = 1024;

/**
 * Detect exact suffix cycles using Z-algorithm on reversed text.
 * Returns [unit, count] when a cycle is found, null otherwise.
 */
function detectExactSuffixCycle(text: string): [unit: string, count: number] | null {
	if (text.length < EXACT_SHORT_MIN_REPEATED_CHARS) return null;

	const reversed = text.split("").reverse().join("");
	const z = new Uint16Array(reversed.length);
	let left = 0;
	let right = 0;
	for (let i = 1; i < reversed.length; i++) {
		if (i <= right) z[i] = Math.min(right - i + 1, z[i - left]);
		while (i + z[i] < reversed.length && reversed[z[i]] === reversed[i + z[i]]) z[i]++;
		if (i + z[i] - 1 > right) {
			left = i;
			right = i + z[i] - 1;
		}
	}

	const maxUnit = Math.min(EXACT_LONG_MAX_UNIT, Math.floor(reversed.length / 3));
	for (let len = 2; len <= maxUnit; len++) {
		const count = 1 + Math.floor(z[len] / len);
		const minCount = len <= EXACT_SHORT_MAX_UNIT ? 4 : 3;
		const minChars = len <= EXACT_SHORT_MAX_UNIT ? EXACT_SHORT_MIN_REPEATED_CHARS : EXACT_LONG_MIN_REPEATED_CHARS;
		if (count < minCount || len * count < minChars) continue;
		const unit = text.slice(-len);
		// Must contain actual content (not just whitespace/punctuation)
		if (/\p{L}|\p{Extended_Pictographic}/u.test(unit)) return [unit, count];
	}
	return null;
}

// ── Semantic loop detection ───────────────────────────────────────────────────

/**
 * Normalize text for comparison: lowercase, tokenize, drop pure numbers.
 */
function normalizeText(text: string): string {
	return text
		.toLowerCase()
		.replace(/`([^`]*)`/g, " $1 ")
		.replace(/[^a-z0-9]+/g, " ")
		.split(/\s+/)
		.filter(token => /[a-z]/.test(token))
		.join(" ")
		.trim();
}

/** Word-trigram shingle set. */
function trigramShingles(text: string): Set<string> {
	const words = text.split(" ").filter(Boolean);
	if (words.length < 3) return new Set(words.length > 0 ? [words.join(" ")] : new Set());
	const shingles = new Set<string>();
	for (let i = 0; i + 3 <= words.length; i++) {
		shingles.add(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
	}
	return shingles;
}

/** Jaccard similarity between two sets. */
function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
	if (a.size === 0 || b.size === 0) return 0;
	const [small, large] = a.size < b.size ? [a, b] : [b, a];
	let intersection = 0;
	for (const x of small) {
		if (large.has(x)) intersection++;
	}
	const union = a.size + b.size - intersection;
	return union === 0 ? 0 : intersection / union;
}

/**
 * Concrete anchor regex: code spans, file extensions, paths, snake/camel/Pascal case.
 * These are references the model is actually reasoning about.
 */
const CONCRETE_ANCHOR =
	/`[^`]+`|\b\w{2,}\.[a-zA-Z]\w{0,4}\b|[\w-]+(?:\/[\w-]+){2,}|\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z]+[A-Z]\w*\b/g;

// ── Configuration ─────────────────────────────────────────────────────────────

/** How many recent segments to keep for similarity comparison. */
const SEGMENT_WINDOW = 12;
/** Jaccard threshold for near-duplicate detection. */
const SIMILARITY_THRESHOLD = 0.75;
/** Minimum substantial segment length (normalized). */
const SEGMENT_MIN_CHARS = 80;
/** Near-duplicate cluster size that trips detection. */
const CLUSTER_THRESHOLD = 3;
/** Vocabulary novelty floor (0-1). Below this = recycling wording. */
const NOVELTY_FLOOR = 0.25;
/** Consecutive low-novelty segments to trip stall detection. */
const STALL_THRESHOLD = 6;

// ── Main detector ─────────────────────────────────────────────────────────────

export interface TextLoopDetectorOptions {
	/** Disable semantic heuristics (only check exact cycles). */
	disableSemanticHeuristics?: boolean;
}

export class TextLoopDetector {
	#semanticHeuristics: boolean;
	/** Recent trigram fingerprints for similarity comparison. */
	#window: Set<string>[] = [];
	/** Recent word sets for vocabulary novelty. */
	#wordWindow: Set<string>[] = [];
	/** Recent concrete anchors. */
	#anchorWindow: Set<string>[] = [];
	/** Count of substantial segments processed. */
	#count = 0;
	/** Consecutive low-novelty segments. */
	#stallRun = 0;

	constructor(options: TextLoopDetectorOptions = {}) {
		this.#semanticHeuristics = !options.disableSemanticHeuristics;
	}

	/**
	 * Check a completed assistant message for loop/stagnation patterns.
	 * Returns a description of the detected pattern, or null if clean.
	 */
	check(text: string): string | null {
		// 1. Exact suffix cycle detection
		const exact = detectExactSuffixCycle(text);
		if (exact) {
			const [unit, count] = exact;
			return `Exact repetition detected: "${unit.slice(0, 40)}..." repeated ${count}×`;
		}

		if (!this.#semanticHeuristics) return null;

		// 2. Split text into segments (by blank lines or forced cap)
		const segments = splitSegments(text);
		if (segments.length === 0) return null;

		// 3. Check each segment
		for (const raw of segments) {
			const hit = this.#consumeSegment(raw);
			if (hit) return hit;
		}

		return null;
	}

	/**
	 * Check accumulated text from a stream of deltas.
	 * Call after each delta to detect loops incrementally.
	 */
	push(delta: string): string | null {
		// For incremental detection, we accumulate and check the suffix
		if (!delta) return null;
		return this.check(delta);
	}

	/**
	 * Reset the detector state. Call between turns.
	 */
	reset(): void {
		this.#window = [];
		this.#wordWindow = [];
		this.#anchorWindow = [];
		this.#count = 0;
		this.#stallRun = 0;
	}

	#consumeSegment(raw: string): string | null {
		// Strip heading/title lines (they change every paragraph and would mask loops)
		const segment = raw
			.replace(/^[ \t]*#{1,6}[ \t].*$/gm, "")
			.replace(/^[ \t]*\*{2,3}.+?\*{2,3}[ \t]*$/gm, "");

		const normalized = normalizeText(segment);
		if (normalized.length < SEGMENT_MIN_CHARS) return null;

		const trigrams = trigramShingles(normalized);
		const words = new Set<string>(normalized.split(" ").filter(Boolean));

		// (a) Near-duplicate cluster check
		let cluster = 1;
		for (const prev of this.#window) {
			if (jaccardSimilarity(trigrams, prev) >= SIMILARITY_THRESHOLD) cluster++;
		}

		// (b) Vocabulary novelty check
		const priorVocab = new Set<string>();
		for (const set of this.#wordWindow) for (const w of set) priorVocab.add(w);
		let unseen = 0;
		for (const w of words) if (!priorVocab.has(w)) unseen++;
		const novelty = priorVocab.size === 0 ? 1 : unseen / words.size;

		// (c) Concrete anchor check
		const anchors = new Set<string>();
		for (const match of segment.matchAll(CONCRETE_ANCHOR)) {
			anchors.add(match[0].replace(/`/g, "").toLowerCase());
		}
		let newAnchor = false;
		for (const anchor of anchors) {
			if (this.#anchorWindow.every(seen => !seen.has(anchor))) {
				newAnchor = true;
				break;
			}
		}

		// Track stall: low novelty + no new anchor
		if (novelty <= NOVELTY_FLOOR && !newAnchor) {
			this.#stallRun++;
		} else {
			this.#stallRun = 0;
		}

		// Update sliding windows
		this.#window.push(trigrams);
		if (this.#window.length > SEGMENT_WINDOW) this.#window.shift();
		this.#wordWindow.push(words);
		if (this.#wordWindow.length > SEGMENT_WINDOW) this.#wordWindow.shift();
		this.#anchorWindow.push(anchors);
		if (this.#anchorWindow.length > SEGMENT_WINDOW) this.#anchorWindow.shift();
		this.#count++;

		// After warm-up, check for problems
		if (this.#count >= 4) {
			if (cluster >= CLUSTER_THRESHOLD) {
				return `${cluster} near-identical paragraphs within the last ${SEGMENT_WINDOW}`;
			}
			if (this.#stallRun >= STALL_THRESHOLD) {
				return `${this.#stallRun} consecutive paragraphs recycling recent wording with no new references`;
			}
		}

		return null;
	}
}

/** Split text into paragraphs/segments by blank lines, with a char cap. */
function splitSegments(text: string): string[] {
	const segments: string[] = [];
	const CHARS_PER_SEGMENT = 700;

	// Split by double newlines
	const blocks = text.split(/\n\s*\n/);
	for (const block of blocks) {
		// Strip trailing/leading whitespace
		const trimmed = block.trim();
		if (trimmed.length < SEGMENT_MIN_CHARS) continue;

		// If block is too long, split into chunks
		let remaining = trimmed;
		while (remaining.length > 0) {
			const chunk = remaining.length > CHARS_PER_SEGMENT ? remaining.slice(0, CHARS_PER_SEGMENT) : remaining;
			remaining = remaining.slice(chunk.length).trim();
			if (chunk.length >= SEGMENT_MIN_CHARS) {
				segments.push(chunk);
			}
		}
	}

	return segments;
}
