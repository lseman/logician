// ── Utility primitives ────────────────────────────────────────────────────────
// Grapheme segmentation, visible width, text wrapping, fuzzy matching

// ── Grapheme segmenter (Unicode-aware) ────────────────────────────────────────

export const getGraphemeSegmenter = (): Intl.Segmenter => {
	try {
		return new (
			Intl as unknown as { Segmenter: typeof Intl.Segmenter }
		).Segmenter(undefined, {
			segmenter: "grapheme",
		} as unknown as Intl.SegmenterOptions);
	} catch (_e: unknown) {
		// Fallback: Intl.Segmenter not available — BMP-only splitter
		return {
			segment(text: string): Iterable<{ segment: string; segmented: boolean }> {
				const chars = [...text];
				return {
					*[Symbol.iterator]() {
						for (const ch of chars) {
							yield { segment: ch, segmented: false };
						}
					},
				};
			},
		} as unknown as Intl.Segmenter;
	}
};

// ── Word navigation ───────────────────────────────────────────────────────────

export function findWordBackward(text: string, cursor: number): number {
	if (cursor === 0) return 0;
	const segmenter = getGraphemeSegmenter();
	const segments = [...segmenter.segment(text)].map(s => s.segment);

	// Skip non-word chars going backward
	let i = cursor;
	while (i > 0 && !isWordChar(segments[i - 1])) i--;
	const _start = i;

	// Skip word chars going backward
	while (i > 0 && isWordChar(segments[i - 1])) i--;
	return i;
}

export function findWordForward(text: string, cursor: number): number {
	if (cursor >= text.length) return text.length;
	const segmenter = getGraphemeSegmenter();
	const segments = [...segmenter.segment(text)].map(s => s.segment);

	// Skip non-word chars going forward
	let i = cursor;
	while (i < segments.length && !isWordChar(segments[i])) i++;
	const _start = i;

	// Skip word chars going forward
	while (i < segments.length && isWordChar(segments[i])) i++;
	return i;
}

function isWordChar(seg: string): boolean {
	const ch = seg.trim();
	if (!ch) return false;
	const c = ch.charCodeAt(0);
	return (
		(c >= 0x30 && c <= 0x39) || // 0-9
		(c >= 0x41 && c <= 0x5a) || // A-Z
		(c >= 0x61 && c <= 0x7a) || // a-z
		c === 0x5f || // _
		c >= 0x0100
	); // Unicode letter/digit
}

export function isWhitespaceChar(ch: string): boolean {
	return ch.trim().length === 0;
}

// ── Word boundaries (for grapheme cursor) ─────────────────────────────────────

export function graphemeLength(text: string): number {
	const segmenter = getGraphemeSegmenter();
	return [...segmenter.segment(text)].length;
}

export function graphemeSlice(text: string, from: number, to?: number): string {
	const segmenter = getGraphemeSegmenter();
	const segments = [...segmenter.segment(text)];
	const end = to !== undefined ? to : segments.length;
	return segments
		.slice(from, end)
		.map(s => s.segment)
		.join("");
}

// ── Fuzzy matching (for slash commands) ───────────────────────────────────────

export interface FuzzyScore {
	index: number;
	score: number;
}

export function fuzzyMatch(
	query: string,
	candidate: string,
): FuzzyScore | null {
	const q = query.toLowerCase();
	const c = candidate.toLowerCase();

	if (!q) return { index: 0, score: 1000 };

	// Exact match
	if (c === q) return { index: 0, score: 2000 };

	// Prefix match
	if (c.startsWith(q)) {
		return { index: 0, score: 1800 - (c.length - q.length) };
	}

	// Subsequence match
	let qi = 0;
	let gap = 0;
	let lastMatch = -1;

	for (let ci = 0; ci < c.length && qi < q.length; ci++) {
		if (c[ci] === q[qi]) {
			if (lastMatch >= 0) {
				gap += ci - lastMatch - 1;
			}
			lastMatch = ci;
			qi++;
		}
	}

	if (qi === q.length) {
		// Bonus for matching at start
		const startBonus = c.indexOf(q[0]) === 0 ? 500 : 0;
		return { index: 0, score: 1200 - gap * 10 + startBonus };
	}

	// Description match (lower priority)
	if (c.includes(q)) {
		return { index: 0, score: 600 - c.indexOf(q) };
	}

	return null;
}

export function fuzzyFilter<T>(
	items: T[],
	query: string,
	getKey: (item: T) => string,
): (T & { score: number })[] {
	const scored = items
		.map(item => {
			const key = getKey(item);
			const result = fuzzyMatch(query, key);
			if (!result) {
				// Check description too
				const descMatch = fuzzyMatch(query, key.split(" ").pop() || "");
				if (!descMatch) return null;
				return { item, score: descMatch.score };
			}
			return { item, score: result.score };
		})
		.filter(Boolean) as unknown as (T & { score: number })[];

	scored.sort((a, b) => b.score - a.score);
	return scored;
}
