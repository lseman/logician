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

function isWhitespaceChar(ch: string): boolean {
	return ch.trim().length === 0;
}

// ── Word boundaries (for grapheme cursor) ─────────────────────────────────────

function graphemeLength(text: string): number {
	const segmenter = getGraphemeSegmenter();
	return [...segmenter.segment(text)].length;
}

function graphemeSlice(text: string, from: number, to?: number): string {
	const segmenter = getGraphemeSegmenter();
	const segments = [...segmenter.segment(text)];
	const end = to !== undefined ? to : segments.length;
	return segments
		.slice(from, end)
		.map(s => s.segment)
		.join("");
}

// ── Fuzzy matching (for slash commands) ───────────────────────────────────────

interface FuzzyScore {
	index: number;
	score: number;
}

function fuzzyMatch(query: string, candidate: string): FuzzyScore | null {
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

function fuzzyFilter<T>(
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
