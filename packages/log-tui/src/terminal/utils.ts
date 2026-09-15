// ── Utility primitives ────────────────────────────────────────────────────────
// Grapheme segmentation, visible width, text wrapping, fuzzy matching

// ── Grapheme segmenter (Unicode-aware) ────────────────────────────────────────

export const getGraphemeSegmenter = (): Intl.Segmenter => {
	try {
		return new (
			Intl as unknown as { Segmenter: typeof Intl.Segmenter }
		).Segmenter(undefined, {
			granularity: "grapheme",
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
