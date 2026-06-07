// ── Word navigation ───────────────────────────────────────────────────────────
// Grapheme-aware word boundary finding for cursor movement.

import { getGraphemeSegmenter } from "./utils.ts";

const segmenter = getGraphemeSegmenter();

export function findWordBackward(text: string, cursor: number): number {
	if (cursor === 0) return 0;

	const segments = [...segmenter.segment(text)].map((s) => s.segment);
	const segCursor = Math.min(cursor, segments.length);

	// Skip non-word chars going backward
	let i = segCursor;
	while (i > 0 && !isWordChar(segments[i - 1])) i--;
	// Skip word chars going backward
	while (i > 0 && isWordChar(segments[i - 1])) i--;
	return i;
}

export function findWordForward(text: string, cursor: number): number {
	if (cursor >= text.length) return text.length;

	const segments = [...segmenter.segment(text)].map((s) => s.segment);
	const segCursor = Math.min(cursor, segments.length);

	// Skip non-word chars going forward
	let i = segCursor;
	while (i < segments.length && !isWordChar(segments[i])) i++;

	// Skip word chars going forward
	while (i < segments.length && isWordChar(segments[i])) i++;
	return i;
}

function isWordChar(seg: string): boolean {
	if (!seg || seg.length === 0) return false;
	const ch = seg.trim();
	if (!ch) return false;
	const c = ch.charCodeAt(0);
	return (
		(c >= 0x30 && c <= 0x39) || // 0-9
		(c >= 0x41 && c <= 0x5a) || // A-Z
		(c >= 0x61 && c <= 0x7a) || // a-z
		c === 0x5f
	); // _
}
