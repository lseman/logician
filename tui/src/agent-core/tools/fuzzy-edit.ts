// ── Fuzzy Edit ─────────────────────────────────────────────────────────────────
// Fuzzy whitespace matching + edit application for the edit_file tool.
// Extracted from helpers.ts to reduce its size from 595 → ~80 lines.

import { normalizeForFuzzyMatch } from "./helpers.ts";

export interface Edit {
	oldText: string;
	newText: string;
}

export interface ApplyEditsResult {
	baseContent: string;
	newContent: string;
}

/**
 * Map positions from fuzzy-normalized content back to actual content positions.
 * Returns a mapping array where mapping[fuzzyPos] = actualPos.
 */
function buildPosMapping(
	actual: string,
	fuzzy: string,
): number[] {
	const mapping: number[] = [];
	let actualPos = 0;
	let fuzzyPos = 0;

	while (fuzzyPos < fuzzy.length) {
		const fuzzyChar = fuzzy[fuzzyPos];

		if (actualPos >= actual.length) {
			mapping[fuzzyPos] = actualPos;
			fuzzyPos++;
			continue;
		}

		const actualChar = actual[actualPos];

		if (fuzzyChar === ' ') {
			if (actualChar === ' ' || actualChar === '\t' || actualChar === '\n') {
				mapping[fuzzyPos] = actualPos;
				actualPos++;
			} else {
				mapping[fuzzyPos] = actualPos;
				fuzzyPos++;
			}
		} else {
			while (
				actualPos < actual.length &&
				(actual[actualPos] === ' ' || actual[actualPos] === '\t' || actual[actualPos] === '\n')
			) {
				actualPos++;
			}
			mapping[fuzzyPos] = actualPos;
			actualPos++;
			fuzzyPos++;
		}
	}

	return mapping;
}

/**
 * Search for oldText in content starting from a given position,
 * with flexible whitespace matching.
 * Returns the end position after the match, or -1 if not found.
 */
function searchForText(
	content: string,
	oldText: string,
	startPos: number,
): number {
	let ci = startPos;
	let ti = 0;
	let matchEnd = -1;

	while (ci < content.length && ti < oldText.length) {
		const c = content[ci];
		const t = oldText[ti];

		if (c === ' ' || c === '\t' || c === '\n') {
			ci++;
			continue;
		}

		if (t === ' ' && c !== ' ') {
			ti++;
			continue;
		}

		if (c.toLowerCase() === t.toLowerCase()) {
			matchEnd = ci + 1;
			ci++;
			ti++;
		} else {
			if (matchEnd !== -1 && matchEnd !== ci) {
				ci = matchEnd + 1;
				matchEnd = -1;
			} else if (ti > 0) {
				break;
			} else {
				ci++;
			}
		}
	}

	if (ti === oldText.length) {
		return ci;
	} else if (matchEnd !== -1 && ti > 0) {
		let endPos = matchEnd;
		while (endPos < content.length && content[endPos] === ' ') endPos++;
		return endPos;
	}

	return -1;
}

/**
 * Apply one or more edits to normalized content using fuzzy whitespace matching.
 * Falls back to exact matching if fuzzy doesn't find the oldText.
 */
export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	filePath: string,
): ApplyEditsResult {
	const fuzzyNormalized = normalizeForFuzzyMatch(normalizedContent);

	const sortedEdits = edits.map((edit, i) => ({ ...edit, originalIndex: i }))
		.sort((a, b) => a.oldText.length - b.oldText.length);

	// For each edit, find oldText in the content
	const editPositions: Array<{ start: number; end: number; oldText: string; newText: string }> = [];

	for (const edit of sortedEdits) {
		if (!edit.oldText) continue;

		const fuzzyOldText = normalizeForFuzzyMatch(edit.oldText);
		const fuzzyMatchPos = fuzzyNormalized.indexOf(fuzzyOldText);

		if (fuzzyMatchPos !== -1) {
			const fuzzyMapping = buildPosMapping(normalizedContent, fuzzyNormalized);
			const actualStart = fuzzyMapping[fuzzyMatchPos] ?? fuzzyMatchPos;
			const actualEnd = fuzzyMapping[fuzzyMatchPos + fuzzyOldText.length - 1] ?? fuzzyMatchPos + fuzzyOldText.length;

			editPositions.push({
				start: actualStart,
				end: actualEnd,
				oldText: edit.oldText,
				newText: edit.newText,
			});
		} else {
			// Fuzzy failed — try exact match
			const exactPos = normalizedContent.indexOf(edit.oldText);
			if (exactPos !== -1) {
				editPositions.push({
					start: exactPos,
					end: exactPos + edit.oldText.length,
					oldText: edit.oldText,
					newText: edit.newText,
				});
			}
		}
	}

	// Apply edits in reverse order to maintain positions
	let newContent = normalizedContent;
	for (let i = editPositions.length - 1; i >= 0; i--) {
		const { start, end, newText } = editPositions[i];
		newContent = newContent.slice(0, start) + newText + newContent.slice(end);
	}

	return {
		baseContent: normalizedContent,
		newContent,
	};
}
