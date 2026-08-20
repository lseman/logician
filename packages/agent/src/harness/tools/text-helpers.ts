// ── Text helpers ──────────────────────────────────────────────────────────
// BOM handling, line-ending normalization, and mutation-diff summaries shared
// by the edit/write tools. Ported from coding-agent's tools/helpers.ts,
// dropping the raw-fs EditOperations abstraction — tools go through
// ExecutionEnv directly instead.

import { summarizeDiff, syntheticUnifiedDiff } from "./diff-utils.ts";

// ── BOM handling ──────────────────────────────────────────────────────────

const BOM = "﻿";

/** Strip UTF-8 BOM from the start of a string. Returns { bom, text }. */
export function stripBom(content: string): { bom: string; text: string } {
	if (content.charCodeAt(0) === 0xfeff) {
		return { bom: BOM, text: content.slice(1) };
	}
	return { bom: "", text: content };
}

// ── Line ending handling ──────────────────────────────────────────────────

const CRLF = "\r\n";
const LF = "\n";

/** Detect whether the content uses CRLF or LF line endings. */
export function detectLineEnding(content: string): string {
	return content.includes(CRLF) ? CRLF : LF;
}

/** Normalize all line endings to LF. */
export function normalizeToLF(content: string): string {
	return content.replace(/\r\n/g, LF);
}

/** Convert LF line endings back to the detected original line ending style. */
export function restoreLineEndings(
	content: string,
	lineEnding: string,
): string {
	if (lineEnding === CRLF) {
		return content.replace(/\n/g, CRLF);
	}
	return content;
}

/** Generate a summary diff for file mutation reporting. */
export function mutationSummary(
	filePath: string,
	before: string | null,
	after: string,
): string {
	return summarizeDiff(syntheticUnifiedDiff(filePath, before, after));
}
