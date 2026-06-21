// ── Helpers ───────────────────────────────────────────────────────────────────
// Text utilities: BOM handling, line endings, edit operations, mutation summary.

// ============================================================================
// BOM handling
// ============================================================================

const BOM = "\uFEFF";

/** Strip UTF-8 BOM from the start of a string. Returns { bom, text }. */
export function stripBom(content: string): { bom: string; text: string } {
	if (content.charCodeAt(0) === 0xfeff) {
		return { bom: BOM, text: content.slice(1) };
	}
	return { bom: "", text: content };
}

// ============================================================================
// Line ending handling
// ============================================================================

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

// ============================================================================
// Edit operations interface
// ============================================================================

import * as fs from "node:fs";
import { access, readFile, writeFile } from "node:fs/promises";

export interface EditOperations {
	readFile: (absolutePath: string) => Promise<Buffer>;
	writeFile: (absolutePath: string, content: string) => Promise<void>;
	access: (absolutePath: string) => Promise<void>;
}

const defaultEditOperations: EditOperations = {
	readFile: (p) => readFile(p, "utf-8").then((b) => Buffer.from(b)),
	writeFile: (p, content) => writeFile(p, content, "utf-8"),
	access: (p) => access(p, fs.constants.R_OK | fs.constants.W_OK),
};

export { defaultEditOperations };

import { summarizeDiff, syntheticUnifiedDiff } from "../skills/diff-utils.ts";

/** Generate a summary diff for file mutation reporting. */
export async function mutationSummary(
	_cwd: string | undefined,
	filePath: string,
	before: string | null,
	after: string,
): Promise<string> {
	return summarizeDiff(syntheticUnifiedDiff(filePath, before, after));
}
