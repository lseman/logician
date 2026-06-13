// ── Diff Utilities ─────────────────────────────────────────────────────────────
// Diff generation and patch utilities for the edit_file tool.
// Extracted from helpers.ts to reduce its size.

import * as path from "node:path";

export interface EditDiffResult {
	diff: string;
	firstChangedLine: number | undefined;
}

/** Generate a unified diff string with first-line tracking. */
export function generateDiffString(
	before: string,
	after: string,
): EditDiffResult {
	if (before === after) {
		return { diff: "", firstChangedLine: undefined };
	}

	const diff = syntheticUnifiedDiff("/dev/edit", before, after);

	const beforeLines = before.split("\n");
	const afterLines = after.split("\n");
	let firstChangedLine: number | undefined;

	let i = 0;
	let j = 0;
	let found = false;
	while (i < beforeLines.length && j < afterLines.length && !found) {
		if (beforeLines[i] !== afterLines[j]) {
			firstChangedLine = j + 1;
			found = true;
		} else {
			i++;
			j++;
		}
	}

	if (!found && beforeLines.length !== afterLines.length) {
		firstChangedLine = Math.min(afterLines.length, beforeLines.length + 1);
	}

	return { diff, firstChangedLine };
}

/** Generate a unified patch format string. */
export function generateUnifiedPatch(
	filePath: string,
	before: string,
	after: string,
): string {
	if (before === after) return "";
	return syntheticUnifiedDiff(filePath, before, after);
}

/** Generate a minimal unified diff (no file headers beyond the basic ones). */
export function syntheticUnifiedDiff(
	filePath: string,
	before: string | null,
	after: string,
): string {
	const beforeLines = (before ?? "").split("\n");
	const afterLines = after.split("\n");
	const beforeLabel = before === null ? "/dev/null" : `a/${path.basename(filePath)}`;
	const afterLabel = `b/${path.basename(filePath)}`;

	let prefix = 0;
	while (
		prefix < beforeLines.length &&
		prefix < afterLines.length &&
		beforeLines[prefix] === afterLines[prefix]
	) {
		prefix++;
	}

	let beforeSuffix = beforeLines.length - 1;
	let afterSuffix = afterLines.length - 1;
	while (
		beforeSuffix >= prefix &&
		afterSuffix >= prefix &&
		beforeLines[beforeSuffix] === afterLines[afterSuffix]
	) {
		beforeSuffix--;
		afterSuffix--;
	}

	const contextBefore = Math.max(0, prefix - 3);
	const contextAfterBefore = Math.min(beforeLines.length - 1, beforeSuffix + 3);
	const contextAfterAfter = Math.min(afterLines.length - 1, afterSuffix + 3);
	const oldStart = contextBefore + 1;
	const newStart = contextBefore + 1;
	const oldCount = Math.max(0, contextAfterBefore - contextBefore + 1);
	const newCount = Math.max(0, contextAfterAfter - contextBefore + 1);

	const out = [
		`--- ${beforeLabel}`,
		`+++ ${afterLabel}`,
		`@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`,
	];

	for (let i = contextBefore; i < prefix; i++)
		out.push(` ${beforeLines[i] ?? ""}`);
	for (let i = prefix; i <= beforeSuffix; i++)
		out.push(`-${beforeLines[i] ?? ""}`);
	for (let i = prefix; i <= afterSuffix; i++)
		out.push(`+${afterLines[i] ?? ""}`);

	const afterContextStart = Math.max(prefix, afterSuffix + 1);
	for (let i = afterContextStart; i <= contextAfterAfter; i++)
		out.push(` ${afterLines[i] ?? ""}`);

	return out.join("\n");
}

/** Summarize a diff for display when it's too large. */
export function summarizeDiff(diff: string, maxChars = 12000): string {
	if (!diff.trim()) return "(no diff)";
	if (diff.length <= maxChars) return diff;
	return diff.slice(0, maxChars) + "\n\n...(truncated)";
}
