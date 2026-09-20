// ── Diff Utilities ─────────────────────────────────────────────────────────────
// Line-level diff and unified-patch generation for the edit / write tools.
// Backed by @logician/log-natives' native structuredPatchHunks() (a Myers
// diff, ported from oh-my-pi's pi-diff) instead of a hand-rolled JS LCS, so
// disjoint changes still produce separate hunks — a naive common-prefix/suffix
// diff renders everything between the first and last change as removed+re-added,
// which reads as "the whole file changed" for multi-edits and replaceAll.

import * as path from "node:path";
import { loadNative, type NativeModule } from "../native-addon.ts";

type PatchHunk = Awaited<
	ReturnType<NativeModule["structuredPatchHunks"]>
>[number];

// ============================================================================
// Unified format
// ============================================================================

const CONTEXT_LINES = 3;

/** Render precomputed hunks as unified diff text with headers. */
function renderUnifiedFromHunks(
	beforeLabel: string,
	afterLabel: string,
	hunks: PatchHunk[],
): string {
	if (hunks.length === 0) return "";
	const out = [`--- ${beforeLabel}`, `+++ ${afterLabel}`];
	for (const hunk of hunks) {
		out.push(
			`@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`,
		);
		out.push(...hunk.lines);
	}
	return out.join("\n");
}

const NO_NEWLINE_MARKER = "\\ No newline at end of file";

/**
 * Derive `firstChangedLine`/`linesChanged` from the same hunks used to render
 * the diff text, so both come from a single native diff call per edit.
 * `linesChanged` counts by contiguous change run — max(dels, adds) per run,
 * not len(dels) + len(adds) — so a plain 1-line substitution (one removed
 * line + one added line) counts as 1 changed line, not 2.
 */
function changeMetricsFromHunks(hunks: PatchHunk[]): {
	firstChangedLine: number | undefined;
	linesChanged: number;
} {
	let firstChangedLine: number | undefined;
	let linesChanged = 0;
	for (const hunk of hunks) {
		let newLine = hunk.newStart;
		const lines = hunk.lines;
		let i = 0;
		while (i < lines.length) {
			const line = lines[i] ?? "";
			if (line.startsWith(NO_NEWLINE_MARKER)) {
				i++;
				continue;
			}
			if (line.startsWith(" ")) {
				newLine++;
				i++;
				continue;
			}
			firstChangedLine ??= newLine;
			let dels = 0;
			let adds = 0;
			while (i < lines.length) {
				const l = lines[i] ?? "";
				if (l.startsWith(NO_NEWLINE_MARKER)) {
					i++;
					continue;
				}
				if (l.startsWith(" ")) break;
				if (l.startsWith("-")) dels++;
				else {
					adds++;
					newLine++;
				}
				i++;
			}
			linesChanged += Math.max(dels, adds);
		}
	}
	return { firstChangedLine, linesChanged };
}

// ============================================================================
// Public API
// ============================================================================

export interface EditDiffResult {
	diff: string;
	firstChangedLine: number | undefined;
	/** Count of added/removed lines (equal lines don't count). */
	linesChanged: number;
}

/**
 * Generate the display diff and the unified patch from a single native
 * structuredPatchHunks() call. Both formats, plus the change metrics, are
 * derived from the same hunks on every edit call, so computing them once
 * avoids doubling the native diff cost for no benefit.
 */
export async function generateEditDiffs(
	filePath: string,
	before: string,
	after: string,
): Promise<EditDiffResult & { patch: string }> {
	if (before === after) {
		return {
			diff: "",
			firstChangedLine: undefined,
			linesChanged: 0,
			patch: "",
		};
	}

	const native = await loadNative();
	const hunks = native.structuredPatchHunks(before, after, CONTEXT_LINES);
	const diff = renderUnifiedFromHunks("a/edit", "b/edit", hunks);
	const patch = renderUnifiedFromHunks(
		`a/${path.basename(filePath)}`,
		`b/${path.basename(filePath)}`,
		hunks,
	);
	const { firstChangedLine, linesChanged } = changeMetricsFromHunks(hunks);

	return { diff, firstChangedLine, linesChanged, patch };
}

/** Generate a unified diff between two file states (multi-hunk, 3 context lines). */
export async function syntheticUnifiedDiff(
	filePath: string,
	before: string | null,
	after: string,
): Promise<string> {
	const beforeLabel =
		before === null ? "/dev/null" : `a/${path.basename(filePath)}`;
	const afterLabel = `b/${path.basename(filePath)}`;
	const native = await loadNative();
	const hunks = native.structuredPatchHunks(before ?? "", after, CONTEXT_LINES);
	return renderUnifiedFromHunks(beforeLabel, afterLabel, hunks);
}

/** Summarize a diff for display when it's too large. */
export function summarizeDiff(diff: string, maxChars = 512000): string {
	if (!diff.trim()) return "(no diff)";
	if (diff.length <= maxChars) return diff;
	return `${diff.slice(0, maxChars)}\n\n...(truncated)`;
}
