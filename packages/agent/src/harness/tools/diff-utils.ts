// ── Diff Utilities ────────────────────────────────────────────────────────
// Line-level diff and unified-patch generation for the edit/write tools.
// Uses an LCS-based line diff so disjoint changes produce separate hunks — a
// naive common-prefix/suffix diff renders everything between the first and
// last change as removed+re-added, which reads as "the whole file changed"
// for multi-edits and replaceAll. Ported from coding-agent's tools/diff-utils.ts.

import * as path from "node:path";

// ── Line diff (LCS) ──────────────────────────────────────────────────────

interface DiffOp {
	type: "equal" | "del" | "add";
	line: string;
}

// Above this many DP cells, fall back to the cheap prefix/suffix diff rather
// than risk memory/CPU blowups on pathological inputs.
const MAX_LCS_CELLS = 4_000_000;

/**
 * Diff two line arrays into equal/del/add operations. Trims the common
 * prefix/suffix first (typical edits leave a small middle), then runs an LCS
 * over the remainder. Falls back to a whole-block replace when the middle is
 * too large for the DP table.
 */
function diffOps(before: string[], after: string[]): DiffOp[] {
	let prefix = 0;
	while (
		prefix < before.length &&
		prefix < after.length &&
		before[prefix] === after[prefix]
	) {
		prefix++;
	}
	let suffix = 0;
	while (
		suffix < before.length - prefix &&
		suffix < after.length - prefix &&
		before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
	) {
		suffix++;
	}

	const a = before.slice(prefix, before.length - suffix);
	const b = after.slice(prefix, after.length - suffix);

	const ops: DiffOp[] = [];
	for (let i = 0; i < prefix; i++) {
		const line = before[i];
		if (line !== undefined) ops.push({ type: "equal", line });
	}
	ops.push(...diffMiddle(a, b));
	for (let i = before.length - suffix; i < before.length; i++) {
		const line = before[i];
		if (line !== undefined) ops.push({ type: "equal", line });
	}
	return ops;
}

function diffMiddle(a: string[], b: string[]): DiffOp[] {
	if (a.length === 0 && b.length === 0) return [];
	if (a.length === 0) return b.map(line => ({ type: "add" as const, line }));
	if (b.length === 0) return a.map(line => ({ type: "del" as const, line }));

	// Fallback: whole-block replace when LCS would be too expensive.
	if (a.length * b.length > MAX_LCS_CELLS) {
		return [
			...a.map(line => ({ type: "del" as const, line })),
			...b.map(line => ({ type: "add" as const, line })),
		];
	}

	// LCS lengths table (rows a, cols b), then backtrack into ops.
	const cols = b.length + 1;
	const table = new Int32Array((a.length + 1) * cols);
	for (let i = a.length - 1; i >= 0; i--) {
		for (let j = b.length - 1; j >= 0; j--) {
			table[i * cols + j] =
				a[i] === b[j]
					? (table[(i + 1) * cols + j + 1] ?? 0) + 1
					: Math.max(
							table[(i + 1) * cols + j] ?? 0,
							table[i * cols + j + 1] ?? 0,
						);
		}
	}

	const ops: DiffOp[] = [];
	let i = 0;
	let j = 0;
	while (i < a.length && j < b.length) {
		if (a[i] === b[j]) {
			ops.push({ type: "equal", line: a[i] ?? "" });
			i++;
			j++;
		} else if (
			(table[(i + 1) * cols + j] ?? 0) >= (table[i * cols + j + 1] ?? 0)
		) {
			ops.push({ type: "del", line: a[i] ?? "" });
			i++;
		} else {
			ops.push({ type: "add", line: b[j] ?? "" });
			j++;
		}
	}
	for (; i < a.length; i++) ops.push({ type: "del", line: a[i] ?? "" });
	for (; j < b.length; j++) ops.push({ type: "add", line: b[j] ?? "" });
	return ops;
}

// ── Unified format ────────────────────────────────────────────────────────

const CONTEXT_LINES = 3;

/** Render diff ops as unified hunks with headers. */
function renderUnified(
	beforeLabel: string,
	afterLabel: string,
	ops: DiffOp[],
): string {
	// Identify changed-op indices; group into hunks when gaps exceed 2*context.
	const changed: number[] = [];
	for (let i = 0; i < ops.length; i++) {
		if (ops[i]?.type !== "equal") changed.push(i);
	}
	if (changed.length === 0) return "";

	interface Hunk {
		start: number; // first op index (inclusive)
		end: number; // last op index (inclusive)
	}
	const firstChanged = changed[0] ?? 0;
	const hunks: Hunk[] = [];
	let current: Hunk = {
		start: Math.max(0, firstChanged - CONTEXT_LINES),
		end: Math.min(ops.length - 1, firstChanged + CONTEXT_LINES),
	};
	for (let c = 1; c < changed.length; c++) {
		const changedIndex = changed[c] ?? 0;
		const start = Math.max(0, changedIndex - CONTEXT_LINES);
		if (start <= current.end + 1) {
			current.end = Math.min(ops.length - 1, changedIndex + CONTEXT_LINES);
		} else {
			hunks.push(current);
			current = {
				start,
				end: Math.min(ops.length - 1, changedIndex + CONTEXT_LINES),
			};
		}
	}
	hunks.push(current);

	const out = [`--- ${beforeLabel}`, `+++ ${afterLabel}`];
	// Track line numbers: oldLine/newLine are 1-based positions of the NEXT op.
	let oldLine = 1;
	let newLine = 1;
	let opIndex = 0;
	for (const hunk of hunks) {
		// Advance counters over ops before the hunk.
		for (; opIndex < hunk.start; opIndex++) {
			const op = ops[opIndex];
			if (op?.type !== "add") oldLine++;
			if (op?.type !== "del") newLine++;
		}
		const oldStart = oldLine;
		const newStart = newLine;
		let oldCount = 0;
		let newCount = 0;
		const body: string[] = [];
		for (; opIndex <= hunk.end; opIndex++) {
			const op = ops[opIndex];
			if (!op) continue;
			if (op.type === "equal") {
				body.push(` ${op.line}`);
				oldCount++;
				newCount++;
				oldLine++;
				newLine++;
			} else if (op.type === "del") {
				body.push(`-${op.line}`);
				oldCount++;
				oldLine++;
			} else {
				body.push(`+${op.line}`);
				newCount++;
				newLine++;
			}
		}
		out.push(`@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`);
		out.push(...body);
	}
	return out.join("\n");
}

// ── Public API ────────────────────────────────────────────────────────────

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

	const ops = diffOps(before.split("\n"), after.split("\n"));
	const diff = renderUnified("a/edit", "b/edit", ops);

	// First changed line, numbered in the AFTER content.
	let firstChangedLine: number | undefined;
	let newLine = 1;
	for (const op of ops) {
		if (op.type !== "equal") {
			firstChangedLine = newLine;
			break;
		}
		newLine++;
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

/** Generate a unified diff between two file states (multi-hunk, 3 context lines). */
export function syntheticUnifiedDiff(
	filePath: string,
	before: string | null,
	after: string,
): string {
	const beforeLabel =
		before === null ? "/dev/null" : `a/${path.basename(filePath)}`;
	const afterLabel = `b/${path.basename(filePath)}`;
	const ops: DiffOp[] =
		before === null
			? after.split("\n").map(line => ({ type: "add" as const, line }))
			: diffOps(before.split("\n"), after.split("\n"));
	return renderUnified(beforeLabel, afterLabel, ops);
}

/** Summarize a diff for display when it's too large. */
export function summarizeDiff(diff: string, maxChars = 512000): string {
	if (!diff.trim()) return "(no diff)";
	if (diff.length <= maxChars) return diff;
	return `${diff.slice(0, maxChars)}\n\n...(truncated)`;
}
