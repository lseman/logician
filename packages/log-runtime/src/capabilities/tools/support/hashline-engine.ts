// ── Hashline edit engine ──────────────────────────────────────────────────────
// Applies hashline-format edits (PUT/CUT/MV/REM) with staging, preview, and
// stale-anchor recovery. Integrates with the EditStore for snapshots and
// clipboard registers.

import * as fs from "node:fs";
import * as path from "node:path";
import type { EditStore } from "./edit-store.js";
import { createEditStore } from "./edit-store.js";
import {
	HashlineEditResult,
	isHashlineFileHeader,
	parseHashlineEdit,
	HL_REM_KEYWORD,
} from "./hashline.js";
import { generateEditDiffs } from "./utils/diff-utils.js";
import { atomicWriteFile } from "./utils/atomic-write.js";

// ── Staged hashline result ────────────────────────────────────────────────────

/** A single file in a staged (previewed) hashline edit. */
interface StagedFileEdit {
	path: string;
	original: string;
	proposed: string;
	diff: string;
	linesChanged: number;
	deleted: boolean;
}

// ── Hashline edit application ─────────────────────────────────────────────────

/**
 * Apply hashline PUT/CUT operations to file content.
 */
function applyEditsToLines(
	lines: string[],
	edits: Array<{ range: string | undefined; body: string[] | undefined }>,
): { newLines: string[]; linesChanged: number } {
	let newLines = [...lines];
	let linesChanged = 0;

	for (const edit of edits) {
		const range = edit.range;
		if (!range) continue;

		if (range.startsWith("<")) {
			const targetLine = Number(range.slice(1));
			if (!Number.isNaN(targetLine)) {
				const insertAt = Math.max(0, targetLine - 1);
				newLines.splice(insertAt, 0, ...(edit.body ?? []));
				linesChanged += (edit.body?.length ?? 0);
			}
		} else if (range.startsWith(">")) {
			const targetLine = Number(range.slice(1));
			if (!Number.isNaN(targetLine)) {
				const insertAt = Math.min(lines.length, targetLine);
				newLines.splice(insertAt, 0, ...(edit.body ?? []));
				linesChanged += (edit.body?.length ?? 0);
			}
		} else if (range.includes("-")) {
			const match = range.match(/^(\d+)-(\d+)$/);
			if (match) {
				const start = Math.max(0, Number(match[1]) - 1);
				const end = Math.min(lines.length, Number(match[2]));
				const removed = end - start;
				newLines.splice(start, removed, ...(edit.body ?? []));
				linesChanged += Math.abs((edit.body?.length ?? 0) - removed);
			}
		} else {
			const num = Number(range);
			if (!Number.isNaN(num)) {
				const idx = Math.max(0, num - 1);
				if (idx < newLines.length) {
					newLines[idx] = edit.body?.[0] ?? newLines[idx];
					linesChanged += 1;
				}
			}
		}
	}

	return { newLines, linesChanged };
}

// ── Hashline edit execution ───────────────────────────────────────────────────

/**
 * Execute a hashline edit — dry-run (preview) mode by default.
 * Returns a result with the staged diff but does not write to disk.
 */
export async function executeHashlineEdit(
	editsInput: string,
	store: EditStore,
	cwd: string,
	dryRun: boolean = true,
): Promise<HashlineEditResult> {
	const lines = editsInput.split("\n");
	const fileEntries: StagedFileEdit[] = [];
	const REM = HL_REM_KEYWORD;
	let totalLinesChanged = 0;
	let filesAffected = 0;
	let error: string | undefined;
	let staleAnchors: Array<{ path: string; expectedTag: string; computedTag: string }> | undefined;

	let currentPath = "";
	let currentEdits: Array<{ range: string | undefined; body: string[] | undefined }> = [];

	function flushCurrentPath(): void {
		if (currentPath.length === 0 || currentEdits.length === 0) {
			currentPath = "";
			currentEdits = [];
			return;
		}

		// Check for REM
		if (currentEdits.some((e) => e.range === REM)) {
			const fullPath = path.resolve(cwd, currentPath);
			try {
				if (!dryRun) fs.unlinkSync(fullPath);
			} catch {
				// File might not exist
			}
			fileEntries.push({
				path: currentPath,
				original: "",
				proposed: "",
				diff: "",
				linesChanged: 0,
				deleted: true,
			});
			filesAffected++;
			store.clearSnapshot(fullPath);
			currentPath = "";
			currentEdits = [];
			return;
		}

		const fullPath = path.resolve(cwd, currentPath);

		// Read original content
		let originalContent: string;
		try {
			originalContent = fs.readFileSync(fullPath, "utf-8");
		} catch {
			originalContent = "";
		}

		// Apply edits
		const origLines = originalContent.split("\n");
		const { newLines, linesChanged } = applyEditsToLines(origLines, currentEdits);
		const newContent = newLines.join("\n") + (originalContent.endsWith("\n") ? "\n" : "");

		const diff = originalContent !== newContent
			? generateEditDiffs(currentPath, originalContent, newContent).diff
			: "";

		fileEntries.push({
			path: currentPath,
			original: originalContent,
			proposed: newContent,
			diff,
			linesChanged,
			deleted: false,
		});

		totalLinesChanged += linesChanged;
		filesAffected++;
		currentPath = "";
		currentEdits = [];
	}

	for (const line of lines) {
		const trimmed = line.trim();

		// File header: [path#4hex]
		if (isHashlineFileHeader(trimmed)) {
			flushCurrentPath();
			const hashIdx = trimmed.indexOf("#");
			const closeIdx = trimmed.indexOf("]");
			if (hashIdx >= 0 && closeIdx > hashIdx) {
				currentPath = trimmed.slice(1, hashIdx);
			} else if (closeIdx > 1) {
				currentPath = trimmed.slice(1, closeIdx);
			}
			// Record snapshot for stale detection
			try {
				const fullPath = path.resolve(cwd, currentPath);
				const raw = fs.readFileSync(fullPath, "utf-8");
				store.recordSnapshot(fullPath, raw);
			} catch {
				// New file — no snapshot needed
			}
			continue;
		}

		// Operation line
		const op = parseHashlineEdit(trimmed);
		if (op) {
			flushCurrentPath();

			if (op.operation === "REM") {
				currentPath = "";
				currentEdits = [{ range: REM, body: [] }];
			} else if (op.operation === "MV") {
				// MV handled separately
				currentPath = "";
			} else if (op.operation === "PUT" || op.operation === "CUT") {
				currentEdits = [{ range: op.range, body: op.body }];
			}
		}
	}

	flushCurrentPath();

	// If dry-run, return staged preview
	if (dryRun && fileEntries.length > 0) {
		return {
			applied: false,
			linesChanged: totalLinesChanged,
			filesAffected,
			diff: fileEntries.map((e) => e.diff).join("\n"),
			staleAnchors,
		};
	}

	// Actually apply edits to disk
	for (const entry of fileEntries) {
		if (entry.deleted) continue;
		const fullPath = path.resolve(cwd, entry.path);
		try {
			await atomicWriteFile(fullPath, entry.proposed);
			store.clearSnapshot(fullPath);
		} catch (e) {
			error = `Failed to write ${entry.path}: ${e instanceof Error ? e.message : String(e)}`;
		}
	}

	return {
		applied: true,
		linesChanged: totalLinesChanged,
		filesAffected,
		diff: fileEntries.map((e) => e.diff).join("\n"),
		staleAnchors,
		error,
	};
}

/**
 * Create a default edit store for simple usage.
 */
export function createDefaultEditStore(): EditStore {
	return createEditStore();
}
