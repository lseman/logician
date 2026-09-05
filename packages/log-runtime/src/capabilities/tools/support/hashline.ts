// ── Hashline anchor utilities ─────────────────────────────────────────────────
// Generates 4-char hex content hashes for file lines (hashline anchors) and
// provides the hashline format for the model-facing edit protocol.
//
// Hashline format:
//   [path#4hex]        — file header with 4-char content hash
//   1:line content     — numbered lines
//
// Edit operations (PUT/CUT/MV/REM):
//   PUT N.=M: / PUT <N: / PUT >N:          — line edits
//   PUT N*:@reg                              — block edit
//   PUT N.=M @reg / CUT N.=M @reg           — paste register
//   CUT N.=M / CUT N*                        — capture register
//   REM                                        — delete file
//   MV dest                                    — rename/move file

import { createHash } from "node:crypto";

// ── Constants ──────────────────────────────────────────────────────────────────

/** Length of hashline file tags (4 hex chars). */
export const HASHLINE_TAG_LENGTH = 4;

/** File header prefix: [path#4hex] */
export const HL_FILE_PREFIX = "[";
export const HL_FILE_SUFFIX = "]";
export const HL_FILE_HASH_SEP = "#";

/** Operation keywords */
export const HL_MOVE_KEYWORD = "MV";
export const HL_REM_KEYWORD = "REM";

/** Separator between line number and line content */
export const HL_LINE_BODY_SEP = ":";

// ── Hash generation ────────────────────────────────────────────────────────────

/**
 * Generate a 4-char hex content hash from text.
 * Uses a truncated SHA-256 for fast, collision-resistant hashing.
 */
export function hashlineHash(text: string): string {
	return createHash("sha256")
		.update(text, "utf-8")
		.digest("hex")
		.slice(0, HASHLINE_TAG_LENGTH);
}

/**
 * Format a file header with hashline anchors.
 * Returns: [path#4hex]
 */
export function formatHashlineHeader(path: string, tag: string): string {
	return `${HL_FILE_PREFIX}${path}${HL_FILE_HASH_SEP}${tag}${HL_FILE_SUFFIX}`;
}

// ── Hashline parsing ───────────────────────────────────────────────────────────

/**
 * Split text into addressable lines (no trailing empty line).
 */
export function splitAddressableFileLines(text: string): string[] {
	const lines = text.split("\n");
	if (lines.length > 0 && lines.at(-1) === "") lines.pop();
	return lines;
}

// ── Stale anchor detection ─────────────────────────────────────────────────────

/**
 * Compute a quick hash for stale detection (faster than full content hash).
 * Uses a truncated hash of line count + first/last line.
 */
export function quickFileFingerprint(text: string): string {
	const lines = text.split("\n");
	const firstLine = lines[0] ?? "";
	const lastLine = lines[lines.length - 1] ?? "";
	const fingerprint = `${lines.length}:${firstLine}|${lastLine}`;
	return createHash("sha256")
		.update(fingerprint, "utf-8")
		.digest("hex")
		.slice(0, 8);
}

/**
 * Check if a file has been modified since a fingerprint was recorded.
 */
export function isFileStale(currentFingerprint: string, recordedFingerprint: string): boolean {
	return currentFingerprint !== recordedFingerprint;
}

// ── Edit operation types ───────────────────────────────────────────────────────

/** A hashline edit operation. */
export interface HashlineEdit {
	/** The operation type. */
	operation: "PUT" | "CUT" | "MV" | "REM";
	/** Target file path. */
	path: string;
	/** For PUT/CUT: the line range (N.=M or N*). */
	range?: string;
	/** For PUT: replacement content lines. */
	body?: string[];
	/** For MV: destination path. */
	dest?: string;
	/** For PUT/CUT with paste register: register name. */
	register?: string;
	/** For PUT with block edit (N*): block type. */
	block?: string;
}

/**
 * Parse a hashline edit operation from a line.
 * Returns null if the line is not a hashline operation.
 */
export function parseHashlineEdit(line: string): HashlineEdit | null {
	const trimmed = line.trimStart();

	// REM — delete file
	if (trimmed === HL_REM_KEYWORD) {
		return { operation: "REM", path: "" };
	}

	// MV — move/rename file
	if (trimmed.startsWith(HL_MOVE_KEYWORD + " ")) {
		const dest = trimmed.slice(HL_MOVE_KEYWORD.length + 1).trim();
		return { operation: "MV", path: "", dest };
	}

	// PUT — insert/replace lines
	if (trimmed.startsWith("PUT ")) {
		return parsePutOperation(trimmed);
	}

	// CUT — capture lines
	if (trimmed.startsWith("CUT ")) {
		return parseCutOperation(trimmed);
	}

	return null;
}

function parsePutOperation(line: string): HashlineEdit | null {
	const insert = /^PUT ([<>])(\d+):(.*)$/.exec(line);
	if (insert) {
		return { operation: "PUT", path: "", range: insert[1] + insert[2], body: [insert[3]] };
	}
	const replace = /^PUT (\d+)\.=(\d+):(.*)$/.exec(line);
	if (replace) {
		return { operation: "PUT", path: "", range: `${replace[1]}-${replace[2]}`, body: [replace[3]] };
	}
	// Accept the earlier range spelling as well.
	const legacy = /^PUT (\d+)(?:-(\d+))?\.=(.*)$/.exec(line);
	if (legacy) {
		return { operation: "PUT", path: "", range: `${legacy[1]}-${legacy[2] ?? legacy[1]}`, body: [legacy[3]] };
	}
	return null;
}

function parseCutOperation(line: string): HashlineEdit | null {
	const match = /^CUT (\d+)(?:\.=(\d+))?$/.exec(line);
	return match ? { operation: "CUT", path: "", range: `${match[1]}-${match[2] ?? match[1]}` } : null;
}

// ── Edit result ────────────────────────────────────────────────────────────────

/**
 * Result of a hashline edit operation.
 */
export interface HashlineEditResult {
	/** Whether the edit was applied. */
	applied: boolean;
	/** Number of lines changed. */
	linesChanged: number;
	/** Number of files affected. */
	filesAffected: number;
	/** Error message if the edit failed. */
	error?: string;
	/** Diff of the changes (unified format). */
	diff?: string;
	/** Structured receipts for each planned file mutation. */
	receipts?: import("@logician/log-core").MutationReceipt[];
	/** Stale anchor error: which anchor(s) are stale. */
	staleAnchors?: Array<{ path: string; expectedTag: string; computedTag: string }>;
}
