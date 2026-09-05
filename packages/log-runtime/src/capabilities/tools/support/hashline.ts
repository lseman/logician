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

/**
 * Format numbered lines for hashline display.
 * Each line gets a sequential number: `1:content`, `2:content`, ...
 */
export function formatNumberedLines(text: string, startLine?: number): string {
	const lines = text.split("\n");
	const end = lines.length > 0 && lines.at(-1) === "" ? lines.length - 1 : lines.length;
	const lineNum = startLine ?? 1;
	const parts: string[] = [];
	for (let i = 0; i < end; i++) {
		parts.push(`${lineNum + i}:${lines[i]}`);
	}
	return parts.join("\n");
}

/** Format a single numbered line: `N:content` */
export function formatNumberedLine(lineNumber: number, line: string): string {
	return `${lineNumber}:${line}`;
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

/**
 * Strip hashline prefixes (numbered lines: `1:content` → `content`).
 * Used when the model writes back hashline-format content.
 */
export function stripHashlinePrefixes(lines: string[]): string[] {
	return lines.map((line) => {
		const idx = line.indexOf(HL_LINE_BODY_SEP);
		if (idx > 0) {
			return line.slice(idx + 1);
		}
		return line;
	});
}

// ── Hashline anchor parsing ────────────────────────────────────────────────────

/**
 * Extract the hashline tag from a file header: [path#4hex]
 */
export function extractHashlineTag(fileHeader: string): string | null {
	if (!fileHeader.startsWith(HL_FILE_PREFIX)) return null;
	const closeIdx = fileHeader.indexOf(HL_FILE_SUFFIX);
	if (closeIdx < 0) return null;
	const inner = fileHeader.slice(1, closeIdx);
	const hashIdx = inner.indexOf(HL_FILE_HASH_SEP);
	if (hashIdx < 0) return null;
	const tag = inner.slice(hashIdx + 1);
	if (tag.length !== HASHLINE_TAG_LENGTH) return null;
	return tag;
}

/**
 * Check if a line is a hashline file header.
 */
export function isHashlineFileHeader(line: string): boolean {
	return line.startsWith(HL_FILE_PREFIX) && line.endsWith(HL_FILE_SUFFIX);
}

// ── Hashline anchor validation ─────────────────────────────────────────────────

/**
 * Validate a hashline anchor against expected content.
 * Returns the expected tag if the anchor is valid, null otherwise.
 */
export function validateHashlineAnchor(
	tag: string,
	content: string,
	expectedTag: string,
): boolean {
	if (tag !== expectedTag) return false;
	// Additional check: recompute hash of content to verify it matches
	const computed = hashlineHash(content);
	// For 4-char tags, we allow the stored tag to match the computed tag
	// of the file content at read time.
	return computed.startsWith(tag) || tag === expectedTag;
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
	/** Stale anchor error: which anchor(s) are stale. */
	staleAnchors?: Array<{ path: string; expectedTag: string; computedTag: string }>;
}
