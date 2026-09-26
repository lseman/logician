// ── Hashline anchor utilities ─────────────────────────────────────────────────
// Header formatting and stale-detection helpers for the hashline model-facing
// edit protocol. Parsing and applying the edit operations themselves is
// @logician/log-natives' job now (pi_edit's "hashline" mode) — see
// edit-file.ts's runNativeHashlineEdit — this module only covers what's
// still needed on the read/display side.
//
// Hashline format:
//   [path#4hex]        — file header with 4-char content hash
//   1:line content     — numbered lines

import { createHash } from "node:crypto";

/** File header prefix: [path#4hex] */
const HL_FILE_PREFIX = "[";
const HL_FILE_SUFFIX = "]";
const HL_FILE_HASH_SEP = "#";

/**
 * Format a file header with hashline anchors.
 * Returns: [path#4hex]
 */
export function formatHashlineHeader(path: string, tag: string): string {
	return `${HL_FILE_PREFIX}${path}${HL_FILE_HASH_SEP}${tag}${HL_FILE_SUFFIX}`;
}

/**
 * Split text into addressable lines (no trailing empty line).
 */
export function splitAddressableFileLines(text: string): string[] {
	const lines = text.split("\n");
	if (lines.length > 0 && lines.at(-1) === "") lines.pop();
	return lines;
}

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
