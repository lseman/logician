/**
 * Shared truncation utilities for tool outputs.
 *
 * Truncation is based on two independent limits — whichever is hit first wins:
 * - Line limit (default: 2000 lines)
 * - Byte limit (default: 50KB)
 *
 * Never returns partial lines (except bash tail truncation edge case).
 */

export const DEFAULT_MAX_LINES = 2000;
export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB
export const GREP_MAX_LINE_LENGTH = 500; // Max chars per grep match line

export interface TruncationResult {
	/** The truncated content */
	content: string;
	/** Whether truncation occurred */
	truncated: boolean;
	/** Which limit was hit: "lines", "bytes", or null if not truncated */
	truncatedBy: "lines" | "bytes" | null;
	/** Total number of lines in the original content */
	totalLines: number;
	/** Total number of bytes in the original content */
	totalBytes: number;
	/** Number of complete lines in the truncated output */
	outputLines: number;
	/** Number of bytes in the truncated output */
	outputBytes: number;
	/** Whether the last line was partially truncated */
	lastLinePartial: boolean;
	/** Whether the first line exceeded the byte limit */
	firstLineExceedsLimit: boolean;
	/** The max lines limit that was applied */
	maxLines: number;
	/** The max bytes limit that was applied */
	maxBytes: number;
}

export interface TruncationOptions {
	/** Maximum number of lines (default: 2000) */
	maxLines?: number;
	/** Maximum number of bytes (default: 50KB) */
	maxBytes?: number;
}

const nonAsciiPattern = /[^\x00-\x7f]/;

function utf8ByteLength(content: string): number {
	const firstNonAscii = content.search(nonAsciiPattern);
	if (firstNonAscii === -1) return content.length;

	let bytes = firstNonAscii;
	for (let i = firstNonAscii; i < content.length; i++) {
		const code = content.charCodeAt(i);
		if (code <= 0x7f) {
			bytes += 1;
		} else if (code <= 0x7ff) {
			bytes += 2;
		} else if (code >= 0xd800 && code <= 0xdbff && i + 1 < content.length) {
			const next = content.charCodeAt(i + 1);
			if (next >= 0xdc00 && next <= 0xdfff) {
				bytes += 4;
				i++;
			} else {
				bytes += 3;
			}
		} else {
			bytes += 3;
		}
	}
	return bytes;
}

/**
 * Truncate text to fit within line and byte limits.
 * Keeps the tail (recent output) and drops the head (older output).
 */
export function truncateTail(
	content: string,
	options?: TruncationOptions,
): TruncationResult {
	const maxLines = options?.maxLines ?? DEFAULT_MAX_LINES;
	const maxBytes = options?.maxBytes ?? DEFAULT_MAX_BYTES;

	const lines = content.split("\n");
	const totalLines = lines.length;
	const totalBytes = utf8ByteLength(content);

	// Check if first line exceeds byte limit
	const firstLineBytes = utf8ByteLength(lines[0] ?? "");
	const firstLineExceedsLimit = firstLineBytes > maxBytes;
	if (firstLineExceedsLimit) {
		const firstLine = lines[0] ?? "";
		let truncated = "";
		let bytes = 0;
		for (let i = 0; i < firstLine.length; i++) {
			const code = firstLine.charCodeAt(i);
			let charBytes = 1;
			if (code <= 0x7f) charBytes = 1;
			else if (code <= 0x7ff) charBytes = 2;
			else if (code >= 0xd800 && code <= 0xdbff && i + 1 < firstLine.length) {
				const next = firstLine.charCodeAt(i + 1);
				if (next >= 0xdc00 && next <= 0xdfff) {
					charBytes = 4;
					i++;
				} else {
					charBytes = 3;
				}
			} else {
				charBytes = 3;
			}
			if (bytes + charBytes > maxBytes) break;
			truncated += firstLine[i];
			bytes += charBytes;
		}
		return {
			content: truncated,
			truncated: true,
			truncatedBy: "bytes",
			totalLines,
			totalBytes,
			outputLines: 0,
			outputBytes: bytes,
			lastLinePartial: true,
			firstLineExceedsLimit: true,
			maxLines,
			maxBytes,
		};
	}

	// Check if we need to truncate at all
	if (totalLines <= maxLines && totalBytes <= maxBytes) {
		return {
			content,
			truncated: false,
			truncatedBy: null,
			totalLines,
			totalBytes,
			outputLines: totalLines,
			outputBytes: totalBytes,
			lastLinePartial: false,
			firstLineExceedsLimit: false,
			maxLines,
			maxBytes,
		};
	}

	// Determine how many lines to keep based on whichever limit is hit first
	let keepLines = totalLines;
	let keepBytes = totalBytes;
	let truncatedBy: "lines" | "bytes" | null = null;

	if (totalLines > maxLines) {
		keepLines = maxLines;
		truncatedBy = "lines";
	}

	if (totalBytes > maxBytes) {
		// Binary search for the right cutoff point
		let lo = 0;
		let hi = totalLines;
		let best = 0;

		while (lo <= hi) {
			const mid = Math.floor((lo + hi) / 2);
			const candidate = lines.slice(mid).join("\n");
			if (utf8ByteLength(candidate) <= maxBytes) {
				best = mid;
				hi = mid - 1;
			} else {
				lo = mid + 1;
			}
		}

		const remainingLines = totalLines - best;
		if (remainingLines <= maxLines && truncatedBy === null) {
			truncatedBy = "bytes";
		} else if (truncatedBy === null) {
			truncatedBy = "lines";
		}

		if (remainingLines <= maxLines) {
			keepLines = remainingLines;
			keepBytes = utf8ByteLength(lines.slice(best).join("\n"));
		} else {
			keepLines = maxLines;
			keepBytes = utf8ByteLength(lines.slice(totalLines - maxLines).join("\n"));
		}
	}

	const outputLines = lines.slice(lines.length - keepLines);
	const contentOut = outputLines.join("\n");
	const outputBytes = utf8ByteLength(contentOut);

	// Check if the last line is partial (truncated mid-line during binary search)
	const lastLinePartial = totalBytes > maxBytes && truncatedBy === "bytes";

	return {
		content: contentOut,
		truncated: true,
		truncatedBy,
		totalLines,
		totalBytes,
		outputLines: outputLines.length,
		outputBytes,
		lastLinePartial,
		firstLineExceedsLimit: false,
		maxLines,
		maxBytes,
	};
}

/**
 * Remove non-printable binary characters from a string.
 * Keeps tabs (0x09), newlines (0x0A), carriage returns (0x0D).
 * Filters out control characters and Unicode surrogate holes.
 */
export function sanitizeBinaryOutput(str: string): string {
	return Array.from(str)
		.filter((char) => {
			const code = char.codePointAt(0);
			if (code === undefined) return false;
			if (code === 0x09 || code === 0x0a || code === 0x0d) return true;
			if (code <= 0x1f) return false;
			if (code >= 0xfff9 && code <= 0xfffb) return false;
			return true;
		})
		.join("");
}
