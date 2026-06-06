// ── Truncation utilities ─────────────────────────────────────────────────────────
// Two independent limits — whichever is hit first wins:
//   - Line limit  (default 2000 lines)
//   - Byte limit  (default 50KB)
// Never returns partial lines (except the tail-truncation edge case).
// Ported from pi (packages/coding-agent/src/core/tools/truncate.ts).

export const DEFAULT_MAX_LINES = 2000;
export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB
export const GREP_MAX_LINE_LENGTH = 500; // Max chars per match line

export interface TruncationResult {
    content: string;
    truncated: boolean;
    truncatedBy: "lines" | "bytes" | null;
    totalLines: number;
    totalBytes: number;
    outputLines: number;
    outputBytes: number;
    /** Only set by tail truncation when the kept line had to be cut mid-line. */
    lastLinePartial: boolean;
    /** Head truncation: first line alone exceeded the byte limit. */
    firstLineExceedsLimit: boolean;
    maxLines: number;
    maxBytes: number;
}

export interface TruncationOptions {
    maxLines?: number;
    maxBytes?: number;
}

function splitLinesForCounting(content: string): string[] {
    if (content.length === 0) return [];
    const lines = content.split("\n");
    if (content.endsWith("\n")) lines.pop();
    return lines;
}

export function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/**
 * Truncate content from the head (keep first N lines/bytes).
 * For file reads, where the beginning matters. Never returns partial lines.
 */
export function truncateHead(
    content: string,
    options: TruncationOptions = {},
): TruncationResult {
    const maxLines = options.maxLines ?? DEFAULT_MAX_LINES;
    const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;

    const totalBytes = Buffer.byteLength(content, "utf-8");
    const lines = splitLinesForCounting(content);
    const totalLines = lines.length;

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

    const firstLineBytes = Buffer.byteLength(lines[0] ?? "", "utf-8");
    if (firstLineBytes > maxBytes) {
        return {
            content: "",
            truncated: true,
            truncatedBy: "bytes",
            totalLines,
            totalBytes,
            outputLines: 0,
            outputBytes: 0,
            lastLinePartial: false,
            firstLineExceedsLimit: true,
            maxLines,
            maxBytes,
        };
    }

    const outputLinesArr: string[] = [];
    let outputBytesCount = 0;
    let truncatedBy: "lines" | "bytes" = "lines";

    for (let i = 0; i < lines.length && i < maxLines; i++) {
        const lineBytes =
            Buffer.byteLength(lines[i], "utf-8") + (i > 0 ? 1 : 0);
        if (outputBytesCount + lineBytes > maxBytes) {
            truncatedBy = "bytes";
            break;
        }
        outputLinesArr.push(lines[i]);
        outputBytesCount += lineBytes;
    }

    if (outputLinesArr.length >= maxLines && outputBytesCount <= maxBytes) {
        truncatedBy = "lines";
    }

    const outputContent = outputLinesArr.join("\n");
    return {
        content: outputContent,
        truncated: true,
        truncatedBy,
        totalLines,
        totalBytes,
        outputLines: outputLinesArr.length,
        outputBytes: Buffer.byteLength(outputContent, "utf-8"),
        lastLinePartial: false,
        firstLineExceedsLimit: false,
        maxLines,
        maxBytes,
    };
}

/**
 * Truncate content from the tail (keep last N lines/bytes).
 * For command output, where errors and final results live at the end.
 */
export function truncateTail(
    content: string,
    options: TruncationOptions = {},
): TruncationResult {
    const maxLines = options.maxLines ?? DEFAULT_MAX_LINES;
    const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;

    const totalBytes = Buffer.byteLength(content, "utf-8");
    const lines = splitLinesForCounting(content);
    const totalLines = lines.length;

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

    const outputLinesArr: string[] = [];
    let outputBytesCount = 0;
    let truncatedBy: "lines" | "bytes" = "lines";
    let lastLinePartial = false;

    for (
        let i = lines.length - 1;
        i >= 0 && outputLinesArr.length < maxLines;
        i--
    ) {
        const lineBytes =
            Buffer.byteLength(lines[i], "utf-8") +
            (outputLinesArr.length > 0 ? 1 : 0);
        if (outputBytesCount + lineBytes > maxBytes) {
            truncatedBy = "bytes";
            // First (= last in file) line alone exceeds limit: keep its end, partial.
            if (outputLinesArr.length === 0) {
                const truncatedLine = truncateStringToBytesFromEnd(
                    lines[i],
                    maxBytes,
                );
                outputLinesArr.unshift(truncatedLine);
                outputBytesCount = Buffer.byteLength(truncatedLine, "utf-8");
                lastLinePartial = true;
            }
            break;
        }
        outputLinesArr.unshift(lines[i]);
        outputBytesCount += lineBytes;
    }

    if (outputLinesArr.length >= maxLines && outputBytesCount <= maxBytes) {
        truncatedBy = "lines";
    }

    const outputContent = outputLinesArr.join("\n");
    return {
        content: outputContent,
        truncated: true,
        truncatedBy,
        totalLines,
        totalBytes,
        outputLines: outputLinesArr.length,
        outputBytes: Buffer.byteLength(outputContent, "utf-8"),
        lastLinePartial,
        firstLineExceedsLimit: false,
        maxLines,
        maxBytes,
    };
}

/** Truncate a string to fit within a byte limit, keeping the end. UTF-8 safe. */
function truncateStringToBytesFromEnd(str: string, maxBytes: number): string {
    const buf = Buffer.from(str, "utf-8");
    if (buf.length <= maxBytes) return str;
    let start = buf.length - maxBytes;
    while (start < buf.length && (buf[start] & 0xc0) === 0x80) start++;
    return buf.slice(start).toString("utf-8");
}

/** Truncate a single line to max chars, adding a [truncated] suffix. */
export function truncateLine(
    line: string,
    maxChars: number = GREP_MAX_LINE_LENGTH,
): { text: string; wasTruncated: boolean } {
    if (line.length <= maxChars) return { text: line, wasTruncated: false };
    return {
        text: `${line.slice(0, maxChars)}... [truncated]`,
        wasTruncated: true,
    };
}
