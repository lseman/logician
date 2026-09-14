import { formatHashlineHeader, splitAddressableFileLines } from "./hashline.ts";
import type { ReadResource } from "./read-resource.ts";
import {
	DEFAULT_MAX_BYTES,
	formatSize,
	truncateHead,
} from "./utils/truncate.ts";

export interface ReadPage {
	offset: number;
	limit?: number | undefined;
}

/** One presentation contract for files, listings, docs, and protocol responses. */
export function formatResourceRead(read: ReadResource, page: ReadPage): string {
	const { resource } = read;
	const lines = splitAddressableFileLines(resource.content);
	const start = page.offset - 1;
	if (start >= lines.length && (lines.length > 0 || start > 0)) {
		throw new Error(
			`Offset ${page.offset} is beyond end of resource (${lines.length} lines total)`,
		);
	}
	const header =
		read.kind === "file"
			? formatHashlineHeader(resource.url, read.hash)
			: `[${resource.url}]`;
	if (lines.length === 0)
		return `${header}\n[Empty ${resource.isDirectory ? "directory" : "resource"}]`;
	const selected = lines.slice(
		start,
		page.limit === undefined ? undefined : start + page.limit,
	);
	const numbered = selected
		.map((line, i) => `${start + i + 1}:${line}`)
		.join("\n");
	const truncated = truncateHead(numbered);
	if (truncated.firstLineExceedsLimit) {
		throw new Error(
			`Line ${page.offset} exceeds the ${formatSize(DEFAULT_MAX_BYTES)} read limit. Request a narrower resource or inspect the backing file with bash.`,
		);
	}
	const end = start + truncated.outputLines;
	const parts = [header, truncated.content];
	if (end < lines.length) {
		parts.push(
			`\n[Showing lines ${page.offset}-${end} of ${lines.length}${truncated.truncatedBy === "bytes" ? ` (${formatSize(DEFAULT_MAX_BYTES)} limit)` : ""}. Use offset=${end + 1} to continue.]`,
		);
	}
	if (resource.notes?.length)
		parts.push(truncateHead(resource.notes.join("\n")).content);
	return parts.join("\n");
}
