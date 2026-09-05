// ── read_file tool ────────────────────────────────────────────────────────────────
// Read file contents with line-based pagination, two-axis truncation, and
// hashline anchors for edit targeting. Output includes a [path#4hex] header
// followed by numbered lines (1:content).

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import type { Tool } from "@logician/log-core";
import { recordRead } from "./support/read-tracker.js";
import {
	formatHashlineHeader,
	splitAddressableFileLines,
} from "./support/hashline.js";
import {
	ensureInsideCwd,
	resolveReadPath,
} from "./support/utils/path-utils.js";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	truncateHead,
} from "./support/utils/truncate.js";
import {
	parseConflictBlocks,
} from "./support/conflict-resolution.js";

export const read_file: Tool = {
	readOnly: true,
	cacheable: true,
	name: "read_file",
	label: "Read File",
	hookAliases: ["Read"],
	executionMode: "parallel",
	description:
		`Read file contents. Output includes hashline anchors ` +
		`(format: [path#4hex] with numbered lines). ` +
		`Truncated to ${DEFAULT_MAX_LINES} lines or ` +
		`${formatSize(DEFAULT_MAX_BYTES)} (whichever is hit first). ` +
		"Use offset/limit for large files; continue with offset until complete.",
	promptSnippet:
		"Read files; output includes [path#4hex] hashline header + numbered lines for edit targeting",
	promptGuidelines: [
		"Use read_file to read files; the output includes hashline anchors for edit_file targeting",
	],
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path to read" },
			offset: {
				type: "number",
				description: "1-based line number to start reading from",
			},
			limit: {
				type: "number",
				description: "Maximum number of lines to read",
			},
		},
		required: ["path"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") return { path: raw };
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			path: args.path ?? args.file_path ?? args.filename,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const filePath = String(args.path);
		const resolved = resolveReadPath(filePath, ctx.cwd || process.cwd());
		ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);

		if (!fs.existsSync(resolved)) {
			return `Error: File not found: ${resolved}`;
		}
		const stat = fs.statSync(resolved);
		if (stat.isDirectory()) {
			return `Error: Path is a directory: ${resolved}`;
		}

		const offset = Number(args.offset) || 0;
		const limit = Number(args.limit) || 0;

		const buffer = fs.readFileSync(resolved);
		if (buffer.subarray(0, 8192).includes(0)) {
			return (
				`Error: ${resolved} appears to be a binary file ` +
				`(${formatSize(stat.size)}). Use bash tools (file, xxd, strings) to inspect it.`
			);
		}
		const fullContent = buffer.toString("utf-8");
		recordRead(resolved);

		// Compute hashline tag from full file content
		const fileHash = createHash("sha256").update(buffer).digest("hex").slice(0, 4);
		const header = formatHashlineHeader(filePath, fileHash);

		// Split into addressable lines (no trailing empty line)
		const allLines = splitAddressableFileLines(fullContent);
		const totalLines = allLines.length;
		// Detect merge conflicts
		const conflictBlocks = parseConflictBlocks(fullContent, resolved);


		// 1-based offset -> 0-based start.
		const startLine = offset > 0 ? offset - 1 : 0;
		if (startLine >= allLines.length) {
			return `Error: Offset ${offset} is beyond end of file (${totalLines} lines total)`;
		}
		const startDisplay = startLine + 1;

		let selectedLines: string[];
		let userLimited = 0;
		if (limit > 0) {
			const end = Math.min(startLine + limit, allLines.length);
			selectedLines = allLines.slice(startLine, end);
			userLimited = end - startLine;
		} else {
			selectedLines = allLines.slice(startLine);
		}

		const t = truncateHead(selectedLines.join("\n"));

		if (t.firstLineExceedsLimit) {
			const lineSize = formatSize(
				Buffer.byteLength(allLines[startLine], "utf-8"),
			);
			return (
				`[Line ${startDisplay} is ${lineSize}, exceeds ${formatSize(DEFAULT_MAX_BYTES)} limit. ` +
				`Use bash: sed -n '${startDisplay}p' ${filePath} | head -c ${DEFAULT_MAX_BYTES}]`
			);
		}

		if (t.truncated) {
			const endDisplay = startDisplay + t.outputLines - 1;
			const nextOffset = endDisplay + 1;
			const limitNote =
				t.truncatedBy === "lines"
					? `Showing lines ${startDisplay}-${endDisplay} of ${totalLines}.`
					: `Showing lines ${startDisplay}-${endDisplay} of ${totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit).`;
			const conflictNotice = formatConflictNotice(fullContent, conflictBlocks);
			return `${header}\n${t.content}\n\n[${limitNote} Use offset=${nextOffset} to continue.]${conflictNotice}`;
		}

		if (userLimited > 0 && startLine + userLimited < allLines.length) {
			const remaining = allLines.length - (startLine + userLimited);
			const nextOffset = startLine + userLimited + 1;
			const conflictNotice = formatConflictNotice(fullContent, conflictBlocks);
			return `${header}\n${t.content}\n\n[${remaining} more lines in file. Use offset=${nextOffset} to continue.]${conflictNotice}`;
		}

		const conflictNotice = formatConflictNotice(fullContent, conflictBlocks);
		return `${header}\n${t.content}${conflictNotice}`;
	},
};

/** Build a conflict notice to append to read_file output. */
function formatConflictNotice(fullContent: string, blocks: { index: number; oursLabel: string; theirsLabel: string; offset: number; file: string }[]): string {
	if (blocks.length === 0) return "";
	const lines = [
		"",
		`# ⚠ Merge Conflicts: ${blocks.length} block${blocks.length > 1 ? "s" : ""} in ${blocks[0].file}`,
		"",
	];
	for (const block of blocks) {
		const blockLine = fullContent.slice(0, block.offset).split("\n").length;
		lines.push(
			`${block.index}: <<<<<<< ${block.oursLabel} ... ======= ... >>>>>>> ${block.theirsLabel} (around line ${blockLine})`,
		);
	}
	lines.push("");
	lines.push("[Use conflict://N?q=ours|theirs|ours+theirs|base to resolve]");
	lines.push("[Use :conflicts selector to view full conflict blocks]");
	return "\n" + lines.join("\n");
}
