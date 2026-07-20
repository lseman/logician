// ── read_file tool ────────────────────────────────────────────────────────────────
// Read file contents with line-based pagination and two-axis truncation.

import * as fs from "node:fs";
import type { Tool } from "@logician/agent-core/core/types.ts";
import { ensureInsideCwd, resolveReadPath } from "@logician/agent-core/tools/shared/path-utils.ts";
import { recordRead } from "./read-tracker.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	truncateHead,
} from "./truncate.ts";

export const read_file: Tool = {
	readOnly: true,
	name: "read_file",
	label: "Read File",
	hookAliases: ["Read"],
	executionMode: "parallel",
	description:
		`Read file contents. Output is truncated to ${DEFAULT_MAX_LINES} lines or ` +
		`${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files; ` +
		"continue with offset until complete.",
	promptSnippet: "Read file contents with line numbers and truncation support",
	promptGuidelines: ["Use read_file to read files; use bash cat for quick checks"],
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
		ensureInsideCwd(ctx.cwd, resolved);

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
		const text = buffer.toString("utf-8");
		recordRead(resolved);
		const allLines = text.split("\n");
		const totalLines = allLines.length;

		// 1-based offset -> 0-based start.
		const startLine = offset > 0 ? offset - 1 : 0;
		if (startLine >= allLines.length) {
			return `Error: Offset ${offset} is beyond end of file (${totalLines} lines total)`;
		}
		const startDisplay = startLine + 1;

		let selected: string;
		let userLimited = 0;
		if (limit > 0) {
			const end = Math.min(startLine + limit, allLines.length);
			selected = allLines.slice(startLine, end).join("\n");
			userLimited = end - startLine;
		} else {
			selected = allLines.slice(startLine).join("\n");
		}

		const t = truncateHead(selected);

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
			return `${t.content}\n\n[${limitNote} Use offset=${nextOffset} to continue.]`;
		}
		if (userLimited > 0 && startLine + userLimited < allLines.length) {
			const remaining = allLines.length - (startLine + userLimited);
			const nextOffset = startLine + userLimited + 1;
			return `${t.content}\n\n[${remaining} more lines in file. Use offset=${nextOffset} to continue.]`;
		}
		return t.content;
	},
};
