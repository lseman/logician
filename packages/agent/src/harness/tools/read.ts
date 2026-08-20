// ── read tool ─────────────────────────────────────────────────────────────
// Read file contents with line-based pagination and two-axis (line/byte)
// truncation. Ported from coding-agent's tools/read-file.ts, rewritten
// against ExecutionEnv (throw-on-error, per AgentTool.execute's contract)
// instead of raw node:fs. Text-only — coding-agent's version has no image
// support either, unlike pi's read tool.

import { type Static, Type } from "@sinclair/typebox";
import type { AgentTool } from "../../agent/types.ts";
import type { ExecutionEnv } from "../../env/execution-env.ts";
import { ensureInsideCwd, resolveReadPath } from "./path-utils.ts";
import { recordRead } from "./read-tracker.ts";
import { DEFAULT_MAX_BYTES, formatSize, truncateHead } from "./truncate.ts";

const readSchema = Type.Object({
	path: Type.String({ description: "File path to read" }),
	offset: Type.Optional(
		Type.Number({ description: "1-based line number to start reading from" }),
	),
	limit: Type.Optional(
		Type.Number({ description: "Maximum number of lines to read" }),
	),
});

export interface ReadToolOptions {
	env: ExecutionEnv;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
}

export function createReadTool(
	options: ReadToolOptions,
): AgentTool<typeof readSchema, undefined> {
	const { env, allowedPaths, allowAllPaths } = options;
	return {
		name: "read",
		label: "Read File",
		description: `Read file contents. Output is truncated to ${2000} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files; continue with offset until complete.`,
		parameters: readSchema,
		executionMode: "parallel",
		readOnly: true,
		prepareArguments: (raw): unknown => {
			if (typeof raw === "string") return { path: raw };
			if (!raw || typeof raw !== "object") return {};
			const args = raw as Record<string, unknown>;
			return { ...args, path: args.path ?? args.file_path ?? args.filename };
		},
		async execute(_toolCallId, rawParams) {
			const { path, offset, limit } = rawParams as Static<typeof readSchema>;
			const resolved = await resolveReadPath(env, path);
			await ensureInsideCwd(env, resolved, allowedPaths, allowAllPaths);

			const info = await env.fileInfo(resolved);
			if (!info.ok) {
				if (info.error.code === "not_found")
					throw new Error(`File not found: ${resolved}`);
				throw new Error(`Failed to read ${resolved}: ${info.error.message}`);
			}
			if (info.value.kind === "directory")
				throw new Error(`Path is a directory: ${resolved}`);

			const bytesResult = await env.readBinaryFile(resolved);
			if (!bytesResult.ok)
				throw new Error(
					`Failed to read ${resolved}: ${bytesResult.error.message}`,
				);
			const bytes = bytesResult.value;
			if (bytes.subarray(0, 8192).includes(0)) {
				throw new Error(
					`${resolved} appears to be a binary file (${formatSize(info.value.size)}). Use bash tools (file, xxd, strings) to inspect it.`,
				);
			}

			const text = new TextDecoder().decode(bytes);
			await recordRead(env, resolved);
			const allLines = text.split("\n");
			const totalLines = allLines.length;

			const startLine = offset && offset > 0 ? offset - 1 : 0;
			if (startLine >= allLines.length) {
				throw new Error(
					`Offset ${offset} is beyond end of file (${totalLines} lines total)`,
				);
			}
			const startDisplay = startLine + 1;

			let selected: string;
			let userLimited = 0;
			if (limit && limit > 0) {
				const end = Math.min(startLine + limit, allLines.length);
				selected = allLines.slice(startLine, end).join("\n");
				userLimited = end - startLine;
			} else {
				selected = allLines.slice(startLine).join("\n");
			}

			const truncation = truncateHead(selected);
			let outputText: string;

			if (truncation.firstLineExceedsLimit) {
				const lineSize = formatSize(
					new TextEncoder().encode(allLines[startLine] ?? "").byteLength,
				);
				outputText = `[Line ${startDisplay} is ${lineSize}, exceeds ${formatSize(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n '${startDisplay}p' ${path} | head -c ${DEFAULT_MAX_BYTES}]`;
			} else if (truncation.truncated) {
				const endDisplay = startDisplay + truncation.outputLines - 1;
				const nextOffset = endDisplay + 1;
				const limitNote =
					truncation.truncatedBy === "lines"
						? `Showing lines ${startDisplay}-${endDisplay} of ${totalLines}.`
						: `Showing lines ${startDisplay}-${endDisplay} of ${totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit).`;
				outputText = `${truncation.content}\n\n[${limitNote} Use offset=${nextOffset} to continue.]`;
			} else if (userLimited > 0 && startLine + userLimited < allLines.length) {
				const remaining = allLines.length - (startLine + userLimited);
				const nextOffset = startLine + userLimited + 1;
				outputText = `${truncation.content}\n\n[${remaining} more lines in file. Use offset=${nextOffset} to continue.]`;
			} else {
				outputText = truncation.content;
			}

			return {
				content: [{ type: "text", text: outputText }],
				details: undefined,
			};
		},
	};
}
