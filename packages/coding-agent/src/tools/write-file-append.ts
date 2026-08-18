// ── write_file_append tool ────────────────────────────────────────────────────
// Append a chunk of content to a file, creating it (and parent directories) if
// missing. Lets the model stream a large file across multiple tool calls, each
// call stays comfortably under the completion's max_tokens budget instead of
// emitting one giant write_file call that gets truncated mid-argument.

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool, ToolResult } from "@logician/agent-core";
import { ensureInsideCwd, readUtf8IfExists, resolvePath } from "@logician/agent-core";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "./read-tracker.ts";
import { appendToFile } from "./shared/atomic-write.ts";
import { withFileMutationQueue } from "./shared/file-mutation-queue.ts";

export const write_file_append: Tool = {
	name: "write_file_append",
	executionMode: "parallel",
	label: "Append File",
	description:
		"Append a chunk of text to a file, creating it (and parent directories) if it " +
		"doesn't exist yet. Use this instead of write_file when the content is too large " +
		"for a single tool call — split it into chunks and call this tool repeatedly in " +
		"order, same path each time. Overwriting an existing file's start requires reading " +
		"it with read_file first, same as write_file.",
	promptSnippet:
		"Append a chunk to a file, for streaming large files across multiple calls",
	promptGuidelines: [
		"Use write_file_append to build a large file in chunks instead of one huge write_file call",
	],
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path to append to" },
			content: { type: "string", description: "Chunk of content to append" },
		},
		required: ["path", "content"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			path: args.path ?? args.file_path ?? args.filename,
			content: args.content ?? args.text,
		};
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const filePath = String(args.path);
		const content = String(args.content ?? "");
		const resolved = resolvePath(ctx.cwd, filePath);
		ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);

		return withFileMutationQueue(resolved, async () => {
			const before = readUtf8IfExists(resolved);
			if (before !== null) {
				if (!hasBeenRead(resolved)) {
					return (
						`${resolved} already exists but has not been read. ` +
						"Read it with read_file before appending, or use write_file / edit_file for targeted changes."
					);
				}
				if (isStaleSinceRead(resolved)) {
					return (
						`${resolved} has been modified since it was last read. ` +
						"Read it again before appending."
					);
				}
			}

			fs.mkdirSync(path.dirname(resolved), { recursive: true });
			const beforeBytes =
				before === null ? 0 : Buffer.byteLength(before, "utf-8");
			const { newSize } = await appendToFile(resolved, content, {
				expectedSizeBefore: beforeBytes,
			});
			refreshAfterWrite(resolved);

			const chunkBytes = Buffer.byteLength(content, "utf-8");
			const totalLines = fs.readFileSync(resolved, "utf-8").split("\n").length;
			const verb = before === null ? "Created" : "Appended to";
			const lineInfo = totalLines > 0 ? ` · ${totalLines} lines total` : "";
			return (
				`${verb} ${resolved} (+${chunkBytes} bytes, ${newSize} bytes total${lineInfo}). ` +
				"Call write_file_append again with the next chunk, or stop if this was the last one."
			);
		});
	},
};
