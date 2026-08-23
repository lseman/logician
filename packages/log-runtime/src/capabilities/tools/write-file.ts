// ── write_file tool ───────────────────────────────────────────────────────────────
// Create or overwrite a complete file. Creates parent directories. Overwriting an
// existing file requires it to have been read first (and not modified since), so the
// model can never blind-clobber content. Returns the new content with syntax
// highlighting and line numbers (truncated for large files).
//
// With append: true, appends content to the end of the file instead of overwriting
// it. The file must have been read first if it already exists (same safety check).
// Appending is useful for streaming large files across multiple tool calls.

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool, ToolResult } from "@logician/log-core";
import { withFileMutationQueue } from "./support/mutation-queue.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "./support/read-tracker.ts";
import { appendToFile, atomicWriteFile } from "./support/utils/atomic-write.ts";
import { ensureInsideCwd, resolvePath } from "./support/utils/path-utils.ts";
import { highlightAuto } from "./support/utils/syntax-highlighter.ts";
import {
	DEFAULT_MAX_BYTES,
	formatSize,
	truncateHead,
} from "./support/utils/truncate.ts";

export const write_file: Tool = {
	name: "write_file",
	executionMode: "parallel",
	label: "Write File",
	hookAliases: ["Write"],
	description:
		"Create or overwrite a complete file. Creates parent directories. " +
		"Overwriting an existing file requires reading it with read_file first. " +
		"Output is truncated to ${DEFAULT_MAX_LINES} lines or " +
		`${formatSize(DEFAULT_MAX_BYTES)} (whichever is hit first). ` +
		"With append: true, appends content to the end of the file instead of " +
		"overwriting. Useful for streaming large files across multiple tool calls.",
	promptSnippet:
		"Create or overwrite files; automatically create parent directories; use append: true to append",
	promptGuidelines: [
		"Use write_file for new files or complete rewrites",
		"Use append: true to add content to the end of an existing file without overwriting it",
	],
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path to write" },
			content: { type: "string", description: "Complete file contents" },
			append: {
				type: "boolean",
				description:
					"If true, append content to the end of the file instead of overwriting. Defaults to false.",
			},
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
		const append = Boolean(args.append);
		const resolved = resolvePath(ctx.cwd, filePath);
		ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);

		return withFileMutationQueue(resolved, async () => {
			const fileExists = fs.existsSync(resolved);

			if (fileExists && !append) {
				// Overwrite mode: must have been read first.
				if (!hasBeenRead(resolved)) {
					return (
						`${resolved} already exists but has not been read. ` +
						"Read it with read_file before overwriting, or use edit_file for targeted changes."
					);
				}
				if (isStaleSinceRead(resolved)) {
					return (
						`${resolved} has been modified since it was last read. ` +
						"Read it again before overwriting."
					);
				}
			}

			// Append mode: also require read if file exists, but the check is
			// about whether the model last saw this file's contents — same
			// stale-after-write protection, no content comparison needed.
			if (fileExists && append) {
				if (!hasBeenRead(resolved)) {
					return (
						`${resolved} already exists but has not been read. ` +
						"Read it with read_file before appending, or use write_file for a complete overwrite."
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

			const chunkBytes = Buffer.byteLength(content, "utf-8");
			let result: string;

			if (append && fileExists) {
				// Append mode with existing file: size-based concurrency guard.
				const fileStat = fs.statSync(resolved);
				await appendToFile(resolved, content, {
					expectedSizeBefore: fileStat.size,
				});
				refreshAfterWrite(resolved);
				result =
					`Appended to ${resolved} (+${chunkBytes} bytes, file size now ${formatSize(chunkBytes + fileStat.size - Buffer.byteLength(content, "utf-8"))}). ` +
					"Call write_file with the next chunk, or stop if this was the last one.";
			} else if (append && !fileExists) {
				// Append mode with new file: just create and write.
				await appendToFile(resolved, content);
				refreshAfterWrite(resolved);
				result =
					`Created ${resolved} (${chunkBytes} bytes) with append mode. ` +
					"Call write_file with the next chunk, or stop if this was the last one.";
			} else {
				// Overwrite mode (normal).
				await atomicWriteFile(resolved, content, {
					expectedMissing: !fileExists,
				});
				refreshAfterWrite(resolved);

				const lineCount = content === "" ? 0 : content.split("\n").length;
				const byteLen = Buffer.byteLength(content, "utf-8");
				if (!fileExists) {
					return `Created ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
				}

				const highlighted = highlightAuto(content);
				const t = truncateHead(highlighted.value);

				if (t.firstLineExceedsLimit) {
					const firstLineBytes = Buffer.byteLength(
						highlighted.value.split("\n")[0] ?? "",
						"utf-8",
					);
					return (
						`[First line is ${formatSize(firstLineBytes)}, exceeds ${formatSize(DEFAULT_MAX_BYTES)} limit. ` +
						`Use bash: head -c ${DEFAULT_MAX_BYTES} ${filePath}]`
					);
				}
				if (t.truncated) {
					const endDisplay = t.outputLines;
					const nextOffset = endDisplay + 1;
					return (
						`${t.content}\n\n` +
						`[Wrote ${resolved} (${lineCount} lines, ${formatSize(byteLen)}). ` +
						`Showing lines 1-${endDisplay}. ` +
						`Use offset=${nextOffset} to continue.]`
					);
				}

				const gutterWidth = String(lineCount).length + 1;
				const header = `Wrote ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
				const out: string[] = [header];
				const hlLines = t.content.split("\n");
				for (let i = 0; i < hlLines.length; i++) {
					const num = String(i + 1).padStart(gutterWidth, " ");
					out.push(`${num}|${hlLines[i]}`);
				}
				return out.join("\n");
			}

			return result;
		});
	},
};
