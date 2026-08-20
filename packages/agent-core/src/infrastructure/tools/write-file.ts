// ── write_file tool ───────────────────────────────────────────────────────────────
// Create or overwrite a complete file. Creates parent directories. Overwriting an
// existing file requires it to have been read first (and not modified since), so the
// model can never blind-clobber content. Returns the new content with syntax
// highlighting and line numbers.

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool, ToolResult } from "../../core/types/types-messages.ts";
import { withFileMutationQueue } from "../filesystem/mutation-queue.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "../filesystem/read-tracker.ts";
import { atomicWriteFile } from "./utils/atomic-write.ts";
import {
	ensureInsideCwd,
	readUtf8IfExists,
	resolvePath,
} from "./utils/path-utils.ts";
import { highlightAuto } from "./utils/syntax-highlighter.ts";

export const write_file: Tool = {
	name: "write_file",
	executionMode: "parallel",
	label: "Write File",
	hookAliases: ["Write"],
	description:
		"Create or overwrite a complete file. Creates parent directories. " +
		"Overwriting an existing file requires reading it with read_file first. " +
		"For very large files, use write_file_append in chunks instead.",
	promptSnippet:
		"Create or overwrite files; automatically create parent directories",
	promptGuidelines: [
		"Use write_file for new files or complete rewrites",
		"For very large files, prefer write_file_append in chunks over one huge write_file call",
	],
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path to write" },
			content: { type: "string", description: "Complete file contents" },
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
			if (before === content) {
				return `No changes made: ${resolved}`;
			}

			fs.mkdirSync(path.dirname(resolved), { recursive: true });
			await atomicWriteFile(resolved, content, {
				expectedContent: before ?? undefined,
				expectedMissing: before === null,
			});
			refreshAfterWrite(resolved);

			const lineCount = content === "" ? 0 : content.split("\n").length;
			const byteLen = Buffer.byteLength(content, "utf-8");
			if (before === null) {
				return `Created ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
			}

			const highlighted = highlightAuto(content);
			const hlLines = highlighted.value.split("\n");
			const gutterWidth = String(lineCount).length + 1;
			const header = `Wrote ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
			const out: string[] = [header];
			for (let i = 0; i < hlLines.length; i++) {
				const num = String(i + 1).padStart(gutterWidth, " ");
				out.push(`${num}|${hlLines[i]}`);
			}
			return out.join("\n");
		});
	},
};
