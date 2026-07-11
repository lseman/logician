// ── write_file tool ───────────────────────────────────────────────────────────────
// Create or overwrite a complete file. Creates parent directories and returns the highlighted new file content.

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool } from "@logician/agent-core/core/types.ts";
import { withFileMutationQueue } from "./shared/file-mutation-queue.ts";
import {
	ensureInsideCwd,
	readUtf8IfExists,
	resolvePath,
} from "@logician/agent-core/tools/shared/path-utils.ts";
import { refreshAfterWrite } from "./read-tracker.ts";
import { highlight } from "@logician/agent-core/tools/shared/syntax-highlighter.ts";

export const write_file: Tool = {
	name: "write_file",
	label: "Write File",
	hookAliases: ["Write"],
	description:
		"Create or overwrite a complete file. Creates parent directories and returns the highlighted new file content.",
	promptSnippet:
		"Create or overwrite files; automatically create parent directories",
	promptGuidelines: ["Use write_file for new files or complete rewrites"],
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
	execute: async (args, ctx): Promise<string> => {
		const filePath = String(args.path);
		const content = String(args.content ?? "");
		const resolved = resolvePath(ctx.cwd, filePath);
		ensureInsideCwd(ctx.cwd, resolved);

		return withFileMutationQueue(resolved, async () => {
			const before = readUtf8IfExists(resolved);
			if (before === content) {
				return `No changes made: ${resolved}`;
			}

			fs.mkdirSync(path.dirname(resolved), { recursive: true });
			fs.writeFileSync(resolved, content, "utf-8");
			refreshAfterWrite(resolved);

			// Determine language for syntax highlighting based on file extension
			const ext = path.extname(resolved).slice(1).toLowerCase();
			const languageMap: Record<string, string> = {
				ts: "typescript",
				js: "javascript",
				tsx: "typescript",
				jsx: "javascript",
				py: "python",
				rb: "ruby",
				rs: "rust",
				go: "go",
				java: "java",
				cs: "csharp",
				cpp: "cpp",
				c: "c",
				h: "cpp",
				hpp: "cpp",
				php: "php",
				html: "html",
				css: "css",
				scss: "scss",
				less: "less",
				json: "json",
				md: "markdown",
				yaml: "yaml",
				yml: "yaml",
				sh: "bash",
				bash: "bash",
				sql: "sql",
				xml: "xml",
				dockerfile: "dockerfile",
				docker: "dockerfile",
			};
			const language = languageMap[ext] || "plaintext";

			// Syntax-highlight the content for display
			const highlighted = highlight(content, language);
			const langLabel = highlighted.language ? ` (${highlighted.language})` : "";

			return (
				`${before === null ? "Created" : "Wrote"} ${resolved}${langLabel}` +
				"\n\n" +
				highlighted.value
			);
		});
	},
};
