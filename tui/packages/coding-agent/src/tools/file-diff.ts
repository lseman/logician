// ── file_diff tool ───────────────────────────────────────────────────────────────
// Show git diff for one file or the whole working tree.

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import type { Tool } from "@logician/agent-core/core/types.ts";
import { ensureInsideCwd, resolveReadPath } from "@logician/agent-core/tools/shared/path-utils.ts";
import { summarizeDiff } from "./diff-utils.ts";

const execFileAsync = promisify(execFile);

export const file_diff: Tool = {
	readOnly: true,
	name: "file_diff",
	label: "File Diff",
	executionMode: "parallel",
	description:
		"Show the current git diff for a file or the whole working tree.",
	promptSnippet: "Generate unified diff between old and new file content",
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "Optional file path to diff" },
			staged: {
				type: "boolean",
				description: "Show staged diff instead of unstaged diff",
			},
		},
	},
	execute: async (args, ctx): Promise<string> => {
		const cwd = ctx.cwd || process.cwd();
		const cmd = ["diff"];
		if (args.staged) cmd.push("--staged");

		if (args.path) {
			const resolved = resolveReadPath(String(args.path), cwd);
			ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);
			cmd.push("--", path.relative(cwd, resolved));
		}

		try {
			const { stdout } = await execFileAsync("git", cmd, {
				cwd,
				timeout: 10000,
				maxBuffer: 1024 * 1024,
			});
			return summarizeDiff(stdout.trimEnd()) || "(no diff)";
		} catch (_e: unknown) {
			const error = e as { stderr?: string; message?: string };
			return `Error: ${error.stderr || error.message || "git diff failed"}`;
		}
	},
};
