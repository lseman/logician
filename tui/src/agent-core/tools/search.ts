// ── rg_search tool ────────────────────────────────────────────────────────────────
// Search files using ripgrep (rg) with glob filtering.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { Tool } from "../types.ts";
import { truncateLine } from "./truncate.ts";

const execFileAsync = promisify(execFile);

export const rg_search: Tool = {
	name: "rg_search",
	hookAliases: ["Grep"],
	executionMode: "parallel",
	description:
		"Search file contents with ripgrep (supports regex, glob, context lines).",
	parameters: {
		type: "object",
		properties: {
			pattern: { type: "string", description: "Search pattern (regex)" },
			path: {
				type: "string",
				description: "Path to search (defaults to cwd)",
			},
			glob: { type: "string", description: "Glob pattern filter" },
			max_results: {
				type: "number",
				description: "Max result lines to return",
			},
			context: { type: "number", description: "Context lines (-A/-B)" },
		},
		required: ["pattern"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") return { pattern: raw };
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			pattern: args.pattern ?? args.query ?? args.regex,
			path: args.path ?? args.file_path ?? args.directory,
			max_results: args.max_results ?? args.maxResults ?? args.limit,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const pattern = String(args.pattern);
		const searchPath = String(args.path || ctx.cwd || ".");
		const maxResults = Number(args.max_results ?? args.max_files) || 100;
		const context = Number(args.context) || 2;

		const cmd = ["--line-number", "-n"];
		if (context > 0) cmd.push(`-C${context}`);

		const glob = String(args.glob || "");
		if (glob) cmd.push("-g", glob);

		cmd.push(pattern, searchPath);

		try {
			const { stdout } = await execFileAsync("rg", cmd, {
				cwd: ctx.cwd,
				timeout: 10000,
				maxBuffer: 1024 * 1024,
			});
			const allLines = stdout.split("\n").filter(Boolean);
			const lines = allLines
				.slice(0, maxResults)
				.map((l) => truncateLine(l).text);
			const suffix =
				allLines.length > maxResults
					? `\n... [truncated to ${maxResults} result lines]`
					: "";
			return (lines.join("\n") + suffix).trim() || "No matches found.";
		} catch (e: unknown) {
			const error = e as { code?: number; stderr?: string };
			if (error.code === 1) return "No matches found.";
			return `Error: ${error.stderr || String(error)}`;
		}
	},
};
