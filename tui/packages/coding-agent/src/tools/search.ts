// ── grep tool ────────────────────────────────────────────────────────────────
// Search files using ripgrep (rg) with structured JSON output.
// Features: line/byte truncation, context lines, ignoreCase, literal mode,
// file caching for context, AbortSignal support, structured ToolResult.

import { readFile as fsReadFile, stat as fsStat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { spawn } from "node:child_process";
import path from "node:path";
import type { Tool, ToolResult } from "@logician/agent-core/core/types.ts";
import { truncateHead, truncateLine, formatSize } from "./truncate.ts";
import { ensureInsideCwd, resolvePath } from "@logician/agent-core/tools/shared/path-utils.ts";
import { ensureTool } from "./shared/tools-manager.ts";

const grepSchema = {
	type: "object",
	properties: {
		pattern: { type: "string", description: "Search pattern (regex)" },
		path: {
			type: "string",
			description: "Directory or file to search (default: current directory)",
		},
		glob: {
			type: "string",
			description:
				"Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
		},
		ignoreCase: {
			type: "boolean",
			description: "Case-insensitive search (default: false)",
		},
		literal: {
			type: "boolean",
			description:
				"Treat pattern as literal string instead of regex (default: false)",
		},
		context: {
			type: "number",
			description:
				"Number of lines to show before and after each match (default: 0)",
		},
		limit: {
			type: "number",
			description: "Maximum number of matches to return (default: 100)",
		},
	},
	required: ["pattern"],
} as const;

type SearchToolArgs = {
	pattern: string;
	path?: string;
	glob?: string;
	ignoreCase?: boolean;
	literal?: boolean;
	context?: number;
	limit?: number;
};

const DEFAULT_LIMIT = 100;

export interface SearchDetails {
	truncation?: { truncated: boolean; maxBytes?: number };
	matchLimitReached?: number;
	linesTruncated?: boolean;
	[key: string]: unknown;
}

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (typeof raw === "string") return { pattern: raw };
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
	const args = raw as Record<string, unknown>;
	return {
		pattern: args.pattern ?? args.query ?? args.regex,
		path: args.path ?? args.file_path ?? args.directory,
		glob: args.glob,
		ignoreCase: args.ignoreCase,
		literal: args.literal,
		context: args.context ?? 0,
		limit: args.limit ?? args.max_results ?? args.maxResults ?? DEFAULT_LIMIT,
	};
}

/** Pluggable operations for grep. Override to delegate to remote systems. */
interface SearchOperations {
	isDirectory: (p: string) => Promise<boolean>;
	readFile: (p: string) => Promise<string>;
}

const defaultOps: SearchOperations = {
	isDirectory: async (p) => (await fsStat(p)).isDirectory(),
	readFile: (p) => fsReadFile(p, "utf-8"),
};

export const grep: Tool = {
	readOnly: true,
	executionMode: "parallel",
	name: "grep",
	label: "Search Files",
	hookAliases: ["Grep"],
	description:
		"Search file contents for a pattern. Returns matching lines with file paths and line numbers. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.",
	promptSnippet: "Search file contents with pattern matching and line numbers",
	promptGuidelines: ["Use grep to search file contents; use find to search by name"],
	parameters: grepSchema,
	prepareArguments,
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const {
			pattern,
			path: searchDir,
			glob,
			ignoreCase,
			literal,
			context,
			limit,
		} = args as SearchToolArgs;

		if (!pattern) return "Error: pattern is required.";

		const rgPath = await ensureTool("rg");
		if (!rgPath) return "Error: ripgrep (rg) is not installed.";

		const searchPath = searchDir
			? resolvePath(ctx.cwd, searchDir)
			: ctx.cwd || ".";
		ensureInsideCwd(ctx.cwd, searchPath, ctx.allowedPaths, ctx.allowAllPaths);
		const ops = defaultOps;
		let isDirectory: boolean;
		try {
			isDirectory = await ops.isDirectory(searchPath);
		} catch (_e: unknown) {
			return `Error: Path not found: ${searchPath}`;
		}

		const contextValue = context && context > 0 ? context : 0;
		const effectiveLimit = Math.max(1, limit ?? DEFAULT_LIMIT);
		const formatPath = (filePath: string): string => {
			if (isDirectory) {
				const relative = path.relative(searchPath, filePath);
				if (relative && !relative.startsWith("..")) {
					return relative.replace(/\\/g, "/");
				}
			}
			return path.basename(filePath);
		};

		const fileCache = new Map<string, string[]>();
		const getFileLines = async (filePath: string): Promise<string[]> => {
			let lines = fileCache.get(filePath);
			if (!lines) {
				try {
					const content = await ops.readFile(filePath);
					lines = content
						.replace(/\r\n/g, "\n")
						.replace(/\r/g, "\n")
						.split("\n");
				} catch (_e: unknown) {
					lines = [];
				}
				fileCache.set(filePath, lines);
			}
			return lines;
		};

		const argsRg: string[] = ["--json", "--line-number", "--color=never", "--hidden"];
		if (ignoreCase) argsRg.push("--ignore-case");
		if (literal) argsRg.push("--fixed-strings");
		if (glob) argsRg.push("--glob", glob);
		argsRg.push("--", pattern, searchPath);

		return new Promise<string | ToolResult>((resolve) => {
			if (ctx.signal?.aborted) {
				resolve("Error: Command aborted");
				return;
			}
			let settled = false;
			const settle = (value: string | ToolResult) => {
				if (settled) return;
				settled = true;
				resolve(value);
			};
			const child = spawn(rgPath, argsRg, { stdio: ["ignore", "pipe", "pipe"] });
			const rl = createInterface({ input: child.stdout });

			let aborted = false;
			let killedDueToLimit = false;
			const onAbort = () => {
				aborted = true;
				if (!child.killed) child.kill();
			};
			ctx.signal?.addEventListener("abort", onAbort, { once: true });

			let stderr = "";
			let matchCount = 0;
			let matchLimitReached = false;
			let linesTruncated = false;
			const matches: Array<{ filePath: string; lineNumber: number; lineText?: string }> = [];

			rl.on("line", (line) => {
				if (!line.trim() || matchCount >= effectiveLimit) return;
				let event: unknown;
				try {
					event = JSON.parse(line);
				} catch (_e: unknown) {
					return;
				}
				if (
					event &&
					typeof event === "object" &&
					"type" in event &&
					(event as { type: string }).type === "match"
				) {
					const e = event as unknown as {
						data: { path: { text: string }; line_number: number; lines?: { text?: string } };
					};
					if (e.data?.path?.text && typeof e.data.line_number === "number") {
						matchCount++;
						matches.push({
							filePath: e.data.path.text,
							lineNumber: e.data.line_number,
							lineText: e.data.lines?.text,
						});
						if (matchCount >= effectiveLimit) {
							matchLimitReached = true;
							killedDueToLimit = true;
							if (!child.killed) child.kill();
						}
					}
				}
			});

			child.stderr?.on("data", (chunk) => {
				stderr += chunk.toString();
			});

			child.on("close", (code) => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();
				void (async () => {
					if (aborted || ctx.signal?.aborted) {
						settle("Error: Command aborted");
						return;
					}

					if (!killedDueToLimit && code !== 0 && code !== 1) {
						const message = stderr.trim() || `ripgrep exited with code ${code}`;
						settle(`Error: ${message}`);
						return;
					}

					if (matchCount === 0) {
						settle("No matches found.");
						return;
					}

					const outputLines: string[] = [];

					for (const match of matches) {
						const relativePath = formatPath(match.filePath);

						// Fast-path (no context): use lineText from rg JSON, skip file re-read.
						if (contextValue === 0) {
							const rawLine = match.lineText ?? (await getFileLines(match.filePath))[match.lineNumber - 1] ?? "";
							const { text: truncatedText, wasTruncated } = truncateLine(rawLine.replace(/\r\n?/g, ""));
							if (wasTruncated) linesTruncated = true;
							outputLines.push(`${relativePath}:${match.lineNumber}: ${truncatedText}`);
							continue;
						}

						const lines = await getFileLines(match.filePath);
						if (!lines.length) {
							outputLines.push(
								`${relativePath}:${match.lineNumber}: (unable to read file)`,
							);
							continue;
						}

						const start = Math.max(1, match.lineNumber - contextValue);
						const end = Math.min(lines.length, match.lineNumber + contextValue);
						for (let current = start; current <= end; current++) {
							const lineText = lines[current - 1] ?? "";
							const sanitized = lineText.replace(/\r/g, "");
							const { text: truncatedText, wasTruncated } = truncateLine(sanitized);
							if (wasTruncated) linesTruncated = true;
							if (current === match.lineNumber) {
								outputLines.push(`${relativePath}:${current}: ${truncatedText}`);
							} else {
								outputLines.push(`${relativePath}-${current}- ${truncatedText}`);
							}
						}
					}

					const rawOutput = outputLines.join("\n");
					const truncation = truncateHead(rawOutput);
					let output = truncation.content;

					const details: SearchDetails = {};
					const notices: string[] = [];

					if (matchLimitReached) {
						notices.push(
							`${effectiveLimit} matches limit reached. Use limit=${effectiveLimit * 2} for more`,
						);
						details.matchLimitReached = effectiveLimit;
					}
					if (truncation.truncated) {
						notices.push(`${formatSize(truncation.maxBytes)} limit`);
						details.truncation = {
							truncated: true,
							maxBytes: truncation.maxBytes,
						};
					}
					if (linesTruncated) {
						notices.push("some lines truncated");
						details.linesTruncated = true;
					}

					if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;

					settle({
						content: output,
						details,
					});
				})();
			});

			child.on("error", (error) => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();
				settle(`Error: Failed to run ripgrep (rg): ${error.message}`);
			});
		});
	},
};
