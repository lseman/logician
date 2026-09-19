// ── grep tool ────────────────────────────────────────────────────────────────
// Search files using ripgrep (rg) with structured JSON output.
// Features: line/byte truncation, context lines, ignoreCase, case, literal mode,
// cross-line patterns, semicolon-delimited paths, file:LINE-RANGE selectors,
// skip pagination, file caching, AbortSignal support, structured ToolResult.

import { spawn } from "node:child_process";
import { readFile as fsReadFile, stat as fsStat } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline";
import type { Tool, ToolResult } from "@logician/log-core";
import { extractInternalUrlScheme } from "../../runtime/bridge/support/internal-urls/parse.ts";
import type { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";
import { ensureTool } from "./external-tools.ts";
import { resolvePath } from "./support/utils/path-utils.ts";
import {
	formatSize,
	truncateHead,
	truncateLine,
} from "./support/utils/truncate.ts";
import { loadNative } from "./support/native-addon.ts";

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
		skip: {
			type: "number",
			description: "Skip the first N matches before returning results (default: 0)",
		},
		case: {
			type: "boolean",
			description: "Case-sensitive search (default: true). Setting to false is equivalent to ignoreCase: true.",
		},
	},
	required: ["pattern"],
} as const;

type SearchToolArgs = {
	pattern: string;
	path?: string;
	glob?: string;
	ignoreCase?: boolean;
	case?: boolean;
	literal?: boolean;
	context?: number;
	limit?: number;
	skip?: number;
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
		pattern: args.pattern,
		path: args.path,
		glob: args.glob,
		case: args.case,
		literal: args.literal,
		context: args.context ?? 0,
		limit: args.limit ?? DEFAULT_LIMIT,
		skip: args.skip ?? 0,
	};
}
/**
 * Search in-memory content using the native grep engine (PCRE2-backed).
 * Replaces the old JS-only regex search.
 */
async function searchInMemoryContent(
	content: string,
	displayLabel: string,
	opts: {
		pattern: string;
		ignoreCase?: boolean | undefined;
		literal?: boolean | undefined;
		context?: number | undefined;
		limit?: number | undefined;
		skip?: number | undefined;
	},
): Promise<string> {
	const { pattern, ignoreCase, literal, context, limit, skip } = opts;
	const searchPattern = literal
		? pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
		: pattern;

	const native = await loadNative();
	const result = native.search(content, {
		pattern: searchPattern,
		ignoreCase,
		context: context ?? 0,
		maxCount: limit ?? DEFAULT_LIMIT,
		offset: skip ?? 0,
	});

	if (result.error) return `Error: ${result.error}`;

	if (result.matches.length === 0) return "No matches found.";

	const outputLines: string[] = [];
	for (const m of result.matches) {
		outputLines.push(`${displayLabel}:${m.lineNumber}: ${m.line}`);
	}

	const rawOutput = outputLines.join("\n");
	const truncation = truncateHead(rawOutput);
	let output = truncation.content;
	const notices: string[] = [];
	if (result.matchCount > (limit ?? DEFAULT_LIMIT)) {
		notices.push(
			`${limit ?? DEFAULT_LIMIT} matches limit reached. Use limit=${(limit ?? DEFAULT_LIMIT) * 2} for more`,
		);
	}
	if (truncation.truncated)
		notices.push(`${formatSize(truncation.maxBytes)} limit`);
	if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;
	return output;
}

interface RawGrepMatch {
	filePath: string;
	lineNumber: number;
	line?: string | undefined;
}

/**
 * Render matches (from either the native grep engine or the ripgrep
 * fallback) into the tool's text output, applying line-range filtering,
 * context expansion (re-reading the file — neither source is asked for
 * context lines directly), truncation, and result notices.
 */
async function buildGrepOutput(
	matches: RawGrepMatch[],
	matchLimitReached: boolean,
	effectiveLimit: number,
	contextValue: number,
	lineRangeMap: Map<string, [number, number]>,
	formatPath: (filePath: string) => Promise<string>,
	getFileLines: (filePath: string) => Promise<string[]>,
): Promise<string | ToolResult> {
	const outputLines: string[] = [];
	let linesTruncated = false;

	for (const match of matches) {
		const range = lineRangeMap.get(match.filePath);
		if (range && (match.lineNumber < range[0] || match.lineNumber > range[1])) {
			continue;
		}

		const relativePath = await formatPath(match.filePath);

		// Fast-path (no context): use the match's own line text, skip file re-read.
		if (contextValue === 0) {
			const rawLine =
				match.line ??
				(await getFileLines(match.filePath))[match.lineNumber - 1] ??
				"";
			const { text: truncatedText, wasTruncated } = truncateLine(
				rawLine.replace(/\r\n?/g, ""),
			);
			if (wasTruncated) linesTruncated = true;
			outputLines.push(`${relativePath}:${match.lineNumber}: ${truncatedText}`);
			continue;
		}

		const lines = await getFileLines(match.filePath);
		if (!lines.length) {
			outputLines.push(`${relativePath}:${match.lineNumber}: (unable to read file)`);
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
	if (!output) return "No matches found.";

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
		details.truncation = { truncated: true, maxBytes: truncation.maxBytes };
	}
	if (linesTruncated) {
		notices.push("some lines truncated");
		details.linesTruncated = true;
	}

	if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;

	return { content: output, details };
}

/**
 * Search a single directory or file using the native grep engine
 * (ripgrep-equivalent, PCRE2-backed, parallel-streaming). Ported from
 * oh-my-pi's crates/pi-natives/src/grep.rs.
 */
async function runSinglePathGrep(opts: {
	pattern: string;
	searchPath: string;
	glob?: string | undefined;
	effectiveIgnoreCase: boolean;
	literal?: boolean | undefined;
	hasCrossLine: boolean;
	effectiveLimit: number;
	effectiveSkip: number;
	contextValue: number;
	lineRangeMap: Map<string, [number, number]>;
	formatPath: (filePath: string) => Promise<string>;
	getFileLines: (filePath: string) => Promise<string[]>;
}): Promise<string | ToolResult> {
	const {
		pattern,
		searchPath,
		glob,
		effectiveIgnoreCase,
		literal,
		hasCrossLine,
		effectiveLimit,
		effectiveSkip,
		contextValue,
		lineRangeMap,
		formatPath,
		getFileLines,
	} = opts;

	// Native grep has no literal/fixed-strings flag; escape like the
	// in-memory search path already does.
	const searchPattern = literal
		? pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
		: pattern;

	let isDir: boolean;
	try {
		isDir = (await fsStat(searchPath)).isDirectory();
	} catch {
		return `Error: Path not found: ${searchPath}`;
	}

	const native = await loadNative();
	let result: Awaited<ReturnType<typeof native.grep>>;
	try {
		result = await native.grep({
			pattern: searchPattern,
			path: searchPath,
			...(glob !== undefined ? { glob } : {}),
			ignoreCase: effectiveIgnoreCase,
			multiline: hasCrossLine,
			hidden: true,
			gitignore: true,
			maxCount: effectiveLimit,
			offset: effectiveSkip,
		});
	} catch (err) {
		return `Error: ${err instanceof Error ? err.message : String(err)}`;
	}

	if (result.matches.length === 0) return "No matches found.";

	// Directory searches return root-relative paths; file/context lookups
	// need real filesystem paths.
	const matches: RawGrepMatch[] = result.matches.map(m => ({
		filePath: isDir ? path.join(searchPath, m.path) : m.path,
		lineNumber: m.lineNumber,
		line: m.line,
	}));

	return buildGrepOutput(
		matches,
		result.limitReached ?? false,
		effectiveLimit,
		contextValue,
		lineRangeMap,
		formatPath,
		getFileLines,
	);
}

export function createGrepTool(router?: InternalUrlRouter): Tool {
	return {
		readOnly: true,
		executionMode: "parallel",
		name: "grep",
		label: "Search Files",
		hookAliases: ["Grep"],
		description:
			"Search file contents for a pattern. Returns matching lines with file paths and line numbers. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars. " +
			"path also accepts internal resource URLs (e.g. skill://name, memory://list) — resources backed by a real file are searched with full rg feature parity; others are searched in-process. A directory-shaped resource with no backing file is rejected.",
		promptSnippet:
			"Search file contents with pattern matching and line numbers",
		promptGuidelines: [
			"grep also accepts internal resource URLs (skill://, memory://, local://, ...) as path",
		],
		parameters: grepSchema,
		prepareArguments,
		execute: async (args, ctx): Promise<string | ToolResult> => {
			const {
				pattern,
				path: searchDirArg,
				glob,
				ignoreCase,
				case: caseSensitive,
				literal,
				context,
				limit,
				skip,
			} = args as SearchToolArgs;

			if (!pattern) return "Error: pattern is required.";

			let searchDir = searchDirArg;
			let displayLabelOverride: string | undefined;

			if (searchDir && router && extractInternalUrlScheme(searchDir)) {
				try {
					const pathOnly = await router.resolve(searchDir, {
						...ctx,
						pathOnly: true,
					});
					if (pathOnly.isDirectory && !pathOnly.sourcePath) {
						return `Error: grep cannot recurse the listing at ${searchDir}; grep a specific resource under it, or read ${searchDir} to list its entries.`;
					}
					if (pathOnly.sourcePath) {
						displayLabelOverride = searchDir;
						searchDir = pathOnly.sourcePath;
					} else {
						const full = await router.resolve(searchDir, ctx);
						if (full.isDirectory) {
							return `Error: grep cannot recurse the listing at ${searchDir}; grep a specific resource under it, or read ${searchDir} to list its entries.`;
						}
						return await searchInMemoryContent(full.content, searchDir, {
							pattern,
							ignoreCase,
							literal,
							context,
							limit,
						});
					}
				} catch (error) {
					return `Error: ${error instanceof Error ? error.message : String(error)}`;
				}
			}
			// ── semicolon-delimited paths ──────────────────────────────────
			const rawPaths = searchDir ? searchDir.split(";") : [ctx.cwd || "."];

			const resolvedPaths: string[] = [];
			const lineRangeMap = new Map<string, [number, number]>();

			for (const raw of rawPaths) {
				const trimmed = raw.trim();
				if (!trimmed) continue;

				// ── internal URL ──────────────────────────────────────────
				if (router && extractInternalUrlScheme(trimmed)) {
					try {
						const pathOnly = await router.resolve(trimmed, {
							...ctx,
							pathOnly: true,
						});
						if (pathOnly.isDirectory && !pathOnly.sourcePath) {
							return `Error: grep cannot recurse the listing at ${trimmed}; grep a specific resource under it, or read ${trimmed} to list its entries.`;
						}
						if (pathOnly.sourcePath) {
							displayLabelOverride ??= trimmed;
							resolvedPaths.push(pathOnly.sourcePath);
							continue;
						}
						const full = await router.resolve(trimmed, ctx);
						if (full.isDirectory) {
							return `Error: grep cannot recurse the listing at ${trimmed}; grep a specific resource under it, or read ${trimmed} to list its entries.`;
						}
						return await searchInMemoryContent(full.content, trimmed, {
							pattern,
							ignoreCase,
							literal,
							context,
							limit,
						});
					} catch (error) {
						return `Error: ${error instanceof Error ? error.message : String(error)}`;
					}
				}

				// ── file:LINE1-LINE2 selector ─────────────────────────────
				const colonIdx = trimmed.indexOf(":");
				let filePath: string;
				let lineRange: [number, number] | null = null;
				if (colonIdx !== -1) {
					const maybeRange = trimmed.slice(colonIdx + 1).trim();
					const rangeMatch = maybeRange.match(/^(\d+)-(\d+)$/);
					if (rangeMatch) {
						filePath = trimmed.slice(0, colonIdx);
						lineRange = [
							parseInt(rangeMatch[1], 10),
							parseInt(rangeMatch[2], 10),
						];
					} else {
						filePath = trimmed;
					}
				} else {
					filePath = trimmed;
				}

				// ── resolve + validate ────────────────────────────────────
				const absPath = filePath.startsWith("/")
					? filePath
					: resolvePath(ctx.cwd || ".", filePath);

				let isDir: boolean;
				try {
					isDir = (await fsStat(absPath)).isDirectory();
				} catch {
					return `Error: Path not found: ${filePath}`;
				}

				if (isDir) {
					resolvedPaths.push(absPath);
				} else {
					resolvedPaths.push(absPath);
				}

				if (lineRange) {
					lineRangeMap.set(absPath, lineRange);
				}
			}

			if (resolvedPaths.length === 0) {
				return "Error: No valid paths to search.";
			}

			// ── case sensitivity (case takes priority over ignoreCase) ──
			const effectiveIgnoreCase =
				caseSensitive !== undefined
					? !caseSensitive
					: ignoreCase ?? false;

			// ── cross-line pattern detection (literal \n in pattern) ───
			const hasCrossLine = pattern.includes("\\n");

			const contextValue = context && context > 0 ? context : 0;
			const effectiveLimit = Math.max(1, limit ?? DEFAULT_LIMIT);
			const effectiveSkip = Math.max(0, skip ?? 0);

			const formatPath = async (filePath: string): Promise<string> => {
				if (displayLabelOverride) return displayLabelOverride;
				// For multi-path, always use relative paths
				if (resolvedPaths.length > 1) {
					const relative = path.relative(ctx.cwd || ".", filePath);
					return relative.replace(/\\/g, "/");
				}
				const rp = resolvedPaths[0];
				if (rp && (await fsStat(rp)).isDirectory()) {
					const relative = path.relative(rp, filePath);
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
						const content = await fsReadFile(filePath, "utf-8");
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

			if (ctx.signal?.aborted) return "Error: Command aborted";

			// ── Single path: native grep engine ────────────────────────
			if (resolvedPaths.length === 1) {
				return runSinglePathGrep({
					pattern,
					searchPath: resolvedPaths[0] as string,
					glob,
					effectiveIgnoreCase,
					literal,
					hasCrossLine,
					effectiveLimit,
					effectiveSkip,
					contextValue,
					lineRangeMap,
					formatPath,
					getFileLines,
				});
			}

			// ── Multiple paths: fall back to ripgrep ────────────────────
			const argsRg: string[] = [
				"--json",
				"--line-number",
				"--color=never",
				"--hidden",
			];
			if (effectiveIgnoreCase) argsRg.push("--ignore-case");
			if (hasCrossLine) argsRg.push("-U");
			if (literal) argsRg.push("--fixed-strings");
			if (glob) argsRg.push("--glob", glob);
			argsRg.push("--", pattern, ...resolvedPaths);

			const rgPath = await ensureTool("rg");
			if (!rgPath) return "Error: ripgrep (rg) is not installed.";
			return new Promise<string | ToolResult>(resolve => {
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
				const child = spawn(rgPath, argsRg, {
					stdio: ["ignore", "pipe", "pipe"],
				});
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
				let collectedCount = 0;
				let matchLimitReached = false;
				const matches: RawGrepMatch[] = [];

				rl.on("line", line => {
					if (!line.trim()) return;
					let event: unknown;
					try {
						event = JSON.parse(line);
					} catch {
						return;
					}
					if (
						event &&
						typeof event === "object" &&
						"type" in event &&
						(event as { type: string }).type === "match"
					) {
						const e = event as unknown as {
							data: {
								path: { text: string };
								line_number: number;
								lines?: { text?: string };
							};
						};
						if (e.data?.path?.text && typeof e.data.line_number === "number") {
							matchCount++;
							// Skip first N matches
							if (matchCount <= effectiveSkip) return;
							collectedCount++;
							matches.push({
								filePath: e.data.path.text,
								lineNumber: e.data.line_number,
								line: e.data.lines?.text,
							});
							// Kill when we have enough (skip + limit)
							if (collectedCount >= effectiveLimit) {
								matchLimitReached = true;
								killedDueToLimit = true;
								if (!child.killed) child.kill();
							}
						}
					}
				});

				child.stderr?.on("data", chunk => {
					stderr += chunk.toString();
				});

				child.on("close", code => {
					ctx.signal?.removeEventListener("abort", onAbort);
					rl.close();
					void (async () => {
						if (aborted || ctx.signal?.aborted) {
							settle("Error: Command aborted");
							return;
						}

						if (!killedDueToLimit && code !== 0 && code !== 1) {
							const message =
								stderr.trim() || `ripgrep exited with code ${code}`;
							settle(`Error: ${message}`);
							return;
						}

						if (matchCount === 0) {
							settle("No matches found.");
							return;
						}

						settle(
							await buildGrepOutput(
								matches,
								matchLimitReached,
								effectiveLimit,
								contextValue,
								lineRangeMap,
								formatPath,
								getFileLines,
							),
						);
					})();
				});

				child.on("error", error => {
					ctx.signal?.removeEventListener("abort", onAbort);
					rl.close();
					settle(`Error: Failed to run ripgrep (rg): ${error.message}`);
				});
			});
		},
	};
}

export const grep = createGrepTool();
