// ── glob tool ─────────────────────────────────────────────────────────────────
// Merges the former find (glob pattern search) and list_files (directory
// listing) tools into one, matching oh-my-pi's glob: a single `path` that is
// either a bare directory/file or a glob pattern.
//
// - Bare directory (no glob chars) → recursive "**/*" listing of that subtree.
// - Bare file (no glob chars) → short-circuits to that one path.
// - Path containing glob chars (*, ?, [, ], {, }) → split into the longest
//   glob-char-free prefix as the search root and the remainder as the
//   fd pattern, e.g. "src/**/*.ts" → root "src", pattern "**/*.ts".
//
// Uses fd (falls back to rg --files, same as the former find tool) —
// directories are suffixed with "/" in fd's own default output, so no
// separate per-entry stat pass is needed the way list_files used to do.

import { existsSync, statSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { createInterface } from "node:readline";
import type { Tool } from "@logician/log-core";
import { ensureTool } from "./external-tools.ts";
import { ensureInsideCwd, resolvePath } from "./support/utils/path-utils.ts";
import {
	DEFAULT_MAX_BYTES,
	formatSize,
	truncateHead,
} from "./support/utils/truncate.ts";

const DEFAULT_LIMIT = 1000;
const GLOB_CHARS_RE = /[*?[\]{}]/;

function toPosixPath(p: string): string {
	return p.split(path.sep).join("/");
}

/**
 * Split a `path` argument into a glob-char-free search root and the fd
 * pattern to run from it. Returns `pattern: null` when the input has no
 * glob characters at all — the caller decides between a bare-file
 * short-circuit and a recursive (all-files, all-subdirectories) listing
 * based on what that root is.
 */
function splitBaseAndPattern(inputPath: string): {
	base: string;
	pattern: string | null;
} {
	if (!GLOB_CHARS_RE.test(inputPath)) {
		return { base: inputPath || ".", pattern: null };
	}
	const segments = inputPath.split("/");
	const baseSegments: string[] = [];
	let i = 0;
	for (; i < segments.length; i++) {
		const segment = segments[i] ?? "";
		if (GLOB_CHARS_RE.test(segment)) break;
		baseSegments.push(segment);
	}
	const base = baseSegments.length > 0 ? baseSegments.join("/") : ".";
	const pattern = segments.slice(i).join("/") || "*";
	return { base, pattern };
}

export const glob: Tool = {
	readOnly: true,
	cacheable: true,
	name: "glob",
	label: "Glob",
	hookAliases: ["Glob"],
	executionMode: "parallel",
	description:
		"List a directory or match files by glob pattern, e.g. '.', 'src', '*.ts', '**/*.json', 'src/**/*.test.ts'. " +
		"A bare directory lists its whole subtree recursively; a bare file path returns just that file. " +
		"Respects .gitignore. Includes hidden files. Directories are suffixed with '/'. " +
		`Truncated to ${DEFAULT_LIMIT} results or ${DEFAULT_MAX_BYTES / 1024}KB.`,
	promptSnippet: "List a directory (recursively) or find files by glob pattern",
	promptGuidelines: [
		"Use glob to browse structure or find files by name; use grep for content search",
	],
	parameters: {
		type: "object",
		properties: {
			path: {
				type: "string",
				description:
					"Directory, file, or glob pattern, e.g. '.', 'src', '*.ts', 'src/**/*.test.ts' (default: '.')",
			},
			limit: {
				type: "number",
				description: "Max results (default: 1000)",
			},
		},
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") return { path: raw };
		if (!raw || typeof raw !== "object") return {};
		return raw as Record<string, unknown>;
	},
	execute: async (args, ctx): Promise<string> => {
		const inputPath = String(args.path ?? "") || ".";
		const limit = Math.max(1, Number(args.limit) || DEFAULT_LIMIT);

		const { base, pattern: splitPattern } = splitBaseAndPattern(inputPath);
		const searchPath = resolvePath(ctx.cwd, base);
		ensureInsideCwd(ctx.cwd, searchPath, ctx.allowedPaths, ctx.allowAllPaths);

		let pattern = splitPattern;
		if (pattern === null) {
			if (!existsSync(searchPath)) {
				return `Error: Path not found: ${searchPath}`;
			}
			if (statSync(searchPath).isFile()) {
				const cwd = ctx.cwd || process.cwd();
				const rel = path.relative(cwd, searchPath);
				return toPosixPath(rel && !rel.startsWith("..") ? rel : searchPath);
			}
			pattern = "**/*";
		}

		// Resolve fd path before entering the Promise so errors surface cleanly.
		const fdPath = await ensureTool("fd");
		if (!fdPath) {
			return resolveFallback(pattern, searchPath, limit, ctx.signal);
		}

		return new Promise<string>(resolve => {
			if (ctx.signal?.aborted) {
				resolve("Error: Command aborted");
				return;
			}

			// Build fd args. --no-require-git applies .gitignore semantics outside git repos.
			// --full-path needed for path-containing patterns like 'src/**/*.ts' or '**/*'.
			const fdArgs: string[] = [
				"--glob",
				"--color=never",
				"--hidden",
				"--no-require-git",
				"--max-results",
				String(limit),
			];

			let effectivePattern = pattern;
			if (pattern.includes("/")) {
				fdArgs.push("--full-path");
				if (
					!pattern.startsWith("/") &&
					!pattern.startsWith("**/") &&
					pattern !== "**"
				) {
					effectivePattern = `**/${pattern}`;
				}
			}

			fdArgs.push("--", effectivePattern, searchPath);

			const child = spawn(fdPath, fdArgs, {
				stdio: ["ignore", "pipe", "pipe"],
			});
			const rl = createInterface({ input: child.stdout });
			let stderr = "";
			const lines: string[] = [];
			let killedDueToLimit = false;

			const onAbort = () => {
				if (!child.killed) child.kill();
			};
			ctx.signal?.addEventListener("abort", onAbort, { once: true });

			child.stderr?.on("data", (chunk: Buffer) => {
				stderr += chunk.toString();
			});

			rl.on("line", line => {
				if (line) lines.push(line);
				if (lines.length >= limit && !killedDueToLimit) {
					killedDueToLimit = true;
					child.kill();
				}
			});

			child.on("error", err => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();
				resolve(`Error: Failed to run fd: ${err.message}`);
			});

			child.on("close", code => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();

				if (ctx.signal?.aborted) {
					resolve("Error: Command aborted");
					return;
				}

				if (lines.length === 0) {
					if (!killedDueToLimit && code !== 0 && code !== 1) {
						const msg = stderr.trim() || `fd exited with code ${code}`;
						resolve(`Error: ${msg}`);
					} else {
						resolve("No files found matching pattern.");
					}
					return;
				}

				// Relativize paths
				const relativized = lines
					.map(raw => {
						const line = raw.replace(/\r$/, "").trim();
						if (!line) return null;
						const hadSlash = line.endsWith("/") || line.endsWith("\\");
						let rel = line.startsWith(searchPath)
							? line.slice(searchPath.length + 1)
							: path.relative(searchPath, line);
						if (hadSlash && !rel.endsWith("/")) rel += "/";
						return toPosixPath(rel);
					})
					.filter(Boolean) as string[];

				const limitReached = killedDueToLimit || relativized.length >= limit;
				const rawOutput = relativized.join("\n");
				const t = truncateHead(rawOutput, {
					maxLines: Number.MAX_SAFE_INTEGER,
				});
				let out = t.content;
				const notices: string[] = [];
				if (limitReached) {
					notices.push(
						`${limit} results limit reached. Use limit=${limit * 2} or refine pattern`,
					);
				}
				if (t.truncated)
					notices.push(`${formatSize(DEFAULT_MAX_BYTES)} limit reached`);
				if (notices.length) out += `\n\n[${notices.join(". ")}]`;
				resolve(out);
			});
		});
	},
};

/** Fallback: use rg --files when fd is not installed. */
async function resolveFallback(
	pattern: string,
	searchPath: string,
	limit: number,
	signal?: AbortSignal,
): Promise<string> {
	const rgPath = await ensureTool("rg");
	if (!rgPath) return "Error: Neither fd nor rg (ripgrep) is installed.";
	const { execFile } = await import("node:child_process");
	const { promisify } = await import("node:util");
	const execFileAsync = promisify(execFile);
	try {
		const { stdout } = await execFileAsync(
			rgPath,
			["--files", "--hidden", "-g", pattern, searchPath],
			{
				timeout: 10000,
				maxBuffer: 1024 * 1024,
				signal,
				killSignal: "SIGKILL" as const,
			},
		);
		const all = stdout.split("\n").filter(Boolean);
		if (all.length === 0) return "No files found matching pattern.";
		const limited = all.slice(0, limit);
		const t = truncateHead(limited.join("\n"), {
			maxLines: Number.MAX_SAFE_INTEGER,
		});
		let out = t.content;
		if (all.length > limit)
			out += `\n\n[${limit} results limit reached. Use limit=${limit * 2} or refine pattern]`;
		if (t.truncated)
			out += `\n\n[${formatSize(DEFAULT_MAX_BYTES)} limit reached]`;
		return out;
	} catch (err: unknown) {
		const e = err as { name?: string; code?: number | string; stderr?: string };
		if (e.name === "AbortError" || e.code === "ABORT_ERR")
			return "Error: Command aborted";
		if (e.code === 1) return "No files found matching pattern.";
		return `Error: ${e.stderr || String(err)}`;
	}
}
