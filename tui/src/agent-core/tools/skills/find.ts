// ── find tool ─────────────────────────────────────────────────────────────────────
// Find files by glob pattern using fd (falls back to rg --files).
// fd: --glob, --hidden, --no-require-git, --full-path for path-containing patterns.

import { createInterface } from "node:readline";
import { spawn } from "node:child_process";
import path from "node:path";
import type { Tool } from "../../core/types.ts";
import { ensureInsideCwd, resolvePath } from "../shared/helpers.ts";
import { DEFAULT_MAX_BYTES, formatSize, truncateHead } from "./truncate.ts";
import { ensureTool } from "../shared/tools-manager.ts";

const DEFAULT_LIMIT = 1000;

function toPosixPath(p: string): string {
	return p.split(path.sep).join("/");
}

export const find: Tool = {
	readOnly: true,
	name: "find",
	label: "Find Files",
	executionMode: "parallel",
	description:
		"Find files by glob pattern, e.g. '*.ts', '**/*.json', 'src/**/*.test.ts'. " +
		"Respects .gitignore. Includes hidden files. Returns paths relative to the search directory. " +
		`Truncated to ${DEFAULT_LIMIT} results or ${DEFAULT_MAX_BYTES / 1024}KB.`,
	promptSnippet: "Find files by name pattern (supports glob, includes hidden files)",
	promptGuidelines: ["Use find to search by name pattern; use grep for content search"],
	parameters: {
		type: "object",
		properties: {
			pattern: {
				type: "string",
				description: "Glob pattern to match files, e.g. '*.ts', '**/*.json', 'src/**/*.spec.ts'",
			},
			path: {
				type: "string",
				description: "Directory to search (default: cwd)",
			},
			limit: {
				type: "number",
				description: "Max results (default: 1000)",
			},
		},
		required: ["pattern"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") return { pattern: raw };
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			pattern: args.pattern ?? args.glob ?? args.query,
			path: args.path ?? args.directory ?? args.dir,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const pattern = String(args.pattern);
		const searchPath = resolvePath(ctx.cwd, String(args.path || "."));
		ensureInsideCwd(ctx.cwd, searchPath);
		const limit = Math.max(1, Number(args.limit) || DEFAULT_LIMIT);

		// Resolve fd path before entering the Promise so errors surface cleanly.
		const fdPath = await ensureTool("fd");
		if (!fdPath) {
			return resolveFallback(pattern, searchPath, limit, ctx.signal);
		}

		return new Promise<string>((resolve) => {
			if (ctx.signal?.aborted) {
				resolve("Error: Command aborted");
				return;
			}

			// Build fd args. --no-require-git applies .gitignore semantics outside git repos.
			// --full-path needed for path-containing patterns like 'src/**/*.ts'.
			const fdArgs: string[] = [
				"--glob",
				"--color=never",
				"--hidden",
				"--no-require-git",
				"--max-results", String(limit),
			];

			// Patterns containing '/' need --full-path so fd matches against the full path.
			let effectivePattern = pattern;
			if (pattern.includes("/")) {
				fdArgs.push("--full-path");
				if (!pattern.startsWith("/") && !pattern.startsWith("**/") && pattern !== "**") {
					effectivePattern = `**/${pattern}`;
				}
			}

			fdArgs.push("--", effectivePattern, searchPath);

			const child = spawn(fdPath, fdArgs, { stdio: ["ignore", "pipe", "pipe"] });
			const rl = createInterface({ input: child.stdout });
			let stderr = "";
			const lines: string[] = [];
			let killedDueToLimit = false;

			const onAbort = () => {
				if (!child.killed) child.kill();
			};
			ctx.signal?.addEventListener("abort", onAbort, { once: true });

			child.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

			rl.on("line", (line) => {
				if (line) lines.push(line);
				if (lines.length >= limit && !killedDueToLimit) {
					killedDueToLimit = true;
					child.kill();
				}
			});

			child.on("error", (err) => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();
				resolve(`Error: Failed to run fd: ${err.message}`);
			});

			child.on("close", (code) => {
				ctx.signal?.removeEventListener("abort", onAbort);
				rl.close();

				if (ctx.signal?.aborted) {
					resolve("Error: Command aborted");
					return;
				}

				if (lines.length === 0) {
					if (!killedDueToLimit && code !== 0 && code !== 1) {
						const msg = stderr.trim() || `fd exited with code ${code}`;
						// Might mean fd not found at this path — fall back to rg
						resolve(`Error: ${msg}`);
					} else {
						resolve("No files found matching pattern.");
					}
					return;
				}

				// Relativize paths
				const relativized = lines.map((raw) => {
					const line = raw.replace(/\r$/, "").trim();
					if (!line) return null;
					const hadSlash = line.endsWith("/") || line.endsWith("\\");
					let rel = line.startsWith(searchPath)
						? line.slice(searchPath.length + 1)
						: path.relative(searchPath, line);
					if (hadSlash && !rel.endsWith("/")) rel += "/";
					return toPosixPath(rel);
				}).filter(Boolean) as string[];

				const limitReached = killedDueToLimit || relativized.length >= limit;
				const rawOutput = relativized.join("\n");
				const t = truncateHead(rawOutput, { maxLines: Number.MAX_SAFE_INTEGER });
				let out = t.content;
				const notices: string[] = [];
				if (limitReached) {
					notices.push(`${limit} results limit reached. Use limit=${limit * 2} or refine pattern`);
				}
				if (t.truncated) notices.push(`${formatSize(DEFAULT_MAX_BYTES)} limit reached`);
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
			{ timeout: 10000, maxBuffer: 1024 * 1024, signal, killSignal: "SIGKILL" as const },
		);
		const all = stdout.split("\n").filter(Boolean);
		if (all.length === 0) return "No files found matching pattern.";
		const limited = all.slice(0, limit);
		const t = truncateHead(limited.join("\n"), { maxLines: Number.MAX_SAFE_INTEGER });
		let out = t.content;
		if (all.length > limit) out += `\n\n[${limit} results limit reached. Use limit=${limit * 2} or refine pattern]`;
		if (t.truncated) out += `\n\n[${formatSize(DEFAULT_MAX_BYTES)} limit reached]`;
		return out;
	} catch (e: unknown) {
		const err = e as { name?: string; code?: number | string; stderr?: string };
		if (err.name === "AbortError" || err.code === "ABORT_ERR") return "Error: Command aborted";
		if (err.code === 1) return "No files found matching pattern.";
		return `Error: ${err.stderr || String(e)}`;
	}
}
