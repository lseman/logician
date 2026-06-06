// ── find tool ─────────────────────────────────────────────────────────────────────
// Find files by glob pattern, respecting .gitignore via ripgrep's --files scanner.

import { execFile } from "child_process";
import { promisify } from "util";
import type { Tool } from "../types.ts";
import { ensureInsideCwd, resolvePath } from "./helpers.ts";
import { DEFAULT_MAX_BYTES, formatSize, truncateHead } from "./truncate.ts";

const execFileAsync = promisify(execFile);

const DEFAULT_LIMIT = 1000;

export const find: Tool = {
    name: "find",
    description:
        `Find files by glob pattern, e.g. '*.ts', '**/*.json', 'src/**/*.test.ts'. ` +
        `Respects .gitignore. Returns paths relative to the search directory. ` +
        `Truncated to ${DEFAULT_LIMIT} results or ${DEFAULT_MAX_BYTES / 1024}KB.`,
    parameters: {
        type: "object",
        properties: {
            pattern: {
                type: "string",
                description: "Glob pattern to match files",
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
    execute: async (args, ctx): Promise<string> => {
        const pattern = String(args.pattern);
        const searchPath = resolvePath(ctx.cwd, String(args.path || "."));
        ensureInsideCwd(ctx.cwd, searchPath);
        const limit = Math.max(1, Number(args.limit) || DEFAULT_LIMIT);

        try {
            const { stdout } = await execFileAsync(
                "rg",
                ["--files", "-g", pattern, searchPath],
                {
                    cwd: ctx.cwd || process.cwd(),
                    timeout: 10000,
                    maxBuffer: 1024 * 1024,
                    signal: ctx.signal,
                    killSignal: "SIGKILL",
                },
            );
            const all = stdout.split("\n").filter(Boolean);
            if (all.length === 0) return "No files found matching pattern.";

            const limited = all.slice(0, limit);
            const t = truncateHead(limited.join("\n"), {
                maxLines: Number.MAX_SAFE_INTEGER,
            });
            let out = t.content;
            const notices: string[] = [];
            if (all.length > limit) {
                notices.push(
                    `${limit} results limit reached. Use limit=${limit * 2} or refine pattern`,
                );
            }
            if (t.truncated)
                notices.push(`${formatSize(DEFAULT_MAX_BYTES)} limit reached`);
            if (notices.length) out += `\n\n[${notices.join(". ")}]`;
            return out;
        } catch (e: unknown) {
            const error = e as {
                name?: string;
                code?: number | string;
                stderr?: string;
            };
            if (error.name === "AbortError" || error.code === "ABORT_ERR") {
                return "Error: Command aborted";
            }
            // rg exits 1 when no files match the glob.
            if (error.code === 1) return "No files found matching pattern.";
            return `Error: ${error.stderr || String(e)}`;
        }
    },
};
