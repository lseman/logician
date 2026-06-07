// ── list_files tool ───────────────────────────────────────────────────────────────
// Fast repository file listing, preferring ripgrep's gitignore-aware scanner.

import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { promisify } from "node:util";
import type { Tool } from "../types.ts";
import { ensureInsideCwd, resolvePath } from "./helpers.ts";

const execFileAsync = promisify(execFile);

export const list_files: Tool = {
	name: "list_files",
	executionMode: "parallel",
	description:
		"List files under a path, respecting gitignore when rg is available.",
	parameters: {
		type: "object",
		properties: {
			path: {
				type: "string",
				description: "Directory to list, defaults to cwd",
			},
			glob: {
				type: "string",
				description: "Optional glob filter, e.g. '*.ts'",
			},
			limit: {
				type: "number",
				description: "Maximum files to return, default 500",
			},
		},
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			path: args.path ?? args.directory ?? args.dir,
			glob: args.glob ?? args.pattern,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const basePath = resolvePath(ctx.cwd, String(args.path || "."));
		ensureInsideCwd(ctx.cwd, basePath);
		const limit = Math.max(1, Number(args.limit) || 500);
		const glob = String(args.glob || "");

		try {
			const cmd = ["--files"];
			if (glob) cmd.push("-g", glob);
			cmd.push(basePath);
			const { stdout } = await execFileAsync("rg", cmd, {
				cwd: ctx.cwd || process.cwd(),
				timeout: 10000,
				maxBuffer: 1024 * 1024,
			});
			const lines = stdout.split("\n").filter(Boolean).slice(0, limit);
			return lines.join("\n") || "No files found.";
		} catch {
			const files: string[] = [];
			walk(basePath, files, limit, ctx.cwd || process.cwd());
			return files.join("\n") || "No files found.";
		}
	},
};

function walk(dir: string, out: string[], limit: number, cwd: string): void {
	if (out.length >= limit) return;
	if (!fs.existsSync(dir)) return;
	const stat = fs.statSync(dir);
	if (!stat.isDirectory()) {
		out.push(path.relative(cwd, dir));
		return;
	}
	for (const entry of fs.readdirSync(dir).sort()) {
		if (entry === ".git" || entry === "node_modules" || entry === "dist")
			continue;
		const full = path.join(dir, entry);
		const s = fs.statSync(full);
		if (s.isDirectory()) walk(full, out, limit, cwd);
		else out.push(path.relative(cwd, full));
		if (out.length >= limit) return;
	}
}
