// ── glob tool ─────────────────────────────────────────────────────────────────
// Merges the former find (glob pattern search) and list_files (directory
// listing) tools into one, matching oh-my-pi's glob: a single `path` that is
// either a bare directory/file or a glob pattern.
//
// - Bare directory (no glob chars) → recursive "**/*" listing of that subtree.
// - Bare file (no glob chars) → short-circuits to that one path.
// - Path containing glob chars (*, ?, [, ], {, }) → split into the longest
//   glob-char-free prefix as the search root and the remainder as the
//   glob pattern, e.g. "src/**/*.ts" → root "src", pattern "**/*.ts".
//
// Uses @logician/log-natives' native glob() engine (pi-walker-backed, ported
// from oh-my-pi) instead of shelling out to fd/rg — directories come back
// tagged with a file type, so they're suffixed with "/" here rather than via
// a separate per-entry stat pass the way list_files used to do.

import { existsSync, statSync } from "node:fs";
import path from "node:path";
import type { Tool } from "@logician/log-core";
import { loadNative } from "./support/native-addon.ts";
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

		if (ctx.signal?.aborted) return "Error: Command aborted";

		const native = await loadNative();
		let result: Awaited<ReturnType<typeof native.glob>>;
		try {
			result = await native.glob(
				{
					pattern,
					path: searchPath,
					hidden: true,
					gitignore: true,
					maxResults: limit,
				},
				null,
			);
		} catch (err) {
			return `Error: ${err instanceof Error ? err.message : String(err)}`;
		}

		if (result.matches.length === 0) return "No files found matching pattern.";

		const lines = result.matches.map(match =>
			match.fileType === native.FileType.Dir ? `${match.path}/` : match.path,
		);

		const limitReached = lines.length >= limit;
		const rawOutput = lines.join("\n");
		const t = truncateHead(rawOutput, { maxLines: Number.MAX_SAFE_INTEGER });
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
		return out;
	},
};
