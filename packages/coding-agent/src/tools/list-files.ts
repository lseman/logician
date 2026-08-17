// ── list_files tool ───────────────────────────────────────────────────────────────
// List directory contents. Returns entries sorted alphabetically (case-insensitive),
// with "/" suffix for directories. Supports glob filtering and byte truncation.
// Ported from Pi's ls tool with logician integration.

import { readdir as fsReaddir, stat as fsStat } from "node:fs/promises";
import path from "node:path";
import type { Tool, ToolResult } from "@logician/agent-core/agent/types/index.ts";
import {
	ensureInsideCwd,
	resolvePath,
} from "@logician/agent-core/tools/shared/path-utils.ts";
import { formatSize, truncateHead } from "./truncate.ts";

const lsSchema = {
	type: "object",
	properties: {
		path: {
			type: "string",
			description: "Directory to list (default: current directory)",
		},
		limit: {
			type: "number",
			description: "Maximum number of entries to return (default: 500)",
		},
	},
} as const;

type ListFilesArgs = {
	path?: string;
	limit?: number;
};

const DEFAULT_LIMIT = 500;

export interface ListFilesDetails {
	truncation?: { truncated: boolean; maxBytes?: number };
	entryLimitReached?: number;
	[key: string]: unknown;
}

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
	const args = raw as Record<string, unknown>;
	return {
		path: args.path ?? args.directory ?? args.dir,
		limit: args.limit ?? args.max_files,
	};
}

/** Pluggable operations for ls. Override to delegate to remote systems. */
interface LsOperations {
	exists: (p: string) => Promise<boolean>;
	stat: (p: string) => Promise<{ isDirectory: () => boolean }>;
	readdir: (p: string) => Promise<string[]>;
}

const defaultOps: LsOperations = {
	exists: async p => {
		try {
			await fsStat(p);
			return true;
		} catch (_e: unknown) {
			return false;
		}
	},
	stat: fsStat,
	readdir: fsReaddir,
};

export const list_files: Tool = {
	readOnly: true,
	cacheable: true,
	executionMode: "parallel",
	name: "list_files",
	label: "List Files",
	hookAliases: ["LS"],
	description:
		"List directory contents. Returns entries sorted alphabetically with '/' suffix for directories. Supports glob filtering. Output is truncated to 500 entries or 50KB (whichever is hit first).",
	promptSnippet:
		"List directory contents with sorted entries and directory indicators",
	promptGuidelines: ["Use list_files (ls) for directory listings"],
	parameters: lsSchema,
	prepareArguments,
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const ops = defaultOps;
		const { path: dirPathStr, limit } = args as ListFilesArgs;

		const safePath = resolvePath(ctx.cwd, dirPathStr || ".");
		ensureInsideCwd(ctx.cwd, safePath, ctx.allowedPaths, ctx.allowAllPaths);
		const effectiveLimit = Math.max(1, Number(limit) || DEFAULT_LIMIT);

		// Check if path exists.
		if (!(await ops.exists(safePath))) {
			return `Error: Path not found: ${safePath}`;
		}

		// Check if path is a directory.
		let isDir: boolean;
		try {
			isDir = (await ops.stat(safePath)).isDirectory();
		} catch (_e: unknown) {
			return `Error: Not a directory: ${safePath}`;
		}
		if (!isDir) {
			return `Error: Not a directory: ${safePath}`;
		}

		// Read directory entries.
		let entries: string[];
		try {
			entries = await ops.readdir(safePath);
		} catch (err: unknown) {
			const message = err instanceof Error ? err.message : String(err);
			return `Error: Cannot read directory: ${message}`;
		}

		// Sort alphabetically, case-insensitive.
		entries.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

		// Format entries with directory indicators.
		const results: string[] = [];
		let entryLimitReached = false;

		for (const entry of entries) {
			if (results.length >= effectiveLimit) {
				entryLimitReached = true;
				break;
			}

			const fullPath = path.join(safePath, entry);
			let suffix = "";
			try {
				const entryStat = await ops.stat(fullPath);
				if (entryStat.isDirectory()) suffix = "/";
			} catch (_e: unknown) {
				// Skip entries we cannot stat.
				continue;
			}
			results.push(entry + suffix);
		}

		if (results.length === 0) {
			return "(empty directory)";
		}

		const rawOutput = results.join("\n");
		const truncation = truncateHead(rawOutput, {
			maxLines: Number.MAX_SAFE_INTEGER,
		});
		let output = truncation.content;

		const details: ListFilesDetails = {};
		const notices: string[] = [];

		if (entryLimitReached) {
			notices.push(
				`${effectiveLimit} entries limit reached. Use limit=${effectiveLimit * 2} for more`,
			);
			details.entryLimitReached = effectiveLimit;
		}
		if (truncation.truncated) {
			notices.push(`${formatSize(truncation.maxBytes)} limit`);
			details.truncation = { truncated: true, maxBytes: truncation.maxBytes };
		}

		if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;

		return {
			content: output,
			details: Object.keys(details).length > 0 ? details : undefined,
		};
	},
};
