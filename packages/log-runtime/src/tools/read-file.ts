// Generic text-resource reader for files, directories, and internal URLs.
import type { Tool } from "@logician/log-core";
import { InternalUrlRouter } from "../resources/router.ts";
import { parseConflictBlocks } from "../shared/conflict-resolution.ts";
import { recordRead } from "../shared/read-tracker.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
} from "../shared/truncate.ts";
import {
	formatResourceRead,
	type ReadPage,
} from "./support/format-resource-read.ts";
import { readResource } from "./support/read-resource.ts";

function readPage(args: Record<string, unknown>): ReadPage {
	for (const key of ["offset", "limit"] as const) {
		const value = args[key];
		if (
			value !== undefined &&
			(typeof value !== "number" || !Number.isSafeInteger(value) || value < 1)
		) {
			throw new Error(`${key} must be a positive integer.`);
		}
	}
	return {
		offset: (args.offset as number | undefined) ?? 1,
		limit: args.limit as number | undefined,
	};
}

export function createReadTool(router?: InternalUrlRouter): Tool {
	return {
		name: "read",
		label: "Read",
		hookAliases: ["Read"],
		readOnly: true,
		// Reads update file tracking and may resolve mutable device/resource catalogs.
		cacheable: false,
		executionMode: "parallel",
		description:
			"Read a text file, directory, or internal resource URL. Read xd:// to discover tool devices and xd://<name> for a device's documentation and JSON input schema. " +
			"All text uses numbered lines with offset/limit pagination, bounded to " +
			DEFAULT_MAX_LINES +
			" lines or " +
			formatSize(DEFAULT_MAX_BYTES) +
			". " +
			"Direct file reads also include [path#4hex] edit anchors; resource URLs are read-only. Unsupported URLs return an error. Prefix literal paths containing :// with ./. " +
			"Supports .zip-family (.zip, .jar, .war, .ear, .apk) and .tar-family (.tar, .tar.gz, .tgz) archive entries via archive.ext:path/inside/archive (bare archive.ext lists entries; other archive formats are not supported). " +
			"Supports SQLite via db.sqlite:table (schema + sample rows, bare db.sqlite lists tables) and db.sqlite:table:rowid (one row); non-rowid primary-key lookups are not supported.",
		promptSnippet:
			"Read files, directories, archives, SQLite, internal URLs, and xd:// device documentation with offset/limit pagination",
		promptGuidelines: [
			"Use read for files and internal resource URLs; only direct file reads provide edit anchors",
			"Read xd:// to discover devices, then read xd://<name> for its input schema before invoking it with write",
			"Read archive.ext:path/inside/archive for zip/tar members and db.sqlite:table[:rowid] for SQLite rows/tables — these are read-before-write tracked like direct file reads",
		],
		parameters: {
			type: "object",
			properties: {
				path: {
					type: "string",
					description:
						"File/directory path or resource URL, e.g. skill://name or xd://browser",
				},
				offset: {
					type: "integer",
					minimum: 1,
					description: "1-based first line, for files and resource URLs",
				},
				limit: {
					type: "integer",
					minimum: 1,
					description: "Maximum number of lines to read",
				},
			},
			required: ["path"],
		},
		prepareArguments: raw => {
			if (typeof raw === "string") return { path: raw };
			if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
			const args = raw as Record<string, unknown>;
			return { ...args, path: args.path ?? args.file_path ?? args.filename };
		},
		execute: async (args, ctx) => {
			try {
				if (typeof args.path !== "string" || args.path.length === 0)
					throw new Error("path must be a non-empty string.");
				const page = readPage(args);
				const read = await readResource(
					args.path,
					ctx,
					router ?? InternalUrlRouter.instance(),
				);
				const output = formatResourceRead(read, page);
				if (read.kind === "resource") return output;
				recordRead(read.path);
				// Already viewing the formatted conflicts breakdown — don't
				// append a second, redundant notice pointing back at itself.
				if (args.path.endsWith(":conflicts")) return output;
				return (
					output +
					formatConflictNotice(
						read.resource.content,
						parseConflictBlocks(read.resource.content, read.path),
					)
				);
			} catch (error) {
				return {
					content:
						"Error: " +
						(error instanceof Error ? error.message : String(error)),
					isError: true,
				};
			}
		},
	};
}

export const read = createReadTool();

/** Build a conflict notice to append to read output. */
function formatConflictNotice(
	fullContent: string,
	blocks: {
		index: number;
		oursLabel: string;
		theirsLabel: string;
		offset: number;
		file: string;
	}[],
): string {
	const firstBlock = blocks[0];
	if (!firstBlock) return "";
	const lines = [
		"",
		`# ⚠ Merge Conflicts: ${blocks.length} block${blocks.length > 1 ? "s" : ""} in ${firstBlock.file}`,
		"",
	];
	for (const block of blocks) {
		const blockLine = fullContent.slice(0, block.offset).split("\n").length;
		lines.push(
			`${block.index}: <<<<<<< ${block.oursLabel} ... ======= ... >>>>>>> ${block.theirsLabel} (around line ${blockLine})`,
		);
	}
	lines.push("");
	lines.push(
		`[Use write path="conflict://${firstBlock.file}:N" content="ours"|"theirs"|"ours+theirs"|"base" to resolve block N, or path="conflict://${firstBlock.file}" to resolve every block]`,
	);
	lines.push("[Use :conflicts selector to view full conflict blocks]");
	return `\n${lines.join("\n")}`;
}
