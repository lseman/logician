// ── write tool ───────────────────────────────────────────────────────────────
// Create or overwrite a complete file. Creates parent directories. Overwriting an
// existing file requires it to have been read first (and not modified since), so the
// model can never blind-clobber content. Returns the new content with syntax
// highlighting and line numbers (truncated for large files).
//
// With append: true, appends content to the end of the file instead of overwriting
// it. The file must have been read first if it already exists (same safety check).
// Appending is useful for streaming large files across multiple tool calls.

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool, ToolResult } from "@logician/log-core";
import { extractInternalUrlScheme } from "../../runtime/bridge/support/internal-urls/parse.ts";
import type { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";
import { archiveFamilyFromPath, writeArchiveMember } from "./support/archive-resource.ts";
import { createMutationSession } from "./mutation/session.js";
import { createEditStore } from "./support/edit-store.js";
import { withFileMutationQueue } from "./support/mutation-queue.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "./support/read-tracker.ts";
import {
	deleteSqliteRow,
	insertSqliteRow,
	updateSqliteRow,
} from "./support/sqlite-resource.ts";
import { appendToFile } from "./support/utils/atomic-write.ts";
import { ensureInsideCwd, resolvePath } from "./support/utils/path-utils.ts";
import {
	detectArchiveSelector,
	detectSqliteSelector,
} from "./support/utils/selector-path.ts";
import { highlightAuto } from "./support/utils/syntax-highlighter.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	truncateHead,
} from "./support/utils/truncate.ts";
import type { XdDeviceRegistry } from "./support/xd-device-registry.ts";

function buildWriteTool(router?: InternalUrlRouter): Tool {
	return {
	name: "write",
	executionMode: "parallel",
	label: "Write",
	hookAliases: ["Write"],
	description:
		`Write a file or invoke an xd:// tool device. For a device, content is a JSON object encoded as a string; read xd://<name> for its schema. ` +
		`Create or overwrite a complete file. Creates parent directories. ` +
		`Overwriting an existing file requires reading it with read first. ` +
		`Output is truncated to ${DEFAULT_MAX_LINES} lines or ` +
		`${formatSize(DEFAULT_MAX_BYTES)} (whichever is hit first). ` +
		`With append: true, appends content to the end of the file instead of ` +
		`overwriting. Useful for streaming large files across multiple tool calls. ` +
		`Supports .zip-family (.zip, .jar, .war, .ear, .apk) and .tar-family (.tar, .tar.gz, .tgz) ` +
		`archive entries via archive.ext:path/inside/archive (whole-archive rewrite under the hood; ` +
		`other archive formats are not supported at all). Requires reading the archive first when it ` +
		`already exists, same as any other overwrite; a new archive is created on first write. ` +
		`Supports SQLite row writes via db.sqlite:table (insert, auto-creating the table from the JSON ` +
		`body's keys if needed), db.sqlite:table:rowid (update with a JSON object body, delete with empty ` +
		`content). SQLite writes do not require reading the database first.`,
	promptSnippet:
		"Create or overwrite files; automatically create parent directories; use append: true to append",
	promptGuidelines: [
		"Use write for new files or complete rewrites",
		"Use append: true to add content to the end of an existing file without overwriting it",
		"Use write with archive.ext:path/inside/archive or db.sqlite:table[:rowid] to write archive members or SQLite rows; archive writes require reading the container first, SQLite writes do not",
	],
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path or xd://<device> URL" },
			content: {
				type: "string",
				description:
					"Complete file contents, or a JSON object encoded as a string for a device",
			},
			append: {
				type: "boolean",
				description:
					"If true, append content to the end of the file instead of overwriting. Defaults to false.",
			},
		},
		required: ["path", "content"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			path: args.path ?? args.file_path ?? args.filename,
			content: args.content ?? args.text,
		};
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const filePath = String(args.path);
		const content = String(args.content ?? "");
		const append = Boolean(args.append);
		const scheme = extractInternalUrlScheme(filePath);
		if (scheme === "xd") {
			return {
				content:
					"Error: Device calls must be dispatched through ToolRegistry so target permissions and hooks run.",
				isError: true,
			};
		}

		if (scheme) {
			if (!router?.canResolve(filePath)) {
				return `Error: write does not support ${scheme}:// links. Use ./ before the path if a literal file is intended.`;
			}
			if (append) {
				return `Error: append is not supported for ${scheme}:// links.`;
			}
			try {
				await router.write(filePath, content, ctx);
				return `Wrote ${filePath} (${Buffer.byteLength(content, "utf-8")} bytes)`;
			} catch (error) {
				return `Error: ${error instanceof Error ? error.message : String(error)}`;
			}
		}

		if (!scheme) {
			const cwd = ctx.cwd ?? process.cwd();

			const archiveMatch = detectArchiveSelector(filePath, cwd, { requireExisting: false });
			if (archiveMatch && archiveMatch.selector) {
				if (append) return "Error: append is not supported for archive members.";
				return withFileMutationQueue(archiveMatch.absolutePath, async () => {
					const containerExists = fs.existsSync(archiveMatch.absolutePath);
					if (containerExists) {
						if (!hasBeenRead(archiveMatch.absolutePath)) {
							return (
								`${archiveMatch.absolutePath} already exists but has not been read. ` +
								"Read it with read before overwriting."
							);
						}
						if (isStaleSinceRead(archiveMatch.absolutePath)) {
							return (
								`${archiveMatch.absolutePath} has been modified since it was last read. ` +
								"Read it again before overwriting."
							);
						}
					}
					ensureInsideCwd(ctx.cwd, archiveMatch.absolutePath, ctx.allowedPaths, ctx.allowAllPaths);
					const family = archiveFamilyFromPath(archiveMatch.absolutePath);
					if (!family) return `Error: unsupported archive format for ${archiveMatch.absolutePath}.`;
					try {
						const { created } = await writeArchiveMember(
							archiveMatch.absolutePath,
							family,
							archiveMatch.selector,
							content,
						);
						refreshAfterWrite(archiveMatch.absolutePath);
						return `${created ? "Created" : "Updated"} ${filePath} (${Buffer.byteLength(content, "utf-8")} bytes)`;
					} catch (error) {
						return `Error: ${error instanceof Error ? error.message : String(error)}`;
					}
				});
			}

			const sqliteMatch = detectSqliteSelector(filePath, cwd, { requireExisting: false });
			if (sqliteMatch && sqliteMatch.selector) {
				if (append) return "Error: append is not supported for SQLite rows.";
				return withFileMutationQueue(sqliteMatch.absolutePath, async () => {
					ensureInsideCwd(ctx.cwd, sqliteMatch.absolutePath, ctx.allowedPaths, ctx.allowAllPaths);
					const [table, rowid] = sqliteMatch.selector.split(":");
					if (!table) return "Error: SQLite writes require a table name, e.g. db.sqlite:table.";
					const trimmed = content.trim();
					try {
						let resultText: string;
						if (rowid === undefined) {
							resultText = insertSqliteRow(sqliteMatch.absolutePath, table, content);
						} else if (trimmed.length === 0) {
							resultText = deleteSqliteRow(sqliteMatch.absolutePath, table, rowid);
						} else {
							resultText = updateSqliteRow(sqliteMatch.absolutePath, table, rowid, content);
						}
						refreshAfterWrite(sqliteMatch.absolutePath);
						return resultText;
					} catch (error) {
						return `Error: ${error instanceof Error ? error.message : String(error)}`;
					}
				});
			}
		}

		const resolved = resolvePath(ctx.cwd, filePath);
		ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);

		const store = createEditStore();
		const mutation = createMutationSession(store, ctx.cwd || process.cwd(), {
			allowedPaths: ctx.allowedPaths,
			allowAllPaths: ctx.allowAllPaths,
		});

		return withFileMutationQueue(resolved, async () => {
			const fileExists = fs.existsSync(resolved);

			if (fileExists && !append) {
				// Overwrite mode: must have been read first.
				if (!hasBeenRead(resolved)) {
					return (
						`${resolved} already exists but has not been read. ` +
						"Read it with read before overwriting, or use edit for targeted changes."
					);
				}
				if (isStaleSinceRead(resolved)) {
					return (
						`${resolved} has been modified since it was last read. ` +
						"Read it again before overwriting."
					);
				}
			}

			// Append mode: also require read if file exists, but the check is
			// about whether the model last saw this file's contents — same
			// stale-after-write protection, no content comparison needed.
			if (fileExists && append) {
				if (!hasBeenRead(resolved)) {
					return (
						`${resolved} already exists but has not been read. ` +
						"Read it with read before appending, or use write for a complete overwrite."
					);
				}
				if (isStaleSinceRead(resolved)) {
					return (
						`${resolved} has been modified since it was last read. ` +
						"Read it again before appending."
					);
				}
			}

			fs.mkdirSync(path.dirname(resolved), { recursive: true });

			const chunkBytes = Buffer.byteLength(content, "utf-8");
			let result: string;

			if (append && fileExists) {
				// Append mode with existing file: size-based concurrency guard.
				const fileStat = fs.statSync(resolved);
				await appendToFile(resolved, content, {
					expectedSizeBefore: fileStat.size,
				});
				refreshAfterWrite(resolved);
				result =
					`Appended to ${resolved} (+${chunkBytes} bytes, file size now ${formatSize(chunkBytes + fileStat.size - Buffer.byteLength(content, "utf-8"))}). ` +
					"Call write with the next chunk, or stop if this was the last one.";
			} else if (append && !fileExists) {
				// Append mode with new file: just create and write.
				await appendToFile(resolved, content);
				refreshAfterWrite(resolved);
				result =
					`Created ${resolved} (${chunkBytes} bytes) with append mode. ` +
					"Call write with the next chunk, or stop if this was the last one.";
			} else {
				// Overwrite mode (normal) — use mutation session for stale detection.
				const beforeContent = fileExists
					? fs.readFileSync(resolved, "utf-8")
					: "";
				const proposal = {
					path: resolved,
					before: beforeContent,
					beforeHash: createHash("sha256")
						.update(beforeContent, "utf-8")
						.digest("hex"),
					after: content,
				};
				const result2 = await mutation.apply(proposal);
				refreshAfterWrite(resolved);

				if (result2.error) return `Error: ${result2.error}`;
				if (!result2.applied)
					return "No changes made: the file content is unchanged.";

				const lineCount = content === "" ? 0 : content.split("\n").length;
				const byteLen = Buffer.byteLength(content, "utf-8");
				if (!fileExists) {
					return `Created ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
				}

				const highlighted = highlightAuto(content);
				const t = truncateHead(highlighted.value);

				if (t.firstLineExceedsLimit) {
					const firstLineBytes = Buffer.byteLength(
						highlighted.value.split("\n")[0] ?? "",
						"utf-8",
					);
					return (
						`[First line is ${formatSize(firstLineBytes)}, exceeds ${formatSize(DEFAULT_MAX_BYTES)} limit. ` +
						`Use bash: head -c ${DEFAULT_MAX_BYTES} ${filePath}]`
					);
				}
				if (t.truncated) {
					const endDisplay = t.outputLines;
					const nextOffset = endDisplay + 1;
					return (
						`${t.content}\n\n` +
						`[Wrote ${resolved} (${lineCount} lines, ${formatSize(byteLen)}). ` +
						`Showing lines 1-${endDisplay}. ` +
						`Use offset=${nextOffset} to continue.]`
					);
				}

				const gutterWidth = String(lineCount).length + 1;
				const header = `Wrote ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
				const out: string[] = [header];
				const hlLines = t.content.split("\n");
				for (let i = 0; i < hlLines.length; i++) {
					const num = String(i + 1).padStart(gutterWidth, " ");
					out.push(`${num}|${hlLines[i]}`);
				}
				return out.join("\n");
			}

			return result;
		});
	},
	};
}

export function createWriteTool(
	devices?: XdDeviceRegistry,
	router?: InternalUrlRouter,
): Tool {
	return {
		...buildWriteTool(router),
		resolveCall: args => {
			if (
				typeof args.path !== "string" ||
				extractInternalUrlScheme(args.path) !== "xd"
			)
				return undefined;
			if (args.append)
				throw new Error("append is not supported for xd:// devices.");
			if (!devices)
				throw new Error("No xd:// devices are mounted in this session.");
			return devices.resolveCall(args.path.slice("xd://".length), args.content);
		},
	};
}

export const write = createWriteTool();
