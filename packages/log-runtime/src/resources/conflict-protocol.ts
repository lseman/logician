// ── conflict:// protocol handler ─────────────────────────────────────────────
// Resolves and applies resolutions to merge conflict blocks in a file.
// URL forms:
//   conflict://<file>          — list conflict blocks in <file>
//   conflict://<file>:<index>  — read conflict block <index> from <file>
//
// write(path="conflict://<file>", content="ours"|"theirs"|"ours+theirs"|"base")
//   resolves every block in <file> with that strategy.
// write(path="conflict://<file>:<index>", content=<same strategies>)
//   resolves only block <index>, leaving the rest of the file's conflict
//   markers untouched.
//
// "base" currently behaves like "ours" in practice: parseConflictBlocks only
// recognizes two-way markers (<<<<<<< / ======= / >>>>>>>), not diff3's
// ||||||| base section, so no real base content is ever available.

import { atomicWriteFile } from "../shared/atomic-write.ts";
import {
	getConflictBlocks,
	resolveConflictsInFile,
} from "../shared/conflict-resolution.ts";
import { ensureInsideCwd, resolvePath } from "../shared/path-utils.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "../shared/read-tracker.ts";
import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	WriteContext,
} from "./types";

type ConflictStrategy = "ours" | "theirs" | "ours+theirs" | "base";
const STRATEGIES: readonly ConflictStrategy[] = [
	"ours",
	"theirs",
	"ours+theirs",
	"base",
];

interface FileAndIndex {
	file: string;
	index: number | null;
}

/**
 * Parse `<file>` or `<file>:<index>` from the URL. The shared parser only
 * puts the first path segment into `url.host`; the rest lands in
 * `url.pathname` (and never stops at `:` either) — reassemble the full
 * address before splitting, the same pattern log-protocol.ts uses for
 * `/`-separated paths and artifact-protocol.ts uses for its `:` selector.
 */
function parseFileAndIndex(url: InternalUrl): FileAndIndex {
	const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;
	if (!full) {
		throw new Error("conflict:// URL requires a file path: conflict://<file>");
	}
	const colonIdx = full.lastIndexOf(":");
	if (colonIdx >= 0) {
		const suffix = full.slice(colonIdx + 1);
		if (/^\d+$/.test(suffix)) {
			return { file: full.slice(0, colonIdx), index: Number(suffix) };
		}
	}
	return { file: full, index: null };
}

export class ConflictProtocolHandler implements ProtocolHandler {
	readonly scheme = "conflict";
	readonly immutable = false;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const cwd = context?.cwd ?? process.cwd();
		const { file, index } = parseFileAndIndex(url);
		const resolved = resolvePath(cwd, file);
		ensureInsideCwd(
			cwd,
			resolved,
			context?.allowedPaths,
			context?.allowAllPaths,
		);

		const blocks = getConflictBlocks(resolved);
		if (blocks.length === 0) {
			return {
				url: url.href,
				content: `# No conflicts in ${file}\n\nThis file has no unresolved merge conflicts.`,
				contentType: "text/markdown",
			};
		}

		// conflict://<file> — list all conflicts
		if (index === null) {
			const lines = blocks.map(
				b =>
					`- Block ${b.index}: ${b.oursLabel} ... ${b.theirsLabel} at offset ${b.offset}`,
			);
			return {
				url: url.href,
				content: `# Merge Conflicts in ${file}\n\n${lines.join("\n")}`,
				contentType: "text/markdown",
			};
		}

		// conflict://<file>:<index> — read a specific block
		const block = blocks.find(b => b.index === index);
		if (!block) {
			throw new Error(`No conflict block at index ${index} in ${file}`);
		}

		const content = [
			`# Conflict Block ${index} in ${file}`,
			``,
			`**Conflict:** ${block.oursLabel} ... ${block.theirsLabel}`,
			`**Offset:** ${block.offset}`,
			``,
			`--- ours ---`,
			``,
			block.ours,
			``,
			`--- theirs ---`,
			``,
			block.theirs,
		].join("\n");

		return {
			url: url.href,
			content,
			contentType: "text/plain",
			sourcePath: resolved,
		};
	}

	async write(
		url: InternalUrl,
		content: string,
		context?: WriteContext,
	): Promise<void> {
		const cwd = context?.cwd ?? process.cwd();
		const { file, index } = parseFileAndIndex(url);
		const resolved = resolvePath(cwd, file);
		ensureInsideCwd(
			cwd,
			resolved,
			context?.allowedPaths,
			context?.allowAllPaths,
		);

		if (!hasBeenRead(resolved)) {
			throw new Error(
				`${resolved} has not been read. Read it with read before resolving conflicts.`,
			);
		}
		if (isStaleSinceRead(resolved)) {
			throw new Error(
				`${resolved} has been modified since it was last read. Read it again before resolving conflicts.`,
			);
		}

		const strategy = content.trim();
		if (!(STRATEGIES as readonly string[]).includes(strategy)) {
			throw new Error(
				`conflict:// write requires content to be one of ${STRATEGIES.map(s => `"${s}"`).join(", ")}; got: ${JSON.stringify(content)}`,
			);
		}

		const resolvedContent = resolveConflictsInFile(
			resolved,
			strategy as ConflictStrategy,
			undefined,
			index ?? undefined,
		);
		await atomicWriteFile(resolved, resolvedContent);
		refreshAfterWrite(resolved);
	}
}
