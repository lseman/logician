import * as fs from "node:fs";
import type { ToolContext } from "@logician/log-core";
import { extractInternalUrlScheme } from "../../resources/parse.ts";
import type { InternalUrlRouter } from "../../resources/router.ts";
import type { InternalResource } from "../../resources/types.ts";
import {
	formatConflictBlocks,
	parseConflictBlocks,
} from "../../shared/conflict-resolution.ts";
import { getNativeEditStore } from "../../shared/native-addon.ts";
import { ensureInsideCwd, resolveReadPath } from "../../shared/path-utils.ts";
import {
	type ContainerSelectorMatch,
	detectArchiveSelector,
	detectSqliteSelector,
} from "../../shared/selector-path.ts";
import { formatSize } from "../../shared/truncate.ts";
import {
	archiveFamilyFromPath,
	listArchiveEntries,
	readArchiveMember,
} from "./archive-resource.ts";
import {
	isSqliteFile,
	listSqliteTables,
	readSqliteRow,
	readSqliteTable,
} from "./sqlite-resource.ts";

/**
 * Only a direct filesystem read grants file edit anchors and read tracking.
 * `"container"` reads (archive members, SQLite tables/rows) are tracked the
 * same way `"file"` reads are — `hasBeenRead`/`isStaleSinceRead` key on
 * `.path`, which is the container's absolute path — but get no hashline
 * header, since there's no meaningful edit anchor into archive bytes or a
 * SQL row (write, not edit, is the only mutation path for either).
 */
export type ReadResource =
	| { kind: "file"; resource: InternalResource; path: string; hash: string }
	| { kind: "container"; resource: InternalResource; path: string }
	| { kind: "resource"; resource: InternalResource };

async function readArchiveResource(
	input: string,
	match: ContainerSelectorMatch,
	ctx: ToolContext,
): Promise<ReadResource> {
	ensureInsideCwd(
		ctx.cwd,
		match.absolutePath,
		ctx.allowedPaths,
		ctx.allowAllPaths,
	);
	const family = archiveFamilyFromPath(match.absolutePath);
	if (!family)
		throw new Error(`Unsupported archive format: ${match.absolutePath}`);

	if (!match.selector) {
		const content = await listArchiveEntries(match.absolutePath, family);
		return {
			kind: "container",
			path: match.absolutePath,
			resource: {
				url: input,
				content,
				sourcePath: match.absolutePath,
				isDirectory: true,
			},
		};
	}

	const { content: buffer, size } = await readArchiveMember(
		match.absolutePath,
		family,
		match.selector,
	);
	if (buffer.subarray(0, 8192).includes(0)) {
		throw new Error(
			`${match.selector} appears to be a binary archive member (${formatSize(size)}).`,
		);
	}
	return {
		kind: "container",
		path: match.absolutePath,
		resource: {
			url: input,
			content: buffer.toString("utf-8"),
			size,
			sourcePath: match.absolutePath,
		},
	};
}

function readSqliteResource(
	input: string,
	match: ContainerSelectorMatch,
	ctx: ToolContext,
): ReadResource {
	ensureInsideCwd(
		ctx.cwd,
		match.absolutePath,
		ctx.allowedPaths,
		ctx.allowAllPaths,
	);
	if (!isSqliteFile(match.absolutePath)) {
		throw new Error(
			`${match.absolutePath} does not look like a SQLite database.`,
		);
	}

	let content: string;
	if (!match.selector) {
		content = listSqliteTables(match.absolutePath);
	} else {
		const [table, rowid] = match.selector.split(":");
		if (!table) throw new Error(`Invalid SQLite selector: ${match.selector}`);
		if (rowid === undefined) {
			content = readSqliteTable(match.absolutePath, table);
		} else {
			const row = readSqliteRow(match.absolutePath, table, rowid);
			if (row === undefined) throw new Error(`No such row: ${table}:${rowid}`);
			content = row;
		}
	}
	return {
		kind: "container",
		path: match.absolutePath,
		resource: { url: input, content, sourcePath: match.absolutePath },
	};
}

/** Resolve the input before presentation. New URL schemes only need a handler. */
export async function readResource(
	input: string,
	ctx: ToolContext,
	router: InternalUrlRouter,
): Promise<ReadResource> {
	ctx.signal?.throwIfAborted();
	if (extractInternalUrlScheme(input)) {
		return { kind: "resource", resource: await router.resolve(input, ctx) };
	}
	const cwd = ctx.cwd ?? process.cwd();
	const archiveMatch = detectArchiveSelector(input, cwd);
	if (archiveMatch) return readArchiveResource(input, archiveMatch, ctx);
	const sqliteMatch = detectSqliteSelector(input, cwd);
	if (sqliteMatch) return readSqliteResource(input, sqliteMatch, ctx);

	// `path:conflicts` — the read tool's own conflict-notice hint advertises
	// this selector; only treat the suffix as a selector when the truncated
	// path actually exists as a file, so a literal filename ending in
	// ":conflicts" (however unlikely) still reads as itself.
	const CONFLICTS_SUFFIX = ":conflicts";
	let readInput = input;
	let conflictsView = false;
	if (input.endsWith(CONFLICTS_SUFFIX)) {
		const candidate = input.slice(0, -CONFLICTS_SUFFIX.length);
		const candidateResolved = resolveReadPath(candidate, cwd);
		if (
			fs.existsSync(candidateResolved) &&
			fs.statSync(candidateResolved).isFile()
		) {
			readInput = candidate;
			conflictsView = true;
		}
	}

	const resolved = resolveReadPath(readInput, ctx.cwd ?? process.cwd());
	ensureInsideCwd(ctx.cwd, resolved, ctx.allowedPaths, ctx.allowAllPaths);
	if (!fs.existsSync(resolved)) throw new Error(`File not found: ${resolved}`);
	const stat = fs.statSync(resolved);
	if (stat.isDirectory()) {
		const entries = fs
			.readdirSync(resolved, { withFileTypes: true })
			.sort((a, b) => a.name.localeCompare(b.name))
			.map(entry => `${entry.name}${entry.isDirectory() ? "/" : ""}`);
		return {
			kind: "resource",
			resource: {
				url: input,
				content: entries.join("\n"),
				isDirectory: true,
			},
		};
	}
	if (!stat.isFile())
		throw new Error(`Path is not a regular file: ${resolved}`);
	const buffer = fs.readFileSync(resolved);
	if (buffer.subarray(0, 8192).includes(0)) {
		throw new Error(
			`${resolved} appears to be a binary file (${formatSize(stat.size)}). Use bash tools (file, xxd, strings) to inspect it.`,
		);
	}
	ctx.signal?.throwIfAborted();
	const rawContent = buffer.toString("utf-8");
	const content = conflictsView
		? formatConflictBlocks(parseConflictBlocks(rawContent, resolved))
		: rawContent;
	// Record this read in the same persistent native EditStore the hashline
	// edit engine validates `[path#tag]` headers against (it looks tags up
	// by recorded snapshot, not by recomputing a hash from disk) — the tag
	// shown here must be the one that store just assigned, not a separately
	// computed hash that happens to look similar.
	const store = await getNativeEditStore();
	const hash = store.recordSnapshot(resolved, rawContent);
	return {
		kind: "file",
		path: resolved,
		hash,
		resource: {
			url: input,
			content,
			size: buffer.length,
			sourcePath: resolved,
		},
	};
}
