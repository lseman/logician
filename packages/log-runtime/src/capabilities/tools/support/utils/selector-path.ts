/**
 * Detects `container.ext:selector` paths for archives and SQLite databases,
 * embedded directly in the `path` argument `read`/`write` already take.
 * Longest-extension-first matching with a `:`-or-end-of-string lookahead
 * means a literal file named e.g. `notes.tar.gz.txt` never matches — the
 * lookahead fails mid-filename, so callers fall through to today's
 * unchanged plain-file handling whenever nothing here matches.
 */

import * as fs from "node:fs";
import { resolvePath } from "./path-utils.ts";

const ARCHIVE_EXTENSIONS = [
	".tar.gz",
	".tgz",
	".tar",
	".zip",
	".jar",
	".war",
	".ear",
	".apk",
] as const;

const SQLITE_EXTENSIONS = [".sqlite3", ".sqlite", ".db3", ".db"] as const;

export interface ContainerSelectorMatch {
	/** Input up to and including the extension, as written by the caller. */
	containerPath: string;
	/** Text after the separating ':', "" for a bare container path. */
	selector: string;
	/** Resolved absolute path of the container. */
	absolutePath: string;
}

function buildPattern(extensions: readonly string[]): RegExp {
	const alternation = extensions
		.map(ext => ext.replace(/\./g, "\\."))
		.join("|");
	return new RegExp(`(?:${alternation})(?=:|$)`, "gi");
}

const ARCHIVE_PATTERN = buildPattern(ARCHIVE_EXTENSIONS);
const SQLITE_PATTERN = buildPattern(SQLITE_EXTENSIONS);

function candidatesFor(
	input: string,
	pattern: RegExp,
): ContainerSelectorMatch[] {
	pattern.lastIndex = 0;
	const candidates: Array<{ containerPath: string; selector: string }> = [];
	let match: RegExpExecArray | null;
	while ((match = pattern.exec(input))) {
		const end = match.index + match[0].length;
		candidates.push({
			containerPath: input.slice(0, end),
			selector: input.slice(end).replace(/^:/, ""),
		});
	}
	return candidates
		.sort((a, b) => b.containerPath.length - a.containerPath.length)
		.map(c => ({ ...c, absolutePath: "" }));
}

/**
 * Detect a `container.ext:selector` match against `pattern`. When
 * `requireExisting` is true (the default, used by read call sites), the
 * container must already exist as a regular file. Write call sites pass
 * `false` so a not-yet-created archive/database can still be targeted.
 */
function detect(
	input: string,
	cwd: string,
	pattern: RegExp,
	requireExisting: boolean,
): ContainerSelectorMatch | undefined {
	for (const candidate of candidatesFor(input, pattern)) {
		const absolutePath = resolvePath(cwd, candidate.containerPath);
		if (requireExisting) {
			let stat: fs.Stats;
			try {
				stat = fs.statSync(absolutePath);
			} catch {
				continue;
			}
			if (!stat.isFile()) continue;
		} else if (
			fs.existsSync(absolutePath) &&
			!fs.statSync(absolutePath).isFile()
		) {
			continue;
		}
		return { ...candidate, absolutePath };
	}
	return undefined;
}

export function detectArchiveSelector(
	input: string,
	cwd: string,
	options: { requireExisting?: boolean } = {},
): ContainerSelectorMatch | undefined {
	return detect(input, cwd, ARCHIVE_PATTERN, options.requireExisting ?? true);
}

export function detectSqliteSelector(
	input: string,
	cwd: string,
	options: { requireExisting?: boolean } = {},
): ContainerSelectorMatch | undefined {
	return detect(input, cwd, SQLITE_PATTERN, options.requireExisting ?? true);
}
