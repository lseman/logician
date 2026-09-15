// ── ast-grep integration for structural AST queries ────────────────────────────
// Wraps @logician/log-natives (the pi-ast/ast-grep-core N-API binding) to
// perform AST-aware pattern matching and rewriting — no external CLI needed.
//
// The ast_edit tool uses this to find structural patterns, compute diffs, and
// produce staged edit proposals.

import { readFileSync, statSync } from "node:fs";
import * as path from "node:path";

// ── Native addon loading ─────────────────────────────────────────────────────

interface NativeAstFindMatch {
	path: string;
	text: string;
	byteStart: number;
	byteEnd: number;
	startLine: number;
	startColumn: number;
	endLine: number;
	endColumn: number;
}

interface NativeAstFindResult {
	matches: NativeAstFindMatch[];
	totalMatches: number;
}

interface NativeAstReplaceChange {
	path: string;
	before: string;
	after: string;
	byteStart: number;
	byteEnd: number;
}

interface NativeAstReplaceResult {
	changes: NativeAstReplaceChange[];
}

interface LogNatives {
	astGrep(options: {
		patterns: string[];
		lang?: string;
		path: string;
	}): Promise<NativeAstFindResult>;
	astEdit(options: {
		rewrites: Record<string, string>;
		lang?: string;
		path: string;
		dryRun: boolean;
	}): Promise<NativeAstReplaceResult>;
}

let nativePromise: Promise<LogNatives> | undefined;

/** Load the native addon lazily so a missing/unbuilt build only breaks callers who need it. */
function loadNative(): Promise<LogNatives> {
	if (!nativePromise) {
		nativePromise = import("@logician/log-natives").catch(error => {
			nativePromise = undefined;
			throw new Error(
				`@logician/log-natives addon not available (run \`bun run build\` in packages/log-natives): ${
					error instanceof Error ? error.message : String(error)
				}`,
			);
		}) as Promise<LogNatives>;
	}
	return nativePromise;
}

// ── Types ──────────────────────────────────────────────────────────────────────

/** A match found by ast-grep. */
export interface AstMatch {
	/** The matched text. */
	text: string;
	/** Byte offset and 0-based line/column range. */
	range: {
		byteOffset: { start: number; end: number };
		start: { line: number; column: number };
		end: { line: number; column: number };
	};
	/** File path. */
	file: string;
}

/** An AST edit operation. */
export interface AstOp {
	/** AST-grep pattern (uses metavariables like $A, $$$ARGS). */
	pat: string;
	/** Replacement template (uses same metavariables). */
	out: string;
}

/** Result of running an AST edit operation. */
export interface AstEditResult {
	/** The file being edited. */
	file: string;
	/** The replacement text. */
	replacement: string;
	/** Byte offset of the original text. */
	originalStart: number;
	/** Byte offset after the original text. */
	originalEnd: number;
	/** The original text. */
	original: string;
}

// ── Language detection ─────────────────────────────────────────────────────────
// Only used as an explicit override; when omitted, the native binding infers
// language per file from its extension, including for mixed-language paths.

const EXT_TO_LANG: Record<string, string> = {
	ts: "typescript",
	tsx: "tsx",
	js: "javascript",
	jsx: "jsx",
	rs: "rust",
	py: "python",
	go: "go",
	java: "java",
	c: "c",
	cpp: "cpp",
	cs: "csharp",
	rb: "ruby",
	php: "php",
	swift: "swift",
	kt: "kotlin",
};

/** Detect the ast-grep language for a file path, for an explicit `language` override. */
function detectLanguage(filePath: string): string | undefined {
	const ext = path.extname(filePath).slice(1).toLowerCase();
	return EXT_TO_LANG[ext];
}

/**
 * Resolve a native match/change's `path` back to an absolute filesystem path.
 * The native side reports a "display path": just the basename when the scan
 * root was a single file, or the path relative to the root when it was a
 * directory — never the absolute path we originally passed in.
 */
function resolveDisplayPath(scanRoot: string, displayPath: string): string {
	if (path.isAbsolute(displayPath)) return displayPath;
	return statSync(scanRoot).isFile()
		? scanRoot
		: path.resolve(scanRoot, displayPath);
}

// ── Diff application ───────────────────────────────────────────────────────────

/**
 * Apply a file's edits (byte-offset spans) to its on-disk content and return
 * the resulting full text. Edits are applied in descending offset order so
 * earlier spans aren't shifted by later replacements; splicing happens on the
 * raw UTF-8 bytes since `originalStart`/`originalEnd` are byte offsets, not
 * UTF-16 code-unit indices.
 */
function applyByteEdits(filePath: string, edits: AstEditResult[]): string {
	let buf = readFileSync(filePath);
	const sorted = [...edits].sort((a, b) => b.originalStart - a.originalStart);
	for (const edit of sorted) {
		buf = Buffer.concat([
			buf.subarray(0, edit.originalStart),
			Buffer.from(edit.replacement, "utf8"),
			buf.subarray(edit.originalEnd),
		]);
	}
	return buf.toString("utf8");
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Execute an AST edit operation across the given paths.
 * Returns the file edits that would result from applying the operation.
 */
export async function executeAstOp(
	ops: AstOp[],
	paths: string[],
): Promise<Array<{ file: string; edits: AstEditResult[] }>> {
	const native = await loadNative();
	const rewrites = Object.fromEntries(ops.map(op => [op.pat, op.out]));

	const fileMap = new Map<string, AstEditResult[]>();
	for (const scanPath of paths) {
		const lang = detectLanguage(scanPath);
		const result = await native.astEdit({
			rewrites,
			...(lang !== undefined ? { lang } : {}),
			path: scanPath,
			dryRun: true,
		});
		for (const change of result.changes) {
			const file = resolveDisplayPath(scanPath, change.path);
			const edits = fileMap.get(file) ?? [];
			fileMap.set(file, edits);
			edits.push({
				file,
				replacement: change.after,
				originalStart: change.byteStart,
				originalEnd: change.byteEnd,
				original: change.before,
			});
		}
	}

	return Array.from(fileMap.entries()).map(([file, edits]) => ({
		file,
		edits,
	}));
}

/**
 * Compute the full post-edit content for a file's staged edits, applying
 * byte-offset splices against the file's current on-disk content.
 */
export function applyFileEdits(file: string, edits: AstEditResult[]): string {
	return applyByteEdits(file, edits);
}

/**
 * Run a read-only structural query (no rewrite) across the given paths.
 * Language is detected from each match's own file unless `language` overrides it.
 */
export async function queryAstPattern(
	pattern: string,
	paths: string[],
	language?: string,
): Promise<AstMatch[]> {
	const native = await loadNative();

	const matches: AstMatch[] = [];
	for (const scanPath of paths) {
		const result = await native.astGrep({
			patterns: [pattern],
			...(language !== undefined ? { lang: language } : {}),
			path: scanPath,
		});
		for (const m of result.matches) {
			matches.push({
				text: m.text,
				file: resolveDisplayPath(scanPath, m.path),
				range: {
					byteOffset: { start: m.byteStart, end: m.byteEnd },
					// Native offsets are 1-based; keep this module's contract 0-based,
					// matching how callers already render it (`line + 1`).
					start: { line: m.startLine - 1, column: m.startColumn - 1 },
					end: { line: m.endLine - 1, column: m.endColumn - 1 },
				},
			});
		}
	}
	return matches;
}
