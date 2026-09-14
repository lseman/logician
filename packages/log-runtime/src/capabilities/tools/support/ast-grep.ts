// ── ast-grep integration for structural AST queries ────────────────────────────
// Wraps the ast-grep CLI to perform AST-aware pattern matching and rewriting.
//
// The ast_edit tool uses this to find structural patterns, compute diffs, and
// produce staged edit proposals.

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Run the ast-grep CLI and parse its JSON output. ast-grep exits with code 1
 * (not 0) whenever a run produces zero matches — same convention as
 * grep/rg — even though stdout still holds a valid `[]`. `execFile`'s
 * promisified form rejects on any non-zero exit, so a plain `await
 * execFileAsync(...)` would surface "no matches" as a thrown error. Recover
 * by parsing `error.stdout` (still populated by Node on a non-zero exit)
 * as JSON; only propagate the error when that doesn't yield a valid result.
 */
async function runAstGrepCli(bin: string, args: string[]): Promise<AstMatch[]> {
	let stdout: string;
	try {
		stdout = (await execFileAsync(bin, args)).stdout;
	} catch (error) {
		const stdoutFromError =
			error && typeof error === "object" && "stdout" in error
				? String((error as { stdout: unknown }).stdout ?? "")
				: "";
		if (!stdoutFromError.trim()) throw error;
		try {
			return JSON.parse(stdoutFromError) as AstMatch[];
		} catch {
			throw error;
		}
	}
	if (!stdout.trim()) return [];
	try {
		return JSON.parse(stdout) as AstMatch[];
	} catch {
		return [];
	}
}

// ── Types ──────────────────────────────────────────────────────────────────────

/** A match found by ast-grep. */
export interface AstMatch {
	/** The matched text. */
	text: string;
	/** Byte offset and line/column range. */
	range: {
		byteOffset: { start: number; end: number };
		start: { line: number; column: number };
		end: { line: number; column: number };
	};
	/** File path. */
	file: string;
	/** The matched lines. */
	lines: string;
	/** Character counts for context. */
	charCount: { leading: number; trailing: number };
	/** Replacement text (only present when using --rewrite). */
	replacement?: string;
	/** Byte offsets for the replacement. */
	replacementOffsets?: { start: number; end: number };
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

// ── CLI detection ──────────────────────────────────────────────────────────────

const AST_GREP_BINARIES = ["sg", "ast-grep"];

/**
 * Find the ast-grep binary.
 */
async function findAstGrepBin(): Promise<string | null> {
	for (const bin of AST_GREP_BINARIES) {
		try {
			await execFileAsync(bin, ["--version"]);
			return bin;
		} catch {}
	}
	return null;
}

let cachedBin: string | null | undefined;

/**
 * Get the ast-grep binary path, cached.
 */
async function getAstGrepBin(): Promise<string | null> {
	if (cachedBin === undefined) {
		cachedBin = await findAstGrepBin();
	}
	return cachedBin;
}

// ── Language detection ─────────────────────────────────────────────────────────

/** Map file extensions to ast-grep language names. */
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

/**
 * Detect the ast-grep language for a file path.
 */
function detectLanguage(filePath: string): string {
	const ext = path.extname(filePath).slice(1).toLowerCase();
	return EXT_TO_LANG[ext] || "typescript";
}

// ── Query execution ────────────────────────────────────────────────────────────

/**
 * Run ast-grep rewrite and return matches with replacements.
 *
 * Does NOT pass `--no-ignore` — current ast-grep CLI versions (0.4x+) require
 * it to take a FILE_TYPE value (`hidden`/`dot`/`exclude`/`global`/`parent`/
 * `vcs`); passing it bare (as this used to) greedily consumes the first file
 * path as that value and errors out, breaking every call. Omitting it just
 * means normal .gitignore-respecting behavior, same as every other tool here.
 */
async function runRewriteQuery(
	bin: string,
	language: string,
	pattern: string,
	rewrite: string,
	filePaths: string[],
): Promise<AstMatch[]> {
	return runAstGrepCli(bin, [
		"run",
		"--lang",
		language,
		"--pattern",
		pattern,
		"--rewrite",
		rewrite,
		"--json=compact",
		...filePaths,
	]);
}

/**
 * Run a pure ast-grep query (no --rewrite) and return matches. Separate from
 * runRewriteQuery because the CLI always computes a replacement when
 * --rewrite is passed — a read-only query has no replacement to offer and
 * shouldn't be forced to invent one.
 */
async function runQuery(
	bin: string,
	language: string,
	pattern: string,
	filePaths: string[],
): Promise<AstMatch[]> {
	return runAstGrepCli(bin, [
		"run",
		"--lang",
		language,
		"--pattern",
		pattern,
		"--json=compact",
		...filePaths,
	]);
}

// ── Diff computation ───────────────────────────────────────────────────────────

/**
 * Compute file edits from ast-grep rewrite matches.
 * Groups matches by file and sorts by byte offset (descending) so replacements
 * can be applied without offset shifts.
 */
export function computeFileEdits(matches: AstMatch[]): Array<{
	file: string;
	edits: AstEditResult[];
}> {
	const fileMap = new Map<string, AstEditResult[]>();

	for (const match of matches) {
		const origStart = match.range.byteOffset.start;
		const origEnd = match.range.byteOffset.end;
		const repl = match.replacement ?? "";
		const file = match.file;

		const edits = fileMap.get(file) ?? [];
		fileMap.set(file, edits);
		edits.push({
			file,
			replacement: repl,
			originalStart: origStart,
			originalEnd: origEnd,
			original: match.text,
		});
	}

	return Array.from(fileMap.entries()).map(([file, edits]) => ({
		file,
		// Sort descending by originalStart so earlier edits aren't shifted
		edits: edits.sort((a, b) => b.originalStart - a.originalStart),
	}));
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
	const bin = await getAstGrepBin();
	if (!bin) {
		throw new Error(
			"ast-grep CLI not found. Install with: npm install -g @ast-grep/cli",
		);
	}

	// Collect all matches across all operations
	const allMatches: AstMatch[] = [];

	for (const op of ops) {
		// Detect language from first path
		const lang = detectLanguage(paths[0] ?? "");

		const matches = await runRewriteQuery(bin, lang, op.pat, op.out, paths);
		allMatches.push(...matches);
	}

	return computeFileEdits(allMatches);
}

/**
 * Execute a single AST operation and return matches for preview.
 */
export async function previewAstOp(
	pat: string,
	out: string,
	paths: string[],
): Promise<AstMatch[]> {
	const bin = await getAstGrepBin();
	if (!bin) {
		throw new Error(
			"ast-grep CLI not found. Install with: npm install -g @ast-grep/cli",
		);
	}

	const lang = detectLanguage(paths[0] ?? "");
	return runRewriteQuery(bin, lang, pat, out, paths);
}

/**
 * Run a read-only structural query (no rewrite) across the given paths.
 * Language is detected from the first path unless `language` overrides it.
 */
export async function queryAstPattern(
	pattern: string,
	paths: string[],
	language?: string,
): Promise<AstMatch[]> {
	const bin = await getAstGrepBin();
	if (!bin) {
		throw new Error(
			"ast-grep CLI not found. Install with: npm install -g @ast-grep/cli",
		);
	}

	const lang = language || detectLanguage(paths[0] ?? "");
	return runQuery(bin, lang, pattern, paths);
}
