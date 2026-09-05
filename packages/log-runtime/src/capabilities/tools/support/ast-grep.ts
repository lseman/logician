// ── ast-grep integration for structural AST queries ────────────────────────────
// Wraps the ast-grep CLI to perform AST-aware pattern matching and rewriting.
//
// The ast_edit tool uses this to find structural patterns, compute diffs, and
// produce staged edit proposals.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as path from "node:path";

const execFileAsync = promisify(execFile);

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
		} catch {
			continue;
		}
	}
	return null;
}

let cachedBin: string | null | undefined = undefined;

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
 */
async function runRewriteQuery(
	bin: string,
	language: string,
	pattern: string,
	rewrite: string,
	filePaths: string[],
): Promise<AstMatch[]> {
	const result = await execFileAsync(bin, [
		"run",
		"--lang",
		language,
		"--pattern",
		pattern,
		"--rewrite",
		rewrite,
		"--json=compact",
		"--no-ignore",
		...filePaths,
	]);

	if (!result.stdout.trim()) {
		return [];
	}

	try {
		const matches = JSON.parse(result.stdout) as AstMatch[];
		return matches;
	} catch {
		return [];
	}
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

		if (!fileMap.has(file)) {
			fileMap.set(file, []);
		}

		fileMap.get(file)!.push({
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
