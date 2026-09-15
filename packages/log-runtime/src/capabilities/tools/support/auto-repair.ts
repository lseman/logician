// ── Auto-repair: parse failure detection ───────────────────────────────────────
// After an edit, run the file through tsx to catch syntax errors it introduced.

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Extensions tsx can actually load. Anything else (plain text, markdown,
 * JSON, config files, ...) isn't a module Node's ESM loader can format —
 * running it through `import()` fails with a loader error (e.g. "Unknown
 * file extension"), not a syntax error, and that failure doesn't match any
 * of the recognized SyntaxError patterns below. Without this guard, the
 * catch-all fallback (first line of stderr) mistakes that loader error for
 * a syntax error and reports a false-positive parse failure on every edit
 * to a non-JS/TS file.
 */
const PARSEABLE_EXTENSIONS = new Set([
	".js",
	".jsx",
	".mjs",
	".cjs",
	".ts",
	".tsx",
	".mts",
	".cts",
]);

/** Substrings that identify a line as the actual syntax-error message, across
 * both Node's own parser ("SyntaxError", "Unexpected token", ...) and esbuild's
 * (tsx's transform), which reports errors as `<file>:<line>:<col>: ERROR: <msg>`
 * rather than throwing a `SyntaxError`. */
const SYNTAX_ERROR_MARKERS = [
	"SyntaxError",
	"Unexpected token",
	"Cannot use import statement",
	"Unterminated",
	"Unexpected end",
	": ERROR:",
];

/**
 * Pick the line that actually describes the failure. `npx` prints "npm
 * notice ..." hints to stderr before running the command, so the naive
 * "first line" fallback would report a notice instead of an error (or, on
 * newer npm, instead of nothing at all when there's no real error).
 */
function relevantErrorLine(stderr: string): string | undefined {
	const lines = stderr.split("\n");
	const marked = lines.find(line =>
		SYNTAX_ERROR_MARKERS.some(marker => line.includes(marker)),
	);
	if (marked) return marked.trim();
	const meaningful = lines.find(
		line =>
			line.trim() &&
			!line.startsWith("npm notice") &&
			!line.startsWith("npm warn"),
	);
	return meaningful?.trim();
}

/**
 * Check if a file parses as valid TypeScript/JavaScript by running it through tsx.
 * Returns an error message if there's a parse failure, null if valid.
 */
export async function checkFileParse(filePath: string): Promise<string | null> {
	if (!PARSEABLE_EXTENSIONS.has(path.extname(filePath).toLowerCase())) {
		return null;
	}
	try {
		// Try to load the file with tsx and capture syntax errors
		const { stderr } = await execFileAsync(
			"npx",
			[
				"tsx",
				"--no-warnings",
				"--eval",
				`import("${path.resolve(filePath)}");`,
			],
			{ timeout: 5000 },
		);

		if (SYNTAX_ERROR_MARKERS.some(marker => stderr.includes(marker))) {
			return relevantErrorLine(stderr) ?? "Syntax error";
		}
		return null;
	} catch (e: unknown) {
		const err = e as { stderr?: string; code?: number };
		// tsx returns non-zero for syntax errors
		if (err.code !== 0 && err.stderr) {
			return relevantErrorLine(err.stderr) ?? "Syntax error";
		}
		// File doesn't exist or other error — not a syntax issue
		return null;
	}
}
