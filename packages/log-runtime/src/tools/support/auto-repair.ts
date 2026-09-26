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
 * Find the line that actually describes a syntax error, if any. `import()`
 * executes the file, so a non-zero exit or non-empty stderr can just as
 * easily mean a runtime throw, a missing import, or a loader quirk (e.g. an
 * unrecognized-case extension) — none of which are syntax regressions this
 * check should report. Only a line matching a known marker counts; anything
 * else returns `undefined` (not "some other line", which used to make every
 * unrelated failure look like a syntax error).
 */
function syntaxErrorLine(stderr: string): string | undefined {
	return stderr
		.split("\n")
		.find(line => SYNTAX_ERROR_MARKERS.some(marker => line.includes(marker)))
		?.trim();
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

		return syntaxErrorLine(stderr) ?? null;
	} catch (e: unknown) {
		const err = e as { stderr?: string; code?: number };
		// tsx returns non-zero for syntax errors, but also for runtime throws,
		// missing imports, and other non-syntax failures triggered by actually
		// executing the file — only report a match against a known marker.
		if (err.code !== 0 && err.stderr) {
			return syntaxErrorLine(err.stderr) ?? null;
		}
		// File doesn't exist or other error — not a syntax issue
		return null;
	}
}
