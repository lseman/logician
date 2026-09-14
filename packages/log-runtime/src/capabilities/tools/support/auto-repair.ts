// ── Auto-repair: parse failure detection ───────────────────────────────────────
// After an edit, run the file through tsx to catch syntax errors it introduced.

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Check if a file parses as valid TypeScript/JavaScript by running it through tsx.
 * Returns an error message if there's a parse failure, null if valid.
 */
export async function checkFileParse(filePath: string): Promise<string | null> {
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

		// Check for common syntax error patterns
		if (stderr.includes("SyntaxError") || stderr.includes("Unexpected token")) {
			// Extract the relevant error line
			const errorLine =
				stderr
					.split("\n")
					.find(
						line =>
							line.includes("SyntaxError") ||
							line.includes("Unexpected token") ||
							line.includes("Cannot use import statement") ||
							line.includes("Unterminated") ||
							line.includes("Unexpected end"),
					) ?? stderr.split("\n")[0];
			return errorLine?.trim() ?? "Syntax error";
		}

		return null;
	} catch (e: unknown) {
		const err = e as { stderr?: string; code?: number };
		// tsx returns non-zero for syntax errors
		if (err.code !== 0 && err.stderr) {
			const errorLine =
				err.stderr
					.split("\n")
					.find(
						line =>
							line.includes("SyntaxError") ||
							line.includes("Unexpected token") ||
							line.includes("Cannot use import statement"),
					) ?? err.stderr.split("\n")[0];
			return errorLine?.trim() ?? "Syntax error";
		}
		// File doesn't exist or other error — not a syntax issue
		return null;
	}
}
