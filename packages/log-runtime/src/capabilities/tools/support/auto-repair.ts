// ── Auto-repair: parse failure detection, hunk isolation, and repair ──────────
//
// When an edit causes a parse failure, this module:
// 1. Detects the failure by running the file through tsx and catching errors
// 2. Isolates the culprit hunk (the edit that caused the parse error)
// 3. Generates a repair suggestion (for consumption by a small model)

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as fs from "node:fs";
import * as path from "node:path";

const execFileAsync = promisify(execFile);

// ── Types ──────────────────────────────────────────────────────────────────────

/** An edit that was applied to a file. */
export interface FileEdit {
	/** Byte offset of the original text. */
	originalStart: number;
	/** Byte offset after the original text. */
	originalEnd: number;
	/** The replacement text. */
	replacement: string;
	/** The original text. */
	original: string;
}

// ── Parse failure detection ────────────────────────────────────────────────────

/**
 * Check if a file parses as valid TypeScript/JavaScript by running it through tsx.
 * Returns an error message if there's a parse failure, null if valid.
 */
export async function checkFileParse(filePath: string): Promise<string | null> {
	try {
		// Try to load the file with tsx and capture syntax errors
		const { stderr } = await execFileAsync("npx", [
			"tsx",
			"--no-warnings",
			"--eval",
			`import("${path.resolve(filePath)}");`,
		], { timeout: 5000 });

		// Check for common syntax error patterns
		if (stderr.includes("SyntaxError") || stderr.includes("Unexpected token")) {
			// Extract the relevant error line
			const errorLine = stderr
				.split("\n")
				.find((line) =>
					line.includes("SyntaxError") ||
					line.includes("Unexpected token") ||
					line.includes("Cannot use import statement") ||
					line.includes("Unterminated") ||
					line.includes("Unexpected end")
				) ?? stderr.split("\n")[0];
			return errorLine?.trim() ?? "Syntax error";
		}

		return null;
	} catch (e: unknown) {
		const err = e as { stderr?: string; code?: number };
		// tsx returns non-zero for syntax errors
		if (err.code !== 0 && err.stderr) {
			const errorLine = err.stderr
				.split("\n")
				.find((line) =>
					line.includes("SyntaxError") ||
					line.includes("Unexpected token") ||
					line.includes("Cannot use import statement")
				) ?? err.stderr.split("\n")[0];
			return errorLine?.trim() ?? "Syntax error";
		}
		// File doesn't exist or other error — not a syntax issue
		return null;
	}
}

/**
 * Lightweight syntax check using bracket matching.
 * Good for quick pre-flight checks before heavier parsing.
 */
export function checkBracketBalance(content: string): string | null {
	const stack: string[] = [];
	const pairs: Record<string, string> = {
		"{": "}",
		"(": ")",
		"[": "]",
	};
	const closing: Record<string, string> = {
		"}": "{",
		")": "(",
		"]": "[",
	};
	let inString = false;
	let stringChar = "";
	let inComment = false;
	let prevChar = "";

	for (let i = 0; i < content.length; i++) {
		const ch = content[i];

		// Skip escaped characters
		if (ch === "\\" && inString && prevChar === "\\") {
			prevChar = ch;
			continue;
		}

		// Handle string boundaries
		if ((ch === '"' || ch === "'" || ch === "`") && !inComment) {
			if (inString && ch === stringChar && prevChar !== "\\") {
				inString = false;
			} else if (!inString) {
				inString = true;
				stringChar = ch;
			}
			prevChar = ch;
			continue;
		}

		if (inString) {
			prevChar = ch;
			continue;
		}

		// Handle comments
		if (ch === "/" && prevChar === "/") {
			// Line comment — skip to end of line
			const nextLineIdx = content.indexOf("\n", i);
			i = nextLineIdx === -1 ? content.length - 1 : nextLineIdx;
			prevChar = ch;
			continue;
		}
		if (ch === "/" && prevChar === "*") {
			// Block comment — skip to */
			const endIdx = content.indexOf("*/", i + 1);
			i = endIdx === -1 ? content.length - 1 : endIdx + 1;
			prevChar = ch;
			continue;
		}

		// Track brackets
		if (pairs[ch]) {
			stack.push(ch);
		} else if (closing[ch]) {
			const expected = closing[ch];
			if (stack.length === 0 || stack[stack.length - 1] !== expected) {
				return `Unmatched closing '${ch}' at position ${i}`;
			}
			stack.pop();
		}

		prevChar = ch;
	}

	if (stack.length > 0) {
		const unclosed = stack.join(", ");
		return `Unclosed ${unclosed} (expected closing)`;
	}

	return null;
}

// ── Culprit hunk isolation ─────────────────────────────────────────────────────

/**
 * Isolate the culprit edit that caused a parse failure.
 * Given a list of edits and a file path, returns the edit(s) most likely
 * responsible based on byte offset proximity to the error.
 */
export function isolateCulpritHunks(
	filePath: string,
	edits: FileEdit[],
	parseError: string,
): Array<{
	edits: FileEdit[];
	reason: string;
}> {
	// Read the file content
	const content = fs.readFileSync(filePath, "utf-8");

	// Try to extract position from error message
	let errorPosition: number | null = null;
	const posMatch = parseError.match(/position\s+(\d+)/i);
	if (posMatch) {
		errorPosition = parseInt(posMatch[1], 10);
	} else {
		// Try to find line number and estimate position
		const lineMatch = parseError.match(/line\s+(\d+)/i);
		if (lineMatch) {
			const lineNumber = parseInt(lineMatch[1], 10);
			const lines = content.split("\n");
			let pos = 0;
			for (let i = 0; i < Math.min(lineNumber - 1, lines.length - 1); i++) {
				pos += lines[i].length + 1;
			}
			errorPosition = pos;
		}
	}

	// Find the edit closest to the error position
	let closestEdit: FileEdit | null = null;
	let closestDistance = Infinity;

	for (const edit of edits) {
		const midPoint = (edit.originalStart + edit.originalEnd) / 2;
		const distance = errorPosition !== null
			? Math.abs(midPoint - errorPosition!)
			: edit.originalStart;

		if (distance < closestDistance) {
			closestDistance = distance;
			closestEdit = edit;
		}
	}

	if (!closestEdit) {
		return [{ edits, reason: parseError }];
	}

	return [
		{
			edits: [closestEdit],
			reason: parseError,
		},
	];
}

// ── Repair suggestion generation ───────────────────────────────────────────────

/**
 * Generate a repair suggestion for a parse failure.
 */
export function generateRepairSuggestion(
	filePath: string,
	culpritEdits: Array<{
		edits: FileEdit[];
		reason: string;
	}>,
): string {
	const suggestions: string[] = [];
	suggestions.push(`# Auto-Repair Suggestion for \`${filePath}\``);
	suggestions.push("");
	suggestions.push("## Parse Error");

	for (const { reason } of culpritEdits) {
		suggestions.push(`- ${reason}`);
	}

	suggestions.push("");
	suggestions.push("## Suggested Fixes");

	for (const { edits } of culpritEdits) {
		for (const edit of edits) {
			suggestions.push("");
			suggestions.push("### Offending edit:");
			const origPreview = edit.original.length > 100
				? edit.original.slice(0, 100) + "..."
				: edit.original;
			const replPreview = edit.replacement.length > 100
				? edit.replacement.slice(0, 100) + "..."
				: edit.replacement;
			suggestions.push(`**Original:** \`${origPreview}\``);
			suggestions.push(`**Replacement:** \`${replPreview}\``);
			suggestions.push("");
			suggestions.push("Common fixes:");
			suggestions.push("- Check for unbalanced braces, brackets, or parentheses");
			suggestions.push("- Look for missing semicolons or commas");
			suggestions.push("- Verify proper JSX/TSX syntax if applicable");
			suggestions.push("- Check for unterminated strings or comments");
		}
	}

	return suggestions.join("\n");
}
