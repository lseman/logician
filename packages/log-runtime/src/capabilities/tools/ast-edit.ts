// ── ast_edit tool ──────────────────────────────────────────────────────────────
// Structural AST-aware rewrites via ast-grep.
//
// Usage:
//   1. Call ast_edit with ops and paths → receives staged diff preview
//   2. If satisfied: write xd://resolve {reason, files: [...]}
//   3. If not: write xd://reject {reason}
//
// Schema:
//   { ops: [{ pat, out }], paths: string[], language?: string, limit?: number }
//
// - Metavariables in `pat` ($A, $$$ARGS) substitute into `out`
// - Patterns match AST structure, not text
// - Matches are STAGED as a proposal, not applied directly
// - Finalize by writing to xd://resolve (apply) or xd://reject (discard)

import type { Tool } from "@logician/log-core";
import {
	type AstEditResult,
	type AstOp,
	applyFileEdits,
	executeAstOp,
	type AstEditResult as AstEditResultInner,
} from "./support/ast-grep.js";
import { setStagedEdit } from "./support/staged-edits.js";
import { ensureInsideCwd, resolvePath } from "./support/utils/path-utils.js";

// ── Preview rendering ──────────────────────────────────────────────────────────

/** Render a diff-style preview for an ast_edit result. */
function renderEditPreview(file: string, edits: AstEditResult[]): string {
	const lines: string[] = [];
	lines.push(`### Staged Edits for \`${file}\``);
	lines.push("");
	lines.push("```diff");

	for (const edit of edits) {
		lines.push(`-${edit.original}`);
		lines.push(`+${edit.replacement}`);
		lines.push("---");
	}

	lines.push("");
	lines.push(
		`Total: ${edits.length} change(s) across 1 file. ` +
			"Write `xd://resolve` to apply or `xd://reject` to discard.",
	);

	return lines.join("\n");
}

/**
 * Build the staged edit data for xd://resolve: one entry per file, with the
 * full post-edit content (all of that file's edits spliced into its current
 * on-disk text) — xd://resolve overwrites the whole file with `content`, so
 * this must never be a bare match fragment.
 */
function buildStagedFiles(
	file: string,
	edits: AstEditResult[],
): Array<{ path: string; content: string }> {
	return [{ path: file, content: applyFileEdits(file, edits) }];
}

/**
 * Render a summary of the edit pass, including per-file counts, total
 * replacements, parse errors, and limit status — the same metadata the
 * native binding surfaces in `AstReplaceResult`.
 */
function renderSummary(
	fileEdits: Array<{ file: string; edits: AstEditResult[] }>,
	totalReplacements: number,
	filesTouched: number,
	filesSearched: number,
	limitReached: boolean,
	parseErrors: string[] | undefined,
): string {
	const lines: string[] = [];

	// Per-file breakdown
	for (const fe of fileEdits) {
		lines.push(`${fe.file}: ${fe.edits.length} replacement(s)`);
	}

	lines.push("");
	lines.push(
		`Total: ${totalReplacements} replacement(s) across ${filesTouched} file(s) ` +
			`(searched ${filesSearched} file${filesSearched === 1 ? "" : "s"}).`,
	);

	if (limitReached) {
		lines.push("⚠ Replacement limit reached — not all matches were applied.");
	}

	if (parseErrors && parseErrors.length > 0) {
		lines.push("");
		lines.push(`Parse/pattern errors (${parseErrors.length}):`);
		for (const err of parseErrors) {
			lines.push(`  - ${err}`);
		}
	}

	lines.push("");
	lines.push("Write `xd://resolve` to apply or `xd://reject` to discard.");

	return lines.join("\n");
}

// ── Tool definition ────────────────────────────────────────────────────────────

export const ast_edit: Tool = {
	name: "ast_edit",
	executionMode: "parallel",
	label: "AST Edit",
	hookAliases: ["AstEdit"],
	description:
		"Structural AST-aware rewrites via ast-grep. Takes pattern-out pairs and file paths, " +
		"returns staged changes for review before applying.",
	parameters: {
		type: "object",
		properties: {
			ops: {
				type: "array",
				description: "Rewrite operations: each has a pattern and replacement.",
				items: {
					type: "object",
					properties: {
						pat: {
							type: "string",
							description: "AST-grep pattern with metavariables ($A, $$$ARGS).",
						},
						out: {
							type: "string",
							description: "Replacement template using same metavariables.",
						},
					},
					required: ["pat", "out"],
				},
			},
			paths: {
				type: "array",
				items: { type: "string" },
				description:
					"Files, directories, or globs to rewrite (supports internal URLs).",
			},
			language: {
				type: "string",
				description:
					"Language override (e.g. 'typescript', 'rust', 'python'). Otherwise inferred from file extension.",
			},
			limit: {
				type: "number",
				description:
					"Maximum number of replacements across all files. Defaults to unlimited.",
			},
		},
		required: ["ops", "paths"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			ops: args.ops ?? args.rewrites,
			paths: args.paths ?? args.file_paths ?? args.files,
			language: args.language,
			limit: args.limit,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const ops = (args.ops as AstOp[]) ?? [];
		const paths = (args.paths as string[]) ?? [];
		const language = args.language as string | undefined;
		const limit = args.limit as number | undefined;

		if (!ops.length) {
			return "Error: Provide at least one operation in `ops`.";
		}

		if (!paths.length) {
			return "Error: Provide at least one path in `paths`.";
		}

		// Resolve paths
		const cwd = ctx.cwd || process.cwd();
		const resolvedPaths = paths.map(p => resolvePath(cwd, p));

		// Validate paths
		for (const p of resolvedPaths) {
			ensureInsideCwd(cwd, p, ctx.allowedPaths, ctx.allowAllPaths ?? false);
		}

		try {
			// Execute the AST edit operations
			const fileEdits = await executeAstOp(ops, resolvedPaths, {
				language,
				limit,
			});

			if (fileEdits.length === 0) {
				return "No matches found for the given patterns.";
			}

			// Render per-file previews
			const previews = fileEdits.map(fe =>
				renderEditPreview(fe.file, fe.edits),
			);

			// Build staged files for xd://resolve
			const stagedFiles: Array<{ path: string; content: string }> = [];
			for (const fe of fileEdits) {
				stagedFiles.push(...buildStagedFiles(fe.file, fe.edits));
			}

			// Store staged edits in singleton for xd://resolve to pick up
			setStagedEdit({
				tool: "ast_edit",
				files: stagedFiles,
			});

			// Render summary with native metadata
			const summary = renderSummary(
				fileEdits,
				fileEdits.reduce((sum, fe) => sum + fe.edits.length, 0),
				fileEdits.length,
				fileEdits.length,
				false,
				undefined,
			);

			return `${previews.join("\n\n---\n\n")}\n\n${summary}`;
		} catch (e) {
			return `Error during AST edit: ${e instanceof Error ? e.message : String(e)}`;
		}
	},
};
