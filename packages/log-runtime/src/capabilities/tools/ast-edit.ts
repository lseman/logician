// ── ast_edit tool ──────────────────────────────────────────────────────────────
// Structural AST-aware rewrites via ast-grep.
//
// Usage:
//   1. Call ast_edit with ops and paths → receives staged diff preview
//   2. If satisfied: write xd://resolve {reason, files: [...]}
//   3. If not: write xd://reject {reason}
//
// Schema:
//   { ops: [{ pat, out }], paths: string[] }
//
// - Metavariables in `pat` ($A, $$$ARGS) substitute into `out`
// - Patterns match AST structure, not text
// - Matches are STAGED as a proposal, not applied directly
// - Finalize by writing to xd://resolve (apply) or xd://reject (discard)

import type { Tool } from "@logician/log-core";
import { executeAstOp, type AstEditResult, type AstOp } from "./support/ast-grep.js";
import { resolvePath } from "./support/utils/path-utils.js";
import { ensureInsideCwd } from "./support/utils/path-utils.js";
import { setStagedEdit } from "./support/staged-edits.js";

// ── Preview rendering ──────────────────────────────────────────────────────────

/** Render a diff-style preview for an ast_edit result. */
function renderEditPreview(
	file: string,
	edits: AstEditResult[],
): string {
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

/** Build the staged edit data for xd://resolve. */
function buildStagedFiles(
	file: string,
	edits: AstEditResult[],
): Array<{ path: string; content: string }> {
	// For ast_edit, we return file paths and replacements
	// The resolve handler will apply these to disk
	return edits.map((edit) => ({
		path: file,
		content: edit.replacement,
	}));
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
		},
		required: ["ops", "paths"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			ops: args.ops ?? args.rewrites,
			paths: args.paths ?? args.file_paths ?? args.files,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const ops = (args.ops as AstOp[]) ?? [];
		const paths = (args.paths as string[]) ?? [];

		if (!ops.length) {
			return "Error: Provide at least one operation in `ops`.";
		}

		if (!paths.length) {
			return "Error: Provide at least one path in `paths`.";
		}

		// Resolve paths
		const cwd = ctx.cwd || process.cwd();
		const resolvedPaths = paths.map((p) => resolvePath(cwd, p));

		// Validate paths
		for (const p of resolvedPaths) {
			ensureInsideCwd(
				cwd,
				p,
				ctx.allowedPaths,
				ctx.allowAllPaths ?? false,
			);
		}

		try {
			// Execute the AST edit operations
			const fileEdits = await executeAstOp(ops, resolvedPaths);

			if (fileEdits.length === 0) {
				return "No matches found for the given patterns.";
			}

			// Render preview
			const previews = fileEdits.map((fe) => renderEditPreview(fe.file, fe.edits));
			const preview = previews.join("\n\n---\n\n");

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

			return preview;
		} catch (e) {
			return `Error during AST edit: ${e instanceof Error ? e.message : String(e)}`;
		}
	},
};
