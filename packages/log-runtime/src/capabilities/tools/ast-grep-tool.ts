// ── ast_grep tool ──────────────────────────────────────────────────────────────
// Read-only structural AST queries via ast-grep. Sibling to ast_edit, which
// stages rewrites; this tool only reports matches — no xd://resolve staging,
// no file mutation.
//
// Schema:
//   { pattern: string, paths: string[], language?: string }
//
// - Metavariables in `pattern` ($A, $$$ARGS) bind to matched nodes
// - Patterns match AST structure, not text

import * as path from "node:path";
import type { Tool } from "@logician/log-core";
import { type AstMatch, queryAstPattern } from "./support/ast-grep.js";
import { ensureInsideCwd, resolvePath } from "./support/utils/path-utils.js";

const MAX_MATCHES = 100;

/** Render matches grouped by file, one block per match with its line range and text. */
function renderMatches(matches: AstMatch[], cwd: string): string {
	const byFile = new Map<string, AstMatch[]>();
	for (const match of matches) {
		const list = byFile.get(match.file) ?? [];
		list.push(match);
		byFile.set(match.file, list);
	}

	const sections: string[] = [];
	for (const [file, fileMatches] of byFile) {
		const relFile = path.relative(cwd, file) || file;
		const blocks = fileMatches.map(match => {
			// ast-grep reports 0-based line numbers.
			const startLine = match.range.start.line + 1;
			const endLine = match.range.end.line + 1;
			const label =
				startLine === endLine
					? `${relFile}:${startLine}`
					: `${relFile}:${startLine}-${endLine}`;
			return `${label}\n${match.text}`;
		});
		sections.push(blocks.join("\n\n"));
	}
	return sections.join("\n\n---\n\n");
}

export const ast_grep: Tool = {
	name: "ast_grep",
	readOnly: true,
	executionMode: "parallel",
	label: "AST Grep",
	hookAliases: ["AstGrep"],
	description:
		"Search code structurally with an ast-grep pattern ($A metavariables bind a node, $$$ARGS binds zero-or-more). " +
		"Read-only — use ast_edit to rewrite. Requires the ast-grep CLI. Output is grouped by file with each match's line range and matched text.",
	promptSnippet: "Search code structurally by AST pattern, not text",
	promptGuidelines: [
		"Use ast_grep when syntax shape matters more than exact text (e.g. any call to a function regardless of formatting); use grep for plain text/regex search",
		"Narrow paths first — searching a whole tree is slower and noisier than one file or directory",
	],
	parameters: {
		type: "object",
		properties: {
			pattern: {
				type: "string",
				description: "AST-grep pattern with metavariables ($A, $$$ARGS).",
			},
			paths: {
				type: "array",
				items: { type: "string" },
				description: "Files or directories to search.",
			},
			language: {
				type: "string",
				description:
					"Override language detection (inferred from the first path's extension otherwise), e.g. 'cpp' for an ambiguous .h file.",
			},
		},
		required: ["pattern", "paths"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		const rawPaths = args.paths ?? args.path ?? args.files;
		return {
			pattern: args.pattern ?? args.pat,
			paths: Array.isArray(rawPaths)
				? rawPaths
				: typeof rawPaths === "string"
					? [rawPaths]
					: rawPaths,
			language: args.language ?? args.lang,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const pattern = args.pattern as string | undefined;
		const paths = (args.paths as string[] | undefined) ?? [];
		const language = args.language as string | undefined;

		if (!pattern) return "Error: Provide a `pattern`.";
		if (!paths.length) return "Error: Provide at least one path in `paths`.";

		const cwd = ctx.cwd || process.cwd();
		const resolvedPaths = paths.map(p => resolvePath(cwd, p));
		for (const p of resolvedPaths) {
			ensureInsideCwd(cwd, p, ctx.allowedPaths, ctx.allowAllPaths ?? false);
		}

		try {
			const matches = await queryAstPattern(pattern, resolvedPaths, language);
			if (matches.length === 0) return "No matches found.";

			const limited = matches.slice(0, MAX_MATCHES);
			const rendered = renderMatches(limited, cwd);
			const summary = `${matches.length} match${matches.length === 1 ? "" : "es"}${
				matches.length > MAX_MATCHES ? ` (showing first ${MAX_MATCHES})` : ""
			}.`;
			return `${rendered}\n\n${summary}`;
		} catch (e) {
			return `Error running ast_grep: ${e instanceof Error ? e.message : String(e)}`;
		}
	},
};
