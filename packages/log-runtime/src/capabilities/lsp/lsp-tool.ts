// ── LSP tool ────────────────────────────────────────────────────────────────────
// Full LSP (Language Server Protocol) operations for the agent: diagnostics,
// definition, references, hover, symbols, code actions, rename, etc.
//
// Wraps the LspClientPool's action() dispatch method and formats results
// into human-readable output for the agent to consume and act on.

import type { LspClientPool } from "./lsp-client-pool.ts";
import type { Tool } from "@logician/log-core";

// ── Action enum ────────────────────────────────────────────────────────────────

const ACTIONS = [
	"diagnostics",
	"go-to-definition",
	"references",
	"hover",
	"symbols",
	"workspace-symbols",
	"code-actions",
	"rename",
	"type-definition",
	"implementation",
	"status",
	"capabilities",
	"reload",
] as const;


// ── Formatting helpers ─────────────────────────────────────────────────────────

function formatLocation(loc: { file: string; line: number; column: number }): string {
	const file = loc.file.replace(/^file:\/\//, "");
	return `  ${file}:${loc.line}:${loc.column}`;
}

function formatHover(hover: { contents: string; signature?: string } | null): string {
	if (!hover || !hover.contents?.trim()) return "No hover information.";
	const lines = [hover.contents.trim()];
	if (hover.signature) lines.push(hover.signature);
	return lines.join("\n");
}

function formatSymbols(symbols: { name: string; kind: string; file: string; line: number; column: number }[]): string {
	if (symbols.length === 0) return "No symbols found.";
	const lines = symbols.map(s => {
		const file = s.file.replace(/^file:\/\//, "");
		return `  ${s.kind.padEnd(10)} ${s.name.padEnd(30)} ${file}:${s.line}`;
	});
	return lines.join("\n");
}

function formatCodeActions(actions: { title: string; kind?: string; command?: string }[]): string {
	if (actions.length === 0) return "No code actions available.";
	const lines = actions.map((a, i) => {
		const kind = a.kind ? ` [${a.kind}]` : "";
		return `  ${i + 1}. ${a.title}${kind}`;
	});
	return lines.join("\n");
}

function formatDiagnostics(
	diagnostics: { line: number; column: number; message: string; code?: string | number; severity?: number; source?: string }[],
): string {
	if (diagnostics.length === 0) return "No diagnostics.";
	const lines = diagnostics.map((d, i) => {
		const code = d.code ? ` (${d.code})` : "";
		const source = d.source ? ` [${d.source}]` : "";
		return `  ${i + 1}. ${d.line}:${d.column}${code}${source} ${d.message}`;
	});
	return lines.join("\n");
}

function formatLocations(locations: { file: string; line: number; column: number }[]): string {
	if (locations.length === 0) return "No locations found.";
	return locations.map(formatLocation).join("\n");
}

// ── Tool ───────────────────────────────────────────────────────────────────────

export function createLspTool(pool: LspClientPool): Tool {

	return {
		readOnly: true,
		cacheable: false,
		name: "lsp",
		label: "LSP",
		description:
			`Query the language server for code intelligence. ` +
			`Actions: ${ACTIONS.join(", ")}. ` +
			`Most actions require a file path and optional line/column. ` +
			`For code-actions, use the title number to apply a fix.`,
		promptSnippet:
			"LSP: diagnostics, go-to-definition, references, hover, symbols, workspace-symbols, code-actions, rename, type-definition, implementation, status, capabilities, reload",
		promptGuidelines: [
			"Use lsp to query the language server for code intelligence",
			"go-to-definition: resolve a symbol to its definition location",
			"references: find all usages of a symbol in the codebase",
			"hover: get type info and documentation at a cursor position",
			"symbols: list document symbols in a file",
			"workspace-symbols: search all symbols by name across the workspace",
			"code-actions: list available quickfixes and refactorings",
			"rename: rename a symbol across all references",
		],
		parameters: {
			type: "object",
			properties: {
				action: {
					type: "string",
					enum: ACTIONS,
					description: "LSP action to perform",
				},
				file: {
					type: "string",
					description: "File path to query",
				},
				line: {
					type: "number",
					description: "Line number (1-based)",
				},
				column: {
					type: "number",
					description: "Column number (1-based)",
				},
				query: {
					type: "string",
					description: "Search query (for workspace-symbols and code-actions)",
				},
				new_name: {
					type: "string",
					description: "New name for rename action",
				},
			},
			required: ["action"],
		},
		prepareArguments: (raw): Record<string, unknown> => {
			if (typeof raw === "string") return { action: raw };
			if (!raw || typeof raw !== "object") return {};
			const args = raw as Record<string, unknown>;
			return {
				...args,
				file: args.file ?? args.filePath ?? args.path ?? undefined,
				query: args.query ?? args.q ?? undefined,
				new_name: args.newName ?? args.new_name ?? undefined,
			};
		},
		execute: async (args, _ctx): Promise<string> => {
			const action = String(args.action).toLowerCase().replace(/-/g, "-");
			const file = String(args.file ?? "");
			const line = Number(args.line) || 1;
			const column = Number(args.column) || 1;
			const query = String(args.query ?? "");
			const newName = String(args.new_name ?? "");

			let result: unknown;

			// Actions that don't need a file
			if (!file) {
				switch (action) {
					case "status": {
						const statuses = await pool.action("status");
						return `LSP Server Status:\n${formatStatusList(Array.isArray(statuses) ? statuses : [])}`;
					}
					case "capabilities":
						result = await pool.action("capabilities");
						return `LSP Server Capabilities:\n${JSON.stringify(result, null, 2)}`;
					case "reload":
						result = await pool.action("reload");
						return String(result);
					default:
						return `Action '${action}' requires a file path. Use: lsp(action: '${action}', file: '/path/to/file.ts', line: N, column: N)`;
				}
			}

			result = await pool.action(action, file, line, column, query, newName);

			if (result && typeof result === "object" && "error" in result) {
				return `LSP Error: ${String((result as { error: string }).error)}`;
			}

			switch (action) {
				case "diagnostics":
					return formatDiagnostics(result as Array<{ line: number; column: number; message: string; code?: string | number; severity?: number; source?: string }>);
				case "go-to-definition":
				case "definition":
					return formatLocations(result as Array<{ file: string; line: number; column: number }>);
				case "references":
					return formatLocations(result as Array<{ file: string; line: number; column: number }>);
				case "hover":
					return formatHover(result as { contents: string; signature?: string } | null);
				case "symbols":
					return formatSymbols(result as Array<{ name: string; kind: string; file: string; line: number; column: number }>);
				case "workspace-symbols":
					return formatSymbols(result as Array<{ name: string; kind: string; file: string; line: number; column: number }>);
				case "code-actions":
					return formatCodeActions(result as Array<{ title: string; kind?: string; command?: string }>);
				case "rename":
					if (result) {
						const edit = result as { changes?: Array<{ file: string; range: { startLine: number; startCol: number; endLine: number; endCol: number }; newText: string }> };
						if (edit.changes) {
							const locations = edit.changes.map(c => ({
								file: c.file,
								line: c.range.startLine,
								column: c.range.startCol,
							}));
							return `Rename to '${newName}' affects ${locations.length} location(s):\n` + locations.map(formatLocation).join("\n");
						}
						return "No references found to rename.";
					}
					return "No references found to rename.";
				case "type-definition":
					return formatLocations(result as Array<{ file: string; line: number; column: number }>);
				case "implementation":
					return formatLocations(result as Array<{ file: string; line: number; column: number }>);
				case "status": {
					const statuses = await pool.action("status");
					return `LSP Server Status:\n${formatStatusList(Array.isArray(statuses) ? statuses : [])}`;
				}
				case "capabilities":
					return `LSP Server Capabilities:\n${JSON.stringify(result, null, 2)}`;
				default:
					return JSON.stringify(result, null, 2);
			}
		},
	};
}

function formatStatusList(statuses: unknown[]): string {
	if (!Array.isArray(statuses) || statuses.length === 0) return "  No LSP servers active.";
	return statuses
		.map(s => {
			if (typeof s !== "object" || !s) return "  unknown";
			const server = s as { command?: string; languageId?: string; ready?: boolean };
			const state = server.ready ? "ready" : "starting";
			return `  ${server.languageId ?? "?"} → ${server.command ?? "?"} (${state})`;
		})
		.join("\n");
}
