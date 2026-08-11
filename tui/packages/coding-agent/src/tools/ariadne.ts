// ── ariadne tool ──────────────────────────────────────────────────────────────
// Interface to the Ariadne code graph for semantic code navigation.
// Uses `ariadne agent tool <operation> --params '{...}'` for graph queries.
// Falls back to "not available" when the CLI is missing or the graph is empty.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import type { Tool, ToolResult } from "@logician/agent-core/agent/types.ts";
import { formatSize, truncateHead } from "./truncate.ts";

const DEFAULT_LIMIT = 50;
const DEFAULT_DB = "ariadne.db";

// Map of operations that accept a "path" parameter (file-level targets)
const PATH_OPERATIONS = new Set([
	"minimal_context",
	"context",
	"impact",
	"callers_of",
	"callees_of",
	"detect_changes",
	"risk",
	"review_context",
	"affected_flows",
	"test_coverage",
	"suggested_questions",
	"counterfactual",
	"rename_preview",
	"find_related",
]);

function resolveAriadneDb(cwd: string): string {
	// Check for --db flag in environment or common locations
	const envDb = process.env.ARIADNE_DB;
	if (envDb) return envDb;
	const localDb = path.join(cwd, DEFAULT_DB);
	return localDb;
}

/** Parse ariadne agent tool output, handling JSON wrapping */
function parseOutput(raw: string): string {
	// Ariadne may return JSON with a "result" field or plain text
	const trimmed = raw.trim();
	if (!trimmed) return "(empty response)";

	// Try to extract meaningful content from JSON
	try {
		const parsed = JSON.parse(trimmed);
		// If it's a JSON object with a result/message field, use that
		if (parsed.result && typeof parsed.result === "string") {
			return parsed.result;
		}
		if (parsed.message && typeof parsed.message === "string") {
			return parsed.message;
		}
		// Otherwise stringify the whole thing
		return JSON.stringify(parsed, null, 2);
	} catch {
		// Not JSON, return as-is
		return trimmed;
	}
}

export const ariadne: Tool = {
	readOnly: true,
	name: "ariadne",
	label: "Ariadne Code Graph",
	description:
		"Query the Ariadne code graph for semantic code analysis. " +
		"Use for: finding symbol context, impact analysis, dependency tracing, " +
		"change risk assessment, caller/callee relationships, and structural analysis. " +
		"Returns structured, bounded context — ideal when you need to understand " +
		"code relationships without reading entire files. " +
		"Available operations: minimal_context, search, impact, callers_of, callees_of, " +
		"paths, traverse, detect_changes, risk, review_context, affected_flows, " +
		"test_coverage, suggested_questions, architecture, communities, cycles, " +
		"bridge_nodes, hub_nodes, god_nodes, gaps, knowledge_gaps, dead_code, " +
		"flows, large_functions, counterfactual, motifs, graph_diff, health.",
	promptSnippet:
		"Query the Ariadne code graph for semantic analysis: minimal_context, search, impact, callers_of, callees_of, paths, traverse, detect_changes, risk, architecture, etc.",
	promptGuidelines: [
		"Use ariadne for semantic code analysis (symbol context, impact, dependencies)",
		"Prefer ariadne over find/grep when you need relationship analysis",
		"Use minimal_context to resolve a symbol and get bounded neighborhood",
		"Use impact to see what breaks if a symbol changes",
		"Use callers_of / callees_of for call graph traversal",
		"Use detect_changes for diff-based risk analysis",
		"Use search for hybrid FTS5 + topology search",
		"Use traverse for bounded graph traversal with token budget",
		"Use architecture for community/coupling overview",
		"Fall back to find/grep/read_file when ariadne has no data for the target",
	],
	parameters: {
		type: "object",
		properties: {
			operation: {
				type: "string",
				description:
					"Operation to perform. Common: minimal_context, search, impact, callers_of, callees_of, paths, traverse, detect_changes, risk, review_context, affected_flows, test_coverage, suggested_questions, architecture, communities, cycles, bridge_nodes, hub_nodes, god_nodes, gaps, knowledge_gaps, dead_code, flows, large_functions, counterfactual, motifs, graph_diff, health",
			},
			target: {
				type: "string",
				description:
					"Target symbol, file path, or search query (operation-dependent)",
			},
			params: {
				type: "string",
				description:
					"Additional JSON parameters for the operation (optional, merged with other params)",
			},
			base: {
				type: "string",
				description:
					"Git reference for diff-based operations (e.g. HEAD~1, main)",
			},
			max_hops: {
				type: "number",
				description: "Maximum hops for graph traversal operations",
			},
			max_depth: {
				type: "number",
				description: "Maximum depth for traversal operations",
			},
			token_budget: {
				type: "number",
				description: "Token budget for bounded operations",
			},
			limit: {
				type: "number",
				description: `Result limit (default: ${DEFAULT_LIMIT})`,
			},
			direction: {
				type: "string",
				description:
					"Traversal direction: 'forward', 'backward', or 'both'",
			},
			algorithm: {
				type: "string",
				description:
					"Algorithm for community detection: 'louvain', 'leiden', 'infomap'",
			},
		},
		required: ["operation"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") return { operation: raw };
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		const params: Record<string, unknown> = {
			operation: args.operation ?? args.op ?? args.command,
			target: args.target ?? args.symbol ?? args.file,
			base: args.base,
			max_hops: args.max_hops ?? args.maxHops,
			max_depth: args.max_depth ?? args.maxDepth,
			token_budget: args.token_budget ?? args.tokenBudget,
			limit: args.limit ?? args.response_limit ?? args.limit,
			direction: args.direction ?? args.dir,
			algorithm: args.algorithm ?? args.algo,
		};
		// Merge additional params if provided as JSON string or object
		if (args.params) {
			if (typeof args.params === "string") {
				try {
					const extra = JSON.parse(args.params);
					Object.assign(params, extra);
				} catch {
					// Keep as string, ariadne CLI will parse it
				}
			} else if (typeof args.params === "object" && !Array.isArray(args.params)) {
				Object.assign(params, args.params);
			}
		}
		return params;
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const operation = String(args.operation);
		if (!operation) return "Error: operation is required.";

		// Check if ariadne CLI is available
		const { execFile } = await import("node:child_process");
		const { promisify } = await import("node:util");
		const execFileAsync = promisify(execFile);
		let ariadnePath: string | null = null;
		try {
			const { stdout } = await execFileAsync("which", ["ariadne"], {
				timeout: 3000,
			});
			ariadnePath = stdout.trim().split("\n")[0].trim() || null;
		} catch {
			// not on PATH
		}
		if (!ariadnePath) {
			return [
				"Ariadne CLI is not installed or not in PATH.",
				"",
				"To install:",
				"  cargo install --path crates/ariadne-graph",
				"",
				"Or use the system package if available.",
				"",
				"Without Ariadne, use find/grep/read_file for code navigation.",
			].join("\n");
		}

		const dbPath = resolveAriadneDb(ctx.cwd || ".");
		const target = args.target ? String(args.target) : "";

		// Build params object for the CLI
		const cliParams: Record<string, unknown> = {};
		if (target) cliParams.target = target;
		if (args.base) cliParams.base = String(args.base);
		if (args.max_hops) cliParams.max_hops = Number(args.max_hops);
		if (args.max_depth) cliParams.max_depth = Number(args.max_depth);
		if (args.token_budget) cliParams.token_budget = Number(args.token_budget);
		if (args.limit) cliParams.limit = Number(args.limit);
		if (args.direction) cliParams.direction = String(args.direction);
		if (args.algorithm) cliParams.algorithm = String(args.algorithm);

		// Merge any additional params from the params field
		if (args.params) {
			if (typeof args.params === "string") {
				try {
					const extra = JSON.parse(args.params);
					Object.assign(cliParams, extra);
				} catch {
					// If it's not valid JSON, pass as-is
					cliParams._raw = args.params;
				}
			} else if (typeof args.params === "object" && !Array.isArray(args.params)) {
				Object.assign(cliParams, args.params);
			}
		}

		// Ensure the db exists
		if (!existsSync(dbPath)) {
			return [
				`Ariadne database not found at ${dbPath}.`,
				"",
				"To build the graph:",
				`  ariadne --db ${dbPath} build .`,
				"",
				"Or set ARIADNE_DB environment variable to point to your database.",
			].join("\n");
		}

		// Build the CLI command
		const paramsJson = JSON.stringify(cliParams);
		const cliArgs = [
			"--db",
			dbPath,
			"agent",
			"tool",
			operation,
			"--params",
			paramsJson,
		];

		return new Promise<string | ToolResult>(resolve => {
			if (ctx.signal?.aborted) {
				resolve("Error: Command aborted");
				return;
			}

			const child = spawn(ariadnePath, cliArgs, {
				stdio: ["ignore", "pipe", "pipe"],
				timeout: 30000, // 30s timeout for graph queries
			});

			let stdout = "";
			let stderr = "";
			let killed = false;

			const onAbort = () => {
				killed = true;
				if (!child.killed) child.kill("SIGKILL");
			};
			ctx.signal?.addEventListener("abort", onAbort, { once: true });

			child.stdout?.on("data", (chunk: Buffer) => {
				stdout += chunk.toString();
			});

			child.stderr?.on("data", (chunk: Buffer) => {
				stderr += chunk.toString();
			});

			child.on("error", err => {
				ctx.signal?.removeEventListener("abort", onAbort);
				resolve(`Error: Failed to run ariadne: ${err.message}`);
			});

			child.on("close", code => {
				ctx.signal?.removeEventListener("abort", onAbort);

				if (killed || ctx.signal?.aborted) {
					resolve("Error: Command aborted");
					return;
				}

				if (code === 0) {
					const parsed = parseOutput(stdout);
					const truncated = truncateHead(parsed, {
						maxBytes: 50 * 1024,
					});
					let result = truncated.content;
					const notices: string[] = [];
					if (truncated.truncated) {
						notices.push(`${formatSize(truncated.maxBytes)} limit reached`);
					}
					if (notices.length) {
						result += `\n\n[${notices.join(". ")}]`;
					}
					resolve({
						content: result,
						details: {
							operation,
							target,
							db: dbPath,
						},
					});
					return;
				}

				// Handle non-zero exit codes
				const errorMsg = stderr.trim() || stdout.trim();
				if (code === 1 && !errorMsg) {
					resolve(`No results for operation '${operation}'.`);
					return;
				}
				resolve(`Error (exit ${code}): ${errorMsg || "Unknown error"}`);
			});

			// Set timeout
			setTimeout(() => {
				if (!child.killed) {
					killed = true;
					child.kill("SIGKILL");
				}
			}, 30000);
		});
	},
};
