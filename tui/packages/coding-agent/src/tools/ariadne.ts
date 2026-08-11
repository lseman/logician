// ── ariadne tool ──────────────────────────────────────────────────────────────
// Interface to the Ariadne code graph for semantic code navigation.
// Uses `ariadne agent tool <operation> --params '{...}'` for graph queries.
// Falls back to "not available" when the CLI is missing or the graph is empty.

import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Tool, ToolResult } from "@logician/agent-core/agent/types.ts";
import { formatSize, truncateHead } from "./truncate.ts";

const DEFAULT_LIMIT = 50;
const DEFAULT_DB = "ariadne.db";
const INDEX_REFRESH_INTERVAL_MS = 5_000;
const MAX_PROCESS_OUTPUT = 2 * 1024 * 1024;
const bundledAriadneRoot = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"../../../../../ariadne",
);
const indexRefreshes = new Map<string, Promise<string | null>>();
const indexRefreshedAt = new Map<string, number>();
const cliDialects = new Map<string, Promise<"direct" | "agent">>();

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

async function executable(pathname: string): Promise<boolean> {
	try {
		await access(pathname, constants.X_OK);
		return true;
	} catch {
		return false;
	}
}

/** Resolve the submodule build first, then an explicitly configured/system CLI. */
export async function resolveAriadneBinary(): Promise<string | null> {
	const candidates = [
		process.env.ARIADNE_BIN,
		path.join(bundledAriadneRoot, "target/release/ariadne"),
		path.join(bundledAriadneRoot, "target/debug/ariadne"),
	].filter((candidate): candidate is string => Boolean(candidate));
	for (const candidate of candidates) {
		if (await executable(candidate)) return candidate;
	}
	const { execFile } = await import("node:child_process");
	const { promisify } = await import("node:util");
	try {
		const { stdout } = await promisify(execFile)("which", ["ariadne"], {
			timeout: 3_000,
		});
		return stdout.trim().split("\n")[0]?.trim() || null;
	} catch {
		return null;
	}
}

function runAriadne(
	binary: string,
	args: string[],
	options: { cwd: string; signal?: AbortSignal; timeoutMs?: number },
): Promise<{
	code: number | null;
	stdout: string;
	stderr: string;
	aborted: boolean;
}> {
	return new Promise(resolve => {
		if (options.signal?.aborted) {
			resolve({ code: null, stdout: "", stderr: "", aborted: true });
			return;
		}
		const child = spawn(binary, args, {
			cwd: options.cwd,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		let aborted = false;
		const append = (current: string, chunk: Buffer) =>
			(current + chunk.toString()).slice(-MAX_PROCESS_OUTPUT);
		child.stdout.on("data", chunk => (stdout = append(stdout, chunk)));
		child.stderr.on("data", chunk => (stderr = append(stderr, chunk)));
		const stop = () => {
			aborted = true;
			if (!child.killed) child.kill("SIGKILL");
		};
		options.signal?.addEventListener("abort", stop, { once: true });
		const timer = setTimeout(stop, options.timeoutMs ?? 30_000);
		child.on("error", error => {
			stderr = error.message;
		});
		child.on("close", code => {
			clearTimeout(timer);
			options.signal?.removeEventListener("abort", stop);
			resolve({ code, stdout, stderr, aborted });
		});
	});
}

async function detectCliDialect(
	binary: string,
	cwd: string,
): Promise<"direct" | "agent"> {
	let pending = cliDialects.get(binary);
	if (!pending) {
		pending = runAriadne(binary, ["tool", "--help"], {
			cwd,
			timeoutMs: 3_000,
		}).then(result => (result.code === 0 ? "direct" : "agent"));
		cliDialects.set(binary, pending);
	}
	return pending;
}

async function refreshIndex(
	binary: string,
	dbPath: string,
	cwd: string,
	signal?: AbortSignal,
): Promise<string | null> {
	const lastRefresh = indexRefreshedAt.get(dbPath) ?? 0;
	if (Date.now() - lastRefresh < INDEX_REFRESH_INTERVAL_MS) return null;
	const existing = indexRefreshes.get(dbPath);
	if (existing) return existing;
	const pending = (async () => {
		let dbExists = false;
		try {
			await stat(dbPath);
			dbExists = true;
		} catch {}
		const args = dbExists
			? ["--db", dbPath, "build", "update", "."]
			: ["--db", dbPath, "build", "."];
		const result = await runAriadne(binary, args, {
			cwd,
			signal,
			timeoutMs: dbExists ? 60_000 : 5 * 60_000,
		});
		if (result.aborted) return "Ariadne index refresh was aborted.";
		if (result.code !== 0) {
			return `Ariadne index refresh failed: ${result.stderr.trim() || result.stdout.trim() || `exit ${result.code}`}`;
		}
		indexRefreshedAt.set(dbPath, Date.now());
		return null;
	})().finally(() => indexRefreshes.delete(dbPath));
	indexRefreshes.set(dbPath, pending);
	return pending;
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
		"The graph is built or incrementally refreshed automatically before queries. " +
		"Available operations: status, minimal_context, search, impact, callers_of, callees_of, " +
		"paths, traverse, detect_changes, risk, review_context, affected_flows, " +
		"test_coverage, suggested_questions, architecture, communities, cycles, " +
		"bridge_nodes, hub_nodes, god_nodes, gaps, knowledge_gaps, dead_code, " +
		"flows, large_functions, counterfactual, motifs, graph_diff, health.",
	promptSnippet:
		"Query the Ariadne code graph for semantic analysis: minimal_context, search, impact, callers_of, callees_of, paths, traverse, detect_changes, risk, architecture, etc.",
	promptGuidelines: [
		"Use Ariadne early to orient in unfamiliar code; its workspace index refreshes automatically",
		"Use ariadne for semantic code analysis (symbol context, impact, dependencies)",
		"Prefer ariadne over find/grep when you need relationship analysis",
		"Use minimal_context to resolve a symbol and get bounded neighborhood",
		"Use impact to see what breaks if a symbol changes",
		"Before editing shared symbols, call impact; after a change set, call detect_changes or risk",
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
				oneOf: [
					{ type: "object", additionalProperties: true },
					{ type: "string" },
				],
				description:
					"Additional operation parameters as an object (or JSON string), merged with top-level convenience fields",
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
				description: "Traversal direction: 'forward', 'backward', or 'both'",
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
			} else if (
				typeof args.params === "object" &&
				!Array.isArray(args.params)
			) {
				Object.assign(params, args.params);
			}
		}
		return params;
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const operation = String(args.operation);
		if (!operation) return "Error: operation is required.";

		const ariadnePath = await resolveAriadneBinary();
		if (!ariadnePath) {
			return [
				"Ariadne CLI is unavailable.",
				"",
				"Build the bundled submodule:",
				`  cargo build --release --manifest-path ${path.join(bundledAriadneRoot, "Cargo.toml")}`,
				"",
				"You can also set ARIADNE_BIN or install ariadne on PATH.",
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
			} else if (
				typeof args.params === "object" &&
				!Array.isArray(args.params)
			) {
				Object.assign(cliParams, args.params);
			}
		}

		// Build or incrementally refresh the workspace graph before querying. Calls
		// within a short burst share one refresh and queries remain bounded.
		const refreshWarning = await refreshIndex(
			ariadnePath,
			dbPath,
			ctx.cwd || ".",
			ctx.signal,
		);
		if (ctx.signal?.aborted) return "Error: Command aborted";

		const dialect = await detectCliDialect(ariadnePath, ctx.cwd || ".");
		const paramsJson = JSON.stringify(cliParams);
		const cliArgs = [
			"--db",
			dbPath,
			...(dialect === "agent" ? ["agent"] : []),
			"tool",
			operation,
			"--params",
			paramsJson,
		];
		const query = await runAriadne(ariadnePath, cliArgs, {
			cwd: ctx.cwd || ".",
			signal: ctx.signal,
			timeoutMs: 30_000,
		});
		if (query.aborted) return "Error: Command aborted";
		if (query.code !== 0) {
			const error = query.stderr.trim() || query.stdout.trim();
			if (query.code === 1 && !error)
				return `No results for operation '${operation}'.`;
			return `Error (exit ${query.code}): ${error || "Unknown error"}`;
		}

		const parsed = parseOutput(query.stdout);
		const truncated = truncateHead(parsed, { maxBytes: 50 * 1024 });
		let content = truncated.content;
		const notices: string[] = [];
		if (refreshWarning) notices.push(refreshWarning);
		if (truncated.truncated)
			notices.push(`${formatSize(truncated.maxBytes)} limit reached`);
		if (notices.length) content += `\n\n[${notices.join(". ")}]`;
		return {
			content,
			details: {
				operation,
				target,
				db: dbPath,
				binary: ariadnePath,
				dialect,
				indexFresh: !refreshWarning,
			},
		};
	},
};
