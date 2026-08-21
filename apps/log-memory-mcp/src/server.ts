import { createHash } from "node:crypto";
import type {
	HookPhase,
	MemoryStore,
	MemoryType,
	RawObservation,
} from "@logician/log-memory";

export const MCP_PROTOCOL_VERSION = "2025-03-26";

interface JsonRpcRequest {
	jsonrpc?: unknown;
	id?: unknown;
	method?: unknown;
	params?: unknown;
}

interface ToolDefinition {
	name: string;
	description: string;
	inputSchema: Record<string, unknown>;
}

interface SearchEntry {
	id: string;
	kind: "memory" | "observation" | "claim";
	type: string;
	title: string;
	summary: string;
	score: number;
	reasons: string[];
}

const MEMORY_TYPES: MemoryType[] = [
	"pattern",
	"preference",
	"architecture",
	"bug",
	"workflow",
	"fact",
];

const HOOK_PHASES: HookPhase[] = [
	"session_start",
	"prompt_submit",
	"pre_tool_use",
	"post_tool_use",
	"post_tool_failure",
	"pre_compact",
	"stop",
	"notification",
];

export const MEMORY_MCP_TOOLS: ToolDefinition[] = [
	{
		name: "memory_search",
		description:
			"Search workspace-scoped durable memories and observations. Returns compact results with stable IDs.",
		inputSchema: {
			type: "object",
			properties: {
				query: { type: "string", minLength: 1 },
				limit: { type: "integer", minimum: 1, maximum: 20 },
			},
			required: ["query"],
			additionalProperties: false,
		},
	},
	{
		name: "memory_get",
		description:
			"Expand up to 20 stable memory or observation IDs in the configured workspace.",
		inputSchema: {
			type: "object",
			properties: {
				ids: {
					type: "array",
					items: { type: "string", minLength: 1 },
					minItems: 1,
					maxItems: 20,
				},
			},
			required: ["ids"],
			additionalProperties: false,
		},
	},
	{
		name: "memory_save",
		description:
			"Save explicit durable knowledge. Reusing the idempotency key returns the original memory.",
		inputSchema: {
			type: "object",
			properties: {
				content: { type: "string", minLength: 1 },
				idempotencyKey: { type: "string", minLength: 1 },
				type: { type: "string", enum: MEMORY_TYPES },
				strength: { type: "integer", minimum: 1, maximum: 10 },
				concepts: { type: "array", items: { type: "string" }, maxItems: 20 },
				files: { type: "array", items: { type: "string" }, maxItems: 20 },
				sessionId: { type: "string" },
				agentId: { type: "string" },
			},
			required: ["content", "idempotencyKey"],
			additionalProperties: false,
		},
	},
	{
		name: "memory_observe",
		description:
			"Record structured prompt or tool evidence. Reusing the idempotency key does not duplicate it.",
		inputSchema: {
			type: "object",
			properties: {
				sessionId: { type: "string", minLength: 1 },
				idempotencyKey: { type: "string", minLength: 1 },
				agentId: { type: "string" },
				hookType: { type: "string", enum: HOOK_PHASES },
				toolName: { type: "string" },
				toolInput: {},
				toolOutput: {},
				userPrompt: { type: "string" },
				raw: {},
			},
			required: ["sessionId", "idempotencyKey", "hookType"],
			additionalProperties: false,
		},
	},
	{
		name: "memory_feedback",
		description:
			"Record an independently measured outcome for a prior retrieval trace.",
		inputSchema: {
			type: "object",
			properties: {
				retrievalTraceId: { type: "string", minLength: 1 },
				taskId: { type: "string", minLength: 1 },
				idempotencyKey: { type: "string", minLength: 1 },
				trialId: { type: "string" },
				outcome: {
					type: "object",
					properties: {
						environmentPassed: { type: "boolean" },
						corrected: { type: "boolean" },
						reverted: { type: "boolean" },
						unauthorizedSideEffect: { type: "boolean" },
						tokens: { type: "number", minimum: 0 },
						durationMs: { type: "number", minimum: 0 },
					},
					required: ["environmentPassed"],
					additionalProperties: false,
				},
			},
			required: ["retrievalTraceId", "taskId", "idempotencyKey", "outcome"],
			additionalProperties: false,
		},
	},
];

function asObject(value: unknown, name = "arguments"): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`${name} must be an object`);
	}
	return value as Record<string, unknown>;
}

function stringArg(
	args: Record<string, unknown>,
	name: string,
	required = true,
): string | undefined {
	const value = args[name];
	if (value === undefined && !required) return undefined;
	if (typeof value !== "string" || !value.trim()) {
		throw new Error(`${name} must be a non-empty string`);
	}
	return value.trim();
}

function stringsArg(args: Record<string, unknown>, name: string): string[] {
	const value = args[name];
	if (value === undefined) return [];
	if (!Array.isArray(value) || value.some(item => typeof item !== "string")) {
		throw new Error(`${name} must be an array of strings`);
	}
	return [...new Set(value.map(item => item.trim()).filter(Boolean))].slice(
		0,
		20,
	);
}

function boundedInteger(
	value: unknown,
	fallback: number,
	minimum: number,
	maximum: number,
): number {
	if (value === undefined) return fallback;
	if (typeof value !== "number" || !Number.isInteger(value)) {
		throw new Error("expected an integer");
	}
	return Math.max(minimum, Math.min(maximum, value));
}

function stableId(
	kind: "memory" | "observation",
	workspace: string,
	idempotencyKey: string,
): string {
	const digest = createHash("sha256")
		.update(`${workspace}\0${idempotencyKey}`)
		.digest("hex")
		.slice(0, 32);
	return `mcp:${kind}:${digest}`;
}

function brief(value: string, maximum = 220): string {
	const normalized = value.replace(/\s+/g, " ").trim();
	return normalized.length <= maximum
		? normalized
		: `${normalized.slice(0, maximum - 1).trimEnd()}…`;
}

function textResult(text: string, structuredContent?: unknown) {
	return {
		content: [{ type: "text", text }],
		...(structuredContent === undefined ? {} : { structuredContent }),
	};
}

function errorResult(error: unknown) {
	return {
		isError: true,
		content: [
			{
				type: "text",
				text: error instanceof Error ? error.message : String(error),
			},
		],
	};
}

export function createMemoryMcpServer(store: MemoryStore) {
	const workspace = store.getCurrentWorkspace();

	async function callTool(name: string, args: Record<string, unknown>) {
		switch (name) {
			case "memory_search": {
				const query = stringArg(args, "query") as string;
				const limit = boundedInteger(args.limit, 8, 1, 20);
				const retrieval = store.retrieve("mcp", 4000, {
					objective: query,
					maxItems: limit,
				});
				const selected = retrieval.trace.selected.slice(0, limit);
				const stableIds = selected.flatMap(item => {
					const separator = item.id.indexOf(":");
					return separator < 0 || item.type === "summary"
						? []
						: [item.id.slice(separator + 1)];
				});
				const expanded = new Map(
					store.expandEntries(stableIds).map(entry => [entry.id, entry]),
				);
				const claims = new Map(
					store.listClaims({ limit: 1000 }).map(claim => [claim.id, claim]),
				);
				const entries = selected.flatMap<SearchEntry>(item => {
					const id = item.id.slice(item.id.indexOf(":") + 1);
					const entry = expanded.get(id);
					if (entry)
						return [
							{
								id,
								kind: entry.kind,
								type: entry.type,
								title: entry.title,
								summary: brief(entry.content),
								score: item.score,
								reasons: item.reasons,
							},
						];
					const claim = item.type === "claim" ? claims.get(id) : undefined;
					return claim
						? [
								{
									id,
									kind: "claim" as const,
									type: claim.status,
									title: "Claim",
									summary: brief(claim.text),
									score: item.score,
									reasons: item.reasons,
								},
							]
						: [];
				});
				const text = entries.length
					? entries
							.map(
								entry =>
									`- [${entry.id}] ${entry.kind} · ${entry.type} · ${entry.title} — ${entry.summary}`,
							)
							.join("\n")
					: `No memory entries matched "${query}".`;
				return textResult(text, {
					workspace,
					query,
					traceId: retrieval.trace.id,
					abstained: retrieval.trace.abstained,
					entries,
				});
			}
			case "memory_get": {
				const ids = stringsArg(args, "ids");
				if (!ids.length) throw new Error("ids must contain at least one ID");
				const entries = store.expandEntries(ids);
				const found = new Set(entries.map(entry => entry.id));
				const missing = ids.filter(id => !found.has(id));
				return textResult(
					entries.length
						? entries
								.map(entry => `## ${entry.title}\n\n${entry.content}`)
								.join("\n\n---\n\n")
						: "No matching entries found in this workspace.",
					{ workspace, entries, missing },
				);
			}
			case "memory_save": {
				const content = stringArg(args, "content") as string;
				const idempotencyKey = stringArg(args, "idempotencyKey") as string;
				const type = stringArg(args, "type", false);
				if (type && !MEMORY_TYPES.includes(type as MemoryType)) {
					throw new Error(`type must be one of: ${MEMORY_TYPES.join(", ")}`);
				}
				const strength =
					args.strength === undefined
						? undefined
						: boundedInteger(args.strength, 5, 1, 10);
				const sessionId = stringArg(args, "sessionId", false);
				const memory = store.create(content, {
					id: stableId("memory", workspace, idempotencyKey),
					type: type as MemoryType | undefined,
					strength,
					concepts: stringsArg(args, "concepts"),
					files: stringsArg(args, "files"),
					sessionIds: sessionId ? [sessionId] : [],
					workspace,
					project: stringArg(args, "agentId", false),
				});
				return textResult(`Saved memory ${memory.id}.`, { workspace, memory });
			}
			case "memory_observe": {
				const sessionId = stringArg(args, "sessionId") as string;
				const idempotencyKey = stringArg(args, "idempotencyKey") as string;
				const hookType = stringArg(args, "hookType") as HookPhase;
				if (!HOOK_PHASES.includes(hookType)) {
					throw new Error(`hookType must be one of: ${HOOK_PHASES.join(", ")}`);
				}
				const id = stableId("observation", workspace, idempotencyKey);
				const existing = store.getObservation(id, sessionId);
				const raw: RawObservation = {
					id,
					sessionId,
					timestamp: new Date().toISOString(),
					hookType,
					toolName: stringArg(args, "toolName", false),
					toolInput: args.toolInput,
					toolOutput: args.toolOutput,
					userPrompt: stringArg(args, "userPrompt", false),
					workspace,
					raw: args.raw ?? {
						agentId: stringArg(args, "agentId", false),
						toolInput: args.toolInput,
						toolOutput: args.toolOutput,
					},
				};
				const observation = existing || store.observe(raw);
				if (!observation) throw new Error("observation was not persisted");
				return textResult(`Recorded observation ${observation.id}.`, {
					workspace,
					observation,
				});
			}
			case "memory_feedback": {
				const idempotencyKey = stringArg(args, "idempotencyKey") as string;
				const idempotentTrialId = `mcp:${idempotencyKey}`;
				const existing = store
					.listOutcomeReceipts(1_000)
					.find(receipt => receipt.trialId === idempotentTrialId);
				if (existing) {
					return textResult(`Feedback already recorded as ${existing.id}.`, {
						workspace,
						receipt: existing,
					});
				}
				const outcome = asObject(args.outcome, "outcome");
				if (typeof outcome.environmentPassed !== "boolean") {
					throw new Error("outcome.environmentPassed must be a boolean");
				}
				const receipt = store.recordOutcomeReceipt({
					retrievalTraceId: stringArg(args, "retrievalTraceId") as string,
					taskId: stringArg(args, "taskId") as string,
					trialId: idempotentTrialId,
					outcome: {
						environmentPassed: outcome.environmentPassed,
						corrected:
							typeof outcome.corrected === "boolean"
								? outcome.corrected
								: undefined,
						reverted:
							typeof outcome.reverted === "boolean"
								? outcome.reverted
								: undefined,
						unauthorizedSideEffect:
							typeof outcome.unauthorizedSideEffect === "boolean"
								? outcome.unauthorizedSideEffect
								: undefined,
						tokens:
							typeof outcome.tokens === "number" ? outcome.tokens : undefined,
						durationMs:
							typeof outcome.durationMs === "number"
								? outcome.durationMs
								: undefined,
					},
				});
				return textResult(`Recorded feedback ${receipt.id}.`, {
					workspace,
					receipt,
				});
			}
			default:
				throw new Error(`Unknown memory tool: ${name}`);
		}
	}

	return {
		async handle(
			request: JsonRpcRequest,
		): Promise<Record<string, unknown> | null> {
			const id = request.id;
			const method = typeof request.method === "string" ? request.method : "";
			if (id === undefined) return null;
			try {
				let result: Record<string, unknown>;
				switch (method) {
					case "initialize":
						result = {
							protocolVersion: MCP_PROTOCOL_VERSION,
							capabilities: { tools: { listChanged: false } },
							serverInfo: { name: "logician-memory", version: "0.1.0" },
						};
						break;
					case "ping":
						result = {};
						break;
					case "tools/list":
						result = { tools: MEMORY_MCP_TOOLS };
						break;
					case "tools/call": {
						const params = asObject(request.params, "params");
						const name = stringArg(params, "name") as string;
						const args = asObject(params.arguments ?? {});
						try {
							result = await callTool(name, args);
						} catch (error) {
							result = errorResult(error);
						}
						break;
					}
					default:
						return {
							jsonrpc: "2.0",
							id,
							error: { code: -32601, message: `Method not found: ${method}` },
						};
				}
				return { jsonrpc: "2.0", id, result };
			} catch (error) {
				return {
					jsonrpc: "2.0",
					id,
					error: {
						code: -32602,
						message: error instanceof Error ? error.message : String(error),
					},
				};
			}
		},
	};
}
