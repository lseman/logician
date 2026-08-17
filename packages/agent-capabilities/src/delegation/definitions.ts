// ── Subagents ────────────────────────────────────────────────────────────────
// spawn_agent tool: run a task in a child functional agent runner with its own context
// window and a scoped tool set, returning only the child's final message to
// the parent. This is the main context-economy lever — explorations and
// reviews burn the child's window, not the parent's.
//
// Agent definitions are markdown files with YAML frontmatter (mirroring
// skills): name, description, optional tools allowlist, model, max-turns. The
// body becomes the child's system prompt. Two built-in agents always exist:
// "general" (all parent tools) and "explorer" (read-only tools).

import { randomUUID } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import {
	type AgentConfig,
	type AgentEvent,
	DEFAULT_TRUNCATION,
	type Tool,
	type ToolResult,
} from "@logician/agent-core";
import type { LLMBackend } from "@logician/agent-core/agent/core/backend.ts";
import { parseFrontmatter } from "@logician/agent-core/tools/shared/frontmatter.ts";
import {
	budgetFromArgs,
	contractFromArgs,
	DELEGATION_CONTRACT_PROPERTIES,
	runDelegatedAgent,
	type SpawnAgentsTask,
} from "./runtime.ts";

// ── Agent definitions ────────────────────────────────────────────────────────

export interface AgentDefinition {
	name: string;
	description: string;
	/** System prompt body. Empty = generic subagent prompt. */
	prompt: string;
	/** Tool-name allowlist. Empty/undefined = all parent tools. */
	tools?: string[];
	/** Model override (must be in the parent's model list to take effect). */
	model?: string;
	maxIterations?: number;
	maxExecutionTimeMs?: number;
	maxToolCalls?: number;
	toolLimits?: Record<string, number>;
}

const GENERIC_SUBAGENT_PROMPT =
	"You are a subagent completing one delegated task. Work autonomously — " +
	"you cannot ask the user questions. When done, end with a final message " +
	"that fully reports your findings/results: it is the ONLY thing returned " +
	"to the caller, so include every detail that matters (paths, names, " +
	"conclusions). Do not pad it with process narration. The message content " +
	"in the SAME turn as your task_status call is what gets returned — a " +
	'closing line like "the task is complete" with no restated findings ' +
	"returns nothing useful. Restate the actual result there, not just that " +
	"you finished.";

export const BUILTIN_AGENTS: AgentDefinition[] = [
	{
		name: "general",
		description:
			"General-purpose agent with the full tool set. Use for multi-step " +
			"subtasks that need both reading and editing.",
		prompt: GENERIC_SUBAGENT_PROMPT,
	},
	{
		name: "explorer",
		description:
			"Read-only researcher. Locates code, maps structure, answers " +
			'"where/how is X done" questions without modifying anything.',
		prompt:
			`${GENERIC_SUBAGENT_PROMPT}\n\nYou are read-only: explore and report, ` +
			"never modify. Report concrete file paths and line references.",
		// Resolved against the parent registry by readOnly flag at spawn time.
		tools: ["__read_only__"],
	},
];

interface AgentFrontmatter {
	name?: string;
	description?: string;
	tools?: string[] | string;
	model?: string;
	"max-turns"?: number;
	"max-execution-seconds"?: number;
	"max-tool-calls"?: number;
	"tool-limits"?: Record<string, number>;
	[key: string]: unknown;
}

/**
 * Load agent definitions from directories of markdown files (one agent per
 * .md file). Invalid files are skipped silently — agents are optional sugar
 * on top of the built-ins. Later directories and files override earlier ones
 * (and built-ins) by name.
 */
export async function loadAgentDefinitions(
	dirs: string[],
): Promise<AgentDefinition[]> {
	const byName = new Map<string, AgentDefinition>(
		BUILTIN_AGENTS.map(a => [a.name, a]),
	);
	for (const dir of dirs) {
		let entries: string[];
		try {
			entries = await readdir(dir);
		} catch {
			continue;
		}
		for (const entry of entries.sort()) {
			if (!entry.endsWith(".md")) continue;
			try {
				const raw = await readFile(join(dir, entry), "utf8");
				const parsed = parseFrontmatter<AgentFrontmatter>(raw);
				if (!parsed.ok) continue;
				const { frontmatter, body } = parsed.value;
				const name =
					typeof frontmatter.name === "string"
						? frontmatter.name
						: entry.replace(/\.md$/, "");
				const description =
					typeof frontmatter.description === "string"
						? frontmatter.description
						: "";
				if (!name || !description) continue;
				const tools = Array.isArray(frontmatter.tools)
					? frontmatter.tools.map(String)
					: typeof frontmatter.tools === "string"
						? frontmatter.tools.split(",").map(t => t.trim())
						: undefined;
				byName.set(name, {
					name,
					description,
					prompt: body.trim() || GENERIC_SUBAGENT_PROMPT,
					tools: tools?.filter(Boolean),
					model:
						typeof frontmatter.model === "string"
							? frontmatter.model
							: undefined,
					maxIterations:
						typeof frontmatter["max-turns"] === "number"
							? frontmatter["max-turns"]
							: undefined,
					maxExecutionTimeMs:
						typeof frontmatter["max-execution-seconds"] === "number"
							? frontmatter["max-execution-seconds"] * 1000
							: undefined,
					maxToolCalls:
						typeof frontmatter["max-tool-calls"] === "number"
							? frontmatter["max-tool-calls"]
							: undefined,
					toolLimits:
						typeof frontmatter["tool-limits"] === "object" &&
						frontmatter["tool-limits"] !== null
							? frontmatter["tool-limits"]
							: undefined,
				});
			} catch {
				// Skip unreadable definitions.
			}
		}
	}
	return [...byName.values()];
}

// ── spawn_agent tool ─────────────────────────────────────────────────────────

export interface SpawnAgentDeps {
	/** Parent config snapshot provider (read at spawn time, not build time). */
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	/** Available agent definitions (read at spawn time). */
	agents: () => AgentDefinition[];
	/** Receives child events wrapped as subagent_* on the parent stream. */
	emit: (event: AgentEvent) => void;
	/** Cap on the child's loop iterations when the definition sets none. */
	defaultMaxIterations?: number;
	/** Session-local concurrency limiter shared by both subagent tools. */
	concurrencyLimiter?: SubagentConcurrencyLimiter;
}

// ── Session-local concurrency limiter ───────────────────────────────────────

export interface SpawnCtx {
	signal?: AbortSignal;
	onUpdate?: (delta: string) => void;
	/** Position within a spawn_agents batch, if run as part of one. */
	taskIndex?: number;
}

interface _PendingSpawn {
	resolve: () => void;
	reject: (error: Error) => void;
	signal?: AbortSignal;
	onAbort?: () => void;
}

const DEFAULT_CONCURRENCY_LIMIT = 4;

export interface SubagentConcurrencyLimiter {
	run<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T>;
}

export function createSubagentConcurrencyLimiter(
	maxParallelAgents: number | undefined,
): SubagentConcurrencyLimiter {
	let limit = DEFAULT_CONCURRENCY_LIMIT;
	if (
		typeof maxParallelAgents === "number" &&
		Number.isInteger(maxParallelAgents) &&
		maxParallelAgents > 0
	) {
		limit = maxParallelAgents;
	}
	let activeCount = 0;
	const pendingQueue: _PendingSpawn[] = [];

	const drainQueue = (): void => {
		while (pendingQueue.length > 0 && activeCount < limit) {
			const next = pendingQueue.shift();
			if (!next) return;
			if (next.signal?.aborted) {
				next.reject(
					new DOMException("Subagent spawn cancelled.", "AbortError"),
				);
				continue;
			}
			if (next.onAbort && next.signal) {
				next.signal.removeEventListener("abort", next.onAbort);
			}
			activeCount++;
			next.resolve();
		}
	};

	const acquire = async (signal?: AbortSignal): Promise<void> => {
		if (signal?.aborted) {
			throw new DOMException("Subagent spawn cancelled.", "AbortError");
		}
		if (activeCount < limit) {
			activeCount++;
			return;
		}
		await new Promise<void>((resolve, reject) => {
			const pending: _PendingSpawn = { resolve, reject, signal };
			if (signal) {
				pending.onAbort = () => {
					const index = pendingQueue.indexOf(pending);
					if (index >= 0) pendingQueue.splice(index, 1);
					reject(new DOMException("Subagent spawn cancelled.", "AbortError"));
				};
				signal.addEventListener("abort", pending.onAbort, { once: true });
			}
			pendingQueue.push(pending);
		});
	};

	return {
		async run<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
			await acquire(signal);
			try {
				return await work();
			} finally {
				activeCount--;
				drainQueue();
			}
		},
	};
}

function limiterFor(deps: SpawnAgentDeps): SubagentConcurrencyLimiter {
	if (!deps.concurrencyLimiter)
		deps.concurrencyLimiter = createSubagentConcurrencyLimiter(undefined);
	return deps.concurrencyLimiter;
}

async function _runSpawn(
	args: Record<string, unknown>,
	ctx: SpawnCtx,
	deps: SpawnAgentDeps,
): Promise<string | ToolResult> {
	const task = typeof args.task === "string" ? args.task : "";
	const agentName =
		typeof args.agent === "string" && args.agent ? args.agent : "general";
	const def = deps.agents().find(a => a.name === agentName);
	if (!def) {
		const names = deps
			.agents()
			.map(a => a.name)
			.join(", ");
		return `Error: Unknown agent "${agentName}". Available: ${names}`;
	}

	const agentId = `agent_${randomUUID()}`;
	const parent = deps.config();
	deps.emit({
		type: "subagent_start",
		agentId,
		agent: def.name,
		task,
		taskIndex: ctx.taskIndex,
	});

	// Child config: parent's provider settings, but its own prompt, scoped
	// tools, and NO parent hooks/queues/events — the child is isolated.
	const childConfig: AgentConfig = {
		baseUrl: parent.baseUrl,
		model: def.model || parent.model,
		cwd: deps.cwd ?? parent.cwd,
		temperature: parent.temperature,
		maxTokens: parent.maxTokens,
		contextWindowTokens: parent.contextWindowTokens,
		systemPrompt: def.prompt,
		tools: resolveChildTools(def, parent.tools ?? []),
		toolExecution: parent.toolExecution,
		permissions: parent.permissions,
		onPermissionRequest: parent.onPermissionRequest,
		allowedPaths: parent.allowedPaths,
		allowAllPaths: parent.allowAllPaths,
		hooks: parent.hooks,
		thinkingLevel: parent.thinkingLevel,
		autoRetryEnabled: parent.autoRetryEnabled,
		maxRetries: parent.maxRetries,
		turnTimeoutMs: parent.turnTimeoutMs,
		webSearch: parent.webSearch,
		truncation: parent.truncation,
		runBudget: parent.runBudget,
		maxTotalTokens: parent.maxTotalTokens,
		// External settings-hooks (PreToolUse shell hooks etc.) stay enabled
		// so safety hooks also govern subagent tool use.
		runtimeHooksEnabled: parent.runtimeHooksEnabled,
		// Enable continuation so the child model doesn't abandon mid-task.
		// Subagents frequently get a premature "stop" — the default nudge
		// "You stopped without completing your work. Continue." is enough
		// to push them past that. Bounded by maxIterations on the child.
		continuationEnabled: true,
		onEvent: event => {
			if (event.type === "text_delta") {
				ctx.onUpdate?.(event.delta);
			}
			deps.emit({
				type: "subagent_event",
				agentId,
				event,
				taskIndex: ctx.taskIndex,
			});
		},
	};

	const backend = def.model ? deps.backend.withModel(def.model) : deps.backend;

	try {
		const run = await runDelegatedAgent({
			task,
			config: childConfig,
			backend,
			tools: childConfig.tools ?? [],
			maxIterations: def.maxIterations ?? deps.defaultMaxIterations ?? 15,
			signal: ctx.signal,
			contract: contractFromArgs(args),
			budget: budgetFromArgs(args, {
				timeoutMs: def.maxExecutionTimeMs,
				maxToolCalls: def.maxToolCalls,
				toolLimits: def.toolLimits,
			}),
			onEvent: event => childConfig.onEvent?.(event),
		});
		const result = truncateResult(
			run.content,
			parent.truncation?.subagentResultMaxChars ??
				DEFAULT_TRUNCATION.subagentResultMaxChars,
		);
		deps.emit({
			type: "subagent_end",
			agentId,
			agent: def.name,
			result,
			turns: run.turns,
			isError: run.status !== "completed",
			taskIndex: ctx.taskIndex,
		});
		return {
			content: result,
			isError: run.status !== "completed",
			details: {
				agent: def.name,
				agentId,
				status: run.status,
				metrics: {
					turns: run.turns,
					durationMs: run.durationMs,
					toolCalls: run.toolCalls,
					toolCallsByName: run.toolCallsByName,
					validationAttempts: run.validationAttempts,
				},
				acceptance: run.acceptance,
			},
		};
	} catch (err: unknown) {
		const message = `Error: subagent failed: ${(err as Error).message}`;
		deps.emit({
			type: "subagent_end",
			agentId,
			agent: def.name,
			result: message,
			isError: true,
			taskIndex: ctx.taskIndex,
		});
		return { content: message, isError: true };
	}
}

/** Resolve the child tool set: allowlist (or all), never subagent spawners. */
function resolveChildTools(def: AgentDefinition, parentTools: Tool[]): Tool[] {
	const base = parentTools.filter(
		t => t.name !== "spawn_agent" && t.name !== "spawn_agents",
	);
	if (!def.tools?.length) return base;
	if (def.tools.includes("__read_only__")) {
		return base.filter(t => t.readOnly === true);
	}
	const allowed = new Set(def.tools);
	return base.filter(t => allowed.has(t.name));
}

export function createSpawnAgentTool(deps: SpawnAgentDeps): Tool {
	return {
		name: "spawn_agent",
		description:
			"Delegate a self-contained task to a subagent with its own context " +
			"window. The subagent works autonomously and returns only its final " +
			"report — use it for explorations, reviews, or subtasks whose " +
			"intermediate output would waste your context. The task prompt must " +
			"be fully self-contained (the subagent sees none of this " +
			"conversation). Agents: " +
			deps
				.agents()
				.map(a => `${a.name} (${a.description})`)
				.join("; "),
		parameters: {
			type: "object",
			properties: {
				task: {
					type: "string",
					description:
						"Complete, self-contained task prompt, including every needed " +
						"path/constraint and what the final report must contain.",
				},
				agent: {
					type: "string",
					description:
						"Agent definition to use (default: general). Use explorer for " +
						"read-only research.",
				},
				...DELEGATION_CONTRACT_PROPERTIES,
			},
			required: ["task"],
		},
		execute: async (args, ctx) => {
			// Fast path: validation before any queuing.
			const task = typeof args.task === "string" ? args.task : "";
			if (!task.trim()) return "Error: spawn_agent requires a task.";
			const agentName =
				typeof args.agent === "string" && args.agent ? args.agent : "general";
			const def = deps.agents().find(a => a.name === agentName);
			if (!def) {
				const names = deps
					.agents()
					.map(a => a.name)
					.join(", ");
				return `Error: Unknown agent "${agentName}". Available: ${names}`;
			}

			return limiterFor(deps).run(
				() =>
					_runSpawn(args, { signal: ctx.signal, onUpdate: ctx.onUpdate }, deps),
				ctx.signal,
			);
		},
	};
}

function truncateResult(text: string, maxChars: number): string {
	return text.length > maxChars
		? `${text.slice(0, maxChars)}\n… [subagent report truncated]`
		: text;
}

// ── spawn_agents tool ────────────────────────────────────────────────────────
// Spawns multiple subagents concurrently, bounded by maxParallelAgents.

export function createSpawnAgentsTool(deps: SpawnAgentDeps): Tool {
	return {
		name: "spawn_agents",
		description:
			"Spawn multiple subagents concurrently, bounded by maxParallelAgents. " +
			"Returns results in the same order as the input tasks. " +
			"Agents: " +
			deps
				.agents()
				.map(a => `${a.name} (${a.description})`)
				.join("; "),
		parameters: {
			type: "object",
			properties: {
				tasks: {
					type: "array",
					minItems: 2,
					items: {
						type: "object",
						properties: {
							task: {
								type: "string",
								description: "Complete, self-contained task prompt.",
							},
							agent: {
								type: "string",
								description: "Agent definition to use (default: general).",
							},
							expected_output: {
								type: "string",
								description:
									"Concrete shape and contents required in the final result.",
							},
							success_criteria: {
								type: "array",
								items: { type: "string" },
								description:
									"Criteria the subagent must explicitly satisfy with evidence.",
							},
							max_validation_retries: {
								type: "integer",
								minimum: 0,
								maximum: 5,
								description:
									"Correction attempts after contract validation fails (default: 2).",
							},
							timeout_ms: {
								type: "integer",
								minimum: 1000,
								description:
									"Whole-task deadline including tools and validation retries.",
							},
							max_tool_calls: {
								type: "integer",
								minimum: 1,
								description:
									"Maximum total tool calls allowed for this delegated task.",
							},
						},
						required: ["task"],
					},
					description: "Array of tasks to execute concurrently.",
				},
			},
			required: ["tasks"],
		},
		execute: async (args, ctx) => {
			const rawTasks = args.tasks;
			if (!Array.isArray(rawTasks) || rawTasks.length < 2) {
				return "Error: spawn_agents requires at least two tasks.";
			}

			// Validate all tasks first.
			const agents = deps.agents();
			const parsedTasks: SpawnAgentsTask[] = [];
			for (let i = 0; i < rawTasks.length; i++) {
				const raw = rawTasks[i];
				if (
					typeof raw !== "object" ||
					raw === null ||
					typeof raw.task !== "string" ||
					!raw.task.trim()
				) {
					return `Error: tasks[${i}] is invalid. Each task must have a non-empty 'task' string.`;
				}
				const agentName =
					typeof raw.agent === "string" && raw.agent ? raw.agent : "general";
				if (!agents.find(a => a.name === agentName)) {
					const names = agents.map(a => a.name).join(", ");
					return `Error: tasks[${i}] uses unknown agent "${agentName}". Available: ${names}`;
				}
				parsedTasks.push({
					task: raw.task,
					agent: agentName,
					expected_output: raw.expected_output,
					success_criteria: raw.success_criteria,
					max_validation_retries: raw.max_validation_retries,
					timeout_ms: raw.timeout_ms,
					max_tool_calls: raw.max_tool_calls,
				});
			}

			// Build per-task spawn args for _runSpawn.
			const taskArgs: Array<{
				spawnArgs: Record<string, unknown>;
				taskIndex: number;
				task: SpawnAgentsTask;
			}> = parsedTasks.map((task, i) => ({
				spawnArgs: {
					task: task.task,
					agent: task.agent,
					expected_output: task.expected_output,
					success_criteria: task.success_criteria,
					max_validation_retries: task.max_validation_retries,
					timeout_ms: task.timeout_ms,
					max_tool_calls: task.max_tool_calls,
				},
				taskIndex: i,
				task,
			}));

			// Execute through the limiter owned by this runtime/session.
			const results: Array<{
				index: number;
				result: string | ToolResult;
				isError: boolean;
			}> = [];

			const runOne = async (item: (typeof taskArgs)[number]) => {
				try {
					const result = await limiterFor(deps).run(
						() =>
							_runSpawn(
								item.spawnArgs,
								{
									signal: ctx.signal,
									taskIndex: item.taskIndex,
								},
								deps,
							),
						ctx.signal,
					);
					const isError = typeof result !== "string" && result.isError === true;
					results.push({
						index: item.taskIndex,
						result,
						isError,
					});
				} catch (error) {
					// Failed before _runSpawn could emit its own subagent_end (e.g.
					// the concurrency limiter's queue was aborted) — emit one here
					// so the TUI's per-task status doesn't hang on "running".
					deps.emit({
						type: "subagent_end",
						agentId: `agent_task${item.taskIndex}_failed`,
						agent: item.task.agent || "general",
						result: `Error: ${error instanceof Error ? error.message : String(error)}`,
						isError: true,
						taskIndex: item.taskIndex,
					});
					results.push({
						index: item.taskIndex,
						result: `Error: ${error instanceof Error ? error.message : String(error)}`,
						isError: true,
					});
				}
			};

			// Launch all tasks; the session-local limiter enforces the cap.
			await Promise.all(taskArgs.map(runOne));

			// Sort results by original index and return.
			results.sort((a, b) => a.index - b.index);
			const toolResults = results.map(r => {
				if (typeof r.result === "string") {
					return {
						index: r.index,
						content: r.result,
						isError: r.isError,
					};
				}
				const tr = r.result as ToolResult;
				return {
					index: r.index,
					content: tr.content,
					isError: tr.isError,
					details: tr.details,
				};
			});
			const content = toolResults
				.map(result => {
					const task = parsedTasks[result.index];
					const agent = task?.agent || "general";
					const status = result.isError ? "failed" : "completed";
					const report = result.content.trim() || "(No final report returned.)";
					return [
						`## Subagent ${result.index + 1}: ${agent} (${status})`,
						report,
					].join("\n");
				})
				.join("\n\n");
			return {
				content,
				details: {
					results: toolResults,
					total: results.length,
					completed: results.filter(r => !r.isError).length,
					failed: results.filter(r => r.isError).length,
				},
			};
		},
	};
}
