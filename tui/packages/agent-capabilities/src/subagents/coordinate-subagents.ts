// ── Coordinate Subagents ──────────────────────────────────────────────────────
// coordinate_subagents tool: run multiple tasks in parallel subagents and merge
// their results intelligently. This is useful for complex tasks that can be
// decomposed into independent subtasks that need to be coordinated.

import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LLMBackend } from "@logician/agent-core/core/backend.ts";
import { runAgentLoop } from "@logician/agent-core/core/agent-loop-runner.ts";
import type {
	AgentConfig,
	AgentEvent,
	Message,
	Tool,
} from "@logician/agent-core";
import type { AgentDefinition } from "./subagent.ts";

// ── Options ──────────────────────────────────────────────────────────────────

export interface CoordinateSubagentsOptions {
	/**
	 * Base directory for worktrees. Default: a `.logician/worktrees` dir inside
	 * the repo root.
	 */
	worktreeDir?: string;
	/** Max number of concurrent parallel agents. Excess spawns queue. */
	maxConcurrent?: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Resolve the child tool set: all parent tools except spawn_agent variants. */
function resolveCoordinateChildTools(
	def: AgentDefinition,
	parentTools: Tool[],
): Tool[] {
	return parentTools.filter(
		(t) => t.name !== "spawn_agent" && t.name !== "spawn_agent_parallel" && t.name !== "coordinate_subagents",
	);
}

let coordinateAgentSeq = 0;
const MAX_RESULT_CHARS = 32_000;

function truncateResult(text: string): string {
	return text.length > MAX_RESULT_CHARS
		? `${text.slice(0, MAX_RESULT_CHARS)}\n… [coordinated subagent report truncated]`
		: text;
}

// ── Git worktree management ──────────────────────────────────────────────────

/**
 * Create a git worktree for the given task.
 * Returns { dir, branch } — the worktree path and branch name.
 * The caller must `pruneWorktree(branch)` when done.
 */
async function createWorktree(
	repoDir: string,
	options: CoordinateSubagentsOptions,
): Promise<{ dir: string; branch: string }> {
	const base = options.worktreeDir || join(repoDir, ".logician", "worktrees");
	const seq = ++coordinateAgentSeq;
	const branch = `logician-coordinate-${seq}`;
	const dir = join(base, branch);

	// Ensure the worktree base dir exists
	const { execFile: exec } = await import("node:child_process");
	await new Promise<void>((resolve, reject) => {
		execFile("mkdir", ["-p", base], (err) => {
			if (err) reject(err);
			else resolve();
		});
	});

	// Create the worktree. We use --no-checkout so the worktree is lightweight
	// (no working tree yet), then checkout to populate it.
	await new Promise<void>((resolve, reject) => {
		execFile(
			"git",
			["worktree", "add", "--detach", "--no-checkout", dir, branch],
			{ cwd: repoDir },
			(err) => {
				if (err) reject(new Error(`git worktree add failed: ${err.message}`));
				else resolve();
			},
		);
	});

	// Checkout HEAD to populate the working tree with the current repo state.
	// This gives the child agent the full file tree to work with.
	await new Promise<void>((resolve, reject) => {
		execFile("git", ["checkout", "HEAD"], { cwd: dir }, (err) => {
			if (err)
				reject(new Error(`git checkout in worktree failed: ${err.message}`));
			else resolve();
		});
	});

	// Enable the worktree's git (it should already be valid, but ensure).
	return { dir, branch };
}

/**
 * Remove a worktree and its branch from the repo.
 */
async function pruneWorktree(repoDir: string, branch: string): Promise<void> {
	try {
		await new Promise<void>((resolve, reject) => {
			execFile(
				"git",
				["worktree", "remove", "--force", join(repoDir, branch)],
				(err) => {
					if (err) {
						// Fall back to git worktree prune + branch delete.
						execFile("git", ["worktree", "prune"], { cwd: repoDir }, (err2) => {
							execFile(
								"git",
								["branch", "-D", branch],
								{ cwd: repoDir },
								(err3) => {
									resolve();
								},
							);
						});
					} else {
						resolve();
					}
				},
			);
		});
	} catch {
		// Best-effort cleanup — don't throw if worktree is already gone.
	}
}

// ── coordinate_subagents tool ─────────────────────────────────────────────────

export interface CoordinateSubagentsDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	agents: () => AgentDefinition[];
	emit: (event: AgentEvent) => void;
	options: CoordinateSubagentsOptions;
	defaultMaxIterations?: number;
}

export function createCoordinateSubagentsTool(
	deps: CoordinateSubagentsDeps,
): Tool {
	const repoDir = deps.cwd || process.cwd();

	return {
		name: "coordinate_subagents",
		description:
			"Coordinate multiple subagents to work on independent subtasks concurrently " +
			"and merge their results intelligently. Each subagent gets its own isolated " +
			"git worktree (independent branch + working directory). Returns a merged " +
			"report with all findings. Use this when you need multiple independent " +
			"subtasks to proceed concurrently and their results need to be synthesized. " +
			"Agents: " +
			deps
				.agents()
				.map((a) => `${a.name} (${a.description})`)
				.join("; "),
		parameters: {
			type: "object",
			properties: {
				tasks: {
					type: "array",
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
							id: {
								type: "string",
								description: "Unique identifier for this task (e.g., 'task-a', 'task-b').",
							},
						},
						required: ["task", "id"],
					},
					description: "Array of tasks to coordinate.",
				},
				mergeStrategy: {
					type: "string",
					enum: ["concatenate", "synthesize", "compare"],
					description: "Strategy to merge results: concatenate (simple concat), synthesize (LLM synthesis), compare (highlight differences).",
					default: "synthesize",
				},
			},
			required: ["tasks"],
		},
		execute: async (args, ctx) => {
			const tasksArg = args.tasks;
			if (!Array.isArray(tasksArg) || tasksArg.length === 0) {
				return "Error: coordinate_subagents requires a non-empty 'tasks' array.";
			}

			const mergeStrategy = (args.mergeStrategy as string) || "synthesize";
			const parent = deps.config();

			// Process each task
			const results: Array<{ id: string; agent: string; result: string; worktree: string; branch: string; turns: number; errors?: string; changes?: string }> = [];
			
			// Create worktrees and execute tasks
			const worktreeTasks = await Promise.all(
				tasksArg.map(async (taskArg: any) => {
					const task = typeof taskArg.task === "string" ? taskArg.task : "";
					if (!task.trim()) {
						throw new Error("Each task must have a non-empty 'task' string.");
					}
					const taskId = taskArg.id || `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
					const agentName = typeof taskArg.agent === "string" && taskArg.agent ? taskArg.agent : "general";
					const def = deps.agents().find((a) => a.name === agentName);
					if (!def) {
						const names = deps
							.agents()
							.map((a) => a.name)
							.join(", ");
						throw new Error(`Unknown agent "${agentName}". Available: ${names}`);
					}

					const agentId = `coord_${++coordinateAgentSeq}`;
					deps.emit({ type: "subagent_start", agentId, agent: def.name, task: `${taskId}: ${task}` });

					// Create isolated worktree
					let worktreeDir = "";
					let worktreeBranch = "";
					try {
						const wt = await createWorktree(repoDir, deps.options);
						worktreeDir = wt.dir;
						worktreeBranch = wt.branch;
					} catch (e: unknown) {
						const msg = `Error: failed to create worktree: ${(e as Error).message}`;
						deps.emit({
							type: "subagent_end",
							agentId,
							agent: def.name,
							result: msg,
							isError: true,
						});
						throw new Error(msg);
					}

					// Child config: same as parallel, but cwd points to the worktree.
					let lastText = "";
					const childConfig: AgentConfig = {
						baseUrl: parent.baseUrl,
						model: def.model || parent.model,
						cwd: worktreeDir,
						temperature: parent.temperature,
						maxTokens: parent.maxTokens,
						contextWindowTokens: parent.contextWindowTokens,
						systemPrompt: def.prompt,
						tools: resolveCoordinateChildTools(def, parent.tools ?? []),
						toolExecution: parent.toolExecution,
						thinkingLevel: parent.thinkingLevel,
						autoRetryEnabled: parent.autoRetryEnabled,
						maxRetries: parent.maxRetries,
						turnTimeoutMs: parent.turnTimeoutMs,
						webSearch: parent.webSearch,
						runtimeHooksEnabled: parent.runtimeHooksEnabled,
						continuationEnabled: true,
						onEvent: (event) => {
							if (event.type === "text_delta") {
								lastText += event.delta;
								ctx.onUpdate?.(truncateResult(lastText));
							}
							if (event.type === "message_start") lastText = "";
							deps.emit({ type: "subagent_event", agentId, event });
						},
					};

					const backend = def.model
						? deps.backend.withModel(def.model)
						: deps.backend;
					let turns = 0;

					try {
						const newMessages = await runAgentLoop(
							{
								systemPrompt: childConfig.systemPrompt,
								messages: [],
								tools: childConfig.tools,
								cwd: worktreeDir,
							},
							[{ role: "user", content: task } satisfies Message],
							{
								...childConfig,
								backend,
								maxIterations: def.maxIterations ?? deps.defaultMaxIterations ?? 15,
								signal: ctx.signal,
							},
							(event) => {
								if (event.type === "turn_start") turns++;
								childConfig.onEvent?.(event);
							},
						);
						const final = [...newMessages]
							.reverse()
							.find((m) => m.role === "assistant" && m.content?.trim());
						const result = truncateResult(
							final?.content?.trim() ||
								`(coordinated subagent ${taskId} produced no final message)`,
						);

						// Capture worktree diff for the caller to review/apply later.
						let diff = "";
						try {
							const { stdout } = await new Promise<{ stdout: string }>(
								(resolve, reject) => {
									execFile(
										"git",
										["diff", "--stat", "HEAD"],
										{ cwd: worktreeDir, maxBuffer: 10 * 1024 * 1024 },
										(err, stdout, stderr) => {
											if (err) reject(err);
											else resolve({ stdout });
										},
									);
								},
							);
							diff = stdout;
						} catch {
							// Best-effort — diff is informational only.
						}

						deps.emit({
							type: "subagent_end",
							agentId,
							agent: def.name,
							result,
							turns,
						});

						return {
							id: taskId,
							agent: def.name,
							result,
							worktree: worktreeDir,
							branch: worktreeBranch,
							turns,
							changes: diff || "(no diff)",
						};
					} catch (e: unknown) {
						const message = `Error: coordinated subagent ${taskId} failed: ${(e as Error).message}`;
						deps.emit({
							type: "subagent_end",
							agentId,
							agent: def.name,
							result: message,
							isError: true,
						});
						return {
							id: taskId,
							agent: def.name,
							result: message,
							worktree: worktreeDir,
							branch: worktreeBranch,
							turns,
							errors: message,
						};
					}
				})
			);

			// Store results
			for (const wt of worktreeTasks) {
				const resultEntry: any = {
					id: wt.id,
					agent: wt.agent,
					result: wt.result,
					worktree: wt.worktree,
					branch: wt.branch,
					turns: wt.turns,
				};
				if (wt.errors) resultEntry.errors = wt.errors;
				if (wt.changes) resultEntry.changes = wt.changes;
				results.push(resultEntry);
			}

			// Merge results based on strategy
			let mergedResult = "";
			if (mergeStrategy === "concatenate") {
				mergedResult = results.map(r => 
					`--- Task: ${r.id} (Agent: ${r.agent}) ---\n${r.result}\n`
				).join('\n\n');
			} else if (mergeStrategy === "synthesize") {
				// Use LLM to synthesize results
				const synthesisPrompt = `You are a coordinator synthesizing results from multiple subagents working on independent tasks.\n\nHere are the results from the subagents:\n\n${results.map(r => 
					`--- Task: ${r.id} (Agent: ${r.agent}) ---\n${r.result}\n`
				).join('\n\n')}\n\nPlease synthesize these results into a coherent, comprehensive report. Highlight any conflicts or important findings.`;
				
				const synthesisResponse = await deps.backend.generate([
					{ role: "user", content: synthesisPrompt }
				], { temperature: 0.3, maxTokens: 2048 });
				
				mergedResult = synthesisResponse.content?.trim() || "Failed to synthesize results.";
			} else if (mergeStrategy === "compare") {
				// Highlight differences and similarities
				mergedResult = "Comparison of subagent results:\n\n";
				for (const r of results) {
					mergedResult += `--- Task: ${r.id} (Agent: ${r.agent}) ---\n`;
					mergedResult += `Result:\n${r.result}\n\n`;
					if (r.errors) {
						mergedResult += `Errors: ${r.errors}\n\n`;
					}
					if (r.changes) {
						mergedResult += `Changes: ${r.changes}\n\n`;
					}
				}
			}

			// Clean up all worktrees
			for (const r of results) {
				try {
					await pruneWorktree(repoDir, r.branch);
				} catch {
					// Best-effort cleanup
				}
			}

			return {
				content: mergedResult,
				details: {
					mergeStrategy,
					taskCount: results.length,
					tasks: results.map(r => ({
						id: r.id,
						agent: r.agent,
						success: !r.errors,
						turns: r.turns
					})),
				},
			};
		},
	};
}