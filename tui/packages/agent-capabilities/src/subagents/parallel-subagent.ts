// ── Parallel Subagents ────────────────────────────────────────────────────────
// spawn_agent_parallel tool: run a task in a child agent with its own git
// worktree, so the child can freely edit files without blocking or conflicting
// with the parent or other parallel children.
//
// Each parallel spawn creates a new branch + working directory via
// `git worktree add`. The child runs in that worktree, edits freely, and on
// completion the worktree is pruned (leaving the parent repo pristine).
//
// Use this when the parent needs to delegate multiple independent edits — e.g.
// refactor A, add B, fix C — concurrently.

import { execFile } from "node:child_process";
import { join } from "node:path";
import type { LLMBackend } from "@logician/agent-core/core/backend.ts";
import type {
	AgentConfig,
	AgentEvent,
	Message,
	Tool,
} from "@logician/agent-core";
import type { AgentDefinition } from "./subagent.ts";
import {
	budgetFromArgs,
	contractFromArgs,
	DELEGATION_CONTRACT_PROPERTIES,
	runDelegatedAgent,
} from "./delegation-runtime.ts";

// ── Options ──────────────────────────────────────────────────────────────────

export interface ParallelSpawnOptions {
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
function resolveParallelChildTools(
	def: AgentDefinition,
	parentTools: Tool[],
): Tool[] {
	return parentTools.filter(
		(t) => t.name !== "spawn_agent" && t.name !== "spawn_agent_parallel",
	);
}

let parallelAgentSeq = 0;
const MAX_RESULT_CHARS = 16_000;

function truncateResult(text: string): string {
	return text.length > MAX_RESULT_CHARS
		? `${text.slice(0, MAX_RESULT_CHARS)}\n… [parallel subagent report truncated]`
		: text;
}

// ── Git worktree management ──────────────────────────────────────────────────

/**
 * Create a git worktree for the given task.
 * Returns { dir, branch } — the worktree path and branch name.
 * The caller must `pruneWorktree(repoDir, dir, branch)` when done.
 */
async function createWorktree(
	repoDir: string,
	options: ParallelSpawnOptions,
): Promise<{ dir: string; branch: string }> {
	const base = options.worktreeDir || join(repoDir, ".logician", "worktrees");
	const seq = ++parallelAgentSeq;
	const branch = `logician-parallel-${seq}`;
	const dir = join(base, branch);

	// Ensure the worktree base dir exists
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
			["worktree", "add", "-b", branch, "--no-checkout", dir, "HEAD"],
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
 * Remove a worktree while preserving its branch. Successful child changes are
 * committed to that branch before cleanup so the parent can inspect or merge it.
 */
async function pruneWorktree(
	repoDir: string,
	dir: string,
): Promise<void> {
	try {
		await new Promise<void>((resolve) => {
			execFile(
				"git",
				["worktree", "remove", "--force", dir],
				(err) => {
					if (err) {
						// Fall back to pruning stale worktree metadata. Never delete the
						// branch: it is the durable hand-off from child to parent.
						execFile("git", ["worktree", "prune"], { cwd: repoDir }, () => {
							resolve();
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

// ── spawn_agent_parallel tool ─────────────────────────────────────────────────

export interface ParallelSpawnAgentDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	agents: () => AgentDefinition[];
	emit: (event: AgentEvent) => void;
	options: ParallelSpawnOptions;
	defaultMaxIterations?: number;
}

export function createParallelSpawnAgentTool(
	deps: ParallelSpawnAgentDeps,
): Tool {
	const repoDir = deps.cwd || process.cwd();
	const maxConcurrent = Math.max(1, deps.options.maxConcurrent ?? 2);
	let active = 0;
	const waiters: Array<() => void> = [];
	const acquire = async (): Promise<void> => {
		if (active >= maxConcurrent) {
			await new Promise<void>((resolve) => waiters.push(resolve));
		}
		active++;
	};
	const release = (): void => {
		active--;
		waiters.shift()?.();
	};

	return {
		name: "spawn_agent_parallel",
		description:
			"Delegate a self-contained task to a subagent with its own isolated git " +
			"worktree (independent branch + working directory). The subagent can freely " +
			"edit, create, and delete files without affecting the parent repo or other " +
			"parallel agents. Returns the final report. Use this when you need multiple " +
			"independent subtasks to proceed concurrently — e.g. refactor module A, add " +
			"feature B, and fix bug C in parallel. Agents: " +
			deps
				.agents()
				.map((a) => `${a.name} (${a.description})`)
				.join("; "),
		parameters: {
			type: "object",
			properties: {
				task: {
					type: "string",
					description:
						"Complete, self-contained task prompt. The subagent sees the full " +
						"repo at HEAD in its isolated worktree.",
				},
				agent: {
					type: "string",
					description: "Agent definition to use (default: general).",
				},
				...DELEGATION_CONTRACT_PROPERTIES,
			},
			required: ["task"],
		},
			execute: async (args, ctx) => {
			const task = typeof args.task === "string" ? args.task : "";
			if (!task.trim()) return "Error: spawn_agent_parallel requires a task.";

			const agentName =
				typeof args.agent === "string" && args.agent ? args.agent : "general";
			const def = deps.agents().find((a) => a.name === agentName);
			if (!def) {
				const names = deps
					.agents()
					.map((a) => a.name)
					.join(", ");
				return `Error: Unknown agent "${agentName}". Available: ${names}`;
			}

			const agentId = `par_${++parallelAgentSeq}`;
			const parent = deps.config();
			await acquire();

			deps.emit({ type: "subagent_start", agentId, agent: def.name, task });

			// Create isolated worktree
			let worktreeDir = "";
			let worktreeBranch = "";
			try {
				const wt = await createWorktree(repoDir, deps.options);
				worktreeDir = wt.dir;
				worktreeBranch = wt.branch;
			} catch (e: unknown) {
				release();
				const msg = `Error: failed to create worktree: ${(e as Error).message}`;
				deps.emit({
					type: "subagent_end",
					agentId,
					agent: def.name,
					result: msg,
					isError: true,
				});
				return msg;
			}

			// Child config: same as sequential, but cwd points to the worktree.
			let lastText = "";
			const childConfig: AgentConfig = {
				baseUrl: parent.baseUrl,
				model: def.model || parent.model,
				cwd: worktreeDir,
				temperature: parent.temperature,
				maxTokens: parent.maxTokens,
				contextWindowTokens: parent.contextWindowTokens,
				systemPrompt: def.prompt,
				tools: resolveParallelChildTools(def, parent.tools ?? []),
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
					onEvent: (event) => childConfig.onEvent?.(event),
				});
				const result = truncateResult(run.content);

				// Commit every tracked and untracked child change so removing the
				// worktree cannot destroy the result. The branch is the hand-off.
				let commit: string | undefined;
				try {
					await new Promise<void>((resolve, reject) => {
						execFile("git", ["add", "-A"], { cwd: worktreeDir }, (error) =>
							error ? reject(error) : resolve(),
						);
					});
					await new Promise<void>((resolve, reject) => {
						execFile(
							"git",
							[
								"-c", "user.name=Logician Agent",
								"-c", "user.email=logician@local",
								"commit", "--allow-empty", "-m", `logician: parallel agent ${agentId}`,
							],
							{ cwd: worktreeDir },
							(error) => error ? reject(error) : resolve(),
						);
					});
					commit = await new Promise<string>((resolve, reject) => {
						execFile("git", ["rev-parse", "HEAD"], { cwd: worktreeDir }, (error, stdout) =>
							error ? reject(error) : resolve(stdout.trim()),
						);
					});
				} catch {
					// Leave the worktree registered if durable hand-off failed. Cleanup
					// below only runs after the result has been assembled successfully.
				}

				// Capture committed change summary for the caller.
				let diff = "";
				try {
					const { stdout } = await new Promise<{ stdout: string }>(
						(resolve, reject) => {
							execFile(
								"git",
								["diff", "--stat", "HEAD^", "HEAD"],
								{ cwd: worktreeDir, maxBuffer: 10 * 1024 * 1024 },
								(err, stdout) => {
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
					turns: run.turns,
					isError: run.status !== "completed",
				});

				return {
					content: result,
					details: {
						agent: def.name,
						agentId,
						worktree: worktreeDir,
						branch: worktreeBranch,
						commit,
						status: run.status,
						metrics: {
							turns: run.turns,
							durationMs: run.durationMs,
							toolCalls: run.toolCalls,
							toolCallsByName: run.toolCallsByName,
							validationAttempts: run.validationAttempts,
						},
						acceptance: run.acceptance,
						changes: diff || "(no diff)",
					},
				};
			} catch (e: unknown) {
				const message = `Error: parallel subagent failed: ${(e as Error).message}`;
				deps.emit({
					type: "subagent_end",
					agentId,
					agent: def.name,
					result: message,
					isError: true,
				});
				return message;
			} finally {
				// Only remove worktrees whose changes reached a durable branch.
				// A failed commit leaves the worktree available for recovery.
				try {
					const clean = await new Promise<boolean>((resolve) => {
						execFile("git", ["status", "--porcelain"], { cwd: worktreeDir }, (error, stdout) =>
							resolve(!error && stdout.trim().length === 0),
						);
					});
					if (clean) await pruneWorktree(repoDir, worktreeDir);
				} finally {
					release();
				}
			}
		},
	};
}
