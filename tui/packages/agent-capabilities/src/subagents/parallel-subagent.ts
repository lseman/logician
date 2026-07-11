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
 * Remove a worktree and its branch from the repo.
 */
async function pruneWorktree(
	repoDir: string,
	dir: string,
	branch: string,
): Promise<void> {
	try {
		await new Promise<void>((resolve, reject) => {
			execFile(
				"git",
				["worktree", "remove", "--force", dir],
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

			deps.emit({ type: "subagent_start", agentId, agent: def.name, task });

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
						"(parallel subagent produced no final message)",
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
					content: result,
					details: {
						agent: def.name,
						agentId,
						worktree: worktreeDir,
						branch: worktreeBranch,
						metrics: { turns },
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
				// Always clean up the worktree.
				await pruneWorktree(repoDir, worktreeDir, worktreeBranch);
			}
		},
	};
}
