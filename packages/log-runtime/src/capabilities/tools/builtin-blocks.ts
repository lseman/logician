// ── Built-in Tools Registry ──────────────────────────────────────────────────
// Returns all built-in tools for automatic registration at startup.

import type {
	AgentConfig,
	AgentEvent,
	LLMBackend,
	Tool,
} from "@logician/log-core";
import {
	createSpawnAgentsTool,
	createSpawnAgentTool,
	createSubagentConcurrencyLimiter,
	type SpawnAgentDeps,
} from "../delegation/definitions.ts";
import { ask_user } from "../ask-user/index.ts";
import { rag_tools } from "../rag/index.ts";
import { todo_tool } from "../tasks/todo.ts";

export interface SubagentToolDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd: string;
	agents: () => import("../delegation/definitions.ts").AgentDefinition[];
	emit: (event: AgentEvent) => void;
	/** Max concurrent subagent executions (default: 4). */
	maxParallelAgents?: number;
}

/** Get all built-in tools as an array. */
export function getBuiltInTools(): Tool[] {
	return [todo_tool, ask_user, ...rag_tools];
}

/** Get subagent tools with dependencies. */
export function getBuiltInSubagentTools(deps: SubagentToolDeps): Tool[] {
	const spawnDeps: SpawnAgentDeps = {
		config: deps.config,
		backend: deps.backend,
		cwd: deps.cwd,
		agents: deps.agents,
		emit: deps.emit,
		defaultMaxIterations: deps.config().maxIterations || 30,
		concurrencyLimiter: createSubagentConcurrencyLimiter(
			deps.maxParallelAgents,
		),
	};
	const spawn = createSpawnAgentTool(spawnDeps);
	const spawnMany = createSpawnAgentsTool(spawnDeps);

	return [spawn, spawnMany];
}
