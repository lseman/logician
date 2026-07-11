// ── Built-in Tools Registry ──────────────────────────────────────────────────
// Returns all built-in tools for automatic registration at startup.

import type { Tool, AgentEvent, AgentConfig } from "@logician/agent-core";
import type { LLMBackend } from "@logician/agent-core/core/backend.ts";
import { ask_user } from "./ask-user/ask-user.ts";
import { task_status } from "./todo/task-status.ts";
import { todo_tool } from "./todo/todo.ts";
import {
	createSpawnAgentTool,
	type SpawnAgentDeps,
} from "./subagents/subagent.ts";
import {
	createParallelSpawnAgentTool,
	type ParallelSpawnAgentDeps,
} from "./subagents/parallel-subagent.ts";
import {
	createCoordinateSubagentsTool,
	type CoordinateSubagentsDeps,
} from "./subagents/coordinate-subagents.ts";

export interface SubagentToolDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd: string;
	agents: () => import("./subagents/subagent.ts").AgentDefinition[];
	emit: (event: AgentEvent) => void;
	parallelOptions?: import("./subagents/parallel-subagent.ts").ParallelSpawnOptions;
	coordinateOptions?: import("./subagents/coordinate-subagents.ts").CoordinateSubagentsOptions;
}

/** Get all built-in tools as an array. */
export function getBuiltInTools(): Tool[] {
	return [todo_tool, ask_user, task_status];
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
	};
	const spawn = createSpawnAgentTool(spawnDeps);

	const parallelDeps: ParallelSpawnAgentDeps = {
		config: deps.config,
		backend: deps.backend,
		cwd: deps.cwd,
		agents: deps.agents,
		emit: deps.emit,
		options: deps.parallelOptions || {},
	};
	const parallelSpawn = createParallelSpawnAgentTool(parallelDeps);

	const coordinateDeps: CoordinateSubagentsDeps = {
		config: deps.config,
		backend: deps.backend,
		cwd: deps.cwd,
		agents: deps.agents,
		emit: deps.emit,
		options: deps.coordinateOptions || {},
		defaultMaxIterations: deps.config().maxIterations || 30,
	};
	const coordinateSpawn = createCoordinateSubagentsTool(coordinateDeps);

	return [spawn, parallelSpawn, coordinateSpawn];
}
