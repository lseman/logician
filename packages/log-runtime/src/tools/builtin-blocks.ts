// ── Built-in Tools Registry ──────────────────────────────────────────────────
// Returns all built-in tools for automatic registration at startup.

import type {
	AgentConfig,
	AgentEvent,
	LLMBackend,
	Tool,
} from "@logician/log-core";
import { ask } from "../capabilities/ask/index.ts";
import {
	createSpawnAgentsTool,
	createSpawnAgentTool,
	createSubagentConcurrencyLimiter,
	type SpawnAgentDeps,
} from "../capabilities/delegation/definitions.ts";
import { rag_tools } from "../capabilities/rag/index.ts";
import { todo_tool } from "../capabilities/tasks/todo.ts";

interface BuiltinToolsOptions {
	todoEnabled?: boolean;
}

export interface SubagentToolDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd: string;
	agents: () => import("../capabilities/delegation/definitions.ts").AgentDefinition[];
	emit: (event: AgentEvent) => void;
	/** Max concurrent subagent executions (default: 4). */
	maxParallelAgents?: number;
}

/** Get all built-in tools as an array. */
export function getBuiltInTools(opts: BuiltinToolsOptions = {}): Tool[] {
	const tools: Tool[] = [];
	if (opts.todoEnabled !== false) tools.push(todo_tool);
	tools.push(ask, ...rag_tools);
	return tools;
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
