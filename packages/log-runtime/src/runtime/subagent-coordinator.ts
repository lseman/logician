// ── SubagentCoordinator ────────────────────────────────────────────────────────
// Owns subagent/spawn concerns extracted from agent-bridge.ts: discovering agent
// definitions (.logician/agents/*.md + plugin agents/*.md + built-ins),
// registering the spawn_agent/spawn_agents tools, and running a direct /spawn
// invocation (bypassing the LLM) with its own synthetic turn + transcript
// lifecycle events.

import path from "node:path";
import type {
	AgentConfig,
	LLMBackend,
	Message,
	Tool,
	ToolContext,
} from "@logician/log-core";
import type { AgentSession } from "@logician/log-core/harness";
import {
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
} from "@logician/log-core/runtime";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	type AgentDefinition,
	loadAgentDefinitions,
} from "../capabilities/delegation/definitions.ts";
import {
	getBuiltInSubagentTools,
	type SubagentToolDeps,
} from "../capabilities/tools/builtin-blocks.ts";

export interface SubagentCoordinatorDeps {
	config: () => AgentConfig;
	backend: LLMBackend;
	cwd: string;
	projectTrusted: boolean;
	maxParallelAgents?: number;
	getEnabledPluginRoots: () => Array<{ name: string; installPath: string }>;
	/** Add a tool to the bridge's default tool set (also wires config.tools / harness.setTools / system prompt). */
	onToolAdded: (tool: Tool) => void;
	getDefaultTools: () => Tool[];
	ensureSession: () => AgentSession;
	emit: (event: RuntimeEvent) => void;
	reportError: (error: unknown) => void;
}

export class SubagentCoordinator {
	private agentDefs: AgentDefinition[] = [];
	private pendingSpawnTasks: Array<{ task: string; agent?: string }> = [];
	private injected = false;

	constructor(private readonly deps: SubagentCoordinatorDeps) {}

	getAgentDefs(): AgentDefinition[] {
		return this.agentDefs;
	}

	isInjected(): boolean {
		return this.injected;
	}

	/**
	 * Register the spawn_agent and spawn_agents tools bound to discovered
	 * definitions. Subagent events are forwarded into the normal event stream
	 * as subagent_* envelopes via deps.config().onEvent.
	 */
	async inject(): Promise<void> {
		const cwd = this.deps.config().cwd || process.cwd();
		this.agentDefs = await loadAgentDefinitions([
			...(this.deps.projectTrusted
				? [path.join(cwd, ".logician", "agents")]
				: []),
			// Claude Code plugin agents (agents/*.md in each enabled plugin).
			...this.deps
				.getEnabledPluginRoots()
				.map(p => path.join(p.installPath, "agents")),
		]);

		const subagentDeps: SubagentToolDeps = {
			config: this.deps.config,
			backend: this.deps.backend,
			cwd,
			agents: () => this.agentDefs,
			emit: event => this.deps.config().onEvent?.(event),
			maxParallelAgents: this.deps.maxParallelAgents,
		};
		const existing = new Set(this.deps.getDefaultTools().map(t => t.name));
		for (const tool of getBuiltInSubagentTools(subagentDeps)) {
			if (!existing.has(tool.name)) this.deps.onToolAdded(tool);
		}

		this.injected = true;
		// Drain any /spawn tasks that arrived before init completed.
		// spawnDirectly feeds result back; only process the first one.
		const first = this.pendingSpawnTasks.shift();
		if (first) this.spawnDirectly(first.task, first.agent);
	}

	/**
	 * Directly invoke the spawn_agent tool without going through the LLM.
	 * Emits tool execution events to the transcript so the subagent output
	 * renders as a tool chunk with integrated streaming output.
	 *
	 * Lifecycle/chunk events come from the tool itself via
	 * `deps.emit → config.onEvent → mapAgentEvent` (same path as LLM-mode).
	 * On completion, the /spawn arg and final result are appended to harness
	 * history (user + spawn_agent tool call + tool result) so later turns
	 * can see them — without followUp/sendMessage (those stall outside a loop).
	 */
	spawnDirectly(task: string, agent?: string): void {
		// If init hasn't finished, queue the task for later.
		if (!this.injected) {
			this.pendingSpawnTasks.push({ task, agent });
			return;
		}
		const spawnTool = this.deps
			.getDefaultTools()
			.find(t => t.name === "spawn_agent");
		if (!spawnTool) {
			this.deps.reportError(
				new Error(
					"spawn_agent tool not available (subagent tools not injected)",
				),
			);
			return;
		}

		const toolCallId = `spawn_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		const turnId = `spawn_turn_${Date.now()}`;
		const controller = new AbortController();

		// Create a synthetic turn so the transcript has a context for the
		// tool chunk. handleToolStart / subagent lifecycle both need a
		// current turn.
		this.deps.emit({ type: "turn_start", turnId: turnId });

		// Emit tool_execution_start so the transcript creates a tool chunk
		// before the tool fires subagent_start/subagent_event.
		this.deps.emit({
			type: "tool_execution_start",
			toolName: "spawn_agent",
			args: { task, agent },
			toolCallId: toolCallId,
		});

		const ctx: ToolContext = {
			signal: controller.signal,
			onUpdate: delta => {
				// Live stream into the open tool chunk. Child chunks (thinking,
				// tool calls, content with agentId) arrive separately via
				// deps.emit(subagent_event) → mapAgentEvent.
				this.deps.emit({
					type: "tool_execution_update",
					toolName: "spawn_agent",
					partialResult: delta,
					toolCallId: toolCallId,
				});
			},
		};

		void spawnTool
			.execute({ task, agent }, ctx)
			.then(result => {
				const content = typeof result === "string" ? result : result.content;
				const isError =
					typeof result === "string" ? false : result.isError === true;

				// tool_execution_end closes the card. subagent_end was already
				// emitted by the tool via deps.emit during execute().
				this.deps.emit({
					type: "tool_execution_end",
					toolName: "spawn_agent",
					result: content,
					isError: isError,
					toolCallId: toolCallId,
					details:
						typeof result === "object" && result.details
							? result.details
							: undefined,
				});
				if (isError) {
					this.deps.reportError(new Error(content));
				}
				this.recordInHistory(task, agent, toolCallId, content, isError);
			})
			.catch(err => {
				const error = err as Error;
				this.deps.emit({
					type: "tool_execution_end",
					toolName: "spawn_agent",
					result: error.message,
					isError: true,
					toolCallId: toolCallId,
				});
				this.deps.reportError(error);
				this.recordInHistory(task, agent, toolCallId, error.message, true);
			})
			.finally(() => {
				// Close the synthetic turn and return the UI to ready. Mirror
				// runMessage's phase emit so status/animation settle.
				this.deps.emit({ type: "turn_end", turnId });
				this.deps.emit({ type: "phase", state: "ready" });
			});
	}

	/**
	 * Persist a direct /spawn exchange into harness history so subsequent
	 * agent turns can see the request and the subagent's final report.
	 * Shape mirrors LLM-mode spawn_agent: user command → tool call → result.
	 */
	private recordInHistory(
		task: string,
		agent: string | undefined,
		toolCallId: string,
		result: string,
		isError: boolean,
	): void {
		try {
			const session = this.deps.ensureSession();
			const agentName = agent?.trim() || "general";
			const userText = agent?.trim()
				? `/spawn agent=${agent.trim()} ${task}`
				: `/spawn ${task}`;
			const messages: Message[] = [
				createUserMessage(userText),
				createAssistantMessage("", [
					{
						id: toolCallId,
						name: "spawn_agent",
						arguments: JSON.stringify({ task, agent: agentName }),
					},
				]),
				createToolResultMessage(toolCallId, "spawn_agent", result, isError),
			];
			session.appendMessages(messages);
		} catch (err: unknown) {
			this.deps.reportError(err as Error);
		}
	}
}
