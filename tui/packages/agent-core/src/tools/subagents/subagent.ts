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

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { LLMBackend } from "../../core/backend.ts";
import { runAgentLoop } from "../../core/agent-loop-runner.ts";
import { parseFrontmatter } from "../shared/skills.ts";
import type {
	AgentConfig,
	AgentEvent,
	Message,
	Tool,
} from "../../core/types.ts";

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
}

const GENERIC_SUBAGENT_PROMPT =
	"You are a subagent completing one delegated task. Work autonomously — " +
	"you cannot ask the user questions. When done, end with a final message " +
	"that fully reports your findings/results: it is the ONLY thing returned " +
	"to the caller, so include every detail that matters (paths, names, " +
	"conclusions). Do not pad it with process narration.";

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
		BUILTIN_AGENTS.map((a) => [a.name, a]),
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
						? frontmatter.tools.split(",").map((t) => t.trim())
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
}

let agentSeq = 0;

// Cap what a child can return into the parent context.
const MAX_RESULT_CHARS = 16_000;

/** Resolve the child tool set: allowlist (or all), never spawn_agent itself. */
function resolveChildTools(def: AgentDefinition, parentTools: Tool[]): Tool[] {
	const base = parentTools.filter((t) => t.name !== "spawn_agent");
	if (!def.tools?.length) return base;
	if (def.tools.includes("__read_only__")) {
		return base.filter((t) => t.readOnly === true);
	}
	const allowed = new Set(def.tools);
	return base.filter((t) => allowed.has(t.name));
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
				.map((a) => `${a.name} (${a.description})`)
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
			},
			required: ["task"],
		},
		execute: async (args, ctx) => {
			const task = typeof args.task === "string" ? args.task : "";
			if (!task.trim()) return "Error: spawn_agent requires a task.";
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

			const agentId = `agent_${++agentSeq}`;
			const parent = deps.config();
			deps.emit({ type: "subagent_start", agentId, agent: def.name, task });

			// Child config: parent's provider settings, but its own prompt, scoped
			// tools, and NO parent hooks/queues/events — the child is isolated.
			let lastText = "";
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
				thinkingLevel: parent.thinkingLevel,
				autoRetryEnabled: parent.autoRetryEnabled,
				maxRetries: parent.maxRetries,
				turnTimeoutMs: parent.turnTimeoutMs,
				webSearch: parent.webSearch,
				// External settings-hooks (PreToolUse shell hooks etc.) stay enabled
				// so safety hooks also govern subagent tool use.
				runtimeHooksEnabled: parent.runtimeHooksEnabled,
				// Enable continuation so the child model doesn't abandon mid-task.
				// Subagents frequently get a premature "stop" — the default nudge
				// "You stopped without completing your work. Continue." is enough
				// to push them past that. Bounded by maxIterations on the child.
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
						cwd: childConfig.cwd,
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
					final?.content?.trim() || "(subagent produced no final message)",
				);
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
						metrics: { turns },
					},
				};
			} catch (e: unknown) {
				const message = `Error: subagent failed: ${(e as Error).message}`;
				deps.emit({
					type: "subagent_end",
					agentId,
					agent: def.name,
					result: message,
					isError: true,
				});
				return message;
			}
		},
	};
}

function truncateResult(text: string): string {
	return text.length > MAX_RESULT_CHARS
		? `${text.slice(0, MAX_RESULT_CHARS)}\n… [subagent report truncated]`
		: text;
}
