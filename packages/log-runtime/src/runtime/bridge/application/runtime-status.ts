import type { AgentConfig, Tool } from "@logician/log-core";
import type { PluginCommandResult } from "../../../adapters/claude-code/plugin-runtime.ts";
import type { Skill } from "../../../capabilities/skills/loader.ts";

const IDLE_RUNTIME_STATE = {
	phase: "idle",
	isStreaming: false,
	pendingToolCalls: [],
	abortRequested: false,
} as const;

export interface RuntimeStatusInput {
	config: AgentConfig;
	toolNames: string[];
	mcpServerCount: number;
	mcpToolCount: number;
	mcpErrors: unknown[];
	contextTokens: number;
	contextMaxTokens?: number;
	runtimeState?: unknown;
	configPath?: string | null;
	reasoner: string;
}

export interface InitializationStatusInput extends RuntimeStatusInput {
	mcpLoaded: boolean;
	mcpLoading: boolean;
	enabledPluginRoots: Array<{ name: string }>;
	loadedSkills: Skill[];
	skillsInjected: boolean;
	skillsVisible: boolean;
	pluginCount: number;
	hookResult: PluginCommandResult | null;
}

/** Projects the stable lightweight state returned to runtime clients. */
export function projectRuntimeStatus(
	input: RuntimeStatusInput,
): Record<string, unknown> {
	return {
		agent_name: "logician",
		model: input.config.model,
		base_url: input.config.baseUrl,
		web_search_url: input.config.webSearch?.baseUrl || "",
		web_search_enabled: input.toolNames.includes("web_search"),
		tools: input.toolNames,
		mcp_servers: input.mcpServerCount,
		mcp_tools: input.mcpToolCount,
		mcp_errors: input.mcpErrors,
		context_tokens: input.contextTokens,
		context_max_tokens: input.contextMaxTokens,
		runtime_state: input.runtimeState ?? IDLE_RUNTIME_STATE,
		config_path: input.configPath || "",
		connected: true,
		reasoner: input.reasoner,
	};
}

/** Projects the richer initialization report without performing initialization. */
export function projectInitializationStatus(
	input: InitializationStatusInput,
): Record<string, unknown> {
	const base = projectRuntimeStatus(input);
	const hookResult = input.hookResult;
	return {
		...base,
		mcp_deferred: !input.mcpLoaded && process.env.LOGICIAN_MCP !== "0",
		mcp_loading: input.mcpLoading,
		mcp_servers_loaded: input.mcpServerCount,
		mcp_tools_loaded: input.mcpToolCount,
		hooks_enabled: input.config.runtimeHooksEnabled !== false,
		hook_transcript_path: input.config.hookTranscriptPath || "",
		startup_plugins_loaded: input.pluginCount,
		startup_plugins: input.enabledPluginRoots.map(plugin => plugin.name),
		startup_hooks_loaded: hookResult?.hook_count || 0,
		startup_hook_contexts: hookResult?.additional_contexts || [],
		startup_hook_messages: hookResult?.context_messages || [],
		startup_hook_initial_message: hookResult?.initial_user_message || "",
		startup_hook_errors: hookResult?.errors || [],
		skills_injected: input.skillsInjected
			? input.loadedSkills.filter(skill => !skill.disableModelInvocation).length
			: 0,
		skills_visible: input.skillsVisible,
		loaded_skills: input.loadedSkills.map(skill => ({
			name: skill.name,
			slash_name: skill.slashName,
			description: skill.description,
			model_visible: !skill.disableModelInvocation,
		})),
	};
}

export function runtimeToolNames(
	liveTools: { list(): Tool[] } | undefined,
	defaultTools: Tool[],
): string[] {
	return (liveTools?.list() ?? defaultTools).map(tool => tool.name);
}
