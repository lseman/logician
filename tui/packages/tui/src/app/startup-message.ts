import { formatStartupMemory } from "./startup-memory.ts";

export interface StartupMessageOptions {
	configPath?: string;
	project: string;
	themeName: string;
}

export function formatStartupMessage(
	state: Record<string, unknown>,
	options: StartupMessageOptions,
): string {
	const pluginCount = Number(state.startup_plugins_loaded || 0);
	const hookCount = Number(state.startup_hooks_loaded || 0);
	const mcpServerCount = Number(state.mcp_servers_loaded || 0);
	const mcpToolCount = Number(state.mcp_tools_loaded || 0);
	const plugins = stringList(state.startup_plugins);
	const skills = normalizeSkills(state.loaded_skills);
	const contexts = stringList(state.startup_hook_contexts);
	const hookMessages = Array.isArray(state.startup_hook_messages)
		? state.startup_hook_messages
				.map(normalizeStartupHookMessage)
				.filter((item) => item.content)
		: [];
	const initialMessage = String(
		state.startup_hook_initial_message || "",
	).trim();
	const errors = stringList(state.startup_hook_errors);
	const mcpErrors = stringList(state.mcp_errors);
	const dim = "\x1b[2m";
	const reset = "\x1b[0m";
	const model = String(state.model || "unknown");
	const agent = String(state.agent_name || "logician");
	const mcpState = state.mcp_loading
		? "MCP loading"
		: state.mcp_deferred
			? "MCP deferred"
			: `MCP ${mcpServerCount}/${mcpToolCount}`;
	const searchUrl = String(state.web_search_url || "");
	const searchEnabled =
		state.web_search_enabled === true ||
		(Array.isArray(state.tools) && state.tools.includes("web_search"));
	const searchState = searchEnabled ? searchUrl || "enabled" : "disabled";

	const lines = [
		"# Logician",
		`${agent} is ready · ${model}`,
		"",
		`${dim}${options.project} · theme ${options.themeName} · plugins ${pluginCount} · skills ${skills.length} · hooks ${hookCount} · ${mcpState}${reset}`,
		`${dim}web search ${searchState}${reset}`,
		`${dim}config ${options.configPath || "-"} · ${String(state.base_url || "unknown")}${reset}`,
	];

	if (initialMessage) {
		lines.push("", "## Startup message", initialMessage);
	}
	if (contexts.length) {
		lines.push("", "## Plugin startup messages");
		if (hookMessages.length) {
			for (const message of hookMessages) {
				lines.push("", `### ${message.title}`, message.content);
			}
		} else {
			contexts.forEach((context, index) => {
				lines.push("", `### Startup hook ${index + 1}`, context);
			});
		}
	}
	if (errors.length) {
		lines.push(
			"",
			"## Startup hook errors",
			...errors.map((error) => `- ${error}`),
		);
	}
	if (mcpErrors.length) {
		lines.push(
			"",
			"## MCP errors",
			...mcpErrors.map((error) => `- ${error}`),
		);
	}
	lines.push(...formatStartupMemory(state));
	lines.push(
		"",
		"## Loaded resources",
		`${dim}Plugins (${plugins.length})${reset}`,
		plugins.length ? plugins.join(" · ") : `${dim}None${reset}`,
		"",
		`${dim}Skills (${skills.length})${reset}`,
		skills.length
			? skills.map((skill) => `/${skill.slashName}`).join(" · ")
			: `${dim}None${reset}`,
	);
	return lines.join("\n");
}

function stringList(value: unknown): string[] {
	return Array.isArray(value)
		? value.map((item) => String(item || "").trim()).filter(Boolean)
		: [];
}

function normalizeSkills(value: unknown): Array<{
	slashName: string;
	description: string;
}> {
	if (!Array.isArray(value)) return [];
	return value
		.map((item) => {
			if (!item || typeof item !== "object") return null;
			const skill = item as Record<string, unknown>;
			const slashName = String(skill.slash_name || skill.name || "").trim();
			const description = String(skill.description || "").trim();
			return slashName ? { slashName, description } : null;
		})
		.filter(
			(item): item is { slashName: string; description: string } =>
				item !== null,
		);
}

function normalizeStartupHookMessage(item: unknown): {
	title: string;
	content: string;
} {
	if (!item || typeof item !== "object") {
		return { title: "Startup hook", content: String(item || "").trim() };
	}
	const raw = item as Record<string, unknown>;
	const pluginName = String(raw.plugin_name || "").trim();
	const pluginId = String(raw.plugin_id || "").trim();
	const matcher = String(raw.matcher || "").trim();
	const label = pluginName || pluginId || "Startup hook";
	const suffix =
		pluginName && pluginId && pluginName !== pluginId ? ` (${pluginId})` : "";
	const matcherText = matcher && matcher !== "*" ? ` · ${matcher}` : "";
	return {
		title: `${label}${suffix}${matcherText}`,
		content: String(raw.content || "").trim(),
	};
}
