import type { AgentConfig } from "../../../system/types/types-config.ts";
import type { Tool } from "../../../system/types/types-messages.ts";
import type { HarnessModule } from "../types.ts";

export class HarnessConfigurationError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "HarnessConfigurationError";
	}
}

/** Compose inert harness modules while rejecting ambiguous names and tools. */
export function composeHarnessConfig(
	modules: HarnessModule[],
	config: AgentConfig,
): AgentConfig {
	const moduleNames = new Set<string>();
	const tools: Tool[] = [];
	const toolNames = new Set<string>();
	const addTools = (items: Tool[] | undefined): void => {
		for (const tool of items ?? []) {
			if (toolNames.has(tool.name)) {
				throw new HarnessConfigurationError(
					`Duplicate harness tool: ${tool.name}`,
				);
			}
			toolNames.add(tool.name);
			tools.push(tool);
		}
	};

	let composed: AgentConfig = { ...config };
	for (const module of modules) {
		if (moduleNames.has(module.name)) {
			throw new HarnessConfigurationError(
				`Duplicate harness module: ${module.name}`,
			);
		}
		moduleNames.add(module.name);
		composed = { ...composed, ...module.config };
		addTools(module.config?.tools);
	}
	composed = { ...composed, ...config };
	addTools(config.tools);
	return tools.length > 0 ? { ...composed, tools } : composed;
}
