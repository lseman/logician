// Claude Code compatibility boundary.
//
// Everything exported here exists to let Logician consume Claude Code plugin
// manifests and hook protocols. Native Logician hooks live under `hooks/`;
// TypeScript extensions live under `extensions/`.

export {
	type ClaudeCodeHookLayer,
	type ClaudeCodeHookLayerOptions,
	createClaudeCodeHookLayer,
} from "./hook-layer.ts";

export {
	buildHookInput,
	executeLoadedHook,
	type HookCommand as ClaudeCodeHookCommand,
	type HookDefinition as ClaudeCodeHookDefinition,
	type HookEventType as ClaudeCodeHookEventType,
	type HookExecutionResult as ClaudeCodeHookExecutionResult,
	loadPluginHooks,
	parseHookResponse,
} from "./plugin-executor.ts";
export * from "./plugin-registry.ts";
