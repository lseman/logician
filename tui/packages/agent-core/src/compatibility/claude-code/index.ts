// Claude Code compatibility boundary.
//
// Everything exported here exists to let Logician consume Claude Code plugin
// manifests and hook protocols. Native Logician hooks live under `hooks/`;
// TypeScript extensions live under `extensions/`.

export {
	createClaudeCodeHookLayer,
	type ClaudeCodeHookLayer,
	type ClaudeCodeHookLayerOptions,
} from "./hook-layer.ts";

export {
	loadPluginHooks,
	executeLoadedHook,
	parseHookResponse,
	buildHookInput,
	type HookEventType as ClaudeCodeHookEventType,
	type HookCommand as ClaudeCodeHookCommand,
	type HookDefinition as ClaudeCodeHookDefinition,
	type HookExecutionResult as ClaudeCodeHookExecutionResult,
} from "./plugin-executor.ts";
export * from "./plugin-manager.ts";
