// ── Tools barrel ─────────────────────────────────────────────────────────

export * from "./async-utils.ts";
export { parseFrontmatter } from "./frontmatter.ts";
export * from "./json-utils.ts";
export { parseToolInput } from "./parser.ts";
export {
	ensureInsideCwd,
	markPathIgnoredByCloudSync,
	normalizePath,
	readUtf8IfExists,
	resolvePath,
	resolveReadPath,
	resolveToCwd,
} from "./path-utils.ts";
export {
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
	type PermissionVerdict,
	primaryArgString,
} from "./permissions.ts";
export {
	configurePluginRuntimeEnv,
	type PluginCommandResult,
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
	TsPluginManager,
} from "./plugins.ts";
export {
	type PreparedToolCall,
	ToolRegistry,
	type ToolRegistryOptions,
} from "./registry.ts";
export {
	type HighlightResult,
	highlight,
	highlightAuto,
	listLanguages,
} from "./syntax-highlighter.ts";
export {
	parseTextToolCalls,
	stripTextToolCalls,
} from "./text-to-tool-calls.ts";
