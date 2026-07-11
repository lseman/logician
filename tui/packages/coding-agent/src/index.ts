// ── Coding Agent — Orchestration Layer ────────────────────────────────────────
// Exports the orchestration layer: bridge, sessions, config, slash commands,
// transcript, MCP, plugins, loop detection, and utilities.

export * from "./runtime/bridge.ts";
export { SessionStore } from "./sessions/session-store.ts";
export {
	configBool,
	configNumber,
	configString,
	findLogicianConfig,
	loadLogicianConfig,
	saveConfigField,
	type LogicianTuiConfig,
} from "./configuration/config.ts";
export {
	createSlashCommands,
	type SlashCommandDef,
	type SlashCommandCategory,
	type SlashCommandSource,
	type SlashDispatch,
} from "./commands/slash-commands.ts";
export { Transcript, type Turn } from "./sessions/transcript.ts";
export { type ParsedBridgeEvent } from "./runtime/events.ts";
export { formatContextSize, envNumber, tableRow } from "./tui-utils.ts";
export { LoopManager } from "./runtime/loop-manager.ts";
export * from "./tools/index.ts";
export * from "./mcp/index.ts";
export * from "./skills.ts";
export * from "./system-prompt.ts";
export * from "./context-files/index.ts";
export * from "./prompts/index.ts";
export * from "./trust/index.ts";
