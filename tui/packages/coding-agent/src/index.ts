// ── Coding Agent — Orchestration Layer ────────────────────────────────────────
// Exports the orchestration layer: bridge, sessions, config, slash commands,
// transcript, MCP, plugins, reasoners, loop detection, and utilities.

export * from "./bridge.ts";
export { SessionStore } from "./session-store.ts";
export {
	configBool,
	configNumber,
	configString,
	findLogicianConfig,
	loadLogicianConfig,
	saveConfigField,
	type LogicianTuiConfig,
} from "./config.ts";
export {
	createSlashCommands,
	type SlashCommandDef,
	type SlashCommandCategory,
	type SlashCommandSource,
	type SlashDispatch,
} from "./slash-commands.ts";
export { Transcript, type Turn } from "./transcript.ts";
export { type ParsedBridgeEvent } from "./events.ts";
export { formatContextSize, envNumber, tableRow } from "./tui-utils.ts";
export { LoopManager } from "./loop-manager.ts";
export * from "./tools/index.ts";
export * from "./mcp/index.ts";
export * from "./skills.ts";
export * from "./subagents/subagent.ts";
export * from "./system-prompt.ts";
export * from "./reasoners/index.ts";
export * from "./context-files/index.ts";
export * from "./prompts/index.ts";
export * from "./trust/index.ts";
