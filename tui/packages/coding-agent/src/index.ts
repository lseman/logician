// ── Coding Agent — Orchestration Layer ────────────────────────────────────────
// Exports the orchestration layer: bridge, sessions, config, slash commands,
// transcript, MCP, plugins, loop detection, and utilities.

export * from "./application/index.ts";
export {
	createSlashCommands,
	type SlashCommandCategory,
	type SlashCommandDef,
	type SlashCommandSource,
	type SlashDispatch,
} from "./commands/slash-commands.ts";
export {
	configBool,
	configNumber,
	configString,
	findLogicianConfig,
	type LogicianTuiConfig,
	loadLogicianConfig,
	saveConfigField,
} from "./configuration/config.ts";
export * from "./context/index.ts";
export * from "./mcp/index.ts";
export * from "./prompts/index.ts";
export type { ParsedBridgeEvent } from "./runtime/events.ts";
export { SessionStore } from "./sessions/session-store.ts";
export { Transcript, type Turn } from "./sessions/transcript.ts";
export * from "./skills/index.ts";
export * from "./tools/index.ts";
export * from "./trust/index.ts";
export { envNumber, formatContextSize, tableRow } from "./tui-utils.ts";
