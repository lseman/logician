// ── MCP (barrel) ──────────────────────────────────────────────────────────
// Public MCP API.
// Implementation split into:
//   mcp-client.ts   : StdioMcpClient, HttpMcpClient, JSON-RPC helpers
//   mcp-manager.ts  : McpManager class and config functions

export {
	createMcpClient,
	createMcpTool,
	encodeMcpMessage,
	formatMcpToolResult,
	parseMcpToolDefinition,
	tryDecodeMcpMessage,
} from "./client.ts";
export {
	type McpLoadResult,
	McpManager,
	type McpServerInfo,
	type McpSnapshotResult,
	type McpToggleResult,
} from "./manager.ts";
