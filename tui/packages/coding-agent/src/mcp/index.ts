// ── MCP (barrel) ──────────────────────────────────────────────────────────
// Public MCP API.
// Implementation split into:
//   mcp-client.ts   : StdioMcpClient, HttpMcpClient, JSON-RPC helpers
//   mcp-manager.ts  : McpManager class and config functions

export {
	McpManager,
	type McpLoadResult,
	type McpServerInfo,
	type McpSnapshotResult,
	type McpToggleResult,
} from "./manager.ts";
export {
	createMcpClient,
	createMcpTool,
	encodeMcpMessage,
	tryDecodeMcpMessage,
	parseMcpToolDefinition,
	formatMcpToolResult,
} from "./client.ts";
