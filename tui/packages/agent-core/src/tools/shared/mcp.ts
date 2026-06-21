// ── MCP (barrel) ──────────────────────────────────────────────────────────
// Re-exports McpManager and all types for backward compatibility.
// Implementation split into:
//   mcp-client.ts   : StdioMcpClient, HttpMcpClient, JSON-RPC helpers
//   mcp-manager.ts  : McpManager class and config functions

export {
	McpManager,
	type McpLoadResult,
	type McpServerInfo,
	type McpSnapshotResult,
	type McpToggleResult,
} from "./mcp-manager.ts";
export {
	createMcpClient,
	createMcpTool,
	encodeMcpMessage,
	tryDecodeMcpMessage,
	parseMcpToolDefinition,
	formatMcpToolResult,
} from "./mcp-client.ts";
