// ── GitHub MCP client lookup ─────────────────────────────────────────────────
// Shared helper for pr:// and issue:// protocol handlers.

import { getMcpRegistryInstance } from "../../../../capabilities/mcp/mcp-server-registry.ts";

/**
 * Find the first GitHub MCP client by inspecting server names.
 * Returns null when no registry or no GitHub server is available.
 */
export function findGithubClient() {
	const registry = getMcpRegistryInstance();
	if (!registry) return null;
	return (
		registry.clients.find(c => c.name.toLowerCase().includes("github")) ?? null
	);
}
