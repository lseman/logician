// ── mcp:// protocol handler ───────────────────────────────────────────────────
// Resolves MCP server resources.
// URL forms:
//   mcp://                          — list all available MCP resources
//   mcp://<server>/<resource-uri>   — read a resource from a specific server
//   mcp://<resource-uri>            — auto-discover the server that owns the resource

import { getMcpRegistryInstance } from "../../../../capabilities/mcp/mcp-server-registry.ts";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

export class McpProtocolHandler implements ProtocolHandler {
	readonly scheme = "mcp";

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const registry = getMcpRegistryInstance();
		if (!registry) {
			return {
				url: url.href,
				content: "# MCP Resources\n\nMCP server registry is not available.",
				contentType: "text/markdown",
			};
		}

		const host = url.rawHost || url.hostname;
		const pathname = url.pathname;

		// No host — list all servers and their resource status
		if (!host) {
			const servers = registry.servers;
			if (!servers || servers.length === 0) {
				return {
					url: url.href,
					content: "# MCP Resources\n\nNo MCP servers configured.",
					contentType: "text/markdown",
				};
			}

			const lines = servers
				.filter((s: { enabled?: boolean }) => s.enabled !== false)
				.map(
					(s: { serverName: string; toolCount: number; loaded: boolean }) => {
						const status = s.loaded ? "✅" : "❌";
						return `- ${status} \`${s.serverName}\` (${s.toolCount} tools, ${s.loaded ? "loaded" : "unavailable"})`;
					},
				);

			return {
				url: url.href,
				content: `# MCP Resources\n\n${lines.length} server${lines.length === 1 ? "" : "s"}:\n\n${lines.join("\n")}`,
				contentType: "text/markdown",
			};
		}

		// Extract resource URI
		const rawPathname = pathname || "/";
		const hasServerPath = rawPathname !== "/" && !rawPathname.startsWith("/0");
		const resourceUri = hasServerPath ? `${host}${rawPathname}` : host;

		// Find a client that supports resource reads
		const client = registry.clients.find(
			c => typeof c.readResource === "function",
		);
		if (!client) {
			return {
				url: url.href,
				content: "# MCP Resources\n\nNo MCP servers support resource reads.",
				contentType: "text/markdown",
			};
		}

		// Attempt to read the resource
		try {
			if (!client.readResource)
				throw new Error("No MCP server supports resource reads");
			const result = await client.readResource(resourceUri);
			const textParts = result.contents
				.filter(
					(c: { text?: string }) => c.text !== undefined && c.text !== null,
				)
				.map(c => c.text as string);

			if (textParts.length === 0) {
				return {
					url: url.href,
					content: `# MCP Resource: ${resourceUri}\n\n[Binary content: ${(result as { mimeType?: string }).mimeType ?? "unknown"}]`,
					contentType: "text/plain",
				};
			}

			return {
				url: url.href,
				content: textParts.join("\n---\n"),
				contentType:
					(result as { mimeType?: string }).mimeType === "application/json"
						? "application/json"
						: "text/plain",
			};
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`MCP resource read error: ${message}`);
		}
	}
}
