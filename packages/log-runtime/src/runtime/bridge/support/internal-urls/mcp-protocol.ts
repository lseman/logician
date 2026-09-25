// ── mcp:// protocol handler ───────────────────────────────────────────────────
// Resolves MCP server resources.
// URL forms:
//   mcp://                          — list configured MCP servers
//   mcp://<server>/<resource-uri>   — read a resource from a specific server

import { getMcpRegistryInstance } from "../../../../capabilities/mcp/mcp-server-registry.ts";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

export class McpProtocolHandler implements ProtocolHandler {
	readonly scheme = "mcp";
	readonly immutable = true;

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const registry = getMcpRegistryInstance();
		if (!registry) {
			if (url.target && url.target !== "/") {
				throw new Error("MCP server registry is not available");
			}
			return {
				url: url.href,
				content: "# MCP Resources\n\nMCP server registry is not available.",
				contentType: "text/markdown",
			};
		}

		const host = url.host;

		// No host — list all servers, their status, and the resources each
		// loaded server exposes (one failing server must not break the list).
		if (!url.target || url.target === "/") {
			const servers = registry.servers;
			if (!servers || servers.length === 0) {
				return {
					url: url.href,
					content: "# MCP Resources\n\nNo MCP servers configured.",
					contentType: "text/markdown",
				};
			}

			const lines = await Promise.all(
				servers
					.filter((s: { enabled?: boolean }) => s.enabled !== false)
					.map(
						async (s: {
							serverName: string;
							toolCount: number;
							loaded: boolean;
						}) => {
							const status = s.loaded ? "✅" : "❌";
							let resourceNote = "";
							if (s.loaded) {
								try {
									const listed = await registry.listResources(s.serverName);
									const shown = listed.resources.slice(0, 10);
									const resourceLines = shown
										.map(
											r =>
												`  - \`${r.uri}\`${r.description ? ` — ${r.description}` : ""}`,
										)
										.join("\n");
									const extra =
										listed.resources.length > shown.length
											? `\n  - … and ${listed.resources.length - shown.length} more`
											: "";
									resourceNote =
										listed.resources.length === 0
											? ""
											: `\n  resources:\n${resourceLines}${extra}`;
								} catch {
									resourceNote =
										"\n  resources: (server does not expose resources)";
								}
							}
							return `- ${status} \`${s.serverName}\` (${s.toolCount} tools, ${s.loaded ? "loaded" : "unavailable"})${resourceNote}`;
						},
					),
			);

			return {
				url: url.href,
				content: `# MCP Resources\n\n${lines.length} server${lines.length === 1 ? "" : "s"}:\n\n${lines.join("\n")}`,
				contentType: "text/markdown",
			};
		}

		// Require a server and preserve its resource URI verbatim. Never select
		// a server based on connection order or fall back after a failed lookup.
		const resourceUri = url.target.slice(host.length + 1);
		if (!host || !url.target.startsWith(`${host}/`) || !resourceUri) {
			throw new Error(
				"MCP links require mcp://<server>/<resource-uri>. Read mcp:// to list servers.",
			);
		}

		// Attempt to read the resource
		try {
			const result = await registry.readResource(host, resourceUri);
			const textParts = result.contents.flatMap(c =>
				typeof c.text === "string" ? [c.text] : [],
			);

			if (textParts.length === 0) {
				return {
					url: url.href,
					content: `# MCP Resource: ${resourceUri}\n\n[Binary content: ${result.mimeType ?? "unknown"}]`,
					contentType: "text/plain",
				};
			}

			return {
				url: url.href,
				content: textParts.join("\n---\n"),
				contentType:
					result.mimeType === "application/json"
						? "application/json"
						: "text/plain",
			};
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`MCP resource read error: ${message}`);
		}
	}
}
