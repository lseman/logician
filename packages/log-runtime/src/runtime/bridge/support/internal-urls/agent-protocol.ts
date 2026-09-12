// ── agent:// protocol handler ──────────────────────────────────────────────────
// Resolves subagent execution results.
// URL forms:
//   agent://<id>                  — full result as JSON
//   agent://<id>/content          — final output text
//   agent://<id>/status           — completion status
//   agent://<id>/details.metrics  — nested dot-notation access

import { AgentOutputRegistry, resolvePath } from "./agent-registry";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

export class AgentProtocolHandler implements ProtocolHandler {
	readonly scheme = "agent";

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const registry = AgentOutputRegistry.instance();
		const agentId = url.host;
		const pathname = url.pathname;

		if (!agentId) {
			const ids = registry.ids();
			return {
				url: url.href,
				content:
					ids.length > 0
						? `# Agent Results\n\n${ids.map(id => `  - ${id}`).join("\n")}`
						: "# Agent Results\n\nNo agents have completed yet.",
				contentType: "text/markdown",
			};
		}

		const entry = registry.get(agentId);
		if (!entry) {
			const ids = registry.ids();
			const hint =
				ids.length > 0
					? `\nAvailable: ${ids.join(", ")}`
					: "\nNo agents have completed yet.";
			throw new Error(`Unknown agent: ${agentId}${hint}`);
		}

		// No path segment or root — return full result as JSON
		if (!pathname || pathname === "/" || pathname === "/0") {
			const result = {
				agentId: entry.agentId,
				agent: entry.agent,
				status: entry.status,
				timestamp: new Date(entry.timestamp).toISOString(),
				content: entry.content,
				details: entry.details,
			};
			return {
				url: url.href,
				content: JSON.stringify(result, null, 2),
				contentType: "application/json",
			};
		}

		// Path segment — resolve dot-notation into result object
		const path = pathname.slice(1); // strip leading /
		const resultObj = {
			content: entry.content,
			status: entry.status,
			agentId: entry.agentId,
			agent: entry.agent,
			timestamp: new Date(entry.timestamp).toISOString(),
			details: entry.details,
		};

		const value = resolvePath(resultObj, path);

		if (value === undefined) {
			throw new Error(
				`Path not found: ${path}\nAvailable keys: content, status, agentId, agent, timestamp, details`,
			);
		}

		if (typeof value === "object" && value !== null) {
			return {
				url: url.href,
				content: JSON.stringify(value, null, 2),
				contentType: "application/json",
			};
		}

		return {
			url: url.href,
			content: String(value),
			contentType: "text/plain",
		};
	}
}
