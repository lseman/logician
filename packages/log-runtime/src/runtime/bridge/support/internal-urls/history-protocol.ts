// ── history:// protocol handler ────────────────────────────────────────────────
// Lists completed subagents and returns their results.
// URL forms:
//   history://              — list all completed agents
//   history://<id>          — full result for agent <id>
//   history://<id>/content  — final output text only
//   history://<id>/status   — completion status

import { AgentOutputRegistry, resolvePath } from "./agent-registry";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

/** Format a timestamp as "Ns/Nm/Nh/Nd ago". */
function formatAgo(timestamp: number): string {
	const diffMs = Math.max(0, Date.now() - timestamp);
	const secs = Math.floor(diffMs / 1000);
	if (secs < 60) return `${secs}s ago`;
	const mins = Math.floor(secs / 60);
	if (mins < 60) return `${mins}m ago`;
	const hours = Math.floor(mins / 60);
	if (hours < 24) return `${hours}h ago`;
	return `${Math.floor(hours / 24)}d ago`;
}

export class HistoryProtocolHandler implements ProtocolHandler {
	readonly scheme = "history";

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const registry = AgentOutputRegistry.instance();
		const agentId = url.rawHost || url.hostname;

		// No agent ID — list all completed agents
		if (!agentId) {
			const ids = registry.ids();
			if (ids.length === 0) {
				return {
					url: url.href,
					content: "# Agent History\n\nNo agents have completed yet.",
					contentType: "text/markdown",
				};
			}

			// Sort by timestamp (most recent first)
			const entries = ids
				.map(id => registry.get(id))
				.filter((e): e is NonNullable<typeof e> => e !== undefined);
			entries.sort((a, b) => b.timestamp - a.timestamp);
			const lines = entries.map(e => {
				const statusIcon = e.status === "completed" ? "✅" : "❌";
				const ago = formatAgo(e.timestamp);
				return `- ${statusIcon} \`${e.agentId}\` — ${e.agent} — ${ago}`;
			});

			return {
				url: url.href,
				content: `# Agent History\n\n${entries.length} agent${entries.length === 1 ? "" : "s"} completed:\n\n${lines.join("\n")}`,
				contentType: "text/markdown",
			};
		}

		const entry = registry.get(agentId);
		if (!entry) {
			const ids = registry.ids();
			const hint =
				ids.length > 0
					? `\nKnown agents: ${ids.join(", ")}`
					: "\nNo agents have completed yet.";
			throw new Error(`Unknown agent: ${agentId}${hint}`);
		}

		// No path segment or root — return full result as JSON
		if (!url.pathname || url.pathname === "/" || url.pathname === "/0") {
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
		const path = url.pathname.slice(1);
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
