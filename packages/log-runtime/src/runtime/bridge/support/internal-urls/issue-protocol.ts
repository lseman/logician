// ── issue:// protocol handler ────────────────────────────────────────────────
// Resolves GitHub issues via the GitHub MCP server.
// URL forms:
//   issue://                         — prompt for repo
//   issue://owner/repo               — list open issues
//   issue://owner/repo/42            — get issue #42
//   issue://owner/repo/42/comments   — get issue comments
//   issue://owner/repo/42/sub-issues — get sub-issues

import { findGithubClient } from "./github-client";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

/** Format a tool result value for display. */
function formatValue(value: unknown): string {
	if (value === null || value === undefined) return "[null]";
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean")
		return String(value);
	if (Array.isArray(value)) {
		if (value.length === 0) return "[]";
		if (value.length <= 5) {
			return JSON.stringify(value, null, 2);
		}
		return `${JSON.stringify(value.slice(0, 5), null, 2)}\n  ... (${value.length - 5} more items)`;
	}
	if (typeof value === "object") {
		return JSON.stringify(value, null, 2);
	}
	return String(value);
}

interface IssuePathParts {
	owner: string;
	repo: string;
	number?: number;
	action?: "comments" | "sub-issues";
}

function parseIssuePath(url: InternalUrl): IssuePathParts | null {
	// The shared parser only puts the first path segment into `url.host`; the
	// rest lands in `url.pathname`. Reassemble the full address before
	// splitting — see log-protocol.ts for the same pattern.
	const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

	if (!full) return null;

	const segments = full.split("/").filter(Boolean);
	if (segments.length < 2) return null;

	const owner = segments[0];
	const repo = segments[1];
	const rest = segments.slice(2);

	const number = rest.length > 0 ? Number(rest[0]) : undefined;
	if (number !== undefined && !Number.isFinite(number)) return null;

	const actionStr = rest.length > 1 ? rest[1] : undefined;
	const action =
		actionStr === "comments"
			? "comments"
			: actionStr === "sub-issues"
				? "sub-issues"
				: undefined;

	return { owner, repo, number, action };
}

export class IssueProtocolHandler implements ProtocolHandler {
	readonly scheme = "issue";
	readonly immutable = true;

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const parts = parseIssuePath(url);
		if (!parts) {
			return {
				url: url.href,
				content:
					"# Issues\n\nUse `issue://owner/repo` to list issues or `issue://owner/repo/<number>` to read a specific issue.",
				contentType: "text/markdown",
			};
		}

		const client = findGithubClient();
		if (!client) {
			return {
				url: url.href,
				content:
					"# Issues\n\nGitHub MCP server is not available. Configure the GitHub MCP server to access issues.",
				contentType: "text/markdown",
				notes: [
					"Ensure a GitHub MCP server is configured in your MCP configuration.",
				],
			};
		}

		try {
			// issue://owner/repo — list open issues
			if (parts.number === undefined) {
				const result = await client.callTool("issues_list", {
					owner: parts.owner,
					repo: parts.repo,
					state: "open",
					sort: "created",
					direction: "desc",
					per_page: 30,
				});
				const issues =
					(result as { issues?: Array<Record<string, unknown>> })?.issues ??
					(result as { items?: Array<Record<string, unknown>> })?.items ??
					[];

				const lines = (Array.isArray(issues) ? issues : []).map(
					(issue: Record<string, unknown>) => {
						const num = issue.number ?? issue.issue_number ?? "?";
						const title = issue.title ?? "Untitled";
						return `- #${num}: ${title}`;
					},
				);
				return {
					url: url.href,
					content: `# Issues: ${parts.owner}/${parts.repo}\n\n${lines.join("\n") || "No open issues."}`,
					contentType: "text/markdown",
				};
			}

			const issueNum = parts.number;

			// issue://owner/repo/42 — get issue details
			if (parts.action === undefined) {
				const result = await client.callTool("issues_read", {
					owner: parts.owner,
					repo: parts.repo,
					issue_number: issueNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// issue://owner/repo/42/comments — get issue comments
			if (parts.action === "comments") {
				const result = await client.callTool("issues_get_comments", {
					owner: parts.owner,
					repo: parts.repo,
					issue_number: issueNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// issue://owner/repo/42/sub-issues — get sub-issues
			if (parts.action === "sub-issues") {
				const result = await client.callTool("issues_get_sub_issues", {
					owner: parts.owner,
					repo: parts.repo,
					issue_number: issueNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`issue:// read error: ${message}`);
		}

		throw new Error(`Unknown issue:// action: ${parts.action}`);
	}
}
