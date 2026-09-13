// ── pr:// protocol handler ───────────────────────────────────────────────────
// Resolves GitHub pull requests via the GitHub MCP server.
// URL forms:
//   pr://                          — prompt for repo
//   pr://owner/repo                — list open PRs
//   pr://owner/repo/1428           — get PR #1428
//   pr://owner/repo/1428/files     — get PR files
//   pr://owner/repo/1428/comments  — get PR comments
//   pr://owner/repo/1428/reviews   — get PR reviews
//   pr://owner/repo/1428/commits   — get PR commits

import { findGithubClient } from "./github-client";
import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";

/** Format a tool result value for display. */
function formatValue(value: unknown): string {
	if (value === null || value === undefined) return "[null]";
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	if (Array.isArray(value)) {
		if (value.length === 0) return "[]";
		if (value.length <= 5) {
			return JSON.stringify(value, null, 2);
		}
		return JSON.stringify(value.slice(0, 5), null, 2) + `\n  ... (${value.length - 5} more items)`;
	}
	if (typeof value === "object") {
		return JSON.stringify(value, null, 2);
	}
	return String(value);
}

interface PrPathParts {
	owner: string;
	repo: string;
	number?: number;
	action?: "files" | "comments" | "reviews" | "commits";
}

function parsePrPath(url: InternalUrl): PrPathParts | null {
	const hostname = url.host;

	if (!hostname) return null;

	const segments = hostname.split("/").filter(Boolean);
	if (segments.length < 2) return null;

	const owner = segments[0];
	const repo = segments[1];
	const rest = segments.slice(2);

	const number = rest.length > 0 ? Number(rest[0]) : undefined;
	if (number && !Number.isFinite(number)) return null;

	const actionStr = rest.length > 1 ? rest[1] : undefined;
	const action = actionStr === "files" ? "files"
		: actionStr === "comments" ? "comments"
		: actionStr === "reviews" ? "reviews"
		: actionStr === "commits" ? "commits"
		: undefined;

	return { owner, repo, number, action };
}

export class PrProtocolHandler implements ProtocolHandler {
	readonly scheme = "pr";

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const parts = parsePrPath(url);
		if (!parts) {
			return {
				url: url.href,
				content: "# Pull Requests\n\nUse `pr://owner/repo` to list PRs or `pr://owner/repo/<number>` to read a specific PR.",
				contentType: "text/markdown",
			};
		}

		const client = findGithubClient();
		if (!client) {
			return {
				url: url.href,
				content: "# Pull Requests\n\nGitHub MCP server is not available. Configure the GitHub MCP server to access pull requests.",
				contentType: "text/markdown",
				notes: ["Ensure a GitHub MCP server is configured in your MCP configuration."],
			};
		}

		try {
			// pr://owner/repo — list open PRs
			if (parts.number === undefined) {
				const result = await client.callTool("pull_requests_list", {
					owner: parts.owner,
					repo: parts.repo,
					state: "open",
					sort: "created",
					direction: "desc",
					per_page: 30,
				});
				const prs = (result as { pull_requests?: Array<Record<string, unknown>> })?.pull_requests
					?? (result as { items?: Array<Record<string, unknown>> })?.items
					?? (result as { pullRequests?: Array<Record<string, unknown>> })?.pullRequests
					?? [];

				const lines = (Array.isArray(prs) ? prs : []).map((pr: Record<string, unknown>) => {
					const num = pr.number ?? pr.pr_number ?? "?";
					const head = pr.head as Record<string, unknown> | undefined;
					const title = pr.title ?? (head?.label as string | undefined) ?? "Untitled";
					return `- #${num}: ${title}`;
				});
				return {
					url: url.href,
					content: `# Pull Requests: ${parts.owner}/${parts.repo}\n\n${lines.join("\n") || "No open pull requests."}`,
					contentType: "text/markdown",
				};
			}

			const prNum = parts.number;

			// pr://owner/repo/1428 — get PR details
			if (parts.action === undefined) {
				const result = await client.callTool("pull_requests_read", {
					owner: parts.owner,
					repo: parts.repo,
					number: prNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// pr://owner/repo/1428/files — get PR files
			if (parts.action === "files") {
				const result = await client.callTool("pull_requests_get_files", {
					owner: parts.owner,
					repo: parts.repo,
					number: prNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// pr://owner/repo/1428/comments — get PR comments
			if (parts.action === "comments") {
				const result = await client.callTool("pull_requests_get_comments", {
					owner: parts.owner,
					repo: parts.repo,
					number: prNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// pr://owner/repo/1428/reviews — get PR reviews
			if (parts.action === "reviews") {
				const result = await client.callTool("pull_requests_get_reviews", {
					owner: parts.owner,
					repo: parts.repo,
					number: prNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}

			// pr://owner/repo/1428/commits — get PR commits
			if (parts.action === "commits") {
				const result = await client.callTool("pull_requests_get_commits", {
					owner: parts.owner,
					repo: parts.repo,
					number: prNum,
				});
				return {
					url: url.href,
					content: formatValue(result),
					contentType: "application/json",
				};
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`pr:// read error: ${message}`);
		}

		throw new Error(`Unknown pr:// action: ${parts.action}`);
	}
}
