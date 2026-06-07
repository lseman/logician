// ── web_search tool ──────────────────────────────────────────────────────────────
// Search the web via SearXNG (self-hosted privacy-respecting metasearch engine).

import type { Tool, ToolContext, WebSearchConfig } from "../types.ts";
import { DEFAULT_MAX_BYTES, truncateTail } from "./truncate.ts";

export function createWebSearchTool(config: WebSearchConfig): Tool {
	const baseUrl = config.baseUrl.replace(/\/+$/, "");
	const maxResults = config.maxResults ?? 10;

	return {
		name: "web_search",
		executionMode: "parallel",
		description: `Search the web via SearXNG (${baseUrl}). Returns title, URL, and snippet for each result.`,
		parameters: {
			type: "object",
			properties: {
				query: {
					type: "string",
					description: "Search query",
				},
				engines: {
					type: "string",
					description:
						"Comma-separated SearXNG engines (e.g. google,bing,duckduckgo). Default: all enabled.",
				},
				categories: {
					type: "string",
					description:
						"SearXNG category (e.g. general, images, news, videos). Default: general.",
				},
				language: {
					type: "string",
					description: "Language code (e.g. en, en-US, de). Default: auto.",
				},
				time_range: {
					type: "string",
					description: "Time range: day, week, month, year. Default: none.",
				},
				max_results: {
					type: "number",
					description: `Max results to return (default ${maxResults}).`,
				},
			},
			required: ["query"],
		},
		execute: async (
			args: Record<string, unknown>,
			ctx: ToolContext,
		): Promise<string> => {
			const query = String(args.query);
			if (!query.trim()) return "Error: query is required";

			const engines = String(args.engines || "");
			const categories = String(args.categories || "general");
			const language = String(args.language || "");
			const timeRange = String(args.time_range || "");
			const resultsLimit = Number(args.max_results) || maxResults;

			const params = new URLSearchParams({
				q: query,
				format: "json",
				categories,
				...(engines && { engines }),
				...(language && { language }),
				...(timeRange && { time_range: timeRange }),
				pageno: "1",
			});

			const url = `${baseUrl}/search?${params.toString()}`;

			try {
				const response = await fetch(url, {
					signal: ctx.signal,
					headers: { Accept: "application/json" },
				});

				if (!response.ok) {
					const text = await response.text();
					return `Error: SearXNG request failed (${response.status}): ${text.slice(0, 500)}`;
				}

				const data = (await response.json()) as {
					results?: Array<{
						title?: string;
						url?: string;
						content?: string;
						engine?: string;
					}>;
				};

				const results = data.results || [];
				if (!results.length) return "No results found.";

				const limited = results.slice(0, resultsLimit);
				const lines = limited.map((r, i) =>
					[
						`${i + 1}. ${r.title || "(no title)"}`,
						`   URL: ${r.url || "(no URL)"}`,
						...(r.content ? [`   Snippet: ${r.content}`] : []),
						...(r.engine ? [`   Engine: ${r.engine}`] : []),
					].join("\n"),
				);

				const body = lines.join("\n\n");
				const t = truncateTail(body, { maxBytes: DEFAULT_MAX_BYTES });
				return t.truncated
					? `${t.content}\n... [truncated, ${results.length} total results]`
					: body;
			} catch (e: unknown) {
				const error = e as Error;
				return `Error: ${error.message || String(e)}`;
			}
		},
	};
}
