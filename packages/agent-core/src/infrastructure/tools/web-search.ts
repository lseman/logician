// ── web_search tool ──────────────────────────────────────────────────────────────
// Search the web via SearXNG (self-hosted privacy-respecting metasearch engine).

import type { WebSearchConfig } from "../../core/types/types-config.ts";
import type { Tool, ToolContext } from "../../core/types/types-messages.ts";
import { DEFAULT_MAX_BYTES, truncateTail } from "./utils/truncate.ts";

type FetchLike = (
	input: string | URL | Request,
	init?: RequestInit,
) => Promise<Response>;

export function createWebSearchTool(
	config: WebSearchConfig,
	fetchImpl: FetchLike = globalThis.fetch.bind(globalThis),
): Tool {
	const baseUrl = config.baseUrl.replace(/\/+$/, "");
	const maxResults = config.maxResults ?? 10;

	return {
		name: "web_search",
		label: "Web Search",
		readOnly: true,
		hookAliases: ["WebSearch"],
		executionMode: "parallel",
		description: `Search the web via SearXNG (${baseUrl}). Returns title, URL, and snippet for each result.`,
		promptSnippet: "Search the web for current information",
		promptGuidelines: [
			"Use web_search for recent information not in training data",
		],
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
				page: {
					type: "number",
					description: "Results page, from 1 to 20. Default: 1.",
				},
				safesearch: {
					type: "number",
					enum: [0, 1, 2],
					description: "Safe-search level: 0 off, 1 moderate, 2 strict.",
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
			const requestedLimit = Number(args.max_results) || maxResults;
			const resultsLimit = Math.min(
				100,
				Math.max(1, Math.floor(requestedLimit)),
			);
			const page = Math.min(
				20,
				Math.max(1, Math.floor(Number(args.page) || 1)),
			);
			const safesearch = [0, 1, 2].includes(Number(args.safesearch))
				? Number(args.safesearch)
				: 1;

			const params = new URLSearchParams({
				q: query,
				format: "json",
				categories,
				...(engines && { engines }),
				...(language && { language }),
				...(timeRange && { time_range: timeRange }),
				pageno: String(page),
				safesearch: String(safesearch),
			});

			const url = `${baseUrl}/search?${params.toString()}`;

			try {
				const response = await fetchImpl(url, {
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
						engines?: string[];
						publishedDate?: string | null;
						score?: number;
					}>;
					answers?: string[];
					suggestions?: string[];
				};

				const results = data.results || [];
				if (!results.length) return "No results found.";

				const limited = results.slice(0, resultsLimit);
				const lines = limited.map((r, i) =>
					[
						`${i + 1}. ${r.title || "(no title)"}`,
						`   URL: ${r.url || "(no URL)"}`,
						...(r.content ? [`   Snippet: ${r.content}`] : []),
						...(r.publishedDate ? [`   Published: ${r.publishedDate}`] : []),
						...(r.engines?.length
							? [`   Engines: ${r.engines.join(", ")}`]
							: r.engine
								? [`   Engine: ${r.engine}`]
								: []),
					].join("\n"),
				);

				const prefix = data.answers?.length
					? [`Answers: ${data.answers.join(" · ")}`, ""]
					: [];
				const suffix = data.suggestions?.length
					? ["", `Related: ${data.suggestions.slice(0, 8).join(" · ")}`]
					: [];
				const body = [...prefix, ...lines, ...suffix].join("\n\n");
				const t = truncateTail(body, { maxBytes: DEFAULT_MAX_BYTES });
				return t.truncated
					? `${t.content}\n... [truncated, ${results.length} total results]`
					: body;
			} catch (err: unknown) {
				const error = err as Error;
				return `Error: ${error.message || String(err)}`;
			}
		},
	};
}
