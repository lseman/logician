import assert from "node:assert/strict";
import { test } from "node:test";
import { createWebSearchTool } from "../tools/web-search.ts";

void test("web_search sends bounded SearXNG parameters and renders rich metadata", async () => {
	let requestedUrl = "";
	const fakeFetch = async (
		input: string | URL | Request,
	): Promise<Response> => {
		requestedUrl = String(input);
		return new Response(
			JSON.stringify({
				answers: ["Direct answer"],
				results: [
					{
						title: "Result",
						url: "https://example.test/result",
						content: "Useful snippet",
						engines: ["duckduckgo", "startpage"],
						publishedDate: "2026-07-20T00:00:00Z",
					},
				],
				suggestions: ["related query"],
			}),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	};
	const tool = createWebSearchTool(
		{ baseUrl: "http://127.0.0.1:8090/", maxResults: 10 },
		fakeFetch,
	);
	const result = await tool.execute(
		{
			query: "agent harness",
			page: 999,
			safesearch: 2,
			max_results: 999,
			engines: "duckduckgo,startpage",
		},
		{},
	);
	const output = typeof result === "string" ? result : result.content;
	const url = new URL(requestedUrl);

	assert.equal(url.origin, "http://127.0.0.1:8090");
	assert.equal(url.searchParams.get("format"), "json");
	assert.equal(url.searchParams.get("pageno"), "20");
	assert.equal(url.searchParams.get("safesearch"), "2");
	assert.match(output, /Direct answer/);
	assert.match(output, /duckduckgo, startpage/);
	assert.match(output, /Published: 2026-07-20/);
	assert.match(output, /Related: related query/);
});

void test("web_search reports SearXNG HTTP errors without throwing", async () => {
	const tool = createWebSearchTool(
		{ baseUrl: "http://search.test" },
		async () => new Response("JSON output is disabled", { status: 403 }),
	);
	const result = await tool.execute({ query: "test" }, {});
	const output = typeof result === "string" ? result : result.content;
	assert.match(output, /SearXNG request failed \(403\)/);
	assert.match(output, /JSON output is disabled/);
});
