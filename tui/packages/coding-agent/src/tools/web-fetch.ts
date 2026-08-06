// ── web_fetch tool ───────────────────────────────────────────────────────────────
// Fetch and extract readable content from a web page.

import type { Tool, ToolContext } from "@logician/agent-core/agent/types.ts";
import { DEFAULT_MAX_BYTES, truncateTail } from "./truncate.ts";

export const web_fetch: Tool = {
	readOnly: true,
	name: "web_fetch",
	label: "Web Fetch",
	hookAliases: ["WebFetch"],
	executionMode: "parallel",
	description: `Fetch and extract readable content from a web page. Strips navigation, ads, scripts. Returns title, description, and main text content. Truncated to ${DEFAULT_MAX_BYTES / 1024}KB.`,
	promptSnippet: "Fetch and read content from a specific URL",
	promptGuidelines: [
		"Use web_fetch to read full URL content; web_search finds URLs, fetch reads them",
	],
	parameters: {
		type: "object",
		properties: {
			url: {
				type: "string",
				description: "URL to fetch",
			},
			max_length: {
				type: "number",
				description: `Max output length in characters (default ${DEFAULT_MAX_BYTES}).`,
			},
			timeout: {
				type: "number",
				description: "Request timeout in ms (default 15000)",
			},
		},
		required: ["url"],
	},
	execute: async (
		args: Record<string, unknown>,
		ctx: ToolContext,
	): Promise<string> => {
		const url = String(args.url);
		if (!url.trim()) return "Error: url is required";

		// Basic URL validation
		try {
			new URL(url);
		} catch (_e: unknown) {
			return `Error: Invalid URL: ${url}`;
		}

		// Only allow http/https
		const parsed = new URL(url);
		if (!["http:", "https:"].includes(parsed.protocol)) {
			return `Error: Only http/https URLs allowed, got: ${parsed.protocol}`;
		}

		const maxLength = Number(args.max_length) || DEFAULT_MAX_BYTES;
		const timeout = Number(args.timeout) || 15000;

		try {
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), timeout);
			ctx.signal?.addEventListener("abort", () => controller.abort(), {
				once: true,
			});

			const response = await fetch(url, {
				signal: controller.signal,
				headers: {
					Accept:
						"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
					"User-Agent":
						"LogicianTUI/1.0 (coding agent; +https://github.com/tui)",
				},
			});

			clearTimeout(timer);

			if (!response.ok) {
				return `Error: HTTP ${response.status} ${response.statusText} for ${url}`;
			}

			const html = await response.text();
			const extracted = extractTextFromHtml(html, maxLength);

			return extracted;
		} catch (err: unknown) {
			const error = err as Error;
			if (error.name === "AbortError") {
				return `Error: Request timed out after ${timeout}ms`;
			}
			return `Error: ${error.message || String(err)}`;
		}
	},
};

// ── Lightweight HTML text extraction ──────────────────────────────────────────────
// Strips script, style, nav, header, footer, sidebar, and other non-content
// elements. Extracts text from heading and paragraph tags. No external deps.

function extractTextFromHtml(html: string, maxLength: number): string {
	// Remove script and style blocks
	let text = html
		.replace(/<script[\s\S]*?<\/script>/gi, "")
		.replace(/<style[\s\S]*?<\/style>/gi, "")
		.replace(/<noscript[\s\S]*?<\/noscript>/gi, "");

	// Remove common non-content elements
	text = text.replace(
		/<(nav|header|footer|aside|sidebar|form|button|iframe|ad|advertisement)[^>]*>[\s\S]*?<\/\1>/gi,
		"",
	);

	// Replace block-level tags with newlines
	text = text.replace(
		/<\/(div|p|br|h[1-6]|li|tr|section|article|main|blockquote)>/gi,
		"\n",
	);

	// Remove remaining tags
	text = text.replace(/<[^>]+>/g, " ");

	// Decode common HTML entities
	text = text
		.replace(/&nbsp;/g, " ")
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")
		.replace(/&apos;/g, "'");

	// Collapse whitespace
	text = text.replace(/\s+/g, " ").trim();

	// Extract page title if present
	let title = "";
	const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
	if (titleMatch) {
		title = titleMatch[1].trim();
	}

	// Extract meta description if present
	let description = "";
	const descMatch = html.match(
		/<meta[^>]*name=["']description["'][^>]*content=["']([^"']+)["']/i,
	);
	if (descMatch) {
		description = descMatch[1].trim();
	}

	// Build output
	const parts: string[] = [];
	if (title) parts.push(`Title: ${title}`);
	if (description) parts.push(`Description: ${description}`);
	parts.push("");
	parts.push("Content:");
	parts.push(text);

	const body = parts.join("\n");
	const t = truncateTail(body, { maxBytes: maxLength });
	return t.truncated
		? `${t.content}\n... [truncated, ${body.length} chars total]`
		: body;
}
