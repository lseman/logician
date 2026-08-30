// ── Ink TUI — Markdown Renderer (synchronous) ────────────────────────────────
// Renders markdown as Ink React components using marked's lexer (AST).
// Supports: headings, bold, italic, code spans, fenced code blocks, lists,
// blockquotes, links, horizontal rules, paragraphs, tables.

import React from "react";
import { Box, Text } from "ink";
import { marked } from "marked";
import { getCurrentTheme } from "../theme";

// ── Syntax highlighting (reuses log-runtime's emphasize-based highlighter) ───
let highlightFn: ((code: string, language: string) => any) | null = null;

function ensureHighlight(): void {
	if (highlightFn) return;
	try {
		const mod = require("@logician/log-runtime/formatting");
		highlightFn = mod.highlight ?? mod.highlightAuto ?? null;
	} catch {
		highlightFn = null;
	}
}

/** Render a code block with optional syntax highlighting. */
function renderCodeBlock(code: string, lang: string): React.ReactNode[] {
	const lines = code.split("\n");
	if (!highlightFn || !lang) {
		return lines.map((line, i) => (
			<Text key={i} color="muted" wrap="wrap">{line}</Text>
		));
	}

	try {
		const result = highlightFn(code, lang);
		const text = typeof result === "string" ? result : (result as any).value || code;
		return text.split("\n").map((line: string, i: number) => (
			<Text key={i} color="muted" wrap="wrap">{line}</Text>
		));
	} catch {
		return lines.map((line, i) => (
			<Text key={i} color="muted" wrap="wrap">{line}</Text>
		));
	}
}

// ── AST walker → React nodes ────────────────────────────────────────────────

interface RenderOptions {
	maxLength?: number;
}

function renderInlineTokens(tokens: any[], opts: RenderOptions): React.ReactNode[] {
	if (!tokens) return [];
	const theme = getCurrentTheme();
	return tokens.map((token: any) => {
		switch (token.type) {
			case "text":
				return (
					<Text key={token.raw} wrap="wrap">
						{String(token.text ?? "").slice(0, opts.maxLength || 4000)}
					</Text>
				);

			case "strong":
				return (
					<Text key={token.raw} bold wrap="wrap">
						{renderInlineTokens(token.tokens || [], opts)}
					</Text>
				);

			case "em":
				return (
					<Text key={token.raw} dimColor wrap="wrap">
						{renderInlineTokens(token.tokens || [], opts)}
					</Text>
				);

			case "codespan":
				return (
					<Text key={token.raw} color={theme.fg.info as string} wrap="wrap">
						{String(token.text ?? "")}
					</Text>
				);

			case "link":
				return (
					<Text key={token.raw} color={theme.fg.accent as string} wrap="wrap">
						{String(token.text || token.href)}
					</Text>
				);

			default:
				return (
					<Text key={token.raw} wrap="wrap">
						{String(token.text ?? "")}
					</Text>
				);
		}
	});
}

function renderToken(token: any, opts: RenderOptions): React.ReactNode {
	const theme = getCurrentTheme();
	const maxLen = opts.maxLength || 4000;

	switch (token.type) {
		case "heading": {
			const depth = token.depth || 1;
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Text color={theme.fg.accent as string} bold wrap="wrap">
						{"#".repeat(depth)} {" "}
						{renderInlineTokens(token.tokens || [], opts)}
					</Text>
				</Box>
			);
		}

		case "paragraph": {
			const text = String(token.text ?? "").slice(0, maxLen);
			if (!text) return null;
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Text color={theme.fg.primary as string} wrap="wrap">
						{renderInlineTokens(token.tokens || [], opts)}
					</Text>
				</Box>
			);
		}

		case "code": {
			const lines = renderCodeBlock(token.text, token.lang || "");
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Box borderStyle="single" borderColor={theme.fg.muted as string} paddingX={1}>
						{lines}
					</Box>
				</Box>
			);
		}

		case "list": {
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					{(token.items || []).map((item: any, i: number) => renderListItem(item, opts))}
				</Box>
			);
		}

		case "blockquote": {
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1} paddingLeft={2}>
					<Text color={theme.fg.secondary as string} wrap="wrap">
						{"│ "}
						{renderInlineTokens(token.tokens || [], opts)}
					</Text>
				</Box>
			);
		}

		case "hr":
			return (
				<Box key={token.raw} marginBottom={1}>
					<Text color={theme.fg.muted as string} wrap="wrap">{"─".repeat(40)}</Text>
				</Box>
			);

		case "table": {
			const headerCells = token.header || [];
			const rows = token.rows || [];
			if (headerCells.length === 0) return null;

			const lines: React.ReactNode[] = [];
			lines.push(
				<Box key="hdr" flexDirection="row">
					{headerCells.map((cell: any, i: number) => (
						<Text key={i} color={theme.fg.accent as string} bold wrap="truncate-end">
							{String(cell.text ?? "").slice(0, 30)}{"\t"}
						</Text>
					))}
				</Box>,
			);
			lines.push(
				<Box key="sep" flexDirection="row">
					<Text color={theme.fg.muted as string} wrap="wrap">{"─".repeat(Math.min(80, headerCells.length * 20))}</Text>
				</Box>,
			);
			for (const row of rows.slice(0, 20)) {
				lines.push(
					<Box key={`row-${row[0]}`} flexDirection="row">
						{row.map((cell: any, i: number) => (
							<Text key={i} color={theme.fg.primary as string} wrap="truncate-end">
								{String(cell ?? "").slice(0, 30)}{"\t"}
							</Text>
						))}
					</Box>,
				);
			}
			if (rows.length > 20) {
				lines.push(
					<Text key="more" color={theme.fg.muted as string} wrap="wrap">
						{`… ${rows.length - 20} more rows`}
					</Text>,
				);
			}

			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					{lines}
				</Box>
			);
		}

		default: {
			const text = String(token.text ?? "").slice(0, maxLen);
			if (!text) return null;
			return (
				<Text key={token.raw} color={theme.fg.primary as string} wrap="wrap">
					{text}
				</Text>
			);
		}
	}
}

function renderListItem(item: any, opts: RenderOptions): React.ReactNode {
	const theme = getCurrentTheme();
	const marker = item.ordered ? `${item.start || 1}.` : "•";
	return (
		<Box key={item.raw} flexDirection="row" marginBottom={0}>
			<Text color={theme.fg.secondary as string} bold>{marker}</Text>
			<Text color={theme.fg.primary as string} wrap="wrap">
				{" "}{renderInlineTokens(item.tokens || [], opts)}
			</Text>
		</Box>
	);
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface MarkdownRendererProps {
	/** Raw markdown string to render. */
	markdown: string;
	/** Max characters to render (truncation). */
	maxLength?: number;
}

/**
 * Parse markdown synchronously and render as Ink React components.
 * Syntax highlighting is loaded once on first call (cached).
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
	markdown,
	maxLength = 4000,
}) => {
	ensureHighlight();

	const theme = getCurrentTheme();
	const truncated = markdown.length > maxLength ? markdown.slice(0, maxLength) + "…" : markdown;

	try {
		const tokens = marked.lexer(truncated, { gfm: true });
		return (
			<Box flexDirection="column">
				{tokens.map(token => renderToken(token, { maxLength }))}
			</Box>
		);
	} catch {
		return (
			<Text color={theme.fg.primary as string} wrap="wrap">
				{truncated}
			</Text>
		);
	}
};
