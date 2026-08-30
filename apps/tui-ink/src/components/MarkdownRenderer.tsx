// ── Ink TUI — Markdown Renderer (synchronous) ────────────────────────────────
// Renders markdown with the same semantic color tokens as the old TUI:
//   mdHeading, mdCode, mdCodeBlock, mdCodeBlockBg, mdCodeBlockBorder,
//   mdLink, mdQuote, mdListBullet, jsonKey/jsonString/jsonNumber/
//   jsonKeyword/jsonPunctuation, diffAdded/diffRemoved/diffHunk/diffMeta.

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

/** Colorize a single inline token for the old TUI's renderInline behavior. */
function colorizeInlineToken(
	token: any,
	baseColor: string | undefined,
	opts: RenderOptions,
): React.ReactNode[] {
	const theme = getCurrentTheme();
	switch (token.type) {
		case "text":
			return [
				<Text key={token.raw} color={baseColor} wrap="wrap">
					{String(token.text ?? "").slice(0, opts.maxLength || 4000)}
				</Text>,
			];

		case "strong":
			return [
				<Text key={token.raw} bold color={baseColor} wrap="wrap">
					{colorizeInlineToken(token.tokens || [], baseColor, opts)}
				</Text>,
			];

		case "em":
			return [
				<Text key={token.raw} color={baseColor} italic wrap="wrap">
					{colorizeInlineToken(token.tokens || [], baseColor, opts)}
				</Text>,
			];

		case "codespan": {
			const code = String(token.text ?? "");
			return [
				<Text key={token.raw} color={theme.fg.mdCode} bold wrap="wrap">
					{`\`${code}\``}
				</Text>,
			];
		}

		case "link": {
			const text = String(token.text || token.href);
			return [
				<Text key={token.raw} color={theme.fg.mdLink} wrap="wrap">
					{text}
				</Text>,
			];
		}

		default:
			return [
				<Text key={token.raw} color={baseColor} wrap="wrap">
					{String(token.text ?? "")}
				</Text>,
			];
	}
}

/** Render a code block with syntax highlighting, matching old TUI style. */
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

// ── Heading styles — per-level color + decoration (matches old TUI) ──────────

interface HeadingStyle {
	color: string | undefined;
	decoration: "bold-underline" | "bold" | "dim" | "none";
}

function getHeadingStyles(): HeadingStyle[] {
	const theme = getCurrentTheme();
	return [
		{ color: theme.fg.mdHeading, decoration: "bold-underline" }, // H1
		{ color: theme.fg.accent, decoration: "bold" },              // H2
		{ color: theme.fg.mdHeading, decoration: "bold" },           // H3
		{ color: theme.fg.warning, decoration: "none" },             // H4
		{ color: theme.fg.muted, decoration: "none" },               // H5
		{ color: theme.fg.dim, decoration: "dim" },                  // H6
	];
}

// ── Block-level token rendering ──────────────────────────────────────────────

interface RenderOptions {
	maxLength?: number;
	baseColor?: string; // base text color for inline content (e.g. assistantText)
}

function renderToken(token: any, opts: RenderOptions): React.ReactNode {
	const theme = getCurrentTheme();
	const maxLen = opts.maxLength || 4000;
	const baseColor = opts.baseColor ?? theme.fg.text ?? "";

	switch (token.type) {
		case "heading": {
			const depth = token.depth || 1;
			const style = getHeadingStyles()[Math.min(depth - 1, 5)];
			const marker = depth <= 2 ? "▌ " : "";
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Text color={style.color} bold wrap="wrap">
						{marker}{renderInlineTokens(token.tokens || [], baseColor, opts)}
					</Text>
				</Box>
			);
		}

		case "paragraph": {
			const text = String(token.text ?? "").slice(0, maxLen);
			if (!text) return null;
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Text color={baseColor} wrap="wrap">
						{renderInlineTokens(token.tokens || [], baseColor, opts)}
					</Text>
				</Box>
			);
		}

		case "code": {
			const lines = renderCodeBlock(token.text, token.lang || "");
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Box borderStyle="single" borderColor={theme.fg.mdCodeBlockBorder} paddingX={1}>
						{lines}
					</Box>
				</Box>
			);
		}

		case "code-block": {
			const fenceLang = token.lang || "";
			const lines = renderCodeBlock(token.text, fenceLang);
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					<Box borderStyle="single" borderColor={theme.fg.mdCodeBlockBorder} paddingX={1}>
						<Text color={theme.fg.dim} dimColor wrap="wrap">
							{"┌─ "}{fenceLang || "code"}
						</Text>
						{lines}
						<Text color={theme.fg.dim} dimColor wrap="wrap">
							{"\n└─ "}{token.text.split("\n").length} line{token.text.split("\n").length === 1 ? "" : "s"}
						</Text>
					</Box>
				</Box>
			);
		}

		case "list": {
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1}>
					{(token.items || []).map((item: any, i: number) => renderListItem(item, baseColor, opts))}
				</Box>
			);
		}

		case "blockquote": {
			return (
				<Box key={token.raw} flexDirection="column" marginBottom={1} paddingLeft={2}>
					<Text color={theme.fg.mdQuote} dimColor wrap="wrap">
						{"▏ "}
						{renderInlineTokens(token.tokens || [], baseColor, opts)}
					</Text>
				</Box>
			);
		}

		case "hr":
			return (
				<Box key={token.raw} marginBottom={1}>
					<Text color={theme.fg.dim} dimColor wrap="wrap">{"─".repeat(40)}</Text>
				</Box>
			);

		case "table": {
			const headerCells = token.header || [];
			const rows = token.rows || [];
			if (headerCells.length === 0) return null;

			const lines: React.ReactNode[] = [];
			const borderColor = theme.fg.borderMuted;

			// Top frame
			lines.push(
				<Box key="top" flexDirection="row">
					<Text color={borderColor} wrap="wrap">{"┌"}</Text>
					{headerCells.map((_: any, i: number) => (
						<Text key={`top-${i}`} color={borderColor} wrap="wrap">{"─".repeat(20)}</Text>
					))}
					<Text key="top-r" color={borderColor} wrap="wrap">{"┐"}</Text>
				</Box>,
			);

			// Header row
			lines.push(
				<Box key="hdr" flexDirection="row">
					<Text color={borderColor} wrap="wrap">{"│"}</Text>
					{headerCells.map((cell: any, i: number) => (
						<Text key={i} color={theme.fg.assistantText} bold wrap="truncate-end">
							{` ${String(cell.text ?? "").slice(0, 30)} `}
						</Text>
					))}
					<Text color={borderColor} wrap="wrap">{"│"}</Text>
				</Box>,
			);

			// Separator
			lines.push(
				<Box key="sep" flexDirection="row">
					<Text color={borderColor} wrap="wrap">{"├"}</Text>
					{headerCells.map((_: any, i: number) => (
						<Text key={`sep-${i}`} color={borderColor} wrap="wrap">{"─".repeat(20)}</Text>
					))}
					<Text key="sep-r" color={borderColor} wrap="wrap">{"┤"}</Text>
				</Box>,
			);

			// Data rows
			for (const row of rows.slice(0, 20)) {
				const isAlt = rows.indexOf(row) % 2 === 1;
				const rowColor = isAlt ? theme.fg.dim : theme.fg.assistantText;
				lines.push(
					<Box key={`row-${rows.indexOf(row)}`} flexDirection="row">
						<Text color={borderColor} wrap="wrap">{"│"}</Text>
						{row.map((cell: any, i: number) => (
							<Text key={i} color={rowColor} wrap="truncate-end">
								{` ${String(cell ?? "").slice(0, 30)} `}
							</Text>
						))}
						<Text color={borderColor} wrap="wrap">{"│"}</Text>
					</Box>,
				);
			}

			// Bottom frame
			lines.push(
				<Box key="bot" flexDirection="row">
					<Text color={borderColor} wrap="wrap">{"└"}</Text>
					{headerCells.map((_: any, i: number) => (
						<Text key={`bot-${i}`} color={borderColor} wrap="wrap">{"─".repeat(20)}</Text>
					))}
					<Text key="bot-r" color={borderColor} wrap="wrap">{"┘"}</Text>
				</Box>,
			);

			if (rows.length > 20) {
				lines.push(
					<Text key="more" color={theme.fg.dim} dimColor wrap="wrap">
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
				<Text key={token.raw} color={baseColor} wrap="wrap">
					{text}
				</Text>
			);
		}
	}
}

function renderListItem(item: any, baseColor: string, opts: RenderOptions): React.ReactNode {
	const theme = getCurrentTheme();
	const marker = item.ordered ? `${item.start || 1}.` : "•";
	return (
		<Box key={item.raw} flexDirection="row" marginBottom={0}>
			<Text color={theme.fg.mdListBullet} bold>{marker}</Text>
			<Text color={baseColor} wrap="wrap">
				{" "}{renderInlineTokens(item.tokens || [], baseColor, opts)}
			</Text>
		</Box>
	);
}

function renderInlineTokens(tokens: any[], baseColor: string | undefined, opts: RenderOptions): React.ReactNode {
	if (!tokens) return [];
	return tokens.map((token: any) => {
		switch (token.type) {
			case "text":
				return (
					<Text key={token.raw} color={baseColor} wrap="wrap">
						{String(token.text ?? "").slice(0, opts.maxLength || 4000)}
					</Text>
				);

			case "strong":
				return (
					<Text key={token.raw} bold color={baseColor} wrap="wrap">
						{renderInlineTokens(token.tokens || [], baseColor, opts)}
					</Text>
				);

			case "em":
				return (
					<Text key={token.raw} color={baseColor} italic wrap="wrap">
						{renderInlineTokens(token.tokens || [], baseColor, opts)}
					</Text>
				);

			case "codespan": {
				const code = String(token.text ?? "");
				return (
					<Text key={token.raw} color={getCurrentTheme().fg.mdCode} bold wrap="wrap">
						{`\`${code}\``}
					</Text>
				);
			}

			case "link": {
				const text = String(token.text || token.href);
				return (
					<Text key={token.raw} color={getCurrentTheme().fg.mdLink} wrap="wrap">
						{text}
					</Text>
				);
			}

			default:
				return (
					<Text key={token.raw} color={baseColor} wrap="wrap">
						{String(token.text ?? "")}
					</Text>
				);
		}
	});
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface MarkdownRendererProps {
	/** Raw markdown string to render. */
	markdown: string;
	/** Max characters to render (truncation). */
	maxLength?: number;
	/** Base color for content text (defaults to assistantText). */
	baseColor?: string;
}

/**
 * Parse markdown synchronously and render as Ink React components.
 * Syntax highlighting is loaded once on first call (cached).
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
	markdown,
	maxLength = 4000,
	baseColor,
}) => {
	ensureHighlight();

	const theme = getCurrentTheme();
	const truncated = markdown.length > maxLength ? markdown.slice(0, maxLength) + "…" : markdown;
	const effectiveBaseColor = baseColor ?? theme.fg.assistantText;

	try {
		const tokens = marked.lexer(truncated, { gfm: true });
		return (
			<Box flexDirection="column">
				{tokens.map(token => renderToken(token, { maxLength, baseColor: effectiveBaseColor }))}
			</Box>
		);
	} catch {
		return (
			<Text color={effectiveBaseColor} wrap="wrap">
				{truncated}
			</Text>
		);
	}
};
