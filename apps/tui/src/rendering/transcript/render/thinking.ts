// ── Transcript thinking/reasoning rendering ─────────────────────────────────
// Renders assistant reasoning chunks (collapsed/summary/expanded modes) with
// code-block syntax highlighting in expanded mode.

import type {
	AssistantChunk,
	ThinkingDisplayStyle,
} from "@logician/log-runtime/sessions";
import { RESET } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import { highlight, highlightAuto } from "../highlight.ts";
import { wrapText } from "../layout.ts";
import {
	extractLangFromFence,
	renderInline,
	stripThinkingToolMarkup,
	unwrapThinkingChannel,
} from "../text-utils.ts";



export function renderThinkingChunk(
	chunk: AssistantChunk,
	_streaming: boolean,
	thinkingMode: ThinkingDisplayStyle,
	currentWidth: number,
): string[] {
	const text = stripThinkingToolMarkup(
		unwrapThinkingChannel(chunk.contentText || ""),
	);
	if (!text) return [];

	const lines: string[] = [];

	switch (thinkingMode) {
		case "collapsed": {
			const preview = text.trim().slice(0, 100);

			lines.push(
				`${theme.fgRaw("thinkingText")}${preview ? `${preview}...` : "thinking"}${RESET}`,
			);
			break;
		}
		case "summary": {
			lines.push(
				`${theme.fgRaw("thinkingText")}${text.trim().slice(0, 150)}${RESET}`,
			);
			break;
		}
		case "expanded": {
			renderThinkingExpanded(text, lines, currentWidth);
			break;
		}
	}

	return lines;
}

/**
 * Render thinking text in expanded mode with code block syntax highlighting.
 * Parses fenced code blocks, applies highlightAuto, and wraps plain text.
 */
function renderThinkingExpanded(
	text: string,
	lines: string[],
	currentWidth: number,
): void {
	const rawLines = text.split("\n");
	let inCodeBlock = false;
	let codeContent = "";
	let codeBlockLang: string | null = null;
	const fg = theme.fgRaw("thinkingText");

	for (const rawLine of rawLines) {
		if (rawLine.startsWith("```")) {
			if (inCodeBlock) {
				renderThinkingCodeBlock(codeContent, codeBlockLang, lines, false);
				inCodeBlock = false;
				codeContent = "";
				codeBlockLang = null;
			} else {
				inCodeBlock = true;
				codeBlockLang = extractLangFromFence(rawLine);
			}
			continue;
		}

		if (inCodeBlock) {
			codeContent += `${rawLine}\n`;
		} else {
			// Wrap plain text
			const wrapped = wrapText(rawLine, currentWidth - 4);
			for (const w of wrapped) {
				lines.push(`${fg}  ${renderInline(w, fg)}${RESET}`);
			}
		}
	}

	// Flush any unterminated code block
	if (inCodeBlock && codeContent) {
		renderThinkingCodeBlock(codeContent, codeBlockLang, lines, true);
	}
}

function renderThinkingCodeBlock(
	content: string,
	language: string | null,
	lines: string[],
	streaming: boolean,
): void {
	const code = content.replace(/\n$/, "");
	if (!code) return;

	let highlightedCode = code;
	try {
		const highlighted = language
			? highlight(code, language)
			: highlightAuto(code);
		highlightedCode = highlighted.value;
	} catch {
		// Unknown or incomplete languages remain readable as plain code.
	}

	const codeLines = highlightedCode.split("\n");
	const label = language || "code";
	const meta = `${label} · ${codeLines.length} line${codeLines.length === 1 ? "" : "s"}${streaming ? " · streaming" : ""}`;
	const border = theme.fgRaw("separator");
	lines.push(`${border}  ┌─${RESET} ${theme.fg("mdCode", meta)}${RESET}`);
	for (const line of codeLines) {
		lines.push(`${border}  │${RESET} ${line}${RESET}`);
	}
}
