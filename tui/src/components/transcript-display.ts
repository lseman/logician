// ── Transcript display component ────────────────────────────────────────────────
// Renders the full conversation history with streaming support and markdown.
// Chunks are interleaved in chronological order: thinking → content → tool → ...

import { highlightAuto } from "../agent-core/syntax-highlighter.ts";
import type {
	AssistantChunk,
	ThinkingDisplayStyle,
	ToolExecution,
	Turn,
} from "../transcript.ts";
import {
	type Component,
	clampLineToWidth,
	type Scrollable,
	visibleWidth,
} from "../tui-core.ts";

interface Theme {
	userColor: string;
	assistantColor: string;
	thinkingColor: string;
	toolColor: string;
	errorColor: string;
	separatorColor: string;
}

const DEFAULT_THEME: Theme = {
	userColor: "\x1b[38;5;111m",
	assistantColor: "\x1b[38;5;188m",
	thinkingColor: "\x1b[38;5;220m",
	toolColor: "\x1b[38;5;141m",
	errorColor: "\x1b[38;5;203m",
	separatorColor: "\x1b[38;5;240m",
};

const USER_PREFIX = "\x1b[1m\x1b[38;5;111mYOU \x1b[0m";
const ASSISTANT_PREFIX = "";
const SYSTEM_PREFIX = "";
const THINKING_PREFIX = "\x1b[38;5;220mTHINK \x1b[0m";
const TOOL_PREFIX = "\x1b[38;5;141mTOOL \x1b[0m";
const ERROR_PREFIX = "\x1b[38;5;203m✗ \x1b[0m";
const TOOL_DETAIL_PREFIX = "\x1b[38;5;239m│ \x1b[0m";
const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const UNDERLINE = "\x1b[4m";
const CODE_BLOCK_COLOR = "\x1b[38;5;241m";
const CODE_INLINE_COLOR = "\x1b[38;5;80m";
const CODE_BLOCK_BG = "\x1b[48;5;235m";
const DIFF_ADD_COLOR = "\x1b[38;5;114m";
const DIFF_REMOVE_COLOR = "\x1b[38;5;203m";
const DIFF_HUNK_COLOR = "\x1b[38;5;220m";
const DIFF_META_COLOR = "\x1b[38;5;111m";
const TERMINAL_COLOR = "\x1b[38;5;250m";
const WARNING_COLOR = "\x1b[38;5;214m";

// Heading palette — distinct color + weight per level so #/##/### read apart.
const HEADING_STYLES: Array<{ color: string; deco: string }> = [
	{ color: "\x1b[38;5;213m", deco: BOLD + UNDERLINE }, // h1 — bright magenta, bold underline
	{ color: "\x1b[38;5;81m", deco: BOLD }, // h2 — cyan, bold
	{ color: "\x1b[38;5;114m", deco: BOLD }, // h3 — green, bold
	{ color: "\x1b[38;5;179m", deco: "" }, // h4 — gold
	{ color: "\x1b[38;5;146m", deco: "" }, // h5 — lavender
	{ color: "\x1b[38;5;245m", deco: DIM }, // h6 — grey, dim
];

// Pseudo-XML emphasis tags like <EXTREMELY-IMPORTANT> … </EXTREMELY-IMPORTANT>.
const TAG_COLOR = "\x1b[38;5;208m"; // orange, attention-grabbing

// Bullet glyphs cycle by nesting depth.
const BULLET_GLYPHS = ["•", "◦", "▪", "‣"];
const BULLET_COLOR = "\x1b[38;5;147m"; // soft periwinkle, distinct from text
const ORDERED_COLOR = "\x1b[38;5;147m";

// ── Code fence language extraction ────────────────────────────────────────────

function extractLangFromFence(line: string): string | null {
	const m = /^```(\w+)/.exec(line.trim());
	return m ? m[1].toLowerCase() : null;
}

// ── Embedded reasoning stripping ──────────────────────────────────────────────
function stripThinkTags(text: string): string {
	if (!text?.includes("<think")) return text;
	return text
		.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, "")
		.replace(/<think(?:ing)?>[\s\S]*$/i, "")
		.trimStart();
}

// ── Inline markdown renderer ──────────────────────────────────────────────────

function matchTagAt(text: string, i: number): number {
	const m = /^<\/?[A-Za-z][\w-]*(?:\s[^<>]*)?\/?>/.exec(text.slice(i, i + 200));
	return m ? m[0].length : 0;
}

function renderInline(text: string, baseColor: string): string {
	let out = baseColor;
	let i = 0;
	while (i < text.length) {
		if (text[i] === "<") {
			const tag = matchTagAt(text, i);
			if (tag) {
				out += TAG_COLOR + BOLD + text.slice(i, i + tag) + RESET + baseColor;
				i += tag;
				continue;
			}
		}
		if (text.startsWith("```", i)) {
			const end = text.indexOf("```", i + 3);
			if (end !== -1) {
				out +=
					CODE_BLOCK_COLOR +
					"```" +
					text.slice(i + 3, end) +
					"```" +
					RESET +
					baseColor;
				i = end + 3;
				continue;
			}
		}
		if (text.startsWith("**", i)) {
			const end = text.indexOf("**", i + 2);
			if (end !== -1) {
				out += BOLD + text.slice(i + 2, end) + RESET + baseColor;
				i = end + 2;
				continue;
			}
		}
		if (text[i] === "`") {
			const end = text.indexOf("`", i + 1);
			if (end !== -1) {
				out +=
					CODE_INLINE_COLOR +
					BOLD +
					"`" +
					text.slice(i + 1, end) +
					"`" +
					RESET +
					baseColor;
				i = end + 1;
				continue;
			}
		}
		if (text[i] === "*" && i + 1 < text.length && text[i + 1] !== "*") {
			const end = text.indexOf("*", i + 1);
			if (end !== -1 && end !== i + 1) {
				out += `\x1b[3m${text.slice(i + 1, end)}${RESET}${baseColor}`;
				i = end + 1;
				continue;
			}
		}
		out += text[i];
		i++;
	}
	return out + RESET;
}

// ── Block-level markdown ──────────────────────────────────────────────────────

function renderMarkdownLine(line: string, baseColor: string): string {
	// Headings
	const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
	if (heading) {
		const level = heading[1].length;
		const style = HEADING_STYLES[level - 1];
		const marker = level <= 2 ? "▌ " : "";
		return `${style.color}${marker}${style.deco}${style.color}${renderInlinePlain(heading[2])}${RESET}`;
	}

	// Horizontal rule
	if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
		return `${DIM}\x1b[38;5;239m${"─".repeat(40)}${RESET}`;
	}

	// List items
	const listMatch = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
	if (listMatch) {
		const indent = listMatch[1];
		const lmarker = listMatch[2];
		const rest = listMatch[3];
		const depth = Math.floor(indent.length / 2);
		if (/^\d/.test(lmarker)) {
			return `${indent}${ORDERED_COLOR}${BOLD}${lmarker}${RESET} ${renderInline(rest, baseColor)}`;
		}
		const glyph = BULLET_GLYPHS[depth % BULLET_GLYPHS.length];
		return `${indent}${BULLET_COLOR}${glyph}${RESET} ${renderInline(rest, baseColor)}`;
	}

	// Blockquote
	const quote = line.match(/^(\s*)>\s?(.*)$/);
	if (quote) {
		return `${quote[1]}${DIM}\x1b[38;5;245m▏ ${renderInlinePlain(quote[2])}${RESET}`;
	}

	return renderInline(line, baseColor);
}

// JSON syntax colors
const JSON_KEY_COLOR = "\x1b[38;5;111m";
const JSON_STR_COLOR = "\x1b[38;5;114m";
const JSON_NUM_COLOR = "\x1b[38;5;179m";
const JSON_KW_COLOR = "\x1b[38;5;213m";
const JSON_PUNCT_COLOR = "\x1b[38;5;244m";

function formatJsonLine(rawLine: string): string[] | null {
	const trimmed = rawLine.trim();
	if (trimmed.length < 2) return null;
	const first = trimmed[0];
	const last = trimmed[trimmed.length - 1];
	const looksJson =
		(first === "{" && last === "}") || (first === "[" && last === "]");
	if (!looksJson) return null;
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
	} catch {
		return null;
	}
	if (parsed === null || typeof parsed !== "object") return null;
	if (
		Array.isArray(parsed)
			? parsed.length === 0
			: Object.keys(parsed).length === 0
	) {
		return null;
	}
	const pretty = JSON.stringify(parsed, null, 2);
	return pretty.split("\n").map(colorizeJsonRow);
}

function colorizeJsonRow(row: string): string {
	const indentMatch = row.match(/^(\s*)(.*)$/s);
	const indent = indentMatch?.[1] ?? "";
	let body = indentMatch?.[2] ?? row;

	const keyMatch = body.match(/^("(?:[^"\\]|\\.)*")(\s*:\s*)(.*)$/s);
	let prefix = "";
	if (keyMatch) {
		prefix = `${JSON_KEY_COLOR}${keyMatch[1]}${RESET}${JSON_PUNCT_COLOR}${keyMatch[2]}${RESET}`;
		body = keyMatch[3];
	}

	let trailing = "";
	const commaMatch = body.match(/^(.*?)(,)\s*$/s);
	if (commaMatch) {
		body = commaMatch[1];
		trailing = `${JSON_PUNCT_COLOR},${RESET}`;
	}

	let valued: string;
	if (/^".*"$/.test(body)) {
		valued = `${JSON_STR_COLOR}${body}${RESET}`;
	} else if (/^-?\d/.test(body)) {
		valued = `${JSON_NUM_COLOR}${body}${RESET}`;
	} else if (body === "true" || body === "false" || body === "null") {
		valued = `${JSON_KW_COLOR}${body}${RESET}`;
	} else if (/^[{}[\]]+$/.test(body)) {
		valued = `${JSON_PUNCT_COLOR}${body}${RESET}`;
	} else {
		valued = body;
	}

	return `${indent}${prefix}${valued}${trailing}`;
}

function stringArg(
	args: Record<string, unknown>,
	key: string,
): string | undefined {
	const value = args[key];
	return typeof value === "string" ? value : undefined;
}

function compactText(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

function extractDiff(result: string | undefined): string | null {
	if (!result) return null;
	const marker = "\nDiff:\n";
	const idx = result.indexOf(marker);
	if (idx >= 0) return result.slice(idx + marker.length);
	if (
		result.startsWith("diff --git") ||
		result.includes("\n@@ ") ||
		result.includes("\n--- ") ||
		result.includes("\n+++ ")
	) {
		return result;
	}
	return null;
}

function diffLineColor(line: string): string {
	if (line.startsWith("@@")) return DIFF_HUNK_COLOR;
	if (
		line.startsWith("diff --git") ||
		line.startsWith("index ") ||
		line.startsWith("---") ||
		line.startsWith("+++")
	) {
		return DIFF_META_COLOR;
	}
	if (line.startsWith("+")) return DIFF_ADD_COLOR;
	if (line.startsWith("-")) return DIFF_REMOVE_COLOR;
	return CODE_BLOCK_COLOR;
}

function parseJsonMaybe(value: string): unknown | null {
	const trimmed = value.trim();
	if (!trimmed || !/^[[{]/.test(trimmed)) return null;
	try {
		return JSON.parse(trimmed);
	} catch {
		return null;
	}
}

function isPermissionRejection(value: string): boolean {
	const text = value.toLowerCase();
	return [
		"permission denied",
		"not granted",
		"requires permission",
		"outside allowed",
		"denied",
		"blocked",
		"rejected",
	].some((pattern) => text.includes(pattern));
}

function normalizeEditArgs(
	args: Record<string, unknown>,
): Array<{ oldText: string; newText: string }> {
	const edits: Array<{ oldText: string; newText: string }> = [];
	if (typeof args.old_text === "string" || typeof args.oldText === "string") {
		edits.push({
			oldText: String(args.old_text ?? args.oldText ?? ""),
			newText: String(args.new_text ?? args.newText ?? ""),
		});
	}

	let rawEdits = args.edits;
	if (typeof rawEdits === "string") {
		try {
			rawEdits = JSON.parse(rawEdits);
		} catch {
			rawEdits = undefined;
		}
	}
	if (Array.isArray(rawEdits)) {
		for (const item of rawEdits) {
			if (!item || typeof item !== "object") continue;
			const edit = item as Record<string, unknown>;
			edits.push({
				oldText: String(edit.old_text ?? edit.oldText ?? ""),
				newText: String(edit.new_text ?? edit.newText ?? ""),
			});
		}
	}
	return edits;
}

function renderInlinePlain(text: string): string {
	let out = "";
	let i = 0;
	while (i < text.length) {
		if (text.startsWith("**", i)) {
			const end = text.indexOf("**", i + 2);
			if (end !== -1) {
				out += `${BOLD + text.slice(i + 2, end)}\x1b[22m`;
				i = end + 2;
				continue;
			}
		}
		if (text[i] === "`") {
			const end = text.indexOf("`", i + 1);
			if (end !== -1) {
				out += text.slice(i, end + 1);
				i = end + 1;
				continue;
			}
		}
		out += text[i];
		i++;
	}
	return out;
}

function escapeMarkdownTableCell(value: string): string {
	return value.replace(/\\/g, "\\\\").replace(/\|/g, "\\|");
}

// ── Options ────────────────────────────────────────────────────────────────────

interface TranscriptDisplayOptions {
	theme?: Partial<Theme>;
	thinkingMode?: ThinkingDisplayStyle;
	maxMessageLength?: number;
}

function hasStreamingChunk(chunks: AssistantChunk[]): boolean {
	return chunks.some((c) => !c.isComplete);
}

// ── Component ──────────────────────────────────────────────────────────────────

export class TranscriptDisplay implements Component, Scrollable {
	public readonly rendersViewport = true;
	private cachedWidth: number = 0;
	private cachedLines: string[] | null = null;
	private currentWidth: number = 80;
	private _scrollOffset: number = 0;
	private _totalHeight: number = 0;
	private _atBottom: boolean = true;
	private _pendingScrollBottom: boolean = false;

	private theme: Theme;
	private thinkingMode: ThinkingDisplayStyle;
	private toolsExpanded = false;
	private maxMessageLength: number;
	private turns: Turn[] = [];

	constructor(options: TranscriptDisplayOptions = {}) {
		this.theme = { ...DEFAULT_THEME, ...options.theme };
		this.thinkingMode = options.thinkingMode ?? "collapsed";
		this.maxMessageLength = options.maxMessageLength ?? 4000;
	}

	setThinkingMode(mode: ThinkingDisplayStyle): void {
		this.thinkingMode = mode;
		this.invalidate();
	}

	setToolsExpanded(expanded: boolean): void {
		if (this.toolsExpanded === expanded) return;
		this.toolsExpanded = expanded;
		this.invalidate();
	}

	toggleToolsExpanded(): boolean {
		this.setToolsExpanded(!this.toolsExpanded);
		return this.toolsExpanded;
	}

	areToolsExpanded(): boolean {
		return this.toolsExpanded;
	}

	invalidate(): void {
		this.cachedLines = null;
		this._totalHeight = 0;
	}

	setViewportHeight(height: number): void {
		if (this._viewportHeight === height) return;
		this._viewportHeight = height;
	}

	// ── Scroll interface ─────────────────────────────────────────────────────

	get scrollOffset(): number {
		return this._scrollOffset;
	}

	get totalHeight(): number {
		return this._totalHeight;
	}

	get isAtBottom(): boolean {
		return this._atBottom;
	}

	scroll(delta: number): void {
		const maxScroll = Math.max(
			0,
			this._totalHeight - (this._viewportHeight || 0),
		);
		this._scrollOffset = Math.min(
			maxScroll,
			Math.max(0, this._scrollOffset - delta),
		);
		this._atBottom = this._scrollOffset >= maxScroll;
	}

	scrollToBottom(): void {
		this._pendingScrollBottom = true;
		this._atBottom = true;
	}

	set scrollOffset(offset: number) {
		this._scrollOffset = offset;
		const maxScroll = Math.max(
			0,
			this._totalHeight - (this._viewportHeight || 0),
		);
		this._atBottom = offset >= maxScroll;
	}

	render(width: number): string[] {
		// Fast path: content unchanged → reuse cached body, only repaint viewport
		if (width === this.cachedWidth && this.cachedLines !== null) {
			this.resolvePendingScroll();
			return this.renderViewport(this.cachedLines, width);
		}

		this.currentWidth = width;
		this.cachedWidth = width;

		const renderedLines: string[] = [];
		const frameWidth = Math.max(1, width - 2);
		const contentWidth = Math.max(1, frameWidth - 2);

		const padToWidth = (line: string): string => {
			const clipped = clampLineToWidth(line, frameWidth);
			const w = visibleWidth(clipped);
			return clipped + " ".repeat(Math.max(0, frameWidth - w));
		};

		const emptyLine = " ".repeat(frameWidth);
		renderedLines.push(padToWidth(emptyLine));

		for (let ti = 0; ti < this.turns.length; ti++) {
			const turn = this.turns[ti];
			if (ti > 0) renderedLines.push(padToWidth(emptyLine));

			// User or system message
			if (turn.userMessage) {
				const content = turn.userMessage.content;
				if (content.startsWith("[System] ")) {
					const sysLines = this.renderMarkdownLines(
						content.slice(9),
						contentWidth - 2,
						false,
						this.theme.assistantColor,
						SYSTEM_PREFIX,
					);
					for (const line of sysLines) renderedLines.push(padToWidth(line));
				} else {
					const lines = this.wrapText(
						USER_PREFIX + BOLD + this.truncateText(content) + RESET,
						contentWidth,
					);
					for (const line of lines) renderedLines.push(padToWidth(line));
				}
			}

			// Assistant message — render chunks in seq order (chronological)
			if (turn.assistantMessage) {
				const msg = turn.assistantMessage;
				const chunks = msg.chunks;
				const streaming = !msg.isComplete || hasStreamingChunk(chunks);
				let lastThinkingSection = false;

				// Buffer consecutive content chunks so block-level markdown
				// (tables, code fences) that spans chunk boundaries renders whole.
				let contentBuffer = "";
				const flushContent = () => {
					if (!contentBuffer) return;
					const answer = stripThinkTags(contentBuffer);
					contentBuffer = "";
					if (lastThinkingSection) {
						renderedLines.push(
							padToWidth(
								`${this.theme.separatorColor}${DIM}  ─── response ───${RESET}`,
							),
						);
						lastThinkingSection = false;
					}
					if (answer) {
						const contentLines = this.renderMarkdownLines(
							answer,
							contentWidth - 2,
							streaming,
						);
						for (const line of contentLines)
							renderedLines.push(padToWidth(line));
					}
				};

				for (const chunk of chunks) {
					if (chunk.type === "content") {
						contentBuffer += chunk.contentText || "";
						continue;
					}
					flushContent();
					if (chunk.type === "thinking") {
						// Render thinking block
						const thinkLines = this.renderThinkingChunk(chunk, streaming);
						for (const line of thinkLines) renderedLines.push(padToWidth(line));
						lastThinkingSection = true;
					} else if (chunk.type === "tool" && chunk.tool) {
						lastThinkingSection = false;
						const toolLines = this.renderTool(chunk.tool, width);
						for (const line of toolLines) renderedLines.push(padToWidth(line));
						renderedLines.push(padToWidth(emptyLine));
					}
				}
				flushContent();

				// No streaming cursor — messages display as-is
			}
		}

		renderedLines.push(padToWidth(emptyLine));
		this._totalHeight = renderedLines.length;

		this.cachedLines = renderedLines;
		this.resolvePendingScroll();
		return this.renderViewport(renderedLines, width);
	}

	// ── Chunk rendering ──────────────────────────────────────────────────────

	private renderThinkingChunk(
		chunk: AssistantChunk,
		_streaming: boolean,
	): string[] {
		const text = chunk.contentText || "";
		if (text.trim().length === 0) return [];

		const lines: string[] = [];

		switch (this.thinkingMode) {
			case "collapsed": {
				const preview = text.trim().slice(0, 100);
				lines.push(
					`${THINKING_PREFIX}${DIM}${preview ? `thinking · ${preview}...` : "thinking"}${RESET}`,
				);
				break;
			}
			case "summary": {
				lines.push(
					`${THINKING_PREFIX}\x1b[2m${text.trim().slice(0, 150)}\x1b[0m`,
				);
				break;
			}
			case "expanded": {
				lines.push(`${THINKING_PREFIX}${BOLD}reasoning${RESET}`);
				const wrapped = this.wrapText(text, this.currentWidth - 4);
				for (let li = 0; li < wrapped.length; li++) {
					const rendered = `${DIM}  ${renderInline(wrapped[li], this.theme.thinkingColor + DIM)}`;
					lines.push(rendered);
				}
				break;
			}
		}

		return lines;
	}

	// ── Scroll helpers ───────────────────────────────────────────────────────

	private resolvePendingScroll(): void {
		if (!this._pendingScrollBottom) return;
		this._scrollOffset = Math.max(0, this._totalHeight - this._viewportHeight);
		this._atBottom = true;
		this._pendingScrollBottom = false;
	}

	private renderViewport(content: string[], width: number): string[] {
		const viewportHeight = this._viewportHeight || 0;
		if (viewportHeight <= 0) return content;
		if (content.length <= viewportHeight) return content;

		const maxScroll = content.length - viewportHeight;
		this._scrollOffset = Math.min(maxScroll, Math.max(0, this._scrollOffset));
		this._atBottom = this._scrollOffset >= maxScroll;

		const visible = content.slice(
			this._scrollOffset,
			this._scrollOffset + viewportHeight,
		);
		const thumbHeight = Math.max(
			1,
			Math.floor((viewportHeight * viewportHeight) / content.length),
		);
		const thumbStart =
			maxScroll > 0
				? Math.floor(
						(this._scrollOffset / maxScroll) * (viewportHeight - thumbHeight),
					)
				: 0;
		const thumbColor = "\x1b[38;5;111m";
		const barColor = "\x1b[38;5;236m";
		const reset = "\x1b[0m";
		for (let i = 0; i < visible.length; i++) {
			const line = visible[i];
			const w = visibleWidth(line);
			const pad = " ".repeat(Math.max(0, width - 2 - w));
			const isThumb = i >= thumbStart && i < thumbStart + thumbHeight;
			const bar = isThumb ? `${thumbColor}█${reset}` : `${barColor}│${reset}`;
			visible[i] = line + pad + bar;
		}
		return visible;
	}

	setTurns(turns: Turn[]): void {
		this.turns = turns;
		this.invalidate();
	}
	private _viewportHeight: number = 0;

	private renderMarkdownLines(
		text: string,
		maxLen: number,
		_streaming: boolean,
		baseColor = this.theme.assistantColor,
		firstLinePrefix = ASSISTANT_PREFIX,
	): string[] {
		const lines: string[] = [];
		const rawLines = text.split("\n");
		let inCodeBlock = false;
		let codeContent = "";
		let codeBlockLang: string | null = null;
		const bg = CODE_BLOCK_BG;
		const bgReset = RESET;

		for (let li = 0; li < rawLines.length; li++) {
			const rawLine = rawLines[li];

			if (rawLine.startsWith("```")) {
				if (inCodeBlock) {
					// Flush code block with syntax highlighting
					const lang = codeBlockLang || null;
					if (lang) {
						const highlighted = highlightAuto(codeContent);
						const langLabel = highlighted.language
							? ` ${highlighted.language} · ${codeContent.split("\n").length} lines`
							: "";
						lines.push(`${bg}${DIM}  ${rawLine}${langLabel}${bgReset}`);
						for (const cl of highlighted.value.split("\n")) {
							lines.push(`${bg}${DIM}  ${cl}${bgReset}`);
						}
					} else {
						const codeLines = codeContent.split("\n");
						for (const cl of codeLines) {
							lines.push(`${bg}${DIM}  ${cl}${bgReset}`);
						}
					}
					lines.push(`${bg}${DIM}  \x60\x60\x60${bgReset}`);
					codeContent = "";
					codeBlockLang = null;
					inCodeBlock = false;
				} else {
					inCodeBlock = true;
					codeBlockLang = extractLangFromFence(rawLine);
					lines.push(`${bg}${DIM}  ${rawLine}${bgReset}`);
				}
				continue;
			}

			if (inCodeBlock) {
				codeContent += (codeContent ? "\n" : "") + rawLine;
				continue;
			}

			if (this.isMemorySummaryRow(rawLine)) {
				const rows: string[][] = [];
				while (li < rawLines.length && this.isMemorySummaryRow(rawLines[li])) {
					const parsed = this.parseMemorySummaryRow(rawLines[li]);
					if (parsed) rows.push(parsed);
					li++;
				}
				li--;

				const renderedTable = this.renderTable(
					[
						"| ID | Time | Type | Title |",
						"| --- | --- | --- | --- |",
						...rows.map(
							(row) => `| ${row.map(escapeMarkdownTableCell).join(" | ")} |`,
						),
					],
					maxLen,
				);
				for (const line of renderedTable) lines.push(line);
				continue;
			}

			if (this.isTableStart(rawLines, li)) {
				const tableLines: string[] = [];
				tableLines.push(rawLines[li]);
				tableLines.push(rawLines[li + 1]);
				li += 2;
				while (li < rawLines.length && this.isTableRow(rawLines[li])) {
					tableLines.push(rawLines[li]);
					li++;
				}
				li--;

				const renderedTable = this.renderTable(tableLines, maxLen);
				for (let ti = 0; ti < renderedTable.length; ti++) {
					const line = renderedTable[ti];
					lines.push(line);
				}
				continue;
			}

			const jsonLines = formatJsonLine(rawLine);
			if (jsonLines) {
				for (let ji = 0; ji < jsonLines.length; ji++) {
					const line =
						ji === 0 ? firstLinePrefix + jsonLines[ji] : `  ${jsonLines[ji]}`;
					// No streaming cursor
					lines.push(line);
				}
				continue;
			}

			const rendered = renderMarkdownLine(rawLine, baseColor);
			const wrapped = this.wrapText(rendered, maxLen);
			if (wrapped.length === 0 || (wrapped.length === 1 && wrapped[0] === "")) {
				lines.push(`  ${baseColor} ${RESET}`);
				continue;
			}
			for (let wi = 0; wi < wrapped.length; wi++) {
				const seg = `${baseColor}${wrapped[wi]}${RESET}`;
				const line =
					wi === 0
						? firstLinePrefix + seg
						: `  ${baseColor}${wrapped[wi]}${RESET}`;
				// No streaming cursor
				lines.push(line);
			}
		}

		if (inCodeBlock && codeContent) {
			lines.push(`${bg}${DIM}  [code block open]${bgReset}`);
			for (const cl of codeContent.split("\n")) {
				lines.push(`${bg}${DIM}  ${cl}${bgReset}`);
			}
		}

		return lines;
	}

	private isMemorySummaryRow(line: string): boolean {
		return this.parseMemorySummaryRow(line) !== null;
	}

	private parseMemorySummaryRow(line: string): string[] | null {
		const match = line.match(
			/^([A-Za-z]?\d+)\s+((?:\d{1,2}:\d{2}[ap])|")\s+(\S+)\s+(.+)$/,
		);
		if (match) return [match[1], match[2], match[3], match[4]];
		const sessionMatch = line.match(/^(S\d+)\s+(.+)$/);
		if (sessionMatch) return [sessionMatch[1], "", "", sessionMatch[2]];
		return null;
	}

	private isTableStart(lines: string[], index: number): boolean {
		return (
			this.isTableRow(lines[index] || "") &&
			this.isTableSeparator(lines[index + 1] || "")
		);
	}

	private isTableRow(line: string): boolean {
		const trimmed = line.trim();
		return trimmed.includes("|") && trimmed.split("|").length >= 3;
	}

	private isTableSeparator(line: string): boolean {
		const cells = this.splitTableRow(line);
		if (cells.length < 2) return false;
		return cells.every((cell) => /^:?-+:?$/.test(cell.trim()));
	}

	private splitTableRow(line: string): string[] {
		const trimmed = line.trim();
		const withoutEdges = trimmed.replace(/^\|/, "").replace(/\|$/, "");
		const cells: string[] = [];
		let current = "";
		let escaped = false;
		for (const ch of withoutEdges) {
			if (escaped) {
				current += ch === "|" ? "|" : `\\${ch}`;
				escaped = false;
				continue;
			}
			if (ch === "\\") {
				escaped = true;
				continue;
			}
			if (ch === "|") {
				cells.push(current.trim());
				current = "";
				continue;
			}
			current += ch;
		}
		if (escaped) current += "\\";
		cells.push(current.trim());
		return cells;
	}

	private renderTable(rawLines: string[], maxLen: number): string[] {
		if (rawLines.length < 2) return rawLines;

		const header = this.splitTableRow(rawLines[0]);
		const rows = rawLines.slice(2).map((line) => this.splitTableRow(line));
		const columnCount = Math.max(
			header.length,
			...rows.map((row) => row.length),
		);
		if (columnCount < 2) return rawLines;

		const normalizedRows = [header, ...rows].map((row) => {
			const next = row.slice(0, columnCount);
			while (next.length < columnCount) next.push("");
			return next;
		});

		const minColumnWidth = 8;
		const gapWidth = 3 * (columnCount - 1) + 4;
		const usableWidth = Math.max(
			columnCount * minColumnWidth,
			maxLen - gapWidth,
		);
		const naturalWidths = Array.from({ length: columnCount }, (_, col) => {
			return Math.max(
				minColumnWidth,
				...normalizedRows.map((row) => visibleWidth(row[col] || "")),
			);
		});

		let widths = naturalWidths.slice();
		const total = widths.reduce((a, b) => a + b, 0);
		if (total > usableWidth) {
			const widest = widths.indexOf(Math.max(...widths));
			widths = widths.map((width, idx) => {
				if (idx === widest) return width;
				return Math.min(width, 24);
			});
			const reserved = widths.reduce(
				(sum, width, idx) => (idx === widest ? sum : sum + width),
				0,
			);
			widths[widest] = Math.max(minColumnWidth, usableWidth - reserved);
			while (widths.reduce((a, b) => a + b, 0) > usableWidth) {
				const shrinkIndex = widths.indexOf(Math.max(...widths));
				if (widths[shrinkIndex] <= minColumnWidth) break;
				widths[shrinkIndex]--;
			}
		}

		const borderColor = "\x1b[38;5;238m";
		const headerColor = "\x1b[38;5;188m";
		const rowColor = this.theme.assistantColor;
		const altRowColor = "\x1b[38;5;239m";
		const border = (
			left: string,
			fill: string,
			join: string,
			right: string,
		) => {
			return (
				borderColor +
				left +
				widths.map((width) => fill.repeat(width + 2)).join(join) +
				right +
				RESET
			);
		};

		const out: string[] = [];
		out.push(border("+", "-", "+", "+"));
		out.push(...this.renderTableRow(header, widths, headerColor, true));
		out.push(border("+", "-", "+", "+"));
		for (let ri = 0; ri < rows.length; ri++) {
			const row = rows[ri];
			const normalized = row.slice(0, columnCount);
			while (normalized.length < columnCount) normalized.push("");
			const rowColor_ = ri % 2 === 0 ? rowColor : altRowColor;
			out.push(...this.renderTableRow(normalized, widths, rowColor_, false));
		}
		out.push(border("+", "-", "+", "+"));
		return out;
	}

	private renderTableRow(
		cells: string[],
		widths: number[],
		color: string,
		header: boolean,
	): string[] {
		const wrappedCells = cells.map((cell, idx) =>
			this.wrapPlainCell(cell, widths[idx]),
		);
		const rowHeight = Math.max(1, ...wrappedCells.map((cell) => cell.length));
		const lines: string[] = [];
		const borderColor = "\x1b[38;5;238m";

		for (let row = 0; row < rowHeight; row++) {
			const renderedCells = wrappedCells.map((cellLines, idx) => {
				const raw = cellLines[row] || "";
				const styled = renderInline(raw, header ? color + BOLD : color);
				const padding = " ".repeat(
					Math.max(0, widths[idx] - visibleWidth(raw)),
				);
				return ` ${styled}${padding} `;
			});
			lines.push(
				`${borderColor}|${RESET}${renderedCells.join(
					`${borderColor}|${RESET}`,
				)}${borderColor}|${RESET}`,
			);
		}

		return lines;
	}

	private wrapPlainCell(text: string, width: number): string[] {
		if (!text) return [""];
		const words = text.split(/\s+/);
		const lines: string[] = [];
		let current = "";

		const pushHardWrapped = (word: string) => {
			let remaining = word;
			while (visibleWidth(remaining) > width) {
				lines.push(remaining.slice(0, width));
				remaining = remaining.slice(width);
			}
			current = remaining;
		};

		for (const word of words) {
			if (!current) {
				if (visibleWidth(word) > width) pushHardWrapped(word);
				else current = word;
			} else if (visibleWidth(current) + 1 + visibleWidth(word) <= width) {
				current += ` ${word}`;
			} else {
				lines.push(current);
				if (visibleWidth(word) > width) pushHardWrapped(word);
				else current = word;
			}
		}
		if (current) lines.push(current);
		return lines.length > 0 ? lines : [""];
	}

	private renderTool(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const isError = tool.isError ? ERROR_PREFIX : TOOL_PREFIX;
		const status = tool.isError
			? "\x1b[38;5;203merror\x1b[0m"
			: tool.isComplete
				? "\x1b[38;5;118m✓ done\x1b[0m"
				: tool.partialResult
					? "\x1b[38;5;220m▸ streaming\x1b[0m"
					: "\x1b[38;5;141m▸ running\x1b[0m";
		const summary = this.toolSummary(tool);
		const hint = this.toolsExpanded
			? `${DIM}ctrl+o collapse${RESET}`
			: `${DIM}ctrl+o expand${RESET}`;
		const base =
			`${isError}${BOLD}${tool.tool_name}${RESET} ` +
			`${DIM}(${RESET}${status}${DIM})${RESET}`;

		lines.push(
			[base, summary ? `${DIM}${summary}${RESET}` : "", hint]
				.filter(Boolean)
				.join(" "),
		);

		// Details only when the global tool-expand toggle is on (ctrl+o).
		if (!this.toolsExpanded) return lines;

		for (const detailLine of this.toolDetailLines(tool, width - 2)) {
			const wrapped = this.wrapText(detailLine, Math.max(20, width - 4));
			for (const line of wrapped) {
				lines.push(`${TOOL_DETAIL_PREFIX}${line}`);
			}
		}

		return lines;
	}

	private toolSummary(tool: ToolExecution): string {
		const args = tool.args || {};
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		if (tool.tool_name === "write_file") {
			const content = stringArg(args, "content") || "";
			const lineCount = content ? content.split("\n").length : 0;
			return [
				path,
				content ? `${content.length} bytes` : "",
				lineCount ? `${lineCount} lines` : "",
			]
				.filter(Boolean)
				.join(" · ");
		}
		if (tool.tool_name === "edit_file") {
			const editCount = Array.isArray(args.edits) ? args.edits.length : 1;
			return [path, `${editCount} edit${editCount === 1 ? "" : "s"}`]
				.filter(Boolean)
				.join(" · ");
		}
		if (tool.tool_name === "bash") {
			return compactText(stringArg(args, "command") || "");
		}
		if (tool.tool_name === "read_file") {
			return path || "";
		}
		if (tool.tool_name === "rg_search") {
			return compactText(stringArg(args, "pattern") || "");
		}
		if (tool.tool_name.startsWith("mcp__")) {
			return [
				tool.tool_name.replace(/^mcp__/, "").replace(/__/g, "."),
				tool.result ? compactText(tool.result).slice(0, 80) : "",
			]
				.filter(Boolean)
				.join(" · ");
		}
		if (path) return path;
		const result = tool.result ?? tool.partialResult;
		return result ? compactText(result).slice(0, 80) : "";
	}

	private toolDetailLines(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const result = tool.result ?? tool.partialResult;

		if (tool.isError && result && isPermissionRejection(result)) {
			lines.push(...this.renderPermissionBlock(result, width));
			return lines;
		}

		if (tool.tool_name === "write_file") {
			lines.push(...this.renderWriteDetails(tool, width));
		} else if (tool.tool_name === "edit_file") {
			lines.push(...this.renderEditDetails(tool, width));
		} else if (tool.tool_name === "file_diff") {
			lines.push(...this.renderFileDiffDetails(tool, width));
		} else if (tool.tool_name === "bash") {
			lines.push(...this.renderBashDetails(tool, width));
		} else if (tool.tool_name.startsWith("mcp__")) {
			lines.push(...this.renderMcpDetails(tool, width));
		} else {
			const argText = JSON.stringify(args, null, 2);
			if (argText && argText !== "{}") {
				lines.push(`${BOLD}args${RESET}`);
				lines.push(...argText.split("\n"));
			}
		}

		if (
			result &&
			!["write_file", "edit_file", "file_diff", "bash"].includes(
				tool.tool_name,
			) &&
			!tool.tool_name.startsWith("mcp__")
		) {
			lines.push(`${BOLD}${tool.isError ? "error" : "result"}${RESET}`);
			lines.push(...this.previewBlock(result, width));
		} else if (!result) {
			lines.push(`${DIM}waiting for result...${RESET}`);
		}

		return lines;
	}

	private renderWriteDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const path =
			stringArg(tool.args || {}, "path") ||
			stringArg(tool.args || {}, "file_path");

		// Try to parse accumulated partialResult (streaming JSON args) to
		// extract the actual content being written.
		const parsedArgs = this.parsePartialArgs(tool.partialResult);
		const content =
			parsedArgs?.content || stringArg(tool.args || {}, "content");
		const resolvedPath = parsedArgs?.path || path;

		if (resolvedPath) lines.push(`${BOLD}path${RESET} ${resolvedPath}`);
		if (content !== undefined) {
			lines.push(
				`${BOLD}content${RESET} ${DIM}${content.length} bytes · ${content.split("\n").length} lines${RESET}`,
			);
			lines.push(...this.previewBlock(content, width));
		}
		const result = tool.result ?? tool.partialResult;
		const diff = extractDiff(result);
		if (diff) {
			lines.push(`${BOLD}diff${RESET}`);
			lines.push(...this.renderDiffBlock(diff, width));
		} else if (tool.result && !content) {
			// Only show the *finished* tool output here. While streaming,
			// `partialResult` is the raw (escaped) JSON args being accumulated —
			// rendering it verbatim leaks literal "\n" into the display.
			lines.push(`${BOLD}${tool.isError ? "error" : "result"}${RESET}`);
			lines.push(...this.previewBlock(tool.result, width));
		} else if (!content && !tool.result) {
			lines.push(`${DIM}writing…${RESET}`);
		}
		return lines;
	}

	/** Parse accumulated partialResult JSON to extract tool args. */
	private parsePartialArgs(
		raw: string | undefined,
	): Record<string, string> | null {
		if (!raw) return null;
		// Try the accumulated string as-is first.
		const parsed = parseJsonMaybe(raw);
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			return parsed as Record<string, string>;
		}
		// Try to find a complete JSON object inside the accumulated string.
		// The model may stream partial JSON fragments before completing the object.
		let depth = 0;
		let start = -1;
		let inString = false;
		let escape = false;
		for (let i = 0; i < raw.length; i++) {
			const ch = raw[i];
			if (escape) {
				escape = false;
				continue;
			}
			if (ch === "\\") {
				escape = true;
				continue;
			}
			if (ch === "\"") {
				inString = !inString;
				continue;
			}
			if (inString) continue;
			if (ch === "{" && depth === 0) {
				start = i;
			}
			if (ch === "}") {
				depth--;
				if (depth === 0 && start >= 0) {
					const candidate = raw.slice(start, i + 1);
					const parsedInner = parseJsonMaybe(candidate);
					if (
						parsedInner &&
						typeof parsedInner === "object" &&
						!Array.isArray(parsedInner)
					) {
						return parsedInner as Record<string, string>;
					}
				}
			}
			if (ch === "{") depth++;
		}
		return null;
	}

	private renderEditDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];

		// Parse accumulated partialResult for streaming args.
		const parsedArgs = this.parsePartialArgs(tool.partialResult);
		const path =
			parsedArgs?.path ||
			stringArg(tool.args || {}, "path") ||
			stringArg(tool.args || {}, "file_path");

		if (path) lines.push(`${BOLD}path${RESET} ${path}`);

		// Build edits from parsed streaming args, falling back to tool.args.
		const editsFromParsed = parsedArgs
			? this.normalizeEditArgsFromParsed(parsedArgs)
			: [];
		const editsFromArgs = editsFromParsed.length
			? editsFromParsed
			: normalizeEditArgs(tool.args || {});

		for (let i = 0; i < editsFromArgs.length; i++) {
			lines.push(`${BOLD}edit ${i + 1}${RESET}`);
			lines.push(`${DIFF_REMOVE_COLOR}- old${RESET}`);
			lines.push(...this.previewBlock(editsFromArgs[i].oldText, width));
			lines.push(`${DIFF_ADD_COLOR}+ new${RESET}`);
			lines.push(...this.previewBlock(editsFromArgs[i].newText, width));
		}

		const result = tool.result ?? tool.partialResult;
		const diff = extractDiff(result);
		if (diff) {
			lines.push(`${BOLD}diff${RESET}`);
			lines.push(...this.renderDiffBlock(diff, width));
		} else if (result && !editsFromArgs.length) {
			lines.push(`${BOLD}${tool.isError ? "error" : "result"}${RESET}`);
			lines.push(...this.previewBlock(result, width));
		}
		return lines;
	}

	/** Normalize edit args from parsed partialResult JSON. */
	private normalizeEditArgsFromParsed(
		args: Record<string, string>,
	): Array<{ oldText: string; newText: string }> {
		const edits: Array<{ oldText: string; newText: string }> = [];
		if (typeof args.old_text === "string" || typeof args.oldText === "string") {
			edits.push({
				oldText: String(args.old_text ?? args.oldText ?? ""),
				newText: String(args.new_text ?? args.newText ?? ""),
			});
		}
		return edits;
	}

	private renderFileDiffDetails(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		if (path) lines.push(`${BOLD}path${RESET} ${path}`);
		if (args.staged) lines.push(`${BOLD}mode${RESET} staged`);
		const result = tool.result ?? tool.partialResult;
		if (result) lines.push(...this.renderDiffBlock(result, width));
		return lines;
	}

	private renderBashDetails(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const command = stringArg(args, "command") || "";
		const timeout = args.timeout ? `${Number(args.timeout)}ms` : "30000ms";
		if (command) {
			lines.push(`${BOLD}command${RESET} ${DIM}${timeout}${RESET}`);
			lines.push(...this.previewBlock(command, width));
		}
		const result = tool.result ?? tool.partialResult;
		if (result) {
			const label = tool.result
				? tool.isError
					? "error output"
					: "output"
				: "streaming output";
			lines.push(`${BOLD}${label}${RESET}`);
			lines.push(...this.renderTerminalBlock(result, width));
		} else {
			lines.push(`${DIM}waiting for command output...${RESET}`);
		}
		return lines;
	}

	private renderMcpDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const args = tool.args || {};
		const serverParts = tool.tool_name.replace(/^mcp__/, "").split("__");
		if (serverParts.length >= 2) {
			lines.push(`${BOLD}server${RESET} ${serverParts[0]}`);
			lines.push(`${BOLD}tool${RESET} ${serverParts.slice(1).join("__")}`);
		}
		const argText = JSON.stringify(args, null, 2);
		if (argText && argText !== "{}") {
			lines.push(`${BOLD}args${RESET}`);
			lines.push(...this.previewBlock(argText, width));
		}
		const result = tool.result ?? tool.partialResult;
		if (result) {
			lines.push(`${BOLD}${tool.isError ? "mcp error" : "mcp result"}${RESET}`);
			lines.push(...this.renderMcpResultBlocks(result, width));
		}
		return lines;
	}

	private renderPermissionBlock(result: string, width: number): string[] {
		const lines = [
			`${WARNING_COLOR}${BOLD}permission / rejection${RESET}`,
			...this.previewBlock(result, width),
		];
		return lines;
	}

	private renderMcpResultBlocks(result: string, width: number): string[] {
		const parsed = parseJsonMaybe(result);
		if (!parsed) return this.previewBlock(result, width);
		const content =
			parsed && typeof parsed === "object"
				? (parsed as Record<string, unknown>).content
				: undefined;
		if (Array.isArray(content)) {
			const lines: string[] = [];
			content.forEach((item, index) => {
				const block = item as Record<string, unknown>;
				lines.push(
					`${DIM}block ${index + 1}: ${String(
						block.type || "content",
					)}${RESET}`,
				);
				if (typeof block.text === "string") {
					lines.push(...this.previewBlock(block.text, width));
				} else {
					lines.push(
						...this.previewBlock(JSON.stringify(block, null, 2), width),
					);
				}
			});
			return lines;
		}
		return this.previewBlock(JSON.stringify(parsed, null, 2), width);
	}

	private renderDiffBlock(diff: string, width: number): string[] {
		if (!diff.trim()) return [`${DIM}(no diff)${RESET}`];
		const rawLines = this.truncateText(diff).split("\n");
		const lines: string[] = [];
		const bg = CODE_BLOCK_BG;
		const bgReset = RESET;
		for (const raw of rawLines) {
			const color = diffLineColor(raw);
			const content = raw.length ? raw.replace(/\t/g, "    ") : " ";
			if (visibleWidth(content) <= width) {
				lines.push(`${bg}${color}${content}${bgReset}`);
			} else {
				for (const wrapped of this.wrapText(content, width)) {
					lines.push(`${bg}${color}${wrapped}${bgReset}`);
				}
			}
		}
		return lines;
	}

	private renderTerminalBlock(text: string, width: number): string[] {
		if (!text) return [`${DIM}(no output)${RESET}`];
		const rawLines = this.truncateText(text).split("\n");
		const lines: string[] = [];
		const bg = CODE_BLOCK_BG;
		const bgReset = RESET;
		for (const raw of rawLines) {
			const content = raw.length ? raw.replace(/\t/g, "    ") : " ";
			const color = raw.startsWith("Error:")
				? DIFF_REMOVE_COLOR
				: TERMINAL_COLOR;
			if (visibleWidth(content) <= width) {
				lines.push(`${bg}${color}${content}${bgReset}`);
			} else {
				for (const wrapped of this.wrapText(content, width)) {
					lines.push(`${bg}${color}${wrapped}${bgReset}`);
				}
			}
		}
		return lines;
	}

	private previewBlock(text: string, width: number): string[] {
		if (!text) return [`${DIM}(empty)${RESET}`];
		const rawLines = this.truncateText(text).split("\n");
		const lines: string[] = [];
		const bg = CODE_BLOCK_BG;
		const bgReset = RESET;
		for (const raw of rawLines) {
			const formatted = raw.length ? raw.replace(/\t/g, "    ") : " ";
			if (visibleWidth(formatted) <= width) {
				lines.push(`${bg}${formatted}${bgReset}`);
			} else {
				for (const wrapped of this.wrapText(formatted, width)) {
					lines.push(`${bg}${wrapped}${bgReset}`);
				}
			}
		}
		return lines;
	}

	private wrapText(text: string, maxLineLength: number): string[] {
		const lines: string[] = [];
		const rawLines = text.split("\n");
		for (const rawLine of rawLines) {
			if (visibleWidth(rawLine) <= maxLineLength) {
				lines.push(rawLine);
			} else {
				const words = rawLine.split(/\s+/);
				let current = "";
				for (const word of words) {
					if (current.length === 0) {
						current = word;
					} else if (
						visibleWidth(current) + 1 + visibleWidth(word) <=
						maxLineLength
					) {
						current += ` ${word}`;
					} else {
						lines.push(current);
						current = word;
					}
				}
				if (current) lines.push(current);
			}
		}
		return lines;
	}

	private truncateText(text: string): string {
		if (text.length <= this.maxMessageLength) return text;
		return `${text.slice(0, this.maxMessageLength)}\n\n\x1b[2m[truncated]\x1b[0m`;
	}
}
