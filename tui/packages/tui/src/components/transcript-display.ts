// ── Transcript display component ────────────────────────────────────────────────
// Renders the full conversation history with streaming support and markdown.
// Chunks are interleaved in chronological order: thinking → content → tool → ...

import {
	highlight,
	highlightAuto,
} from "@logician/agent-core/tools/shared/syntax-highlighter.ts";
import { stripTextToolCalls } from "@logician/agent-core/tools/shared/text-to-tool-calls.ts";
import { stripAcceptanceReport } from "@logician/agent-core/core/guards/acceptance-contract.ts";
import { DEFAULT_TRUNCATION } from "@logician/agent-core";
import { theme } from "../layers/theme/theme.ts";
import type {
	AssistantChunk,
	ChildChunk,
	ChildToolCall,
	ThinkingDisplayStyle,
	ToolExecution,
	Turn,
} from "@logician/coding-agent/transcript";
import {
	type Component,
	clampLineToWidth,
	type Scrollable,
	visibleWidth,
	RESET,
	BOLD,
	DIM,
} from "../layers/core/tui-core.ts";

const UNDERLINE = "\x1b[4m";

// Heading palette — distinct color + weight per level.
const getHeadingStyles = (): Array<{ color: string; deco: string }> => [
	{ color: theme.fgRaw("mdHeading") + RESET, deco: BOLD + UNDERLINE },
	{ color: theme.fgRaw("accent") + RESET, deco: BOLD },
	{ color: theme.fgRaw("mdHeading") + RESET, deco: BOLD },
	{ color: theme.fgRaw("warning") + RESET, deco: "" },
	{ color: theme.fgRaw("muted") + RESET, deco: "" },
	{ color: theme.fgRaw("dim") + DIM + RESET, deco: DIM },
];

// ── Code fence language extraction ────────────────────────────────────────────

function extractLangFromFence(line: string): string | null {
	const m = /^```(\w+)/.exec(line.trim());
	return m ? m[1].toLowerCase() : null;
}

// ── Embedded reasoning stripping ──────────────────────────────────────────────
function stripThinkTags(text: string): string {
	if (!text?.includes("<think")) return text;
	return text
		.replace(/<think(?:ing)?>\s*[\s\S]*?<\/think(?:ing)?>\s*/gi, "")
		.replace(/<think(?:ing)?>\s*[\s\S]*$/i, "")
		.trimStart();
}

function unwrapThinkingChannel(text: string): string {
	return text.replace(/<\/?think(?:ing)?>/gi, "").trim();
}

function stripThinkingToolMarkup(text: string): string {
	if (!/<(?:tool\\?_call|function\s*=)/i.test(text)) return text;
	return stripTextToolCalls(text)
		.replace(
			/\n*\**\s*<(?:tool\\?_call|function\s*=\s*[a-zA-Z_][\w.-]*)[^>]*>[\s\S]*$/i,
			"",
		)
		.trimEnd();
}

function stripAcceptanceForDisplay(text: string): string {
	const marker = text.indexOf("```acceptance-report");
	if (marker < 0) return text;
	const stripped = stripAcceptanceReport(text);
	// While the report is still streaming there is no closing fence to strip.
	// Hide the internal report from its opening marker onward.
	return stripped === text ? text.slice(0, marker).trimEnd() : stripped.trimEnd();
}

function stripInternalHookGuidance(text: string | undefined): string | undefined {
	if (!text?.includes("-hook>")) return text;
	const visible = text
		.replace(
			/\n*<(?:post-tool-use|pre-tool-use|stop)-hook>[\s\S]*?<\/(?:post-tool-use|pre-tool-use|stop)-hook>\n*/gi,
			"\n",
		)
		.replace(/\n{3,}/g, "\n\n")
		.trimEnd();
	return visible || undefined;
}

// ── Inline markdown renderer ──────────────────────────────────────────────────

interface ParsedPostEditDiagnostic {
	line: number;
	column: number;
	label?: string;
	message: string;
}

interface PostEditDiagnosticBlock {
	file: string;
	diagnostics: ParsedPostEditDiagnostic[];
}

function extractPostEditDiagnostics(text: string | undefined): {
	text: string | undefined;
	blocks: PostEditDiagnosticBlock[];
} {
	if (!text?.includes("<post_edit_diagnostics")) {
		return { text, blocks: [] };
	}
	const blocks: PostEditDiagnosticBlock[] = [];
	const cleaned = text.replace(
		/\n*<post_edit_diagnostics\s+file="([^"]*)">([\s\S]*?)<\/post_edit_diagnostics>\n*/gi,
		(_match, fileValue: string, body: string) => {
			const diagnostics = body
				.split("\n")
				.flatMap((line): ParsedPostEditDiagnostic[] => {
					const parsed =
						/^-\s+.*?:(\d+):(\d+)(?:\s+(.+?))?:\s+(.+)$/.exec(
							line.trim(),
						);
					if (!parsed) return [];
					return [{
						line: Number(parsed[1]),
						column: Number(parsed[2]),
						label: parsed[3]?.trim(),
						message: parsed[4].trim(),
					}];
				});
			blocks.push({ file: fileValue, diagnostics });
			return "\n";
		},
	);
	const visible = cleaned.replace(/\n{3,}/g, "\n\n").trimEnd();
	return { text: visible || undefined, blocks };
}

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
				out +=
					theme.fgRaw("warning") +
					BOLD +
					text.slice(i, i + tag) +
					RESET +
					baseColor;
				i += tag;
				continue;
			}
		}
		if (text.startsWith("```", i)) {
			const end = text.indexOf("```", i + 3);
			if (end !== -1) {
				out +=
					theme.fgRaw("mdCodeBlock") +
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
					theme.fgRaw("mdCode") +
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
		const style = getHeadingStyles()[level - 1];
		const marker = level <= 2 ? "▌ " : "";
		return `${style.color}${marker}${style.deco}${style.color}${renderInlinePlain(heading[2])}${RESET}`;
	}

	// Horizontal rule
	if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
		return `${DIM}${theme.fgRaw("dim")}${"─".repeat(40)}${RESET}`;
	}

	// List items
	const listMatch = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
	if (listMatch) {
		const indent = listMatch[1];
		const lmarker = listMatch[2];
		const rest = listMatch[3];
		if (/^\d/.test(lmarker)) {
			return `${indent}${theme.fgRaw("mdListBullet")}${BOLD}${lmarker}${RESET} ${renderInline(rest, baseColor)}`;
		}
		const glyph = "•";
		return `${indent}${theme.fgRaw("mdListBullet")}${glyph}${RESET} ${renderInline(rest, baseColor)}`;
	}

	// Blockquote
	const quote = line.match(/^(\s*)>\s?(.*)$/);
	if (quote) {
		return `${quote[1]}${DIM}${theme.fgRaw("mdQuote")}▏ ${renderInlinePlain(quote[2])}${RESET}`;
	}

	return renderInline(line, baseColor);
}

// JSON syntax color helpers
const getJsonKeyCol = (): string => theme.fgRaw("jsonKey");
const getJsonStringCol = (): string => theme.fgRaw("jsonString");
const getJsonNumCol = (): string => theme.fgRaw("jsonNumber");
const getJsonKwCol = (): string => theme.fgRaw("jsonKeyword");
const getJsonPunctCol = (): string => theme.fgRaw("jsonPunctuation");

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
		prefix = `${getJsonKeyCol()}${keyMatch[1]}${RESET}${getJsonPunctCol()}${keyMatch[2]}${RESET}`;
		body = keyMatch[3];
	}

	let trailing = "";
	const commaMatch = body.match(/^(.*?)(,)\s*$/s);
	if (commaMatch) {
		body = commaMatch[1];
		trailing = `${getJsonPunctCol()},${RESET}`;
	}

	let valued: string;
	if (/^".*"$/.test(body)) {
		valued = `${getJsonStringCol()}${body}${RESET}`;
	} else if (/^-?\d/.test(body)) {
		valued = `${getJsonNumCol()}${body}${RESET}`;
	} else if (body === "true" || body === "false" || body === "null") {
		valued = `${getJsonKwCol()}${body}${RESET}`;
	} else if (/^[{}[\]]+$/.test(body)) {
		valued = `${getJsonPunctCol()}${body}${RESET}`;
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

/** Read an early string field from streamed JSON before the full args parse. */
function streamedStringArg(json: string | undefined, key: string): string | undefined {
	if (!json) return undefined;
	const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const match = new RegExp(`"${escapedKey}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`).exec(json);
	if (!match) return undefined;
	try {
		return JSON.parse(`"${match[1]}"`) as string;
	} catch {
		return match[1];
	}
}

function compactText(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

function formatDurationMs(ms: number): string {
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function diffLineColor(line: string): string {
	if (line.startsWith("@@")) return theme.fgRaw("diffHunk");
	if (
		line.startsWith("diff --git") ||
		line.startsWith("index ") ||
		line.startsWith("---") ||
		line.startsWith("+++")
	) {
		return theme.fgRaw("diffMeta");
	}
	if (line.startsWith("+")) return theme.fgRaw("diffAdded");
	if (line.startsWith("-")) return theme.fgRaw("diffRemoved");
	return theme.fgRaw("mdCodeBlock");
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
	thinkingMode?: ThinkingDisplayStyle;
	maxMessageLength?: number;
	/** Max turns to keep in memory. Older turns are dropped. */
	maxTurns?: number;
	/** Max rendered lines before cutting off older content. */
	maxRenderedLines?: number;
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

	private thinkingMode: ThinkingDisplayStyle;
	private toolsExpanded = false;
	private maxMessageLength: number;
	private maxTurns: number;
	private maxRenderedLines: number;
	private turns: Turn[] = [];
	private spinnerTick = 0;
	private spinnerTimer: ReturnType<typeof setInterval> | null = null;
	private onAnimationTick: (() => void) | null = null;
	/**
	 * Wall-clock timing for spawn_agents per-task rows, keyed by the batch
	 * tool's call id then task index. The batch tool only reports per-task
	 * status via a `▶/✓/×` marker stream with no timestamps, so the first
	 * time a task is observed running/finished here we stamp it ourselves —
	 * this is a rendering-side approximation, not the tool's real timing.
	 */
	private batchTaskTiming = new Map<string, Map<number, { startedAt: number; endedAt?: number }>>();

	constructor(options: TranscriptDisplayOptions = {}) {
		this.thinkingMode = options.thinkingMode ?? "collapsed";
		this.maxMessageLength =
			options.maxMessageLength ?? DEFAULT_TRUNCATION.transcriptMessageMaxChars;
		this.maxTurns = options.maxTurns ?? 200;
		this.maxRenderedLines = options.maxRenderedLines ?? 2000;
	}

	private static readonly SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

	private spinnerFrame(): string {
		return TranscriptDisplay.SPINNER_FRAMES[
			this.spinnerTick % TranscriptDisplay.SPINNER_FRAMES.length
		];
	}

	/** Called once, wiring a redraw request for each spinner animation frame. */
	setOnAnimationTick(cb: () => void): void {
		this.onAnimationTick = cb;
	}

	/** Start ticking the running-tool spinner. Idempotent while already running. */
	startAnimation(): void {
		if (this.spinnerTimer) return;
		this.spinnerTimer = setInterval(() => {
			this.spinnerTick = (this.spinnerTick + 1) % TranscriptDisplay.SPINNER_FRAMES.length;
			this.invalidate();
			this.onAnimationTick?.();
		}, 150);
	}

	stopAnimation(): void {
		if (this.spinnerTimer) {
			clearInterval(this.spinnerTimer);
			this.spinnerTimer = null;
		}
	}

	setThinkingMode(mode: ThinkingDisplayStyle): void {
		if (this.thinkingMode === mode) return;
		const keepBottomAnchored = this._atBottom;
		this.thinkingMode = mode;
		this.invalidate();
		if (keepBottomAnchored) this._pendingScrollBottom = true;
	}

	setToolsExpanded(expanded: boolean): void {
		if (this.toolsExpanded === expanded) return;
		const keepBottomAnchored = this._atBottom;
		this.toolsExpanded = expanded;
		this.invalidate();
		if (keepBottomAnchored) this._pendingScrollBottom = true;
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
		// Apply offset immediately so isAtBottom reflects the new position
		// right away — the TUI checks isAtBottom in the same frame loop.
		const maxScroll = Math.max(
			0,
			this._totalHeight - (this._viewportHeight || 0),
		);
		this._scrollOffset = maxScroll;
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
		const turnStartLines: number[] = [];
		const frameWidth = Math.max(1, width - 2);
		const contentWidth = Math.max(1, frameWidth - 2);

		const padToWidth = (line: string): string => {
			const clipped = clampLineToWidth(line, frameWidth);
			const w = visibleWidth(clipped);
			return clipped + " ".repeat(Math.max(0, frameWidth - w));
		};

		// Render oldest-to-newest so the newest turns always end up at the
		// bottom of the buffer — the viewport slices from scrollOffset and
		// shows the bottom, meaning new messages are always visible even when
		// maxRenderedLines truncates older content.
		const emptyLine = " ".repeat(frameWidth);
		renderedLines.push(padToWidth(emptyLine));

		for (let ti = 0; ti < this.turns.length; ti++) {
			turnStartLines.push(renderedLines.length);
			const turn = this.turns[ti];
			if (ti > 0) renderedLines.push(padToWidth(emptyLine));

			// User or system message
			if (turn.userMessage) {
				const content = turn.userMessage.content;
				if (content.startsWith("[System] ")) {
					renderedLines.push(
						padToWidth(`${theme.fgRaw("systemText")}◇ SYSTEM${RESET}`),
					);
					const sysLines = this.renderMarkdownLines(
						content.slice(9),
						contentWidth - 2,
						false,
						theme.fgRaw("systemText") + RESET,
						"",
					);
					for (const line of sysLines)
						renderedLines.push(padToWidth(`${theme.fgRaw("separator")}│${RESET} ${line}`));
				} else {
					renderedLines.push(
						padToWidth(`${theme.fgRaw("userText")}╭─ ${BOLD}YOU${RESET}`),
					);
					const lines = this.wrapText(
						theme.fgRaw("userText") + this.truncateText(content) + RESET,
						Math.max(1, contentWidth - 2),
					);
					for (const line of lines)
						renderedLines.push(
							padToWidth(`${theme.fgRaw("userText")}│${RESET} ${line}`),
						);
					renderedLines.push(
						padToWidth(`${theme.fgRaw("userText")}╰─${RESET}`),
					);
				}
			}

			// Assistant message — render chunks in seq order (chronological)
			if (turn.assistantMessage) {
				const msg = turn.assistantMessage;
				const chunks = msg.chunks;
				const streaming = !msg.isComplete || hasStreamingChunk(chunks);
				let lastThinkingSection = false;
				renderedLines.push(
					padToWidth(`${theme.fgRaw("assistantText")}◆ ${BOLD}LOGICIAN${RESET}`),
				);

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
								`${theme.fgRaw("separator")}${DIM}  ─── response ───${RESET}`,
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
							renderedLines.push(padToWidth(`  ${line}`));
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
						for (const line of thinkLines)
							renderedLines.push(padToWidth(`  ${line}`));
						lastThinkingSection = true;
					} else if (chunk.type === "tool" && chunk.tool) {
						lastThinkingSection = false;
						const toolLines = this.renderTool(chunk.tool, width);
						for (const line of toolLines)
							renderedLines.push(padToWidth(`  ${line}`));
					} else if (chunk.type === "notice" && chunk.notice) {
						const n = chunk.notice;
						if (n.label === "Skills" && n.level === "info") {
							renderedLines.push(
								padToWidth(
									`${theme.fg("active", "✦")} ${BOLD}${theme.fg("toolTitle", "Skills")}${RESET}  ${theme.fg("systemText", n.text)}${RESET}`,
								),
							);
							continue;
						}
						const icon = n.level === "error"
							? "✗"
							: n.level === "warn"
								? "⚠"
									: n.level === "success"
										? "✓"
											: "●";
						const color = n.level === "error"
							? theme.fgRaw("error")
							: n.level === "warn"
								? theme.fgRaw("warning")
									: theme.fgRaw("systemText");
						renderedLines.push(
							padToWidth(`${color}${icon} ${n.label}: ${n.text}${RESET}`),
						);
					}
				}
				flushContent();

				// No streaming cursor — messages display as-is
			}
		}


		// Bound the render buffer from the *front*. The previous early-break
		// implementation retained old turns and discarded the newest ones while
		// labelling them as "older", which became especially visible after Ctrl+O
		// expanded tool output. Prefer a complete recent-turn boundary; if the
		// newest turn alone exceeds the budget, retain its tail.
		let visibleBuffer = renderedLines;
		const newestAssistant = this.turns.at(-1)?.assistantMessage;
		const newestTurnIsStreaming =
			newestAssistant !== null &&
			newestAssistant !== undefined &&
			(!newestAssistant.isComplete ||
				hasStreamingChunk(newestAssistant.chunks));
		if (
			!newestTurnIsStreaming &&
			renderedLines.length > this.maxRenderedLines
		) {
			const desiredStart = renderedLines.length - (this.maxRenderedLines - 1);
			const firstCompleteTurn = turnStartLines.findIndex(
				(line) => line >= desiredStart,
			);
			const sliceStart =
				firstCompleteTurn >= 0
					? turnStartLines[firstCompleteTurn]
					: desiredStart;
			const olderCount =
				firstCompleteTurn >= 0 ? firstCompleteTurn : Math.max(0, this.turns.length - 1);
			const omittedLabel =
				olderCount > 0
					? `${olderCount} older turn(s) not shown`
					: "earlier lines not shown";
			visibleBuffer = [
				padToWidth(
					`${theme.fgRaw("dim")}… ${omittedLabel}${RESET}`,
				),
				...renderedLines.slice(sliceStart),
			];
		}

		this._totalHeight = visibleBuffer.length;

		this.cachedLines = visibleBuffer;
		this.resolvePendingScroll();
		return this.renderViewport(visibleBuffer, width);
	}

	// ── Chunk rendering ──────────────────────────────────────────────────────

	private renderThinkingChunk(
		chunk: AssistantChunk,
		_streaming: boolean,
	): string[] {
		const text = stripThinkingToolMarkup(
			unwrapThinkingChannel(chunk.contentText || ""),
		);
		if (!text) return [];

		const lines: string[] = [];

		switch (this.thinkingMode) {
			case "collapsed": {
				const preview = text.trim().slice(0, 100);
				lines.push(
					`${theme.fgRaw("thinkingText")}THINK ${DIM}${preview ? `thinking · ${preview}...` : "thinking"}${RESET}`,
				);
				break;
			}
			case "summary": {
				lines.push(
					`${theme.fgRaw("thinkingText")}THINK \x1b[2m${text.trim().slice(0, 150)}\x1b[0m`,
				);
				break;
			}
			case "expanded": {
				lines.push(
					`${theme.fgRaw("thinkingText")}THINK ${BOLD}reasoning${RESET}`,
				);
				this.renderThinkingExpanded(text, lines);
				break;
			}
		}

		return lines;
	}

	/**
	 * Render thinking text in expanded mode with code block syntax highlighting.
	 * Parses fenced code blocks, applies highlightAuto, and wraps plain text.
	 */
	private renderThinkingExpanded(text: string, lines: string[]): void {
		const rawLines = text.split("\n");
		let inCodeBlock = false;
		let codeContent = "";
		let codeBlockLang: string | null = null;
		const fg = theme.fgRaw("thinkingText") + DIM;

		for (const rawLine of rawLines) {
			if (rawLine.startsWith("```")) {
				if (inCodeBlock) {
					// Flush code block with syntax highlighting
					const lang = codeBlockLang || null;
					if (lang) {
						let highlighted;
						try {
							highlighted = highlight(codeContent, lang);
						} catch {
							highlighted = highlightAuto(codeContent);
						}
						const langLabel = highlighted.language
							? ` ${highlighted.language} · ${codeContent.split("\n").length} lines`
							: "";
						lines.push(`${fg}  \`${rawLine}\`${langLabel}${RESET}`);
						for (const cl of highlighted.value.split("\n")) {
							lines.push(`${fg}  ${cl}${RESET}`);
						}
					} else {
						const codeLines = codeContent.split("\n");
						for (const cl of codeLines) {
							lines.push(`${fg}  ${cl}${RESET}`);
						}
					}
					inCodeBlock = false;
					codeContent = "";
					codeBlockLang = null;
				} else {
					inCodeBlock = true;
					codeBlockLang = extractLangFromFence(rawLine);
					lines.push(`${fg}  ${rawLine}${RESET}`);
				}
				continue;
			}

			if (inCodeBlock) {
				codeContent += rawLine + "\n";
			} else {
				// Wrap plain text
				const wrapped = this.wrapText(rawLine, this.currentWidth - 4);
				for (const w of wrapped) {
					lines.push(`${fg}  ${renderInline(w, fg)}${RESET}`);
				}
			}
		}

		// Flush any unterminated code block
		if (inCodeBlock && codeContent) {
			lines.push(`${fg}  [code block open]${RESET}`);
			for (const cl of codeContent.split("\n")) {
				lines.push(`${fg}  ${cl}${RESET}`);
			}
		}
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
		const thumbColor = theme.fgRaw("selected");
		const barColor = theme.fgRaw("separator");
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
		const keepBottomAnchored = this._atBottom;
		// Drop oldest turns beyond the cap to keep memory and render time bounded.
		if (turns.length > this.maxTurns) {
			this.turns = turns.slice(turns.length - this.maxTurns);
		} else {
			this.turns = turns;
		}
		this.pruneBatchTaskTiming();
		this.invalidate();
		if (keepBottomAnchored) this._pendingScrollBottom = true;
	}

	/**
	 * batchTaskTiming is keyed by a spawn_agents tool_call_id and grows one
	 * entry per batch call for the life of the session — evict entries whose
	 * batch tool is no longer present in the retained turns (dropped by the
	 * maxTurns cap above, or the batch simply scrolled out).
	 */
	private pruneBatchTaskTiming(): void {
		if (this.batchTaskTiming.size === 0) return;
		const liveIds = new Set<string>();
		for (const turn of this.turns) {
			for (const chunk of turn.assistantMessage?.chunks ?? []) {
				if (chunk.type === "tool" && chunk.tool?.tool_name === "spawn_agents" && chunk.tool.tool_call_id) {
					liveIds.add(chunk.tool.tool_call_id);
				}
			}
		}
		for (const key of this.batchTaskTiming.keys()) {
			if (!liveIds.has(key)) this.batchTaskTiming.delete(key);
		}
	}
	private _viewportHeight: number = 0;

	private renderMarkdownLines(
		text: string,
		maxLen: number,
		_streaming: boolean,
		baseColor = theme.fg("assistantText", ""),
		firstLinePrefix = "",
	): string[] {
		const lines: string[] = [];
		const rawLines = text.split("\n");
		let inCodeBlock = false;
		let codeContent = "";
		let codeBlockLang: string | null = null;
		let prevEmptyLine = false;
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;
		for (let li = 0; li < rawLines.length; li++) {
			const rawLine = rawLines[li];

			if (rawLine.startsWith("```")) {
				if (inCodeBlock) {
					// Flush code block with syntax highlighting
					const lang = codeBlockLang || null;
					let renderedCode = codeContent;
					try {
						renderedCode = lang
							? highlight(codeContent, lang).value
							: highlightAuto(codeContent).value;
					} catch {
						// Fall back to the original code when no grammar matches.
					}
					for (const cl of renderedCode.split("\n")) {
						lines.push(`${bg}  ${cl}${bgReset}`);
					}
					const count = codeContent.split("\n").length;
					lines.push(
						`${bg}${DIM}  └─ ${count} line${count === 1 ? "" : "s"}${bgReset}`,
					);
					codeContent = "";
					codeBlockLang = null;
					inCodeBlock = false;
				} else {
					inCodeBlock = true;
					codeBlockLang = extractLangFromFence(rawLine);
					lines.push(
						`${bg}${DIM}  ┌─ ${codeBlockLang || "code"}${bgReset}`,
					);
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

			const effectiveColor = baseColor;
			const rendered = renderMarkdownLine(rawLine, effectiveColor);
			const wrapped = this.wrapText(rendered, maxLen);
			if (wrapped.length === 0 || (wrapped.length === 1 && wrapped[0] === "")) {
				if (!prevEmptyLine) {
					lines.push(`  ${effectiveColor} ${RESET}`);
				}
				prevEmptyLine = true;
				continue;
			}
			prevEmptyLine = false;
			for (let wi = 0; wi < wrapped.length; wi++) {
				const seg = `${effectiveColor}${wrapped[wi]}${RESET}`;
				const line =
					wi === 0
						? firstLinePrefix + seg
						: `  ${effectiveColor}${wrapped[wi]}${RESET}`;
				// No streaming cursor
				lines.push(line);
			}
		}

		if (inCodeBlock && codeContent) {
			let renderedCode = codeContent;
			try {
				renderedCode = codeBlockLang
					? highlight(codeContent, codeBlockLang).value
					: highlightAuto(codeContent).value;
			} catch {
				// Keep incomplete streaming code readable when its grammar is unknown.
			}
			for (const cl of renderedCode.split("\n")) {
				lines.push(`${bg}  ${cl}${bgReset}`);
			}
			lines.push(`${bg}${DIM}  └─ streaming${bgReset}`);
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

	// Strip ANSI escape codes for plain-text analysis (table detection, etc.)
	private static stripAnsi(s: string): string {
		return s.replace(/\x1b\[[0-9;]*m/g, "");
	}

	private isTableStart(lines: string[], index: number): boolean {
		return (
			this.isTableRow(lines[index] || "") &&
			this.isTableSeparator(lines[index + 1] || "")
		);
	}

	private isTableRow(line: string): boolean {
		const plain = TranscriptDisplay.stripAnsi(line);
		const trimmed = plain.trim();
		return trimmed.includes("|") && trimmed.split("|").length >= 3;
	}

	private isTableSeparator(line: string): boolean {
		const cells = this.splitTableRow(line);
		if (cells.length < 2) return false;
		return cells.every((cell) =>
			/^:?-+:?$/.test(TranscriptDisplay.stripAnsi(cell.trim())),
		);
	}

	private splitTableRow(line: string): string[] {
		const plain = TranscriptDisplay.stripAnsi(line);
		const trimmed = plain.trim();
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

		const borderColor = theme.fgRaw("borderMuted");
		const headerColor = theme.fgRaw("assistantText");
		const rowColor = theme.fgRaw("assistantText");
		const altRowColor = theme.fgRaw("dim");
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
		const borderColor = theme.fgRaw("borderMuted");

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
		// Hook guidance is part of the model-visible tool result. Strip it only
		// from this local display copy so internal instructions never leak into
		// the user-facing transcript.
		const postEdit = extractPostEditDiagnostics(tool.result);
		tool = {
			...tool,
			result: stripInternalHookGuidance(postEdit.text),
			partialResult: stripInternalHookGuidance(tool.partialResult),
		};
		const lines: string[] = [];
		const subagent = tool.tool_name === "spawn_agent";
		const subagentBatch = tool.tool_name === "spawn_agents";
		const batchTally = subagentBatch ? this.computeBatchTally(tool) : null;
		const batchFailed = batchTally?.failed ?? 0;
		const glyph = tool.isError
			? theme.fg("toolError", "×")
			: tool.isComplete && batchFailed > 0
				? theme.fg("warning", "!")
			: tool.isComplete
				? theme.fg("toolSuccess", "✓")
				: theme.fg("toolRunning", this.spinnerFrame());
		const status = tool.isError
			? theme.fg("toolError", "error")
			: tool.isComplete && batchFailed > 0
				? theme.fg("warning", `partial · ${batchFailed} failed`)
			: tool.isComplete
				? theme.fg("toolSuccess", "done")
				: batchTally
					? theme.fg(
							"toolStreaming",
							`${batchTally.completed + batchTally.running}/${batchTally.total} running${
								batchTally.failed > 0 ? ` · ${batchTally.failed} failed` : ""
							}`,
						)
					: tool.partialResult || tool.streamOutput
						? theme.fg("toolStreaming", "streaming")
						: theme.fg("toolRunning", "running");
		const subagentAgent = subagent
			? String(tool.details?.agent || tool.args?.agent || "general")
			: "";

		// Extract file path for write_file / edit_file to show in header
		const filePath = (() => {
			const args = tool.args || {};
			const path =
				stringArg(args, "path") ||
				stringArg(args, "file_path") ||
				streamedStringArg(tool.partialResult, "path") ||
				streamedStringArg(tool.partialResult, "file_path") ||
				(tool.tool_name === "write_file"
					? /^Created\s+(.+?)(?:\s+\([^\n]*\))?(?:\n|$)/.exec(tool.result ?? "")?.[1]
					: undefined);
			if (!path) return "";
			if (tool.tool_name === "write_file" || tool.tool_name === "edit_file") {
				return path;
			}
			return "";
		})();

		const summary = this.toolSummary(tool);
		const elapsed =
			tool.durationMs !== undefined ? formatDurationMs(tool.durationMs) : "";
		const base = subagent
			? `${glyph} ${theme.fg("toolTitle", "subagent")} ${theme.fg("active", subagentAgent)} ${status}`
			: subagentBatch
				? `${glyph} ${theme.fg("toolTitle", "subagents")} ${status}`
			: filePath
				? `${glyph} ${theme.fg("toolTitle", tool.tool_name)} ${DIM}${filePath}${RESET} ${status}`
				: `${glyph} ${theme.fg("toolTitle", tool.tool_name)} ${status}`;
		const middle = summary ? `${DIM}${summary}${RESET}` : "";
		const right = elapsed ? `${DIM}${elapsed}${RESET}` : "";
		let row = [base, middle].filter(Boolean).join(` ${DIM}·${RESET} `);
		if (right) {
			const available = Math.max(1, width - 4);
			const gap = available - visibleWidth(row) - visibleWidth(right);
			row = gap >= 2 ? `${row}${" ".repeat(gap)}${right}` : `${row} ${right}`;
		}
		lines.push(clampLineToWidth(row, Math.max(1, width - 4)) + RESET);

		// Always show the result (diff) for edit_file and write_file even when collapsed.
		const showDiffResult =
			!this.toolsExpanded &&
			["edit_file", "write_file"].includes(tool.tool_name) &&
			!!tool.result;
		if (showDiffResult) {
			const resultText = tool.result ?? "";
			const label = tool.isError ? "error" : "result";
			if (tool.isError) {
				const resultLines = resultText.split("\n");
				lines.push(
					`${theme.fg("toolError", "│ ")}${BOLD}${theme.fg("toolError", label)}${RESET} ${resultLines[0]}`,
				);
				for (let ri = 1; ri < resultLines.length; ri++) {
					lines.push(`${theme.fg("toolError", "│ ")}${resultLines[ri]}`);
				}
			} else {
				// Syntax-highlight the diff in collapsed view.
				const diffLines = this.renderDiffBlock(
					resultText,
					Math.max(20, width - 4),
					this.detectLanguage(filePath),
				);
				lines.push(`${theme.fg("dim", "│ ")}${BOLD}${label}${RESET}`);
				for (const dl of diffLines) {
					lines.push(`${theme.fg("dim", "│ ")}${dl}`);
				}
			}
		}
		const compactPreview =
			!showDiffResult &&
				!this.toolsExpanded &&
				!subagent &&
				!subagentBatch
				? this.collapsedToolPreview(tool)
				: "";
		if (compactPreview) {
			const label = tool.isError
				? theme.fg("toolError", "error")
				: !tool.isComplete
					? theme.fg("toolRunning", "live")
					: theme.fg("muted", "output");
			lines.push(
				clampLineToWidth(
					`${theme.fg("dim", "└─")} ${label} ${compactPreview}${RESET}`,
					Math.max(1, width - 4),
				),
			);
		}
		for (const block of postEdit.blocks) {
			lines.push(
				...this.renderPostEditDiagnostics(
					block,
					Math.max(20, width - 4),
				),
			);
		}
		if (!this.toolsExpanded && !subagent && !subagentBatch) return lines;
		if (!subagent && !subagentBatch) {
			lines.push(
				`${theme.fg("dim", "│ ")}${theme.fg("active", "◆ details")}`,
			);
		}
		for (const detailLine of this.toolDetailLines(tool, width - 2)) {
			const wrapped = this.wrapText(detailLine, Math.max(20, width - 4));
			for (const line of wrapped) {
				lines.push(`${theme.fg("dim", "│ ")}${line}`);
			}
		}

		return lines;
	}

	private collapsedToolPreview(tool: ToolExecution): string {
		const raw = tool.streamOutput || tool.result || "";
		if (!raw.trim()) return "";
		const firstLine = raw
			.split("\n")
			.map((line) => line.trim())
			.find(Boolean);
		if (!firstLine) return "";
		const preview = compactText(firstLine);
		const summary = this.toolSummary(tool);
		if (!tool.isError && preview === summary) return "";
		return clampLineToWidth(preview, 120);
	}

	private renderPostEditDiagnostics(
		block: PostEditDiagnosticBlock,
		width: number,
	): string[] {
		const count = block.diagnostics.length;
		const lines = [
			`${theme.fg("dim", "│ ")}${theme.fg("warning", "◆")} ${BOLD}${theme.fg("warning", "DIAGNOSTICS")}${RESET} ${DIM}${count} issue${count === 1 ? "" : "s"}${RESET}`,
			`${theme.fg("dim", "│ ")}${theme.fg("muted", block.file)}${RESET}`,
		];
		if (count === 0) {
			lines.push(
				`${theme.fg("dim", "│ ")}${DIM}Diagnostics were reported but could not be parsed.${RESET}`,
			);
			return lines;
		}
		for (const diagnostic of block.diagnostics) {
			const label = diagnostic.label ? ` ${diagnostic.label}` : "";
			lines.push(
				`${theme.fg("dim", "│ ")}${theme.fg("toolError", "×")} ${theme.fg("active", `${diagnostic.line}:${diagnostic.column}`)}${theme.fg("muted", label)}${RESET}`,
			);
			for (const messageLine of this.wrapText(
				diagnostic.message,
				Math.max(16, width - 6),
			)) {
				lines.push(`${theme.fg("dim", "│   ")}${messageLine}${RESET}`);
			}
		}
		return lines;
	}

	private detailSection(label: string, meta = ""): string {
		return `${theme.fg("active", "── ")}${BOLD}${label.toUpperCase()}${RESET}${meta ? `  ${DIM}${meta}${RESET}` : ""}`;
	}

	private toolSummary(tool: ToolExecution): string {
		const args = tool.args || {};
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		if (tool.tool_name === "write_file") {
			const content = stringArg(args, "content") || "";
			const lineCount = content ? content.split("\n").length : 0;
			const parts = [];
			if (content) parts.push(`${content.length} bytes`);
			if (lineCount) parts.push(`${lineCount} lines`);
			return parts.join(" · ");
		}
		if (tool.tool_name === "edit_file") {
			const editCount = Array.isArray(args.edits) ? args.edits.length : 1;
			return `${editCount} edit${editCount === 1 ? "" : "s"}`;
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
		if (tool.tool_name === "spawn_agent") {
			return compactText(stringArg(args, "task") || "").slice(0, 80);
		}
		if (tool.tool_name === "spawn_agents") {
			const tasks = Array.isArray(args.tasks) ? args.tasks.length : 0;
			return tasks ? `${tasks} tasks` : "";
		}
		if (path) return path;
		const result = tool.result ?? tool.partialResult;
		return result ? compactText(result).slice(0, 80) : "";
	}

	private toolDetailLines(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const result = tool.result ?? tool.partialResult;

		if (tool.tool_name === "spawn_agent") {
			return this.renderSubagentDetails(tool, width);
		}
		if (tool.tool_name === "spawn_agents") {
			return this.renderSubagentBatchDetails(tool, width);
		}

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
				lines.push(this.detailSection("arguments"));
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
			lines.push(this.detailSection(tool.isError ? "error" : "result"));
			lines.push(...this.previewBlock(result, width));
		} else if (!result && !tool.isComplete) {
			lines.push(`${DIM}waiting for result...${RESET}`);
		}

		return lines;
	}

	/**
	 * Live progress tally for a spawn_agents batch. Before completion the only
	 * signal available is the `▶/✓/×` marker stream on streamOutput (the tool
	 * only reports structured total/completed/failed once every task
	 * finishes), so this is the single place that parses it — reused by both
	 * the collapsed header and the expanded per-task breakdown.
	 */
	private computeBatchTally(tool: ToolExecution): {
		total: number;
		completed: number;
		failed: number;
		running: number;
		liveStatus: Map<number, "running" | "completed" | "failed">;
		taskElapsedMs: Map<number, number>;
	} {
		const tasks = Array.isArray(tool.args?.tasks)
			? tool.args.tasks.filter(
					(task): task is Record<string, unknown> =>
						typeof task === "object" && task !== null,
				)
			: [];
		const details = tool.details ?? {};
		const liveStatus = new Map<number, "running" | "completed" | "failed">();
		for (const line of (tool.streamOutput ?? "").split("\n")) {
			const match = /^([▶✓×])\s+(\d+)\s+/.exec(line.trim());
			if (!match) continue;
			liveStatus.set(
				Number(match[2]),
				match[1] === "▶"
					? "running"
					: match[1] === "✓"
						? "completed"
						: "failed",
			);
		}

		const timingKey = tool.tool_call_id ?? "";
		let timing = this.batchTaskTiming.get(timingKey);
		if (!timing) {
			timing = new Map();
			this.batchTaskTiming.set(timingKey, timing);
		}
		const now = Date.now();
		const taskElapsedMs = new Map<number, number>();
		for (const [index, status] of liveStatus) {
			let entry = timing.get(index);
			if (!entry) {
				entry = { startedAt: now };
				timing.set(index, entry);
			}
			if (status !== "running" && entry.endedAt === undefined) {
				entry.endedAt = now;
			}
			taskElapsedMs.set(index, (entry.endedAt ?? now) - entry.startedAt);
		}

		const completed =
			typeof details.completed === "number" && Number.isFinite(details.completed)
				? Math.max(0, Math.trunc(details.completed))
				: [...liveStatus.values()].filter((s) => s === "completed").length;
		const failed =
			typeof details.failed === "number" && Number.isFinite(details.failed)
				? Math.max(0, Math.trunc(details.failed))
				: [...liveStatus.values()].filter((s) => s === "failed").length;
		const running = [...liveStatus.values()].filter(
			(status) => status === "running",
		).length;
		const reportedTotal =
			typeof details.total === "number" && Number.isFinite(details.total)
				? Math.max(0, Math.trunc(details.total))
				: 0;
		const observedTotal =
			liveStatus.size > 0 ? Math.max(...liveStatus.keys()) + 1 : 0;
		// Tool arguments can still be streaming when the first progress marker
		// arrives. Never render an impossible ratio such as "1/0", and also
		// defend against stale or malformed structured totals.
		const total = Math.max(
			reportedTotal,
			tasks.length,
			observedTotal,
			completed + failed + running,
		);
		return { total, completed, failed, running, liveStatus, taskElapsedMs };
	}

	private batchAgentStreams(tool: ToolExecution): Map<number, string> {
		const streams = new Map<number, string>();
		const stored = tool.details?.streamTranscript;
		const source =
			typeof stored === "string" ? stored : (tool.streamOutput ?? "");
		for (const line of source.split("\n")) {
			const match = /^↳\s+(\d+)\s+(.+)$/.exec(line);
			if (!match) continue;
			try {
				const delta = JSON.parse(match[2]) as unknown;
				if (typeof delta !== "string") continue;
				const index = Number(match[1]);
				streams.set(index, (streams.get(index) ?? "") + delta);
			} catch {
				// Ignore an incomplete update line while it is still streaming.
			}
		}
		return streams;
	}

	private renderSubagentText(
		text: string,
		width: number,
		streaming: boolean,
	): string[] {
		const visibleText = stripAcceptanceForDisplay(text);
		if (!visibleText) return [];
		const markdown =
			!this.toolsExpanded && visibleText.length > 800
				? this.withTruncationMarker(visibleText.slice(0, 800))
				: visibleText;
		return this.renderMarkdownLines(
			markdown,
			Math.max(16, width),
			streaming,
			theme.fg("assistantText", ""),
		);
	}

	private distinctSubagentOutputs(liveText: string, finalText: string): string[] {
		const live = stripAcceptanceForDisplay(liveText).trim();
		const final = stripAcceptanceForDisplay(finalText).trim();
		if (!live) return final ? [final] : [];
		if (!final || live === final || live.includes(final)) return [live];
		return [live, final];
	}

	private childToolExecution(call: ChildToolCall): ToolExecution {
		const parsedArgs = parseJsonMaybe(call.args);
		const args =
			parsedArgs &&
			typeof parsedArgs === "object" &&
			!Array.isArray(parsedArgs)
				? (parsedArgs as Record<string, unknown>)
				: call.args
					? { input: call.args }
					: {};
		const status =
			call.status ?? (call.isError ? "failed" : "completed");
		return {
			tool: call.toolName,
			tool_name: call.toolName,
			tool_call_id: call.toolCallId,
			args,
			result: call.resultPreview,
			isError: status === "failed" || call.isError === true,
			isComplete: status !== "running",
		};
	}

	/**
	 * Render a child agent's stream using the same chronological model as the
	 * parent transcript: thinking → response → tool → response, in arrival
	 * order. Tool rows stay where they were called instead of being collected
	 * into a separate activity section.
	 */
	private renderSubagentFlow(
		chunks: ChildChunk[],
		width: number,
		showAgent: boolean,
	): string[] {
		const lines: string[] = [];
		const agentIds = [...new Set(
			chunks.map((chunk) => chunk.agentId).filter(Boolean),
		)];
		const plural = showAgent || agentIds.length > 1;
		const runLabel = plural
			? `SUBAGENTS · ${agentIds.length || "?"} CHILD RUNS`
			: `SUBAGENT${agentIds[0] ? ` · ${agentIds[0]}` : ""}`;
		lines.push(
			`${theme.fg("active", `╭─ ${runLabel}`)}${RESET}`,
		);
		let contentBuffer = "";
		let contentAgent = "";
		let lastAgent = "";
		let lastWasThinking = false;

		const showAgentBoundary = (agentId: string) => {
			if (!showAgent || !agentId || agentId === lastAgent) return;
			lines.push(
				`${theme.fg("active", `◇ CHILD · ${agentId}`)}${RESET}`,
			);
			lastAgent = agentId;
		};
		const flushContent = () => {
			if (!contentBuffer) return;
			showAgentBoundary(contentAgent);
			if (lastWasThinking) {
				lines.push(
					`${theme.fgRaw("separator")}${DIM}─── response ───${RESET}`,
				);
			}
			const visible = stripAcceptanceForDisplay(
				stripThinkTags(contentBuffer),
			);
			for (const line of this.renderMarkdownLines(
				visible,
				Math.max(16, width),
				true,
			)) {
				lines.push(line);
			}
			contentBuffer = "";
			contentAgent = "";
			lastWasThinking = false;
		};

		for (const chunk of chunks) {
			if (chunk.type === "content") {
				if (contentBuffer && contentAgent !== chunk.agentId) {
					flushContent();
				}
				contentAgent = chunk.agentId;
				contentBuffer += chunk.contentText ?? "";
				continue;
			}

			flushContent();
			showAgentBoundary(chunk.agentId);
			if (chunk.type === "thinking") {
				const thinkingLines = this.renderThinkingChunk(
					{
						seq: chunk.seq,
						type: "thinking",
						contentText: chunk.contentText,
						isComplete: chunk.isComplete,
					},
					!chunk.isComplete,
				);
				lines.push(...thinkingLines);
				lastWasThinking = thinkingLines.length > 0;
				continue;
			}
			if (chunk.type === "tool" && chunk.tool) {
				lines.push(
					...this.renderTool(
						this.childToolExecution(chunk.tool),
						Math.max(20, width),
					),
				);
				lastWasThinking = false;
			}
		}
		flushContent();
		lines.push(
			`${theme.fg("active", "╰─ RETURN TO PARENT")}${RESET}`,
		);
		return lines;
	}

	private renderSubagentBatchDetails(
		tool: ToolExecution,
		width: number,
	): string[] {
		const tasks = Array.isArray(tool.args?.tasks)
			? tool.args.tasks.filter(
					(task): task is Record<string, unknown> =>
						typeof task === "object" && task !== null,
				)
			: [];
		const details = tool.details ?? {};
		const results = Array.isArray(details.results)
			? details.results.filter(
					(result): result is Record<string, unknown> =>
						typeof result === "object" && result !== null,
				)
			: [];
		const resultByIndex = new Map(
			results.map((result) => [Number(result.index), result]),
		);
		const { liveStatus, taskElapsedMs } = this.computeBatchTally(tool);
		const agentStreams = this.batchAgentStreams(tool);
		const childChunks = Array.isArray(details.childChunks)
			? (details.childChunks as ChildChunk[])
			: [];
		const hasOrderedFlow = childChunks.length > 0;
		// The N/M tally already appears in the collapsed header row above this
		// detail block — repeating it here as its own line was pure duplication.
		const lines: string[] = [];

		for (let index = 0; index < tasks.length; index++) {
			const task = tasks[index];
			const result = resultByIndex.get(index);
			const resultError = result?.isError === true;
			const state = result
				? resultError
					? "failed"
					: "completed"
				: liveStatus.get(index) ?? "queued";
			const icon =
				state === "failed"
					? theme.fg("toolError", "×")
					: state === "completed"
						? theme.fg("toolSuccess", "✓")
						: state === "running"
							? theme.fg("toolRunning", this.spinnerFrame())
							: theme.fg("dim", "·");
			const agent =
				typeof task.agent === "string" && task.agent
					? task.agent
					: "general";
			const taskText =
				typeof task.task === "string"
					? compactText(task.task).slice(0, 100)
					: `Task ${index + 1}`;
			const elapsedMs = taskElapsedMs.get(index);
			const elapsed =
				elapsedMs !== undefined ? ` ${DIM}${formatDurationMs(elapsedMs)}${RESET}` : "";
			const queuedTag = state === "queued" ? ` ${DIM}queued${RESET}` : "";
			lines.push(
				clampLineToWidth(
					`${icon} ${theme.fg("active", `${index + 1}. ${agent}`)} ${DIM}${taskText}${RESET}${queuedTag}${elapsed}`,
					Math.max(20, width),
				),
			);
			if (this.toolsExpanded && !hasOrderedFlow) {
				const liveText = agentStreams.get(index) ?? "";
				const resultText =
					typeof result?.content === "string" ? result.content : "";
				const texts = this.distinctSubagentOutputs(liveText, resultText);
				for (const text of texts) {
					for (const preview of this.renderSubagentText(
						text,
						Math.max(16, width - 4),
						!result,
					)) {
						lines.push(`  ${preview}`);
					}
				}
			}
		}
		if (hasOrderedFlow) {
			lines.push(...this.renderSubagentFlow(childChunks, width, true));
		} else {
			const childToolCalls =
				details.childToolCalls as ChildToolCall[] | undefined;
			lines.push(
				...this.renderSubagentActivity(
					childToolCalls,
					width,
					this.toolsExpanded ? Number.POSITIVE_INFINITY : 4,
				),
			);
		}
		return lines;
	}

	private renderSubagentDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const args = tool.args || {};
		const details = tool.details || {};
		const metrics =
			details.metrics && typeof details.metrics === "object"
				? (details.metrics as Record<string, unknown>)
				: {};
		const liveElapsedMs =
			!tool.isComplete && tool.startedAt !== undefined
				? Date.now() - tool.startedAt
				: undefined;
		const metadata = [
			typeof metrics.turns === "number" ? `${metrics.turns} turn(s)` : "",
			typeof metrics.toolCalls === "number"
				? `${metrics.toolCalls} tool call(s)`
				: "",
			typeof metrics.durationMs === "number"
				? formatDurationMs(metrics.durationMs)
				: liveElapsedMs !== undefined
					? `${formatDurationMs(liveElapsedMs)} elapsed`
					: "",
		]
			.filter(Boolean)
			.join(" · ");
		if (metadata) {
			lines.push(`${DIM}${metadata}${RESET}`);
		}

		const branch = typeof details.branch === "string" ? details.branch : "";
		const commit = typeof details.commit === "string" ? details.commit : "";
		if (branch || commit) {
			lines.push(
				`${DIM}${[branch && `branch ${branch}`, commit && `commit ${commit.slice(0, 12)}`].filter(Boolean).join(" · ")}${RESET}`,
			);
		}

		const task = stringArg(args, "task");
		if (task) {
			for (const line of this.previewBlock(task, Math.max(16, width - 4), 800)) {
				lines.push(`${theme.fg("dim", "→ ")}${line}`);
			}
		}

		const childChunks = Array.isArray(details.childChunks)
			? (details.childChunks as ChildChunk[])
			: [];
		if (childChunks.length > 0) {
			lines.push(...this.renderSubagentFlow(childChunks, width, false));
		} else {
			const childToolCalls =
				details.childToolCalls as ChildToolCall[] | undefined;
			lines.push(
				...this.renderSubagentActivity(
					childToolCalls,
					width,
					this.toolsExpanded ? Number.POSITIVE_INFINITY : 4,
					false,
					false,
				),
			);
		}

		const storedTranscript =
			typeof details.streamTranscript === "string"
				? details.streamTranscript
				: "";
		const liveOutput = tool.isComplete ? storedTranscript : (tool.streamOutput ?? "");
		const finalOutput = tool.isComplete ? (tool.result ?? "") : "";
		const orderedContent = childChunks
			.filter((chunk) => chunk.type === "content")
			.map((chunk) => chunk.contentText ?? "")
			.join("");
		const outputs =
			this.toolsExpanded && childChunks.length === 0
				? this.distinctSubagentOutputs(liveOutput, finalOutput)
				: !this.toolsExpanded &&
						tool.isComplete &&
						finalOutput &&
						!orderedContent.includes(finalOutput)
					? [finalOutput]
					: [];
		if (outputs.length > 0) {
			// Ctrl+O is the explicit full-detail view. Keep collapsed tool rows
			// compact, but never discard child-agent progress or the final report here.
			for (const output of outputs) {
				for (const line of this.renderSubagentText(
					output,
					Math.max(16, width - 4),
					!tool.isComplete,
				)) {
					lines.push(`  ${line}`);
				}
			}
		} else if (!tool.isComplete && this.toolsExpanded) {
			lines.push(`${theme.fg("dim", "  waiting for agent output…")}${RESET}`);
		}

		return lines;
	}

	private renderSubagentActivity(
		calls: ChildToolCall[] | undefined,
		width: number,
		limit = Number.POSITIVE_INFINITY,
		showHeading = true,
		showAgent = true,
	): string[] {
		if (!calls?.length) return [];
		const visible = calls.slice(-limit);
		const hidden = calls.length - visible.length;
		const lines = showHeading
			? [this.detailSection(
				"activity",
				`${calls.length} tool call${calls.length === 1 ? "" : "s"}${hidden ? ` · latest ${visible.length}` : ""}`,
			)]
			: hidden
				? [`  ${theme.fg("dim", `⋯ ${hidden} earlier tool call${hidden === 1 ? "" : "s"} hidden`)}`]
				: [];
		const bg = theme.bg("mdCodeBlockBg", "");
		for (const call of visible) {
			const status =
				call.status ?? (call.isError ? "failed" : "completed");
			const icon =
				status === "failed"
					? theme.fg("toolError", "×")
					: status === "running"
						? theme.fg("toolRunning", this.spinnerFrame())
						: theme.fg("toolSuccess", "✓");
			const summary = this.subagentCallSummary(call.args);
			const row = [
				`${icon} ${theme.fg("toolTitle", call.toolName)}`,
				showAgent && call.agentId ? `${DIM}${call.agentId}${RESET}` : "",
				summary ? `${DIM}${summary}${RESET}` : "",
			]
				.filter(Boolean)
				.join(` ${DIM}·${RESET} `);
			lines.push(`${bg}${clampLineToWidth(row, Math.max(20, width))}${RESET}`);
			if (this.toolsExpanded && call.resultPreview) {
				const result = compactText(call.resultPreview);
				lines.push(
					`${bg}${DIM}  └ ${clampLineToWidth(result, Math.max(16, width - 4))}${RESET}`,
				);
			}
		}
		return lines;
	}

	private subagentCallSummary(raw: string): string {
		const text = raw.trim();
		if (!text || text === "{}") return "";
		const parsed = parseJsonMaybe(text);
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			const args = parsed as Record<string, unknown>;
			for (const key of ["path", "file_path", "pattern", "command", "query"]) {
				if (typeof args[key] === "string") {
					return `${key}=${compactText(args[key] as string).slice(0, 96)}`;
				}
			}
		}
		return compactText(text).slice(0, 100);
	}

	private renderWriteDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const args = tool.args || {};
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		const content = stringArg(args, "content");
		const streaming = !tool.isComplete;

		if (path) lines.push(this.detailSection("file", path));

		if (content !== undefined && content !== "") {
			const lineCount = content.split("\n").length;
			const meta = streaming
				? `${DIM}${content.length} bytes · ${lineCount} lines · streaming${RESET}`
				: `${DIM}${content.length} bytes · ${lineCount} lines${RESET}`;
			lines.push(this.detailSection("content", meta));
			const lang = this.detectLanguage(path);
			lines.push(...this.renderFileContent(content, width, lineCount, lang));
		} else if (streaming) {
			lines.push(`${DIM}writing…${RESET}`);
		}

		// Show error result only (skip diff — content is already rendered above).
		if (tool.result) {
			const resultText = tool.result;
			if (tool.isError) {
				lines.push(this.detailSection("error"));
				lines.push(...this.previewBlock(resultText, width));
			} else if (!content) {
				// No content shown above; show the diff result.
				lines.push(this.detailSection("result"));
				lines.push(...this.renderDiffBlock(resultText, width, this.detectLanguage(path)));
			}
		} else if (!streaming && !content) {
			lines.push(`${DIM}no output${RESET}`);
		}

		return lines;
	}

	/** Parse accumulated partialResult JSON to extract tool args. */
	private renderEditDetails(tool: ToolExecution, width: number): string[] {
		const lines: string[] = [];
		const args = tool.args || {};
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		const streaming = !tool.isComplete;
		const edits = normalizeEditArgs(args);
		const language = this.detectLanguage(path);

		if (path) lines.push(this.detailSection("file", path));

		for (let i = 0; i < edits.length; i++) {
			lines.push(this.detailSection(`edit ${i + 1}`, `${i + 1} of ${edits.length}`));
			const oldText = edits[i].oldText;
			const newText = edits[i].newText;

			if (oldText) {
				const oldLineCount = oldText.split("\n").length;
				const oldMeta = streaming
					? `${oldText.length} bytes · ${oldLineCount} lines · streaming`
					: `${oldText.length} bytes · ${oldLineCount} lines`;
				lines.push(
					`${theme.fgRaw("diffRemoved")}── - OLD${RESET}  ${DIM}${oldMeta}${RESET}`,
				);
				lines.push(
					...this.renderFileContent(
						oldText,
						width,
						oldLineCount,
						language,
					),
				);
			}
			if (newText) {
				const newLineCount = newText.split("\n").length;
				const newMeta = streaming
					? `${newText.length} bytes · ${newLineCount} lines · streaming`
					: `${newText.length} bytes · ${newLineCount} lines`;
				lines.push(
					`${theme.fgRaw("diffAdded")}── + NEW${RESET}  ${DIM}${newMeta}${RESET}`,
				);
				lines.push(
					...this.renderFileContent(
						newText,
						width,
						newLineCount,
						language,
					),
				);
			}
		}

		if (edits.length === 0 && streaming) {
			lines.push(`${DIM}editing…${RESET}`);
		}

		if (tool.result) {
			const resultText = tool.result.startsWith("Error:")
				? tool.result
				: tool.result;
			lines.push(this.detailSection(tool.isError ? "error" : "result"));
			if (tool.isError) {
				lines.push(...this.previewBlock(resultText, width));
			} else {
				lines.push(...this.renderDiffBlock(resultText, width, language));
			}
		}

		return lines;
	}

	private renderFileDiffDetails(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const path = stringArg(args, "path") || stringArg(args, "file_path");
		if (path) lines.push(this.detailSection("file", path));
		if (args.staged) lines.push(`${DIM}staged changes${RESET}`);
		const result = tool.result ?? tool.partialResult;
		if (result) {
			lines.push(...this.renderDiffBlock(result, width, this.detectLanguage(path)));
		}
		return lines;
	}

	private renderBashDetails(tool: ToolExecution, width: number): string[] {
		const args = tool.args || {};
		const lines: string[] = [];
		const command = stringArg(args, "command") || "";
		const timeout = args.timeout ? `${Number(args.timeout)}ms` : "30000ms";
		if (command) {
			lines.push(this.detailSection("command", `timeout ${timeout}`));
			lines.push(...this.previewBlock(command, width));
		}
		const result = tool.result ?? tool.partialResult;
		if (result) {
			const label = tool.result
				? tool.isError
					? "error output"
					: "output"
				: "streaming output";
			lines.push(this.detailSection(label));
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
			lines.push(this.detailSection("mcp", `${serverParts[0]} · ${serverParts.slice(1).join("__")}`));
		}
		const argText = JSON.stringify(args, null, 2);
		if (argText && argText !== "{}") {
			lines.push(this.detailSection("arguments"));
			lines.push(...this.previewBlock(argText, width));
		}
		const result = tool.result ?? tool.partialResult;
		if (result) {
			lines.push(this.detailSection(tool.isError ? "mcp error" : "mcp result"));
			lines.push(...this.renderMcpResultBlocks(result, width));
		}
		return lines;
	}

	private renderPermissionBlock(result: string, width: number): string[] {
		const lines = [
			`${theme.fgRaw("warning")}${BOLD}permission / rejection${RESET}`,
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

	private renderDiffBlock(
		diff: string,
		width: number,
		language?: string,
	): string[] {
		if (!diff.trim()) return [`${DIM}(no diff)${RESET}`];
		const rawLines = this.truncateText(diff).split("\n");
		const lines: string[] = [];
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;

		for (const raw of rawLines) {
			const color = diffLineColor(raw);
			const content = raw.length ? raw.replace(/\t/g, "    ") : " ";

			// Keep the diff marker in its semantic color, but highlight the code
			// after it with the grammar selected from the edited file's path.
			if (
				(raw.startsWith("+") && !raw.startsWith("+++")) ||
				(raw.startsWith("-") && !raw.startsWith("---"))
			) {
				const prefix = raw[0];
				const codeText = raw.slice(1).replace(/\t/g, "    ");
				try {
					const highlighted = language
						? highlight(codeText, language)
						: highlightAuto(codeText);
					if (highlighted.value && highlighted.value !== codeText) {
						if (visibleWidth(content) <= width) {
							lines.push(
								`${bg}${color}${prefix}${RESET}${bg}${highlighted.value}${bgReset}`,
							);
							continue;
						}
					}
				} catch {
					// No highlighting available, fall through to plain rendering.
				}
			}

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
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;
		for (const raw of rawLines) {
			const content = raw.length ? raw.replace(/\t/g, "    ") : " ";
			const color = raw.startsWith("Error:")
				? theme.fgRaw("diffRemoved")
				: theme.fgRaw("terminalOutput");
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

	private previewBlock(
		text: string,
		width: number,
		maxChars = this.maxMessageLength,
	): string[] {
		if (!text) return [`${DIM}(empty)${RESET}`];
		const preview =
			text.length > maxChars
				? this.withTruncationMarker(text.slice(0, maxChars))
				: text;
		const rawLines = preview.split("\n");
		const lines: string[] = [];
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;
		let prevEmpty = false;
		for (const raw of rawLines) {
			const isEmpty = raw.length === 0;
			const formatted = isEmpty ? " " : raw.replace(/\t/g, "    ");
			if (isEmpty && prevEmpty) continue; // collapse consecutive blanks
			prevEmpty = isEmpty;
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

	// ── Pi-style line-numbered content rendering ────────────────────────────
	// Shows content with line numbers, collapsed to a preview when expanded,
	// with a line-number gutter like Pi's write tool.

	private renderPiContent(
		text: string,
		width: number,
		totalLines: number,
	): string[] {
		const lines: string[] = [];
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;
		const gutterColor = theme.fgRaw("dim");
		const contentColor = theme.fgRaw("assistantText");

		// Determine how many lines to show in collapsed mode
		const collapsedPreviewLines = 8;
		const showAll = totalLines <= collapsedPreviewLines;

		const rawLines = text.split("\n");
		const displayLines = showAll
			? rawLines
			: rawLines.slice(0, collapsedPreviewLines);

		// Calculate gutter width (line number column)
		const gutterWidth = String(totalLines).length + 1;
		const availableContentWidth = Math.max(20, width - gutterWidth - 2);

		for (let i = 0; i < displayLines.length; i++) {
			const lineNum = i + 1;
			const rawLine = displayLines[i];
			const formatted = rawLine.length ? rawLine.replace(/\t/g, "    ") : " ";

			// Truncate line to fit available width
			const displayContent =
				visibleWidth(formatted) > availableContentWidth
					? this.wrapText(formatted, availableContentWidth)
					: [formatted];

			for (let wi = 0; wi < displayContent.length; wi++) {
				const content = displayContent[wi];
				const numStr = String(lineNum + (wi > 0 ? 0 : 0)).padStart(
					gutterWidth - 1,
					" ",
				);
				lines.push(
					`${bg}${gutterColor}${numStr}│${RESET}${bg}${contentColor}${content}${bgReset}`,
				);
			}
		}

		// Add truncation hint if collapsed
		if (!showAll) {
			const remaining = totalLines - collapsedPreviewLines;
			lines.push(
				`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
			);
		}

		return lines;
	}

	wrapText(text: string, maxLineLength: number): string[] {
		const width = Math.max(1, Math.floor(maxLineLength));
		const lines: string[] = [];
		const rawLines = text.split("\n");
		for (const rawLine of rawLines) {
			if (visibleWidth(rawLine) <= width) {
				lines.push(rawLine);
			} else {
				const words = rawLine.split(/\s+/);
				let current = "";
				for (const word of words) {
					// A streamed tool result can contain hashes, minified JSON, or
					// other tokens with no whitespace. Split those tokens as well so
					// the frame-level width clamp does not silently discard their tail.
					const chunks = this.hardWrapVisible(word, width);
					for (let i = 0; i < chunks.length; i++) {
						const chunk = chunks[i];
						if (i > 0) {
							if (current) lines.push(current);
							current = chunk;
							continue;
						}
					if (current.length === 0) {
						current = chunk;
					} else if (
						visibleWidth(current) + 1 + visibleWidth(chunk) <= width
					) {
						current += ` ${chunk}`;
					} else {
						lines.push(current);
						current = chunk;
					}
						if (visibleWidth(current) === width && i < chunks.length - 1) {
							lines.push(current);
							current = "";
						}
					}
				}
				if (current) lines.push(current);
			}
		}
		return lines;
	}

	private hardWrapVisible(text: string, width: number): string[] {
		if (visibleWidth(text) <= width) return [text];
		const chunks: string[] = [];
		let chunk = "";
		let chunkWidth = 0;
		for (const char of text) {
			const charWidth = visibleWidth(char);
			if (chunk && chunkWidth + charWidth > width) {
				chunks.push(chunk);
				chunk = "";
				chunkWidth = 0;
			}
			chunk += char;
			chunkWidth += charWidth;
		}
		if (chunk) chunks.push(chunk);
		return chunks;
	}

	/** Detect language from file extension. */
	private detectLanguage(filePath: string | undefined): string | undefined {
		if (!filePath) return undefined;
		const ext = filePath.split(".").pop()?.toLowerCase();
		if (!ext) return undefined;
		// Map common extensions to highlight.js language names
		const extMap: Record<string, string> = {
			ts: "typescript",
			tsx: "tsx",
			js: "javascript",
			jsx: "jsx",
			py: "python",
			rs: "rust",
			go: "go",
			java: "java",
			rb: "ruby",
			php: "php",
			cs: "csharp",
			cpp: "cpp",
			c: "c",
			h: "c",
			hs: "haskell",
			ml: "ocaml",
			sh: "bash",
			zsh: "bash",
			bash: "bash",
			md: "markdown",
			html: "html",
			css: "css",
			json: "json",
			yaml: "yaml",
			yml: "yaml",
			xml: "xml",
			sql: "sql",
			toml: "toml",
			ini: "ini",
			conf: "ini",
			gitignore: "plaintext",
			env: "plaintext",
			svelte: "svelte",
			vue: "vue",
			astro: "astro",
			kt: "kotlin",
			swift: "swift",
			dart: "dart",
			lua: "lua",
			r: "r",
			pl: "perl",
			pm: "perl",
		};
		return extMap[ext];
	}

	/** Render file content with syntax highlighting and line numbers (Pi-style). */
	private renderFileContent(
		text: string,
		width: number,
		totalLines: number,
		language: string | undefined,
	): string[] {
		const lines: string[] = [];
		const bg = theme.bg("mdCodeBlockBg", "");
		const bgReset = RESET;
		const gutterColor = theme.fgRaw("dim");
		const plainColor = theme.fgRaw("assistantText");

		const collapsedPreviewLines = 8;
		const showAll = totalLines <= collapsedPreviewLines;

		const rawLines = text.split("\n");
		const displayLines = showAll
			? rawLines
			: rawLines.slice(0, collapsedPreviewLines);

		// Calculate gutter width
		const gutterWidth = String(totalLines).length + 1;
		const availableContentWidth = Math.max(20, width - gutterWidth - 2);

		if (language) {
			// Highlighted rendering: split each line by ANSI sequences, apply line numbers
			const highlighted = language
				? highlight(text, language)
				: highlightAuto(text);

			// Parse highlighted output into lines, each line may have ANSI color spans
			const hlLines = highlighted.value.split("\n");
			for (let i = 0; i < displayLines.length; i++) {
				const lineNum = i + 1;
				const hlLine = hlLines[i] || "";

				// If highlighted output is empty for this line, fall back to plain
				const content = hlLine.replace(/\x1b\[[\d;]*m/g, "");
				const displayContent =
					visibleWidth(content) > availableContentWidth
						? this.wrapText(content, availableContentWidth)
						: [content];

				for (let wi = 0; wi < displayContent.length; wi++) {
					const displayLine = displayContent[wi];
					const numStr = String(lineNum).padStart(gutterWidth - 1, " ");
					// Extract ANSI spans from highlighted line at the same wrap position
					let hlContent = this.extractHlSpan(hlLine, displayLine);
					if (!hlContent) hlContent = plainColor + displayLine + plainColor;
					lines.push(
						`${bg}${gutterColor}${numStr}│${RESET}${bg}${hlContent}${bgReset}`,
					);
				}
			}

			// Add language label in the gutter area if collapsed
			if (!showAll) {
				const remaining = totalLines - collapsedPreviewLines;
				lines.push(
					`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
				);
			}
		} else {
			// No language detection — plain text with line numbers
			for (let i = 0; i < displayLines.length; i++) {
				const lineNum = i + 1;
				const rawLine = displayLines[i];
				const formatted = rawLine.length ? rawLine.replace(/\t/g, "    ") : " ";

				const displayContent =
					visibleWidth(formatted) > availableContentWidth
						? this.wrapText(formatted, availableContentWidth)
						: [formatted];

				for (let wi = 0; wi < displayContent.length; wi++) {
					const content = displayContent[wi];
					const numStr = String(lineNum).padStart(gutterWidth - 1, " ");
					lines.push(
						`${bg}${gutterColor}${numStr}│${RESET}${bg}${plainColor}${content}${bgReset}`,
					);
				}
			}
			if (!showAll) {
				const remaining = totalLines - collapsedPreviewLines;
				lines.push(
					`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
				);
			}
		}

		return lines;
	}

	/** Extract the portion of a highlighted line corresponding to a display line. */
	private extractHlSpan(hlLine: string, displayLine: string): string | null {
		if (!hlLine || hlLine.trim().length === 0) return null;
		// If the plain text of the hl line matches the display line, use it directly
		const stripped = hlLine.replace(/\x1b\[[\d;]*m/g, "");
		if (stripped === displayLine) return hlLine;
		// Otherwise approximate: if displayLine is shorter (wrapped), take first N chars of hl
		if (displayLine.length < hlLine.length) {
			// Count visible chars needed
			const visLen = displayLine.length;
			let idx = 0;
			let visible = 0;
			let inSeq = false;
			while (idx < hlLine.length && visible < visLen) {
				if (hlLine[idx] === "\x1b") {
					inSeq = true;
				}
				if (inSeq && hlLine[idx] === "m") {
					inSeq = false;
				} else if (!inSeq) {
					visible++;
				}
				idx++;
			}
			return hlLine.slice(0, idx);
		}
		return hlLine;
	}

	private truncateText(text: string): string {
		if (text.length <= this.maxMessageLength) return text;
		return this.withTruncationMarker(text.slice(0, this.maxMessageLength));
	}

	/** Appends the shared "content cut off" marker used by every truncation path. */
	private withTruncationMarker(text: string): string {
		return `${text}\n\n${DIM}[truncated]${RESET}`;
	}
}
