// ── Transcript display component ────────────────────────────────────────────────
// Renders the full conversation history with streaming support and markdown.
// Chunks are interleaved in chronological order: thinking → content → tool → ...

import { DEFAULT_TRUNCATION } from "@logician/agent-core";
import type {
	ThinkingDisplayStyle,
	ToolExecution,
	Turn,
} from "@logician/coding-agent/sessions";
import {
	BOLD,
	type Component,
	clampLineToWidth,
	DIM,
	RESET,
	type Scrollable,
	visibleWidth,
} from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";
import { hasStreamingChunk, revisionText, stripThinkTags } from "./text-utils.ts";
import { wrapText } from "./layout.ts";
import { renderMarkdownLines } from "./render/markdown-table.ts";
import { renderThinkingChunk } from "./render/thinking.ts";
import { truncateText } from "./render/content.ts";
import {
	getSanitizationMetrics,
	renderTool,
	type RenderCtx,
	type SanitizedToolCache,
} from "./render/tool.ts";

// ── Options ────────────────────────────────────────────────────────────────────

interface TranscriptDisplayOptions {
	thinkingMode?: ThinkingDisplayStyle;
	maxMessageLength?: number;
	/** Max turns to keep in memory. Older turns are dropped. */
	maxTurns?: number;
	/** Max rendered lines before cutting off older content. */
	maxRenderedLines?: number;
}

// ── Component ──────────────────────────────────────────────────────────────────

export class TranscriptDisplay implements Component, Scrollable, RenderCtx {
	public readonly rendersViewport = true;
	private cachedWidth: number = 0;
	private cachedLines: string[] | null = null;
	// Not private: read by the extracted render-*.ts functions through the
	// RenderCtx interface, which TranscriptDisplay satisfies structurally.
	currentWidth: number = 80;
	private _scrollOffset: number = 0;
	private _totalHeight: number = 0;
	private _atBottom: boolean = true;
	private _pendingScrollBottom: boolean = false;
	private _newOutputBelow = false;
	private contentRevision = "";

	thinkingMode: ThinkingDisplayStyle;
	toolsExpanded = false;
	private expandedToolKeys = new Set<string>();
	private toolHitRegions: Array<{ start: number; end: number; key: string }> = [];
	private focusedToolKey: string | null = null;
	maxMessageLength: number;
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
	batchTaskTiming = new Map<
		string,
		Map<number, { startedAt: number; endedAt?: number }>
	>();
	sanitizedToolCache = new WeakMap<ToolExecution, SanitizedToolCache>();
	sanitizationMetrics = { cacheHits: 0, scannedCharacters: 0 };

	constructor(options: TranscriptDisplayOptions = {}) {
		this.thinkingMode = options.thinkingMode ?? "collapsed";
		this.maxMessageLength =
			options.maxMessageLength ?? DEFAULT_TRUNCATION.transcriptMessageMaxChars;
		this.maxTurns = options.maxTurns ?? 200;
		this.maxRenderedLines = options.maxRenderedLines ?? 2000;
	}

	private static readonly SPINNER_FRAMES = [
		"⠋",
		"⠙",
		"⠹",
		"⠸",
		"⠼",
		"⠴",
		"⠦",
		"⠧",
	];

	spinnerFrame(): string {
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
			this.spinnerTick =
				(this.spinnerTick + 1) % TranscriptDisplay.SPINNER_FRAMES.length;
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

	handleMouse(_column: number, row: number): boolean {
		if (
			this._newOutputBelow &&
			row === Math.max(0, this._viewportHeight - 1)
		) {
			this.scrollToBottom();
			this.invalidate();
			return true;
		}
		const contentRow = this._scrollOffset + row;
		const region = this.toolHitRegions.find(
			(candidate) => contentRow >= candidate.start && contentRow < candidate.end,
		);
		if (!region) return false;
		const keepBottomAnchored = this._atBottom;
		if (this.expandedToolKeys.has(region.key)) {
			this.expandedToolKeys.delete(region.key);
		} else {
			this.expandedToolKeys.add(region.key);
		}
		this.invalidate();
		if (keepBottomAnchored) this._pendingScrollBottom = true;
		return true;
	}

	/** Move keyboard focus between rendered tool cards and reveal the target. */
	focusTool(direction: 1 | -1): { index: number; total: number } | null {
		if (this.toolHitRegions.length === 0) return null;
		const currentIndex = this.toolHitRegions.findIndex(
			(region) => region.key === this.focusedToolKey,
		);
		const nextIndex =
			currentIndex < 0
				? direction > 0
					? 0
					: this.toolHitRegions.length - 1
				: (currentIndex + direction + this.toolHitRegions.length) %
					this.toolHitRegions.length;
		const region = this.toolHitRegions[nextIndex];
		this.focusedToolKey = region.key;
		const viewportHeight = Math.max(1, this._viewportHeight);
		if (region.start < this._scrollOffset) {
			this._scrollOffset = region.start;
		} else if (region.end > this._scrollOffset + viewportHeight) {
			this._scrollOffset = Math.max(0, region.end - viewportHeight);
		}
		this._atBottom =
			this._scrollOffset >= Math.max(0, this._totalHeight - viewportHeight);
		this._pendingScrollBottom = false;
		this.invalidate();
		return { index: nextIndex + 1, total: this.toolHitRegions.length };
	}

	/** Expand or collapse the keyboard-focused tool card. */
	toggleFocusedTool(): boolean | null {
		if (!this.focusedToolKey) return null;
		if (this.expandedToolKeys.has(this.focusedToolKey)) {
			this.expandedToolKeys.delete(this.focusedToolKey);
		} else {
			this.expandedToolKeys.add(this.focusedToolKey);
		}
		this.invalidate();
		return this.expandedToolKeys.has(this.focusedToolKey);
	}

	invalidate(): void {
		this.cachedLines = null;
	}

	getSanitizationMetrics(): {
		cacheHits: number;
		scannedCharacters: number;
	} {
		return getSanitizationMetrics(this);
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
		if (this._atBottom) this._newOutputBelow = false;
		// A streamed update may have invalidated the cached lines without being
		// rendered yet. If the user reaches the bottom of the last committed
		// layout, keep that intent through the next render as its height grows.
		this._pendingScrollBottom = this._atBottom;
	}

	scrollToBottom(): void {
		this._pendingScrollBottom = true;
		this._atBottom = true;
		this._newOutputBelow = false;
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
		if (this._atBottom) this._newOutputBelow = false;
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
		const pendingToolRegions: Array<{
			start: number;
			end: number;
			key: string;
		}> = [];
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
					const sysLines = renderMarkdownLines(
						content.slice(9),
						contentWidth - 2,
						false,
						theme.fgRaw("systemText") + RESET,
						"",
					);
					for (const line of sysLines)
						renderedLines.push(
							padToWidth(`${theme.fgRaw("separator")}│${RESET} ${line}`),
						);
				} else {
					renderedLines.push(
						padToWidth(
							`${theme.fgRaw("separator")}›${RESET} ${theme.fgRaw("userText")}${BOLD}YOU${RESET}`,
						),
					);
					const lines = wrapText(
						theme.fgRaw("userText") + truncateText(content, this.maxMessageLength) + RESET,
						Math.max(1, contentWidth),
					);
					for (const line of lines)
						renderedLines.push(padToWidth(`  ${line}`));
				}
			}

			// Assistant message — render chunks in seq order (chronological)
			if (turn.assistantMessage) {
				const msg = turn.assistantMessage;
				const chunks = msg.chunks;
				const streaming = !msg.isComplete || hasStreamingChunk(chunks);
				let lastThinkingSection = false;
				renderedLines.push(
					padToWidth(
						`${theme.fgRaw("assistantText")}◆ ${BOLD}LOGICIAN${RESET}`,
					),
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
								`${theme.fgRaw("separator")}${DIM}  ─────────────────${RESET}`,
							),
						);
						lastThinkingSection = false;
					}
					if (answer) {
						renderedLines.push(
							padToWidth(
								`  ${theme.fgRaw("assistantText")}${BOLD}RESPONSE${RESET}`,
							),
						);
						const contentLines = renderMarkdownLines(
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
						const thinkLines = renderThinkingChunk(
							chunk,
							streaming,
							this.thinkingMode,
							this.currentWidth,
						);
						for (const line of thinkLines)
							renderedLines.push(padToWidth(`  ${line}`));
						lastThinkingSection = true;
					} else if (chunk.type === "tool" && chunk.tool) {
						lastThinkingSection = false;
						const toolKey =
							chunk.tool.tool_call_id ?? `${turn.id}:${chunk.seq}`;
						const regionStart = renderedLines.length;
						const toolLines = renderTool(
							this,
							chunk.tool,
							width,
							this.toolsExpanded || this.expandedToolKeys.has(toolKey),
						);
						for (let lineIndex = 0; lineIndex < toolLines.length; lineIndex++) {
							const prefix =
								lineIndex === 0 && toolKey === this.focusedToolKey
									? `${theme.fg("selected", "›")} `
									: "  ";
							renderedLines.push(padToWidth(`${prefix}${toolLines[lineIndex]}`));
						}
						pendingToolRegions.push({
							start: regionStart,
							end: renderedLines.length,
							key: toolKey,
						});
					} else if (chunk.type === "notice" && chunk.notice) {
						const n = chunk.notice;
						if (n.label === "Skills" && n.level === "info") {
							renderedLines.push(
								padToWidth(
									`${theme.fg("active", "✦ NOTICE")} ${BOLD}${theme.fg("toolTitle", "Skills")}${RESET}  ${theme.fg("systemText", n.text)}${RESET}`,
								),
							);
							continue;
						}
						const icon =
							n.level === "error"
								? "✗"
								: n.level === "warn"
									? "⚠"
									: n.level === "success"
										? "✓"
										: "●";
						const color =
							n.level === "error"
								? theme.fgRaw("error")
								: n.level === "warn"
									? theme.fgRaw("warning")
									: theme.fgRaw("systemText");
						renderedLines.push(
							padToWidth(
								`${color}${icon} NOTICE${RESET} ${BOLD}${n.label}${RESET}  ${color}${n.text}${RESET}`,
							),
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
		let visibleStart = 0;
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
				firstCompleteTurn >= 0
					? firstCompleteTurn
					: Math.max(0, this.turns.length - 1);
			const omittedLabel =
				olderCount > 0
					? `${olderCount} older turn(s) not shown`
					: "earlier lines not shown";
			visibleBuffer = [
				padToWidth(`${theme.fgRaw("dim")}… ${omittedLabel}${RESET}`),
				...renderedLines.slice(sliceStart),
			];
			visibleStart = sliceStart - 1;
		}
		this.toolHitRegions = pendingToolRegions
			.map((region) => ({
				...region,
				start: region.start - visibleStart,
				end: region.end - visibleStart,
			}))
			.filter((region) => region.end > 0);

		this._totalHeight = visibleBuffer.length;

		this.cachedLines = visibleBuffer;
		this.resolvePendingScroll();
		return this.renderViewport(visibleBuffer, width);
	}

	// ── Scroll helpers ───────────────────────────────────────────────────────

	private resolvePendingScroll(): void {
		if (!this._pendingScrollBottom) return;
		this._scrollOffset = Math.max(0, this._totalHeight - this._viewportHeight);
		this._atBottom = true;
		this._newOutputBelow = false;
		this._pendingScrollBottom = false;
	}

	private renderViewport(content: string[], width: number): string[] {
		const viewportHeight = this._viewportHeight || 0;
		if (viewportHeight <= 0) return content;
		if (content.length <= viewportHeight) return content;

		const maxScroll = content.length - viewportHeight;
		this._scrollOffset = Math.min(maxScroll, Math.max(0, this._scrollOffset));
		this._atBottom = this._scrollOffset >= maxScroll;
		if (this._atBottom) this._newOutputBelow = false;

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
		if (this._newOutputBelow && !this._atBottom && visible.length > 0) {
			const indicator = `${theme.fg("accent", "↓")} ${theme.fg("muted", "new output below")}`;
			const clipped = clampLineToWidth(indicator, Math.max(1, width - 2));
			visible[visible.length - 1] =
				" ".repeat(Math.max(0, width - 2 - visibleWidth(clipped))) +
				clipped +
				`${barColor}│${reset}`;
		}
		return visible;
	}

	setTurns(turns: Turn[]): void {
		const keepBottomAnchored = this._atBottom;
		const nextRevision = this.revisionFor(turns);
		if (
			!keepBottomAnchored &&
			this.contentRevision !== "" &&
			nextRevision !== this.contentRevision
		) {
			this._newOutputBelow = true;
		}
		this.contentRevision = nextRevision;
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

	private revisionFor(turns: Turn[]): string {
		const turn = turns.at(-1);
		const message = turn?.assistantMessage;
		const chunks = message?.chunks ?? [];
		const chunkRevision = chunks
			.map((chunk) => {
				const tool = chunk.tool;
				return [
					chunk.seq,
					chunk.type,
					chunk.isComplete ? 1 : 0,
					revisionText(chunk.contentText),
					revisionText(tool?.result),
					revisionText(tool?.streamOutput),
					tool?.isComplete ? 1 : 0,
				].join(":");
			})
			.join("|");
		return [
			turns.length,
			turn?.id ?? "",
			message?.isComplete ? 1 : 0,
			chunkRevision,
		].join("/");
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
				if (
					chunk.type === "tool" &&
					chunk.tool?.tool_name === "spawn_agents" &&
					chunk.tool.tool_call_id
				) {
					liveIds.add(chunk.tool.tool_call_id);
				}
			}
		}
		for (const key of this.batchTaskTiming.keys()) {
			if (!liveIds.has(key)) this.batchTaskTiming.delete(key);
		}
	}
	private _viewportHeight: number = 0;
}
