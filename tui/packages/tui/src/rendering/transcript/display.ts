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
import {
	hasStreamingChunk,
	renderInline,
	revisionText,
	stripThinkTags,
} from "./text-utils.ts";
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

interface TurnRenderCache {
	width: number;
	/** Content fingerprint — turns mutate in place while streaming, so turn
	 * object identity alone is not a valid cache-hit signal. */
	turnRevision: string;
	/** This turn's own narrow slice of style/expand state (see
	 * `turnStyleRevision`), not the whole transcript's. */
	styleRevision: string;
	lines: string[];
	/** Relative to `lines[0]`, rebased to absolute offsets by the caller. */
	hitRegions: Array<{ start: number; end: number; key: string }>;
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
	/** Per-turn render cache — a turn's own lines/hitRegions are only rebuilt
	 * when its own content, width, or narrow style slice actually changes.
	 * WeakMap so turns dropped by `setTurns()`'s maxTurns cap become
	 * GC-eligible automatically. */
	private turnCache = new WeakMap<Turn, TurnRenderCache>();

	thinkingMode: ThinkingDisplayStyle;
	toolsExpanded = false;
	private expandedToolKeys = new Set<string>();
	/** Per-task expand state for spawn_agents cards, keyed by `${toolCallId}:task:${index}`. */
	private expandedAgentKeys = new Set<string>();
	/** Per-child-tool expand state within a subagent flow, keyed by `${parentKey}:child:${childToolCallId}`. */
	private expandedChildToolKeys = new Set<string>();
	private toolHitRegions: Array<{ start: number; end: number; key: string }> = [];
	private focusedToolKey: string | null = null;
	/** Populated by renderTool for spawn_agent(s) per-task/per-child-tool hit regions. */
	_taskHitRegions?: Array<{ start: number; end: number; key: string }>;
	maxMessageLength: number;
	private maxTurns: number;
	private maxRenderedLines: number;
	private turns: Turn[] = [];
	private spinnerTick = 0;
	private spinnerTimer: ReturnType<typeof setInterval> | null = null;
	private onAnimationTick: (() => void) | null = null;
	sanitizedToolCache = new WeakMap<ToolExecution, SanitizedToolCache>();
	sanitizationMetrics = { cacheHits: 0, scannedCharacters: 0 };

	constructor(options: TranscriptDisplayOptions = {}) {
		this.thinkingMode = options.thinkingMode ?? "collapsed";
		this.maxMessageLength =
			options.maxMessageLength ?? DEFAULT_TRUNCATION.transcriptMessageMaxChars;
		// A terminal viewport is typically 30-60 rows, so both budgets stay a
		// small multiple of that rather than the old 200-turn/2000-line
		// defaults — those forced every full rebuild (any cache miss, e.g. a
		// theme change or a fresh streaming turn joining the prefix) to
		// markdown-parse and word-wrap far more content than a screen could
		// ever show, while still only ever displaying the last viewportHeight
		// rows. Scrollback is intentionally shallow now, not unbounded.
		this.maxTurns = options.maxTurns ?? 40;
		this.maxRenderedLines = options.maxRenderedLines ?? 400;
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

	/** Whether a specific spawn_agents task card is expanded. */
	isAgentExpanded(toolCallId: string, taskIndex: number): boolean {
		const key = `${toolCallId}:task:${taskIndex}`;
		return this.expandedAgentKeys.has(key);
	}

	/** Toggle expand state for a specific spawn_agents task card. */
	toggleAgentExpanded(toolCallId: string, taskIndex: number): void {
		const key = `${toolCallId}:task:${taskIndex}`;
		if (this.expandedAgentKeys.has(key)) {
			this.expandedAgentKeys.delete(key);
		} else {
			this.expandedAgentKeys.add(key);
		}
		this.invalidate();
	}

	/** Whether a specific child tool call within a subagent flow is expanded. */
	isChildToolExpanded(parentKey: string, childToolCallId: string): boolean {
		const key = `${parentKey}:child:${childToolCallId}`;
		return this.expandedChildToolKeys.has(key);
	}

	/** Toggle expand state for a specific child tool call within a subagent flow. */
	toggleChildToolExpanded(parentKey: string, childToolCallId: string): void {
		const key = `${parentKey}:child:${childToolCallId}`;
		if (this.expandedChildToolKeys.has(key)) {
			this.expandedChildToolKeys.delete(key);
		} else {
			this.expandedChildToolKeys.add(key);
		}
		this.invalidate();
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
		this.toggleHitRegionKey(region.key);
		if (keepBottomAnchored) this._pendingScrollBottom = true;
		return true;
	}

	/**
	 * Toggle whichever expand state a hit-region key addresses — a spawn_agents
	 * per-task card, a per-child-tool call inside a subagent flow, or a plain
	 * tool card. Shared by mouse clicks and keyboard-focused toggling so both
	 * paths dispatch identically instead of the keyboard path assuming every
	 * key belongs to `expandedToolKeys`.
	 */
	private toggleHitRegionKey(key: string): void {
		// toolCallId can itself contain colons (e.g. the `${turn.id}:${chunk.seq}`
		// fallback), so match the trailing `:task:<n>` / `:child:<id>` suffix
		// instead of assuming the id has none.
		const taskMatch = /^(.+):task:(\d+)$/.exec(key);
		if (taskMatch) {
			const [, toolCallId, taskIndexStr] = taskMatch;
			this.toggleAgentExpanded(toolCallId, Number(taskIndexStr));
			return;
		}
		const childMatch = /^(.+):child:(.+)$/.exec(key);
		if (childMatch) {
			const [, parentKey, childToolCallId] = childMatch;
			this.toggleChildToolExpanded(parentKey, childToolCallId);
			return;
		}
		if (this.expandedToolKeys.has(key)) {
			this.expandedToolKeys.delete(key);
		} else {
			this.expandedToolKeys.add(key);
		}
		this.invalidate();
	}

	/** Whether a hit-region key's addressed expand state is currently on. */
	private isHitRegionKeyExpanded(key: string): boolean {
		const taskMatch = /^(.+):task:(\d+)$/.exec(key);
		if (taskMatch) {
			return this.isAgentExpanded(taskMatch[1], Number(taskMatch[2]));
		}
		const childMatch = /^(.+):child:(.+)$/.exec(key);
		if (childMatch) {
			return this.isChildToolExpanded(childMatch[1], childMatch[2]);
		}
		return this.expandedToolKeys.has(key);
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
		this.toggleHitRegionKey(this.focusedToolKey);
		return this.isHitRegionKeyExpanded(this.focusedToolKey);
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
		const padToWidth = (line: string): string => {
			const clipped = clampLineToWidth(line, frameWidth);
			const w = visibleWidth(clipped);
			return clipped + " ".repeat(Math.max(0, frameWidth - w));
		};
		const emptyLine = " ".repeat(frameWidth);

		// Render oldest-to-newest so the newest turns always end up at the
		// bottom of the buffer — the viewport slices from scrollOffset and
		// shows the bottom, meaning new messages are always visible even when
		// maxRenderedLines truncates older content.
		renderedLines.push(padToWidth(emptyLine));

		for (let ti = 0; ti < this.turns.length; ti++) {
			const turn = this.turns[ti];
			if (ti > 0) renderedLines.push(padToWidth(emptyLine));
			const turnStart = renderedLines.length;
			turnStartLines.push(turnStart);

			// Per-turn cache: a turn's own lines/hitRegions are only rebuilt when
			// its own content, width, or narrow style slice actually changes —
			// toggling one tool card or streaming one active turn no longer
			// forces every other turn to re-render.
			const cache = this.renderTurn(turn, width);
			renderedLines.push(...cache.lines);
			for (const region of cache.hitRegions) {
				pendingToolRegions.push({
					start: turnStart + region.start,
					end: turnStart + region.end,
					key: region.key,
				});
			}
		}

		// Bound the render buffer from the *front*. The previous early-break
		// implementation retained old turns and discarded the newest ones while
		// labelling them as "older", which became especially visible after Ctrl+O
		// expanded tool output. Prefer a complete recent-turn boundary; if the
		// newest turn alone exceeds the budget, retain its tail.
		//
		// This must stay active while the newest turn is streaming, not just once
		// it settles — streaming is exactly when the spinner forces a re-render
		// every 150ms, so skipping the cut here meant every one of those frames
		// carried the full, unbounded transcript back through the render/diff
		// pipeline for as long as a turn was in flight.
		const { visibleBuffer, visibleStart } = this.truncateToRenderedLines(
			renderedLines,
			turnStartLines,
			padToWidth,
		);
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
		this.invalidate();
		if (keepBottomAnchored) this._pendingScrollBottom = true;
	}

	/** Fingerprints only the newest turn — drives the "new output below"
	 * scroll indicator, unrelated to per-turn render caching. */
	private revisionFor(turns: Turn[]): string {
		const turn = turns.at(-1);
		return [turns.length, this.turnRevisionFor(turn)].join("/");
	}

	/** Content fingerprint for one turn. Turns/chunks mutate in place while
	 * streaming, so this — not object identity — is what a cache-hit check
	 * must compare. */
	private turnRevisionFor(turn: Turn | undefined): string {
		const message = turn?.assistantMessage;
		const chunks = message?.chunks ?? [];
		const chunkRevision = chunks
			.map((chunk) => {
				const tool = chunk.tool;
				const details = tool?.details;
				const childChunks = Array.isArray(details?.childChunks)
					? details.childChunks
					: [];
				const childRevision = childChunks
					.map((child) =>
						[
							child.seq,
							child.type,
							child.isComplete ? 1 : 0,
							revisionText(child.contentText),
							child.tool?.status ?? "",
							revisionText(child.tool?.resultPreview),
						].join("."),
					)
					.join(",");
				return [
					chunk.seq,
					chunk.type,
					chunk.isComplete ? 1 : 0,
					revisionText(chunk.contentText),
					revisionText(tool?.result),
					revisionText(tool?.streamOutput),
					revisionText(
						typeof details?.streamTranscript === "string"
							? details.streamTranscript
							: undefined,
					),
					childRevision,
					tool?.isComplete ? 1 : 0,
				].join(":");
			})
			.join("|");
		return [turn?.id ?? "", message?.isComplete ? 1 : 0, chunkRevision].join(
			"/",
		);
	}

	/** This turn's narrow slice of style/expand state. Global fields (theme,
	 * thinkingMode, toolsExpanded, maxMessageLength) are folded in
	 * unconditionally so they still invalidate every turn's cache when
	 * changed; expand-key sets and focus are narrowed to just the keys that
	 * belong to this turn, so toggling one tool card no longer invalidates
	 * every other turn the way one global revision string used to. */
	private turnStyleRevision(turn: Turn): string {
		const keysForTurn = (set: Set<string>) =>
			[...set]
				.filter((key) => this.keyBelongsToTurn(key, turn))
				.sort()
				.join(",");
		return [
			theme.name,
			this.thinkingMode,
			this.toolsExpanded ? 1 : 0,
			this.maxMessageLength,
			this.focusedToolKey && this.keyBelongsToTurn(this.focusedToolKey, turn)
				? this.focusedToolKey
				: "",
			keysForTurn(this.expandedToolKeys),
			keysForTurn(this.expandedAgentKeys),
			keysForTurn(this.expandedChildToolKeys),
		].join("|");
	}

	/** Whether an expand/focus key (a plain tool_call_id, or a composite
	 * `${toolCallId}:task:${n}` / `${parentKey}:child:${childId}` key) refers
	 * to a tool call that belongs to this turn. Reuses the same trailing-anchor
	 * regexes as toggleHitRegionKey/isHitRegionKeyExpanded — a tool_call_id can
	 * itself contain colons, so a naive split is not safe here either. */
	private keyBelongsToTurn(key: string, turn: Turn): boolean {
		const taskMatch = /^(.+):task:(\d+)$/.exec(key);
		const childMatch = /^(.+):child:(.+)$/.exec(key);
		const rootId = taskMatch?.[1] ?? childMatch?.[1] ?? key;
		const chunks = turn.assistantMessage?.chunks ?? [];
		return chunks.some(
			(chunk) =>
				chunk.type === "tool" &&
				chunk.tool &&
				(chunk.tool.tool_call_id === rootId ||
					`${turn.id}:${chunk.seq}` === rootId),
		);
	}

	/** Look up or (re)build this turn's cached lines/hitRegions. */
	private renderTurn(turn: Turn, width: number): TurnRenderCache {
		const turnRevision = this.turnRevisionFor(turn);
		const styleRevision = this.turnStyleRevision(turn);
		const cached = this.turnCache.get(turn);
		if (
			cached &&
			cached.width === width &&
			cached.turnRevision === turnRevision &&
			cached.styleRevision === styleRevision
		) {
			return cached;
		}
		const built = this.buildTurnLines(turn, width, turnRevision, styleRevision);
		this.turnCache.set(turn, built);
		return built;
	}

	/** Renders one turn's user/system message, assistant content, thinking
	 * chunks, tool calls, and notices into its own line buffer. This is the
	 * unmodified per-turn rendering logic, just scoped to a single turn
	 * instead of running inline across the whole transcript. */
	private buildTurnLines(
		turn: Turn,
		width: number,
		turnRevision: string,
		styleRevision: string,
	): TurnRenderCache {
		const lines: string[] = [];
		const hitRegions: Array<{ start: number; end: number; key: string }> = [];
		const frameWidth = Math.max(1, width - 2);
		const contentWidth = Math.max(1, frameWidth - 2);
		const padToWidth = (line: string): string => {
			const clipped = clampLineToWidth(line, frameWidth);
			const w = visibleWidth(clipped);
			return clipped + " ".repeat(Math.max(0, frameWidth - w));
		};

		// User or system message
		if (turn.userMessage) {
			const content = turn.userMessage.content;
			if (content.startsWith("[System] ")) {
				lines.push(padToWidth(`${theme.fgRaw("systemText")}◇ SYSTEM${RESET}`));
				const sysLines = renderMarkdownLines(
					content.slice(9),
					contentWidth - 2,
					false,
					theme.fgRaw("systemText") + RESET,
					"",
				);
				for (const line of sysLines)
					lines.push(padToWidth(`${theme.fgRaw("separator")}│${RESET} ${line}`));
			} else {
				lines.push(
					padToWidth(
						`${theme.fgRaw("separator")}›${RESET} ${theme.fgRaw("userLabel")}${BOLD}YOU${RESET}`,
					),
				);
				const colored = theme.fgRaw("userText") + truncateText(content, this.maxMessageLength) + RESET;
				for (const rawLine of colored.split("\n")) {
					for (const line of wrapText(rawLine, Math.max(1, contentWidth)))
						lines.push(padToWidth(`  ${line}`));
				}
			}
		}

		// Assistant message — render chunks in seq order (chronological)
		if (turn.assistantMessage) {
			const msg = turn.assistantMessage;
			const chunks = msg.chunks;
			const streaming = !msg.isComplete || hasStreamingChunk(chunks);
			let lastThinkingSection = false;
			lines.push(
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
					lines.push(
						padToWidth(
							`${theme.fgRaw("separator")}${DIM}  ─────────────────${RESET}`,
						),
					);
					lastThinkingSection = false;
				}
				if (answer) {
					lines.push(
						padToWidth(`  ${theme.fgRaw("responseLabel")}${BOLD}RESPONSE${RESET}`),
					);
					const contentLines = renderMarkdownLines(
						answer,
						contentWidth - 2,
						streaming,
					);
					for (const line of contentLines) lines.push(padToWidth(`  ${line}`));
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
					for (const line of thinkLines) lines.push(padToWidth(`  ${line}`));
					lastThinkingSection = true;
				} else if (chunk.type === "tool" && chunk.tool) {
					lastThinkingSection = false;
					const toolKey = chunk.tool.tool_call_id ?? `${turn.id}:${chunk.seq}`;
					const regionStart = lines.length;
					// Clear per-task hit regions before rendering so this tool's
					// renderer starts from an empty list instead of inheriting
					// (and then wiping) the previous tool's regions.
					this._taskHitRegions = [];
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
						lines.push(padToWidth(`${prefix}${toolLines[lineIndex]}`));
					}
					// Merge per-task/per-child-tool regions from the renderer first
					// so they take precedence.
					for (const region of this._taskHitRegions ?? []) {
						hitRegions.push({
							start: regionStart + region.start,
							end: regionStart + region.end,
							key: region.key,
						});
					}
					// Parent tool region as fallback — except for spawn_agents,
					// where only the per-task rows should be clickable. A
					// whole-block fallback there would catch clicks on the
					// header/blank rows between tasks and toggle every task's
					// detail at once, which reads as agents' streams and
					// responses getting mixed together.
					if (chunk.tool.tool_name !== "spawn_agents") {
						hitRegions.push({
							start: regionStart,
							end: lines.length,
							key: toolKey,
						});
					}
				} else if (chunk.type === "notice" && chunk.notice) {
					const n = chunk.notice;
					if (n.label === "Skills" && n.level === "info") {
						lines.push(
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
					const labelText = n.label.replace(/^\*\*(.*?)\*\*$/, "$1");
					const noticePrefix = `${icon} NOTICE `;
					const bodyIndent = " ".repeat(visibleWidth(noticePrefix));
					const reasonMatch = /^(\[[^\]\n]+\])(?:\s+|$)/.exec(n.text);
					const bodyText = reasonMatch
						? `${theme.fg("accent", reasonMatch[1])} ${color}${n.text.slice(reasonMatch[0].length).trimStart()}${RESET}`
						: `${color}${n.text}${RESET}`;
					lines.push(
						padToWidth(
							`${color}${noticePrefix.trimEnd()}${RESET} ${BOLD}${renderInline(labelText, color)}${RESET}`,
						),
					);
					for (const line of wrapText(
						bodyText,
						Math.max(1, contentWidth - visibleWidth(bodyIndent)),
					)) {
						lines.push(padToWidth(`${bodyIndent}${line}`));
					}
				}
			}
			flushContent();

			// No streaming cursor — messages display as-is
		}

		return { width, turnRevision, styleRevision, lines, hitRegions };
	}

	/** Bound the render buffer from the front once it exceeds maxRenderedLines,
	 * prepending a "N older turn(s) not shown" banner. Runs on every render()
	 * call regardless of streaming state — skipping this while the newest turn
	 * streams would carry the full, unbounded transcript through the render
	 * pipeline for as long as a turn is in flight. */
	private truncateToRenderedLines(
		renderedLines: string[],
		turnStartLines: number[],
		padToWidth: (line: string) => string,
	): { visibleBuffer: string[]; visibleStart: number } {
		if (renderedLines.length <= this.maxRenderedLines) {
			return { visibleBuffer: renderedLines, visibleStart: 0 };
		}
		const desiredStart = renderedLines.length - (this.maxRenderedLines - 1);
		const firstCompleteTurn = turnStartLines.findIndex(
			(line) => line >= desiredStart,
		);
		const sliceStart =
			firstCompleteTurn >= 0 ? turnStartLines[firstCompleteTurn] : desiredStart;
		const olderCount =
			firstCompleteTurn >= 0
				? firstCompleteTurn
				: Math.max(0, this.turns.length - 1);
		const omittedLabel =
			olderCount > 0
				? `${olderCount} older turn(s) not shown`
				: "earlier lines not shown";
		return {
			visibleBuffer: [
				padToWidth(`${theme.fgRaw("dim")}… ${omittedLabel}${RESET}`),
				...renderedLines.slice(sliceStart),
			],
			visibleStart: sliceStart - 1,
		};
	}

	private _viewportHeight: number = 0;
}
