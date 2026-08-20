// ── Transcript display component ────────────────────────────────────────────────
// Renders the full conversation history with streaming support and markdown.
// Chunks are interleaved in chronological order: thinking → content → tool → ...

import { DEFAULT_TRUNCATION } from "@logician/agent-core";
import type {
	ThinkingDisplayStyle,
	ToolExecution,
	Turn,
} from "@logician/agent-core/sessions";
import {
	BOLD,
	type Component,
	clampLineToWidth,
	DIM,
	RESET,
	visibleWidth,
} from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";
import type { ScrollView } from "../scroll-view.ts";
import { wrapText } from "./layout.ts";
import { truncateText } from "./render/content.ts";
import { renderMarkdownLines } from "./render/markdown-table.ts";
import { renderThinkingChunk } from "./render/thinking.ts";
import {
	getSanitizationMetrics,
	type RenderCtx,
	renderTool,
	type SanitizedToolCache,
} from "./render/tool.ts";
import {
	hasStreamingChunk,
	renderInline,
	renderMarkdownLine,
	revisionText,
	stripThinkTags,
} from "./text-utils.ts";

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

export class TranscriptDisplay implements Component, RenderCtx {
	private cachedWidth: number = 0;
	private cachedLines: string[] | null = null;
	/** Pre-truncation assembly from the last render(): the full spliced line
	 * buffer, each turn's start offset within it, and the per-turn revisions
	 * used to build it. Lets render() reuse the unchanged prefix of turns
	 * (the common case — only the newest turn is streaming) instead of
	 * re-fingerprinting and re-splicing all `maxTurns` turns every frame. */
	private assembledLines: string[] = [];
	private assembledTurnStarts: number[] = [];
	/** Full revision per turn: `${turnRevision}::${styleRevision}`. */
	private assembledTurnRevisions: string[] = [];
	/** Per-turn contentRevision (O(1) number) for fast prefix-scan comparison.
	 * When this matches the turn's current value AND the turn object identity
	 * matches, we can skip the expensive turnRevisionFor() call — but we still
	 * need to compare the stored styleRevision vs current styleRevision.
	 * Stored alongside assembledTurnRevisions for O(1) content checks. */
	private assembledTurnContentRevisions: number[] = [];
	private assembledTurns: Turn[] = [];
	private assembledWidth: number = -1;
	// Not private: read by the extracted render-*.ts functions through the
	// RenderCtx interface, which TranscriptDisplay satisfies structurally.
	currentWidth: number = 80;
	/** Owning ScrollView, set once by app/tui.ts after construction. Scroll
	 * position/clipping/scrollbar all live there now — this component only
	 * renders full unbounded content and answers hit-testing in content-
	 * relative coordinates. */
	private scrollView: ScrollView | undefined;
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
	private toolHitRegions: Array<{ start: number; end: number; key: string }> =
		[];
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
		// Unbounded by default: painting is already clipped to the viewport
		// (paintBox only walks firstRow..lastRow of the visible rect) and
		// render() caches its output and returns the identical array reference
		// when nothing changed, so keeping the full history around doesn't add
		// per-frame cost — it just lets the user scroll back through the whole
		// session instead of hitting a truncation banner. Callers that want a
		// hard cap (e.g. to bound memory on very long-running sessions) can
		// still pass maxTurns/maxRenderedLines explicitly.
		this.maxTurns = options.maxTurns ?? Number.POSITIVE_INFINITY;
		this.maxRenderedLines =
			options.maxRenderedLines ?? Number.POSITIVE_INFINITY;
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
			// Active text/thinking streams have no transcript spinner. Avoid
			// invalidating and scheduling a frame until a running tool actually
			// needs the animation; token events already request their own frames.
			if (!this.hasRunningTool()) return;
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

	/** `row` is content-relative (i.e. already translated by the caller using
	 * the owning ScrollView's scrollTop — see getComponentBoxAt in
	 * rendering/layout.ts), not screen-relative. */
	handleMouse(_column: number, row: number): boolean {
		const region = this.toolHitRegions.find(
			candidate => row >= candidate.start && row < candidate.end,
		);
		if (!region) return false;
		this.toggleHitRegionKey(region.key);
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
			region => region.key === this.focusedToolKey,
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
		if (this.scrollView) {
			const viewportHeight = Math.max(1, this.scrollView.viewportHeight);
			const scrollTop = this.scrollView.scrollTop;
			if (region.start < scrollTop) {
				this.scrollView.scrollTo(region.start);
			} else if (region.end > scrollTop + viewportHeight) {
				this.scrollView.scrollTo(Math.max(0, region.end - viewportHeight));
			}
		}
		this.invalidate();
		return { index: nextIndex + 1, total: this.toolHitRegions.length };
	}

	/** Called once by app/tui.ts after wrapping this component in a ScrollView. */
	setScrollView(view: ScrollView): void {
		this.scrollView = view;
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

	// ── Scroll-adjacent state ────────────────────────────────────────────────
	// Actual scroll position/clipping/scrollbar live on the wrapping
	// ScrollView (see setScrollView). This is just the "new output arrived
	// while scrolled away from the end" signal, driven by scrollView's own
	// isFollowingEnd — app/tui.ts surfaces it as a bottom-anchored overlay.

	hasNewOutputBelow(): boolean {
		return this._newOutputBelow && this.scrollView?.isFollowingEnd === false;
	}

	clearNewOutputIndicator(): void {
		this._newOutputBelow = false;
	}

	render(width: number): string[] {
		// Fast path: content unchanged → reuse cached body
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.currentWidth = width;
		this.cachedWidth = width;

		const frameWidth = Math.max(1, width - 2);
		const padToWidth = (line: string): string => {
			const clipped = clampLineToWidth(line, frameWidth);
			const w = visibleWidth(clipped);
			return clipped + " ".repeat(Math.max(0, frameWidth - w));
		};
		const emptyLine = " ".repeat(frameWidth);

		// Reuse the assembled prefix from the last frame up to the first turn
		// whose identity or revision actually changed. During normal streaming
		// only the newest turn's revision moves, so this turns an O(all turns)
		// re-fingerprint-and-resplice into O(1) for every animation tick.
		//
		// O(1) content check via per-turn contentRevision: the Transcript bumps
		// this counter on every chunk mutation. We compare it against the stored
		// contentRevision from the previous frame — a match means the Transcript
		// has not mutated the turn's content since we last rendered, so we can
		// skip the expensive turnRevisionFor() call entirely. We still compare
		// the style revision (spinner state, expand toggles) because those can
		// change without content mutations.
		//
		// When width changes we still need a full rebuild because
		// assembledLines carries the old width's padding. But the prefix scan
		// now finds the first dirty turn in O(1) per turn, and renderTurn()
		// below rebuilds only the dirty turns at the new width.
		const turnRevisions: string[] = new Array(this.turns.length);
		let firstDirty = 0;
		if (this.assembledTurns.length > 0 && this.assembledWidth === width) {
			const minLen = Math.min(this.turns.length, this.assembledTurns.length);
			while (firstDirty < minLen) {
				const turn = this.turns[firstDirty];
				const storedContentRevision =
					this.assembledTurnContentRevisions[firstDirty];
				// Fast path: content unchanged AND turn identity unchanged
				if (
					storedContentRevision !== undefined &&
					storedContentRevision === turn.contentRevision &&
					turn === this.assembledTurns[firstDirty]
				) {
					// Content is the same — only style might have changed.
					// Extract the stored style revision from the full revision
					// string (format: "contentRevision::styleRevision").
					const storedFullRev = this.assembledTurnRevisions[firstDirty];
					const storedStyleRev = storedFullRev.slice(
						storedFullRev.indexOf("::") + 2,
					);
					const currentStyleRev = this.turnStyleRevision(turn);
					if (currentStyleRev === storedStyleRev) {
						// Nothing changed — reuse stored revision
						turnRevisions[firstDirty] = storedFullRev;
						firstDirty++;
						continue;
					}
					// Style changed — fall through to recompute full revision
				}
				// Content or style changed — compute full revision
				const revision = `${this.turnRevisionFor(turn)}::${this.turnStyleRevision(turn)}`;
				if (
					turn !== this.assembledTurns[firstDirty] ||
					revision !== this.assembledTurnRevisions[firstDirty]
				) {
					break;
				}
				turnRevisions[firstDirty] = revision;
				firstDirty++;
			}
		}

		const allTurnsUnchanged =
			firstDirty === this.turns.length &&
			this.turns.length === this.assembledTurns.length;

		// Fastest path: nothing changed at all — return the exact same array
		// reference. The layout engine sees the same array and skips its own
		// work for this component, and the diff engine downstream sees the
		// same raw lines as last frame and skips the per-row diff too.
		if (allTurnsUnchanged && this.cachedLines !== null) {
			return this.cachedLines;
		}

		const renderedLines: string[] = allTurnsUnchanged
			? this.assembledLines.slice()
			: firstDirty > 0
				? this.assembledLines.slice(0, this.assembledTurnStarts[firstDirty])
				: [];
		const turnStartLines: number[] = this.assembledTurnStarts.slice(
			0,
			firstDirty,
		);
		const pendingToolRegions: Array<{
			start: number;
			end: number;
			key: string;
		}> = [];
		// Hit regions for the reused prefix must be recovered from those turns'
		// own per-turn cache (still valid — width/revision unchanged) rather
		// than dropped, since toolHitRegions is rebuilt fresh every render().
		for (let ti = 0; ti < firstDirty; ti++) {
			const cache = this.renderTurn(this.turns[ti], width);
			const turnStart = turnStartLines[ti];
			for (const region of cache.hitRegions) {
				pendingToolRegions.push({
					start: turnStart + region.start,
					end: turnStart + region.end,
					key: region.key,
				});
			}
		}

		if (firstDirty === 0) {
			// Render oldest-to-newest so the newest turns always end up at the
			// bottom of the buffer — the viewport slices from scrollOffset and
			// shows the bottom, meaning new messages are always visible even when
			// maxRenderedLines truncates older content.
			renderedLines.push(padToWidth(emptyLine));
		}

		for (let ti = firstDirty; ti < this.turns.length; ti++) {
			const turn = this.turns[ti];
			// The separator above turn `firstDirty` is already included in the
			// sliced prefix when firstDirty > 0 (turnStartLines records the
			// offset *after* it) — only push one for turns after that, and for
			// turn 0 when there's no reused prefix at all.
			const needsSeparator = ti > 0 && !(ti === firstDirty && firstDirty > 0);
			if (needsSeparator) renderedLines.push(padToWidth(emptyLine));
			const turnStart = renderedLines.length;
			turnStartLines.push(turnStart);
			turnRevisions[ti] =
				`${this.turnRevisionFor(turn)}::${this.turnStyleRevision(turn)}`;

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

		this.assembledLines = renderedLines;
		this.assembledTurnStarts = turnStartLines;
		this.assembledTurnRevisions = turnRevisions;
		// Store per-turn content revisions for O(1) prefix-scan comparison on
		// the next frame. A matching contentRevision + turn identity means we
		// can skip the expensive turnRevisionFor() call entirely.
		this.assembledTurnContentRevisions = new Array(this.turns.length);
		for (let ti = 0; ti < this.turns.length; ti++) {
			this.assembledTurnContentRevisions[ti] =
				this.turns[ti].contentRevision ?? -1;
		}
		this.assembledTurns = this.turns.slice();
		this.assembledWidth = width;

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
			.map(region => ({
				...region,
				start: region.start - visibleStart,
				end: region.end - visibleStart,
			}))
			.filter(region => region.end > 0);

		this.cachedLines = visibleBuffer;
		return visibleBuffer;
	}

	setTurns(turns: Turn[]): void {
		const wasFollowingEnd = this.scrollView?.isFollowingEnd ?? true;
		const nextRevision = this.revisionFor(turns);
		if (
			!wasFollowingEnd &&
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
			.map(chunk => {
				const tool = chunk.tool;
				const details = tool?.details;
				const childChunks = Array.isArray(details?.childChunks)
					? details.childChunks
					: [];
				const childRevision = childChunks
					.map(child =>
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
	 * every other turn the way one global revision string used to.
	 *
	 * spinnerTick is folded in only while the turn has a running tool: a
	 * running tool's glyph (renderTool's ctx.spinnerFrame()) is read purely
	 * from spinner state, not from any turn content field, so without this
	 * the per-turn cache never saw the tick change and the glyph only ever
	 * advanced when something else (e.g. clicking the tool) happened to
	 * touch a field that IS in this revision. Gating on a running tool avoids
	 * paying a spinner-driven rebuild for incomplete text-only turns. */
	private turnStyleRevision(turn: Turn): string {
		const keysForTurn = (set: Set<string>) =>
			[...set]
				.filter(key => this.keyBelongsToTurn(key, turn))
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
			this.turnHasRunningTool(turn) ? this.spinnerTick : "",
		].join("|");
	}

	private hasRunningTool(): boolean {
		return this.turns.some(turn => this.turnHasRunningTool(turn));
	}

	private turnHasRunningTool(turn: Turn): boolean {
		return (turn.assistantMessage?.chunks ?? []).some(
			chunk => chunk.type === "tool" && chunk.tool?.isComplete !== true,
		);
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
			chunk =>
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
					lines.push(
						padToWidth(`${theme.fgRaw("separator")}│${RESET} ${line}`),
					);
			} else {
				lines.push(
					padToWidth(
						`${theme.fgRaw("separator")}›${RESET} ${theme.fgRaw("userLabel")}${BOLD}YOU${RESET}`,
					),
				);
				const colored =
					theme.fgRaw("userText") +
					truncateText(content, this.maxMessageLength) +
					RESET;
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
				if (!contentBuffer.trim()) {
					contentBuffer = "";
					return;
				}
				const answer = stripThinkTags(contentBuffer).trim();
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
						padToWidth(
							`  ${theme.fgRaw("responseLabel")}${BOLD}RESPONSE${RESET}`,
						),
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
					const levelColor =
						n.level === "error"
							? theme.fgRaw("error")
							: n.level === "warn"
								? theme.fgRaw("warning")
								: theme.fgRaw("systemText");
					const labelColor = theme.fgRaw("toolTitle");
					const labelText = n.label.replace(/^\*\*(.*?)\*\*$/, "$1");
					const noticePrefix = `${icon} NOTICE `;
					const bodyIndent = " ".repeat(visibleWidth(noticePrefix));
					const reasonMatch = /^(\[[^\]\n]+\])(?:\s+|$)/.exec(n.text);
					const bodyColor = theme.fgRaw("muted");
					// Split body at first newline (if any) so reason-only first line
					// gets accent+body coloring while continuation lines stay muted.
					const bodyLines = n.text.split("\n");
					const firstLineBody = bodyLines[0] ?? "";
					const continuationLines = bodyLines.slice(1);

					// Apply markdown rendering to the body with muted base color.
					// renderMarkdownLine adds inline highlighting for **bold**, `code`, etc.
					let renderedFirst = renderMarkdownLine(firstLineBody, bodyColor);

					if (reasonMatch) {
						const reason = reasonMatch[1];
						const bodyAfterReason = firstLineBody
							.slice(reasonMatch[0].length)
							.trimStart();
						renderedFirst = `${theme.fg("accent", reason)} ${renderMarkdownLine(bodyAfterReason, bodyColor)}`;
					}

					lines.push(
						padToWidth(
							`${levelColor}${noticePrefix.trimEnd()}${RESET} ${BOLD}${renderInline(labelText, labelColor)}${RESET}`,
						),
					);

					// Collect all body lines (first line body + continuations) and
					// wrap them individually so each line gets proper markdown + color.
					const maxBodyWidth = Math.max(
						1,
						contentWidth - visibleWidth(bodyIndent),
					);
					const allBodyRendered: string[] = [renderedFirst];
					for (let ci = 0; ci < continuationLines.length; ci++) {
						allBodyRendered.push(
							renderMarkdownLine(continuationLines[ci], bodyColor),
						);
					}
					for (const rendered of allBodyRendered) {
						const wrapped = wrapText(rendered, maxBodyWidth);
						for (const wl of wrapped) {
							lines.push(padToWidth(`${bodyIndent}${wl}`));
						}
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
			line => line >= desiredStart,
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
}
