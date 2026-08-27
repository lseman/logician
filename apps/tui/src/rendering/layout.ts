import {
	type Component,
	CURSOR_MARKER,
	clampLineToWidth,
	compositeTuiLine,
	visibleWidth,
} from "../terminal/primitives.ts";
import { allocateFlexSizes, visibleFlexEntries } from "./flex.ts";
import { getLayoutNode } from "./layout-node.ts";
import type { ScrollView } from "./scroll-view.ts";

export interface LayoutRect {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface LayoutBox {
	component: Component;
	rect: LayoutRect;
	clip: LayoutRect;
	children: LayoutBox[];
	parent?: LayoutBox;
	lines?: readonly string[];
	lineOffset?: number;
	scrollView?: ScrollView;
	scrollContentLines?: readonly string[];
	layer: number;
}

// ── Layout helpers ───────────────────────────────────────────────────────────

function allocateBox(
	component: Component,
	x: number,
	y: number,
	width: number,
	height: number,
	clipX: number,
	clipY: number,
	clipWidth: number,
	clipHeight: number,
	layer: number,
	lines?: readonly string[],
	lineOffset?: number,
): LayoutBox {
	return {
		component,
		rect: { x, y, width, height },
		clip: { x: clipX, y: clipY, width: clipWidth, height: clipHeight },
		children: [],
		lines,
		lineOffset,
		parent: undefined,
		layer,
	};
}

export interface LayoutFrame {
	root: LayoutBox;
	width: number;
	height: number;
	lines: string[];
	primaryScrollView?: ScrollView;
}

interface ScrollbarGeometry {
	column: number;
	trackTop: number;
	trackHeight: number;
	thumbTop: number;
	thumbHeight: number;
	maxScrollTop: number;
}

interface LayoutContext {
	viewport: { width: number; height: number };
	/** Render cache: Component → width → cached lines. Cleared each frame so
	 * old frame data does not accumulate and cause GC pressure. Matches pi's
	 * simpler design (no LRU, no WeakMap, no string-key hashing). */
	renderCache: Map<Component, Map<number, string[]>>;
	/** Cached visibleWidth results — avoids re-parsing ANSI escape sequences
	 * on every line every frame. The layout engine calls visibleWidth on
	 * nearly every rendered line during layout, so this cache eliminates
	 * significant per-frame allocation and computation. */
	widthCache: Map<string, number>;
	requestRender: () => void;
	primaryScrollView: ScrollView | undefined;
}

function intersect(a: LayoutRect, b: LayoutRect): LayoutRect {
	const x = Math.max(a.x, b.x);
	const y = Math.max(a.y, b.y);
	const right = Math.min(a.x + a.width, b.x + b.width);
	const bottom = Math.min(a.y + a.height, b.y + b.height);
	return {
		x,
		y,
		width: Math.max(0, right - x),
		height: Math.max(0, bottom - y),
	};
}

/* Cached render with nested Map cache matching pi's design.
 * Component identity → width → cached lines. The cache is cleared each frame
 * so old frame data does not accumulate and cause GC pressure. */
function renderCached(
	context: LayoutContext,
	component: Component,
	width: number,
): string[] {
	const safeWidth = Math.max(1, Math.floor(width));
	let widths = context.renderCache.get(component);
	if (!widths) {
		widths = new Map<number, string[]>();
		context.renderCache.set(component, widths);
	}
	let lines = widths.get(safeWidth);
	if (!lines) {
		lines = component.render(safeWidth);
		widths.set(safeWidth, lines);
	}
	return lines;
}

/** Cached visibleWidth — avoids re-parsing ANSI escape sequences on every line.
 * Uses a string key "line|width" for the cache, but since width is constant
 * within a layout frame, we just key by the line text itself. */
function visibleWidthCached(context: LayoutContext, line: string): number {
	let cached = context.widthCache.get(line);
	if (cached === undefined) {
		cached = visibleWidth(line);
		context.widthCache.set(line, cached);
	}
	return cached;
}

function measureHeight(
	context: LayoutContext,
	component: Component,
	width: number,
): number {
	return renderCached(context, component, width).length;
}

function measureWidth(
	context: LayoutContext,
	component: Component,
	width: number,
): number {
	return renderCached(context, component, width).reduce(
		(max, line) => Math.max(max, visibleWidthCached(context, line)),
		0,
	);
}

function withParent(box: LayoutBox, parent: LayoutBox): LayoutBox {
	box.parent = parent;
	return box;
}

function translateBox(box: LayoutBox, deltaY: number): void {
	box.rect.y += deltaY;
	for (const child of box.children) translateBox(child, deltaY);
}

function updateClips(box: LayoutBox, parentClip: LayoutRect): void {
	box.clip = intersect(parentClip, box.rect);
	for (const child of box.children) updateClips(child, box.clip);
}

/** Cross-frame leaf cache: skips recomputing a leaf's height/offset/clip math
 * when its own render() returned the exact same array reference as last
 * frame (i.e. the leaf itself decided nothing changed — every leaf component
 * in this codebase already implements that fast path internally, e.g.
 * InputBar/TranscriptDisplay's `cachedLines`) AND its layout inputs
 * (position/size/clip) are unchanged. Keyed by component identity via
 * WeakMap, so entries for dropped components become GC-eligible on their
 * own — no manual invalidation, no shared mutable slot to get out of sync
 * with reality (see the removed frame-level cache in renderLayoutFrame,
 * which used one global slot compared against itself and therefore always
 * hit after the first frame).
 *
 * A cached `box` is held across frames — it must not be mutated by the
 * allocateBox call (allocateBox always returns a fresh object, never a
 * recycled one). */
interface LeafLayoutCache {
	x: number;
	y: number;
	width: number;
	height: number | undefined;
	clip: LayoutRect;
	lines: readonly string[];
	box: LayoutBox;
}
const leafCache = new WeakMap<Component, LeafLayoutCache>();

function sameClip(a: LayoutRect, b: LayoutRect): boolean {
	return (
		a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height
	);
}

function layoutComponent(
	context: LayoutContext,
	component: Component,
	x: number,
	y: number,
	width: number,
	height: number | undefined,
	clip: LayoutRect,
): LayoutBox {
	const safeWidth = Math.max(1, Math.floor(width));
	const node = getLayoutNode(component);
	if (!node) {
		const lines = renderCached(context, component, safeWidth);

		const cached = leafCache.get(component);
		if (
			cached &&
			cached.lines === lines &&
			cached.x === x &&
			cached.y === y &&
			cached.width === safeWidth &&
			cached.height === height &&
			sameClip(cached.clip, clip)
		) {
			return cached.box;
		}

		const allocatedHeight =
			height === undefined ? lines.length : Math.max(0, Math.floor(height));
		let lineOffset = 0;
		if (lines.length > allocatedHeight && allocatedHeight > 0) {
			const cursorLine = lines.findIndex(line => line.includes(CURSOR_MARKER));
			if (cursorLine >= allocatedHeight)
				lineOffset = cursorLine - allocatedHeight + 1;
		}
		const clipRect = intersect(clip, {
			x,
			y,
			width: safeWidth,
			height: allocatedHeight,
		});
		const box = allocateBox(
			component,
			x,
			y,
			safeWidth,
			allocatedHeight,
			clipRect.x,
			clipRect.y,
			clipRect.width,
			clipRect.height,
			0,
			lines,
			lineOffset,
		);
		leafCache.set(component, {
			x,
			y,
			width: safeWidth,
			height,
			clip,
			lines,
			box,
		});
		return box;
	}

	if (node.type === "scroll") {
		const previousScrollTop = node.state.scrollTop;
		const contentWidth = node.state.getContentWidth(safeWidth);
		const childBox = layoutComponent(
			context,
			node.component,
			x,
			y - previousScrollTop,
			contentWidth,
			undefined,
			clip,
		);
		const contentHeight = childBox.rect.height;
		const viewportHeight =
			height === undefined ? contentHeight : Math.max(0, Math.floor(height));
		node.state.updateLayout(
			contentHeight,
			viewportHeight,
			context.requestRender,
		);
		translateBox(childBox, previousScrollTop - node.state.scrollTop);
		const scrollView = node.state as ScrollView;
		if (node.state.primary || !context.primaryScrollView)
			context.primaryScrollView = scrollView;
		const rect = { x, y, width: safeWidth, height: viewportHeight };
		const childClip = intersect(clip, rect);
		const box = allocateBox(
			component,
			x,
			y,
			safeWidth,
			viewportHeight,
			childClip.x,
			childClip.y,
			childClip.width,
			childClip.height,
			0,
		);
		box.children.push(childBox);
		box.scrollView = scrollView;
		box.scrollContentLines = renderCached(
			context,
			node.component,
			contentWidth,
		);
		childBox.parent = box;
		updateClips(childBox, childClip);
		return box;
	}

	const entries = visibleFlexEntries(node.entries, context.viewport);
	const gapTotal = Math.max(0, entries.length - 1) * node.gap;
	if (node.type === "vstack") {
		const intrinsicHeights = entries.map(entry =>
			typeof entry.basis === "number"
				? entry.basis
				: measureHeight(context, entry.component, safeWidth),
		);
		const sizes = allocateFlexSizes(
			entries,
			intrinsicHeights,
			height,
			node.gap,
		);
		const naturalHeight = sizes.reduce((sum, size) => sum + size, 0) + gapTotal;
		const allocatedHeight =
			height === undefined ? naturalHeight : Math.max(0, Math.floor(height));
		const rect = { x, y, width: safeWidth, height: allocatedHeight };
		const clipRect = intersect(clip, rect);
		const box = allocateBox(
			component,
			x,
			y,
			safeWidth,
			allocatedHeight,
			clipRect.x,
			clipRect.y,
			clipRect.width,
			clipRect.height,
			0,
		);
		let childY = y;
		for (let index = 0; index < entries.length; index++) {
			box.children.push(
				withParent(
					layoutComponent(
						context,
						entries[index]?.component,
						x,
						childY,
						safeWidth,
						sizes[index]!,
						box.clip,
					),
					box,
				),
			);
			childY += sizes[index]! + node.gap;
		}
		return box;
	}

	const intrinsicWidths = entries.map(entry =>
		typeof entry.basis === "number"
			? entry.basis
			: measureWidth(context, entry.component, safeWidth),
	);
	const widths = allocateFlexSizes(
		entries,
		intrinsicWidths,
		safeWidth,
		node.gap,
	);
	const intrinsicHeights = entries.map((entry, index) =>
		measureHeight(context, entry.component, Math.max(1, widths[index]!)),
	);
	const allocatedHeight =
		height === undefined
			? intrinsicHeights.reduce(
					(max, childHeight) => Math.max(max, childHeight),
					0,
				)
			: Math.max(0, height);
	const rect = { x, y, width: safeWidth, height: allocatedHeight };
	const clipRect = intersect(clip, rect);
	const box = allocateBox(
		component,
		x,
		y,
		safeWidth,
		allocatedHeight,
		clipRect.x,
		clipRect.y,
		clipRect.width,
		clipRect.height,
		0,
	);
	let childX = x;
	for (let index = 0; index < entries.length; index++) {
		const naturalChildHeight = intrinsicHeights[index]!;
		const childHeight =
			node.align === "stretch"
				? allocatedHeight
				: Math.min(allocatedHeight, naturalChildHeight);
		let childY = y;
		if (node.align === "center")
			childY += Math.floor((allocatedHeight - childHeight) / 2);
		else if (node.align === "end") childY += allocatedHeight - childHeight;
		const childWidth = widths[index]!;
		if (childWidth === 0) {
			box.children.push(
				allocateBox(
					entries[index]?.component,
					childX,
					childY,
					0,
					childHeight,
					childX,
					childY,
					0,
					0,
					0,
				),
			);
		} else {
			box.children.push(
				withParent(
					layoutComponent(
						context,
						entries[index]?.component,
						childX,
						childY,
						childWidth,
						childHeight,
						box.clip,
					),
					box,
				),
			);
		}
		childX += childWidth + node.gap;
	}
	return box;
}

// Our scrollbar always repaints the whole trailing column, so painting it is
// a plain per-row cell replacement rather than pi's grapheme-aware in-line
// styling — nothing else ever shares that column.
function paintScrollbarCell(
	line: string,
	column: number,
	glyph: string,
	isThumb: boolean,
	style: (glyph: string, isThumb: boolean) => string,
): string {
	const before = clampLineToWidth(line, column);
	const beforePad = " ".repeat(Math.max(0, column - visibleWidth(before)));
	return `${before}${beforePad}${style(glyph, isThumb)}`;
}

function getScrollbarGeometry(box: LayoutBox): ScrollbarGeometry | undefined {
	if (
		!box.scrollView?.isScrollbarVisible ||
		box.rect.width <= 0 ||
		box.rect.height <= 0
	)
		return undefined;

	const contentHeight =
		box.children[0]?.rect.height ?? box.scrollContentLines?.length ?? 0;
	const trackHeight = box.rect.height;

	const minThumbHeight = Math.min(2, trackHeight);
	const thumbHeight = Math.max(
		minThumbHeight,
		Math.min(
			trackHeight,
			Math.round((trackHeight * trackHeight) / contentHeight),
		),
	);
	const maxScrollTop = Math.max(0, contentHeight - trackHeight);
	const maxThumbTop = trackHeight - thumbHeight;
	const thumbOffset =
		maxScrollTop === 0
			? 0
			: Math.round((box.scrollView.scrollTop / maxScrollTop) * maxThumbTop);
	const column = box.rect.x + box.rect.width - 1;
	if (column < box.clip.x || column >= box.clip.x + box.clip.width)
		return undefined;

	return {
		column,
		trackTop: box.rect.y,
		trackHeight,
		thumbTop: box.rect.y + thumbOffset,
		thumbHeight,
		maxScrollTop,
	};
}

function paintScrollbar(
	box: LayoutBox,
	screen: string[],
	_totalWidth: number,
): void {
	const geometry = getScrollbarGeometry(box);
	if (!geometry || !box.scrollView) return;

	for (let offset = 0; offset < geometry.trackHeight; offset++) {
		const row = geometry.trackTop + offset;
		if (
			row < box.clip.y ||
			row >= box.clip.y + box.clip.height ||
			row < 0 ||
			row >= screen.length
		)
			continue;
		const isThumb =
			row >= geometry.thumbTop &&
			row < geometry.thumbTop + geometry.thumbHeight;
		const glyph = isThumb ? "█" : "│";
		screen[row] = paintScrollbarCell(
			screen[row] ?? "",
			geometry.column,
			glyph,
			isThumb,
			box.scrollView.scrollbarStyle,
		);
	}
}

function paintBox(box: LayoutBox, screen: string[], totalWidth: number): void {
	if (box.lines) {
		const offset = box.lineOffset ?? 0;
		const firstRow = Math.max(box.rect.y, box.clip.y, 0);
		const lastRow = Math.min(
			box.rect.y + box.rect.height,
			box.clip.y + box.clip.height,
			screen.length,
		);
		for (let row = firstRow; row < lastRow; row++) {
			const sourceLine = box.lines[offset + row - box.rect.y];
			if (sourceLine === undefined) continue;
			screen[row] = compositeTuiLine(
				screen[row] ?? "",
				sourceLine,
				box.rect.x,
				box.rect.width,
				totalWidth,
			);
		}
	}
	for (const child of box.children) paintBox(child, screen, totalWidth);
	paintScrollbar(box, screen, totalWidth);
}

export function renderLayoutFrame(
	root: Component,
	width: number,
	height: number,
	requestRender: () => void,
): LayoutFrame {
	const safeWidth = Math.max(1, Math.floor(width));
	const safeHeight = Math.max(1, Math.floor(height));

	const context: LayoutContext = {
		viewport: { width: safeWidth, height: safeHeight },
		renderCache: new Map<Component, Map<number, string[]>>(),
		widthCache: new Map(),
		requestRender,
		primaryScrollView: undefined,
	};
	const rootBox = layoutComponent(context, root, 0, 0, safeWidth, safeHeight, {
		x: 0,
		y: 0,
		width: safeWidth,
		height: safeHeight,
	});
	const lines = Array.from({ length: safeHeight }, () => "");
	paintBox(rootBox, lines, safeWidth);

	const frame: LayoutFrame = {
		root: rootBox,
		width: safeWidth,
		height: safeHeight,
		lines,
		...(context.primaryScrollView === undefined
			? {}
			: { primaryScrollView: context.primaryScrollView }),
	};

	return frame;
}

function containsPoint(rect: LayoutRect, x: number, y: number): boolean {
	return (
		x >= rect.x &&
		x < rect.x + rect.width &&
		y >= rect.y &&
		y < rect.y + rect.height
	);
}

function getScrollViewBox(
	frame: LayoutFrame,
	scrollView: ScrollView,
): LayoutBox | undefined {
	const visit = (box: LayoutBox): LayoutBox | undefined => {
		if (box.scrollView === scrollView) return box;
		for (const child of box.children) {
			const match = visit(child);
			if (match) return match;
		}
		return undefined;
	};
	return visit(frame.root);
}

/**
 * Find the leaf box for a specific component instance, so a click's absolute
 * terminal (column, row) can be translated to that component's own
 * content-relative coordinates: `row - box.rect.y`, `column - box.rect.x`.
 * Returns undefined if the component isn't part of the committed frame, or
 * the point falls outside its clip (e.g. scrolled out of view).
 */
export function getComponentBoxAt(
	frame: LayoutFrame,
	component: Component,
	x: number,
	y: number,
): LayoutBox | undefined {
	const visit = (box: LayoutBox): LayoutBox | undefined => {
		if (box.component === component) {
			return containsPoint(box.clip, x, y) && containsPoint(box.rect, x, y)
				? box
				: undefined;
		}
		for (const child of box.children) {
			const match = visit(child);
			if (match) return match;
		}
		return undefined;
	};
	return visit(frame.root);
}

export function getScrollViewsAt(
	frame: LayoutFrame,
	x: number,
	y: number,
): ScrollView[] {
	const result: Array<{ scrollView: ScrollView; depth: number }> = [];
	const visit = (box: LayoutBox, depth: number): void => {
		if (!containsPoint(box.clip, x, y)) return;
		if (box.scrollView && containsPoint(box.rect, x, y))
			result.push({ scrollView: box.scrollView, depth });
		for (const child of box.children) visit(child, depth + 1);
	};
	visit(frame.root, 0);
	result.sort((a, b) => b.depth - a.depth);
	return result.map(entry => entry.scrollView);
}
