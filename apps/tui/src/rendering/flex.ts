// ── Flex — single flexbox-style stack ──────────────────────────────────────
// One implementation covers both directions: `direction: "column"` stacks
// children top-to-bottom, while `"row"` composes them left-to-right on shared
// rows. Only "column" has a live caller today, but the
// layout engine (rendering/layout.ts) already models both as one node type
// (`StackLayoutNode.type: "vstack" | "hstack"`), so keeping "row" here costs
// nothing and avoids re-deriving the algorithm if a row layout is needed.

import {
	type Component,
	Container,
	compositeTuiLine,
	visibleWidth,
} from "../terminal/primitives.ts";
import {
	LAYOUT_NODE,
	type LayoutViewport,
	type StackLayoutEntry,
	type StackLayoutNode,
} from "./layout-node.ts";

export interface FlexEntryOptions {
	basis?: number | "auto";
	grow?: number;
	shrink?: number;
	minSize?: number;
	maxSize?: number;
	visible?: (viewport: LayoutViewport) => boolean;
}

export interface FlexEntry extends FlexEntryOptions {
	component: Component;
}

export type FlexChild = Component | FlexEntry;

export interface FlexOptions {
	direction?: "row" | "column";
	gap?: number;
	align?: "stretch" | "start" | "center" | "end";
}

function isFlexEntry(child: FlexChild): child is FlexEntry {
	return !("render" in child);
}

function normalizeSize(value: number | undefined, fallback: number): number {
	return value === undefined || !Number.isFinite(value)
		? fallback
		: Math.max(0, Math.floor(value));
}

export class Flex extends Container {
	private readonly entries: StackLayoutEntry[] = [];
	private readonly direction: "row" | "column";
	private readonly gap: number;
	private readonly align: "stretch" | "start" | "center" | "end";

	constructor(children: FlexChild[] = [], options: FlexOptions = {}) {
		super();
		this.direction = options.direction ?? "column";
		this.gap = normalizeSize(options.gap, 0);
		this.align = options.align ?? "stretch";
		for (const child of children) {
			if (isFlexEntry(child)) this.addChild(child.component, child);
			else this.addChild(child);
		}
	}

	override addChild(
		component: Component,
		options: FlexEntryOptions = {},
	): void {
		super.addChild(component);
		this.entries.push({
			component,
			...(options.basis === undefined ? {} : { basis: options.basis }),
			...(options.grow === undefined
				? {}
				: { grow: normalizeSize(options.grow, 0) }),
			...(options.shrink === undefined
				? {}
				: { shrink: normalizeSize(options.shrink, 1) }),
			...(options.minSize === undefined
				? {}
				: { minSize: normalizeSize(options.minSize, 0) }),
			...(options.maxSize === undefined
				? {}
				: { maxSize: normalizeSize(options.maxSize, Number.MAX_SAFE_INTEGER) }),
			...(options.visible === undefined ? {} : { visible: options.visible }),
		});
	}

	override removeChild(component: Component): void {
		super.removeChild(component);
		const index = this.entries.findIndex(
			entry => entry.component === component,
		);
		if (index !== -1) this.entries.splice(index, 1);
	}

	override clear(): void {
		super.clear();
		this.entries.length = 0;
	}

	[LAYOUT_NODE](): StackLayoutNode {
		return {
			type: this.direction === "column" ? "vstack" : "hstack",
			entries: this.entries,
			gap: this.gap,
			align: this.align,
		};
	}

	override render(width: number): string[] {
		return this.direction === "column"
			? this.renderColumn(width)
			: this.renderRow(width);
	}

	private renderColumn(width: number): string[] {
		const viewport = {
			width: Math.max(1, width),
			height: Number.MAX_SAFE_INTEGER,
		};
		const entries = visibleFlexEntries(this.entries, viewport);
		const rendered = entries.map(entry =>
			entry.component.render(viewport.width),
		);
		const sizes = allocateFlexSizes(
			entries,
			rendered.map(lines => lines.length),
			undefined,
			this.gap,
		);
		const lines: string[] = [];
		for (let index = 0; index < entries.length; index++) {
			if (index > 0) {
				for (let gap = 0; gap < this.gap; gap++) lines.push("");
			}
			const childLines = rendered[index]?.slice(0, sizes[index]);
			lines.push(...childLines);
			for (let padding = childLines.length; padding < sizes[index]!; padding++)
				lines.push("");
		}
		return lines;
	}

	private renderRow(width: number): string[] {
		const safeWidth = Math.max(1, width);
		const viewport = { width: safeWidth, height: Number.MAX_SAFE_INTEGER };
		const entries = visibleFlexEntries(this.entries, viewport);
		if (entries.length === 0) return [];

		const intrinsicWidths = entries.map(entry => {
			const lines = entry.component.render(safeWidth);
			return lines.reduce((max, line) => Math.max(max, visibleWidth(line)), 0);
		});
		const widths = allocateFlexSizes(
			entries,
			intrinsicWidths,
			safeWidth,
			this.gap,
		);
		const rendered = entries.map((entry, index) =>
			widths[index] === 0 ? [] : entry.component.render(widths[index]!),
		);
		const height = rendered.reduce(
			(max, lines) => Math.max(max, lines.length),
			0,
		);
		const result = Array.from({ length: height }, () => "");
		let x = 0;
		for (let index = 0; index < rendered.length; index++) {
			const lines = rendered[index]!;
			const childWidth = widths[index]!;
			let offset = 0;
			if (this.align === "center")
				offset = Math.floor((height - lines.length) / 2);
			else if (this.align === "end") offset = height - lines.length;
			for (let row = 0; row < lines.length; row++) {
				const target = row + offset;
				if (target < 0 || target >= result.length) continue;
				result[target] = compositeTuiLine(
					result[target]!,
					lines[row]!,
					x,
					childWidth,
					safeWidth,
				);
			}
			x += childWidth + this.gap;
		}
		return result;
	}
}

export function visibleFlexEntries(
	entries: readonly StackLayoutEntry[],
	viewport: LayoutViewport,
): StackLayoutEntry[] {
	return entries.filter(entry => entry.visible?.(viewport) ?? true);
}

function clampSize(size: number, entry: StackLayoutEntry): number {
	const min = Math.max(0, Math.floor(entry.minSize ?? 0));
	const max = Math.max(
		min,
		Math.floor(entry.maxSize ?? Number.MAX_SAFE_INTEGER),
	);
	return Math.max(min, Math.min(max, Math.max(0, Math.floor(size))));
}

function distribute(
	sizes: number[],
	entries: readonly StackLayoutEntry[],
	amount: number,
	mode: "grow" | "shrink",
): void {
	let remaining = amount;
	while (remaining > 0) {
		const candidates = entries
			.map((entry, index) => ({ entry, index }))
			.filter(({ entry, index }) => {
				if (mode === "grow") {
					return (
						(entry.grow ?? 0) > 0 &&
						sizes[index]! < (entry.maxSize ?? Number.MAX_SAFE_INTEGER)
					);
				}
				return (entry.shrink ?? 1) > 0 && sizes[index]! > (entry.minSize ?? 0);
			});
		if (candidates.length === 0) return;

		const totalWeight = candidates.reduce((sum, { entry, index }) => {
			return (
				sum +
				(mode === "grow"
					? (entry.grow ?? 0)
					: (entry.shrink ?? 1) * Math.max(1, sizes[index]!))
			);
		}, 0);
		let distributed = 0;
		for (const { entry, index } of candidates) {
			if (remaining <= 0) break;
			const weight =
				mode === "grow"
					? (entry.grow ?? 0)
					: (entry.shrink ?? 1) * Math.max(1, sizes[index]!);
			const proposed = Math.max(
				1,
				Math.floor((remaining * weight) / totalWeight),
			);
			const capacity =
				mode === "grow"
					? (entry.maxSize ?? Number.MAX_SAFE_INTEGER) - sizes[index]!
					: sizes[index]! - (entry.minSize ?? 0);
			const delta = Math.min(remaining, proposed, capacity);
			if (delta <= 0) continue;
			sizes[index] = sizes[index]! + (mode === "grow" ? delta : -delta);
			remaining -= delta;
			distributed += delta;
		}
		if (distributed === 0) return;
	}
}

export function allocateFlexSizes(
	entries: readonly StackLayoutEntry[],
	intrinsicSizes: readonly number[],
	availableSize: number | undefined,
	gap: number,
): number[] {
	const sizes = entries.map((entry, index) =>
		clampSize(
			entry.basis === undefined || entry.basis === "auto"
				? (intrinsicSizes[index] ?? 0)
				: entry.basis,
			entry,
		),
	);
	if (availableSize === undefined) return sizes;

	const contentSize = Math.max(
		0,
		Math.floor(availableSize) - Math.max(0, entries.length - 1) * gap,
	);
	const total = sizes.reduce((sum, size) => sum + size, 0);
	if (total < contentSize)
		distribute(sizes, entries, contentSize - total, "grow");
	else if (total > contentSize)
		distribute(sizes, entries, total - contentSize, "shrink");
	return sizes;
}
