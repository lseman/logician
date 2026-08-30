// ── Flex — single flexbox-style stack ──────────────────────────────────────
// `direction: "column"` stacks children top-to-bottom, `"row"` composes them
// left-to-right; the choice only selects the `StackLayoutNode.type`
// (`"vstack" | "hstack"`) emitted from `[LAYOUT_NODE]()`. Flex has no render
// path of its own — the layout engine (rendering/layout.ts) consumes the node
// and drives sizing through `allocateFlexSizes`. Outside the layout engine it
// falls back to `Container`'s naive vertical concatenation.

import { type Component, Container } from "../terminal/primitives.ts";
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

interface FlexEntry extends FlexEntryOptions {
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
}

export function visibleFlexEntries(
	entries: readonly StackLayoutEntry[],
	viewport: LayoutViewport,
): StackLayoutEntry[] {
	return entries.filter(entry => entry.visible?.(viewport) ?? true);
}

function clampSize(size: number, entry: StackLayoutEntry): number {
	const min = normalizeSize(entry.minSize, 0);
	const max = Math.max(
		min,
		normalizeSize(entry.maxSize, Number.MAX_SAFE_INTEGER),
	);
	return Math.max(min, Math.min(max, normalizeSize(size, 0)));
}

interface DistributionCandidate {
	index: number;
	weight: number;
	capacity: number;
}

function distribute(
	sizes: number[],
	entries: readonly StackLayoutEntry[],
	amount: number,
	mode: "grow" | "shrink",
): void {
	let remaining = normalizeSize(amount, 0);
	while (remaining > 0) {
		const candidates: DistributionCandidate[] = [];
		for (let index = 0; index < entries.length; index++) {
			const entry = entries[index];
			if (!entry) continue;
			const size = sizes[index] ?? 0;
			const factor = mode === "grow" ? (entry.grow ?? 0) : (entry.shrink ?? 1);
			const limit =
				mode === "grow"
					? normalizeSize(entry.maxSize, Number.MAX_SAFE_INTEGER)
					: normalizeSize(entry.minSize, 0);
			const capacity = mode === "grow" ? limit - size : size - limit;
			const weight = mode === "grow" ? factor : factor * Math.max(1, size);
			if (factor > 0 && capacity > 0 && weight > 0) {
				candidates.push({ index, weight, capacity });
			}
		}
		if (candidates.length === 0) return;

		const totalWeight = candidates.reduce((sum, item) => sum + item.weight, 0);
		const roundAmount = remaining;
		const allocations = candidates.map(candidate => {
			const ideal = (roundAmount * candidate.weight) / totalWeight;
			return {
				...candidate,
				ideal,
				delta: Math.min(candidate.capacity, Math.floor(ideal)),
			};
		});
		let distributed = allocations.reduce((sum, item) => sum + item.delta, 0);
		let remainder = Math.min(
			roundAmount - distributed,
			allocations.reduce(
				(sum, item) => sum + Math.max(0, item.capacity - item.delta),
				0,
			),
		);
		allocations.sort(
			(a, b) =>
				b.ideal - Math.floor(b.ideal) - (a.ideal - Math.floor(a.ideal)) ||
				a.index - b.index,
		);
		for (const item of allocations) {
			if (remainder <= 0) break;
			if (item.delta >= item.capacity) continue;
			item.delta++;
			remainder--;
			distributed++;
		}
		for (const { index, delta } of allocations) {
			const size = sizes[index] ?? 0;
			sizes[index] = size + (mode === "grow" ? delta : -delta);
		}
		remaining -= distributed;
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

	const safeGap = normalizeSize(gap, 0);
	const contentSize = Math.max(
		0,
		normalizeSize(availableSize, 0) - Math.max(0, entries.length - 1) * safeGap,
	);
	const total = sizes.reduce((sum, size) => sum + size, 0);
	if (total < contentSize)
		distribute(sizes, entries, contentSize - total, "grow");
	else if (total > contentSize)
		distribute(sizes, entries, total - contentSize, "shrink");
	return sizes;
}
