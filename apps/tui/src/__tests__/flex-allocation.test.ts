import { expect, test } from "bun:test";
import { allocateFlexSizes } from "../rendering/flex.ts";
import type { StackLayoutEntry } from "../rendering/layout-node.ts";
import type { Component } from "../terminal/primitives.ts";

const component: Component = { render: () => [] };
const entry = (
	overrides: Omit<StackLayoutEntry, "component"> = {},
): StackLayoutEntry => ({ component, ...overrides });

test("equal flex weights receive equal integer growth", () => {
	expect(
		allocateFlexSizes(
			[entry({ basis: 0, grow: 1 }), entry({ basis: 0, grow: 1 })],
			[0, 0],
			10,
			0,
		),
	).toEqual([5, 5]);
});

test("growth redistributes capacity left unused by a max constraint", () => {
	expect(
		allocateFlexSizes(
			[entry({ basis: 0, grow: 1, maxSize: 2 }), entry({ basis: 0, grow: 1 })],
			[0, 0],
			10,
			0,
		),
	).toEqual([2, 8]);
});

test("scaled shrink stays proportional and respects minimum sizes", () => {
	expect(
		allocateFlexSizes(
			[
				entry({ basis: 10, shrink: 1, minSize: 5 }),
				entry({ basis: 20, shrink: 1, minSize: 8 }),
			],
			[10, 20],
			15,
			0,
		),
	).toEqual([5, 10]);
});

test("allocation fills exactly the width left by the gap", () => {
	const widths = allocateFlexSizes(
		[entry({ basis: 0, grow: 1 }), entry({ basis: 0, grow: 1 })],
		[0, 0],
		11,
		1,
	);
	expect(widths).toEqual([5, 5]);
	expect(widths.reduce((sum, w) => sum + w, 0) + 1).toBe(11);
});
