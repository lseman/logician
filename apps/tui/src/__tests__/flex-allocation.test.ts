import { expect, test } from "bun:test";
import { allocateFlexSizes, Flex } from "../rendering/flex.ts";
import type { StackLayoutEntry } from "../rendering/layout-node.ts";
import { type Component, visibleWidth } from "../terminal/primitives.ts";

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

test("row rendering remains width-safe after allocation", () => {
	const left: Component = { render: width => ["L".repeat(width)] };
	const right: Component = { render: width => ["R".repeat(width)] };
	const flex = new Flex(
		[
			{ component: left, basis: 0, grow: 1 },
			{ component: right, basis: 0, grow: 1 },
		],
		{ direction: "row", gap: 1 },
	);
	const [line] = flex.render(11);
	expect(visibleWidth(line ?? "")).toBe(11);
	expect(line).toContain("LLLLL");
	expect(line).toContain("RRRRR");
});
