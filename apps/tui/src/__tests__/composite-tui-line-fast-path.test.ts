import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	clampLineToWidth,
	compositeTuiLine,
	visibleWidth,
} from "../terminal/primitives.ts";

// Reference (slow-path) implementation, copied inline so the fast path can be
// checked against it directly without relying on compositeTuiLine's own
// internal branch selection.
function slowCompositeTuiLine(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string {
	const RESET = "\x1b[0m";
	const before = clampLineToWidth(baseLine, startCol);
	const beforeWidth = visibleWidth(before);
	const beforePad = " ".repeat(Math.max(0, startCol - beforeWidth));
	const overlay = clampLineToWidth(overlayLine, overlayWidth);
	const overlayPad = " ".repeat(
		Math.max(0, overlayWidth - visibleWidth(overlay)),
	);
	const afterStart = startCol + overlayWidth;
	const afterWidth = Math.max(0, totalWidth - afterStart);
	// Reimplement skipColumns inline (not exported) using clampLineToWidth's
	// sibling behavior: strip the first afterStart visible columns.
	let visible = 0;
	let i = 0;
	let skipped = "";
	while (i < baseLine.length && visible < afterStart) {
		const ch = baseLine[i];
		if (ch === "\x1b") {
			const next = baseLine[i + 1];
			if (next === "[") {
				let j = i + 2;
				while (
					j < baseLine.length &&
					!(baseLine.charCodeAt(j) >= 0x40 && baseLine.charCodeAt(j) <= 0x7e)
				)
					j++;
				i = j + 1;
				continue;
			}
			if (next === "]" || next === "_") {
				let j = i + 2;
				while (
					j < baseLine.length &&
					baseLine[j] !== "\x07" &&
					!(baseLine[j] === "\x1b" && baseLine[j + 1] === "\\")
				)
					j++;
				i = baseLine[j] === "\x07" ? j + 1 : j + 2;
				continue;
			}
			i++;
			continue;
		}
		visible++;
		i++;
	}
	skipped = baseLine.slice(i);
	const after = afterWidth > 0 ? clampLineToWidth(skipped, afterWidth) : "";
	const afterPad = " ".repeat(Math.max(0, afterWidth - visibleWidth(after)));
	return `${before}${beforePad}${RESET}${overlay}${overlayPad}${RESET}${after}${afterPad}`;
}

const cases: Array<{
	label: string;
	base: string;
	overlay: string;
	startCol: number;
	overlayWidth: number;
	totalWidth: number;
}> = [
	{
		label: "plain ascii",
		base: "hello world",
		overlay: "XX",
		startCol: 3,
		overlayWidth: 2,
		totalWidth: 20,
	},
	{
		label: "colored base",
		base: "\x1b[31mhello\x1b[0m world",
		overlay: "YY",
		startCol: 3,
		overlayWidth: 2,
		totalWidth: 20,
	},
	{
		label: "colored overlay",
		base: "plain text here",
		overlay: "\x1b[32mZZ\x1b[0m",
		startCol: 5,
		overlayWidth: 2,
		totalWidth: 20,
	},
	{
		label: "overlay wider than slot",
		base: "abcdefghij",
		overlay: "\x1b[33mtoolong\x1b[0m",
		startCol: 2,
		overlayWidth: 3,
		totalWidth: 15,
	},
	{
		label: "startCol beyond base length",
		base: "short",
		overlay: "XX",
		startCol: 10,
		overlayWidth: 2,
		totalWidth: 20,
	},
	{
		label: "zero overlay width",
		base: "hello world",
		overlay: "",
		startCol: 3,
		overlayWidth: 0,
		totalWidth: 20,
	},
	{
		label: "multiple ansi codes in base",
		base: "\x1b[1m\x1b[31mbold red\x1b[0m normal \x1b[32mgreen\x1b[0m",
		overlay: "MM",
		startCol: 4,
		overlayWidth: 2,
		totalWidth: 30,
	},
	{
		label: "empty base and overlay",
		base: "",
		overlay: "",
		startCol: 0,
		overlayWidth: 0,
		totalWidth: 10,
	},
	{
		label: "tab in base",
		base: "a\tb\tc",
		overlay: "X",
		startCol: 2,
		overlayWidth: 1,
		totalWidth: 20,
	},
	{
		label: "hyperlink osc8",
		base: "\x1b]8;;https://example.com\x07link text\x1b]8;;\x07 rest",
		overlay: "OO",
		startCol: 3,
		overlayWidth: 2,
		totalWidth: 30,
	},
	{
		label: "overlay at totalWidth boundary",
		base: "0123456789",
		overlay: "XY",
		startCol: 8,
		overlayWidth: 2,
		totalWidth: 10,
	},
	{
		label: "after region larger than remaining base",
		base: "short",
		overlay: "AB",
		startCol: 0,
		overlayWidth: 2,
		totalWidth: 20,
	},
];

for (const c of cases) {
	test(`compositeTuiLine fast path matches slow path: ${c.label}`, () => {
		const fast = compositeTuiLine(
			c.base,
			c.overlay,
			c.startCol,
			c.overlayWidth,
			c.totalWidth,
		);
		const slow = slowCompositeTuiLine(
			c.base,
			c.overlay,
			c.startCol,
			c.overlayWidth,
			c.totalWidth,
		);
		assert.equal(fast, slow);
	});
}

test("compositeTuiLine fast path: randomized ascii fuzz", () => {
	const chars =
		"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?";
	const colors = [
		"\x1b[31m",
		"\x1b[32m",
		"\x1b[1m",
		"\x1b[0m",
		"\x1b[38;5;120m",
	];
	function randomLine(len: number): string {
		let s = "";
		for (let i = 0; i < len; i++) {
			if (Math.random() < 0.1)
				s += colors[Math.floor(Math.random() * colors.length)];
			s += chars[Math.floor(Math.random() * chars.length)];
		}
		return s;
	}
	let _seed = 0;
	for (let trial = 0; trial < 2000; trial++) {
		_seed++;
		const base = randomLine(20);
		const overlay = randomLine(8);
		const startCol = Math.floor(Math.random() * 15);
		const overlayWidth = Math.floor(Math.random() * 10);
		const totalWidth = 30;
		const fast = compositeTuiLine(
			base,
			overlay,
			startCol,
			overlayWidth,
			totalWidth,
		);
		const slow = slowCompositeTuiLine(
			base,
			overlay,
			startCol,
			overlayWidth,
			totalWidth,
		);
		assert.equal(
			fast,
			slow,
			`mismatch on trial ${trial}: base=${JSON.stringify(base)} overlay=${JSON.stringify(overlay)} startCol=${startCol} overlayWidth=${overlayWidth}`,
		);
	}
});
