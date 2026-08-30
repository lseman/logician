import { expect, test } from "bun:test";
import {
	normalizeKeyboardInput,
	parseSgrMouseInput,
	splitNavigationBatch,
} from "../terminal/input-protocol.ts";

test("Kitty control reports preserve ambiguous Ctrl+I and Ctrl+M", () => {
	expect(normalizeKeyboardInput("\x1b[99;5u")).toBe("\x03");
	expect(normalizeKeyboardInput("\x1b[105;5u")).toBe("\x1b[105;5u");
	expect(normalizeKeyboardInput("\x1b[109;5u")).toBe("\x1b[109;5u");
});

test("navigation batches split only when the entire chunk is recognized", () => {
	expect(splitNavigationBatch("\x1b[6~\x1b[6~")).toEqual([
		"\x1b[6~",
		"\x1b[6~",
	]);
	expect(splitNavigationBatch("\x1b[A\x1b[A")).toBeNull();
	expect(splitNavigationBatch("\x1b[6~text")).toBeNull();
});

test("SGR mouse decoding coalesces wheel ticks and retains click order", () => {
	expect(parseSgrMouseInput("\x1b[<64;4;7M\x1b[<65;8;9M\x1b[<0;3;5M")).toEqual({
		clicks: [{ column: 2, row: 4 }],
		consumedLength: 29,
		wheel: { column: 7, row: 8, ticks: 0 },
	});
});

test("SGR mouse decoding stops at the first malformed byte", () => {
	const first = "\x1b[<65;2;3M";
	expect(parseSgrMouseInput(`${first}x\x1b[<65;4;5M`)).toEqual({
		clicks: [],
		consumedLength: first.length,
		wheel: { column: 1, row: 2, ticks: 1 },
	});
});

test("SGR mouse decoding exposes an unconsumed suffix", () => {
	const data = "\x1b[<65;2;3Mtail";
	const parsed = parseSgrMouseInput(data);
	expect(parsed?.consumedLength).toBe(data.length - 4);
	expect(parsed?.wheel).toEqual({ column: 1, row: 2, ticks: 1 });
	expect(parseSgrMouseInput("plain text")).toBeNull();
});
