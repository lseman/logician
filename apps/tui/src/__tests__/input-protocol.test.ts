import { expect, test } from "bun:test";
import {
	normalizeKeyboardInput,
	parseSgrMouseInput,
	splitNavigationBatch,
	TerminalInputBuffer,
} from "../terminal/input-protocol.ts";

test("Kitty control reports preserve ambiguous Ctrl+I and Ctrl+M", () => {
	expect(normalizeKeyboardInput("\x1b[99;5u")).toBe("\x03");
	expect(normalizeKeyboardInput("\x1b[99;133u")).toBe("\x03");
	expect(normalizeKeyboardInput("\x1b[27;129u")).toBe("\x1b");
	expect(normalizeKeyboardInput("\x1b[105;5u")).toBe("\x1b[105;5u");
	expect(normalizeKeyboardInput("\x1b[105;133u")).toBe("\x1b[105;133u");
	expect(normalizeKeyboardInput("\x1b[109;5u")).toBe("\x1b[109;5u");
});

test("Kitty key releases are ignored", () => {
	expect(normalizeKeyboardInput("\x1b[99;133:3u")).toBe("");
});

test("terminal input buffering splits batched interrupt keys", () => {
	const sequences: string[] = [];
	const buffer = new TerminalInputBuffer(sequence => sequences.push(sequence));
	buffer.process("a\x1b[99;5u\x03");
	expect(sequences.map(normalizeKeyboardInput)).toEqual(["a", "\x03", "\x03"]);
});

test("terminal input buffering reassembles fragmented Escape and Ctrl+C", async () => {
	const sequences: string[] = [];
	const buffer = new TerminalInputBuffer(
		sequence => sequences.push(sequence),
		5,
	);
	buffer.process("\x1b[");
	buffer.process("99;5u");
	buffer.process("\x1b");
	await Bun.sleep(10);
	expect(sequences.map(normalizeKeyboardInput)).toEqual(["\x03", "\x1b"]);
});

test("xterm modifyOtherKeys Ctrl+C is normalized", () => {
	expect(normalizeKeyboardInput("\x1b[27;5;99~")).toBe("\x03");
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
