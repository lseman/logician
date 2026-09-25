import { expect, test } from "bun:test";
import { InputBar } from "../input/input-bar.ts";

const schemes = ["memory", "ssh", "log", "local"];

function barWith(value: string): InputBar {
	const bar = new InputBar();
	bar.valueText = value;
	return bar;
}

test("getActiveUrlQuery detects a scheme:// token immediately before the cursor", () => {
	const bar = barWith("read memory://mem");
	expect(bar.getActiveUrlQuery(schemes)).toEqual({
		scheme: "memory",
		token: "memory://mem",
	});
});

test("getActiveUrlQuery keeps a path with a trailing slash inside the token", () => {
	const bar = barWith("see ssh://prod/");
	expect(bar.getActiveUrlQuery(schemes)).toEqual({
		scheme: "ssh",
		token: "ssh://prod/",
	});
});

test("getActiveUrlQuery ignores schemes embedded in longer words", () => {
	const bar = barWith("blog://x is not a scheme match");
	expect(bar.getActiveUrlQuery(schemes)).toBeNull();
});

test("getActiveUrlQuery disarms the token once whitespace follows it", () => {
	// Regression: the token sits at index 0 and its remainder holds a
	// whitespace, so the marker scan must stop instead of re-finding
	// index 0 forever (lastIndexOf clamps a negative fromIndex by the
	// string length).
	const bar = barWith("memory://mem then more text");
	expect(bar.getActiveUrlQuery(schemes)).toBeNull();
});

test("getActiveUrlQuery disarms a mid-line token followed by whitespace", () => {
	const bar = barWith("read memory://mem and more");
	expect(bar.getActiveUrlQuery(schemes)).toBeNull();
});

test("insertUrl replaces the active token and appends a trailing space", () => {
	const bar = barWith("read memory://mem");
	bar.insertUrl(schemes, "memory://memories");
	expect(bar.valueText).toBe("read memory://memories ");
});

test("insertUrl leaves the text untouched when no token is active", () => {
	const bar = barWith("memory://mem and more");
	bar.insertUrl(schemes, "memory://memories");
	expect(bar.valueText).toBe("memory://mem and more");
});
