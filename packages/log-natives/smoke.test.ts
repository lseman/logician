import { expect, test } from "bun:test";
import {
	astMatch,
	countTokens,
	diffLines,
	editDiffString,
	search,
} from "./index.js";

test("Bun loads and executes the native capabilities", async () => {
	expect(countTokens("hello world")).toBeGreaterThan(0);
	expect(search("first\nneedle\nlast", { pattern: "needle" }).matchCount).toBe(
		1,
	);
	expect(diffLines("old\n", "new\n").length).toBeGreaterThan(0);
	expect(editDiffString("old\n", "new\n", "example.txt")).toBeDefined();
	const matches = await astMatch({
		source: "const value = 42;",
		lang: "typescript",
		patterns: ["const $NAME = $VALUE"],
	});
	expect(matches.totalMatches).toBe(1);
});
