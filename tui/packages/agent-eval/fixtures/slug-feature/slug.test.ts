import { expect, test } from "bun:test";
import { slugify } from "./src/slug.ts";

test("normalizes words and punctuation", () => {
	expect(slugify("  Hello, Logician!  ")).toBe("hello-logician");
});

test("folds diacritics and collapses separators", () => {
	expect(slugify("Crème  brûlée -- recipe")).toBe("creme-brulee-recipe");
});

test("truncates at a word boundary without a trailing dash", () => {
	expect(slugify("alpha beta gamma", { maxLength: 12 })).toBe("alpha-beta");
});

test("returns an empty slug for punctuation-only input", () => {
	expect(slugify("... !!!")).toBe("");
});
