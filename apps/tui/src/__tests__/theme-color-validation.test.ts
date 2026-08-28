import { expect, test } from "bun:test";
import { initTheme, theme } from "../terminal/theme.ts";

test("theme color validation rejects unknown footer configuration tokens", () => {
	initTheme("dark");
	expect(theme.hasColor("accent")).toBe(true);
	expect(theme.hasColor("not-a-theme-token")).toBe(false);
});
