import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { renderFileContent } from "../rendering/transcript/render/content.ts";
import { renderMarkdownLines } from "../rendering/transcript/render/markdown-table.ts";
import { initTheme, type ThemeColor, theme } from "../terminal/theme.ts";

const home = process.env.HOME;
const colorterm = process.env.COLORTERM;
let tempHome: string;
beforeEach(() => {
	tempHome = mkdtempSync(join(tmpdir(), "logician-themes-"));
	process.env.HOME = tempHome;
});
afterEach(() => {
	if (home === undefined) delete process.env.HOME;
	else process.env.HOME = home;
	if (colorterm === undefined) delete process.env.COLORTERM;
	else process.env.COLORTERM = colorterm;
	rmSync(tempHome, { recursive: true, force: true });
	initTheme("dark");
});

function rgb(ansi: string): number[] {
	const values = ansi.slice(2, -1).split(";").map(Number);
	if (values[1] === 2) return values.slice(2);
	const index = values[2];
	if (index >= 232) return Array(3).fill(8 + (index - 232) * 10);
	if (index < 16)
		throw new Error(
			"Bundled colors must not depend on terminal ANSI overrides",
		);
	const n = index - 16,
		cube = [0, 95, 135, 175, 215, 255];
	return [cube[Math.floor(n / 36)], cube[Math.floor(n / 6) % 6], cube[n % 6]];
}
function luminance(color: number[]): number {
	return color
		.map(v => {
			const c = v / 255;
			return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
		})
		.reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
}
function contrast(a: number[], b: number[]): number {
	const x = luminance(a),
		y = luminance(b);
	return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

for (const mode of ["truecolor", "256color"]) {
	test(`all bundled themes keep text and borders legible in ${mode}`, () => {
		process.env.COLORTERM = mode === "truecolor" ? "truecolor" : "";
		const dir = join(import.meta.dir, "../../themes");
		for (const file of readdirSync(dir).filter(f => f.endsWith(".json"))) {
			const json = JSON.parse(readFileSync(join(dir, file), "utf8"));
			initTheme(json.name);
			expect(theme.name).toBe(json.name);
			for (const surface of ["mdCodeBlockBg", "toolBlockBg"] as const) {
				const background = rgb(theme.bgRaw(surface));
				for (const token of [
					"text",
					"toolTitle",
					"toolOutput",
					"terminalOutput",
					"mdCodeBlock",
					"thinkingText",
					"muted",
					"dim",
					"jsonKey",
					"jsonString",
					"jsonNumber",
					"jsonKeyword",
					"jsonPunctuation",
					"toolSuccess",
					"toolError",
					"toolRunning",
				] as ThemeColor[]) {
					expect(
						contrast(rgb(theme.fgRaw(token)), background),
						`${json.name}: ${token} on ${surface}`,
					).toBeGreaterThanOrEqual(4.5);
				}
				for (const token of [
					"border",
					"borderMuted",
					"mdCodeBlockBorder",
				] as ThemeColor[]) {
					expect(
						contrast(rgb(theme.fgRaw(token)), background),
						`${json.name}: ${token} on ${surface}`,
					).toBeGreaterThanOrEqual(2.5);
				}
			}
		}
	});
}

test("code surfaces survive nested resets and syntax follows theme switches", () => {
	process.env.COLORTERM = "truecolor";
	initTheme("dark");
	const code = '```typescript\nconst greeting = "hello";\n```';
	const dark = renderMarkdownLines(code, 80, false).join("\n");
	initTheme("light");
	const light = renderMarkdownLines(code, 80, false).join("\n");
	expect(light).not.toBe(dark);
	expect(light).toContain(theme.fgRaw("jsonKeyword"));
	expect(light).toContain(theme.fgRaw("jsonString"));
	const bg = theme.bgRaw("mdCodeBlockBg");
	const nested = theme.bg(
		"mdCodeBlockBg",
		theme.fg("text", "first") + " second",
	);
	expect(nested).toContain(`\x1b[0m${bg} second`);
	const file = renderFileContent("hello", 80, 1, undefined, true)[0];
	expect(file.startsWith(bg)).toBe(true);
	expect(file).toContain(`\x1b[0m${bg}`);
	expect(file.endsWith("\x1b[0m")).toBe(true);
});
