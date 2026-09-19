import { expect, test } from "bun:test";
import type { ToolExecution } from "@logician/log-runtime/sessions";
import {
	type RenderCtx,
	renderTool,
} from "../rendering/transcript/render/tool.ts";
import { visibleWidth } from "../terminal/core.ts";
import { initTheme, type ThemeColor, theme } from "../terminal/theme.ts";

function context(): RenderCtx {
	return {
		toolsExpanded: false,
		spinnerFrame: () => "*",
		maxMessageLength: 10000,
		sanitizedToolCache: new WeakMap(),
		sanitizationMetrics: { cacheHits: 0, scannedCharacters: 0 },
		currentWidth: 80,
		thinkingMode: "collapsed",
	};
}

test("tool output ends at the bottom border without trailing blank rows", () => {
	initTheme("dark");
	for (const expanded of [false, true]) {
		for (const ending of ["", "\n", "\n \n"]) {
			const lines = renderTool(
				context(),
				{
					tool_name: "bash",
					args: { command: "run" },
					result: `first\n\nlast${ending}`,
					isComplete: true,
					isError: false,
				},
				80,
				expanded,
			);
			expect(lines.at(-2)).toContain("last");
			const first = lines.findIndex(line => line.includes("first"));
			const last = lines.findIndex(line => line.includes("last"));
			expect(last - first).toBe(2);
		}
	}
});

const cases: { tool: ToolExecution; text: string; color: ThemeColor }[] = [
	{
		tool: {
			tool_name: "lsp",
			args: { action: "symbols" },
			result: "exampleSymbol",
			isComplete: true,
			isError: false,
		},
		text: "RESULTS",
		color: "active",
	},
	{
		tool: {
			tool_name: "eval",
			args: { code: 'const greeting = "hello";', language: "typescript" },
			result: "hello",
			isComplete: true,
			isError: false,
		},
		text: "greeting",
		color: "jsonKeyword",
	},
	{
		tool: {
			tool_name: "bash",
			args: { command: "run" },
			streamOutput: "progress message",
			isComplete: false,
			isError: false,
		},
		text: "progress message",
		color: "terminalOutput",
	},
	{
		tool: {
			tool_name: "bash",
			args: { command: "run" },
			result: "Error: failed",
			isComplete: true,
			isError: true,
		},
		text: "Error: failed",
		color: "diffRemoved",
	},
];

for (const { tool, text, color } of cases) {
	test(`collapsed ${tool.tool_name} preserves expanded styling for ${text}`, () => {
		initTheme("dark");
		for (const width of [40, 100]) {
			const ctx = context();
			const collapsed = renderTool(ctx, tool, width, false);
			const expanded = renderTool(ctx, tool, width, true);
			const row = collapsed.find(line => line.includes(text));
			expect(row).toBeDefined();
			expect(row).toContain(theme.fgRaw(color));
			expect(expanded).toContain(row);
			expect(collapsed.every(line => visibleWidth(line) <= width)).toBe(true);
		}
	});
}

test("cropped output retains colors and excludes terminal control injection", () => {
	initTheme("dark");
	const tool: ToolExecution = {
		tool_name: "bash",
		args: { command: "run" },
		result:
			Array.from({ length: 15 }, (_, i) => `line-${i}`).join("\n") +
			"\nError: failed\x1b[2J\n\n",
		isComplete: true,
		isError: true,
	};
	const lines = renderTool(context(), tool, 80, false);
	expect(lines.some(line => line.includes("line-0"))).toBe(false);
	const error = lines.find(line => line.includes("Error: failed"));
	expect(error).toContain(theme.fgRaw("diffRemoved"));
	expect(lines.join("\n")).not.toContain("\x1b[2J");
});
