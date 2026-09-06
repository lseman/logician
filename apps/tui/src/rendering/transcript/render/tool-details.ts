// ── Transcript per-tool-type detail renderers ───────────────────────────────
// Expanded-detail rendering for write/edit/file_diff/bash/mcp tool executions.

import type { ToolExecution } from "@logician/log-runtime/sessions";
import { DIM, RESET } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import { detectLanguage } from "../file-language.ts";
import {
	normalizeEditArgs,
	streamedStringArg,
	stringArg,
} from "../text-utils.ts";
import { renderFileContent } from "./content.ts";
import type { RenderCtx } from "./tool-context.ts";

export interface ToolDetailHelpers {
	detailSection: (label: string, meta?: string) => string;
	detailSectionFile: (path: string) => string;
	previewBlock: (
		ctx: RenderCtx,
		text: string,
		width: number,
		maxChars?: number,
	) => string[];
	renderDiffBlock: (
		ctx: RenderCtx,
		text: string,
		width: number,
		language?: string,
	) => string[];
	renderMcpResultBlocks: (
		ctx: RenderCtx,
		text: string,
		width: number,
	) => string[];
	renderTerminalBlock: (
		ctx: RenderCtx,
		text: string,
		width: number,
	) => string[];
	writeFileContent: (tool: ToolExecution) => string | undefined;
}

export function renderWriteDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
	helpers: ToolDetailHelpers,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const path =
		stringArg(args, "path") ||
		stringArg(args, "file_path") ||
		streamedStringArg(tool.partialResult, "path") ||
		streamedStringArg(tool.partialResult, "file_path");
	const content = helpers.writeFileContent(tool);
	const streaming = !tool.isComplete;
	const appending = Boolean(args.append);

	if (path) lines.push(helpers.detailSectionFile(path));

	if (content !== undefined && content !== "") {
		const lineCount = content.split("\n").length;
		const meta = streaming
			? `${DIM}${content.length} bytes · ${lineCount} lines · streaming${RESET}`
			: `${DIM}${content.length} bytes · ${lineCount} lines${RESET}`;
		lines.push(
			helpers.detailSection(appending ? "append content" : "content", meta),
		);
		const lang = detectLanguage(path);
		lines.push(...renderFileContent(content, width, lineCount, lang, expanded));
	} else if (streaming) {
		lines.push(`${DIM}${appending ? "appending" : "writing"}…${RESET}`);
	}

	// Show error result only (skip diff — content is already rendered above).
	if (tool.result) {
		const resultText = tool.result;
		if (tool.isError) {
			lines.push(helpers.detailSection("error"));
			lines.push(...helpers.previewBlock(ctx, resultText, width));
		} else if (!content) {
			// No content shown above; show the diff result.
			lines.push(helpers.detailSection("result"));
			lines.push(
				...helpers.renderDiffBlock(
					ctx,
					resultText,
					width,
					detectLanguage(path),
				),
			);
		}
	} else if (!streaming && !content) {
		lines.push(`${DIM}no output${RESET}`);
	}

	return lines;
}

/** Parse accumulated partialResult JSON to extract tool args. */
export function renderEditDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
	helpers: ToolDetailHelpers,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const path = stringArg(args, "path") || stringArg(args, "file_path");
	const streaming = !tool.isComplete;
	const edits = normalizeEditArgs(args);
	const language = detectLanguage(path);

	if (path) lines.push(helpers.detailSectionFile(path));

	for (let i = 0; i < edits.length; i++) {
		lines.push(
			helpers.detailSection(`edit ${i + 1}`, `${i + 1} of ${edits.length}`),
		);
		const oldText = edits[i].oldText;
		const newText = edits[i].newText;

		if (oldText) {
			const oldLineCount = oldText.split("\n").length;
			const oldMeta = streaming
				? `${oldText.length} bytes · ${oldLineCount} lines · streaming`
				: `${oldText.length} bytes · ${oldLineCount} lines`;
			lines.push(
				`${theme.fgRaw("diffRemoved")}── - OLD${RESET}  ${DIM}${oldMeta}${RESET}`,
			);
			lines.push(
				...renderFileContent(oldText, width, oldLineCount, language, expanded),
			);
		}
		if (newText) {
			const newLineCount = newText.split("\n").length;
			const newMeta = streaming
				? `${newText.length} bytes · ${newLineCount} lines · streaming`
				: `${newText.length} bytes · ${newLineCount} lines`;
			lines.push(
				`${theme.fgRaw("diffAdded")}── + NEW${RESET}  ${DIM}${newMeta}${RESET}`,
			);
			lines.push(
				...renderFileContent(newText, width, newLineCount, language, expanded),
			);
		}
	}

	if (edits.length === 0 && streaming) {
		lines.push(`${DIM}editing…${RESET}`);
	}

	if (tool.result) {
		const resultText = tool.result.startsWith("Error:")
			? tool.result
			: tool.result;
		lines.push(helpers.detailSection(tool.isError ? "error" : "result"));
		if (tool.isError) {
			lines.push(...helpers.previewBlock(ctx, resultText, width));
		} else {
			lines.push(...helpers.renderDiffBlock(ctx, resultText, width, language));
		}
	}

	return lines;
}

export function renderFileDiffDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	helpers: ToolDetailHelpers,
): string[] {
	const args = tool.args || {};
	const lines: string[] = [];
	const path = stringArg(args, "path") || stringArg(args, "file_path");
	if (path) lines.push(helpers.detailSectionFile(path));
	if (args.staged) lines.push(`${DIM}staged changes${RESET}`);
	const result = tool.result ?? tool.partialResult;
	if (result) {
		lines.push(
			...helpers.renderDiffBlock(ctx, result, width, detectLanguage(path)),
		);
	}
	return lines;
}

export function renderBashDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	helpers: ToolDetailHelpers,
): string[] {
	const args = tool.args || {};
	const lines: string[] = [];
	const command = stringArg(args, "command") || "";
	const timeout = args.timeout ? `${Number(args.timeout)}ms` : "30000ms";
	if (command) {
		lines.push(helpers.detailSection("command", `timeout ${timeout}`));
		lines.push(...helpers.previewBlock(ctx, command, width));
	}
	const result = tool.result ?? tool.partialResult;
	if (result) {
		const label = tool.result
			? tool.isError
				? "error output"
				: "output"
			: "streaming output";
		lines.push(helpers.detailSection(label));
		lines.push(...helpers.renderTerminalBlock(ctx, result, width));
	} else if (tool.streamOutput) {
		// For live streaming before the tool completes, show the current
		// stream content so the collapsed preview isn't just "waiting…".
		lines.push(helpers.detailSection("streaming"));
		lines.push(...helpers.renderTerminalBlock(ctx, tool.streamOutput, width));
	} else {
		lines.push(`${DIM}waiting for command output...${RESET}`);
	}
	return lines;
}

export function renderMcpDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	helpers: ToolDetailHelpers,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const serverParts = tool.tool_name.replace(/^mcp__/, "").split("__");
	if (serverParts.length >= 2) {
		lines.push(
			helpers.detailSection(
				"mcp",
				`${serverParts[0]} · ${serverParts.slice(1).join("__")}`,
			),
		);
	}
	const argText = JSON.stringify(args, null, 2);
	if (argText && argText !== "{}") {
		lines.push(helpers.detailSection("arguments"));
		lines.push(...helpers.previewBlock(ctx, argText, width));
	}
	const result = tool.result ?? tool.partialResult;
	if (result) {
		lines.push(
			helpers.detailSection(tool.isError ? "mcp error" : "mcp result"),
		);
		lines.push(...helpers.renderMcpResultBlocks(ctx, result, width));
	}
	return lines;
}

export function renderEvalDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	helpers: ToolDetailHelpers,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const code = stringArg(args, "code") || "";
	const language = stringArg(args, "language") || "";

	if (code) {
		const lineCount = code.split("\n").length;
		const meta = `${DIM}${code.length} bytes · ${lineCount} lines${RESET}`;
		lines.push(helpers.detailSection("code", meta));
		lines.push(
			...renderFileContent(code, width, lineCount, language || undefined, true),
		);
	}

	const result = tool.result ?? tool.partialResult;
	if (result) {
		lines.push(helpers.detailSection(tool.isError ? "error" : "output"));
		lines.push(...helpers.previewBlock(ctx, result, width));
	}

	return lines;
}
// ── LSP tool detail renderer ─────────────────────────────────────────────────

export function renderLspDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	helpers: ToolDetailHelpers,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const action = String(args.action || "").toLowerCase();
	const file = String(args.file || args.filePath || "");

	// Show the action header
	if (file) {
		lines.push(helpers.detailSection("lsp", action));
	} else {
		lines.push(helpers.detailSection("lsp", action));
	}

	if (file && action !== "status" && action !== "capabilities") {
		lines.push(helpers.detailSectionFile(file));
	}

	if (args.line) {
		lines.push(`${DIM}line ${args.line}${args.column ? `:${args.column}` : ""}${RESET}`);
	}

	// Parse result text and format based on action type
	const result = tool.result ?? tool.partialResult;
	if (result) {
		const resultText = String(result);

		// Check for error prefix
		if (resultText.startsWith("LSP Error:")) {
			lines.push(helpers.detailSection(tool.isError ? "lsp error" : "lsp result"));
			lines.push(...helpers.previewBlock(ctx, resultText, width));
			return lines;
		}

		if (action === "diagnostics") {
			if (resultText === "No diagnostics.") {
				lines.push(`${theme.fg("success", "✓ No diagnostics")}${RESET}`);
			} else {
				lines.push(helpers.detailSection("diagnostics"));
				lines.push(...helpers.previewBlock(ctx, resultText, width));
			}
		} else if (action === "hover") {
			if (resultText === "No hover information.") {
				lines.push(`${DIM}No hover information${RESET}`);
			} else {
				lines.push(helpers.detailSection("hover"));
				lines.push(...helpers.previewBlock(ctx, resultText, width));
			}
		} else if (action === "status") {
			lines.push(...resultText.split("\n").map(line => `${DIM}${line}${RESET}`));
		} else if (action === "capabilities") {
			lines.push(helpers.detailSection("capabilities"));
			lines.push(...helpers.previewBlock(ctx, resultText, width));
		} else if (action === "code-actions") {
			lines.push(helpers.detailSection("available actions"));
			lines.push(...helpers.previewBlock(ctx, resultText, width));
		} else if (action === "symbols" || action === "workspace-symbols") {
			lines.push(helpers.detailSection("results"));
			lines.push(...helpers.previewBlock(ctx, resultText, width));
		} else {
			// Generic: go-to-definition, references, rename, type-definition, implementation
			const hasLocations = resultText.includes(":") && resultText.split("\n").some(l => /^\s+\S+:\d+:\d+/.test(l));
			if (hasLocations) {
				lines.push(helpers.detailSection("locations"));
				lines.push(...helpers.previewBlock(ctx, resultText, width));
			} else {
				lines.push(helpers.detailSection(tool.isError ? "error" : "result"));
				lines.push(...helpers.previewBlock(ctx, resultText, width));
			}
		}
	} else if (!tool.isComplete) {
		lines.push(`${DIM}waiting for result...${RESET}`);
	}

	return lines;
}
