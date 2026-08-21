// ── Transcript per-tool-type detail renderers ───────────────────────────────
// Expanded-detail rendering for write/edit/file_diff/bash/mcp tool executions.

import type { ToolExecution } from "@logician/agent-runtime/sessions";
import { DIM, RESET } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import { detectLanguage } from "../file-language.ts";
import {
	normalizeEditArgs,
	streamedStringArg,
	stringArg,
} from "../text-utils.ts";
import { renderFileContent } from "./content.ts";
import {
	detailSection,
	detailSectionFile,
	previewBlock,
	type RenderCtx,
	renderDiffBlock,
	renderMcpResultBlocks,
	renderTerminalBlock,
	writeFileContent,
} from "./tool.ts";

export function renderWriteDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const path =
		stringArg(args, "path") ||
		stringArg(args, "file_path") ||
		streamedStringArg(tool.partialResult, "path") ||
		streamedStringArg(tool.partialResult, "file_path");
	const content = writeFileContent(tool);
	const streaming = !tool.isComplete;
	const appending = tool.tool_name === "write_file_append";

	if (path) lines.push(detailSectionFile(path));

	if (content !== undefined && content !== "") {
		const lineCount = content.split("\n").length;
		const meta = streaming
			? `${DIM}${content.length} bytes · ${lineCount} lines · streaming${RESET}`
			: `${DIM}${content.length} bytes · ${lineCount} lines${RESET}`;
		lines.push(detailSection(appending ? "append content" : "content", meta));
		const lang = detectLanguage(path);
		lines.push(...renderFileContent(content, width, lineCount, lang, expanded));
	} else if (streaming) {
		lines.push(`${DIM}${appending ? "appending" : "writing"}…${RESET}`);
	}

	// Show error result only (skip diff — content is already rendered above).
	if (tool.result) {
		const resultText = tool.result;
		if (tool.isError) {
			lines.push(detailSection("error"));
			lines.push(...previewBlock(ctx, resultText, width));
		} else if (!content) {
			// No content shown above; show the diff result.
			lines.push(detailSection("result"));
			lines.push(
				...renderDiffBlock(ctx, resultText, width, detectLanguage(path)),
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
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const path = stringArg(args, "path") || stringArg(args, "file_path");
	const streaming = !tool.isComplete;
	const edits = normalizeEditArgs(args);
	const language = detectLanguage(path);

	if (path) lines.push(detailSectionFile(path));

	for (let i = 0; i < edits.length; i++) {
		lines.push(detailSection(`edit ${i + 1}`, `${i + 1} of ${edits.length}`));
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
		lines.push(detailSection(tool.isError ? "error" : "result"));
		if (tool.isError) {
			lines.push(...previewBlock(ctx, resultText, width));
		} else {
			lines.push(...renderDiffBlock(ctx, resultText, width, language));
		}
	}

	return lines;
}

export function renderFileDiffDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
): string[] {
	const args = tool.args || {};
	const lines: string[] = [];
	const path = stringArg(args, "path") || stringArg(args, "file_path");
	if (path) lines.push(detailSectionFile(path));
	if (args.staged) lines.push(`${DIM}staged changes${RESET}`);
	const result = tool.result ?? tool.partialResult;
	if (result) {
		lines.push(...renderDiffBlock(ctx, result, width, detectLanguage(path)));
	}
	return lines;
}

export function renderBashDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
): string[] {
	const args = tool.args || {};
	const lines: string[] = [];
	const command = stringArg(args, "command") || "";
	const timeout = args.timeout ? `${Number(args.timeout)}ms` : "30000ms";
	if (command) {
		lines.push(detailSection("command", `timeout ${timeout}`));
		lines.push(...previewBlock(ctx, command, width));
	}
	const result = tool.result ?? tool.partialResult;
	if (result) {
		const label = tool.result
			? tool.isError
				? "error output"
				: "output"
			: "streaming output";
		lines.push(detailSection(label));
		lines.push(...renderTerminalBlock(ctx, result, width));
	} else {
		lines.push(`${DIM}waiting for command output...${RESET}`);
	}
	return lines;
}

export function renderMcpDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const serverParts = tool.tool_name.replace(/^mcp__/, "").split("__");
	if (serverParts.length >= 2) {
		lines.push(
			detailSection(
				"mcp",
				`${serverParts[0]} · ${serverParts.slice(1).join("__")}`,
			),
		);
	}
	const argText = JSON.stringify(args, null, 2);
	if (argText && argText !== "{}") {
		lines.push(detailSection("arguments"));
		lines.push(...previewBlock(ctx, argText, width));
	}
	const result = tool.result ?? tool.partialResult;
	if (result) {
		lines.push(detailSection(tool.isError ? "mcp error" : "mcp result"));
		lines.push(...renderMcpResultBlocks(ctx, result, width));
	}
	return lines;
}
