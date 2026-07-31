// ── Transcript general tool rendering ───────────────────────────────────────
// Renders a single tool execution row (header + collapsed preview + expanded
// details), tool-result sanitization, and the shared low-level content-block
// helpers (diff/terminal/preview/mcp/permission blocks) used by both the
// per-tool-type detail renderers and the subagent renderers.

import {
	highlight,
	highlightAuto,
} from "@logician/agent-core/tools/shared/syntax-highlighter.ts";
import type {
	ThinkingDisplayStyle,
	ToolExecution,
} from "@logician/coding-agent/sessions";
import {
	BOLD,
	clampLineToWidth,
	DIM,
	RESET,
	visibleWidth,
} from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import {
	compactText,
	diffLineColor,
	extractPostEditDiagnostics,
	formatDurationMs,
	isPermissionRejection,
	parseJsonMaybe,
	type PostEditDiagnosticBlock,
	stringArg,
	streamedStringArg,
	streamedStringArgLive,
	stripInternalHookGuidance,
} from "../text-utils.ts";
import { detectLanguage } from "../file-language.ts";
import { wrapText } from "../layout.ts";
import { truncateText, withTruncationMarker } from "./content.ts";
import {
	renderSubagentBatchDetails,
	renderSubagentDetails,
} from "./subagent.ts";
import {
	renderBashDetails,
	renderEditDetails,
	renderFileDiffDetails,
	renderMcpDetails,
	renderWriteDetails,
} from "./tool-details.ts";
import {
	sanitizeTerminalText,
	sanitizeTerminalValue,
} from "../../terminal-sanitize.ts";

// ── Shared render context ────────────────────────────────────────────────────
// TranscriptDisplay's instance state, as read by the free functions extracted
// from it. TranscriptDisplay satisfies this shape naturally.

export interface SanitizedStringCache {
	raw?: string;
	safe?: string;
}

export interface SanitizedToolCache {
	result: SanitizedStringCache;
	partialResult: SanitizedStringCache;
	streamOutput: SanitizedStringCache;
	argsSource?: ToolExecution["args"];
	argsSafe?: ToolExecution["args"];
}

export interface RenderCtx {
	toolsExpanded: boolean;
	spinnerFrame: () => string;
	maxMessageLength: number;
	batchTaskTiming: Map<string, Map<number, { startedAt: number; endedAt?: number }>>;
	sanitizedToolCache: WeakMap<ToolExecution, SanitizedToolCache>;
	sanitizationMetrics: { cacheHits: number; scannedCharacters: number };
	currentWidth: number;
	thinkingMode: ThinkingDisplayStyle;
}

export function renderTool(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded = ctx.toolsExpanded,
): string[] {
	// Tool results, streamed output, arguments, and nested subagent details
	// are untrusted terminal input. Clone and remove every terminal control
	// before any markdown, highlighting, wrapping, or ANSI styling is added.
	tool = sanitizeToolForDisplay(ctx, tool);
	// Hook guidance is part of the model-visible tool result. Strip it only
	// from this local display copy so internal instructions never leak into
	// the user-facing transcript.
	const postEdit = extractPostEditDiagnostics(tool.result);
	tool = {
		...tool,
		result: stripInternalHookGuidance(postEdit.text),
		partialResult: stripInternalHookGuidance(tool.partialResult),
	};
	const lines: string[] = [];
	const subagent = tool.tool_name === "spawn_agent";
	const subagentBatch = tool.tool_name === "spawn_agents";
	const batchTally = subagentBatch ? computeBatchTally(ctx, tool) : null;
	const batchFailed = batchTally?.failed ?? 0;
	const glyph = tool.isError
		? theme.fg("toolError", "×")
		: tool.isComplete && batchFailed > 0
			? theme.fg("warning", "!")
			: tool.isComplete
				? theme.fg("toolSuccess", "✓")
				: theme.fg("toolRunning", ctx.spinnerFrame());
	const status = tool.isError
		? theme.fg("toolError", "error")
		: tool.isComplete && batchFailed > 0
			? theme.fg("warning", `partial · ${batchFailed} failed`)
			: tool.isComplete
				? theme.fg("toolSuccess", "done")
				: batchTally
					? theme.fg(
							"toolStreaming",
							`${batchTally.completed + batchTally.running}/${batchTally.total} running${
								batchTally.failed > 0 ? ` · ${batchTally.failed} failed` : ""
							}`,
						)
					: tool.partialResult || tool.streamOutput
						? theme.fg("toolStreaming", "streaming")
						: theme.fg("toolRunning", "running");
	const subagentAgent = subagent
		? String(tool.details?.agent || tool.args?.agent || "general")
		: "";

	// Extract file path for write_file / edit_file to show in header
	const filePath = (() => {
		const args = tool.args || {};
		const path =
			stringArg(args, "path") ||
			stringArg(args, "file_path") ||
			streamedStringArg(tool.partialResult, "path") ||
			streamedStringArg(tool.partialResult, "file_path") ||
			(tool.tool_name === "write_file"
				? /^Created\s+(.+?)(?:\s+\([^\n]*\))?(?:\n|$)/.exec(
						tool.result ?? "",
					)?.[1]
				: undefined);
		if (!path) return "";
		if (
			["write_file", "write_file_append", "edit_file"].includes(tool.tool_name)
		) {
			return path;
		}
		return "";
	})();

	const summary = toolSummary(tool);
	const elapsed =
		tool.durationMs !== undefined ? formatDurationMs(tool.durationMs) : "";
	const base = subagent
		? `${glyph} ${theme.fg("toolTitle", "subagent")} ${theme.fg("active", subagentAgent)} ${status}`
		: subagentBatch
			? `${glyph} ${theme.fg("toolTitle", "subagents")} ${status}`
			: filePath
				? `${glyph} ${theme.fg("toolTitle", tool.tool_name)} ${DIM}${filePath}${RESET} ${status}`
				: `${glyph} ${theme.fg("toolTitle", tool.tool_name)} ${status}`;
	const middle = summary ? `${DIM}${summary}${RESET}` : "";
	const right = elapsed ? `${DIM}${elapsed}${RESET}` : "";
	let row = [base, middle].filter(Boolean).join(` ${DIM}·${RESET} `);
	if (right) {
		const available = Math.max(1, width - 4);
		const gap = available - visibleWidth(row) - visibleWidth(right);
		row = gap >= 2 ? `${row}${" ".repeat(gap)}${right}` : `${row} ${right}`;
	}
	lines.push(clampLineToWidth(row, Math.max(1, width - 4)) + RESET);

	// Edits keep their compact diff preview. File writes and appends report
	// their live line count in the header and reveal content on expand.
	const showDiffResult =
		!expanded && tool.tool_name === "edit_file" && !!tool.result;
	if (showDiffResult) {
		const resultText = tool.result ?? "";
		const label = tool.isError ? "error" : "result";
		if (tool.isError) {
			const resultLines = resultText.split("\n");
			lines.push(
				`${theme.fg("toolError", "│ ")}${BOLD}${theme.fg("toolError", label)}${RESET} ${resultLines[0]}`,
			);
			for (let ri = 1; ri < resultLines.length; ri++) {
				lines.push(`${theme.fg("toolError", "│ ")}${resultLines[ri]}`);
			}
		} else {
			// Syntax-highlight the diff in collapsed view.
			const diffLines = renderDiffBlock(
				ctx,
				resultText,
				Math.max(20, width - 4),
				detectLanguage(filePath),
			);
			lines.push(`${theme.fg("dim", "│ ")}${BOLD}${label}${RESET}`);
			for (const dl of diffLines) {
				lines.push(`${theme.fg("dim", "│ ")}${dl}`);
			}
		}
	}
	const compactPreview =
		!showDiffResult && !expanded && !subagent && !subagentBatch
			? collapsedToolPreview(tool)
			: "";
	if (compactPreview) {
		const label = tool.isError
			? theme.fg("toolError", "error")
			: !tool.isComplete
				? theme.fg("toolRunning", "live")
				: theme.fg("muted", "output");
		lines.push(
			clampLineToWidth(
				`${theme.fg("dim", "└─")} ${label} ${compactPreview}${RESET}`,
				Math.max(1, width - 4),
			),
		);
	}
	for (const block of postEdit.blocks) {
		lines.push(
			...renderPostEditDiagnostics(block, Math.max(20, width - 4)),
		);
	}
	if (!expanded && !subagent && !subagentBatch) return lines;
	if (!subagent && !subagentBatch) {
		lines.push(`${theme.fg("dim", "│ ")}${theme.fg("active", "◆ details")}`);
	}
	for (const detailLine of toolDetailLines(ctx, tool, width - 2, expanded)) {
		const wrapped = wrapText(detailLine, Math.max(20, width - 4));
		for (const line of wrapped) {
			lines.push(`${theme.fg("dim", "│ ")}${line}`);
		}
	}

	return lines;
}

export function sanitizeToolForDisplay(
	ctx: RenderCtx,
	tool: ToolExecution,
): ToolExecution {
	let cache = ctx.sanitizedToolCache.get(tool);
	if (!cache) {
		cache = {
			result: {},
			partialResult: {},
			streamOutput: {},
		};
		ctx.sanitizedToolCache.set(tool, cache);
	}
	const sanitizeString = (
		value: string | undefined,
		field: SanitizedStringCache,
	): string | undefined => {
		if (value === undefined) return undefined;
		if (value === field.raw) {
			ctx.sanitizationMetrics.cacheHits++;
			return field.safe;
		}
		const incremental =
			field.raw !== undefined &&
			field.safe !== undefined &&
			value.startsWith(field.raw) &&
			!/[\r\x1b\x80-\x9f]/u.test(field.raw);
		const pending = incremental ? value.slice(field.raw?.length ?? 0) : value;
		ctx.sanitizationMetrics.scannedCharacters += pending.length;
		const safe = incremental
			? field.safe + sanitizeTerminalText(pending)
			: sanitizeTerminalText(pending);
		field.raw = value;
		field.safe = safe;
		return safe;
	};
	if (tool.args !== cache.argsSource) {
		cache.argsSource = tool.args;
		cache.argsSafe = sanitizeTerminalValue(tool.args);
	}
	return {
		...tool,
		tool: sanitizeTerminalText(tool.tool),
		tool_name: sanitizeTerminalText(tool.tool_name),
		tool_call_id: tool.tool_call_id
			? sanitizeTerminalText(tool.tool_call_id)
			: tool.tool_call_id,
		result: sanitizeString(tool.result, cache.result),
		partialResult: sanitizeString(tool.partialResult, cache.partialResult),
		streamOutput: sanitizeString(tool.streamOutput, cache.streamOutput),
		args: cache.argsSafe,
		// Subagent detail objects mutate in place as child events arrive, so
		// revalidate that structure rather than caching by object identity.
		details: sanitizeTerminalValue(tool.details),
	};
}

export function getSanitizationMetrics(ctx: RenderCtx): {
	cacheHits: number;
	scannedCharacters: number;
} {
	return { ...ctx.sanitizationMetrics };
}

export function collapsedToolPreview(tool: ToolExecution): string {
	if (
		tool.tool_name === "write_file" ||
		tool.tool_name === "write_file_append"
	) {
		return "";
	}
	const raw = tool.streamOutput || tool.result || "";
	if (!raw.trim()) return "";
	const firstLine = raw
		.split("\n")
		.map((line) => line.trim())
		.find(Boolean);
	if (!firstLine) return "";
	const preview = compactText(firstLine);
	const summary = toolSummary(tool);
	if (!tool.isError && preview === summary) return "";
	return clampLineToWidth(preview, 120);
}

export function renderPostEditDiagnostics(
	block: PostEditDiagnosticBlock,
	width: number,
): string[] {
	const count = block.diagnostics.length;
	const lines = [
		`${theme.fg("dim", "│ ")}${theme.fg("warning", "◆")} ${BOLD}${theme.fg("warning", "DIAGNOSTICS")}${RESET} ${DIM}${count} issue${count === 1 ? "" : "s"}${RESET}`,
		`${theme.fg("dim", "│ ")}${theme.fg("muted", block.file)}${RESET}`,
	];
	if (count === 0) {
		lines.push(
			`${theme.fg("dim", "│ ")}${DIM}Diagnostics were reported but could not be parsed.${RESET}`,
		);
		return lines;
	}
	for (const diagnostic of block.diagnostics) {
		const label = diagnostic.label ? ` ${diagnostic.label}` : "";
		lines.push(
			`${theme.fg("dim", "│ ")}${theme.fg("toolError", "×")} ${theme.fg("active", `${diagnostic.line}:${diagnostic.column}`)}${theme.fg("muted", label)}${RESET}`,
		);
		for (const messageLine of wrapText(diagnostic.message, Math.max(16, width - 6))) {
			lines.push(`${theme.fg("dim", "│   ")}${messageLine}${RESET}`);
		}
	}
	return lines;
}

export function detailSection(label: string, meta = ""): string {
	return `${theme.fg("active", "── ")}${BOLD}${label.toUpperCase()}${RESET}${meta ? `  ${DIM}${meta}${RESET}` : ""}`;
}

export function toolSummary(tool: ToolExecution): string {
	const args = tool.args || {};
	const path = stringArg(args, "path") || stringArg(args, "file_path");
	if (
		tool.tool_name === "write_file" ||
		tool.tool_name === "write_file_append"
	) {
		const content = writeFileContent(tool) || "";
		const lineCount = content ? content.split("\n").length : 0;
		const verb = tool.tool_name === "write_file_append" ? "appended" : "written";
		return `${lineCount} line${lineCount === 1 ? "" : "s"} ${verb}${
			tool.isComplete ? "" : " so far"
		}`;
	}
	if (tool.tool_name === "edit_file") {
		const editCount = Array.isArray(args.edits) ? args.edits.length : 1;
		return `${editCount} edit${editCount === 1 ? "" : "s"}`;
	}
	if (tool.tool_name === "bash") {
		return compactText(stringArg(args, "command") || "");
	}
	if (tool.tool_name === "read_file") {
		return path || "";
	}
	if (tool.tool_name === "rg_search") {
		return compactText(stringArg(args, "pattern") || "");
	}
	if (tool.tool_name.startsWith("mcp__")) {
		return [
			tool.tool_name.replace(/^mcp__/, "").replace(/__/g, "."),
			tool.result ? compactText(tool.result).slice(0, 80) : "",
		]
			.filter(Boolean)
			.join(" · ");
	}
	if (tool.tool_name === "spawn_agent") {
		return compactText(stringArg(args, "task") || "").slice(0, 80);
	}
	if (tool.tool_name === "spawn_agents") {
		const tasks = Array.isArray(args.tasks) ? args.tasks.length : 0;
		return tasks ? `${tasks} tasks` : "";
	}
	if (path) return path;
	const result = tool.result ?? tool.partialResult;
	return result ? compactText(result).slice(0, 80) : "";
}

export function toolDetailLines(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
): string[] {
	const args = tool.args || {};
	const lines: string[] = [];
	const result = tool.result ?? tool.partialResult;

	if (tool.tool_name === "spawn_agent") {
		return renderSubagentDetails(ctx, tool, width, expanded);
	}
	if (tool.tool_name === "spawn_agents") {
		return renderSubagentBatchDetails(ctx, tool, width, expanded);
	}

	if (tool.isError && result && isPermissionRejection(result)) {
		lines.push(...renderPermissionBlock(ctx, result, width));
		return lines;
	}

	if (
		tool.tool_name === "write_file" ||
		tool.tool_name === "write_file_append"
	) {
		lines.push(...renderWriteDetails(ctx, tool, width, expanded));
	} else if (tool.tool_name === "edit_file") {
		lines.push(...renderEditDetails(ctx, tool, width, expanded));
	} else if (tool.tool_name === "file_diff") {
		lines.push(...renderFileDiffDetails(ctx, tool, width));
	} else if (tool.tool_name === "bash") {
		lines.push(...renderBashDetails(ctx, tool, width));
	} else if (tool.tool_name.startsWith("mcp__")) {
		lines.push(...renderMcpDetails(ctx, tool, width));
	} else {
		const argText = JSON.stringify(args, null, 2);
		if (argText && argText !== "{}") {
			lines.push(detailSection("arguments"));
			lines.push(...argText.split("\n"));
		}
	}

	if (
		result &&
		![
			"write_file",
			"write_file_append",
			"edit_file",
			"file_diff",
			"bash",
		].includes(tool.tool_name) &&
		!tool.tool_name.startsWith("mcp__")
	) {
		lines.push(detailSection(tool.isError ? "error" : "result"));
		lines.push(...previewBlock(ctx, result, width));
	} else if (!result && !tool.isComplete) {
		lines.push(`${DIM}waiting for result...${RESET}`);
	}

	return lines;
}

/**
 * Live progress tally for a spawn_agents batch. Before completion the only
 * signal available is the `▶/✓/×` marker stream on streamOutput (the tool
 * only reports structured total/completed/failed once every task
 * finishes), so this is the single place that parses it — reused by both
 * the collapsed header and the expanded per-task breakdown.
 */
export function computeBatchTally(
	ctx: RenderCtx,
	tool: ToolExecution,
): {
	total: number;
	completed: number;
	failed: number;
	running: number;
	liveStatus: Map<number, "running" | "completed" | "failed">;
	taskElapsedMs: Map<number, number>;
} {
	const tasks = Array.isArray(tool.args?.tasks)
		? tool.args.tasks.filter(
				(task): task is Record<string, unknown> =>
					typeof task === "object" && task !== null,
			)
		: [];
	const details = tool.details ?? {};
	const liveStatus = new Map<number, "running" | "completed" | "failed">();
	for (const line of (tool.streamOutput ?? "").split("\n")) {
		const match = /^([▶✓×])\s+(\d+)\s+/.exec(line.trim());
		if (!match) continue;
		liveStatus.set(
			Number(match[2]),
			match[1] === "▶" ? "running" : match[1] === "✓" ? "completed" : "failed",
		);
	}

	const timingKey = tool.tool_call_id ?? "";
	let timing = ctx.batchTaskTiming.get(timingKey);
	if (!timing) {
		timing = new Map();
		ctx.batchTaskTiming.set(timingKey, timing);
	}
	const now = Date.now();
	const taskElapsedMs = new Map<number, number>();
	for (const [index, status] of liveStatus) {
		let entry = timing.get(index);
		if (!entry) {
			entry = { startedAt: now };
			timing.set(index, entry);
		}
		if (status !== "running" && entry.endedAt === undefined) {
			entry.endedAt = now;
		}
		taskElapsedMs.set(index, (entry.endedAt ?? now) - entry.startedAt);
	}

	const completed =
		typeof details.completed === "number" && Number.isFinite(details.completed)
			? Math.max(0, Math.trunc(details.completed))
			: [...liveStatus.values()].filter((s) => s === "completed").length;
	const failed =
		typeof details.failed === "number" && Number.isFinite(details.failed)
			? Math.max(0, Math.trunc(details.failed))
			: [...liveStatus.values()].filter((s) => s === "failed").length;
	const running = [...liveStatus.values()].filter(
		(status) => status === "running",
	).length;
	const reportedTotal =
		typeof details.total === "number" && Number.isFinite(details.total)
			? Math.max(0, Math.trunc(details.total))
			: 0;
	const observedTotal = liveStatus.size > 0 ? Math.max(...liveStatus.keys()) + 1 : 0;
	// Tool arguments can still be streaming when the first progress marker
	// arrives. Never render an impossible ratio such as "1/0", and also
	// defend against stale or malformed structured totals.
	const total = Math.max(
		reportedTotal,
		tasks.length,
		observedTotal,
		completed + failed + running,
	);
	return { total, completed, failed, running, liveStatus, taskElapsedMs };
}

export function writeFileContent(tool: ToolExecution): string | undefined {
	return (
		stringArg(tool.args || {}, "content") ??
		streamedStringArgLive(tool.partialResult, "content")
	);
}

// ── Shared low-level content-block helpers ──────────────────────────────────
// Used by both the per-tool-type detail renderers and the subagent renderers.

export function renderPermissionBlock(
	ctx: RenderCtx,
	result: string,
	width: number,
): string[] {
	const lines = [
		`${theme.fgRaw("warning")}${BOLD}permission / rejection${RESET}`,
		...previewBlock(ctx, result, width),
	];
	return lines;
}

export function renderMcpResultBlocks(
	ctx: RenderCtx,
	result: string,
	width: number,
): string[] {
	const parsed = parseJsonMaybe(result);
	if (!parsed) return previewBlock(ctx, result, width);
	const content =
		parsed && typeof parsed === "object"
			? (parsed as Record<string, unknown>).content
			: undefined;
	if (Array.isArray(content)) {
		const lines: string[] = [];
		content.forEach((item, index) => {
			const block = item as Record<string, unknown>;
			lines.push(
				`${DIM}block ${index + 1}: ${String(block.type || "content")}${RESET}`,
			);
			if (typeof block.text === "string") {
				lines.push(...previewBlock(ctx, block.text, width));
			} else {
				lines.push(...previewBlock(ctx, JSON.stringify(block, null, 2), width));
			}
		});
		return lines;
	}
	return previewBlock(ctx, JSON.stringify(parsed, null, 2), width);
}

export function renderDiffBlock(
	ctx: RenderCtx,
	diff: string,
	width: number,
	language?: string,
): string[] {
	if (!diff.trim()) return [`${DIM}(no diff)${RESET}`];
	const rawLines = truncateText(diff, ctx.maxMessageLength).split("\n");
	const lines: string[] = [];
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;

	for (const raw of rawLines) {
		const color = diffLineColor(raw);
		const content = raw.length ? raw.replace(/\t/g, "    ") : " ";

		// Keep the diff marker in its semantic color, but highlight the code
		// after it with the grammar selected from the edited file's path.
		if (
			(raw.startsWith("+") && !raw.startsWith("+++")) ||
			(raw.startsWith("-") && !raw.startsWith("---"))
		) {
			const prefix = raw[0];
			const codeText = raw.slice(1).replace(/\t/g, "    ");
			try {
				const highlighted = language
					? highlight(codeText, language)
					: highlightAuto(codeText);
				if (highlighted.value && highlighted.value !== codeText) {
					if (visibleWidth(content) <= width) {
						lines.push(
							`${bg}${color}${prefix}${RESET}${bg}${highlighted.value}${bgReset}`,
						);
						continue;
					}
				}
			} catch {
				// No highlighting available, fall through to plain rendering.
			}
		}

		if (visibleWidth(content) <= width) {
			lines.push(`${bg}${color}${content}${bgReset}`);
		} else {
			for (const wrapped of wrapText(content, width)) {
				lines.push(`${bg}${color}${wrapped}${bgReset}`);
			}
		}
	}

	return lines;
}

export function renderTerminalBlock(
	ctx: RenderCtx,
	text: string,
	width: number,
): string[] {
	if (!text) return [`${DIM}(no output)${RESET}`];
	const rawLines = truncateText(text, ctx.maxMessageLength).split("\n");
	const lines: string[] = [];
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;
	for (const raw of rawLines) {
		const content = raw.length ? raw.replace(/\t/g, "    ") : " ";
		const color = raw.startsWith("Error:")
			? theme.fgRaw("diffRemoved")
			: theme.fgRaw("terminalOutput");
		if (visibleWidth(content) <= width) {
			lines.push(`${bg}${color}${content}${bgReset}`);
		} else {
			for (const wrapped of wrapText(content, width)) {
				lines.push(`${bg}${color}${wrapped}${bgReset}`);
			}
		}
	}
	return lines;
}

export function previewBlock(
	ctx: RenderCtx,
	text: string,
	width: number,
	maxChars = ctx.maxMessageLength,
): string[] {
	if (!text) return [`${DIM}(empty)${RESET}`];
	const preview =
		text.length > maxChars ? withTruncationMarker(text.slice(0, maxChars)) : text;
	const rawLines = preview.split("\n");
	const lines: string[] = [];
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;
	let prevEmpty = false;
	for (const raw of rawLines) {
		const isEmpty = raw.length === 0;
		const formatted = isEmpty ? " " : raw.replace(/\t/g, "    ");
		if (isEmpty && prevEmpty) continue; // collapse consecutive blanks
		prevEmpty = isEmpty;
		if (visibleWidth(formatted) <= width) {
			lines.push(`${bg}${formatted}${bgReset}`);
		} else {
			for (const wrapped of wrapText(formatted, width)) {
				lines.push(`${bg}${wrapped}${bgReset}`);
			}
		}
	}
	return lines;
}
