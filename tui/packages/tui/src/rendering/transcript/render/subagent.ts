// ── Transcript subagent/batch rendering ─────────────────────────────────────
// Renders spawn_agent / spawn_agents tool executions: live batch progress
// tallies, per-child-agent streamed output, and the chronological child
// thinking/response/tool flow.

import { clampLineToWidth, DIM, RESET } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import type {
	ChildChunk,
	ChildToolCall,
	ToolExecution,
} from "@logician/coding-agent/sessions";
import {
	compactText,
	formatDurationMs,
	parseJsonMaybe,
	stringArg,
	stripAcceptanceForDisplay,
	stripThinkTags,
} from "../text-utils.ts";
import { renderMarkdownLines } from "./markdown-table.ts";
import { renderThinkingChunk } from "./thinking.ts";
import { withTruncationMarker } from "./content.ts";
import {
	computeBatchTally,
	detailSection,
	previewBlock,
	type RenderCtx,
	renderTool,
} from "./tool.ts";

export function batchAgentStreams(tool: ToolExecution): Map<number, string> {
	const streams = new Map<number, string>();
	const stored = tool.details?.streamTranscript;
	const source = typeof stored === "string" ? stored : (tool.streamOutput ?? "");
	for (const line of source.split("\n")) {
		const match = /^↳\s+(\d+)\s+(.+)$/.exec(line);
		if (!match) continue;
		try {
			const delta = JSON.parse(match[2]) as unknown;
			if (typeof delta !== "string") continue;
			const index = Number(match[1]);
			streams.set(index, (streams.get(index) ?? "") + delta);
		} catch {
			// Ignore an incomplete update line while it is still streaming.
		}
	}
	return streams;
}

export function renderSubagentText(
	text: string,
	width: number,
	streaming: boolean,
	ctx: RenderCtx,
	expanded = ctx.toolsExpanded,
): string[] {
	const visibleText = stripAcceptanceForDisplay(text);
	if (!visibleText) return [];
	const markdown =
		!expanded && visibleText.length > 800
			? withTruncationMarker(visibleText.slice(0, 800))
			: visibleText;
	return renderMarkdownLines(
		markdown,
		Math.max(16, width),
		streaming,
		theme.fg("assistantText", ""),
	);
}

export function distinctSubagentOutputs(
	liveText: string,
	finalText: string,
): string[] {
	const live = stripAcceptanceForDisplay(liveText).trim();
	const final = stripAcceptanceForDisplay(finalText).trim();
	if (!live) return final ? [final] : [];
	if (!final || live === final || live.includes(final)) return [live];
	return [live, final];
}

export function childToolExecution(call: ChildToolCall): ToolExecution {
	const parsedArgs = parseJsonMaybe(call.args);
	const args =
		parsedArgs && typeof parsedArgs === "object" && !Array.isArray(parsedArgs)
			? (parsedArgs as Record<string, unknown>)
			: call.args
				? { input: call.args }
				: {};
	const status = call.status ?? (call.isError ? "failed" : "completed");
	return {
		tool: call.toolName,
		tool_name: call.toolName,
		tool_call_id: call.toolCallId,
		args,
		result: call.resultPreview,
		isError: status === "failed" || call.isError === true,
		isComplete: status !== "running",
	};
}

/**
 * Render a child agent's stream using the same chronological model as the
 * parent transcript: thinking → response → tool → response, in arrival
 * order. Tool rows stay where they were called instead of being collected
 * into a separate activity section.
 */
export function renderSubagentFlow(
	chunks: ChildChunk[],
	width: number,
	showAgent: boolean,
	ctx: RenderCtx,
	expanded = ctx.toolsExpanded,
): string[] {
	const lines: string[] = [];
	const agentIds = [
		...new Set(chunks.map((chunk) => chunk.agentId).filter(Boolean)),
	];
	const plural = showAgent || agentIds.length > 1;
	const runLabel = plural
		? `SUBAGENTS · ${agentIds.length || "?"} CHILD RUNS`
		: `SUBAGENT${agentIds[0] ? ` · ${agentIds[0]}` : ""}`;
	lines.push(`${theme.fg("active", `╭─ ${runLabel}`)}${RESET}`);
	let contentBuffer = "";
	let contentAgent = "";
	let lastAgent = "";
	let lastWasThinking = false;

	const showAgentBoundary = (agentId: string) => {
		if (!showAgent || !agentId || agentId === lastAgent) return;
		lines.push(`${theme.fg("active", `◇ CHILD · ${agentId}`)}${RESET}`);
		lastAgent = agentId;
	};
	const flushContent = () => {
		if (!contentBuffer) return;
		showAgentBoundary(contentAgent);
		if (lastWasThinking) {
			lines.push(`${theme.fgRaw("separator")}${DIM}─── response ───${RESET}`);
		}
		const visible = stripAcceptanceForDisplay(stripThinkTags(contentBuffer));
		for (const line of renderMarkdownLines(visible, Math.max(16, width), true)) {
			lines.push(line);
		}
		contentBuffer = "";
		contentAgent = "";
		lastWasThinking = false;
	};

	for (const chunk of chunks) {
		if (chunk.type === "content") {
			if (contentBuffer && contentAgent !== chunk.agentId) {
				flushContent();
			}
			contentAgent = chunk.agentId;
			contentBuffer += chunk.contentText ?? "";
			continue;
		}

		flushContent();
		showAgentBoundary(chunk.agentId);
		if (chunk.type === "thinking") {
			const thinkingLines = renderThinkingChunk(
				{
					seq: chunk.seq,
					type: "thinking",
					contentText: chunk.contentText,
					isComplete: chunk.isComplete,
				},
				!chunk.isComplete,
				ctx.thinkingMode,
				ctx.currentWidth,
			);
			lines.push(...thinkingLines);
			lastWasThinking = thinkingLines.length > 0;
			continue;
		}
		if (chunk.type === "tool" && chunk.tool) {
			lines.push(
				...renderTool(ctx, childToolExecution(chunk.tool), Math.max(20, width), expanded),
			);
			lastWasThinking = false;
		}
	}
	flushContent();
	lines.push(`${theme.fg("active", "╰─ RETURN TO PARENT")}${RESET}`);
	return lines;
}

export function renderSubagentBatchDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
): string[] {
	const tasks = Array.isArray(tool.args?.tasks)
		? tool.args.tasks.filter(
				(task): task is Record<string, unknown> =>
					typeof task === "object" && task !== null,
			)
		: [];
	const details = tool.details ?? {};
	const results = Array.isArray(details.results)
		? details.results.filter(
				(result): result is Record<string, unknown> =>
					typeof result === "object" && result !== null,
			)
		: [];
	const resultByIndex = new Map(results.map((result) => [Number(result.index), result]));
	const { liveStatus, taskElapsedMs } = computeBatchTally(ctx, tool);
	const agentStreams = batchAgentStreams(tool);
	const childChunks = Array.isArray(details.childChunks)
		? (details.childChunks as ChildChunk[])
		: [];
	const hasOrderedFlow = childChunks.length > 0;
	// The N/M tally already appears in the collapsed header row above this
	// detail block — repeating it here as its own line was pure duplication.
	const lines: string[] = [];

	for (let index = 0; index < tasks.length; index++) {
		const task = tasks[index];
		const result = resultByIndex.get(index);
		const resultError = result?.isError === true;
		const state = result
			? resultError
				? "failed"
				: "completed"
			: (liveStatus.get(index) ?? "queued");
		const icon =
			state === "failed"
				? theme.fg("toolError", "×")
				: state === "completed"
					? theme.fg("toolSuccess", "✓")
					: state === "running"
						? theme.fg("toolRunning", ctx.spinnerFrame())
						: theme.fg("dim", "·");
		const agent =
			typeof task.agent === "string" && task.agent ? task.agent : "general";
		const taskText =
			typeof task.task === "string"
				? compactText(task.task).slice(0, 100)
				: `Task ${index + 1}`;
		const elapsedMs = taskElapsedMs.get(index);
		const elapsed =
			elapsedMs !== undefined ? ` ${DIM}${formatDurationMs(elapsedMs)}${RESET}` : "";
		const queuedTag = state === "queued" ? ` ${DIM}queued${RESET}` : "";
		lines.push(
			clampLineToWidth(
				`${icon} ${theme.fg("active", `${index + 1}. ${agent}`)} ${DIM}${taskText}${RESET}${queuedTag}${elapsed}`,
				Math.max(20, width),
			),
		);
		if (expanded && !hasOrderedFlow) {
			const liveText = agentStreams.get(index) ?? "";
			const resultText = typeof result?.content === "string" ? result.content : "";
			const texts = distinctSubagentOutputs(liveText, resultText);
			for (const text of texts) {
				for (const preview of renderSubagentText(
					text,
					Math.max(16, width - 4),
					!result,
					ctx,
					expanded,
				)) {
					lines.push(`  ${preview}`);
				}
			}
		}
	}
	if (hasOrderedFlow) {
		lines.push(...renderSubagentFlow(childChunks, width, true, ctx, expanded));
	} else {
		const childToolCalls = details.childToolCalls as ChildToolCall[] | undefined;
		lines.push(
			...renderSubagentActivity(
				childToolCalls,
				width,
				ctx,
				expanded ? Number.POSITIVE_INFINITY : 4,
				true,
				true,
				expanded,
			),
		);
	}
	return lines;
}

export function renderSubagentDetails(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
): string[] {
	const lines: string[] = [];
	const args = tool.args || {};
	const details = tool.details || {};
	const metrics =
		details.metrics && typeof details.metrics === "object"
			? (details.metrics as Record<string, unknown>)
			: {};
	const liveElapsedMs =
		!tool.isComplete && tool.startedAt !== undefined
			? Date.now() - tool.startedAt
			: undefined;
	const metadata = [
		typeof metrics.turns === "number" ? `${metrics.turns} turn(s)` : "",
		typeof metrics.toolCalls === "number" ? `${metrics.toolCalls} tool call(s)` : "",
		typeof metrics.durationMs === "number"
			? formatDurationMs(metrics.durationMs)
			: liveElapsedMs !== undefined
				? `${formatDurationMs(liveElapsedMs)} elapsed`
				: "",
	]
		.filter(Boolean)
		.join(" · ");
	if (metadata) {
		lines.push(`${DIM}${metadata}${RESET}`);
	}

	const branch = typeof details.branch === "string" ? details.branch : "";
	const commit = typeof details.commit === "string" ? details.commit : "";
	if (branch || commit) {
		lines.push(
			`${DIM}${[branch && `branch ${branch}`, commit && `commit ${commit.slice(0, 12)}`].filter(Boolean).join(" · ")}${RESET}`,
		);
	}

	const task = stringArg(args, "task");
	if (task) {
		for (const line of previewBlock(ctx, task, Math.max(16, width - 4), 800)) {
			lines.push(`${theme.fg("dim", "→ ")}${line}`);
		}
	}

	const childChunks = Array.isArray(details.childChunks)
		? (details.childChunks as ChildChunk[])
		: [];
	if (childChunks.length > 0) {
		lines.push(...renderSubagentFlow(childChunks, width, false, ctx, expanded));
	} else {
		const childToolCalls = details.childToolCalls as ChildToolCall[] | undefined;
		lines.push(
			...renderSubagentActivity(
				childToolCalls,
				width,
				ctx,
				expanded ? Number.POSITIVE_INFINITY : 4,
				false,
				false,
				expanded,
			),
		);
	}

	const storedTranscript =
		typeof details.streamTranscript === "string" ? details.streamTranscript : "";
	const liveOutput = tool.isComplete ? storedTranscript : (tool.streamOutput ?? "");
	const finalOutput = tool.isComplete ? (tool.result ?? "") : "";
	const orderedContent = childChunks
		.filter((chunk) => chunk.type === "content")
		.map((chunk) => chunk.contentText ?? "")
		.join("");
	const outputs =
		expanded && childChunks.length === 0
			? distinctSubagentOutputs(liveOutput, finalOutput)
			: !expanded && tool.isComplete && finalOutput && !orderedContent.includes(finalOutput)
				? [finalOutput]
				: [];
	if (outputs.length > 0) {
		// Ctrl+O is the explicit full-detail view. Keep collapsed tool rows
		// compact, but never discard child-agent progress or the final report here.
		for (const output of outputs) {
			for (const line of renderSubagentText(
				output,
				Math.max(16, width - 4),
				!tool.isComplete,
				ctx,
				expanded,
			)) {
				lines.push(`  ${line}`);
			}
		}
	} else if (!tool.isComplete && expanded) {
		lines.push(`${theme.fg("dim", "  waiting for agent output…")}${RESET}`);
	}

	return lines;
}

export function renderSubagentActivity(
	calls: ChildToolCall[] | undefined,
	width: number,
	ctx: RenderCtx,
	limit = Number.POSITIVE_INFINITY,
	showHeading = true,
	showAgent = true,
	expanded = ctx.toolsExpanded,
): string[] {
	if (!calls?.length) return [];
	const visible = calls.slice(-limit);
	const hidden = calls.length - visible.length;
	const lines = showHeading
		? [
				detailSection(
					"activity",
					`${calls.length} tool call${calls.length === 1 ? "" : "s"}${hidden ? ` · latest ${visible.length}` : ""}`,
				),
			]
		: hidden
			? [`  ${theme.fg("dim", `⋯ ${hidden} earlier tool call${hidden === 1 ? "" : "s"} hidden`)}`]
			: [];
	const bg = theme.bg("mdCodeBlockBg", "");
	for (const call of visible) {
		const status = call.status ?? (call.isError ? "failed" : "completed");
		const icon =
			status === "failed"
				? theme.fg("toolError", "×")
				: status === "running"
					? theme.fg("toolRunning", ctx.spinnerFrame())
					: theme.fg("toolSuccess", "✓");
		const summary = subagentCallSummary(call.args);
		const row = [
			`${icon} ${theme.fg("toolTitle", call.toolName)}`,
			showAgent && call.agentId ? `${DIM}${call.agentId}${RESET}` : "",
			summary ? `${DIM}${summary}${RESET}` : "",
		]
			.filter(Boolean)
			.join(` ${DIM}·${RESET} `);
		lines.push(`${bg}${clampLineToWidth(row, Math.max(20, width))}${RESET}`);
		if (expanded && call.resultPreview) {
			const result = compactText(call.resultPreview);
			lines.push(
				`${bg}${DIM}  └ ${clampLineToWidth(result, Math.max(16, width - 4))}${RESET}`,
			);
		}
	}
	return lines;
}

export function subagentCallSummary(raw: string): string {
	const text = raw.trim();
	if (!text || text === "{}") return "";
	const parsed = parseJsonMaybe(text);
	if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
		const args = parsed as Record<string, unknown>;
		for (const key of ["path", "file_path", "pattern", "command", "query"]) {
			if (typeof args[key] === "string") {
				return `${key}=${compactText(args[key] as string).slice(0, 96)}`;
			}
		}
	}
	return compactText(text).slice(0, 100);
}
