// ── Transcript subagent/batch rendering ─────────────────────────────────────
// Renders spawn_agent / spawn_agents tool executions: live batch progress
// tallies, per-child-agent streamed output, and the chronological child
// thinking/response/tool flow.

import type {
	ChildChunk,
	ChildToolCall,
	ToolExecution,
} from "@logician/log-runtime/sessions";
import { clampLineToWidth, DIM, RESET } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import {
	compactText,
	formatDurationMs,
	parseJsonMaybe,
	stringArg,
	stripAcceptanceForDisplay,
	stripThinkTags,
} from "../text-utils.ts";
import { withTruncationMarker } from "./content.ts";
import { renderMarkdownLines } from "./markdown-table.ts";
import { renderThinkingChunk } from "./thinking.ts";
import {
	computeBatchTally,
	detailSection,
	previewBlock,
	type RenderCtx,
	renderTool,
} from "./tool.ts";

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
	if (!final) return live ? [live] : [];
	if (!live || live === final) return [final];
	// `live` may be a fuller transcript that already contains `final` as a
	// substring (e.g. a full stream transcript vs. just the final report) —
	// keep the richer one. Or `live` may be a stale prefix that hasn't caught
	// up to `final` yet (task just finished) — `final` alone is authoritative
	// there, since showing both would repeat the same text back to back.
	if (live.includes(final)) return [live];
	if (final.includes(live)) return [final];
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
	parentKey = "",
	includeContent = true,
	showFrame = true,
): string[] {
	const lines: string[] = [];
	const hitRegions = parentKey
		? (ctx._taskHitRegions = ctx._taskHitRegions ?? [])
		: undefined;
	if (showFrame) {
		const agentIds = [
			...new Set(chunks.map(chunk => chunk.agentId).filter(Boolean)),
		];
		const plural = showAgent || agentIds.length > 1;
		const runLabel = plural
			? `SUBAGENTS · ${agentIds.length || "?"} CHILD RUNS`
			: `SUBAGENT${agentIds[0] ? ` · ${agentIds[0]}` : ""}`;
		lines.push(`${theme.fg("active", `╭─ ${runLabel}`)}${RESET}`);
	}
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
		for (const line of renderMarkdownLines(
			visible,
			Math.max(16, width),
			true,
		)) {
			lines.push(line);
		}
		contentBuffer = "";
		contentAgent = "";
		lastWasThinking = false;
	};

	for (const chunk of chunks) {
		if (chunk.type === "content") {
			if (!includeContent) continue;
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
			const childToolCallId = chunk.tool.toolCallId;
			const childExpanded =
				parentKey && childToolCallId
					? (ctx.isChildToolExpanded?.(parentKey, childToolCallId) ?? expanded)
					: expanded;
			const regionStart = lines.length;
			lines.push(
				...renderTool(
					ctx,
					childToolExecution(chunk.tool),
					Math.max(20, width),
					childExpanded,
				),
			);
			if (hitRegions && childToolCallId) {
				hitRegions.push({
					start: regionStart,
					end: regionStart + 1,
					key: `${parentKey}:child:${childToolCallId}`,
				});
			}
			lastWasThinking = false;
		}
	}
	flushContent();
	if (showFrame) {
		lines.push(`${theme.fg("active", "╰─ RETURN TO PARENT")}${RESET}`);
	}
	return lines;
}

/**
 * Chronological tool-call activity appended after the per-task rows once the
 * whole spawn_agents tool is expanded (Ctrl+O or row click). Response text is
 * intentionally excluded here — the per-task rows already show each task's
 * text, and duplicating it in the tail rendered two copies of the same
 * streamed text back to back.
 */
export function renderSubagentBatchActivityTail(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded: boolean,
): string[] {
	const details = tool.details ?? {};
	const childChunks = Array.isArray(details.childChunks)
		? (details.childChunks as ChildChunk[])
		: [];
	if (childChunks.length > 0) {
		return renderSubagentFlow(
			childChunks,
			width,
			false,
			ctx,
			expanded,
			tool.tool_call_id ?? "",
			false,
		);
	}
	const childToolCalls = details.childToolCalls as ChildToolCall[] | undefined;
	return renderSubagentActivity(
		childToolCalls,
		width,
		ctx,
		expanded ? Number.POSITIVE_INFINITY : 4,
		true,
		true,
		expanded,
	);
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
		typeof metrics.toolCalls === "number"
			? `${metrics.toolCalls} tool call(s)`
			: "",
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
		lines.push(
			...renderSubagentFlow(
				childChunks,
				width,
				false,
				ctx,
				expanded,
				tool.tool_call_id ?? "",
			),
		);
	} else {
		const childToolCalls = details.childToolCalls as
			| ChildToolCall[]
			| undefined;
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
		typeof details.streamTranscript === "string"
			? details.streamTranscript
			: "";
	const liveOutput = tool.isComplete ? storedTranscript : "";
	const finalOutput = tool.isComplete ? (tool.result ?? "") : "";
	const orderedContent = childChunks
		.filter(chunk => chunk.type === "content")
		.map(chunk => chunk.contentText ?? "")
		.join("");
	const outputs =
		childChunks.length === 0
			? distinctSubagentOutputs(liveOutput, finalOutput)
			: tool.isComplete && finalOutput && !orderedContent.includes(finalOutput)
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
			? [
					`  ${theme.fg("dim", `⋯ ${hidden} earlier tool call${hidden === 1 ? "" : "s"} hidden`)}`,
				]
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

/**
 * Render live streaming output for a single (spawn_agent) subagent that is
 * still running. Called from the collapsed path so the user can follow
 * progress without manually expanding.
 *
 * Uses `childChunks`, the canonical ordered thinking/content/tool stream.
 */
export function renderSubagentLiveOutput(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
): string[] {
	// Prefer childChunks (ordered, chronological streaming).
	const childChunks = Array.isArray(tool.details?.childChunks)
		? (tool.details.childChunks as ChildChunk[])
		: [];
	if (childChunks.length > 0) {
		const flowLines = renderSubagentFlow(childChunks, width, false, ctx, false);
		// Drop the footer while still running.
		if (!tool.isComplete) {
			flowLines.pop();
		}
		return flowLines;
	}

	return [];
}

/**
 * Render a compact collapsed view for spawn_agents: one line per task
 * showing the task number, agent, description, and status.
 *
 * Each task card is clickable to expand/collapse.  When `expanded`
 * is true, every task's detail lines are shown by default.  When
 * false, only individually expanded tasks (via click) show their
 * detail beneath the card.  Hit regions are registered via
 * `ctx._taskHitRegions` so clicks are forwarded to the transcript
 * display.
 */
export function renderSubagentBatchCollapsed(
	ctx: RenderCtx,
	tool: ToolExecution,
	width: number,
	expanded = false,
): string[] {
	const lines: string[] = [];
	const hitRegions = (ctx._taskHitRegions = ctx._taskHitRegions ?? []);
	const toolCallId = tool.tool_call_id ?? "";

	const tasks = Array.isArray(tool.args?.tasks)
		? tool.args.tasks.filter(
				(task): task is Record<string, unknown> =>
					typeof task === "object" && task !== null,
			)
		: [];
	const { liveStatus, taskElapsedMs } = computeBatchTally(ctx, tool);
	const results = Array.isArray(tool.details?.results)
		? tool.details.results.filter(
				(r): r is Record<string, unknown> =>
					typeof r === "object" && r !== null,
			)
		: [];
	const resultByIndex = new Map(results.map(r => [Number(r.index), r]));

	for (let index = 0; index < tasks.length; index++) {
		const task = tasks[index];
		const result = resultByIndex.get(index);

		// Determine task state.
		const state = result
			? result.isError === true
				? "failed"
				: "completed"
			: (liveStatus.get(index) ?? "queued");

		// Status icon (mirrors renderSubagentBatchDetails's icon logic).
		const icon =
			state === "failed"
				? theme.fg("toolError", "×")
				: state === "completed"
					? theme.fg("toolSuccess", "✓")
					: state === "running"
						? theme.fg("toolRunning", ctx.spinnerFrame())
						: theme.fg("dim", "·");

		// Build task line: "icon N. agent description"
		const agent =
			typeof task.agent === "string" && task.agent ? task.agent : "general";
		const taskText =
			typeof task.task === "string"
				? compactText(task.task).slice(0, 120)
				: `Task ${index + 1}`;

		const elapsedMs = taskElapsedMs.get(index);
		const elapsed =
			elapsedMs !== undefined
				? ` ${DIM}${formatDurationMs(elapsedMs)}${RESET}`
				: "";
		const queuedTag = state === "queued" ? ` ${DIM}queued${RESET}` : "";
		const line =
			"  " +
			icon +
			" " +
			theme.fg("active", `${index + 1}. ${agent}`) +
			" " +
			DIM +
			taskText +
			RESET +
			queuedTag +
			elapsed;
		const maxLine = Math.max(20, width - 4);
		lines.push(clampLineToWidth(line, maxLine));

		// Register hit region for per-task expand/collapse.
		if (hitRegions) {
			hitRegions.push({
				start: lines.length - 1,
				end: lines.length,
				key: `${toolCallId}:task:${index}`,
			});
		}

		// Expanded detail for this task: show when `expanded` is true or when
		// this individual card has been clicked to expand. childChunks now
		// carry their own taskIndex (threaded through subagent_start/_event),
		// so this task's thinking/content/tool-call flow can be filtered out
		// reliably instead of guessing agentId-to-task-index correspondence.
		if (expanded || ctx.isAgentExpanded?.(toolCallId, index)) {
			const allChunks = Array.isArray(tool.details?.childChunks)
				? (tool.details.childChunks as ChildChunk[])
				: [];
			const taskChunks = allChunks.filter(c => c.taskIndex === index);
			if (taskChunks.length > 0) {
				const regionBase = lines.length;
				const hitRegionsBefore = hitRegions?.length ?? 0;
				const flowLines = renderSubagentFlow(
					taskChunks,
					Math.max(16, width - 4),
					false,
					ctx,
					true,
					`${toolCallId}:task:${index}`,
					true,
					// Show header/footer frame so the child agent ID is visible
					// (e.g. "╭─ SUBAGENT · agent_3") even inside spawn_agents batches.
					true,
				);
				for (const fl of flowLines) {
					lines.push(`  ${fl}`);
				}
				// renderSubagentFlow pushed child-tool hit regions relative to its
				// own local `lines` (starting at 0) — translate only the entries it
				// just appended into this function's own `lines` coordinate space.
				if (hitRegions) {
					for (let i = hitRegionsBefore; i < hitRegions.length; i++) {
						hitRegions[i] = {
							...hitRegions[i],
							start: hitRegions[i].start + regionBase,
							end: hitRegions[i].end + regionBase,
						};
					}
				}
			} else {
				const resultText =
					result && typeof result.content === "string" ? result.content : "";
				if (resultText) {
					for (const fl of renderSubagentText(
						resultText,
						Math.max(16, width - 4),
						false,
						ctx,
						true,
					)) {
						lines.push(`  ${fl}`);
					}
				} else if (!tool.isComplete) {
					lines.push(`  ${theme.fg("dim", "waiting for output...")}${RESET}`);
				}
			}
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
