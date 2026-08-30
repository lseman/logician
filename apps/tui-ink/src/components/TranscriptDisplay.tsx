// ── Ink TUI — Transcript Display (with scrollback) ────────────────────────────
// Renders turns with the same semantic color tokens as the old TUI:
//   userLabel + userText for YOU messages, assistantText for LOGICIAN,
//   systemText for system messages, responseLabel before assistant content,
//   thinkingText/thinkingLabel for reasoning blocks, tool colors per status.

import React from "react";
import { Box, Text } from "ink";
import type { Turn, ThinkingDisplayMode, AssistantChunk } from "../types";
import { getCurrentTheme } from "../theme";
import { ellipsis } from "../utils";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface TranscriptDisplayProps {
	turns: Turn[];
	thinkingMode: ThinkingDisplayMode;
	maxMessageLength?: number;
	scrollOffset?: number;
	maxVisibleTurns?: number;
	hasNewOutputBelow?: boolean;
}

export const TranscriptDisplay: React.FC<TranscriptDisplayProps> = ({
	turns,
	thinkingMode,
	maxMessageLength,
	scrollOffset = 0,
	maxVisibleTurns = 60,
	hasNewOutputBelow = false,
}) => {
	const theme = getCurrentTheme();

	// ── Estimate line count for a turn (used by scroll logic) ──────────────
	const estimateLinesForTurn = (turn: Turn): number => {
		let lines = 0;
		if (turn.userMessage) lines += 2;
		if (turn.assistantMessage) {
			lines += 1; // "LOGICIAN" header
			for (const chunk of turn.assistantMessage.chunks) {
				switch (chunk.type) {
					case "thinking":
						lines += thinkingMode === "collapsed" ? 0 : (thinkingMode === "summary" ? 2 : 4);
						break;
					case "tool":
						lines += 1 + (chunk.tool?.streamOutput && !chunk.tool.isComplete ? 1 : 0) + (chunk.tool?.result ? 1 : 0);
						break;
					case "notice":
						lines += 1;
						break;
					default:
						lines += Math.max(1, Math.ceil((chunk.contentText?.length ?? 0) / 80));
				}
			}
		}
		return Math.max(1, lines);
	};

	// ── Compute visible turn range ────────────────────────────────────────
	const totalLines = turns.reduce((sum, t) => sum + estimateLinesForTurn(t), 0);
	let startIdx = 0;
	if (scrollOffset > 0 && totalLines > maxVisibleTurns) {
		let accumulated = 0;
		for (let i = 0; i < turns.length; i++) {
			accumulated += estimateLinesForTurn(turns[i]!);
			if (accumulated >= scrollOffset) {
				startIdx = i;
				break;
			}
		}
	}

	const visibleTurns = turns.slice(startIdx, startIdx + maxVisibleTurns);
	const renderedStartLines = (() => {
		let sum = 0;
		for (let i = 0; i < startIdx && i < turns.length; i++) {
			sum += estimateLinesForTurn(turns[i]!);
		}
		return sum;
	})();

	const renderChunk = (chunk: AssistantChunk, index: number): React.ReactNode => {
		switch (chunk.type) {
			case "thinking": {
				if (thinkingMode === "collapsed") return null;

				const text = chunk.contentText || "";
				const maxLength = thinkingMode === "summary" ? 200 : maxMessageLength || 2000;
				const displayText = text.length <= maxLength ? text : ellipsis(text, maxLength);

				return (
					<Box flexDirection="column" marginBottom={1} key={`thinking-${index}`}>
						<Text color={theme.fg.reasoningLabel} bold>
							{"Thinking"}
						</Text>
						<Text color={theme.fg.thinkingText} wrap="wrap">
							{displayText}
						</Text>
					</Box>
				);
			}

			case "tool": {
				const tool = chunk.tool;
				if (!tool) return null;

				const toolName = tool.tool_name;
				const isComplete = tool.isComplete;
				const isError = tool.isError;
				const duration = tool.durationMs ? ` ${tool.durationMs}ms` : "";

				// Semantic tool colors from theme
				const toolTitleColor = isError ? theme.fg.toolError : isComplete ? theme.fg.toolSuccess : theme.fg.toolRunning;
				const toolOutputColor = theme.fg.toolOutput;

				return (
					<Box flexDirection="column" marginBottom={1} key={`tool-${index}`}>
						<Text color={toolTitleColor} bold>
							{`⚙ ${toolName}${duration}`}
						</Text>
						{tool.streamOutput && !isComplete && (
							<Text color={toolOutputColor} wrap="truncate-end">
								{ellipsis(tool.streamOutput, 150)}
							</Text>
						)}
						{tool.result && (
							<Text color={toolOutputColor} wrap="truncate-end">
								{ellipsis(String(tool.result), maxMessageLength || 500)}
							</Text>
						)}
					</Box>
				);
			}

			case "notice": {
				const notice = chunk.notice;
				if (!notice) return null;

				const noticeColor =
					notice.level === "error"
						? theme.fg.error
						: notice.level === "warn"
							? theme.fg.warning
							: notice.level === "success"
								? theme.fg.success
								: theme.fg.info;

				return (
					<Box flexDirection="column" marginBottom={1} key={`notice-${index}`}>
						<Text color={noticeColor} wrap="wrap">
							{notice.text}
						</Text>
					</Box>
				);
			}

			case "content": {
				const text = chunk.contentText || "";
				if (!text) return null;
				return (
					<MarkdownRenderer
						key={`content-${index}`}
						markdown={text}
						maxLength={maxMessageLength || 4000}
					/>
				);
			}

			case "user":
			default: {
				const text = chunk.contentText || "";
				if (!text) return null;
				return (
					<Box flexDirection="column" marginBottom={1} key={`content-${index}`}>
						<Text color={theme.fg.userText} wrap="wrap">
							{ellipsis(text, maxMessageLength || 4000)}
						</Text>
					</Box>
				);
			}
		}
	};

	const renderTurn = (turn: Turn, index: number): React.ReactNode => {
		const nodes: React.ReactNode[] = [];

		if (turn.userMessage) {
			const content = turn.userMessage.content;
			const isSystem = content.startsWith("[System]");
			nodes.push(
				<Box flexDirection="column" marginBottom={1} key={`user-${index}`}>
					{!isSystem && (
						<Text color={theme.fg.userLabel} bold>
							{"YOU"}
						</Text>
					)}
					<Text color={isSystem ? theme.fg.systemText : theme.fg.userText} wrap="wrap">
						{ellipsis(
							isSystem ? content.replace(/^\[System\]\s*/, "") : content,
							maxMessageLength || 4000,
						)}
					</Text>
				</Box>,
			);
		}

		if (turn.assistantMessage) {
			nodes.push(
				<Box flexDirection="column" marginBottom={2} key={`assistant-${index}`}>
					<Text color={theme.fg.assistantText} bold>
						{"◆ LOGICIAN"}
					</Text>
					<Box flexDirection="column">
						{/* RESPONSE label before first content chunk (matches old TUI) */}
						{turn.assistantMessage.chunks.some(c => c.type === "content") && (
							<Text color={theme.fg.responseLabel} bold>
								{"RESPONSE"}
							</Text>
						)}
						{turn.assistantMessage.chunks.map((chunk, chunkIndex) =>
							renderChunk(chunk, chunkIndex),
						)}
						{turn.assistantMessage.chunks.length === 0 && (
							<Text color={theme.fg.muted}>{"…"}</Text>
						)}
					</Box>
				</Box>,
			);
		}

		if (nodes.length === 0) return null;
		return (
			<Box flexDirection="column" key={`turn-${index}`}>
				{nodes}
			</Box>
		);
	};

	const hasContent = visibleTurns.length > 0 || turns.length === 0;

	return (
		<Box flexDirection="column" flexGrow={1} overflow="hidden">
			{/* Spacer for scrolled-off content */}
			{renderedStartLines > 0 && (
				<Text color={theme.fg.dim} wrap="wrap">
					{`… ${renderedStartLines} lines above …`}
				</Text>
			)}

			{hasContent ? (
				<Box flexDirection="column">
					{visibleTurns.map((turn, index) => renderTurn(turn, startIdx + index))}
				</Box>
			) : (
				<Box flexDirection="column" justifyContent="center" alignItems="center">
					<Text color={theme.fg.accent} bold wrap="wrap">
						{"◆ LOGICIAN"}
					</Text>
					<Text color={theme.fg.muted} wrap="wrap">
						{"Your workspace, ready to reason."}
					</Text>
					<Box flexDirection="column" marginTop={1}>
						<Text color={theme.fg.header} bold wrap="wrap">{"QUICK START"}</Text>
						<Text color={theme.fg.accent} wrap="wrap">
							{"/  "}{"Browse commands"}
						</Text>
						<Text color={theme.fg.accent} wrap="wrap">
							{"@  "}{"Attach a file"}
						</Text>
						<Text color={theme.fg.muted} wrap="wrap">
							{"/sessions  "}{"Resume previous work"}
						</Text>
						<Text color={theme.fg.muted} wrap="wrap">
							{"/help      "}{"See keys and capabilities"}
						</Text>
					</Box>
				</Box>
			)}

			{/* New output indicator */}
			{hasNewOutputBelow && (
				<Box justifyContent="center">
					<Text color={theme.fg.accent} bold>
						{"↓ new output below"}
					</Text>
				</Box>
			)}
		</Box>
	);
};
