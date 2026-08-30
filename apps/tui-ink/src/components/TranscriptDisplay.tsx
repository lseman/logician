// ── Ink TUI — Transcript Display (with scrollback) ────────────────────────────

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
	/** Number of turns to skip from the top (scroll offset). */
	scrollOffset?: number;
	/** Maximum number of turns to render at once. */
	maxVisibleTurns?: number;
	/** Whether there is content below the visible window. */
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
			lines += 1; // "Assistant" header
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
		// Walk forward accumulating lines until we pass scrollOffset
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
						<Text color={theme.fg.secondary} bold>
							{"Thinking"}
						</Text>
						<Text color={theme.fg.secondary} wrap="wrap">
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
				const statusColor = isError ? "error" : isComplete ? "success" : "accent";

				return (
					<Box flexDirection="column" marginBottom={1} key={`tool-${index}`}>
						<Text color={theme.fg[statusColor as keyof typeof theme.fg]} bold>
							{`⚙ ${toolName}${duration}`}
						</Text>
						{tool.streamOutput && !isComplete && (
							<Text color={theme.fg.secondary} wrap="truncate-end">
								{ellipsis(tool.streamOutput, 150)}
							</Text>
						)}
						{tool.result && (
							<Text color={theme.fg.secondary} wrap="truncate-end">
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
						? "error"
						: notice.level === "warn"
							? "warning"
							: notice.level === "success"
								? "success"
								: "info";

				return (
					<Box flexDirection="column" marginBottom={1} key={`notice-${index}`}>
						<Text color={theme.fg[noticeColor as keyof typeof theme.fg]} wrap="wrap">
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
						<Text color={theme.fg.primary} wrap="wrap">
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
						<Text color={theme.fg.accent} bold>
							{"You"}
						</Text>
					)}
					<Text color={isSystem ? theme.fg.secondary : theme.fg.primary} wrap="wrap">
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
					<Text color={theme.fg.primary} bold>
						{"Assistant"}
					</Text>
					<Box flexDirection="column">
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
				<Text color={theme.fg.muted} wrap="wrap">
					{`… ${renderedStartLines} lines above …`}
				</Text>
			)}

			{hasContent ? (
				<Box flexDirection="column">
					{visibleTurns.map((turn, index) => renderTurn(turn, startIdx + index))}
				</Box>
			) : (
				<Box justifyContent="center" alignItems="center" height={10}>
					<Text color={theme.fg.muted}>{"Logician TUI — Ready"}</Text>
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
