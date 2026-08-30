// ── Ink TUI — Transcript Display ──────────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import type { Turn, ThinkingDisplayMode, AssistantChunk } from "../types";
import { getCurrentTheme } from "../theme";
import { ellipsis } from "../utils";

interface TranscriptDisplayProps {
	turns: Turn[];
	thinkingMode: ThinkingDisplayMode;
	maxMessageLength?: number;
}

export const TranscriptDisplay: React.FC<TranscriptDisplayProps> = ({
	turns,
	thinkingMode,
	maxMessageLength,
}) => {
	const theme = getCurrentTheme();

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

			case "content":
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

	return (
		<Box flexGrow={1} flexDirection="column" overflow="hidden">
			{turns.length === 0 ? (
				<Box justifyContent="center" alignItems="center" height={10}>
					<Text color={theme.fg.muted}>{"Logician TUI — Ready"}</Text>
				</Box>
			) : (
				<Box flexDirection="column">{turns.map((turn, index) => renderTurn(turn, index))}</Box>
			)}
		</Box>
	);
};
