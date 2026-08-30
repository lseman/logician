// ── Ink TUI — Thinking Level Selector ─────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import type { ThinkingLevel } from "../types";

interface ThinkingLevelSelectorProps {
	currentLevel: ThinkingLevel;
	isActive: boolean;
	onSelect: (level: ThinkingLevel) => void;
	onClose: () => void;
}

const LEVELS: ThinkingLevel[] = [
	"off",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
];

export const ThinkingLevelSelector: React.FC<ThinkingLevelSelectorProps> = ({
	currentLevel,
	isActive,
	onSelect,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const { index } = useOverlayInput({
		isActive,
		count: LEVELS.length,
		onSelect: i => onSelect(LEVELS[i]!),
		onClose,
	});

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={36}
		>
			<Text color={theme.fg.accent as string} bold>
				Thinking Level
			</Text>
			{LEVELS.map((level, i) => (
				<Text
					key={level}
					color={
						i === index ? (theme.fg.selected as string) : (theme.fg.primary as string)
					}
					bold={i === index}
				>
					{`${i === index ? "▸ " : "  "}${level}${level === currentLevel ? "  ✓" : ""}`}
				</Text>
			))}
		</Box>
	);
};
