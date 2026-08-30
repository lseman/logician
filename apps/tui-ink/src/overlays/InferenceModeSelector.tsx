// ── Ink TUI — Inference Mode Selector ─────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import type { InferenceMode } from "../types";

interface InferenceModeSelectorProps {
	currentMode: InferenceMode;
	isActive: boolean;
	onSelect: (mode: InferenceMode) => void;
	onClose: () => void;
}

const MODES: InferenceMode[] = ["none", "deep", "research", "creative", "debug"];

export const InferenceModeSelector: React.FC<InferenceModeSelectorProps> = ({
	currentMode,
	isActive,
	onSelect,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const { index } = useOverlayInput({
		isActive,
		count: MODES.length,
		onSelect: i => onSelect(MODES[i]!),
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
				Inference Mode
			</Text>
			{MODES.map((mode, i) => (
				<Text
					key={mode}
					color={
						i === index ? (theme.fg.selected as string) : (theme.fg.primary as string)
					}
					bold={i === index}
				>
					{`${i === index ? "▸ " : "  "}${mode}${mode === currentMode ? "  ✓" : ""}`}
				</Text>
			))}
		</Box>
	);
};
