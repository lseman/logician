// ── Ink TUI — Reasoner Selector (read-only status for MVP) ────────────────────

import React from "react";
import { Box, Text, useInput } from "ink";
import { getCurrentTheme } from "../theme";
import type { ReasonerStatus } from "../types";

interface ReasonerSelectorProps {
	reasoner: ReasonerStatus;
	isActive: boolean;
	onClose: () => void;
}

export const ReasonerSelector: React.FC<ReasonerSelectorProps> = ({
	reasoner,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();
	useInput((_input, key) => {
		if (key.escape || key.return) onClose();
	}, { isActive });

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={36}
		>
			<Text color={theme.fg.accent as string} bold>
				Reasoner
			</Text>
			<Text color={theme.fg.primary as string}>
				{`${reasoner.active ? "● " : "○ "}${reasoner.name}`}
			</Text>
			<Text color={theme.fg.muted as string}>esc / ⏎ to close</Text>
		</Box>
	);
};
