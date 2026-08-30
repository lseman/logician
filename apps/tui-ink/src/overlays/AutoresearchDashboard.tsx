// ── Ink TUI — Autoresearch Dashboard ──────────────────────────────────────────

import React from "react";
import { Box, Text, useInput } from "ink";
import { getCurrentTheme } from "../theme";

interface AutoresearchDashboardProps {
	active: boolean;
	status?: string;
	iteration?: number;
	isActive: boolean;
	onClose: () => void;
}

export const AutoresearchDashboard: React.FC<AutoresearchDashboardProps> = ({
	active,
	status,
	iteration = 0,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();
	useInput((_i, key) => {
		if (key.escape || key.return) onClose();
	}, { isActive });

	return (
		<Box
			borderColor={theme.fg.border}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={44}
		>
			<Text color={theme.fg.header} bold>
				Autoresearch
			</Text>
			<Text color={active ? theme.fg.success : theme.fg.muted}>
				{active ? "● active" : "○ inactive"}
				{`  ·  iteration ${iteration}`}
			</Text>
			{status ? (
				<Text color={theme.fg.text} wrap="wrap">
					{status}
				</Text>
			) : null}
			<Text color={theme.fg.muted}>esc / ⏎ to close</Text>
		</Box>
	);
};
