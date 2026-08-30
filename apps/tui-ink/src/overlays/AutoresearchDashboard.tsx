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
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={44}
		>
			<Text color={theme.fg.accent as string} bold>
				Autoresearch
			</Text>
			<Text color={active ? (theme.fg.success as string) : (theme.fg.muted as string)}>
				{active ? "● active" : "○ inactive"}
				{`  ·  iteration ${iteration}`}
			</Text>
			{status ? (
				<Text color={theme.fg.primary as string} wrap="wrap">
					{status}
				</Text>
			) : null}
			<Text color={theme.fg.muted as string}>esc / ⏎ to close</Text>
		</Box>
	);
};
