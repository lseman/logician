// ── Ink TUI — Slash Command Popup ─────────────────────────────────────────────

import React, { useMemo } from "react";
import { Box, Text } from "ink";
import type { SlashCommandDef } from "@logician/log-runtime/commands";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import { filterCommands } from "../slash";

interface SlashPopupProps {
	commands: SlashCommandDef[];
	query: string;
	isActive: boolean;
	onSelect: (command: SlashCommandDef) => void;
	onClose: () => void;
}

const MAX_VISIBLE = 10;

export const SlashPopup: React.FC<SlashPopupProps> = ({
	commands,
	query,
	isActive,
	onSelect,
	onClose,
}) => {
	const theme = getCurrentTheme();

	const filtered = useMemo(
		() => filterCommands(commands, query).slice(0, MAX_VISIBLE),
		[commands, query],
	);

	const { index } = useOverlayInput({
		isActive,
		count: filtered.length,
		onSelect: i => {
			const cmd = filtered[i];
			if (cmd) onSelect(cmd);
		},
		onClose,
	});

	return (
		<Box
			borderColor={theme.fg.border}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={48}
		>
			<Text color={theme.fg.header} bold>
				{`/${query}`}
			</Text>
			{filtered.length === 0 ? (
				<Text color={theme.fg.muted}>no matching commands</Text>
			) : (
				filtered.map((cmd, i) => (
					<Box key={cmd.command} flexDirection="row">
						<Text
							color={
								i === index
									? (theme.fg.selected as string)
									: (theme.fg.text as string)
							}
							bold={i === index}
						>
							{`${i === index ? "▸ " : "  "}${cmd.command}`}
						</Text>
						<Text color={theme.fg.muted}>
							{cmd.argHint ? ` ${cmd.argHint}` : ""}
							{`  ${cmd.description}`}
						</Text>
					</Box>
				))
			)}
		</Box>
	);
};
