// ── Ink TUI — Session Tree (flat list for MVP) ───────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import type { SessionInfo } from "../types";

interface SessionTreeProps {
	sessions: SessionInfo[];
	isActive: boolean;
	onClose: () => void;
}

export const SessionTree: React.FC<SessionTreeProps> = ({
	sessions,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const { index } = useOverlayInput({
		isActive,
		count: sessions.length,
		onSelect: () => onClose(),
		onClose,
	});

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={48}
		>
			<Text color={theme.fg.accent as string} bold>
				Session Tree
			</Text>
			{sessions.length === 0 ? (
				<Text color={theme.fg.muted as string}>no sessions yet</Text>
			) : (
				sessions.map((node, i) => (
					<Text
						key={node.id}
						color={
							i === index ? (theme.fg.selected as string) : (theme.fg.primary as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${node.name}  (${node.messageCount} msgs)`}
					</Text>
				))
			)}
		</Box>
	);
};
