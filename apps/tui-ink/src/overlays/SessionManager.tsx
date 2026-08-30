// ── Ink TUI — Session Manager ─────────────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import type { TuiSessionSummary } from "@logician/log-runtime/sessions";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";

interface SessionManagerProps {
	sessions: TuiSessionSummary[];
	currentSessionId: string;
	isActive: boolean;
	onSelect: (sessionId: string) => void;
	onNew: () => void;
	onDelete: (sessionId: string) => void;
	onClose: () => void;
}

export const SessionManager: React.FC<SessionManagerProps> = ({
	sessions,
	currentSessionId,
	isActive,
	onSelect,
	onNew,
	onDelete,
	onClose,
}) => {
	const theme = getCurrentTheme();

	const { index } = useOverlayInput({
		isActive,
		count: sessions.length,
		onSelect: i => {
			const s = sessions[i];
			if (s) onSelect(s.id);
		},
		onClose,
		keys: {
			n: () => onNew(),
			d: i => {
				const s = sessions[i];
				if (s) onDelete(s.id);
			},
		},
	});

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={52}
		>
			<Text color={theme.fg.accent as string} bold>
				Sessions
			</Text>
			<Text color={theme.fg.muted as string}>
				↑↓ move · ⏎ open · n new · d delete · esc close
			</Text>
			{sessions.length === 0 ? (
				<Text color={theme.fg.muted as string}>no sessions yet</Text>
			) : (
				sessions.map((session, i) => (
					<Text
						key={session.id}
						color={
							i === index
								? (theme.fg.selected as string)
								: (theme.fg.primary as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${session.name}${
							session.id === currentSessionId ? "  (current)" : ""
						}`}
					</Text>
				))
			)}
		</Box>
	);
};
