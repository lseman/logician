// ── Ink TUI — Queue Manager ───────────────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import type { SteerMessage } from "../types";
import { ellipsis } from "../utils";

interface QueueManagerProps {
	bridge: AgentRuntime;
	messages: SteerMessage[];
	isActive: boolean;
	onClose: () => void;
}

export const QueueManager: React.FC<QueueManagerProps> = ({
	bridge,
	messages,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();

	const { index } = useOverlayInput({
		isActive,
		count: messages.length,
		onSelect: () => {
			bridge.flushSteeringNow?.();
			onClose();
		},
		onClose,
		keys: {
			d: i => {
				try {
					bridge.dropQueuedMessage?.(i);
				} catch {
					/* ignore */
				}
			},
		},
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
				{`Steer Queue (${messages.length})`}
			</Text>
			<Text color={theme.fg.muted}>⏎ flush now · d drop · esc close</Text>
			{messages.length === 0 ? (
				<Text color={theme.fg.muted}>queue is empty</Text>
			) : (
				messages.map((msg, i) => (
					<Text
						key={msg.id}
						color={
							i === index ? (theme.fg.selected as string) : (theme.fg.text as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${ellipsis(msg.message, 60)}`}
					</Text>
				))
			)}
		</Box>
	);
};
