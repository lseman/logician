// ── Ink TUI — Permission Popup ────────────────────────────────────────────────

import React from "react";
import { Box, Text, useInput } from "ink";
import { getCurrentTheme } from "../theme";
import { ellipsis } from "../utils";

type Decision = "allow" | "deny" | "always";

interface PermissionPopupProps {
	toolName: string;
	toolCallId: string;
	args?: unknown;
	isActive: boolean;
	onDecision: (decision: Decision) => void;
}

export const PermissionPopup: React.FC<PermissionPopupProps> = ({
	toolName,
	args,
	isActive,
	onDecision,
}) => {
	const theme = getCurrentTheme();

	useInput(
		(input, key) => {
			const c = input.toLowerCase();
			if (c === "a" || c === "y") onDecision("allow");
			else if (c === "s") onDecision("always");
			else if (c === "d" || c === "n" || key.escape) onDecision("deny");
		},
		{ isActive },
	);

	const preview = args ? ellipsis(JSON.stringify(args), 200) : "";

	return (
		<Box
			borderColor={theme.fg.warning}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={48}
		>
			<Text color={theme.fg.warning} bold>
				⚠ Permission required
			</Text>
			<Text color={theme.fg.text}>{`Tool: ${toolName}`}</Text>
			{preview ? (
				<Text color={theme.fg.muted} wrap="truncate-end">
					{preview}
				</Text>
			) : null}
			<Box flexDirection="row">
				<Text color={theme.fg.success} bold>
					{"[a] allow once  "}
				</Text>
				<Text color={theme.fg.info} bold>
					{"[s] always  "}
				</Text>
				<Text color={theme.fg.error} bold>
					{"[d] deny"}
				</Text>
			</Box>
		</Box>
	);
};
