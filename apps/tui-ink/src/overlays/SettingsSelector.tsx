// ── Ink TUI — Settings Selector ───────────────────────────────────────────────
// Enter cycles the highlighted setting to its next value and applies it.

import React from "react";
import { Box, Text } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";
import type { TuiState } from "../state";
import type { AppConfig, InferenceMode, ThinkingLevel } from "../types";

interface SettingsSelectorProps {
	state: TuiState;
	config: AppConfig;
	bridge: AgentRuntime;
	isActive: boolean;
	onClose: () => void;
}

function next<T>(list: readonly T[], current: T): T {
	const i = list.indexOf(current);
	return list[(i + 1) % list.length]!;
}

const THINKING: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const INFERENCE: InferenceMode[] = ["none", "deep", "research", "creative", "debug"];
const DISPLAY = ["collapsed", "summary", "expanded"] as const;
const PERMISSION = ["ask", "acceptEdits", "acceptAll"] as const;
const PROFILE = ["minimal", "autonomous"] as const;

export const SettingsSelector: React.FC<SettingsSelectorProps> = ({
	state,
	bridge,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();

	const rows: Array<{ label: string; value: string; cycle: () => void }> = [
		{
			label: "Thinking level",
			value: state.thinkingLevel,
			cycle: () => {
				const v = next(THINKING, state.thinkingLevel);
				state.setThinkingLevel(v);
				bridge.updateSettings({ thinkingLevel: v as never });
			},
		},
		{
			label: "Inference mode",
			value: state.inferenceMode,
			cycle: () => {
				const v = next(INFERENCE, state.inferenceMode);
				state.setInferenceMode(v);
				bridge.updateSettings({ inferenceMode: v as never });
			},
		},
		{
			label: "Workflow mode",
			value: state.workflowMode,
			cycle: () => state.setWorkflowMode(state.workflowMode === "plan" ? "act" : "plan"),
		},
		{
			label: "Execution profile",
			value: state.executionProfile,
			cycle: () => {
				const v = next(PROFILE, state.executionProfile);
				state.setExecutionProfile(v);
				bridge.updateSettings({ executionProfile: v as never });
			},
		},
		{
			label: "Permission mode",
			value: state.permissionMode,
			cycle: () => {
				const v = next(PERMISSION, state.permissionMode);
				state.setPermissionMode(v);
				bridge.setPermissionMode(v as never);
			},
		},
		{
			label: "Thinking display",
			value: state.thinkingDisplayMode,
			cycle: () => state.setThinkingDisplayMode(next(DISPLAY, state.thinkingDisplayMode)),
		},
	];

	const { index } = useOverlayInput({
		isActive,
		count: rows.length,
		onSelect: i => rows[i]?.cycle(),
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
				Settings
			</Text>
			<Text color={theme.fg.muted as string}>↑↓ move · ⏎ cycle value · esc close</Text>
			{rows.map((row, i) => (
				<Box key={row.label} flexDirection="row">
					<Text
						color={
							i === index
								? (theme.fg.selected as string)
								: (theme.fg.secondary as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${row.label}: `}
					</Text>
					<Text color={theme.fg.primary as string} bold={i === index}>
						{row.value}
					</Text>
				</Box>
			))}
		</Box>
	);
};
