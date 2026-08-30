// ── Ink TUI — Status Bar ──────────────────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import type { TuiState } from "../state";
import type { AppConfig } from "../types";
import { getCurrentTheme } from "../theme";
import {
	formatPhaseEmoji,
	formatPhaseColor,
	formatModelShort,
	formatTokenUsage,
	formatGitStatus,
} from "../utils";

interface StatusBarProps {
	state: TuiState;
	config: AppConfig;
	width: number;
}

export const StatusBar: React.FC<StatusBarProps> = ({ state, config, width }) => {
	const theme = getCurrentTheme();

	const phaseEmoji = formatPhaseEmoji(state.phase);
	const phaseColor = formatPhaseColor(state.phase);
	const modelShort = formatModelShort(state.model);
	const tokenUsage = formatTokenUsage(state.contextTokens, state.contextMaxTokens);
	const gitParts = formatGitStatus(state.git);
	const gitStr = gitParts.length > 0 ? gitParts.join(" ") : "";

	// Build status bar sections
	const leftSections: React.ReactNode[] = [];
	const rightSections: React.ReactNode[] = [];

	// Left: phase indicator, model
	leftSections.push(
		<Text key="phase" color={phaseColor as any} bold>{phaseEmoji}</Text>,
		<Text key="model" color={theme.fg.accent} bold>{" "}{modelShort}</Text>,
	);

	// Token usage
	if (state.contextMaxTokens > 0) {
		leftSections.push(
			<Text key="tokens" color={theme.fg.muted} dimColor>{" "}{tokenUsage}</Text>,
		);
	}

	// Git status
	if (gitStr) {
		leftSections.push(
			<Text key="git" color={theme.fg.muted} dimColor>{" "}{gitStr}</Text>,
		);
	}

	// Right: session, thinking level, inference mode
	rightSections.push(
		<Text key="session" color={theme.fg.muted} dimColor>{state.sessionTitle}</Text>,
	);

	if (state.thinkingLevel !== "off") {
		rightSections.push(
			<Text key="thinking" color={theme.fg.info} dimColor>{" "}{`thinking:${state.thinkingLevel}`}</Text>,
		);
	}

	if (state.inferenceMode !== "none") {
		rightSections.push(
			<Text key="mode" color={theme.fg.info} dimColor>{" "}{`mode:${state.inferenceMode}`}</Text>,
		);
	}

	if (state.workflowMode === "plan") {
		rightSections.push(
			<Text key="plan" color={theme.fg.warning} dimColor>{" "}[plan]</Text>,
		);
	}

	// Turn count
	if (state.transcriptTurns.length > 0) {
		rightSections.push(
			<Text key="turns" color={theme.fg.muted} dimColor>{" "}{`${state.transcriptTurns.length}t`}</Text>,
		);
	}

	// Scroll mode indicator
	rightSections.push(
		<Text key="scroll" color={theme.fg.muted} dimColor>{" "}{state.followMode ? "follow" : "scrolled"}</Text>,
	);

	return (
		<Box flexDirection="row" justifyContent="space-between" paddingX={1}>
			<Box flexDirection="row">
				{leftSections}
			</Box>
			<Box flexDirection="row" flexGrow={1} justifyContent="flex-end">
				{rightSections}
			</Box>
		</Box>
	);
};
