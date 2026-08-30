// ── Ink TUI — Footer Bar (widget-based status bar) ─────────────────────────
// Wraps FooterStatusBar and renders its output as React Text components.
// Feeds TuiState data into the widget factory each render cycle.

import React, { useEffect, useRef } from "react";
import { Box, Text } from "ink";
import type { TuiState } from "../state.ts";
import type { AppConfig } from "../types.ts";
import { FooterStatusBar } from "../rendering/layout.ts";
import type { WidgetFactoryStatus } from "../rendering/widget-factory.ts";

interface FooterBarProps {
	state: TuiState;
	config: AppConfig;
	width: number;
}

export const FooterBar: React.FC<FooterBarProps> = ({ state, config, width }) => {
	const footerRef = useRef<FooterStatusBar | null>(null);

	// Initialize the footer status bar once
	if (!footerRef.current) {
		footerRef.current = new FooterStatusBar();
	}

	// Feed state data into the widget factory each render cycle
	useEffect(() => {
		const footer = footerRef.current;
		if (!footer) return;

		const widgetStatus: WidgetFactoryStatus = {
			phase: state.phase,
			model: state.model.split("/").pop() ?? state.model,
			cwd: state.cwd,
			branch: state.branch ?? "",
			contextTokens: state.contextTokens,
			contextMaxTokens: state.contextMaxTokens,
			contextCompacted: false,
			thinkingLevel: state.thinkingLevel,
			inferenceMode: state.inferenceMode,
			reasoner: state.reasoner.name === "default" ? "none" : state.reasoner.name,
			sessionTitle: state.sessionTitle,
			mcpServerCount: 0, // TODO: get from bridge
			mcpLoading: state.mcpLoading,
			sandboxMode: "code", // TODO: get from config
			permissionMode: state.permissionMode,
			workflowMode: state.workflowMode,
			executionProfile: state.executionProfile,
			rtkProxyEnabled: state.rtkProxyEnabled,
			legroomEnabled: state.legroomEnabled,
			memoriamEnabled: state.memoriamEnabled,
			graphicianEnabled: state.graphicianEnabled,
			fffgrepEnabled: state.fffgrepEnabled,
			gitModified: state.git.modified,
			gitStaged: state.git.staged,
			gitUntracked: state.git.untracked,
			virtualEnv: state.virtualEnv,
			turnCount: state.transcriptTurns.length,
			messageCount: 0,
			cacheReadTokens: state.cacheReadTokens,
			goalCondition: state.goalCondition,
			goalTurnCount: state.goalTurnCount,
			goalElapsed: state.goalElapsed,
		};

		footer.update(widgetStatus);
	}, [
		state.phase, state.model, state.cwd, state.branch,
		state.contextTokens, state.contextMaxTokens,
		state.thinkingLevel, state.inferenceMode,
		state.reasoner, state.sessionTitle, state.mcpLoading,
		state.permissionMode, state.workflowMode, state.executionProfile,
		state.rtkProxyEnabled, state.legroomEnabled, state.memoriamEnabled,
		state.graphicianEnabled, state.fffgrepEnabled,
		state.git, state.virtualEnv, state.transcriptTurns.length,
		state.cacheReadTokens, state.goalCondition,
		state.goalTurnCount, state.goalElapsed,
	]);

	// Start animation timer for phase spinner (only when active)
	useEffect(() => {
		const footer = footerRef.current;
		if (!footer) return;
		const isActive = state.phase === "thinking" ||
			state.phase === "working" ||
			state.phase === "cancelling";
		if (isActive && !footer.timer) {
			footer.startAnimation();
		} else if (!isActive && footer.timer) {
			footer.stopAnimation();
		}
		return () => {
			if (footer.timer) footer.stopAnimation();
		};
	}, [state.phase]);

	// Cleanup on unmount
	useEffect(() => {
		const footer = footerRef.current;
		return () => footer?.dispose();
	}, []);

	// Render the footer output
	const footer = footerRef.current;
	const lines = footer?.render(width) ?? [" ".repeat(width)];

	return (
		<Box flexDirection="column">
			{lines.map((line, i) => (
				<Text key={i} dimColor>{line}</Text>
			))}
		</Box>
	);
};
