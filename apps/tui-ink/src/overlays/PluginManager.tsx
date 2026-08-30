// ── Ink TUI — Plugin Manager ─────────────────────────────────────────────────
// MVP: read-only snapshot. Toggle/CRUD is a parity-phase item.

import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import { getCurrentTheme } from "../theme";

interface PluginManagerProps {
	bridge?: AgentRuntime;
	isActive: boolean;
	onClose: () => void;
}

export const PluginManager: React.FC<PluginManagerProps> = ({
	bridge,
	isActive,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const [lines, setLines] = useState<string[]>(["loading…"]);

	useInput((_i, key) => {
		if (key.escape || key.return) onClose();
	}, { isActive });

	useEffect(() => {
		let cancelled = false;
		void bridge
			?.getPluginSnapshot?.()
			.then((snap: unknown) => {
				if (cancelled) return;
				const items = Array.isArray((snap as { plugins?: unknown[] })?.plugins)
					? (snap as { plugins: Array<{ name?: string; enabled?: boolean }> }).plugins
					: [];
				setLines(
					items.length
						? items.map(p => `${p.enabled ? "✓" : "☐"} ${p.name ?? "?"}`)
						: ["no plugins"],
				);
			})
			.catch(() => !cancelled && setLines(["(unavailable)"]));
		return () => {
			cancelled = true;
		};
	}, [bridge]);

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={40}
		>
			<Text color={theme.fg.accent as string} bold>
				Plugins
			</Text>
			{lines.map((l, i) => (
				<Text key={i} color={theme.fg.primary as string}>
					{l}
				</Text>
			))}
			<Text color={theme.fg.muted as string}>esc / ⏎ to close</Text>
		</Box>
	);
};
