// ── Ink TUI — MCP Manager ────────────────────────────────────────────────────
// MVP: read-only snapshot. Enable/disable is a parity-phase item.

import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import { getCurrentTheme } from "../theme";

interface McpManagerProps {
	bridge?: AgentRuntime;
	isActive: boolean;
	onClose: () => void;
}

export const McpManager: React.FC<McpManagerProps> = ({
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
			?.getMcpSnapshot?.()
			.then((snap: unknown) => {
				if (cancelled) return;
				const servers = Array.isArray((snap as { servers?: unknown[] })?.servers)
					? (snap as { servers: Array<{ name?: string; connected?: boolean; enabled?: boolean }> }).servers
					: [];
				setLines(
					servers.length
						? servers.map(
								s => `${s.connected ?? s.enabled ? "●" : "○"} ${s.name ?? "?"}`,
							)
						: ["no MCP servers"],
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
				MCP Servers
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
