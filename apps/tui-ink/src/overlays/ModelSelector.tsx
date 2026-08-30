// ── Ink TUI — Model Selector ──────────────────────────────────────────────────

import React, { useMemo } from "react";
import { Box, Text } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";

interface ModelSelectorProps {
	bridge: AgentRuntime;
	isActive: boolean;
	onSelect: (model?: string) => void;
	onClose: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
	bridge,
	isActive,
	onSelect,
	onClose,
}) => {
	const theme = getCurrentTheme();

	const { models, current } = useMemo(() => {
		let list: string[] = [];
		try {
			const opts = bridge.models.options?.() ?? [];
			list = opts.map((o: { key?: string; model?: string }) => o.model ?? o.key ?? "");
		} catch {
			/* ignore */
		}
		if (list.length === 0) {
			try {
				list = bridge.models.list();
			} catch {
				/* ignore */
			}
		}
		let cur = "";
		try {
			cur = bridge.models.current();
		} catch {
			/* ignore */
		}
		return { models: Array.from(new Set(list.filter(Boolean))), current: cur };
	}, [bridge]);

	const { index } = useOverlayInput({
		isActive,
		count: models.length,
		onSelect: i => onSelect(models[i]),
		onClose,
	});

	return (
		<Box
			borderColor={theme.fg.border}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={44}
		>
			<Text color={theme.fg.header} bold>
				Select Model
			</Text>
			{models.length === 0 ? (
				<Text color={theme.fg.muted}>no models configured</Text>
			) : (
				models.map((model, i) => (
					<Text
						key={model}
						color={
							i === index
								? (theme.fg.selected as string)
								: (theme.fg.text as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${model}${model === current ? "  (current)" : ""}`}
					</Text>
				))
			)}
		</Box>
	);
};
