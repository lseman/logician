import { Box, Text } from "ink";
import React from "react";
import type { InkListOverlayModel } from "../../overlays/ink-overlay-model.ts";
import { theme, type ThemeColor } from "../../terminal/theme.ts";

export function ListOverlay({
	model,
	width,
}: {
	model: InkListOverlayModel;
	width: number;
}): React.ReactElement {
	const innerWidth = Math.max(1, width - 4);
	const maxRows = model.maxRows ?? 10;
	const start = Math.max(
		0,
		Math.min(
			model.selectedIndex - Math.floor(maxRows / 2),
			Math.max(0, model.items.length - maxRows),
		),
	);
	const end = Math.min(model.items.length, start + maxRows);
	const visible = model.items.slice(start, end);

	return (
		<Box flexDirection="column" width={width} borderStyle="round" borderColor={theme.inkColor("borderMuted")}>
			<Box paddingX={1} justifyContent="space-between">
				<Text bold>{model.title}{model.subtitle ?? ""}</Text>
				{model.hints && <Text color={theme.inkColor("muted")}>{model.hints}</Text>}
			</Box>
			{model.headerLines?.map((line, index) => (
				<Text key={`header:${index}`} color={theme.inkColor("muted")} wrap="truncate-end"> {line}</Text>
			))}
			{start > 0 && <Text color={theme.inkColor("dim")}> ↑ {start} more above</Text>}
			{visible.length === 0 ? (
				<Text color={theme.inkColor("muted")}> {model.emptyText}</Text>
			) : visible.map((item, index) => {
				const marker = item.selected ? "❯" : " ";
				const current = item.current ? " ✓" : "";
				const left = `${marker} ${item.label}${current}`;
				return (
					<Box key={`${start + index}:${item.label}`} paddingX={1} width={innerWidth + 2}>
						<Text bold={item.selected} color={theme.inkColor(item.selected ? "selected" : itemColor(item.status))}>
							{left}
						</Text>
						<Box flexGrow={1} />
						{item.metadata && <Text color={theme.inkColor(item.selected ? "text" : "muted")}>{item.metadata}</Text>}
					</Box>
				);
			})}
			{end < model.items.length && <Text color={theme.inkColor("dim")}> ↓ {model.items.length - end} more below</Text>}
			<Box paddingX={1}>
				<Text color={theme.inkColor("muted")} wrap="truncate-end">{model.footer}</Text>
			</Box>
		</Box>
	);
}

function itemColor(status: InkListOverlayModel["items"][number]["status"]): ThemeColor {
	switch (status) {
		case "active": return "active";
		case "success": return "success";
		case "warning": return "warning";
		case "error": return "error";
		default: return "text";
	}
}

export function listOverlayHeight(model: InkListOverlayModel): number {
	const maxRows = model.maxRows ?? 10;
	const visibleRows = Math.min(model.items.length || 1, maxRows);
	const hiddenIndicators = model.items.length > maxRows ? 1 : 0;
	return 2 + 1 + (model.headerLines?.length ?? 0) + hiddenIndicators + visibleRows + 1;
}
