import { Box, Text } from "ink";
import React from "react";
import type { InkListOverlayModel } from "../../overlays/ink-overlay-model.ts";

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
		<Box flexDirection="column" width={width} borderStyle="round" borderColor="gray">
			<Box paddingX={1} justifyContent="space-between">
				<Text bold>{model.title}{model.subtitle ?? ""}</Text>
				{model.hints && <Text dimColor>{model.hints}</Text>}
			</Box>
			{model.headerLines?.map((line, index) => (
				<Text key={`header:${index}`} dimColor wrap="truncate-end"> {line}</Text>
			))}
			{start > 0 && <Text dimColor> ↑ {start} more above</Text>}
			{visible.length === 0 ? (
				<Text dimColor> {model.emptyText}</Text>
			) : visible.map((item, index) => {
				const marker = item.selected ? "❯" : " ";
				const current = item.current ? " ✓" : "";
				const left = `${marker} ${item.label}${current}`;
				return (
					<Box key={`${start + index}:${item.label}`} paddingX={1} width={innerWidth + 2}>
						<Text bold={item.selected} color={item.selected ? "cyan" : undefined}>
							{left}
						</Text>
						<Box flexGrow={1} />
						{item.metadata && <Text dimColor={!item.selected}>{item.metadata}</Text>}
					</Box>
				);
			})}
			{end < model.items.length && <Text dimColor> ↓ {model.items.length - end} more below</Text>}
			<Box paddingX={1}>
				<Text dimColor wrap="truncate-end">{model.footer}</Text>
			</Box>
		</Box>
	);
}

export function listOverlayHeight(model: InkListOverlayModel): number {
	const maxRows = model.maxRows ?? 10;
	const visibleRows = Math.min(model.items.length || 1, maxRows);
	const hiddenIndicators = model.items.length > maxRows ? 1 : 0;
	return 2 + 1 + (model.headerLines?.length ?? 0) + hiddenIndicators + visibleRows + 1;
}
