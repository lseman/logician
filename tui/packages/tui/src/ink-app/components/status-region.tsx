import { Box, Text } from "ink";
import React, { useMemo } from "react";
import type { InkTextComponent, InkTextRow } from "../../terminal/core.ts";

export function InkTextRows({ rows }: { rows: readonly InkTextRow[] }): React.ReactElement {
	if (rows.length === 0) return <></>;
	return (
		<Box flexDirection="column">
			{rows.map((row, rowIndex) => (
				<Text
					// biome-ignore lint/suspicious/noArrayIndexKey: dock rows are positional
					key={rowIndex}
					wrap="truncate-end"
				>
					{row.map((span, spanIndex) => (
						<Text
							// biome-ignore lint/suspicious/noArrayIndexKey: spans are ordered fragments
							key={spanIndex}
							color={span.color}
							backgroundColor={span.backgroundColor}
							bold={span.bold}
							dimColor={span.dim}
							underline={span.underline}
							italic={span.italic}
							inverse={span.inverse}
						>
							{span.text}
						</Text>
					))}
				</Text>
			))}
		</Box>
	);
}

export function StatusRegion({
	component,
	width,
	renderTick,
}: {
	component: InkTextComponent | null;
	width: number;
	renderTick: number;
}): React.ReactElement {
	const rows = useMemo(
		() => component?.getInkTextRows(width) ?? [[{ text: " ".repeat(width) }]],
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick invalidates stable component instances
		[component, width, renderTick],
	);
	return <InkTextRows rows={rows} />;
}

export function statusHeight(component: InkTextComponent | null, width: number): number {
	return component?.getInkTextRows(width).length ?? 1;
}
