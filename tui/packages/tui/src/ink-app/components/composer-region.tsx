import { Box, Text, useCursor } from "ink";
import React, { useMemo } from "react";
import type {
	InkComposerComponent,
	InkComposerModel,
} from "../../terminal/core.ts";
import { visibleWidth } from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";

export interface ComposerRegionProps {
	component: InkComposerComponent | null;
	width: number;
	originY: number;
	showHardwareCursor: boolean;
	renderTick: number;
}

const EMPTY_COMPOSER: InkComposerModel = {
	prompt: "",
	headerHint: null,
	beforeCursor: "",
	atCursor: " ",
	afterCursor: "",
	isPlaceholder: false,
	leftClipped: false,
	rightClipped: false,
	cursorColumn: 0,
	focused: false,
};

/** Native Ink owner for composer layout, styling, clipping, and cursor placement. */
export function ComposerRegion({
	component,
	width,
	originY,
	showHardwareCursor,
	renderTick,
}: ComposerRegionProps): React.ReactElement {
	const { setCursorPosition } = useCursor();
	const model = useMemo(
		() => component?.getInkComposerModel(width) ?? EMPTY_COMPOSER,
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick invalidates stable component instances
		[component, width, renderTick],
	);
	const cursorRow = model.headerHint ? 1 : 0;

	if (showHardwareCursor && model.focused) {
		setCursorPosition({
			x: model.cursorColumn,
			// Ink fullscreen cursor coordinates are one greater than layout rows.
			y: originY + cursorRow + 1,
		});
	} else {
		setCursorPosition(undefined);
	}

	return (
		<Box flexDirection="column">
			{model.headerHint ? (
				<Box width={width}>
					<Text color={theme.inkColor("separator")}>
						{"─".repeat(
							Math.max(1, width - visibleWidth(` ${model.headerHint} `)),
						)}
					</Text>
					<Text color={theme.inkColor("muted")}>{` ${model.headerHint} `}</Text>
				</Box>
			) : null}
			<Box width={width}>
				<Text color={theme.inkColor("prompt")} bold>{model.prompt}</Text>
				{model.leftClipped ? <Text color={theme.inkColor("dim")}>‹</Text> : null}
				<Text color={theme.inkColor(model.isPlaceholder ? "inputPlaceholder" : "inputText")}>
					{model.beforeCursor}
					<Text inverse={model.focused && !model.isPlaceholder}>
						{model.atCursor}
					</Text>
					{model.afterCursor}
				</Text>
				{model.rightClipped ? <Text color={theme.inkColor("dim")}>›</Text> : null}
			</Box>
		</Box>
	);
}

export function composerHeight(
	component: InkComposerComponent | null,
	width: number,
): number {
	return component?.getInkComposerModel(width).headerHint ? 2 : 1;
}
