// ── Ink TUI — Input Bar ───────────────────────────────────────────────────────

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { getCurrentTheme } from "../theme";

interface InputBarProps {
	value: string;
	onValueChange: (value: string) => void;
	onSubmit: () => void;
	isActive?: boolean;
	placeholder?: string;
}

/** Index of the start of the word left of `pos` (whitespace-delimited). */
function wordStartLeft(text: string, pos: number): number {
	let i = pos;
	while (i > 0 && /\s/.test(text[i - 1]!)) i--;
	while (i > 0 && !/\s/.test(text[i - 1]!)) i--;
	return i;
}

export const InputBar: React.FC<InputBarProps> = ({
	value,
	onValueChange,
	onSubmit,
	isActive = true,
	placeholder = "Type a message…  ( / commands · @ files · Ctrl+C exit )",
}) => {
	const theme = getCurrentTheme();
	const [cursor, setCursor] = useState(0);
	const safeCursor = Math.min(cursor, value.length);

	const setBoth = (next: string, nextCursor: number): void => {
		onValueChange(next);
		setCursor(Math.max(0, Math.min(nextCursor, next.length)));
	};

	useInput(
		(input, key) => {
			if (key.return) {
				onSubmit();
				setCursor(0);
				return;
			}

			// Cursor movement
			if (key.leftArrow) {
				setCursor(c => Math.max(0, Math.min(c, value.length) - 1));
				return;
			}
			if (key.rightArrow) {
				setCursor(c => Math.min(value.length, c + 1));
				return;
			}
			if (key.ctrl && input === "a") {
				setCursor(0);
				return;
			}
			if (key.ctrl && input === "e") {
				setCursor(value.length);
				return;
			}

			// Deletion
			if (key.backspace) {
				if (safeCursor > 0) {
					setBoth(
						value.slice(0, safeCursor - 1) + value.slice(safeCursor),
						safeCursor - 1,
					);
				}
				return;
			}
			if (key.delete) {
				// Some terminals map the Backspace key to `delete`.
				if (safeCursor > 0 && safeCursor >= value.length) {
					setBoth(value.slice(0, safeCursor - 1), safeCursor - 1);
				} else if (safeCursor < value.length) {
					setBoth(value.slice(0, safeCursor) + value.slice(safeCursor + 1), safeCursor);
				}
				return;
			}
			if (key.ctrl && input === "u") {
				setBoth("", 0);
				return;
			}
			if (key.ctrl && input === "w") {
				const start = wordStartLeft(value, safeCursor);
				setBoth(value.slice(0, start) + value.slice(safeCursor), start);
				return;
			}
			if (key.ctrl && input === "k") {
				setBoth(value.slice(0, safeCursor), safeCursor);
				return;
			}

			// Printable input (ignore other control chords)
			if (input && !key.ctrl && !key.meta && !key.escape) {
				setBoth(
					value.slice(0, safeCursor) + input + value.slice(safeCursor),
					safeCursor + input.length,
				);
			}
		},
		{ isActive },
	);

	// Render value with a block cursor.
	const before = value.slice(0, safeCursor);
	const at = value.slice(safeCursor, safeCursor + 1) || " ";
	const after = value.slice(safeCursor + 1);

	return (
		<Box flexDirection="row" flexGrow={1}>
			{value.length === 0 ? (
				<>
					<Text inverse>{" "}</Text>
					<Text color={theme.fg.muted}>{placeholder}</Text>
				</>
			) : (
				<Text color={theme.fg.primary}>
					{before}
					<Text inverse>{at}</Text>
					{after}
				</Text>
			)}
		</Box>
	);
};
