// ── Ink TUI — Input Bar (with undo/redo + kill ring) ─────────────────────────

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

// ── Undo/Redo Stack ──────────────────────────────────────────────────────────

interface UndoEntry {
	value: string;
	cursor: number;
}

class UndoStack {
	private past: UndoEntry[] = [];
	private future: UndoEntry[] = [];

	push(entry: UndoEntry): void {
		this.past.push(entry);
		this.future = []; // clear redo on new edit
	}

	undo(): UndoEntry | null {
		if (this.past.length <= 1) return null;
		const current = this.past.pop()!;
		const previous = this.past[this.past.length - 1]!;
		this.future.push(current);
		return previous;
	}

	redo(): UndoEntry | null {
		if (this.future.length === 0) return null;
		const next = this.future.pop()!;
		this.past.push(next);
		return next;
	}

	get length(): number {
		return this.past.length;
	}
}

// ── Kill Ring ────────────────────────────────────────────────────────────────

class KillRing {
	private entries: string[] = [];
	private latest: string | null = null; // text from most recent kill

	push(text: string): void {
		if (!text) return;
		this.entries.push(text);
		if (this.entries.length > 20) this.entries.shift();
		this.latest = text;
	}

	paste(): string | null {
		if (this.entries.length === 0) return null;
		return this.entries[this.entries.length - 1]!;
	}

	get latestKill(): string | null {
		return this.latest;
	}
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Index of the start of the word left of `pos` (whitespace-delimited). */
function wordStartLeft(text: string, pos: number): number {
	let i = pos;
	while (i > 0 && /\s/.test(text[i - 1]!)) i--;
	while (i > 0 && !/\s/.test(text[i - 1]!)) i--;
	return i;
}

/** Index of the start of the word right of `pos` (whitespace-delimited). */
function wordStartRight(text: string, pos: number): number {
	let i = pos;
	while (i < text.length && /\s/.test(text[i]!)) i++;
	while (i < text.length && !/\s/.test(text[i]!)) i++;
	return i;
}

// ── Shared singleton instances ───────────────────────────────────────────────

const undoStack = new UndoStack();
const killRing = new KillRing();

export { undoStack, killRing };

// ── Component ────────────────────────────────────────────────────────────────

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

	const saveState = (): UndoEntry => ({ value, cursor: safeCursor });

	const setBoth = (next: string, nextCursor: number): void => {
		onValueChange(next);
		setCursor(Math.max(0, Math.min(nextCursor, next.length)));
	};

	useInput(
		(input, key) => {
			if (key.return) {
				onSubmit();
				return;
			}

			// ── Undo / Redo ────────────────────────────────────────────────
			if (key.ctrl && input === "z") {
				const entry = undoStack.undo();
				if (entry) setBoth(entry.value, Math.min(entry.cursor, entry.value.length));
				return;
			}
			if ((key.ctrl && key.shift && input === "z") || (key.ctrl && input === "y")) {
				const entry = undoStack.redo();
				if (entry) setBoth(entry.value, Math.min(entry.cursor, entry.value.length));
				return;
			}

			// ── Cursor movement ────────────────────────────────────────────
			if (key.leftArrow) {
				setCursor(c => Math.max(0, c - 1));
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

			// Word navigation
			if (key.ctrl && key.leftArrow) {
				const start = wordStartLeft(value, safeCursor);
				setBoth(value, start);
				return;
			}
			if (key.ctrl && key.rightArrow) {
				const end = wordStartRight(value, safeCursor);
				setBoth(value, end);
				return;
			}

			// ── Deletion with kill ring ────────────────────────────────────
			if (key.backspace) {
				if (safeCursor > 0) {
					const deleted = value[safeCursor - 1]!;
					const snapshot = saveState();
					setBoth(
						value.slice(0, safeCursor - 1) + value.slice(safeCursor),
						safeCursor - 1,
					);
					undoStack.push(snapshot);
					killRing.push(deleted);
				}
				return;
			}
			if (key.delete) {
				const snapshot = saveState();
				if (safeCursor > 0 && safeCursor >= value.length) {
					setBoth(value.slice(0, safeCursor - 1), safeCursor - 1);
					undoStack.push(snapshot);
					killRing.push(value[safeCursor - 1]!);
				} else if (safeCursor < value.length) {
					const deleted = value.slice(safeCursor, safeCursor + 1);
					setBoth(
						value.slice(0, safeCursor) + value.slice(safeCursor + 1),
						safeCursor,
					);
					undoStack.push(snapshot);
					killRing.push(deleted);
				}
				return;
			}

			// Kill operations (push to kill ring)
			if (key.ctrl && input === "u") {
				const killed = value.slice(0, safeCursor);
				const snapshot = saveState();
				setBoth("", 0);
				undoStack.push(snapshot);
				killRing.push(killed);
				return;
			}
			if (key.ctrl && input === "w") {
				const start = wordStartLeft(value, safeCursor);
				const killed = value.slice(start, safeCursor);
				const snapshot = saveState();
				setBoth(value.slice(0, start) + value.slice(safeCursor), start);
				undoStack.push(snapshot);
				killRing.push(killed);
				return;
			}
			if (key.ctrl && input === "k") {
				const killed = value.slice(safeCursor);
				const snapshot = saveState();
				setBoth(value.slice(0, safeCursor), safeCursor);
				undoStack.push(snapshot);
				killRing.push(killed);
				return;
			}

			// Paste from kill ring (Ctrl+Shift+V or Ctrl+_ on some layouts)
			if ((key.ctrl && key.shift && input === "v") || (key.ctrl && input === "_")) {
				const pasted = killRing.paste();
				if (pasted) {
					const snapshot = saveState();
					setBoth(
						value.slice(0, safeCursor) + pasted + value.slice(safeCursor),
						safeCursor + pasted.length,
					);
					undoStack.push(snapshot);
				}
				return;
			}

			// ── Printable input ────────────────────────────────────────────
			if (input && !key.ctrl && !key.meta && !key.escape) {
				const snapshot = saveState();
				setBoth(
					value.slice(0, safeCursor) + input + value.slice(safeCursor),
					safeCursor + input.length,
				);
				undoStack.push(snapshot);
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
					<Text color={theme.fg.inputPlaceholder}>{placeholder}</Text>
				</>
			) : (
				<Text color={theme.fg.inputText}>
					{before}
					<Text inverse>{at}</Text>
					{after}
				</Text>
			)}
		</Box>
	);
};
