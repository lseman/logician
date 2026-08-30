// ── Ink TUI — shared overlay key handling ─────────────────────────────────────
// A single list-navigation input model for every overlay popup: up/down move a
// clamped selection index, Enter selects, Escape closes. Only the currently
// open overlay should pass `isActive: true` so keystrokes never reach more than
// one handler (Ink has no event bubbling — every active useInput fires).

import { useState } from "react";
import { useInput } from "ink";

export interface UseOverlayInputOptions {
	isActive: boolean;
	count: number;
	onSelect: (index: number) => void;
	onClose: () => void;
	/** Extra single-key handlers, e.g. { d: () => onDelete(index) }. */
	keys?: Record<string, (index: number) => void>;
	pageSize?: number;
}

export function useOverlayInput({
	isActive,
	count,
	onSelect,
	onClose,
	keys,
	pageSize = 5,
}: UseOverlayInputOptions): { index: number; setIndex: (i: number) => void } {
	const [index, setIndex] = useState(0);

	const clamp = (i: number): number =>
		count <= 0 ? 0 : Math.max(0, Math.min(count - 1, i));

	useInput(
		(input, key) => {
			if (key.escape) {
				onClose();
				return;
			}
			if (key.upArrow || (key.ctrl && input === "p")) {
				setIndex(i => clamp(i - 1));
				return;
			}
			if (key.downArrow || (key.ctrl && input === "n")) {
				setIndex(i => clamp(i + 1));
				return;
			}
			if (key.pageUp) {
				setIndex(i => clamp(i - pageSize));
				return;
			}
			if (key.pageDown) {
				setIndex(i => clamp(i + pageSize));
				return;
			}
			if (key.return) {
				if (count > 0) onSelect(clamp(index));
				return;
			}
			if (keys && input && keys[input]) {
				keys[input](clamp(index));
			}
		},
		{ isActive },
	);

	return { index: clamp(index), setIndex: (i: number) => setIndex(clamp(i)) };
}
