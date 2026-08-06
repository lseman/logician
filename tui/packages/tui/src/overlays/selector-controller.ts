export interface SelectorWindow {
	start: number;
	end: number;
}

/** Shared selection, wrapping, paging, and viewport behavior for TUI pickers. */
export class SelectorController {
	index = 0;

	set(index: number, count: number): void {
		this.index = count > 0 ? Math.max(0, Math.min(index, count - 1)) : 0;
	}

	move(delta: number, count: number): void {
		if (count <= 0) return;
		this.index = (this.index + (delta % count) + count) % count;
	}

	window(count: number, maxRows: number): SelectorWindow {
		const start = Math.max(
			0,
			Math.min(
				this.index - Math.floor(maxRows / 2),
				Math.max(0, count - maxRows),
			),
		);
		return { start, end: Math.min(count, start + maxRows) };
	}
}
