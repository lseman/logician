// ── Undo stack ────────────────────────────────────────────────────────────────
// Simple redo-able stack (pi-style).

export interface UndoSnapshot<T> {
	state: T;
}

export class UndoStack<T> {
	private past: UndoSnapshot<T>[] = [];
	private future: UndoSnapshot<T>[] = [];
	private readonly maxDepth: number;

	constructor(maxDepth = 50) {
		this.maxDepth = maxDepth;
	}

	push(state: T): void {
		this.past.push({ state: { ...state } as unknown as T });
		if (this.past.length > this.maxDepth) this.past.shift();
		this.future = [];
	}

	pop(): T | null {
		if (this.past.length === 0) return null;
		// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
		const snapshot = this.past.pop()!;
		this.future.push(snapshot);
		return snapshot.state;
	}

	peek(): T | null {
		if (this.past.length === 0) return null;
		return this.past[this.past.length - 1].state;
	}

	get depth(): number {
		return this.past.length;
	}

	hasPast(): boolean {
		return this.past.length > 0;
	}

	hasFuture(): boolean {
		return this.future.length > 0;
	}

	redo(): T | null {
		if (this.future.length === 0) return null;
		// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
		const snapshot = this.future.pop()!;
		this.past.push(snapshot);
		return snapshot.state;
	}
}
