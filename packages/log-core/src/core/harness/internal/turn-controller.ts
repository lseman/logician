/** Owns abort signaling and settlement for one active harness turn. */
export class HarnessTurnController {
	private abortController: AbortController | null = null;
	private settledPromise?: Promise<void>;
	private resolveSettled?: () => void;

	async run<T>(
		execute: (signal: AbortSignal) => Promise<T>,
		onSettled: () => void,
	): Promise<T> {
		this.abortController = new AbortController();
		this.settledPromise = new Promise<void>(resolve => {
			this.resolveSettled = resolve;
		});
		try {
			return await execute(this.abortController.signal);
		} finally {
			this.abortController = null;
			onSettled();
			this.resolveSettled?.();
			this.settledPromise = undefined;
			this.resolveSettled = undefined;
		}
	}

	abort(reason?: unknown): void {
		this.abortController?.abort(reason);
	}

	async wait(): Promise<void> {
		await this.settledPromise;
	}
}
