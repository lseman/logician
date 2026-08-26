import { CancellationScope } from "../../control/cancellation-scope.ts";

/** Owns abort signaling and settlement for one active harness turn. */
export class HarnessTurnController {
	private scope: CancellationScope | null = null;
	private settledPromise?: Promise<void>;
	private resolveSettled?: () => void;

	async run<T>(
		execute: (signal: AbortSignal) => Promise<T>,
		onSettled: () => void,
	): Promise<T> {
		this.scope = new CancellationScope({ operation: "agent turn" });
		this.settledPromise = new Promise<void>(resolve => {
			this.resolveSettled = resolve;
		});
		try {
			return await this.scope.run(execute, { rejectOnAbort: false });
		} finally {
			this.scope = null;
			onSettled();
			this.resolveSettled?.();
			this.settledPromise = undefined;
			this.resolveSettled = undefined;
		}
	}

	abort(reason?: unknown): void {
		this.scope?.abort(reason);
	}

	async wait(): Promise<void> {
		await this.settledPromise;
	}
}
