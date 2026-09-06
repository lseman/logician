export type CancellationKind =
	| "user"
	| "steering"
	| "timeout"
	| "shutdown"
	| "parent"
	| "failure";

export class CancellationError extends Error {
	constructor(
		message: string,
		readonly kind: CancellationKind,
		readonly operation: string,
		options?: ErrorOptions,
	) {
		super(message, options);
		this.name = "CancellationError";
	}
}

export interface CancellationScopeOptions {
	operation: string;
	parent?: AbortSignal;
	timeoutMs?: number;
}

export type CancellationCleanup = () => void | Promise<void>;

/**
 * Owns cancellation, deadlines, and cleanup for one operation.
 * Callers receive only its AbortSignal; lifecycle mechanics stay here.
 */
export class CancellationScope {
	private readonly controller = new AbortController();
	private readonly cleanups: CancellationCleanup[] = [];
	private readonly detachParent?: () => void;
	private readonly deadlineTimer?: ReturnType<typeof setTimeout>;
	private closed = false;
	private closePromise?: Promise<void>;

	constructor(readonly options: CancellationScopeOptions) {
		const { parent, timeoutMs } = options;
		if (parent) {
			const abortFromParent = () =>
				this.abort(
					parent.reason ??
						new CancellationError(
							`Parent cancelled ${options.operation}`,
							"parent",
							options.operation,
						),
				);
			if (parent.aborted) abortFromParent();
			else {
				parent.addEventListener("abort", abortFromParent, { once: true });
				this.detachParent = () =>
					parent.removeEventListener("abort", abortFromParent);
			}
		}
		if (timeoutMs !== undefined && timeoutMs > 0 && !this.signal.aborted) {
			this.deadlineTimer = setTimeout(
				() =>
					this.abort(
						new CancellationError(
							`${options.operation} timed out after ${timeoutMs}ms`,
							"timeout",
							options.operation,
						),
					),
				timeoutMs,
			);
		}
	}

	get signal(): AbortSignal {
		return this.controller.signal;
	}

	abort(reason?: unknown): void {
		if (this.signal.aborted) return;
		this.controller.abort(
			reason ??
				new CancellationError(
					`Cancelled ${this.options.operation}`,
					"user",
					this.options.operation,
				),
		);
	}

	addCleanup(cleanup: CancellationCleanup): () => void {
		if (this.closed) throw new Error("Cannot add cleanup to a closed scope");
		this.cleanups.push(cleanup);
		return () => {
			const index = this.cleanups.indexOf(cleanup);
			if (index >= 0) this.cleanups.splice(index, 1);
		};
	}

	async run<T>(
		work: (signal: AbortSignal) => Promise<T>,
		options: { rejectOnAbort?: boolean } = {},
	): Promise<T> {
		if (this.closed) throw new Error("Cannot run work in a closed scope");
		let detachAbort: (() => void) | undefined;
		try {
			this.signal.throwIfAborted();
			if (options.rejectOnAbort === false) return await work(this.signal);
			return await new Promise<T>((resolve, reject) => {
				const onAbort = () => reject(this.signal.reason);
				this.signal.addEventListener("abort", onAbort, { once: true });
				detachAbort = () => this.signal.removeEventListener("abort", onAbort);
				Promise.resolve()
					.then(() => {
						this.signal.throwIfAborted();
						return work(this.signal);
					})
					.then(resolve, reject);
			});
		} finally {
			detachAbort?.();
			await this.close();
		}
	}

	close(): Promise<void> {
		if (this.closePromise) return this.closePromise;
		this.closed = true;
		if (this.deadlineTimer) clearTimeout(this.deadlineTimer);
		this.detachParent?.();
		const cleanups = this.cleanups.splice(0).reverse();
		this.closePromise = Promise.resolve().then(async () => {
			const failures: unknown[] = [];
			for (const cleanup of cleanups) {
				try {
					await cleanup();
				} catch (error) {
					failures.push(error);
				}
			}
			if (failures.length) {
				throw new AggregateError(
					failures,
					`Cleanup failed for ${this.options.operation}`,
				);
			}
		});
		return this.closePromise;
	}
}
