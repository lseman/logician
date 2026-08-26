export type RuntimeStartupTask = (source: string) => Promise<void>;

/** Owns exactly-once startup, concurrent joining, retry, and reset semantics. */
export class RuntimeStartupCoordinator {
	private pending: Promise<void> | null = null;
	private ready = false;
	private generation = 0;

	constructor(private readonly initialize: RuntimeStartupTask) {}

	async ensure(source = "startup"): Promise<void> {
		if (this.ready) return;
		if (!this.pending) {
			const generation = this.generation;
			this.pending = this.initialize(source)
				.then(() => {
					if (generation === this.generation) this.ready = true;
				})
				.catch(error => {
					if (generation === this.generation) this.pending = null;
					throw error;
				});
		}
		await this.pending;
	}

	reset(): void {
		this.generation++;
		this.ready = false;
		this.pending = null;
	}
}
