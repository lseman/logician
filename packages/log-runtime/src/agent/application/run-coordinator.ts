export interface RuntimeRunSubmission {
	message: string;
	canSteer: () => boolean;
	steer: (message: string) => void;
	execute: (message: string) => Promise<void>;
}

/** Serializes user-visible runs and owns the runtime's active/idle state. */
export class RuntimeRunCoordinator {
	private tail: Promise<void> = Promise.resolve();
	private active = false;

	isActive(): boolean {
		return this.active;
	}

	submit(submission: RuntimeRunSubmission): Promise<void> {
		if (this.active && submission.canSteer()) {
			submission.steer(submission.message);
			return Promise.resolve();
		}

		const run = this.tail.then(async () => {
			this.active = true;
			try {
				await submission.execute(submission.message);
			} finally {
				this.active = false;
			}
		});
		this.tail = run.catch(() => {});
		return run;
	}

	/** Force idle after the owning runtime has cancelled or discarded its run. */
	reset(): void {
		this.active = false;
	}
}
