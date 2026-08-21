export interface ProgressTask {
	id: string | number;
	status: string;
}

export interface ProgressTrackerOptions {
	minimumChecks?: number;
	stalledChecks?: number;
}

/** Detect repeated autonomous turns with no new observable evidence. */
export class ProgressTracker {
	private readonly minimumChecks: number;
	private readonly stalledChecks: number;
	private readonly evidence = new Set<string>();
	private checks = 0;
	private previousFingerprint = "";
	private consecutiveStalls = 0;

	constructor(options: ProgressTrackerOptions = {}) {
		this.minimumChecks = options.minimumChecks ?? 3;
		this.stalledChecks = options.stalledChecks ?? 2;
	}

	recordToolResult(
		toolName: string,
		argumentsJson: string,
		result: string,
	): void {
		this.evidence.add(
			`${toolName}\0${argumentsJson}\0${result.replace(/\s+/g, " ").trim().slice(0, 500)}`,
		);
	}

	shouldStop(tasks: readonly ProgressTask[]): boolean {
		this.checks++;
		const taskState = tasks
			.map(task => `${task.id}:${task.status}`)
			.sort()
			.join("|");
		const fingerprint = `${this.evidence.size}\0${taskState}`;
		if (fingerprint === this.previousFingerprint) this.consecutiveStalls++;
		else this.consecutiveStalls = 0;
		this.previousFingerprint = fingerprint;
		return (
			this.checks > this.minimumChecks &&
			this.consecutiveStalls >= this.stalledChecks
		);
	}
}
