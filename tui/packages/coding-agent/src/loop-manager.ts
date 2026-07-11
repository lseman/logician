// ── Completion-aware recurring loop manager ──────────────────────────────
// Schedules the next run only after the current run settles. This guarantees
// at-most-one in-flight agent request and avoids setInterval overlap.

export type LoopStatus = "idle" | "scheduled" | "running" | "stopped";

export interface LoopState {
	readonly prompt: string;
	readonly intervalMs: number;
	readonly iteration: number;
	readonly status: Exclude<LoopStatus, "idle">;
	readonly startedAt: number;
	readonly nextRunAt?: number;
	readonly lastStartedAt?: number;
	readonly lastFinishedAt?: number;
	readonly lastError?: string;
}

export type LoopAction =
	| { type: "start"; prompt: string; intervalMs: number }
	| { type: "stop" }
	| { type: "tick"; iteration: number };

export type LoopTickHandler = (
	iteration: number,
	prompt: string,
	signal: AbortSignal,
) => void | Promise<void>;

export type LoopStateHandler = (state: Readonly<LoopState> | null) => void;

const MIN_INTERVAL_MS = 100;

export class LoopManager {
	private state: LoopState | null = null;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private controller: AbortController | null = null;
	private generation = 0;
	private onTick?: LoopTickHandler;
	private onStateChange?: LoopStateHandler;

	setOnTick(cb: LoopTickHandler): void {
		this.onTick = cb;
	}

	setOnStateChange(cb: LoopStateHandler): void {
		this.onStateChange = cb;
	}

	/** Parse an interval token like "5m", "30s", or "1h". */
	static parseInterval(arg: string): number | null {
		const match = arg.trim().toLowerCase().match(/^(\d+)(ms|s|m|h|d)$/);
		if (!match) return null;
		const value = Number(match[1]);
		if (!Number.isSafeInteger(value) || value <= 0) return null;
		const multiplier =
			match[2] === "ms"
				? 1
				: match[2] === "s"
					? 1_000
					: match[2] === "m"
						? 60_000
						: match[2] === "h"
							? 3_600_000
							: 86_400_000;
		const interval = value * multiplier;
		return Number.isSafeInteger(interval) && interval >= MIN_INTERVAL_MS
			? interval
			: null;
	}

	/** Start a loop. The first run occurs after the configured interval. */
	start(prompt: string, intervalMs: number): void {
		const normalizedPrompt = prompt.trim();
		if (!normalizedPrompt) throw new Error("Loop prompt cannot be empty");
		if (!Number.isFinite(intervalMs) || intervalMs < MIN_INTERVAL_MS) {
			throw new Error(`Loop interval must be at least ${MIN_INTERVAL_MS}ms`);
		}
		this.stop();
		const now = Date.now();
		this.state = {
			prompt: normalizedPrompt,
			intervalMs: Math.round(intervalMs),
			iteration: 0,
			status: "scheduled",
			startedAt: now,
			nextRunAt: now + Math.round(intervalMs),
		};
		const generation = ++this.generation;
		this.notify();
		this.schedule(generation, intervalMs);
	}

	/** Stop scheduling and cooperatively cancel an active callback. */
	stop(): void {
		this.generation++;
		if (this.timer) clearTimeout(this.timer);
		this.timer = null;
		this.controller?.abort();
		this.controller = null;
		this.state = null;
		this.notify();
	}

	isActive(): boolean {
		return this.state !== null;
	}

	/** Return a detached immutable snapshot, never the manager's live state. */
	getState(): Readonly<LoopState> | null {
		return this.state ? Object.freeze({ ...this.state }) : null;
	}

	handleAction(action: LoopAction): void {
		if (action.type === "start") this.start(action.prompt, action.intervalMs);
		else if (action.type === "stop") this.stop();
	}

	private schedule(generation: number, delayMs: number): void {
		this.timer = setTimeout(() => {
			void this.runTick(generation);
		}, delayMs);
	}

	private async runTick(generation: number): Promise<void> {
		if (generation !== this.generation || !this.state) return;
		this.timer = null;
		this.controller = new AbortController();
		const iteration = this.state.iteration + 1;
		this.state = {
			...this.state,
			iteration,
			status: "running",
			nextRunAt: undefined,
			lastStartedAt: Date.now(),
			lastError: undefined,
		};
		this.notify();

		let lastError: string | undefined;
		try {
			await this.onTick?.(iteration, this.state.prompt, this.controller.signal);
		} catch (error) {
			lastError = error instanceof Error ? error.message : String(error);
		}

		if (generation !== this.generation || !this.state) return;
		this.controller = null;
		const now = Date.now();
		this.state = {
			...this.state,
			status: "scheduled",
			lastFinishedAt: now,
			lastError,
			nextRunAt: now + this.state.intervalMs,
		};
		this.notify();
		this.schedule(generation, this.state.intervalMs);
	}

	private notify(): void {
		this.onStateChange?.(this.getState());
	}
}
