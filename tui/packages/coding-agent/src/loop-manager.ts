// ── Loop Manager ────────────────────────────────────────────────────────────────
// Runs a prompt repeatedly on a timer. Start with /loop <prompt> [interval],
// stop with /loop stop or pressing Esc while a loop is active.

export interface LoopState {
	prompt: string;
	intervalMs: number;
	timer: ReturnType<typeof setInterval> | null;
	iteration: number;
}

export type LoopAction =
	| { type: "start"; prompt: string; intervalMs: number }
	| { type: "stop" }
	| { type: "tick"; iteration: number };

export class LoopManager {
	private state: LoopState | null = null;
	private onTick?: (iteration: number, prompt: string) => void;

	setOnTick(cb: (iteration: number, prompt: string) => void): void {
		this.onTick = cb;
	}

	/** Parse interval token like "5m", "30s", "1h" into ms. Returns null if not an interval. */
	static parseInterval(arg: string): number | null {
		const m = arg.match(/^(\d+)(s|m|h|d)$/);
		if (!m) return null;
		const [, value, unit] = m;
		const n = parseInt(value, 10);
		switch (unit) {
			case "s":
				return n * 1000;
			case "m":
				return n * 60_000;
			case "h":
				return n * 3_600_000;
			case "d":
				return n * 86_400_000;
		}
		return null;
	}

	/** Start a new loop. Stops any existing loop first. */
	start(prompt: string, intervalMs: number): void {
		this.stop();
		this.state = { prompt, intervalMs, timer: null, iteration: 0 };
		this.state.timer = setInterval(() => {
			if (!this.state) return;
			this.state.iteration++;
			this.onTick?.(this.state.iteration, this.state.prompt);
		}, intervalMs);
	}

	/** Stop the current loop. */
	stop(): void {
		if (this.state?.timer) {
			clearInterval(this.state.timer);
			this.state.timer = null;
		}
		this.state = null;
	}

	/** Check if a loop is currently active. */
	isActive(): boolean {
		return this.state !== null && this.state.timer !== null;
	}

	/** Get the current loop state (for display). */
	getState(): LoopState | null {
		return this.state;
	}

	/** Process an action (start/stop). */
	handleAction(action: LoopAction): void {
		switch (action.type) {
			case "start":
				this.start(action.prompt, action.intervalMs);
				break;
			case "stop":
				this.stop();
				break;
		}
	}
}
