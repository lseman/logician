import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface, type Interface } from "node:readline";

interface PendingRequest {
	resolve: (response: Record<string, unknown>) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
}
interface Generation {
	child: ChildProcessWithoutNullStreams;
	lines: Interface;
	pending: Map<string, PendingRequest>;
	stderr: string;
}

/** Owns process generations and request settlement; adapters own protocol payloads. */
export class JsonlWorker {
	private current?: Generation;
	private nextId = 0;
	constructor(
		private readonly options: {
			name: string;
			python?: string;
			args: string[];
			timeoutMs?: number;
			initialize?: Record<string, unknown>;
		},
	) {}

	async request(
		payload: Record<string, unknown>,
	): Promise<Record<string, unknown>> {
		const id = `${this.options.name.toLowerCase()}-${process.pid}-${++this.nextId}`;
		// Serialize before registering a timer or starting a process.
		const line = `${JSON.stringify({ ...payload, id })}\n`;
		const generation = this.ensureStarted();
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.take(generation, id)?.reject(
					new Error(
						`${this.options.name} SDK request timed out after ${timeoutMs}ms`,
					),
				);
			}, timeoutMs);
			generation.pending.set(id, { resolve, reject, timer });
			try {
				generation.child.stdin.write(line, error => {
					if (error)
						this.take(generation, id)?.reject(
							new Error(
								`Unable to write to ${this.options.name} SDK worker: ${error.message}`,
							),
						);
				});
			} catch (error) {
				this.take(generation, id)?.reject(
					error instanceof Error ? error : new Error(String(error)),
				);
			}
		});
	}

	private take(generation: Generation, id: string): PendingRequest | undefined {
		const pending = generation.pending.get(id);
		if (pending) {
			generation.pending.delete(id);
			clearTimeout(pending.timer);
		}
		return pending;
	}

	private ensureStarted(): Generation {
		if (this.current) return this.current;
		const init = this.options.initialize
			? `${JSON.stringify({ ...this.options.initialize, id: `init-${++this.nextId}` })}\n`
			: undefined;
		const child = spawn(this.options.python ?? "python3", this.options.args, {
			stdio: ["pipe", "pipe", "pipe"],
		});
		const generation: Generation = {
			child,
			lines: createInterface({ input: child.stdout }),
			pending: new Map(),
			stderr: "",
		};
		this.current = generation;
		generation.lines.on("line", line => {
			let value: unknown;
			try {
				value = JSON.parse(line);
			} catch {
				return;
			}
			if (!value || typeof value !== "object") return;
			const response = value as Record<string, unknown>;
			if (typeof response.id !== "string" || typeof response.ok !== "boolean")
				return;
			const pending = this.take(generation, response.id);
			if (!pending) return;
			if (response.ok) pending.resolve(response);
			else
				pending.reject(
					new Error(
						typeof response.error === "string"
							? response.error
							: `${this.options.name} SDK request failed`,
					),
				);
		});
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk: string) => {
			generation.stderr = `${generation.stderr}${chunk}`.slice(-4096);
		});
		child.on("error", error => this.dispose(generation, error));
		child.stdin.on("error", error => this.dispose(generation, error));
		// Drain stdout before rejecting requests left over from an exited process.
		child.on("close", (code, signal) => {
			const detail = generation.stderr.trim();
			this.dispose(
				generation,
				new Error(
					`${this.options.name} SDK worker exited (${signal ?? code ?? "unknown"})${detail ? `: ${detail}` : ""}`,
				),
			);
		});
		// Preserve Memoriam's ordered, fire-and-forget initialization protocol.
		if (init)
			child.stdin.write(init, error => {
				if (error) this.dispose(generation, error);
			});
		return generation;
	}

	private dispose(generation: Generation, error: Error): void {
		if (this.current === generation) this.current = undefined;
		generation.lines.close();
		for (const id of generation.pending.keys())
			this.take(generation, id)?.reject(error);
		generation.child.stdin.end();
		if (generation.child.exitCode === null && !generation.child.killed)
			generation.child.kill("SIGTERM");
	}

	close(): void {
		if (this.current)
			this.dispose(
				this.current,
				new Error(`${this.options.name} SDK worker closed`),
			);
	}
}
