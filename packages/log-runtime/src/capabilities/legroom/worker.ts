import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface, type Interface } from "node:readline";

export interface LegroomSdkConfig {
	mode?: "off" | "sdk";
	python?: string;
	args?: string[];
	failOpen?: boolean;
	timeoutMs?: number;
	config?: Record<string, unknown>;
}

interface PendingRequest {
	resolve: (messages: Record<string, unknown>[]) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
}

interface WorkerResponse {
	id: string;
	ok: boolean;
	messages?: Record<string, unknown>[];
	error?: string;
}

function parseResponse(line: string): WorkerResponse | undefined {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch {
		return undefined;
	}
	if (!value || typeof value !== "object") return undefined;
	const response = value as Record<string, unknown>;
	if (typeof response.id !== "string" || typeof response.ok !== "boolean")
		return undefined;
	if (
		response.messages !== undefined &&
		(!Array.isArray(response.messages) ||
			!response.messages.every(
				message => message !== null && typeof message === "object",
			))
	)
		return undefined;
	return response as unknown as WorkerResponse;
}

/** A lazy, persistent JSONL client for Legroom's Python SDK worker. */
export class LegroomWorker {
	private process?: ChildProcessWithoutNullStreams;
	private lines?: Interface;
	private readonly pending = new Map<string, PendingRequest>();
	private nextId = 0;
	private stopping = false;
	private stderrTail = "";

	constructor(private readonly options: LegroomSdkConfig) {}

	async compress(
		messages: Record<string, unknown>[],
		model: string,
	): Promise<Record<string, unknown>[]> {
		try {
			return await this.request(messages, model);
		} catch (error) {
			if (this.options.failOpen !== false) return messages;
			throw error;
		}
	}

	private request(
		messages: Record<string, unknown>[],
		model: string,
	): Promise<Record<string, unknown>[]> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve, reject, timer });
			const payload = JSON.stringify({
				id,
				method: "compress",
				messages,
				model: model || "gpt-4o",
				config: this.options.config ?? {},
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	private ensureStarted(): ChildProcessWithoutNullStreams {
		if (this.process && this.process.exitCode === null) return this.process;
		this.stopping = false;
		this.stderrTail = "";
		const child = spawn(
			this.options.python ?? "python3",
			this.options.args ?? ["-m", "legroom.sdk_worker"],
			{ stdio: ["pipe", "pipe", "pipe"] },
		);
		this.process = child;
		this.lines = createInterface({ input: child.stdout });
		this.lines.on("line", line => this.handleLine(line));
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk: string) => {
			this.stderrTail = `${this.stderrTail}${chunk}`.slice(-4_096);
		});
		child.on("error", error => this.failAll(error));
		child.on("exit", (code, signal) => {
			if (this.process === child) this.process = undefined;
			if (!this.stopping) {
				const detail = this.stderrTail.trim();
				this.failAll(
					new Error(
						`Legroom SDK worker exited (${signal ?? code ?? "unknown"})${detail ? `: ${detail}` : ""}`,
					),
				);
			}
		});
		return child;
	}

	private handleLine(line: string): void {
		const response = parseResponse(line);
		if (!response) return;
		const pending = this.pending.get(response.id);
		if (!pending) return;
		clearTimeout(pending.timer);
		this.pending.delete(response.id);
		if (!response.ok) {
			pending.reject(new Error(response.error ?? "Legroom SDK request failed"));
			return;
		}
		if (!response.messages) {
			pending.reject(new Error("Legroom SDK response omitted messages"));
			return;
		}
		pending.resolve(response.messages);
	}

	private failAll(error: Error): void {
		for (const pending of this.pending.values()) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}
		this.pending.clear();
	}

	close(): void {
		this.stopping = true;
		this.lines?.close();
		this.lines = undefined;
		this.failAll(new Error("Legroom SDK worker closed"));
		const child = this.process;
		this.process = undefined;
		if (!child) return;
		child.stdin.end();
		if (child.exitCode === null) child.kill("SIGTERM");
	}
}
