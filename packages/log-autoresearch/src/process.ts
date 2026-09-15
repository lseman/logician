/**
 * Generic subprocess execution: run a script with a timeout, capture
 * bounded stdout/stderr, and kill the process tree on timeout. No
 * autoresearch-specific knowledge — used by run_experiment and checks.sh.
 */

import { spawn } from "node:child_process";

export function killTree(
	pid: number,
	signal: NodeJS.Signals = "SIGTERM",
): void {
	try {
		process.kill(-pid, signal);
	} catch {
		try {
			process.kill(pid, signal);
		} catch {
			// Process may have already exited
		}
	}
}

export interface ProcessResult {
	code: number | null;
	stdout: string;
	stderr: string;
	killed: boolean;
}

const PROCESS_OUTPUT_LIMIT_BYTES = 1024 * 1024;

export function appendOutputTail(chunks: Buffer[], chunk: Buffer): void {
	chunks.push(chunk);
	let total = chunks.reduce((sum, current) => sum + current.length, 0);
	while (total > PROCESS_OUTPUT_LIMIT_BYTES && chunks.length > 1) {
		total -= chunks.shift()?.length ?? 0;
	}
	if (total > PROCESS_OUTPUT_LIMIT_BYTES && chunks.length === 1) {
		chunks[0] = chunks[0].subarray(-PROCESS_OUTPUT_LIMIT_BYTES);
	}
}

export function runScript(
	scriptPath: string,
	cwd: string,
	timeoutMs: number,
): Promise<ProcessResult> {
	return new Promise((resolve, reject) => {
		const child = spawn("bash", [scriptPath], {
			cwd,
			detached: true,
			stdio: ["ignore", "pipe", "pipe"],
		});
		const stdout: Buffer[] = [];
		const stderr: Buffer[] = [];
		child.stdout?.on("data", (chunk: Buffer) =>
			appendOutputTail(stdout, chunk),
		);
		child.stderr?.on("data", (chunk: Buffer) =>
			appendOutputTail(stderr, chunk),
		);
		let killed = false;
		let settled = false;
		let forceKillTimer: NodeJS.Timeout | undefined;
		const timer =
			timeoutMs > 0
				? setTimeout(() => {
						killed = true;
						if (child.pid) {
							const pid = child.pid;
							killTree(pid);
							forceKillTimer = setTimeout(
								() => killTree(pid, "SIGKILL"),
								1_000,
							);
						}
					}, timeoutMs)
				: undefined;
		child.once("error", error => {
			if (timer) clearTimeout(timer);
			if (forceKillTimer) clearTimeout(forceKillTimer);
			if (settled) return;
			settled = true;
			reject(error);
		});
		child.once("close", code => {
			if (timer) clearTimeout(timer);
			if (forceKillTimer) clearTimeout(forceKillTimer);
			if (settled) return;
			settled = true;
			resolve({
				code,
				stdout: Buffer.concat(stdout).toString("utf8"),
				stderr: Buffer.concat(stderr).toString("utf8"),
				killed,
			});
		});
	});
}

export function truncateTail(
	output: string,
	maxLines: number,
	maxBytes: number,
): {
	content: string;
	truncated: boolean;
	outputLines: number;
	totalLines: number;
} {
	const lines = output.split("\n");
	const totalLines = lines.length;
	let truncated = false;

	// Limit to maxLines
	if (lines.length > maxLines) {
		lines.splice(0, lines.length - maxLines);
		truncated = true;
	}

	// Limit to maxBytes
	const content = lines.join("\n");
	if (Buffer.byteLength(content) > maxBytes) {
		const limited = output.slice(-maxBytes);
		return {
			content: limited,
			truncated: true,
			outputLines: Math.min(maxLines, output.split("\n").length),
			totalLines,
		};
	}

	return { content, truncated, outputLines: lines.length, totalLines };
}

export function formatSize(bytes: number): string {
	if (bytes < 1024) return `${bytes}B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}
