import { spawn } from "node:child_process";

const OUTPUT_LIMIT = 100_000;

export interface ProcessResult {
	exitCode: number | null;
	stdout: string;
	stderr: string;
	durationMs: number;
	timedOut: boolean;
}

export async function runProcess(
	command: string,
	args: string[],
	options: { cwd: string; timeoutMs?: number; env?: NodeJS.ProcessEnv },
): Promise<ProcessResult> {
	const started = performance.now();
	return new Promise(resolve => {
		const child = spawn(command, args, {
			cwd: options.cwd,
			env: options.env ?? process.env,
			stdio: ["ignore", "pipe", "pipe"],
			detached: process.platform !== "win32",
		});
		let stdout = "";
		let stderr = "";
		let timedOut = false;
		const terminate = (signal: NodeJS.Signals) => {
			if (!child.pid) return;
			try {
				process.kill(
					process.platform === "win32" ? child.pid : -child.pid,
					signal,
				);
			} catch {
				child.kill(signal);
			}
		};
		let forceTimer: NodeJS.Timeout | undefined;
		const timer = options.timeoutMs
			? setTimeout(() => {
					timedOut = true;
					terminate("SIGTERM");
					forceTimer = setTimeout(() => terminate("SIGKILL"), 2000);
				}, options.timeoutMs)
			: undefined;
		// Decode across chunk boundaries and cap retention during execution,
		// not only after a potentially long-running agent has exited.
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = (stdout + chunk).slice(-OUTPUT_LIMIT);
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = (stderr + chunk).slice(-OUTPUT_LIMIT);
		});
		child.on("error", error => {
			stderr = `${stderr}${error.message}\n`.slice(-OUTPUT_LIMIT);
		});
		child.on("close", exitCode => {
			if (timer) clearTimeout(timer);
			if (forceTimer) clearTimeout(forceTimer);
			resolve({
				exitCode,
				stdout,
				stderr,
				durationMs: Math.round(performance.now() - started),
				timedOut,
			});
		});
	});
}
