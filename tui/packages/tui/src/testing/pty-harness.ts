import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

export interface PtyAction {
	afterMs: number;
	send: string;
}

export interface PtyRunOptions {
	command: string;
	args: string[];
	cwd: string;
	env?: Record<string, string>;
	actions?: PtyAction[];
	timeoutMs?: number;
	columns?: number;
	rows?: number;
}

export interface PtyRunResult {
	output: string;
	exitCode: number | null;
}

/** Run a command in a real Unix PTY and capture the raw terminal stream. */
export async function runInPty(options: PtyRunOptions): Promise<PtyRunResult> {
	const driver = fileURLToPath(new URL("./pty-driver.py", import.meta.url));
	const payload = JSON.stringify({
		...options,
		env: { ...process.env, ...(options.env ?? {}) },
	});
	return new Promise<PtyRunResult>((resolve, reject) => {
		const child = spawn("python3", [driver], {
			cwd: path.dirname(driver),
			stdio: ["pipe", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => { stdout += chunk; });
		child.stderr.on("data", (chunk: string) => { stderr += chunk; });
		child.once("error", reject);
		child.once("close", (code) => {
			if (code !== 0) {
				reject(new Error(`PTY driver failed (${code}): ${stderr || stdout}`));
				return;
			}
			try {
				resolve(JSON.parse(stdout) as PtyRunResult);
			} catch {
				reject(new Error(`PTY driver returned invalid JSON: ${stdout}\n${stderr}`));
			}
		});
		child.stdin.end(payload);
	});
}

export function stripTerminalControls(value: string): string {
	return value
		.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
		.replace(/\x1b[()][0-2A-Z]/g, "")
		.replace(/\r/g, "");
}
