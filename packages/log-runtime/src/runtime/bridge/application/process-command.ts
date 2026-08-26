import { spawn } from "node:child_process";
import { getShellConfig } from "../../../capabilities/tools/support/utils/shell.ts";

export interface ProcessCommandResult {
	output: string;
	exitCode: number;
}

/** Executes one user-requested shell command in the runtime working directory. */
export function executeProcessCommand(
	cwd: string,
	command: string,
): Promise<ProcessCommandResult> {
	const { shell, args } = getShellConfig();
	return new Promise((resolve, reject) => {
		const child = spawn(shell, [...args, command], {
			cwd,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let output = "";
		child.stdout?.on("data", (data: Buffer) => {
			output += data.toString();
		});
		child.stderr?.on("data", (data: Buffer) => {
			output += data.toString();
		});
		child.on("close", (code: number | null) => {
			resolve({ output: output || "(no output)", exitCode: code ?? 1 });
		});
		child.on("error", reject);
	});
}
