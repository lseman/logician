import fs from 'fs';

const file = 'packages/agent-core/src/application/agent-bridge.ts';
let content = fs.readFileSync(file, 'utf8');

// Replace the whole problematic block with a cleaner async implementation
const old = `		return (async () => {
			const { spawn } = require("node:child_process");
			const { getShellConfig } = await import("../../infrastructure/tools/shell.ts");
			const { shell, args: shellArgs } = getShellConfig();

			const child = spawn(shell, [...shellArgs, command], {
				cwd: this.cwd,
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
				resolve({
					output: output || "(no output)",
					exitCode: code ?? 1,
				});
			});

			child.on("error", (err: Error) => {
				reject(err);
			});
		});`;

const newCode = `		const { spawn } = await import("node:child_process");
		const { getShellConfig } = await import("../../infrastructure/tools/shell.ts");
		const { shell, args: shellArgs } = getShellConfig();

		return new Promise((resolve, reject) => {
			const child = spawn(shell, [...shellArgs, command], {
				cwd: this.cwd,
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
				resolve({
					output: output || "(no output)",
					exitCode: code ?? 1,
				});
			});

			child.on("error", (err: Error) => {
				reject(err);
			});
		});`;

content = content.replace(old, newCode);
fs.writeFileSync(file, content, 'utf8');
console.log('Fixed');
