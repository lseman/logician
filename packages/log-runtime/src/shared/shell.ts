// ── Shell utilities ──────────────────────────────────────────────────────────────
// Shell configuration, environment, and process tree killing.
// Ported from Pi (packages/coding-agent/src/utils/shell.ts).

import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

export interface ShellConfig {
	shell: string;
	args: string[];
}

/**
 * Find bash on PATH by searching each directory.
 */
function findBashOnPath(): string | null {
	const pathEnv = process.env.PATH ?? "";
	const entries = pathEnv.split(path.delimiter);
	const names = process.platform === "win32" ? ["bash.exe", "bash"] : ["bash"];

	for (const dir of entries) {
		for (const name of names) {
			const fullPath = path.join(dir, name);
			if (existsSync(fullPath)) {
				return fullPath;
			}
		}
	}
	return null;
}

/**
 * Get shell configuration for executing commands.
 */
export function getShellConfig(customShellPath?: string): ShellConfig {
	if (customShellPath) {
		if (existsSync(customShellPath)) {
			return { shell: customShellPath, args: ["-c"] };
		}
		throw new Error(`Custom shell path not found: ${customShellPath}`);
	}

	if (process.platform === "win32") {
		const paths: string[] = [];
		const programFiles = process.env.ProgramFiles;
		if (programFiles) {
			paths.push(`${programFiles}\\Git\\bin\\bash.exe`);
		}
		const programFilesX86 = process.env["ProgramFiles(x86)"];
		if (programFilesX86) {
			paths.push(`${programFilesX86}\\Git\\bin\\bash.exe`);
		}

		for (const p of paths) {
			if (existsSync(p)) {
				return { shell: p, args: ["-c"] };
			}
		}

		const bashOnPath = findBashOnPath();
		if (bashOnPath) {
			return { shell: bashOnPath, args: ["-c"] };
		}

		throw new Error(
			"No bash shell found. Options:\n" +
				"  1. Install Git for Windows: https://git-scm.com/download/win\n" +
				"  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n" +
				"  3. Set shellPath in settings.json\n\n" +
				`Searched Git Bash in:\n${paths.map(p => `  ${p}`).join("\n")}`,
		);
	}

	// Unix: try /bin/bash, then bash on PATH, then fallback to sh
	if (existsSync("/bin/bash")) {
		return { shell: "/bin/bash", args: ["-c"] };
	}

	const bashOnPath = findBashOnPath();
	if (bashOnPath) {
		return { shell: bashOnPath, args: ["-c"] };
	}

	return { shell: "sh", args: ["-c"] };
}

/** Resolve the conventional project-local virtual environment, if present. */
export function getProjectVirtualEnv(cwd?: string): string | undefined {
	if (!cwd) return undefined;
	const virtualEnv = path.join(cwd, ".venv");
	const executables = path.join(
		virtualEnv,
		process.platform === "win32" ? "Scripts" : "bin",
	);
	return existsSync(executables) ? virtualEnv : undefined;
}

/** Read the Python version recorded by a virtual environment. */
export function getVirtualEnvPythonVersion(
	virtualEnv?: string,
): string | undefined {
	if (!virtualEnv) return undefined;
	try {
		const config = readFileSync(path.join(virtualEnv, "pyvenv.cfg"), "utf8");
		const match = config.match(/^version\s*=\s*(\S+)\s*$/im);
		return match?.[1];
	} catch {
		return undefined;
	}
}

/**
 * Get a shell environment with a project-local .venv activated when present.
 */
export function getShellEnv(cwd?: string): NodeJS.ProcessEnv {
	const pathKey =
		Object.keys(process.env).find(key => key.toLowerCase() === "path") ??
		"PATH";
	let currentPath = process.env[pathKey] ?? "";
	const virtualEnv = getProjectVirtualEnv(cwd);

	if (virtualEnv) {
		const executables = path.join(
			virtualEnv,
			process.platform === "win32" ? "Scripts" : "bin",
		);
		const remainingEntries = currentPath
			.split(path.delimiter)
			.filter(entry => entry !== executables);
		currentPath = [executables, ...remainingEntries].join(path.delimiter);
	}

	return {
		...process.env,
		[pathKey]: currentPath,
		...(virtualEnv ? { VIRTUAL_ENV: virtualEnv } : {}),
	};
}

/** Activate a project-local .venv for the Logician process and its children. */
export function activateProjectVirtualEnv(cwd?: string): string | undefined {
	const virtualEnv = getProjectVirtualEnv(cwd);
	if (!virtualEnv) return undefined;
	const env = getShellEnv(cwd);
	const pathKey =
		Object.keys(process.env).find(key => key.toLowerCase() === "path") ??
		"PATH";
	process.env[pathKey] = env[pathKey];
	process.env.VIRTUAL_ENV = virtualEnv;
	return virtualEnv;
}

/**
 * Kill a process and all its children (cross-platform).
 * Uses process group kill on Unix (-pid) and taskkill /T on Windows.
 */
export function killProcessTree(pid: number): void {
	if (process.platform === "win32") {
		try {
			spawn("taskkill", ["/F", "/T", "/PID", String(pid)], {
				stdio: "ignore",
				detached: true,
				windowsHide: true,
			});
		} catch (_e: unknown) {
			// Ignore errors if taskkill fails
		}
	} else {
		try {
			process.kill(-pid, "SIGKILL");
		} catch (_e: unknown) {
			// Fallback to killing just the child if process group kill fails
			try {
				process.kill(pid, "SIGKILL");
			} catch (_e: unknown) {
				// Process already dead
			}
		}
	}
}

// ============================================================================
// Detached child process tracking
// ============================================================================

const trackedDetachedPids = new Set<number>();

/**
 * Track a detached child PID for later cleanup.
 */
export function trackDetachedChildPid(pid: number): void {
	trackedDetachedPids.add(pid);
}

/**
 * Untrack a detached child PID.
 */
export function untrackDetachedChildPid(pid: number): void {
	trackedDetachedPids.delete(pid);
}

/**
 * Kill all tracked detached child processes.
 */
export function killAllTrackedChildren(): void {
	for (const pid of trackedDetachedPids) {
		try {
			killProcessTree(pid);
		} catch (_e: unknown) {
			// Ignore errors for already-dead processes
		}
	}
	trackedDetachedPids.clear();
}
