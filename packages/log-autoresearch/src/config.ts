/** .auto/config.json: max-iteration cap and an optional working-directory override. */

import * as fs from "node:fs";
import * as path from "node:path";
import { sessionFilePath } from "./paths.ts";

interface AutoresearchConfig {
	maxIterations?: number;
	workingDir?: string;
}

function autoresearchConfigPath(dir: string): string {
	return sessionFilePath(dir, "config");
}

function readConfig(cwd: string): AutoresearchConfig {
	try {
		const configPath = autoresearchConfigPath(cwd);
		if (!fs.existsSync(configPath)) return {};
		return JSON.parse(fs.readFileSync(configPath, "utf-8"));
	} catch {
		return {};
	}
}

export function readMaxExperiments(cwd: string): number | null {
	const config = readConfig(cwd);
	return typeof config.maxIterations === "number" && config.maxIterations > 0
		? Math.floor(config.maxIterations)
		: null;
}

export function resolveWorkDir(ctxCwd: string): string {
	const config = readConfig(ctxCwd);
	if (!config.workingDir) return ctxCwd;
	return path.isAbsolute(config.workingDir)
		? config.workingDir
		: path.resolve(ctxCwd, config.workingDir);
}

export function validateWorkDir(ctxCwd: string): string | null {
	const workDir = resolveWorkDir(ctxCwd);
	if (workDir === ctxCwd) return null;
	try {
		const stat = fs.statSync(workDir);
		if (!stat.isDirectory()) {
			return `workingDir "${workDir}" (from .auto/config.json) is not a directory.`;
		}
	} catch {
		return `workingDir "${workDir}" (from .auto/config.json) does not exist.`;
	}
	return null;
}
