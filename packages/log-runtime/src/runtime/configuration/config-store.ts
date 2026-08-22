import {
	existsSync,
	mkdirSync,
	readFileSync,
	renameSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { stripJsonComments } from "../../capabilities/tools/support/utils/json-utils.ts";
import { type ResolvedLogicianConfig, validateConfig } from "./config.ts";

export function loadLogicianConfig(
	cwd = process.cwd(),
): ResolvedLogicianConfig {
	const configPath = findLogicianConfig(cwd);
	if (!configPath) return { config: {}, warnings: [] };
	return loadLogicianConfigFile(configPath);
}

export function loadGlobalLogicianConfig(
	home = process.env.HOME,
): ResolvedLogicianConfig {
	if (!home) return { config: {}, warnings: [] };
	const configPath = join(home, ".logician", "settings.json");
	if (!existsSync(configPath)) return { config: {}, warnings: [] };
	return loadLogicianConfigFile(configPath);
}

function loadLogicianConfigFile(configPath: string): ResolvedLogicianConfig {
	try {
		const raw = JSON.parse(stripJsonComments(readFileSync(configPath, "utf8")));
		const warnings: string[] = [];
		if (raw && typeof raw === "object") {
			return {
				path: configPath,
				config: validateConfig(raw, warnings),
				warnings,
			};
		}
		return { config: {}, warnings: ["Config root is not an object."] };
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`Failed to read ${configPath}: ${message}`);
	}
}

export function findLogicianConfig(cwd = process.cwd()): string | null {
	const envPath = process.env.LOGICIAN_CONFIG?.trim();
	if (envPath) {
		const resolved = resolve(
			envPath.replace(/^~(?=$|\/)/, process.env.HOME || ""),
		);
		return existsSync(resolved) ? resolved : null;
	}
	let directory = resolve(cwd);
	while (true) {
		const candidate = join(directory, ".logician.json");
		if (existsSync(candidate)) return candidate;
		const parent = dirname(directory);
		if (parent === directory) break;
		directory = parent;
	}
	const home = process.env.HOME;
	if (!home) return null;
	const global = join(home, ".logician", "settings.json");
	return existsSync(global) ? global : null;
}

export function saveConfigField(key: string, value: unknown): boolean {
	return updateGlobalConfig(raw => {
		if (value === undefined) delete raw[key];
		else raw[key] = value;
	});
}

export function saveConfigNestedField(
	section: string,
	key: string,
	value: unknown,
): boolean {
	return updateGlobalConfig(raw => {
		const current =
			raw[section] &&
			typeof raw[section] === "object" &&
			!Array.isArray(raw[section])
				? (raw[section] as Record<string, unknown>)
				: {};
		if (value === undefined) {
			const next = { ...current };
			delete next[key];
			if (Object.keys(next).length) raw[section] = next;
			else delete raw[section];
		} else raw[section] = { ...current, [key]: value };
	});
}

export function updateConfigFile(
	configPath: string,
	mutate: (raw: Record<string, unknown>) => void,
): boolean {
	let temporaryPath: string | undefined;
	try {
		const directory = dirname(configPath);
		mkdirSync(directory, { recursive: true });
		const raw = existsSync(configPath)
			? (JSON.parse(
					stripJsonComments(readFileSync(configPath, "utf8")),
				) as Record<string, unknown>)
			: {};
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return false;
		mutate(raw);
		temporaryPath = join(
			directory,
			`.${configPath.split(/[\\/]/).at(-1)}.${process.pid}.${Date.now()}.tmp`,
		);
		const mode = existsSync(configPath) ? statSync(configPath).mode : 0o600;
		writeFileSync(temporaryPath, `${JSON.stringify(raw, null, 2)}\n`, {
			encoding: "utf8",
			mode,
		});
		renameSync(temporaryPath, configPath);
		temporaryPath = undefined;
		return true;
	} catch {
		if (temporaryPath) {
			try {
				rmSync(temporaryPath, { force: true });
			} catch {
				// Best-effort cleanup after a failed atomic replacement.
			}
		}
		return false;
	}
}

function updateGlobalConfig(
	mutate: (raw: Record<string, unknown>) => void,
): boolean {
	const home = process.env.HOME || "";
	if (!home) return false;
	return updateConfigFile(join(home, ".logician", "settings.json"), mutate);
}
