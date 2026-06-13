import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

export interface LogicianTuiConfig {
	baseUrl?: string;
	llmUrl?: string;
	model?: string;
	systemPrompt?: string;
	chatTemplate?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	toolExecution?: "sequential" | "parallel";
	contextWindow?: number;
	contextWindowTokens?: number;
	hooks?: boolean;
	mcp?: Record<string, unknown>;
	mcpServers?: Record<string, unknown>;
	mcpEager?: boolean;
	webSearch?: {
		baseUrl?: string;
		maxResults?: number;
	};
	permissionMode?: "acceptAll" | "acceptEdits" | "ask" | "plan";
	permissions?: {
		allow?: string[];
		deny?: string[];
	};
	steeringInterrupt?: boolean;
	maxTotalTokens?: number;
}

export interface ResolvedLogicianConfig {
	path?: string;
	config: LogicianTuiConfig;
}

export function loadLogicianConfig(
	cwd = process.cwd(),
): ResolvedLogicianConfig {
	const configPath = findLogicianConfig(cwd);
	if (!configPath) return { config: {} };
	try {
		const raw = JSON.parse(readFileSync(configPath, "utf8"));
		return {
			path: configPath,
			config: raw && typeof raw === "object" ? (raw as LogicianTuiConfig) : {},
		};
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

	let dir = resolve(cwd);
	while (true) {
		const candidate = join(dir, ".logician.json");
		if (existsSync(candidate)) return candidate;
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}

	// Fall back to a per-user global config when no project config is found.
	const home = process.env.HOME;
	if (home) {
		const global = join(home, ".logician", "logician.json");
		if (existsSync(global)) return global;
	}
	return null;
}

export function configString(
	value: unknown,
	fallback?: string,
): string | undefined {
	return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

export function configNumber(
	value: unknown,
	fallback?: number,
): number | undefined {
	if (typeof value === "number" && Number.isFinite(value)) return value;
	if (typeof value === "string" && value.trim()) {
		const parsed = Number(value);
		if (Number.isFinite(parsed)) return parsed;
	}
	return fallback;
}

export function configBool(
	value: unknown,
	fallback?: boolean,
): boolean | undefined {
	if (typeof value === "boolean") return value;
	if (typeof value === "string") {
		const clean = value.trim().toLowerCase();
		if (["1", "true", "yes", "on"].includes(clean)) return true;
		if (["0", "false", "no", "off"].includes(clean)) return false;
	}
	return fallback;
}
