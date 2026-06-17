import {
	existsSync,
	readFileSync,
	writeFileSync,
	appendFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";

/** Validated configuration with warnings collected during load. */
export interface ResolvedLogicianConfig {
	path?: string;
	config: LogicianTuiConfig;
	warnings: string[];
}

/** Known config keys for unknown-field detection. */
const KNOWN_KEYS = new Set([
	"baseUrl", "llmUrl", "model", "theme", "systemPrompt", "chatTemplate",
	"temperature", "maxTokens", "maxIterations", "toolExecution",
	"contextWindow", "contextWindowTokens", "hooks", "mcp", "mcpServers",
	"mcpEager", "webSearch", "permissionMode", "permissions",
	"steeringInterrupt", "maxTotalTokens",
	"loopDetectionEnabled", "guardsEnabled", "continuationEnabled",
]);

const WEB_SEARCH_KEYS = new Set(["baseUrl", "maxResults"]);
const PERMISSIONS_KEYS = new Set(["allow", "deny"]);

/** Validate a URL string (non-empty, starts with http:// or https://). */
function isValidUrl(v: unknown): boolean {
	if (typeof v !== "string") return false;
	const s = v.trim();
	return s.startsWith("http://") || s.startsWith("https://");
}

/** Validate range: value >= min and (max undefined or value <= max). */
function inRange(v: number, min: number, max?: number): boolean {
	return Number.isFinite(v) && v >= min && (max === undefined || v <= max);
}

/** Emit a warning and record it. */
function warn(warnings: string[], msg: string): void {
	// eslint-disable-next-line no-console
	console.warn(`[logician config] ${msg}`);
	warnings.push(msg);
}

export function validateConfig(
	raw: unknown,
	warnings: string[],
): LogicianTuiConfig {
	if (typeof raw !== "object" || raw === null) {
		warn(warnings, "Config is not an object — ignoring.");
		return {};
	}

	const obj = raw as Record<string, unknown>;
	const cfg: LogicianTuiConfig = {};

	// Check for unknown top-level keys.
	for (const key of Object.keys(obj)) {
		if (!KNOWN_KEYS.has(key)) {
			warn(warnings, `Unknown config key: "${key}".`);
		}
	}

	// String fields (URLs validated).
	if (obj.baseUrl !== undefined) {
		if (isValidUrl(obj.baseUrl)) {
			cfg.baseUrl = configString(obj.baseUrl);
		} else {
			warn(warnings, "\"baseUrl\" must be a valid http/https URL.");
		}
	}
	if (obj.llmUrl !== undefined) {
		if (isValidUrl(obj.llmUrl)) {
			cfg.llmUrl = configString(obj.llmUrl);
		} else {
			warn(warnings, "\"llmUrl\" must be a valid http/https URL.");
		}
	}

	// Simple strings.
	cfg.model = configString(obj.model);
	cfg.theme = configString(obj.theme);
	cfg.systemPrompt = configString(obj.systemPrompt);
	cfg.chatTemplate = configString(obj.chatTemplate);

	// temperature: 0–2
	if (obj.temperature !== undefined) {
		const t = configNumber(obj.temperature);
		if (t !== undefined) {
			if (!inRange(t, 0, 2)) {
				warn(warnings, `"temperature" out of range [0,2], value: ${t}. Clamping to [0,2].`);
				cfg.temperature = Math.max(0, Math.min(2, t));
			} else {
				cfg.temperature = t;
			}
		}
	}

	// maxTokens: > 0
	if (obj.maxTokens !== undefined) {
		const mt = configNumber(obj.maxTokens);
		if (mt !== undefined) {
			if (mt <= 0) {
				warn(warnings, `"maxTokens" must be > 0, got ${mt}. Ignored.`);
			} else {
				cfg.maxTokens = mt;
			}
		}
	}

	// maxIterations: > 0
	if (obj.maxIterations !== undefined) {
		const mi = configNumber(obj.maxIterations);
		if (mi !== undefined) {
			if (mi <= 0) {
				warn(warnings, `"maxIterations" must be > 0, got ${mi}. Ignored.`);
			} else {
				cfg.maxIterations = mi;
			}
		}
	}

	// maxTotalTokens: > 0
	if (obj.maxTotalTokens !== undefined) {
		const mt = configNumber(obj.maxTotalTokens);
		if (mt !== undefined) {
			if (mt <= 0) {
				warn(warnings, `"maxTotalTokens" must be > 0, got ${mt}. Ignored.`);
			} else {
				cfg.maxTotalTokens = mt;
			}
		}
	}

	// contextWindow / contextWindowTokens: > 0
	if (obj.contextWindow !== undefined) {
		const cw = configNumber(obj.contextWindow);
		if (cw !== undefined && cw > 0) cfg.contextWindow = cw;
	}
	if (obj.contextWindowTokens !== undefined) {
		const cwt = configNumber(obj.contextWindowTokens);
		if (cwt !== undefined && cwt > 0) cfg.contextWindowTokens = cwt;
	}

	// Enum fields.
	if (obj.toolExecution !== undefined) {
		const te = configString(obj.toolExecution);
		if (te !== "sequential" && te !== "parallel") {
			warn(warnings, `"toolExecution" must be "sequential" or "parallel", got: "${te}".`);
		} else {
			cfg.toolExecution = te as "sequential" | "parallel";
		}
	}
	if (obj.permissionMode !== undefined) {
		const pm = configString(obj.permissionMode);
		const validModes = ["acceptAll", "acceptEdits", "ask", "plan"];
		if (!validModes.includes(pm ?? "")) {
			warn(warnings, `"permissionMode" invalid, got: "${pm}".`);
		} else {
			cfg.permissionMode = pm as LogicianTuiConfig["permissionMode"];
		}
	}

	// Boolean fields.
	cfg.hooks = configBool(obj.hooks);
	cfg.mcpEager = configBool(obj.mcpEager);
	cfg.steeringInterrupt = configBool(obj.steeringInterrupt);
	cfg.loopDetectionEnabled = configBool(obj.loopDetectionEnabled);
	cfg.guardsEnabled = configBool(obj.guardsEnabled);
	cfg.continuationEnabled = configBool(obj.continuationEnabled);

	// MCP fields (passthrough, but warn on unknown sub-keys).
	if (obj.mcp !== undefined && typeof obj.mcp === "object") {
		cfg.mcp = obj.mcp as Record<string, unknown>;
	}
	if (obj.mcpServers !== undefined && typeof obj.mcpServers === "object") {
		cfg.mcpServers = obj.mcpServers as Record<string, unknown>;
	}

	// webSearch sub-object.
	if (obj.webSearch !== undefined) {
		if (typeof obj.webSearch !== "object" || obj.webSearch === null) {
			warn(warnings, "\"webSearch\" must be an object.");
		} else {
			const ws = obj.webSearch as Record<string, unknown>;
			const wscfg: NonNullable<LogicianTuiConfig["webSearch"]> = {};
			for (const key of Object.keys(ws)) {
				if (!WEB_SEARCH_KEYS.has(key)) {
					warn(warnings, `Unknown webSearch key: "${key}".`);
				}
			}
			if (ws.baseUrl !== undefined) {
				if (!isValidUrl(ws.baseUrl)) {
					warn(warnings, "\"webSearch.baseUrl\" must be a valid http/https URL.");
				} else {
					wscfg.baseUrl = configString(ws.baseUrl);
				}
			}
			if (ws.maxResults !== undefined) {
				const mr = configNumber(ws.maxResults);
				if (mr !== undefined && mr > 0 && mr <= 100) {
					wscfg.maxResults = mr;
				} else {
					warn(warnings, `"webSearch.maxResults" must be 1–100, got: ${mr}.`);
				}
			}
			cfg.webSearch = Object.keys(wscfg).length > 0 ? wscfg : undefined;
		}
	}

	// permissions sub-object.
	if (obj.permissions !== undefined) {
		if (typeof obj.permissions !== "object" || obj.permissions === null) {
			warn(warnings, "\"permissions\" must be an object.");
		} else {
			const perms = obj.permissions as Record<string, unknown>;
			const percfg: NonNullable<LogicianTuiConfig["permissions"]> = {};
			for (const key of Object.keys(perms)) {
				if (!PERMISSIONS_KEYS.has(key)) {
					warn(warnings, `Unknown permissions key: "${key}".`);
				}
			}
			if (Array.isArray(perms.allow)) {
				percfg.allow = perms.allow.filter((v): v is string => typeof v === "string" && v.trim().length > 0);
			}
			if (Array.isArray(perms.deny)) {
				percfg.deny = perms.deny.filter((v): v is string => typeof v === "string" && v.trim().length > 0);
			}
			cfg.permissions = Object.keys(percfg).length > 0 ? percfg : undefined;
		}
	}

	// Strip undefined values so the returned config only contains set fields.
	return Object.fromEntries(Object.entries(cfg).filter(([, v]) => v !== undefined)) as LogicianTuiConfig;
}

export interface LogicianTuiConfig {
	baseUrl?: string;
	llmUrl?: string;
	model?: string;
	theme?: string;
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
	// Safeguard options (match pi's trust-model approach by default).
	loopDetectionEnabled?: boolean; // OFF by default
	guardsEnabled?: boolean; // OFF by default
	continuationEnabled?: boolean; // OFF by default
}

export function loadLogicianConfig(
	cwd = process.cwd(),
): ResolvedLogicianConfig {
	const configPath = findLogicianConfig(cwd);
	if (!configPath) return { config: {}, warnings: [] };
	try {
		const raw = JSON.parse(readFileSync(configPath, "utf8"));
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

/** Save a single config field to the global user config file (~/.logician/logician.json). */
export function saveConfigField(key: string, value: unknown): boolean {
	try {
		const home = process.env.HOME || "";
		if (!home) return false;
		const configPath = join(home, ".logician", "logician.json");
		const dir = dirname(configPath);
		if (!existsSync(dir)) {
			appendFileSync(configPath, "{}\n");
		}
		const raw = existsSync(configPath)
			? (JSON.parse(readFileSync(configPath, "utf8")) as Record<
					string,
					unknown
				>)
			: {};
		raw[key] = value;
		writeFileSync(configPath, JSON.stringify(raw, null, 2) + "\n");
		return true;
	} catch {
		return false;
	}
}
