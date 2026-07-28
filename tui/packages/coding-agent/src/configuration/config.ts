import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, isAbsolute } from "node:path";
import type { AgentModelConfig, TruncationConfig } from "@logician/agent-core";

/** Validated configuration with warnings collected during load. */
export interface ResolvedLogicianConfig {
	path?: string;
	config: LogicianTuiConfig;
	warnings: string[];
}

/** Known config keys for unknown-field detection. */
const KNOWN_KEYS = new Set([
	"baseUrl",
	"llmUrl",
	"model",
	"models",
	"theme",
	"systemPrompt",
	"chatTemplate",
	"temperature",
	"maxTokens",
	"maxIterations",
	"autoRetryEnabled",
	"maxRetries",
	"retryBaseDelayMs",
	"turnTimeoutMs",
	"cacheSize",
	"cacheTtlMs",
	"toolExecution",
	"contextWindow",
	"contextWindowTokens",
	"hooks",
	"mcp",
	"mcpServers",
	"mcpEager",
	"webSearch",
	"permissionMode",
	"permissions",
	"steeringInterrupt",
	"maxTotalTokens",
	"guardsEnabled",
	"duplicateGuardEnabled",
	"failureGuardEnabled",
	"continuationEnabled",
	"postEditDiagnostics",
	"lsp",
	"compaction",
	"plugins",
	"inferenceMode",
	"allowedPaths",
	"allowAllPaths",
	"maxParallelAgents",
	"cwd",
	"truncation",
]);
const COMPACTION_KEYS = new Set([
	"enabled",
	"reserveTokens",
	"keepRecentTokens",
]);
const TRUNCATION_KEYS = new Set([
	"toolResultMaxChars",
	"maxLines",
	"grepLineMaxChars",
	"subagentResultMaxChars",
	"compactionSummaryMaxChars",
	"microCompactMaxChars",
	"transcriptMessageMaxChars",
]);
const MICRO_COMPACT_MAX_CHARS_KEYS = new Set(["tool", "assistant", "default"]);

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

	// models: array of model objects for cycling (Ctrl+L model selector)
	if (obj.models !== undefined) {
		if (Array.isArray(obj.models)) {
			const parsed: AgentModelConfig[] = [];
			for (const item of obj.models) {
				if (typeof item === "string" && item.trim()) {
					// Legacy string entry: convert to object
					parsed.push({ name: item.trim(), model: item.trim() });
				} else if (
					typeof item === "object" &&
					item !== null &&
					"model" in item &&
					"name" in item &&
					typeof (item as Record<string, unknown>).name === "string" &&
					typeof (item as Record<string, unknown>).model === "string" &&
					((item as AgentModelConfig).name.trim() ||
						(item as AgentModelConfig).model.trim())
				) {
					const m = item as AgentModelConfig;
					parsed.push({
						name: m.name.trim(),
						model: m.model.trim(),
						url: typeof m.url === "string" ? m.url.trim() : m.url,
					});
				} else {
					warn(
						warnings,
						`"models" entry invalid, got: ${JSON.stringify(item)}.`,
					);
				}
			}
			if (parsed.length > 0) {
				cfg.models = parsed as LogicianTuiConfig["models"];
			}
		} else {
			warn(warnings, "\"models\" must be an array.");
		}
	}
	cfg.theme = configString(obj.theme);
	cfg.systemPrompt = configString(obj.systemPrompt);
	cfg.chatTemplate = configString(obj.chatTemplate);

	// temperature: 0–2
	if (obj.temperature !== undefined) {
		const t = configNumber(obj.temperature);
		if (t !== undefined) {
			if (!inRange(t, 0, 2)) {
				warn(
					warnings,
					`"temperature" out of range [0,2], value: ${t}. Clamping to [0,2].`,
				);
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
			warn(
				warnings,
				`"toolExecution" must be "sequential" or "parallel", got: "${te}".`,
			);
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

	// inferenceMode: pre-defined sampling parameter set (Alt+M in the TUI)
	if (obj.inferenceMode !== undefined) {
		const im = configString(obj.inferenceMode);
		const validModes = ["thinking-general", "thinking-coding", "instruct-general", "instruct-reasoning"];
		if (im && !validModes.includes(im)) {
			warn(
				warnings,
				`"inferenceMode" must be one of: ${validModes.join(", ")}, got: "${im}".`,
			);
		} else if (im) {
			cfg.inferenceMode = im as LogicianTuiConfig["inferenceMode"];
		}
	}
	cfg.guardsEnabled = configBool(obj.guardsEnabled);
	cfg.duplicateGuardEnabled = configBool(obj.duplicateGuardEnabled, true);
	cfg.failureGuardEnabled = configBool(obj.failureGuardEnabled);
	cfg.continuationEnabled = configBool(obj.continuationEnabled, true);
	cfg.postEditDiagnostics = configBool(obj.postEditDiagnostics, true);
	cfg.autoRetryEnabled = configBool(obj.autoRetryEnabled, true);

	for (const [key, minimum, inclusive] of [
		["maxRetries", 0, true],
		["retryBaseDelayMs", 0, true],
		["turnTimeoutMs", 0, false],
		["cacheSize", 0, false],
		["cacheTtlMs", 0, false],
	] as const) {
		if (obj[key] === undefined) continue;
		const value = configNumber(obj[key]);
		const valid =
			value !== undefined && (inclusive ? value >= minimum : value > minimum);
		if (!valid) {
			warn(
				warnings,
				`"${key}" must be ${inclusive ? ">=" : ">"} ${minimum}. Ignored.`,
			);
		} else {
			cfg[key] = value;
		}
	}

	// allowedPaths: array of absolute paths allowed outside CWD.
	if (obj.allowedPaths !== undefined) {
		if (Array.isArray(obj.allowedPaths)) {
			const paths: string[] = [];
			for (const p of obj.allowedPaths) {
				if (typeof p === "string" && p.trim()) {
					const trimmed = p.trim();
					if (!isAbsolute(trimmed)) {
						warn(warnings, `"allowedPaths" entry must be an absolute path: "${trimmed}". Ignored.`);
					} else {
						paths.push(trimmed);
					}
				}
			}
			if (paths.length > 0) cfg.allowedPaths = paths;
		} else {
			warn(warnings, "\"allowedPaths\" must be an array.");
		}
	}

	// allowAllPaths: when true, skip CWD/allowedPaths enforcement.
	if (obj.allowAllPaths !== undefined) {
		cfg.allowAllPaths = configBool(obj.allowAllPaths);
	}

	// cwd: explicit project root.
	if (obj.cwd !== undefined) {
		const cwd = configString(obj.cwd);
		if (cwd !== undefined) {
			const resolved = resolve(cwd);
			if (existsSync(resolved)) {
				cfg.cwd = resolved;
			} else {
				warn(warnings, `"cwd" path does not exist: "${cwd}". Ignored.`);
			}
		}
	}

	// lsp sub-object.
	if (obj.lsp !== undefined) {
		if (typeof obj.lsp !== "object" || obj.lsp === null) {
			warn(warnings, "\"lsp\" must be an object.");
		} else {
			const l = obj.lsp as Record<string, unknown>;
			const lc: {
				enabled?: boolean;
				timeoutMs?: number;
				serverOverrides?: Record<string, {
					command: string;
					args?: string[];
					languageId: string;
				}>;
			} = {};
			for (const key of Object.keys(l)) {
				if (key !== "enabled" && key !== "timeoutMs" && key !== "serverOverrides") {
					warn(warnings, `Unknown lsp key: "${key}".`);
				}
			}
			const le = configBool(l.enabled);
			if (le !== undefined) lc.enabled = le;
			const lt = configNumber(l.timeoutMs);
			if (lt !== undefined && lt > 0) lc.timeoutMs = lt;
			if (l.serverOverrides !== undefined && typeof l.serverOverrides === "object" && l.serverOverrides !== null) {
				const overrides = l.serverOverrides as Record<string, unknown>;
				const parsedOverrides: NonNullable<typeof lc.serverOverrides> = {};
				for (const [ext, def] of Object.entries(overrides)) {
					if (typeof def !== "object" || def === null) {
						warn(warnings, `"lsp.serverOverrides.${ext}" must be an object.`);
						continue;
					}
					const d = def as Record<string, unknown>;
					if (typeof d.command !== "string" || !d.command.trim()) {
						warn(warnings, `"lsp.serverOverrides.${ext}.command" must be a non-empty string.`);
						continue;
					}
					if (typeof d.languageId !== "string" || !d.languageId.trim()) {
						warn(warnings, `"lsp.serverOverrides.${ext}.languageId" must be a non-empty string.`);
						continue;
					}
					const args = Array.isArray(d.args)
						? d.args.filter((a): a is string => typeof a === "string")
						: undefined;
					parsedOverrides[ext] = {
						command: d.command.trim(),
						args,
						languageId: d.languageId.trim(),
					};
				}
				if (Object.keys(parsedOverrides).length > 0) {
					lc.serverOverrides = parsedOverrides;
				}
			}
			if (Object.keys(lc).length > 0) cfg.lsp = lc;
		}
	}

	// compaction sub-object.
	if (obj.compaction !== undefined) {
		if (typeof obj.compaction !== "object" || obj.compaction === null) {
			warn(warnings, "\"compaction\" must be an object.");
		} else {
			const c = obj.compaction as Record<string, unknown>;
			const ccfg: {
				enabled?: boolean;
				reserveTokens?: number;
				keepRecentTokens?: number;
			} = {};
			for (const key of Object.keys(c)) {
				if (!COMPACTION_KEYS.has(key)) {
					warn(warnings, `Unknown compaction key: "${key}".`);
				}
			}
			const ce = configBool(c.enabled);
			if (ce !== undefined) ccfg.enabled = ce;
			const crt = configNumber(c.reserveTokens);
			if (crt !== undefined && crt > 0) ccfg.reserveTokens = crt;
			const krt = configNumber(c.keepRecentTokens);
			if (krt !== undefined && krt > 0) ccfg.keepRecentTokens = krt;
			if (Object.keys(ccfg).length > 0) cfg.compaction = ccfg;
		}
	}

	// truncation sub-object: universal output/result size caps.
	if (obj.truncation !== undefined) {
		if (typeof obj.truncation !== "object" || obj.truncation === null) {
			warn(warnings, "\"truncation\" must be an object.");
		} else {
			const t = obj.truncation as Record<string, unknown>;
			const tcfg: TruncationConfig = {};
			for (const key of Object.keys(t)) {
				if (!TRUNCATION_KEYS.has(key)) {
					warn(warnings, `Unknown truncation key: "${key}".`);
				}
			}
			for (const key of [
				"toolResultMaxChars",
				"maxLines",
				"grepLineMaxChars",
				"subagentResultMaxChars",
				"compactionSummaryMaxChars",
				"transcriptMessageMaxChars",
			] as const) {
				const n = configNumber(t[key]);
				if (n !== undefined) {
					if (n <= 0) {
						warn(warnings, `"truncation.${key}" must be > 0, got ${n}. Ignored.`);
					} else {
						tcfg[key] = n;
					}
				}
			}
			if (t.microCompactMaxChars !== undefined) {
				if (
					typeof t.microCompactMaxChars !== "object" ||
					t.microCompactMaxChars === null
				) {
					warn(warnings, "\"truncation.microCompactMaxChars\" must be an object.");
				} else {
					const m = t.microCompactMaxChars as Record<string, unknown>;
					const mcfg: NonNullable<TruncationConfig["microCompactMaxChars"]> = {};
					for (const key of Object.keys(m)) {
						if (!MICRO_COMPACT_MAX_CHARS_KEYS.has(key)) {
							warn(warnings, `Unknown truncation.microCompactMaxChars key: "${key}".`);
						}
					}
					for (const key of ["tool", "assistant", "default"] as const) {
						const n = configNumber(m[key]);
						if (n !== undefined) {
							if (n <= 0) {
								warn(
									warnings,
									`"truncation.microCompactMaxChars.${key}" must be > 0, got ${n}. Ignored.`,
								);
							} else {
								mcfg[key] = n;
							}
						}
					}
					if (Object.keys(mcfg).length > 0) tcfg.microCompactMaxChars = mcfg;
				}
			}
			if (Object.keys(tcfg).length > 0) cfg.truncation = tcfg;
		}
	}

	// MCP fields (passthrough, but warn on unknown sub-keys).
	if (obj.mcp !== undefined && typeof obj.mcp === "object") {
		cfg.mcp = obj.mcp as Record<string, unknown>;
	}
	if (obj.mcpServers !== undefined && typeof obj.mcpServers === "object") {
		cfg.mcpServers = obj.mcpServers as Record<string, unknown>;
	}

	if (obj.plugins !== undefined) {
		if (
			typeof obj.plugins !== "object" ||
			obj.plugins === null ||
			Array.isArray(obj.plugins)
		) {
			warn(warnings, "\"plugins\" must be an object.");
		} else {
			cfg.plugins = obj.plugins as Record<string, unknown>;
		}
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
				percfg.allow = perms.allow.filter(
					(v): v is string => typeof v === "string" && v.trim().length > 0,
				);
			}
			if (Array.isArray(perms.deny)) {
				percfg.deny = perms.deny.filter(
					(v): v is string => typeof v === "string" && v.trim().length > 0,
				);
			}
			cfg.permissions = Object.keys(percfg).length > 0 ? percfg : undefined;
		}
	}

	// Strip undefined values so the returned config only contains set fields.
	return Object.fromEntries(
		Object.entries(cfg).filter(([, v]) => v !== undefined),
	) as LogicianTuiConfig;
}

export interface LogicianTuiConfig {
	baseUrl?: string;
	llmUrl?: string;
	model?: string;
	models?: AgentModelConfig[];
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
	plugins?: Record<string, unknown>;
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
	guardsEnabled?: boolean; // legacy: forces both guards below on when true
	duplicateGuardEnabled?: boolean; // ON by default — blocks exact-repeat tool calls (e.g. re-reading the same file)
	failureGuardEnabled?: boolean; // OFF by default
	continuationEnabled?: boolean; // ON by default — prevents premature stopping when the model says "done" mid-task
	postEditDiagnostics?: boolean; // ON by default — syntax and project-aware diagnostics after edits
	autoRetryEnabled?: boolean;
	maxRetries?: number;
	retryBaseDelayMs?: number;
	turnTimeoutMs?: number;
	cacheSize?: number;
	cacheTtlMs?: number;
	// Absolute paths the agent may read/write outside CWD.
	allowedPaths?: string[];
	// When true, skip CWD/allowedPaths enforcement entirely.
	allowAllPaths?: boolean;
	// Explicit project root (overrides auto-detected CWD).
	cwd?: string;
	// LSP (language server protocol) settings.
	lsp?: {
		enabled?: boolean;
		timeoutMs?: number;
		serverOverrides?: Record<string, {
			command: string;
			args?: string[];
			languageId: string;
		}>;
	};
	// Compaction settings.
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	// Inference mode — pre-defined sampling parameter set, cycled via Alt+M.
	inferenceMode?:
		| "thinking-general"
		| "thinking-coding"
		| "instruct-general"
		| "instruct-reasoning";
	// Universal output/result truncation limits.
	truncation?: TruncationConfig;
}

export function loadLogicianConfig(
	cwd = process.cwd(),
): ResolvedLogicianConfig {
	const configPath = findLogicianConfig(cwd);
	if (!configPath) return { config: {}, warnings: [] };
	return loadLogicianConfigFile(configPath);
}

/**
 * Load trusted per-user settings without consulting project-local config.
 * Project trust gates .logician.json, never ~/.logician/settings.json.
 */
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
		const global = join(home, ".logician", "settings.json");
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

/** Save a single config field to the global user config file (~/.logician/settings.json). */
export function saveConfigField(key: string, value: unknown): boolean {
	try {
		const home = process.env.HOME || "";
		if (!home) return false;
		const configPath = join(home, ".logician", "settings.json");
		const dir = dirname(configPath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
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
	} catch (_e: unknown) {
		return false;
	}
}
