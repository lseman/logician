import { existsSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import type { AgentModelConfig, TruncationConfig } from "@logician/log-core";
import { getReasonerIds } from "../capabilities/reasoning/index.ts";
import {
	getKnownConfigKeys,
	getNestedKeys,
	getSettingSpec,
	type SettingSpec,
} from "./settings-schema.ts";

/** Validated configuration with warnings collected during load. */
export interface ResolvedLogicianConfig {
	path?: string;
	/** Parsed file contents before validation (provenance reporting). */
	raw?: Record<string, unknown>;
	config: LogicianTuiConfig;
	warnings: string[];
}

/**
 * Known config keys for unknown-field detection. Derived from the schema
 * registry (settings-schema.ts) so the two can't drift apart.
 */
const KNOWN_KEYS = new Set(getKnownConfigKeys());
/** Sourced from the reasoner registry itself so this can't drift from the real set. */
const REASONER_IDS = new Set(getReasonerIds());

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

/**
 * Keys validated by dedicated blocks in `validateConfig` rather than the
 * registry pass: semantics the schema registry cannot express (filesystem
 * check, external registry lookup, clamping instead of ignore).
 */
const CUSTOM_KEYS: ReadonlySet<string> = new Set([
	"cwd",
	"reasoner",
	"temperature",
]);

/**
 * Sub-objects whose fields are validated from the schema registry. Objects
 * not listed here are passthroughs (mcp, mcpServers, plugins,
 * reasonerConfig) and keep dedicated blocks in `validateConfig`.
 */
const SUB_OBJECTS = [
	"lsp",
	"legroom",
	"memoriam",
	"compaction",
	"truncation",
	"tools",
	"webSearch",
	"permissions",
	"ttsr",
] as const;

/**
 * Validate one scalar field from its schema spec, writing the accepted
 * value to `out[field]`. `path` is the full dotted key used in warning
 * messages; `field` is the local key in `out`.
 */
function validateScalarField(
	path: string,
	field: string,
	value: unknown,
	spec: SettingSpec,
	out: Record<string, unknown>,
	warnings: string[],
): void {
	if (value === undefined) return;
	switch (spec.type) {
		case "string": {
			const s = configString(value);
			if (s !== undefined) out[field] = s;
			return;
		}
		case "url":
			if (isValidUrl(value)) {
				out[field] = configString(value);
			} else {
				warn(warnings, `"${path}" must be a valid http/https URL.`);
			}
			return;
		case "number": {
			const n = configNumber(value);
			if (n === undefined) return;
			const okMin =
				spec.min === undefined
					? true
					: spec.minExclusive
						? n > spec.min
						: n >= spec.min;
			const okMax = spec.max === undefined || n <= spec.max;
			if (okMin && okMax) {
				out[field] = n;
			} else if (!spec.silent) {
				warn(warnings, rangeMessage(path, spec));
			}
			return;
		}
		case "boolean": {
			const b = configBool(value, spec.default as boolean | undefined);
			if (b !== undefined) out[field] = b;
			return;
		}
		case "enum": {
			const s = configString(value);
			const valid = spec.enum ?? [];
			if (s !== undefined && valid.includes(s)) {
				out[field] = s;
			} else if (s !== undefined) {
				warn(warnings, `"${path}" must be one of: ${valid.join(", ")}.`);
			}
			return;
		}
		default:
			return;
	}
}

/** Unified range-violation message for number settings. */
function rangeMessage(key: string, spec: SettingSpec): string {
	const min = spec.min ?? 0;
	const inclusive = spec.minExclusive !== true;
	const max = spec.max !== undefined ? ` and <= ${spec.max}` : "";
	return `"${key}" must be ${inclusive ? ">=" : ">"} ${min}${max}. Ignored.`;
}

/** Filter an array field down to its string elements. */
function copyStringArray(
	path: string,
	field: string,
	value: unknown,
	out: Record<string, unknown>,
	warnings: string[],
): void {
	if (!Array.isArray(value)) {
		warn(warnings, `"${path}" must be an array of strings.`);
		return;
	}
	out[field] = value.filter((item): item is string => typeof item === "string");
}

/** Filter an array field down to its non-blank string elements. */
function copyTrimmedStringArray(
	path: string,
	field: string,
	value: unknown,
	out: Record<string, unknown>,
	warnings: string[],
): void {
	if (!Array.isArray(value)) {
		warn(warnings, `"${path}" must be an array of strings.`);
		return;
	}
	out[field] = value.filter(
		(item): item is string =>
			typeof item === "string" && item.trim().length > 0,
	);
}

/**
 * Per-sub-object handlers for the fields that need bespoke treatment:
 * string-array element filtering, object pass-through, and the deep
 * serverOverrides shape.
 */
type SubObjectCustom = (
	value: Record<string, unknown>,
	out: Record<string, unknown>,
	warnings: string[],
) => void;

const SUB_OBJECT_CUSTOMS: Readonly<Record<string, SubObjectCustom>> = {
	legroom: (value, out, warnings) => {
		if (value.args !== undefined)
			copyStringArray("legroom.args", "args", value.args, out, warnings);
		if (value.config !== undefined) {
			if (typeof value.config === "object" && value.config !== null)
				out.config = value.config;
			else warn(warnings, '"legroom.config" must be an object.');
		}
	},
	ttsr: (value, out, warnings) => {
		if (value.disabledRules !== undefined)
			copyStringArray(
				"ttsr.disabledRules",
				"disabledRules",
				value.disabledRules,
				out,
				warnings,
			);
	},
	memoriam: (value, out, warnings) => {
		if (value.args !== undefined)
			copyStringArray("memoriam.args", "args", value.args, out, warnings);
		if (value.config !== undefined) {
			if (typeof value.config === "object" && value.config !== null)
				out.config = value.config;
			else warn(warnings, '"memoriam.config" must be an object.');
		}
	},
	permissions: (value, out, warnings) => {
		if (value.allow !== undefined)
			copyTrimmedStringArray(
				"permissions.allow",
				"allow",
				value.allow,
				out,
				warnings,
			);
		if (value.deny !== undefined)
			copyTrimmedStringArray(
				"permissions.deny",
				"deny",
				value.deny,
				out,
				warnings,
			);
	},
	lsp: (value, out, warnings) => {
		if (value.serverOverrides === undefined) return;
		if (
			typeof value.serverOverrides !== "object" ||
			value.serverOverrides === null
		) {
			warn(warnings, '"lsp.serverOverrides" must be an object.');
			return;
		}
		const overrides = value.serverOverrides as Record<string, unknown>;
		const parsed: Record<
			string,
			{ command: string; args?: string[]; languageId: string }
		> = {};
		for (const [ext, def] of Object.entries(overrides)) {
			if (typeof def !== "object" || def === null) {
				warn(warnings, `"lsp.serverOverrides.${ext}" must be an object.`);
				continue;
			}
			const d = def as Record<string, unknown>;
			if (typeof d.command !== "string" || !d.command.trim()) {
				warn(
					warnings,
					`"lsp.serverOverrides.${ext}.command" must be a non-empty string.`,
				);
				continue;
			}
			if (typeof d.languageId !== "string" || !d.languageId.trim()) {
				warn(
					warnings,
					`"lsp.serverOverrides.${ext}.languageId" must be a non-empty string.`,
				);
				continue;
			}
			const args = Array.isArray(d.args)
				? d.args.filter((a): a is string => typeof a === "string")
				: undefined;
			parsed[ext] = {
				command: d.command.trim(),
				args,
				languageId: d.languageId.trim(),
			};
		}
		out.serverOverrides = parsed;
	},
};

/**
 * Validate one sub-object: unknown-key warnings, registry-driven validation
 * of every registered field (recursing into nested sub-objects), then the
 * object's custom handler for bespoke fields. `parent` is the schema path
 * used for lookups and messages; `key` is where the assembled object lands
 * in `sink`.
 */
function validateSubObject(
	parent: string,
	raw: unknown,
	sink: Record<string, unknown>,
	key: string,
	warnings: string[],
): void {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		warn(warnings, `"${parent}" must be an object.`);
		return;
	}
	const value = raw as Record<string, unknown>;
	const known = getNestedKeys(parent);
	for (const k of Object.keys(value)) {
		if (!known.includes(k)) {
			warn(warnings, `Unknown ${parent} key: "${k}".`);
		}
	}
	const out: Record<string, unknown> = {};
	for (const field of known) {
		const fieldPath = `${parent}.${field}`;
		const fv = value[field];
		const spec = getSettingSpec(fieldPath);
		if (spec === undefined || fv === undefined) continue;
		if (spec.type === "object") {
			// Nested sub-object (truncation.microCompactMaxChars) — recurse
			// into the parent's out so it lands under `field`. Objects
			// without registered children (lsp.serverOverrides) are handled
			// by the custom handler.
			if (getNestedKeys(fieldPath).length > 0) {
				validateSubObject(fieldPath, fv, out, field, warnings);
			}
			continue;
		}
		if (spec.type === "array" || spec.type === "models") continue; // custom
		validateScalarField(fieldPath, field, fv, spec, out, warnings);
	}
	const custom = SUB_OBJECT_CUSTOMS[parent];
	if (custom) custom(value, out, warnings);
	if (Object.keys(out).length > 0) sink[key] = out;
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
	const sink = cfg as Record<string, unknown>;

	// Check for unknown top-level keys.
	for (const key of Object.keys(obj)) {
		if (!KNOWN_KEYS.has(key)) {
			warn(warnings, `Unknown config key: "${key}".`);
		}
	}

	// Registry-driven pass: every scalar key is validated from its schema
	// spec (type, range, enum) so adding a key to settings-schema.ts is
	// enough to validate it here. Keys with semantics beyond scalar
	// coercion (CUSTOM_KEYS) and object/array keys keep dedicated blocks
	// below.
	for (const key of getKnownConfigKeys()) {
		if (CUSTOM_KEYS.has(key)) continue;
		const spec = getSettingSpec(key);
		if (spec === undefined) continue;
		const value = obj[key];
		if (value === undefined) {
			// Boolean defaults come from the registry so the UI, docs, and
			// validation share one source of truth.
			if (spec.type === "boolean" && spec.default !== undefined)
				sink[key] = spec.default;
			continue;
		}
		if (
			spec.type === "object" ||
			spec.type === "array" ||
			spec.type === "models"
		) {
			continue;
		}
		validateScalarField(key, key, value, spec, sink, warnings);
	}

	// models: array of named model objects for cycling (Ctrl+L model selector).
	if (obj.models !== undefined) {
		const parsed: AgentModelConfig[] = [];
		if (Array.isArray(obj.models)) {
			for (const item of obj.models) {
				if (
					typeof item === "object" &&
					item !== null &&
					"model" in item &&
					"name" in item &&
					typeof (item as Record<string, unknown>).name === "string" &&
					typeof (item as Record<string, unknown>).model === "string" &&
					((item as AgentModelConfig).name.trim() ||
						(item as AgentModelConfig).model.trim())
				) {
					const m = item as Record<string, unknown> & AgentModelConfig;
					const entry: AgentModelConfig = {
						name: m.name.trim(),
						model: m.model.trim(),
						url: typeof m.url === "string" ? m.url.trim() : m.url,
					};
					if (m.contextWindow !== undefined) {
						const cw = configNumber(m.contextWindow);
						if (cw !== undefined && cw > 0) entry.contextWindow = cw;
						else
							warn(
								warnings,
								`"models" entry "${m.model}" has invalid contextWindow; ignored.`,
							);
					}
					if (m.maxTokens !== undefined) {
						const mt = configNumber(m.maxTokens);
						if (mt !== undefined && mt > 0) entry.maxTokens = mt;
						else
							warn(
								warnings,
								`"models" entry "${m.model}" has invalid maxTokens; ignored.`,
							);
					}
					parsed.push(entry);
				} else {
					warn(
						warnings,
						`"models" entry invalid, got: ${JSON.stringify(item)}.`,
					);
				}
			}
		} else {
			warn(
				warnings,
				'"models" must be an array of { name, model, url?, contextWindow?, maxTokens? }.',
			);
		}
		if (parsed.length > 0) {
			cfg.models = parsed as LogicianTuiConfig["models"];
		}
	}

	// temperature: 0–2 (clamps out-of-range values instead of ignoring).
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

	// reasoner: name from the reasoner registry; unknown names fall back.
	if (obj.reasoner !== undefined) {
		cfg.reasoner = configString(obj.reasoner)?.toLowerCase();
		if (!cfg.reasoner) warn(warnings, '"reasoner" must be a non-empty string.');
		else if (!REASONER_IDS.has(cfg.reasoner)) {
			warn(warnings, `Unknown reasoner: "${cfg.reasoner}". Using "none".`);
			cfg.reasoner = "none";
		}
	}
	// reasonerConfig: object passthrough.
	if (obj.reasonerConfig !== undefined) {
		if (
			obj.reasonerConfig &&
			typeof obj.reasonerConfig === "object" &&
			!Array.isArray(obj.reasonerConfig)
		) {
			cfg.reasonerConfig = {
				...(obj.reasonerConfig as Record<string, unknown>),
			};
		} else {
			warn(warnings, '"reasonerConfig" must be an object.');
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
						warn(
							warnings,
							`"allowedPaths" entry must be an absolute path: "${trimmed}". Ignored.`,
						);
					} else {
						paths.push(trimmed);
					}
				}
			}
			if (paths.length > 0) cfg.allowedPaths = paths;
		} else {
			warn(warnings, '"allowedPaths" must be an array.');
		}
	}

	// cwd: explicit project root (existence checked against the filesystem).
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

	// Sub-objects: registry-driven fields plus per-object custom handlers
	// (SUB_OBJECT_CUSTOMS) for element filtering and deep shapes.
	for (const parent of SUB_OBJECTS) {
		if (obj[parent] !== undefined) {
			validateSubObject(parent, obj[parent], sink, parent, warnings);
		}
	}

	// MCP fields (passthrough, no sub-key validation).
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
			warn(warnings, '"plugins" must be an object.');
		} else {
			cfg.plugins = obj.plugins as Record<string, unknown>;
		}
	}

	// simpleTools: explicit list of tool names that render as simple one-liners.
	if (obj.simpleTools !== undefined) {
		if (Array.isArray(obj.simpleTools)) {
			const tools: string[] = [];
			for (const t of obj.simpleTools) {
				if (typeof t === "string" && t.trim()) {
					tools.push(t.trim());
				}
			}
			if (tools.length > 0) cfg.simpleTools = tools;
		} else {
			warn(warnings, '"simpleTools" must be an array of strings.');
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
	thinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
	thinkingFormat?: "qwen" | "qwen-chat-template";
	executionProfile?: "autonomous" | "minimal";
	toolExecution?: "sequential" | "parallel";
	contextWindow?: number;
	contextWindowTokens?: number;
	hooks?: boolean;
	mcp?: Record<string, unknown>;
	mcpServers?: Record<string, unknown>;
	plugins?: Record<string, unknown>;
	legroom?: {
		mode?: "off" | "sdk";
		python?: string;
		args?: string[];
		failOpen?: boolean;
		timeoutMs?: number;
		config?: Record<string, unknown>;
	};
	memoriam?: {
		mode?: "off" | "sdk";
		python?: string;
		args?: string[];
		failOpen?: boolean;
		timeoutMs?: number;
		config?: Record<string, unknown>;
	};
	webSearch?: {
		baseUrl?: string;
		maxResults?: number;
	};
	permissionMode?: "acceptAll" | "acceptEdits" | "ask" | "plan";
	workflowMode?: "act" | "plan";
	permissions?: {
		allow?: string[];
		deny?: string[];
	};
	steeringInterrupt?: boolean;
	maxTotalTokens?: number;
	// Safeguard options (match pi's trust-model approach by default).
	guardsEnabled?: boolean; // umbrella toggle that enables both guards below
	duplicateGuardEnabled?: boolean; // ON by default — blocks exact-repeat tool calls (e.g. re-reading the same file)
	failureGuardEnabled?: boolean; // OFF by default
	duplicateToolThreshold?: number; // consecutive identical calls before the duplicate guard blocks (default 3)
	toolFailureLoopThreshold?: number; // repeated failures (same call/path/category) before the failure guard blocks (default 3)
	progressStopEnabled?: boolean; // OFF by default — stops after repeated turns produce no new tool/task evidence
	continuationEnabled?: boolean; // ON by default — prevents premature stopping when the model says "done" mid-task
	verifiedStopEnabled?: boolean; // Require successful post-edit verification before settling
	postEditDiagnostics?: boolean; // ON by default — syntax and project-aware diagnostics after edits
	autoRetryEnabled?: boolean;
	// RTK CLI proxy — compresses bash/rg/grep output 60–90%.
	rtkProxyEnabled?: boolean;
	/** Expose the Graphician code-graph tool (ON by default). */
	graphicianEnabled?: boolean;
	/** Prefer the fff MCP indexed grep tool when available (ON by default). */
	fffgrepEnabled?: boolean;
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
		serverOverrides?: Record<
			string,
			{
				command: string;
				args?: string[];
				languageId: string;
			}
		>;
	};
	// Time-traveling stream rules (mid-stream rule enforcement).
	ttsr?: {
		enabled?: boolean;
		builtinRules?: boolean;
		judge?: boolean;
		interruptMode?: "always" | "prose-only" | "tool-only" | "never";
		repeatMode?: "once" | "gap";
		repeatGap?: number;
		disabledRules?: string[];
	};
	// Compaction settings.
	compaction?: {
		enabled?: boolean;
		/**
		 * auto: shake, then LLM summary if still large (default) · llm · shake
		 * (drop recoverable heavy content only) · snapcompact (local bitmap
		 * frames, no LLM) · remote (provider-native endpoint).
		 */
		mode?: "auto" | "llm" | "snapcompact" | "shake" | "remote";
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	// Inference mode — pre-defined sampling parameter set, cycled via Alt+M.
	inferenceMode?:
		| "auto"
		| "none"
		| "thinking-general"
		| "thinking-coding"
		| "instruct-general"
		| "instruct-reasoning"
		| "instruct-coding"
		| "deterministic"
		| "creative"
		| "analytical";
	// Universal output/result truncation limits.
	truncation?: TruncationConfig;
	/** Maximum delegated agents executing concurrently. */
	maxParallelAgents?: number;
	/** Maximum number of turns to keep in the transcript (default: 40). */
	transcriptMaxTurns?: number;
	/** Maximum rendered lines before older transcript content is cut off (default: 400). */
	transcriptMaxRenderedLines?: number;
	/** Structured pre-reasoning mode. Default: "none" (disabled). */
	reasoner?: string;
	reasonerConfig?: Record<string, unknown>;
	/** Explicit list of tool names that should render as simple one-liners.
	 * Merged with the built-in defaults; only adds tools the user wants as simple. */
	simpleTools?: string[];
	/** Enable discoverable tools behind `xd://` URLs (default: true). */
	tools?: { xdev?: boolean };
	/** Enable the `todo` tool (default: true). */
	todoEnabled?: boolean;
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
