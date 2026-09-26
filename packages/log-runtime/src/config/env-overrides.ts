/**
 * Declarative environment-variable override layer for configuration.
 *
 * Every config key listed in ENV_OVERRIDES can be set from the environment
 * without editing a config file. Precedence: defaults < global < project <
 * env. Env values are coerced using the settings-schema type, so the table
 * cannot drift from the registry — a key that leaves the schema fails the
 * load-time check below.
 *
 * Keys intentionally NOT here: operational env (config file path, trust
 * gate, kill switches, debug flags) and endpoint settings that own their
 * env handling at the consumer (webSearch falls back to LOGICIAN_SEARXNG_URL
 * where its defaults are resolved).
 */

import { getKnownConfigKeys, getSettingSpec } from "./settings-schema.ts";

/** config key → env var(s), first set value wins. */
const ENV_OVERRIDES: Readonly<Record<string, readonly string[]>> = {
	model: ["LOGICIAN_MODEL"],
	baseUrl: ["LOGICIAN_LLM_URL"],
	systemPrompt: ["LOGICIAN_SYSTEM_PROMPT"],
	reasoner: ["LOGICIAN_REASONER"],
	contextWindowTokens: ["LOGICIAN_CONTEXT_WINDOW", "LOGICIAN_CTX_SIZE"],
	hooks: ["LOGICIAN_HOOKS"],
	theme: ["LOGICIAN_THEME"],
	postEditDiagnostics: ["LOGICIAN_POST_EDIT_DIAGNOSTICS"],
};

/** Fail at module load if the table names a key the schema no longer knows. */
const KNOWN_KEYS = new Set(getKnownConfigKeys());
for (const key of Object.keys(ENV_OVERRIDES)) {
	if (!KNOWN_KEYS.has(key)) {
		throw new Error(
			`ENV_OVERRIDES references unknown config key "${key}" — update settings-schema.ts or the table`,
		);
	}
}

const BOOL_FALSE = new Set(["0", "off", "false"]);
const BOOL_TRUE = new Set(["1", "on", "true"]);

/**
 * Coerce a raw env string to the schema's type. Returns undefined (and
 * records a warning) when the value cannot be coerced — the config value
 * then stands.
 */
function coerceEnvValue(
	key: string,
	raw: string,
	warnings: string[],
): string | number | boolean | undefined {
	const spec = getSettingSpec(key);
	const type = spec?.type ?? "string";
	const value = raw.trim();
	if (value === "") return undefined;

	switch (type) {
		case "number": {
			const n = Number(value);
			if (!Number.isFinite(n)) {
				warnings.push(
					`Environment ${key} value "${raw}" is not a number; ignored.`,
				);
				return undefined;
			}
			if (spec?.minExclusive && n <= (spec.min ?? 0)) {
				warnings.push(
					`Environment ${key} value "${raw}" must be > ${spec.min}; ignored.`,
				);
				return undefined;
			}
			if (spec?.min !== undefined && n < spec.min && !spec.minExclusive) {
				warnings.push(
					`Environment ${key} value "${raw}" must be >= ${spec.min}; ignored.`,
				);
				return undefined;
			}
			return n;
		}
		case "boolean": {
			const normalized = value.toLowerCase();
			if (BOOL_FALSE.has(normalized)) return false;
			if (BOOL_TRUE.has(normalized)) return true;
			warnings.push(
				`Environment ${key} value "${raw}" is not a boolean (0/off/false, 1/on/true); ignored.`,
			);
			return undefined;
		}
		case "enum": {
			const allowed = spec?.enum;
			if (allowed && !allowed.includes(value)) {
				warnings.push(
					`Environment ${key} value "${raw}" is not one of ${allowed.join(", ")}; ignored.`,
				);
				return undefined;
			}
			return value;
		}
		default:
			// string, url, object, array, models: pass through as-is.
			return value;
	}
}

/**
 * Apply env overrides to a validated config object. Returns a new object
 * (input is not mutated), the coerced overrides that were applied (for
 * provenance reporting), and warnings for values that failed coercion.
 */
export function applyEnvOverrides(
	config: Record<string, unknown>,
	environment: NodeJS.ProcessEnv = process.env,
): {
	config: Record<string, unknown>;
	applied: Record<string, unknown>;
	warnings: string[];
} {
	const patched: Record<string, unknown> = { ...config };
	const applied: Record<string, unknown> = {};
	const warnings: string[] = [];
	for (const [key, vars] of Object.entries(ENV_OVERRIDES)) {
		const raw = vars
			.map(name => environment[name])
			.find(value => value !== undefined && value.trim() !== "");
		if (raw === undefined) continue;
		const coerced = coerceEnvValue(key, raw, warnings);
		if (coerced !== undefined) {
			patched[key] = coerced;
			applied[key] = coerced;
		}
	}
	return { config: patched, applied, warnings };
}
