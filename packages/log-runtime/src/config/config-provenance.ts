/**
 * Config provenance: for every key a user actually set, which layer
 * (global < project < env) supplied the value that won.
 *
 * Built from the RAW layer inputs — the parsed config files and the
 * applied env overrides — so the table shows what the user wrote, not
 * the defaults `validateConfig` injects.
 */

/** Layers that can supply config values, lowest precedence first. */
export const CONFIG_LAYERS = ["global", "project", "env"] as const;

/** A config layer that can supply values. */
export type ConfigLayer = (typeof CONFIG_LAYERS)[number];

/** One row of the config precedence table. */
export interface ConfigProvenanceEntry {
	/** Dotted config path (e.g. "compaction.enabled"). */
	key: string;
	/** The winning value for the key. */
	value: unknown;
	/** The highest-precedence layer that set the key. */
	layer: ConfigLayer;
}

export type ConfigProvenance = Readonly<ConfigProvenanceEntry[]>;

/**
 * Free-form passthrough objects are not flattened: their shape is
 * unconstrained (server lists, plugin maps), so they report at their
 * top-level key.
 */
const PASSTHROUGH_KEYS = new Set([
	"mcp",
	"mcpServers",
	"plugins",
	"reasonerConfig",
]);

/**
 * Flatten a raw config layer to dotted leaf paths. Nested objects flatten
 * recursively except passthrough objects and arrays, which stay whole
 * under their top-level key.
 */
function flattenLayer(
	config: Record<string, unknown>,
	prefix: string,
	out: Record<string, unknown>,
): void {
	for (const [key, value] of Object.entries(config)) {
		const path = prefix ? `${prefix}.${key}` : key;
		if (
			value !== null &&
			typeof value === "object" &&
			!Array.isArray(value) &&
			!PASSTHROUGH_KEYS.has(key)
		) {
			flattenLayer(value as Record<string, unknown>, path, out);
		} else {
			out[path] = value;
		}
	}
}

/**
 * Build the precedence table: one entry per key present in any layer,
 * carrying the value and layer of the highest-precedence writer. Keys
 * shadowed by a higher layer do not surface the shadowed value.
 */
export function buildConfigProvenance(layers: {
	global?: Record<string, unknown>;
	project?: Record<string, unknown>;
	env?: Record<string, unknown>;
}): ConfigProvenance {
	const flat: Record<ConfigLayer, Record<string, unknown>> = {
		global: {},
		project: {},
		env: {},
	};
	if (layers.global) flattenLayer(layers.global, "", flat.global);
	if (layers.project) flattenLayer(layers.project, "", flat.project);
	if (layers.env) flattenLayer(layers.env, "", flat.env);

	const entries: ConfigProvenanceEntry[] = [];
	const seen = new Set<string>();
	// Highest precedence first, so each key's row carries the winning
	// layer's value.
	for (const layer of [...CONFIG_LAYERS].reverse()) {
		for (const key of Object.keys(flat[layer])) {
			if (seen.has(key)) continue;
			seen.add(key);
			entries.push({ key, value: flat[layer][key], layer });
		}
	}
	return entries.sort((a, b) => a.key.localeCompare(b.key));
}

/** Render the precedence table for doctor / CLI output. */
export function formatConfigProvenance(provenance: ConfigProvenance): string {
	if (provenance.length === 0) return "  (no keys set — defaults only)";
	const lines: string[] = [];
	for (const { key, value, layer } of provenance) {
		const text = JSON.stringify(value);
		lines.push(`    ${key.padEnd(28)} ${text}  [${layer}]`);
	}
	return lines.join("\n");
}
