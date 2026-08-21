const ANNOTATION_KEYS = new Set([
	"$id",
	"$schema",
	"default",
	"deprecated",
	"examples",
	"format",
	"readOnly",
	"title",
	"writeOnly",
]);

// Keep this deliberately smaller than JSON Schema. Provider grammar compilers
// vary considerably, while the tool itself remains the authority for validation.
const SUPPORTED_KEYS = new Set([
	"additionalProperties",
	"allOf",
	"anyOf",
	"description",
	"enum",
	"exclusiveMaximum",
	"exclusiveMinimum",
	"items",
	"maxItems",
	"maxLength",
	"maximum",
	"minItems",
	"minLength",
	"minimum",
	"multipleOf",
	"oneOf",
	"properties",
	"required",
	"type",
	"uniqueItems",
]);

function record(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

function resolveLocalReference(
	root: Record<string, unknown>,
	reference: string,
): Record<string, unknown> | undefined {
	if (!reference.startsWith("#/")) return undefined;
	let current: unknown = root;
	for (const segment of reference
		.slice(2)
		.split("/")
		.map(part => part.replaceAll("~1", "/").replaceAll("~0", "~"))) {
		current = record(current)?.[segment];
	}
	return record(current);
}

/**
 * Convert arbitrary extension/MCP JSON Schema into the conservative subset
 * accepted by OpenAI-compatible grammar compilers such as llama.cpp.
 * Runtime validation remains the tool implementation's responsibility.
 */
export function normalizeProviderToolSchema(
	input: unknown,
): Record<string, unknown> {
	const root = record(input) ?? { type: "object", properties: {} };
	const active = new Set<Record<string, unknown>>();

	const visit = (value: unknown): unknown => {
		if (value === true) return {};
		if (value === false || value == null) return {};
		if (Array.isArray(value)) return value.map(visit);
		const source = record(value);
		if (!source) return value;
		if (active.has(source)) return {};
		active.add(source);

		let resolved = source;
		if (typeof source.$ref === "string") {
			const target = resolveLocalReference(root, source.$ref);
			if (target) resolved = { ...target, ...source };
		}

		const output: Record<string, unknown> = {};
		for (const [key, child] of Object.entries(resolved)) {
			if (
				key === "$ref" ||
				key === "$defs" ||
				key === "definitions" ||
				ANNOTATION_KEYS.has(key)
			) {
				continue;
			}
			if (key !== "const" && !SUPPORTED_KEYS.has(key)) continue;
			if (key === "type" && Array.isArray(child)) {
				output.type = child.find(type => type !== "null") ?? "string";
				continue;
			}
			if (key === "const") {
				output.enum = [child];
				continue;
			}
			if (key === "properties" && record(child)) {
				output.properties = Object.fromEntries(
					Object.entries(child as Record<string, unknown>).map(
						([name, schema]) => [name, visit(schema)],
					),
				);
				continue;
			}
			if (key === "required" && Array.isArray(child)) {
				const propertyNames = new Set(
					Object.keys(record(resolved.properties) ?? {}),
				);
				output.required = child.filter(
					name => typeof name === "string" && propertyNames.has(name),
				);
				continue;
			}
			if (key === "items" && Array.isArray(child)) {
				output.items = { anyOf: child.map(visit) };
				continue;
			}
			output[key] = visit(child);
		}
		if (!output.type && output.properties) output.type = "object";
		active.delete(source);
		return output;
	};

	const normalized = record(visit(root)) ?? {};
	if (normalized.type !== "object") {
		return {
			type: "object",
			properties: { value: normalized },
			required: ["value"],
		};
	}
	return { ...normalized, properties: record(normalized.properties) ?? {} };
}
