// ── Compaction Method System ─────────────────────────────────────────────────
// Configurable multi-strategy compaction system inspired by oh-my-pi's
// compaction-method architecture. Supports 5 strategies with ordered fallback.

/** Ordered automatic context-maintenance methods and their settings metadata. */

export type CompactionMethod =
	| "remote"
	| "snapcompact"
	| "handoff"
	| "shake"
	| "soft";

/** Choices presented by the compaction-method setting. */
export const COMPACTION_METHOD_CHOICES: Array<{
	value: CompactionMethod;
	label: string;
	description: string;
}> = [
	{
		value: "remote",
		label: "Server compaction",
		description:
			"Use provider-native server compaction (OpenAI Responses compact, Anthropic compaction) when the active route supports it.",
	},
	{
		value: "snapcompact",
		label: "Snapcompact",
		description:
			"Archive history onto dense bitmap images the active vision model reads back; no LLM call.",
	},
	{
		value: "handoff",
		label: "Handoff",
		description:
			"Generate a handoff document and continue from it as the compaction summary.",
	},
	{
		value: "soft",
		label: "Soft compaction",
		description:
			"Summarize in place with a compaction model without using server compaction.",
	},
	{
		value: "shake",
		label: "Shake",
		description:
			"Drop recoverable heavy content in place without an LLM call.",
	},
] as const;

/** Default fallback order: server-native first, portable summary last. */
export const DEFAULT_COMPACTION_METHOD_ORDER: CompactionMethod[] = [
	"remote",
	"snapcompact",
	"handoff",
	"shake",
	"soft",
];

const COMPACTION_METHODS: Record<CompactionMethod, true> = {
	remote: true,
	snapcompact: true,
	handoff: true,
	soft: true,
	shake: true,
};

/** Whether a string names a supported compaction method. */
export function isCompactionMethod(value: unknown): value is CompactionMethod {
	return typeof value === "string" && Object.hasOwn(COMPACTION_METHODS, value);
}

/**
 * Filter malformed entries and preserve first-occurrence order from a configured
 * compaction-method preference list.
 */
export function resolveCompactionMethodOrder(
	value: unknown,
): CompactionMethod[] {
	if (!Array.isArray(value)) return [];

	const methods: CompactionMethod[] = [];
	for (const method of value) {
		if (isCompactionMethod(method) && !methods.includes(method))
			methods.push(method);
	}
	return methods;
}

/** Map from compaction method to the underlying strategy name. */
export const STRATEGY_BY_COMPACTION_METHOD: Record<
	CompactionMethod,
	"remote" | "snapcompact" | "handoff" | "shake" | "soft"
> = {
	remote: "remote",
	snapcompact: "snapcompact",
	handoff: "handoff",
	soft: "soft",
	shake: "shake",
};

/** Get the label for a compaction method (for UI display). */
export function getCompactionMethodLabel(method: CompactionMethod): string {
	return (
		COMPACTION_METHOD_CHOICES.find(c => c.value === method)?.label ?? method
	);
}

/** Get the description for a compaction method. */
export function getCompactionMethodDescription(
	method: CompactionMethod,
): string {
	return (
		COMPACTION_METHOD_CHOICES.find(c => c.value === method)
			?.description ?? ""
	);
}

// Re-export resolver functions for convenience
export {
	resolveCompactionMethod,
	isMethodAvailable,
	getFallbackChain,
	shouldUseServerCompaction,
} from "./resolver.ts";

export type { CompactionCapabilities, CompactionSettings } from "./resolver.ts";
