// ── Reflector agent ──────────────────────────────────────────────────────
// Synthesizes higher-level reflections from observations.
// Called by the consolidation pipeline when reflection threshold is reached.

import { estimateTokens } from "./tokens.ts";
import type { Reflection } from "./types.ts";
import { REFLECTOR_SYSTEM_PROMPT } from "./prompts.ts";
import { callLLM, extractJsonArray } from "./llm-client.ts";

export interface ReflectorConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	observations: Array<{
		content: string;
		coverage: "none" | "partial" | "strong";
	}>;
	reflections: Reflection[];
	thinkingLevel?: string;
}

export async function runReflector(
	config: ReflectorConfig,
): Promise<Reflection[] | undefined> {
	const {
		model,
		apiKey,
		baseUrl,
		headers,
		observations,
		reflections: _existingReflections,
		thinkingLevel = "off",
	} = config;

	const obsList = observations
		.map((o) => `[${o.coverage}] ${o.content}`)
		.join("\n");

	const systemPrompt = `${REFLECTOR_SYSTEM_PROMPT}\n\n## Active observations to reflect on:\n${obsList}`;

	const userMessage =
		"Analyze the observations above and produce higher-level reflections. Return only JSON.";

	try {
		const response = await callLLM(systemPrompt, userMessage, {
			model,
			apiKey,
			baseUrl,
			headers,
			thinkingLevel,
		});
		return parseReflections(response);
	} catch (e: unknown) {
		return undefined;
	}
}

function parseReflections(raw: string): Reflection[] | undefined {
	const parsed = extractJsonArray(raw);
	if (!parsed) return undefined;

	const reflections: Reflection[] = [];
	for (const item of parsed) {
		if (!isReflection(item)) continue;
		reflections.push({
			...item,
			tokenCount: item.tokenCount ?? estimateTokens(item.content),
		});
	}
	return reflections;
}

function isReflection(value: unknown): value is Reflection {
	if (!value || typeof value !== "object") return false;
	const r = value as Record<string, unknown>;
	if (typeof r.id !== "string" || !/^[a-f0-9]{12}$/.test(r.id)) return false;
	if (typeof r.content !== "string" || !r.content) return false;
	if (/\r|\n/.test(r.content)) return false;
	if (
		!Array.isArray(r.supportingObservationIds) ||
		r.supportingObservationIds.length === 0
	)
		return false;
	return true;
}
