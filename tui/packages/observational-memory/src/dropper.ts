// ── Dropper agent ────────────────────────────────────────────────────────
// Prunes observation pool by identifying safe-to-drop observations.
// Called by the consolidation pipeline only after reflections are recorded.

import type { Observation, Reflection } from "./types.ts";
import { DROPPER_SYSTEM_PROMPT } from "./prompts.ts";
import { callLLM, extractJsonArray } from "./llm-client.ts";

export interface DropperConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	observations: Observation[];
	reflections: Reflection[];
	targetTokens: number;
	thinkingLevel?: string;
}

export async function runDropper(
	config: DropperConfig,
): Promise<string[] | undefined> {
	const {
		model,
		apiKey,
		baseUrl,
		headers,
		observations,
		reflections: _reflections,
		targetTokens: _targetTokens,
		thinkingLevel = "off",
	} = config;

	const obsList = observations
		.map((o) => `[${o.relevance}] ${o.id}: ${o.content}`)
		.join("\n");

	const systemPrompt = `${DROPPER_SYSTEM_PROMPT}\n\n## Active observations:\n${obsList}`;
	const userMessage =
		"Return a JSON array of observation IDs to drop. Return only JSON.";

	try {
		const response = await callLLM(systemPrompt, userMessage, {
			model,
			apiKey,
			baseUrl,
			headers,
			thinkingLevel,
		});
		return parseDropIds(response);
	} catch (e: unknown) {
		return undefined;
	}
}

function parseDropIds(raw: string): string[] | undefined {
	const parsed = extractJsonArray(raw);
	if (!parsed) return undefined;
	return parsed.filter(
		(id: unknown) => typeof id === "string" && /^[a-f0-9]{12}$/.test(id),
	) as string[];
}
