// ── Dropper agent ────────────────────────────────────────────────────────
// Prunes observation pool by identifying safe-to-drop observations.
// Called by the consolidation pipeline only after reflections are recorded.

import { callStructuredLLM } from "./llm-client.ts";
import { DROPPER_SYSTEM_PROMPT } from "./prompts.ts";
import type { Observation, Reflection } from "./types.ts";

export interface DropperConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	observations: Observation[];
	reflections: Reflection[];
	targetTokens: number;
	thinkingLevel?: string;
	signal?: AbortSignal;
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
		reflections,
		targetTokens,
		thinkingLevel = "off",
	} = config;

	const coverage = reflectionCoverage(observations, reflections);
	const obsList = observations
		.map(
			(o) =>
				`[${o.id}] ${o.timestamp} [${o.relevance}] [coverage:${coverage.get(o.id) ?? "none"}] ${o.content}`,
		)
		.join("\n");

	const systemPrompt = `${DROPPER_SYSTEM_PROMPT}\n\n## Active observations:\n${obsList}\n\nTarget tokens: ${targetTokens}`;
	const userMessage =
		"Return a JSON array of observation IDs to drop. Return only JSON.";

	const response = await callStructuredLLM(
		systemPrompt,
		userMessage,
		{
			model,
			apiKey,
			baseUrl,
			headers,
			thinkingLevel,
			signal: config.signal,
		},
		{
			name: "drop_observations",
			description: "Propose active observation IDs that are safe to archive.",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					ids: { type: "array", items: { type: "string" } },
				},
				required: ["ids"],
			},
		},
	);
	return parseDropIds(response);
}

function parseDropIds(raw: unknown): string[] | undefined {
	if (!raw || typeof raw !== "object") return undefined;
	const ids = (raw as Record<string, unknown>).ids;
	if (!Array.isArray(ids)) return undefined;
	return ids.filter(
		(id: unknown) => typeof id === "string" && /^[a-f0-9]{12}$/.test(id),
	) as string[];
}

function reflectionCoverage(
	observations: readonly Observation[],
	reflections: readonly Reflection[],
): Map<string, "partial" | "strong"> {
	const counts = new Map<string, number>();
	const valid = new Set(observations.map((item) => item.id));
	for (const reflection of reflections) {
		for (const id of new Set(reflection.supportingObservationIds)) {
			if (valid.has(id)) counts.set(id, (counts.get(id) ?? 0) + 1);
		}
	}
	return new Map(
		Array.from(counts, ([id, count]) => [
			id,
			count >= 2 ? "strong" : "partial",
		]),
	);
}
