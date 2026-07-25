// ── Reflector agent ──────────────────────────────────────────────────────
// Synthesizes higher-level reflections from observations.
// Called by the consolidation pipeline when reflection threshold is reached.

import { hashId } from "../ids.ts";
import { callStructuredLLM } from "./llm-client.ts";
import { REFLECTOR_SYSTEM_PROMPT } from "./prompts.ts";
import { estimateTokens } from "../tokens.ts";
import type { Reflection, Relevance } from "../types.ts";

export interface ReflectorConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	observations: Array<{
		id: string;
		content: string;
		timestamp: string;
		relevance: Relevance;
		coverage: "none" | "partial" | "strong";
	}>;
	reflections: Reflection[];
	thinkingLevel?: string;
	signal?: AbortSignal;
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
		.map(
			(o) =>
				`[${o.id}] ${o.timestamp} [${o.relevance}] [coverage:${o.coverage}] ${o.content}`,
		)
		.join("\n");

	const systemPrompt = `${REFLECTOR_SYSTEM_PROMPT}\n\n## Active observations to reflect on:\n${obsList}`;

	const userMessage =
		"Analyze the observations above and produce higher-level reflections. Return only JSON.";

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
			name: "record_reflections",
			description:
				"Record durable reflections with valid supporting observation IDs.",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					reflections: {
						type: "array",
						items: {
							type: "object",
							additionalProperties: false,
							properties: {
								content: { type: "string" },
								supportingObservationIds: {
									type: "array",
									items: { type: "string" },
								},
							},
							required: ["content", "supportingObservationIds"],
						},
					},
				},
				required: ["reflections"],
			},
		},
	);
	return parseReflections(
		response,
		observations.map((item) => item.id),
	);
}

export function parseReflections(
	raw: unknown,
	allowedObservationIds: readonly string[],
): Reflection[] | undefined {
	if (!raw || typeof raw !== "object") return undefined;
	const parsed = (raw as Record<string, unknown>).reflections;
	if (!Array.isArray(parsed)) return undefined;
	const allowed = new Set(allowedObservationIds);

	const reflections: Reflection[] = [];
	for (const item of parsed) {
		if (!isReflectionProposal(item)) continue;
		if (item.supportingObservationIds.some((id) => !allowed.has(id))) continue;
		const content = item.content.trim().replace(/\s+/g, " ");
		if (!content || /\r|\n/.test(content)) continue;
		reflections.push({
			id: hashId(content),
			content,
			supportingObservationIds: [...new Set(item.supportingObservationIds)],
			tokenCount: estimateTokens(content),
		});
	}
	return reflections;
}

type ReflectionProposal = Omit<Reflection, "id" | "tokenCount">;

function isReflectionProposal(value: unknown): value is ReflectionProposal {
	if (!value || typeof value !== "object") return false;
	const r = value as Record<string, unknown>;
	if (typeof r.content !== "string" || !r.content) return false;
	if (/\r|\n/.test(r.content)) return false;
	if (
		!Array.isArray(r.supportingObservationIds) ||
		r.supportingObservationIds.length === 0
	)
		return false;
	return true;
}
