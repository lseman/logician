// ── Observer agent ───────────────────────────────────────────────────────
// Extracts structured observations from a chunk of session entries.
// Called by the consolidation pipeline on turn_end.

import { hashId } from "./ids.ts";
import { callStructuredLLM } from "./llm-client.ts";
import { OBSERVER_SYSTEM_PROMPT } from "./prompts.ts";
import { estimateTokens } from "./tokens.ts";
import type { Observation } from "./types.ts";

export interface ObserverConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	priorObservations: string[];
	priorReflections: string[];
	chunk: string;
	allowedSourceEntryIds: string[];
	thinkingLevel?: string;
	signal?: AbortSignal;
}

export async function runObserver(
	config: ObserverConfig,
): Promise<Observation[] | undefined> {
	const {
		model,
		apiKey,
		baseUrl,
		headers,
		priorObservations,
		priorReflections,
		chunk,
		allowedSourceEntryIds,
		thinkingLevel = "off",
	} = config;

	const systemPrompt = [
		OBSERVER_SYSTEM_PROMPT,
		priorObservations.length > 0
			? `\n\n## Prior observations (avoid duplication):\n${priorObservations.map((o) => `  - ${o}`).join("\n")}`
			: "",
		priorReflections.length > 0
			? `\n\n## Prior reflections (inform context):\n${priorReflections.map((r) => `  - ${r}`).join("\n")}`
			: "",
	].join("\n");

	const userMessage = `Extract observations from the following source entries.\n\nSource entries:\n${chunk}`;

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
			name: "record_observations",
			description:
				"Record validated observations from the supplied source entries.",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					observations: {
						type: "array",
						items: {
							type: "object",
							additionalProperties: false,
							properties: {
								content: { type: "string" },
								timestamp: { type: "string" },
								relevance: {
									type: "string",
									enum: ["low", "medium", "high", "critical"],
								},
								sourceEntryIds: {
									type: "array",
									items: { type: "string" },
								},
							},
							required: ["content", "timestamp", "relevance", "sourceEntryIds"],
						},
					},
				},
				required: ["observations"],
			},
		},
	);
	return parseObservations(response, allowedSourceEntryIds);
}

export function parseObservations(
	raw: unknown,
	allowedIds: string[],
): Observation[] | undefined {
	if (!raw || typeof raw !== "object") return undefined;
	const parsed = (raw as Record<string, unknown>).observations;
	if (!Array.isArray(parsed)) return undefined;
	const allowed = new Set(allowedIds);

	const observations: Observation[] = [];
	for (const item of parsed) {
		if (!isObservationProposal(item)) continue;
		// Validate sourceEntryIds against allowed set
		const validIds = item.sourceEntryIds.filter((id) => allowed.has(id));
		if (validIds.length === 0) continue;

		const content = item.content.trim().replace(/\s+/g, " ");
		if (!content) continue;
		observations.push({
			id: hashId(content),
			content,
			timestamp: item.timestamp,
			relevance: item.relevance,
			sourceEntryIds: validIds,
			tokenCount: estimateTokens(content),
		});
	}
	return observations;
}

type ObservationProposal = Omit<Observation, "id" | "tokenCount">;

function isObservationProposal(value: unknown): value is ObservationProposal {
	if (!value || typeof value !== "object") return false;
	const o = value as Record<string, unknown>;
	if (typeof o.content !== "string" || !o.content) return false;
	if (typeof o.timestamp !== "string" || !o.timestamp) return false;
	if (!["low", "medium", "high", "critical"].includes(String(o.relevance)))
		return false;
	if (!Array.isArray(o.sourceEntryIds) || o.sourceEntryIds.length === 0)
		return false;
	return true;
}
