// ── Observer agent ───────────────────────────────────────────────────────
// Extracts structured observations from a chunk of session entries.
// Called by the consolidation pipeline on turn_end.

import { estimateTokens } from "./tokens.ts";
import type { Observation } from "./types.ts";
import { OBSERVER_SYSTEM_PROMPT } from "./prompts.ts";

export interface ObserverConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	priorObservations: string[];
	priorReflections: string[];
	chunk: string;
	allowedSourceEntryIds: string[];
	maxTurns?: number;
	thinkingLevel?: string;
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
		maxTurns: _maxTurns,
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

	try {
		const response = await callLLM(systemPrompt, userMessage, {
			model,
			apiKey,
			baseUrl,
			headers,
			thinkingLevel,
		});
		return parseObservations(response, allowedSourceEntryIds);
	} catch {
		return undefined;
	}
}

async function callLLM(
	systemPrompt: string,
	userMessage: string,
	config: {
		model: unknown;
		apiKey: string;
		baseUrl?: string;
		headers?: Record<string, string>;
		thinkingLevel?: string;
	},
): Promise<string> {
	const { model: modelName, apiKey: key, baseUrl, headers: h, thinkingLevel } = config;

	const body: Record<string, unknown> = {
		model: typeof modelName === "string" ? modelName : "gpt-4o",
		messages: [
			{ role: "system", content: systemPrompt },
			{ role: "user", content: userMessage },
		],
		response_format: { type: "json_object" },
		max_tokens: 2048,
	};

	if (thinkingLevel && thinkingLevel !== "off") {
		body["thinking"] = thinkingLevel;
	}

	const response = await fetch(`${(baseUrl ?? "https://api.openai.com").replace(/\/+$/, "")}/v1/chat/completions`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			...(key ? { Authorization: `Bearer ${key}` } : {}),
			...(h ?? {}),
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		throw new Error(
			`LLM call failed: ${response.status} ${response.statusText}`,
		);
	}

	const data = (await response.json()) as {
		choices: Array<{ message: { content: string } }>;
	};
	return data.choices[0]?.message?.content ?? "";
}

function parseObservations(
	raw: string,
	allowedIds: string[],
): Observation[] | undefined {
	// Extract JSON from response (may be wrapped in markdown code blocks)
	const jsonMatch = raw.match(/\[[\s\S]*\]/);
	if (!jsonMatch) return undefined;

	try {
		const parsed = JSON.parse(jsonMatch[0]) as unknown;
		if (!Array.isArray(parsed)) return undefined;

		const observations: Observation[] = [];
		for (const item of parsed) {
			if (!isObservation(item)) continue;
			// Validate sourceEntryIds against allowed set
			const validIds = item.sourceEntryIds.filter((id) =>
				allowedIds.includes(id),
			);
			if (validIds.length === 0) continue;

			observations.push({
				...item,
				sourceEntryIds: validIds,
				tokenCount: item.tokenCount ?? estimateTokens(item.content),
			});
		}
		return observations;
	} catch {
		return undefined;
	}
}

function isObservation(value: unknown): value is Observation {
	if (!value || typeof value !== "object") return false;
	const o = value as Record<string, unknown>;
	if (typeof o.id !== "string" || !/^[a-f0-9]{12}$/.test(o.id)) return false;
	if (typeof o.content !== "string" || !o.content) return false;
	if (typeof o.timestamp !== "string" || !o.timestamp) return false;
	if (!["low", "medium", "high", "critical"].includes(String(o.relevance)))
		return false;
	if (!Array.isArray(o.sourceEntryIds) || o.sourceEntryIds.length === 0)
		return false;
	return true;
}
