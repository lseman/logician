// ── Reflector agent ──────────────────────────────────────────────────────
// Synthesizes higher-level reflections from observations.
// Called by the consolidation pipeline when reflection threshold is reached.

import { estimateTokens } from "./tokens.ts";
import type { Reflection } from "./types.ts";
import { REFLECTOR_SYSTEM_PROMPT } from "./prompts.ts";

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
	maxTurns?: number;
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
		maxTurns: _maxTurns,
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

function parseReflections(raw: string): Reflection[] | undefined {
	const jsonMatch = raw.match(/\[[\s\S]*\]/);
	if (!jsonMatch) return undefined;

	try {
		const parsed = JSON.parse(jsonMatch[0]) as unknown;
		if (!Array.isArray(parsed)) return undefined;

		const reflections: Reflection[] = [];
		for (const item of parsed) {
			if (!isReflection(item)) continue;
			reflections.push({
				...item,
				tokenCount: item.tokenCount ?? estimateTokens(item.content),
			});
		}
		return reflections;
	} catch (e: unknown) {
		return undefined;
	}
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
