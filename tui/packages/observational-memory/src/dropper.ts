// ── Dropper agent ────────────────────────────────────────────────────────
// Prunes observation pool by identifying safe-to-drop observations.
// Called by the consolidation pipeline only after reflections are recorded.

import type { Observation, Reflection } from "./types.ts";
import { DROPPER_SYSTEM_PROMPT } from "./prompts.ts";

export interface DropperConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	observations: Observation[];
	reflections: Reflection[];
	targetTokens: number;
	maxTurns?: number;
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
		maxTurns: _maxTurns,
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

function parseDropIds(raw: string): string[] | undefined {
	const jsonMatch = raw.match(/\[[\s\S]*\]/);
	if (!jsonMatch) return undefined;

	try {
		const parsed = JSON.parse(jsonMatch[0]) as unknown;
		if (!Array.isArray(parsed)) return undefined;
		return parsed.filter(
			(id: unknown) => typeof id === "string" && /^[a-f0-9]{12}$/.test(id),
		) as string[];
	} catch {
		return undefined;
	}
}
