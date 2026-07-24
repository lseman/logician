// ── Shared LLM client for consolidation agents ───────────────────────────
// Single non-streaming JSON-completion call, used by observer/reflector/dropper.

export interface LLMCallConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	thinkingLevel?: string;
}

export async function callLLM(
	systemPrompt: string,
	userMessage: string,
	config: LLMCallConfig,
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

/** Extract the first JSON array from a raw LLM response, tolerating markdown fences. */
export function extractJsonArray(raw: string): unknown[] | undefined {
	const jsonMatch = raw.match(/\[[\s\S]*\]/);
	if (!jsonMatch) return undefined;
	try {
		const parsed = JSON.parse(jsonMatch[0]) as unknown;
		return Array.isArray(parsed) ? parsed : undefined;
	} catch {
		return undefined;
	}
}
