// ── Shared LLM client for consolidation agents ───────────────────────────
// Single non-streaming JSON-completion call, used by observer/reflector/dropper.

export interface LLMCallConfig {
	model: unknown;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	thinkingLevel?: string;
	signal?: AbortSignal;
}

export async function callStructuredLLM(
	systemPrompt: string,
	userMessage: string,
	config: LLMCallConfig,
	tool: {
		name: string;
		description: string;
		parameters: Record<string, unknown>;
	},
): Promise<unknown> {
	const {
		model: modelName,
		apiKey: key,
		baseUrl,
		headers: h,
		thinkingLevel,
	} = config;

	const body: Record<string, unknown> = {
		model: typeof modelName === "string" ? modelName : "gpt-4o",
		messages: [
			{ role: "system", content: systemPrompt },
			{ role: "user", content: userMessage },
		],
		tools: [{ type: "function", function: tool }],
		tool_choice: "required",
		max_tokens: 2048,
	};

	if (thinkingLevel && thinkingLevel !== "off") {
		body.thinking = thinkingLevel;
	}

	const response = await fetch(
		`${(baseUrl ?? "https://api.openai.com").replace(/\/+$/, "")}/v1/chat/completions`,
		{
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				...(key ? { Authorization: `Bearer ${key}` } : {}),
				...(h ?? {}),
			},
			body: JSON.stringify(body),
			signal: config.signal,
		},
	);

	if (!response.ok) {
		throw new Error(
			`LLM call failed: ${response.status} ${response.statusText}`,
		);
	}

	const data = (await response.json()) as {
		choices?: Array<{
			message?: {
				content?: string;
				tool_calls?: Array<{
					function?: { name?: string; arguments?: string };
				}>;
			};
		}>;
	};
	const message = data.choices?.[0]?.message;
	const call = message?.tool_calls?.find(
		(item) => item.function?.name === tool.name,
	);
	const raw = call?.function?.arguments ?? message?.content;
	if (!raw) return undefined;
	try {
		return JSON.parse(raw) as unknown;
	} catch {
		return extractJsonValue(raw);
	}
}

/** Extract the first JSON object or array, tolerating markdown fences. */
export function extractJsonValue(raw: string): unknown {
	const jsonMatch = raw.match(/(?:\{[\s\S]*\}|\[[\s\S]*\])/);
	if (!jsonMatch) return undefined;
	try {
		return JSON.parse(jsonMatch[0]) as unknown;
	} catch {
		return undefined;
	}
}
