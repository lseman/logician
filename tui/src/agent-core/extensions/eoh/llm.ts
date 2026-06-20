// ── EoH LLM caller ────────────────────────────────────────────────────────────
// Calls OpenAI-compatible API directly (not through harness backend) so EoH
// can run concurrently with the main agent session.

export interface LLMCallOptions {
	baseUrl: string;
	model: string;
	messages: Array<{ role: string; content: string }>;
	temperature?: number;
	maxTokens?: number;
}

export async function callLLM(options: LLMCallOptions): Promise<string> {
	const { baseUrl, model, messages, temperature = 0.8, maxTokens = 2048 } = options;
	const url = baseUrl.replace(/\/$/, "") + "/chat/completions";

	const apiKey = process.env.ANTHROPIC_API_KEY
		?? process.env.OPENAI_API_KEY
		?? process.env.LLM_API_KEY
		?? "sk-no-key";

	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"Authorization": `Bearer ${apiKey}`,
			"x-api-key": apiKey,
		},
		body: JSON.stringify({
			model,
			messages,
			temperature,
			max_tokens: maxTokens,
		}),
	});

	if (!res.ok) {
		const text = await res.text().catch(() => "");
		throw new Error(`LLM API error ${res.status}: ${text.slice(0, 200)}`);
	}

	const data = await res.json() as {
		choices: Array<{ message: { content: string } }>;
	};
	const content = data.choices?.[0]?.message?.content;
	if (!content) throw new Error("Empty LLM response");
	return content;
}
