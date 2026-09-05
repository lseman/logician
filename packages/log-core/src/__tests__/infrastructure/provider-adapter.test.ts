import { expect, test } from "bun:test";
import { OpenAIChatCompletionsAdapter } from "../../capabilities/provider/provider-adapter.ts";

test("chat-completions adapter owns endpoint and provider-specific payload", () => {
	const adapter = new OpenAIChatCompletionsAdapter();
	const payload = adapter.buildPayload({
		model: "model",
		messages: [{ role: "user", content: "hello" }],
		tools: [{ type: "function", function: { name: "read_file" } }],
		temperature: 0.2,
		maxTokens: 2048,
		thinkingLevel: "high",
		thinkingFormat: "qwen",
	});

	expect(adapter.endpoint("http://localhost:8080")).toBe(
		"http://localhost:8080/v1/chat/completions",
	);
	expect(payload).toMatchObject({
		model: "model",
		max_tokens: 2048,
		enable_thinking: true,
		reasoning_effort: "high",
		stream: true,
	});
});
