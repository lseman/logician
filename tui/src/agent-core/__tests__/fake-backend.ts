// Scriptable LLMBackend for tests: each call shifts the next responder.
// A responder gets the generate options (for callbacks/signal) and returns an
// LLMResponse — or a promise that settles however the test scripts it.

import type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "../core/backend.ts";

export type Responder = (
	messages: Record<string, unknown>[],
	options: GenerateOptions,
) => Promise<LLMResponse> | LLMResponse;

export function textResponse(content: string): LLMResponse {
	return { content, toolCalls: [], stopReason: "stop" };
}

export class FakeBackend implements LLMBackend {
	readonly model = "fake";
	calls = 0;
	private responders: Responder[];

	constructor(responders: Responder[]) {
		this.responders = responders;
	}

	withModel(): LLMBackend {
		return this;
	}

	async generate(
		messages: Record<string, unknown>[],
		options: GenerateOptions = {},
	): Promise<LLMResponse> {
		this.calls++;
		const responder = this.responders.shift();
		if (!responder) return textResponse("(out of scripted responses)");
		return responder(messages, options);
	}
}
