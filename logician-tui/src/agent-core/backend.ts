// ── LLM Backend ──────────────────────────────────────────────────────────────────
// OpenAI-compatible HTTP client for streaming LLM responses.
// Mirrors Python LlamaCppClient/VLLMClient but simplified for TS.

import type { ThinkingLevel, ToolCall } from "./types.ts";

export interface LLMResponse {
    content: string | null;
    toolCalls: ToolCall[];
    stopReason: "stop" | "length" | "error";
    errorMessage?: string;
}

export interface LLMBackend {
    generate(
        messages: Record<string, unknown>[],
        tools?: Record<string, unknown>[],
        temperature?: number,
        maxTokens?: number,
        signal?: AbortSignal,
        onDelta?: (delta: string) => void,
        onThinking?: (delta: string) => void,
        thinkingLevel?: ThinkingLevel,
    ): Promise<LLMResponse>;
}

export class OpenAIBackend implements LLMBackend {
    private baseUrl: string;
    private model: string;
    private chatTemplate?: string;
    private stop?: string[];
    private defaultThinkingLevel: ThinkingLevel = "medium";

    constructor(options: {
        baseUrl: string;
        model: string;
        chatTemplate?: string;
        stop?: string[];
        thinkingLevel?: ThinkingLevel;
    }) {
        this.baseUrl = options.baseUrl.replace(/\/+$/, "");
        this.model = options.model;
        this.chatTemplate = options.chatTemplate;
        this.stop = options.stop;
        this.defaultThinkingLevel = options.thinkingLevel ?? "medium";
    }

    /** Update the default thinking level at runtime. */
    setDefaultThinkingLevel(level: ThinkingLevel): void {
        this.defaultThinkingLevel = level;
    }

    async generate(
        messages: Record<string, unknown>[],
        tools?: Record<string, unknown>[] | undefined,
        temperature: number = 0.5,
        maxTokens: number = 4096,
        signal?: AbortSignal | undefined,
        onDelta?: (delta: string) => void,
        onThinking?: (delta: string) => void,
        thinkingLevel?: ThinkingLevel,
    ): Promise<LLMResponse> {
        const body: Record<string, unknown> = {
            model: this.model,
            messages,
            temperature,
            max_tokens: maxTokens,
            stream: true,
            ...(this.stop && { stop: this.stop }),
        };

        // Pass reasoning/thinking level to OpenAI-compatible providers.
        // "off" = omit reasoning field entirely.
        const effectiveLevel = thinkingLevel ?? this.defaultThinkingLevel;
        if (effectiveLevel !== "off") {
            body.reasoning = { effort: effectiveLevel };
        }

        if (tools && tools.length > 0) {
            body.tools = tools;
        }

        const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
            signal,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
                `LLM request failed: ${response.status} ${errorText}`,
            );
        }

        if (!response.body) {
            throw new Error("Response body is unavailable");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullContent = "";
        let toolCalls: ToolCall[] = [];
        let stopReason: LLMResponse["stopReason"] = "stop";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const data = line.slice(6).trim();
                if (!data || data === "[DONE]") continue;

                try {
                    const chunk = JSON.parse(data);
                    const delta = chunk.choices?.[0]?.delta;
                    if (!delta) continue;

                    if (delta.content) {
                        onDelta?.(delta.content);
                        fullContent += delta.content;
                    }

                    if (delta.reasoning_content) {
                        onThinking?.(delta.reasoning_content);
                    }

                    if (delta.tool_calls) {
                        for (const tc of delta.tool_calls) {
                            // Accumulate tool call
                            if (!toolCalls[tc.index]) {
                                toolCalls[tc.index] = {
                                    id: "",
                                    name: "",
                                    arguments: "",
                                };
                            }
                            if (tc.id) toolCalls[tc.index].id = tc.id;
                            if (tc.function?.name)
                                toolCalls[tc.index].name = tc.function.name;
                            if (tc.function?.arguments) {
                                toolCalls[tc.index].arguments +=
                                    tc.function.arguments;
                            }
                        }
                    }
                } catch {
                    // Skip parse errors (partial JSON is normal in streaming)
                }
            }
        }

        toolCalls = toolCalls
            .filter((tc) => tc && tc.name)
            .map((tc, index) => ({
                id: tc.id || `tool_${index}`,
                name: tc.name,
                arguments: tc.arguments || "{}",
            }));

        if (toolCalls.length > 0) {
            stopReason = "stop";
        }

        return {
            content: fullContent || null,
            toolCalls,
            stopReason,
        };
    }
}
