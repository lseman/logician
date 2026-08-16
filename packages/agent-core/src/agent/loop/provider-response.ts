import {
    parseTextToolCalls,
    stripTextToolCalls,
} from "../../tools/shared/text-to-tool-calls.ts";
import type { LLMResponse } from "../backend.ts";
import { stopReasonFor } from "../loop/callbacks.ts";
import type { StopReason, ToolCall } from "../types.ts";

export interface NormalizedProviderResponse {
    assistantContent: string;
    toolCalls: ToolCall[];
    rawStopReason: "stop" | "length" | "error";
    stopReason: StopReason;
}

export function normalizeProviderResponse(
    response: LLMResponse,
    isKnownTool: (name: string) => boolean,
): NormalizedProviderResponse {
    let toolCalls = response.toolCalls ?? [];
    let assistantContent = response.content ?? "";
    if (toolCalls.length === 0 && response.content) {
        const textCalls = parseTextToolCalls(response.content, isKnownTool);
        if (textCalls.length > 0) {
            toolCalls = textCalls;
            assistantContent = stripTextToolCalls(response.content);
        }
    }
    const rawStopReason = response.stopReason ?? "stop";
    return {
        assistantContent,
        toolCalls,
        rawStopReason,
        stopReason: stopReasonFor(rawStopReason, toolCalls),
    };
}
