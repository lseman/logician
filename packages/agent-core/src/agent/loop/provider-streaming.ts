/**
 * Streaming callback builder for provider generate() calls.
 *
 * Wraps each streaming event through queueProviderEvent, which chains
 * its settlement so terminal events (text_end, tool_call_end) cannot
 * overtake delta events on the SSE queue.
 *
 * This is a pure builder — it takes a turnId and a queue function and
 * returns a GenerateCallbacks object suitable for LLMBackend.generate().
 */

import type { GenerateCallbacks } from "../backend.ts";
import type { AgentEvent } from "../types.ts";

/**
 * Build streaming callbacks that funnel through the per-request event chain.
 *
 * The queue function is responsible for ordering: SSE deltas are emitted
 * immediately while terminal events retain their settlement promise so
 * message_end/turn_end cannot overtake them.
 */
export function buildStreamingCallbacks(
	turnId: string,
	queueProviderEvent: (event: AgentEvent) => void,
): GenerateCallbacks {
	return {
		onDelta: delta =>
			queueProviderEvent({ type: "text_delta", turnId, delta }),
		onThinking: delta =>
			queueProviderEvent({ type: "thinking_delta", turnId, delta }),
		onTextStart: () =>
			queueProviderEvent({ type: "text_start", turnId }),
		onTextEnd: () => queueProviderEvent({ type: "text_end", turnId }),
		onToolCallStart: (toolCallId, toolName, args) =>
			queueProviderEvent({
				type: "tool_call_start",
				toolCallId,
				toolName,
				args,
			}),
		onToolCallDelta: (toolCallId, delta) =>
			queueProviderEvent({
				type: "tool_call_delta",
				toolCallId,
				delta,
			}),
		onToolCallIdUpdate: (previousToolCallId, toolCallId) =>
			queueProviderEvent({
				type: "tool_call_id_update",
				previousToolCallId,
				toolCallId,
			}),
	};
}
