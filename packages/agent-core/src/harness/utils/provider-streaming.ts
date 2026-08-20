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

import type { AgentEvent } from "../../types/index.ts";
import { createAssistantMessage } from "../messages.ts";
import type { GenerateCallbacks } from "../utils/backend.ts";

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
		onDelta: delta => queueProviderEvent({ type: "text_delta", turnId, delta }),
		onThinking: delta =>
			queueProviderEvent({ type: "thinking_delta", turnId, delta }),
		onTextStart: () => queueProviderEvent({ type: "text_start", turnId }),
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
		// The backend's own coherent mid-stream accumulation. A consumer (e.g.
		// the TUI's Transcript) can apply this wholesale instead of manually
		// reconciling onDelta/onThinking/onToolCallDelta against a prior snapshot.
		onSnapshot: snapshot => {
			// During a pure-reasoning phase (no content or tool calls yet),
			// createAssistantMessage pads empty content to " " so a *final*
			// assistant message stays API-valid — but that padding isn't
			// meaningful for a mid-stream snapshot. Emitting it as a
			// message_update would inject a phantom whitespace content chunk
			// that closes the in-progress thinking chunk, fragmenting one
			// reasoning segment into two.
			if (snapshot.content || snapshot.toolCalls.length) {
				queueProviderEvent({
					type: "message_update",
					turnId,
					message: createAssistantMessage(snapshot.content, snapshot.toolCalls),
				});
			}
			if (snapshot.reasoning) {
				queueProviderEvent({
					type: "message_reasoning_update",
					turnId,
					reasoning: snapshot.reasoning,
				});
			}
		},
	};
}
