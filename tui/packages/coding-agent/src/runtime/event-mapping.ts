// ── Agent event → bridge event shape mapping ──────────────────────────────────

import {
	STEERING_INTERRUPT_SUMMARY,
	type AgentEvent,
} from "@logician/agent-core";
import type { ParsedBridgeEvent } from "./events.ts";

export function mapAgentEvent(event: AgentEvent): ParsedBridgeEvent | null {
	switch (event.type) {
		case "message_start":
			return {
				type: "message_start",
				turnId: event.turnId,
				role: event.role,
			} as ParsedBridgeEvent;
		case "text_start":
			return { type: "text_start", turnId: event.turnId };
		case "text_delta":
			return { type: "token", token: event.delta };
		case "text_end":
			return { type: "text_end", turnId: event.turnId };
		case "message_update":
			return {
				type: "message_update",
				turnId: event.turnId,
				message: event.message,
			} as ParsedBridgeEvent;
		case "thinking_delta":
			return { type: "thinking_token", token: event.delta };
		case "tool_call_start":
			return {
				type: "tool_execution_start",
				tool: event.toolName,
				tool_name: event.toolName,
				tool_args: parseToolArgs(event.args),
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_call_end":
			return {
				type: "tool_execution_end",
				tool: event.toolName,
				tool_name: event.toolName,
				result: event.result,
				is_error: event.isError,
				tool_call_id: event.toolCallId,
				details: event.details,
			} as ParsedBridgeEvent;
		case "tool_call_delta":
			return {
				type: "tool_execution_update",
				tool: "",
				tool_name: "",
				partial_result: event.delta,
				update_kind: "arguments",
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_execution_start":
			return {
				type: "tool_execution_start",
				tool: event.toolName,
				tool_name: event.toolName,
				tool_args: event.args,
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_execution_end":
			return {
				type: "tool_execution_end",
				tool: event.toolName,
				tool_name: event.toolName,
				result: event.result,
				is_error: event.isError,
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "tool_execution_update":
			return {
				type: "tool_execution_update",
				tool: event.toolName,
				tool_name: event.toolName,
				partial_result: event.partialResult,
				update_kind: "output",
				tool_call_id: event.toolCallId,
			} as ParsedBridgeEvent;
		case "repair_nudge":
			return {
				type: "repair_nudge",
				turn_id: event.turnId,
				repair_stage: event.repairStage,
				tool_name: event.toolName,
				message: event.message,
			};
		case "turn_start":
			return null; // The bridge owns the user-visible turn lifecycle.
		case "turn_end":
			// Core turn_end is per model/tool iteration, not per user-visible turn.
			// Reconcile its final assistant snapshot without completing the UI card;
			// runMessage emits the single terminal turn_end after prompt() settles.
			return event.message?.role === "assistant"
				? {
						type: "message_update",
						turnId: event.turnId,
						message: {
							role: event.message.role,
							content: event.message.content,
							tool_calls: event.message.tool_calls,
						},
					}
				: null;
		case "agent_start":
		case "agent_end":
		case "phase":
			return null; // Handled separately
		case "context_update":
			return {
				type: "context_update",
				tokens: event.tokens,
				max_tokens: event.maxTokens,
				compacted: event.compacted,
				...(event.cachedTokens !== undefined && {
					cached_tokens: event.cachedTokens,
				}),
				...(event.promptTokens !== undefined && {
					prompt_tokens: event.promptTokens,
				}),
				...(event.completionTokens !== undefined && {
					completion_tokens: event.completionTokens,
				}),
			};
		case "compaction":
			return {
				type: "compaction",
				reason: event.reason,
				tokens_before: event.tokensBefore,
				tokens_after: event.tokensAfter,
			};
		case "error":
			return {
				type: "notice",
				level: "error",
				label: "Error",
				text: event.message,
			};
		case "auto_retry_start":
			return {
				type: "notice",
				level: "warn",
				label: `Retry ${event.attempt}/${event.maxRetries}`,
				text: `${event.error} — retrying in ${formatDelay(event.delayMs)}`,
			};
		case "auto_retry_end":
			return {
				type: "notice",
				level: event.success ? "success" : "warn",
				label: `Retry ${event.attempt}`,
				text: event.success ? "succeeded" : "failed",
			};
		case "run_outcome":
			if (event.status === "completed" && event.source === "heuristic") {
				return null;
			}
			if (
				event.status === "cancelled" &&
				event.summary === STEERING_INTERRUPT_SUMMARY
			) {
				return {
					type: "notice",
					level: "info",
					label: "Steering",
					text: STEERING_INTERRUPT_SUMMARY,
				};
			}
			return {
				type: "notice",
				level:
					event.status === "completed"
						? "success"
						: event.status === "failed"
							? "error"
							: "warn",
				label: `Run ${event.status.replace("_", " ")}`,
				text: event.summary || `Run ended with status: ${event.status}`,
			};
		case "model_select":
			return {
				type: "notice",
				level: "info",
				label: "Model",
				text: event.model,
			};
		case "subagent_start":
			return {
				type: "subagent_lifecycle",
				phase: "start",
				agentId: event.agentId,
				agent: event.agent,
				task: event.task,
				taskIndex: event.taskIndex,
			};
		case "subagent_end":
			return {
				type: "subagent_lifecycle",
				phase: "end",
				agentId: event.agentId,
				agent: event.agent,
				result: event.result,
				isError: event.isError === true,
				turns: event.turns,
				taskIndex: event.taskIndex,
			};
		case "subagent_event": {
			// Forward the child's own tool calls and text/thinking deltas as
			// ordered chunks, carrying its emit-time seq, so the transcript can
			// interleave them in true chronological order (same as the parent
			// agent's own chunk stream) instead of grouping tools separately
			// from text.
			const child = event.event;
			const seq = child.seq ?? 0;
			// Streaming providers emit tool_call_*; execution always emits
			// tool_execution_*. Map both so non-streaming backends still fill
			// the expandable agent activity stream.
			if (child.type === "tool_call_start") {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "tool_start",
					toolCallId: child.toolCallId,
					toolName: child.toolName,
					args: child.args,
					taskIndex: event.taskIndex,
				};
			}
			if (child.type === "tool_execution_start") {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "tool_start",
					toolCallId: child.toolCallId,
					toolName: child.toolName,
					args: JSON.stringify(child.args ?? {}),
					taskIndex: event.taskIndex,
				};
			}
			if (
				child.type === "tool_call_end" ||
				child.type === "tool_execution_end"
			) {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "tool_end",
					toolCallId: child.toolCallId,
					toolName: child.toolName,
					result: child.result,
					isError: child.isError === true,
					taskIndex: event.taskIndex,
				};
			}
			if (child.type === "text_delta") {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "content",
					delta: child.delta,
					taskIndex: event.taskIndex,
				};
			}
			if (child.type === "thinking_delta") {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "thinking",
					delta: child.delta,
					taskIndex: event.taskIndex,
				};
			}
			if (child.type === "error") {
				return {
					type: "notice",
					level: "warn",
					label: `↳ ${event.agentId}`,
					text: child.message,
				};
			}
			return null;
		}
		case "tool_permission_request":
			return {
				type: "notice",
				level: "warn",
				label: "Permission",
				text: `${event.toolName} awaiting approval`,
			};
		case "tool_permission_decision":
			return {
				type: "notice",
				level: event.decision === "deny" ? "warn" : "info",
				label: "Permission",
				text: `${event.toolName}: ${event.decision} (${event.source})`,
			};
		case "budget_exhausted":
			return {
				type: "notice",
				level: "warn",
				label: "Budget",
				text: `token budget exhausted (${event.usedTokens}/${event.limitTokens}) — run stopped.`,
			};
		case "max_iterations":
			return {
				type: "notice",
				level: "warn",
				label: "Stopped",
				text: `reached the ${event.limit}-turn safety limit without finishing (${event.iterations} turns).`,
			};
		default:
			return null;
	}
}

// Humanize a backoff delay for retry notices: "500ms", "1.0s", "4.0s".
function formatDelay(ms: number): string {
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function parseToolArgs(args: string): Record<string, unknown> | undefined {
	try {
		const parsed = JSON.parse(args || "{}");
		return parsed && typeof parsed === "object" ? parsed : undefined;
	} catch (_e: unknown) {
		return undefined;
	}
}
