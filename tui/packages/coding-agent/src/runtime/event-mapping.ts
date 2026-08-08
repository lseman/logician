// ── Agent event → bridge event shape mapping ──────────────────────────────────

import {
	type AgentEvent,
	STEERING_INTERRUPT_SUMMARY,
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
		case "task_state_update":
			return null; // Handled separately
		case "inference_mode_selected":
			return {
				type: "notice",
				level: "info",
				label: `Auto → ${event.effectiveMode}`,
				text: `${event.reason} (${event.phase})`,
			};
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
		case "agent_retry_start":
			return {
				type: "agent_retry_start",
				attempt: event.attempt,
				maxRetries: event.maxRetries,
				delayMs: event.delayMs,
				error: event.error,
				reason: event.reason,
			} as ParsedBridgeEvent;
		case "agent_retry_end":
			return {
				type: "agent_retry_end",
				attempt: event.attempt,
				success: event.success,
				reason: event.reason,
			} as ParsedBridgeEvent;
		case "agent_error":
			return {
				type: "agent_error",
				message: event.message,
				phase: event.phase,
				recoverable: event.recoverable,
			} as ParsedBridgeEvent;
		case "session_delete":
			return {
				type: "session_delete",
				sessionFile: event.sessionFile,
				sessionId: event.sessionId,
			} as ParsedBridgeEvent;
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
		case "message_end":
			return {
				type: "message_end",
				turnId: event.turnId,
				message: event.message,
			} as ParsedBridgeEvent;
		case "agent_settled":
			return {
				type: "agent_settled",
				nextTurnCount: event.nextTurnCount,
			} as ParsedBridgeEvent;
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
		case "guard_triggered":
			// Route through the generic "notice" event so Transcript.handleNotice
			// attaches it as an inline chunk on the turn that's still in progress
			// — right where the nudge happened — instead of a separate trailing
			// system message.
			return {
				type: "notice",
				level: "warn",
				label: `Guard: ${event.guard}`,
				text: event.message,
			};
		// ── Acceptance / reflection observability ────────────────────────
		case "acceptance_start":
			return {
				type: "notice",
				level: "info",
				label: "Acceptance",
				text: `Starting acceptance level "${event.level}" (${event.criteriaCount} criteria)`,
			};
		case "acceptance_check":
			return {
				type: "notice",
				level: event.status === "failed" ? "warn" : "info",
				label: `Acceptance: ${event.criterionId}`,
				text: `${event.status} (${event.severity})`,
			};
		case "acceptance_verify":
			return {
				type: "notice",
				level: event.result === "failed" ? "warn" : "info",
				label: `Verify: ${event.command.slice(0, 60)}`,
				text: `${event.result}${event.summary ? ` — ${event.summary}` : ""}`,
			};
		case "acceptance_complete":
			return {
				type: "notice",
				level: event.status === "passed" ? "success" : event.status === "failed" ? "error" : "warn",
				label: "Acceptance",
				text: `Status: ${event.status}`,
			};
		case "reflection_start":
			return {
				type: "notice",
				level: "info",
				label: "Reflection",
				text: "Starting reflection on this turn",
			};
		case "reflection_end":
			return {
				type: "notice",
				level: event.assessment === "complete" ? "info" : "warn",
				label: "Reflection",
				text: event.needsMoreWork
					? `Incomplete — ${event.issues.length} issue(s) need work`
					: "Complete",
			};
		// ── Loop detection ───────────────────────────────────────────────
		case "loop_detected":
			return {
				type: "notice",
				level: "warn",
				label: "Loop detected",
				text: `${event.message}${event.attempt !== undefined ? ` (attempt ${event.attempt})` : ""}`,
			};
		case "thinking_loop_detected":
			return {
				type: "notice",
				level: "warn",
				label: "Thinking loop",
				text: `${event.message} — strategy: ${event.strategy} (iteration ${event.iteration})`,
			};
		case "thinking_loop_stats":
			return {
				type: "notice",
				level: "info",
				label: "Thinking stats",
				text: `consecutive ${event.consecutiveThinkingOnly}, total turns ${event.totalThinkingTurns}, tokens ${event.totalThinkingTokens}, meta ${event.metaReasoningHits}`,
			};
		// ── Model / tool lifecycle ───────────────────────────────────────
		case "model_cycle":
			return {
				type: "notice",
				level: "info",
				label: "Model cycle",
				text: `${event.fromModel} → ${event.model}${event.thinkingLevel ? ` (thinking: ${event.thinkingLevel})` : ""}`,
			};
		case "model_change":
			return {
				type: "notice",
				level: "info",
				label: "Model",
				text: `${event.provider}: ${event.modelId}`,
			};
		case "thinking_level_changed":
			return {
				type: "notice",
				level: "info",
				label: "Thinking",
				text: `Level changed to "${event.level}"`,
			};
		case "thinking_level_clamped":
			return {
				type: "notice",
				level: "warn",
				label: "Thinking",
				text: `Level clamped to "${event.level}" — ${event.reason}`,
			};
		case "tools_update":
			return {
				type: "notice",
				level: "info",
				label: "Tools",
				text: `Active tools: ${event.toolNames.join(", ")}`,
			};
		case "active_tools_change":
			return {
				type: "notice",
				level: "info",
				label: "Active tools",
				text: event.activeToolNames.join(", "),
			};
		case "abort":
			return {
				type: "notice",
				level: "info",
				label: "Aborted",
				text: `Cleared ${event.clearedSteering.length} steering, ${event.clearedFollowUp.length} follow-up, ${event.clearedNextTurn.length} next-turn messages`,
			};
		case "task_failed":
			return {
				type: "notice",
				level: "error",
				label: "Task failed",
				text: `${event.reason} (iteration ${event.iteration})${event.lastContent ? ` — ${event.lastContent.slice(0, 100)}` : ""}`,
			};
		default:
			return null;
	}
}

function parseToolArgs(args: string): Record<string, unknown> | undefined {
	try {
		const parsed = JSON.parse(args || "{}");
		return parsed && typeof parsed === "object" ? parsed : undefined;
	} catch (_e: unknown) {
		return undefined;
	}
}
