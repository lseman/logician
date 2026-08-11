// ── Agent event → bridge event shape mapping ──────────────────────────────────

import {
	type AgentEvent,
	STEERING_INTERRUPT_SUMMARY,
} from "@logician/agent-core";
import type { RuntimeEvent } from "./events.ts";

// Translates core AgentEvent variants to their RuntimeEvent equivalent.
// Not the sole producer of RuntimeEvent: AgentCoreBridge also emits
// UI-only synthesized types directly (e.g. "todos", "steered",
// "notice", "memory_update") for signals that have no core AgentEvent
// counterpart. A core event with no case below returns null and is dropped —
// verify it isn't relied on downstream before adding a new AgentEvent variant.
export function mapAgentEvent(event: AgentEvent): RuntimeEvent | null {
	switch (event.type) {
		case "message_start":
		case "text_start":
		case "text_end":
		case "message_end":
		case "agent_settled":
		case "session_delete":
			return null;
		case "text_delta":
			return { type: "token", token: event.delta };
		case "message_update":
			return {
				type: "message_update",
				turnId: event.turnId,
				message: event.message,
			};
		case "thinking_delta":
			return { type: "thinking_token", token: event.delta };
		case "tool_call_start":
			return {
				type: "tool_call_start",
				toolName: event.toolName,
				args: parseToolArgs(event.args) ?? {},
				toolCallId: event.toolCallId,
			};
		case "tool_call_end":
			// Execution completion is emitted separately. Keeping the provider call
			// completion would create a second terminal lifecycle event for one tool.
			return null;
		case "tool_call_delta":
			return {
				type: "tool_call_update",
				delta: event.delta,
				toolCallId: event.toolCallId,
			};
		case "tool_call_id_update":
			return {
				type: "tool_call_id_update",
				previousToolCallId: event.previousToolCallId,
				toolCallId: event.toolCallId,
			};
		case "tool_execution_start":
			return {
				type: "tool_execution_start",
				toolName: event.toolName,
				args: event.args,
				toolCallId: event.toolCallId,
			};
		case "tool_execution_end":
			return {
				type: "tool_execution_end",
				toolName: event.toolName,
				result: event.result,
				isError: event.isError,
				toolCallId: event.toolCallId,
			};
		case "tool_execution_update":
			return {
				type: "tool_execution_update",
				toolName: event.toolName,
				partialResult: event.partialResult,
				toolCallId: event.toolCallId,
			};
		case "repair_nudge":
			return {
				type: "repair_nudge",
				turnId: event.turnId,
				repairStage: event.repairStage,
				toolName: event.toolName,
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
				maxTokens: event.maxTokens,
				compacted: event.compacted,
				...(event.cachedTokens !== undefined && {
					cachedTokens: event.cachedTokens,
				}),
				...(event.promptTokens !== undefined && {
					promptTokens: event.promptTokens,
				}),
				...(event.completionTokens !== undefined && {
					completionTokens: event.completionTokens,
				}),
			};
		case "compaction":
			return {
				type: "compaction",
				reason: event.reason,
				tokensBefore: event.tokensBefore,
				tokensAfter: event.tokensAfter,
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
			};
		case "agent_retry_end":
			return {
				type: "agent_retry_end",
				attempt: event.attempt,
				success: event.success,
				reason: event.reason,
			};
		case "agent_error":
			return {
				type: "agent_error",
				message: event.message,
				phase: event.phase,
				recoverable: event.recoverable,
			};
		case "run_outcome":
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
				type: "model_select",
				model: event.model,
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
					kind: "tool_execution_start",
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
					kind: "tool_execution_start",
					toolCallId: child.toolCallId,
					toolName: child.toolName,
					args: JSON.stringify(child.args ?? {}),
					taskIndex: event.taskIndex,
				};
			}
			if (child.type === "tool_call_id_update") {
				return {
					type: "subagent_chunk",
					agentId: event.agentId,
					seq,
					kind: "tool_call_id_update",
					previousToolCallId: child.previousToolCallId,
					toolCallId: child.toolCallId,
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
					kind: "tool_execution_end",
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
		case "harness_intervention":
			return {
				type: "notice",
				level:
					event.severity === "error"
						? "error"
						: event.severity === "warning"
							? "warn"
							: "info",
				label: `${event.kind}: ${event.action}`,
				text: `${event.evidence.summary} (attempt ${event.attempt}, incident ${event.id})${event.nextAction ? ` Next: ${event.nextAction}` : ""}`,
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
				level:
					event.status === "passed"
						? "success"
						: event.status === "failed"
							? "error"
							: "warn",
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
