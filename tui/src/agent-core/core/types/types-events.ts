// ── Event types ───────────────────────────────────────────────────────────

import type { MessageRole, StopReason } from "./types-messages.ts";

/**
 * Envelope metadata stamped onto every event at the emit boundary: a
 * monotonic per-loop sequence number and a wall-clock timestamp.
 */
export interface AgentEventEnvelope {
	seq?: number;
	ts?: number;
}

export type AgentEventBody =
	| { type: "agent_start" }
	| { type: "agent_end"; messages?: Message[] }
	| { type: "turn_start"; turnId: string }
	| {
			type: "turn_end";
			turnId: string;
			stopReason?: StopReason;
			message?: Message;
			toolResults?: Message[];
	  }
	| { type: "message_start"; turnId: string; role: MessageRole }
	| { type: "text_start"; turnId: string }
	| { type: "text_delta"; turnId: string; delta: string }
	| { type: "text_end"; turnId: string }
	| { type: "message_update"; turnId: string; message: Message }
	| { type: "message_end"; turnId: string }
	| {
			type: "context_update";
			tokens: number;
			maxTokens?: number;
			compacted?: boolean;
	  }
	| {
			type: "compaction";
			reason: "context_full" | "manual";
			tokensBefore: number;
			tokensAfter: number;
	  }
	| { type: "thinking_delta"; turnId?: string; delta: string }
	| {
			type: "tool_call_start";
			toolName: string;
			toolCallId: string;
			args: string;
	  }
	| {
			type: "tool_call_delta";
			toolCallId: string;
			delta: string;
	  }
	| {
			type: "tool_call_end";
			toolName: string;
			toolCallId: string;
			result: string;
			isError?: boolean;
			details?: Record<string, unknown>;
	  }
	| {
			type: "tool_call_update";
			toolName: string;
			toolCallId: string;
			partialResult: string;
	  }
	| {
			type: "repair_nudge";
			turnId?: string;
			repairStage: string;
			toolName?: string;
			message: string;
	  }
	| { type: "phase"; phase: "thinking" | "tool" | "idle" }
	| {
			type: "auto_retry_start";
			attempt: number;
			maxRetries: number;
			delayMs: number;
			error: string;
	  }
	| { type: "auto_retry_end"; attempt: number; success: boolean }
	| { type: "model_select"; model: string; index: number }
	| { type: "max_iterations"; iterations: number; limit: number }
	| {
			type: "loop_detected";
			message: string;
			attempt?: number;
	  }
	| { type: "subagent_start"; agentId: string; agent: string; task: string }
	| { type: "subagent_event"; agentId: string; event: AgentEvent }
	| {
			type: "subagent_end";
			agentId: string;
			agent: string;
			result: string;
			isError?: boolean;
			turns?: number;
	  }
	| {
			type: "tool_permission_request";
			toolName: string;
			toolCallId: string;
			args: string;
	  }
	| {
			type: "tool_permission_decision";
			toolName: string;
			toolCallId: string;
			decision: "allow" | "deny" | "always";
			source: "rule" | "mode" | "user" | "hook";
	  }
	| { type: "budget_exhausted"; usedTokens: number; limitTokens: number }
	| { type: "error"; message: string; error?: unknown }
	| { type: "model_update"; model: string }
	| { type: "tools_update"; toolNames: string[] };

export type AgentEvent = AgentEventBody & AgentEventEnvelope;
export type EventHandler = (event: AgentEvent) => void;

// Re-export Message for the event types
import type { Message } from "./types-messages.ts";
