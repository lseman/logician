// ── Bridge event types ─────────────────────────────────────────────────────────
// Mirrors logician_bridge.py's _emit() protocol

export type BridgeEventType =
	// Agent lifecycle
	| "agent_start"
	| "agent_end"
	// Turn lifecycle
	| "turn_start"
	| "turn_end"
	// Token streaming
	| "token"
	| "thinking_token"
	// Granular text streaming boundaries
	| "text_start"
	| "text_end"
	// Message update (full partial assistant message)
	| "message_update"
	// Message start (before assistant response, for steering detection)
	| "message_start"
	// Queue update (steering + follow-up)
	| "queue_update"
	// Tool execution
	| "tool_start"
	| "tool_end"
	| "tool_execution_start"
	| "tool_execution_update"
	| "tool_execution_end"
	// Steering
	| "queue_update"
	// Guardrail/repair
	| "guardrail_nudge"
	| "repair_nudge"
	// Classification
	| "classified"
	// UI state
	| "phase"
	| "decision"
	| "context_update"
	| "compaction"
	// Agent question / interactive prompt
	| "question_request"
	// Media
	| "image";

export interface BridgeEvent {
	type: BridgeEventType;
	[key: string]: unknown;
}

// ── Parsed event shapes ───────────────────────────────────────────────────────

export interface TokenEvent {
	type: "token";
	token: string;
}

export interface ThinkingTokenEvent {
	type: "thinking_token";
	token: string;
}

export interface MessageStartEvent {
	type: "message_start";
	turnId: string;
	role: string;
}

export interface TextStartEvent {
	type: "text_start";
	turnId: string;
}

export interface TextEndEvent {
	type: "text_end";
	turnId: string;
}

export interface MessageUpdateEvent {
	type: "message_update";
	turnId: string;
	message: {
		role: string;
		content: string | null;
		tool_calls?: Array<{
			id: string;
			name: string;
			arguments: string;
		}>;
	};
}

export interface TurnStartEvent {
	type: "turn_start";
	turn_id: string;
}

export interface TurnEndEvent {
	type: "turn_end";
	turn_id: string;
	message: string;
}

export interface ToolStartEvent {
	type: "tool_start" | "tool_execution_start";
	tool: string;
	tool_name: string;
	tool_args?: Record<string, unknown>;
	turn_id?: string;
	tool_call_id?: string;
}

export interface ToolEndEvent {
	type: "tool_end" | "tool_execution_end";
	tool: string;
	tool_name: string;
	result?: string;
	is_error?: boolean;
	turn_id?: string;
	tool_call_id?: string;
	// Structured metadata the tool returned alongside its text result.
	details?: Record<string, unknown>;
}

export interface ToolUpdateEvent {
	type: "tool_execution_update";
	tool: string;
	tool_name: string;
	partial_result?: string;
	turn_id?: string;
	tool_call_id?: string;
}

export interface PhaseEvent {
	type: "phase";
	state: string;
	note?: string;
}

export interface DecisionEvent {
	type: "decision";
	stage?: string;
	model?: string;
	turn_id?: string;
}

export interface ContextUpdateEvent {
	type: "context_update";
	tokens: number;
	max_tokens?: number;
	compacted?: boolean;
}

export interface CompactionEvent {
	type: "compaction";
	reason: string;
	tokens_before: number;
	tokens_after: number;
}

export interface GuardrailEvent {
	type: "guardrail_nudge";
	turn_id?: string;
	guard_name?: string;
	nudge?: string;
}

export interface RepairEvent {
	type: "repair_nudge";
	turn_id?: string;
	repair_stage?: string;
	attempt?: number;
	tool_name?: string;
	error_type?: string;
	message?: string;
}

export interface ClassifiedEvent {
	type: "classified";
	turn_id?: string;
	intent?: string;
	domain_groups?: string[];
}

export interface ImageEvent {
	type: "image";
	tool?: string;
	path?: string;
	source?: string;
}

export interface TodosEvent {
	type: "todos";
	todos: Array<{
		content: string;
		status: "pending" | "in_progress" | "completed";
	}>;
}

export interface SteeredEvent {
	type: "steered";
	message: string;
}

export interface QueueUpdateEvent {
	type: "queue_update";
	steering: string[];
	followUp: string[];
	nextTurn?: string[];
}

export interface ModelSelectEvent {
	type: "model_select";
	model: string;
}

// A standalone status line (retry / error / model / stopped) rendered as its
// own iconed, coloured chunk rather than folded into assistant prose.
export interface NoticeEvent {
	type: "notice";
	level: "info" | "warn" | "error" | "success";
	label: string;
	text: string;
}

// A tool call is paused waiting for the user's allow/deny decision. The UI
// answers via bridge.respondToPermission(tool_call_id, decision).
export interface PermissionRequestEvent {
	type: "permission_request";
	tool_name: string;
	tool_call_id: string;
	args: Record<string, unknown>;
}

export interface QuestionRequestEvent {
	type: "question_request";
	question_id: string;
	question: string;
	choices: Array<{ value: string; label: string }>;
}

// Fired after every completed turn — a safe rewind point exists and the
// conversation has been persisted. Use to show autosave indicators.
export interface SavePointEvent {
	type: "save_point";
}

export type ParsedBridgeEvent =
	| TokenEvent
	| ThinkingTokenEvent
	| TextStartEvent
	| TextEndEvent
	| MessageUpdateEvent
	| TurnStartEvent
	| TurnEndEvent
	| ToolStartEvent
	| ToolUpdateEvent
	| ToolEndEvent
	| MessageStartEvent
	| QueueUpdateEvent
	| PhaseEvent
	| DecisionEvent
	| ContextUpdateEvent
	| CompactionEvent
	| GuardrailEvent
	| RepairEvent
	| ClassifiedEvent
	| ImageEvent
	| TodosEvent
	| SteeredEvent
	| ModelSelectEvent
	| NoticeEvent
	| PermissionRequestEvent
	| QuestionRequestEvent
	| SavePointEvent;

// ── Bridge commands (TUI → bridge) ────────────────────────────────────────────

export type BridgeCommand =
	| { type: "message"; message: string }
	| { type: "slash"; command: string; args: string }
	| { type: "cancel" }
	| { type: "thinking"; level: string }
	| { type: "cache"; action: string }
	| { type: "reset" }
	| { type: "config"; key: string; value: unknown };
