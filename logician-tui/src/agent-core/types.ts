// ── Core types ───────────────────────────────────────────────────────────────────
// Mirrors Python AgentEvent/AgentLoop shapes for clean TUI integration.

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface Message {
    role: MessageRole;
    content: string | null;
    tool_call_id?: string;
    tool_calls?: ToolCall[];
    name?: string;
    timestamp?: number;
}

export type AgentEvent =
    | { type: "agent_start" }
    | { type: "agent_end" }
    | { type: "turn_start"; turnId: string }
    | { type: "turn_end"; turnId: string }
    | { type: "message_start"; turnId: string; role: MessageRole }
    | { type: "message_delta"; turnId: string; delta: string }
    | { type: "message_end"; turnId: string }
    | { type: "thinking_start" }
    | { type: "thinking_delta"; delta: string }
    | { type: "thinking_end" }
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
    | {
          type: "tool_call_start";
          toolName: string;
          toolCallId: string;
          args: string;
      }
    | {
          type: "tool_call_end";
          toolName: string;
          toolCallId: string;
          result: string;
          isError?: boolean;
      }
    | {
          type: "tool_call_update";
          toolName: string;
          toolCallId: string;
          partialResult: string;
      }
    | { type: "phase"; phase: "streaming" | "thinking" | "tool" | "idle" }
    | { type: "error"; message: string; error?: unknown };

export type EventHandler = (event: AgentEvent) => void;

// ── Agent-loop contract hooks ──────────────────────────────────────────────
// First-class extension points mirroring Pi's richer loop contract. Each is an
// optional async callback on AgentConfig. Returning undefined = no change.

export interface BeforeToolCallContext {
    toolCall: ToolCall;
    args: Record<string, unknown>;
    iteration: number;
}

// Return `{ content }` to short-circuit execution (tool is NOT run; content is
// used as the result). Return `{ args }` to rewrite the tool input before it
// runs. Return both to short-circuit with a rewritten record (content wins).
export interface BeforeToolCallResult {
    content?: string;
    isError?: boolean;
    args?: Record<string, unknown>;
}

export interface AfterToolCallContext {
    toolCall: ToolCall;
    args: Record<string, unknown>;
    result: string;
    isError: boolean;
    iteration: number;
}

// Return `{ content }` and/or `{ isError }` to rewrite the recorded tool result.
export interface AfterToolCallResult {
    content?: string;
    isError?: boolean;
}

export interface PrepareNextTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
}

// Return rewritten messages to replace the working history before the next
// model call (compaction, steering injection, message rewriting).
export interface PrepareNextTurnResult {
    messages: Message[];
}

export interface ShouldStopAfterTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
}

export interface AgentLoopHooks {
    beforeToolCall?: (
        ctx: BeforeToolCallContext,
    ) => Promise<BeforeToolCallResult | undefined> | BeforeToolCallResult | undefined;
    afterToolCall?: (
        ctx: AfterToolCallContext,
    ) => Promise<AfterToolCallResult | undefined> | AfterToolCallResult | undefined;
    prepareNextTurn?: (
        ctx: PrepareNextTurnContext,
    ) =>
        | Promise<PrepareNextTurnResult | undefined>
        | PrepareNextTurnResult
        | undefined;
    shouldStopAfterTurn?: (
        ctx: ShouldStopAfterTurnContext,
    ) => Promise<boolean | undefined> | boolean | undefined;
}

export interface ToolCall {
    id: string;
    name: string;
    arguments: string;
}

export interface Tool {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    execute: (
        args: Record<string, unknown>,
        ctx: ToolContext,
    ) => Promise<string>;
}

export interface ToolContext {
    cwd?: string;
    maxOutputChars?: number;
    signal?: AbortSignal;
    onUpdate?: (partialResult: string) => void;
}

export interface AgentConfig {
    baseUrl: string;
    model: string;
    cwd?: string;
    temperature?: number;
    maxTokens?: number;
    chatTemplate?: string;
    stop?: string[];
    maxIterations?: number;
    contextWindowTokens?: number;
    systemPrompt?: string;
    tools?: Tool[];
    onEvent?: EventHandler;
    runtimeHooksEnabled?: boolean;
    hookSessionId?: string;
    hookTranscriptPath?: string;
    hooks?: AgentLoopHooks;
}
