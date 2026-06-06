// ── Transcript management ──────────────────────────────────────────────────────
// Maintains the full conversation history with streaming state

import type {
    ParsedBridgeEvent,
    TurnEndEvent,
    ToolStartEvent,
    ToolUpdateEvent,
    ToolEndEvent,
} from "./events.ts";

export type ThinkingDisplayStyle = "collapsed" | "summary" | "expanded";

export interface ToolExecution {
    tool: string;
    tool_name: string;
    tool_call_id?: string;
    args?: Record<string, unknown>;
    result?: string;
    partialResult?: string;
    isError: boolean;
}

export interface AssistantMessage {
    type: "assistant";
    thinkingBlocks: string[];
    content: string;
    tools: ToolExecution[];
    isComplete: boolean;
}

export interface UserMessage {
    type: "user";
    content: string;
}

export interface SystemMessage {
    type: "system";
    content: string;
}

export type Message = UserMessage | AssistantMessage | SystemMessage;

export interface Turn {
    id: string;
    userMessage: UserMessage;
    assistantMessage: AssistantMessage | null;
    isComplete: boolean;
}

export interface SessionState {
    turns: Turn[];
    currentTurnId: string | null;
    thinkingDisplayMode: ThinkingDisplayStyle;
    thinkingLevel: string;
    cacheEnabled: boolean;
}

const DEFAULT_STATE: SessionState = {
    turns: [],
    currentTurnId: null,
    thinkingDisplayMode: "expanded",
    thinkingLevel: "medium",
    cacheEnabled: true,
};

export class Transcript {
    private state: SessionState = { ...DEFAULT_STATE };
    private listeners: Array<() => void> = [];

    // ── Event handling ─────────────────────────────────────────────────────

    handleEvent(event: ParsedBridgeEvent): void {
        switch (event.type) {
            case "turn_start":
                this.handleTurnStart(
                    event as TurnEndEvent & { type: "turn_start" },
                );
                break;
            case "token":
                this.handleToken(String(event.token || ""));
                break;
            case "thinking_token":
                this.handleThinkingToken(String(event.token || ""));
                break;
            case "tool_start":
            case "tool_execution_start":
                this.handleToolStart(event as ToolStartEvent);
                break;
            case "tool_execution_update":
                this.handleToolUpdate(event as ToolUpdateEvent);
                break;
            case "tool_end":
            case "tool_execution_end": {
                const toolEvent = event as ToolEndEvent;
                this.handleToolEnd(toolEvent);
                break;
            }
            case "turn_end":
                this.handleTurnEnd(event as TurnEndEvent);
                break;
        }
        this.notify();
    }

    private handleTurnStart(event: { turn_id: string }): void {
        // The user-submitted turn was created locally with a synthetic id (addTurn).
        // The bridge assigns its own turn_id on turn_start — adopt it onto the most
        // recent turn that has no assistant response yet so streamed tokens land on
        // the right turn. Fall back to creating a new turn if none is pending.
        if (!event.turn_id) return;
        // Adopt the bridge's turn_id onto the most recent turn that has no assistant
        // response yet so streamed tokens land on the right turn. If there is no
        // pending turn (e.g. duplicate turn_start), just point currentTurnId at the
        // latest turn — never fabricate an empty-user turn.
        const pending = [...this.state.turns]
            .reverse()
            .find((t) => !t.isComplete && t.assistantMessage === null);
        if (pending) {
            pending.id = event.turn_id;
            this.state.currentTurnId = event.turn_id;
        } else if (this.state.turns.length > 0) {
            this.state.currentTurnId =
                this.state.turns[this.state.turns.length - 1].id;
        }
    }

    private handleToken(token: string): void {
        const turn = this.getCurrentTurn();
        if (!turn) return;

        if (!turn.assistantMessage) {
            turn.assistantMessage = {
                type: "assistant",
                thinkingBlocks: [],
                content: token,
                tools: [],
                isComplete: false,
            };
        } else {
            turn.assistantMessage.content += token;
        }
    }

    private handleThinkingToken(token: string): void {
        const turn = this.getCurrentTurn();
        if (!turn) return;

        if (!turn.assistantMessage) {
            turn.assistantMessage = {
                type: "assistant",
                thinkingBlocks: [token],
                content: "",
                tools: [],
                isComplete: false,
            };
        } else {
            const blocks = turn.assistantMessage.thinkingBlocks;
            const last = blocks[blocks.length - 1];
            if (last === undefined) {
                blocks.push(token);
            } else {
                blocks[blocks.length - 1] = last + token;
            }
        }
    }

    private handleToolStart(event: ToolStartEvent): void {
        const turn = this.getCurrentTurn();
        if (!turn) return;

        if (!turn.assistantMessage) {
            turn.assistantMessage = {
                type: "assistant",
                thinkingBlocks: [],
                content: "",
                tools: [],
                isComplete: false,
            };
        }

        turn.assistantMessage.tools.push({
            tool: event.tool,
            tool_name: event.tool_name,
            tool_call_id: event.tool_call_id,
            args: event.tool_args as Record<string, unknown> | undefined,
            result: undefined,
            partialResult: undefined,
            isError: false,
        });
    }

    private handleToolUpdate(event: ToolUpdateEvent): void {
        const turn = this.getCurrentTurn();
        if (!turn?.assistantMessage?.tools.length) return;
        const tool = this.findToolForEvent(turn.assistantMessage.tools, event);
        if (!tool) return;
        if (event.partial_result !== undefined) {
            tool.partialResult = String(event.partial_result);
        }
    }

    private handleToolEnd(event: ToolEndEvent): void {
        const turn = this.getCurrentTurn();
        if (!turn) return;

        const assistant = turn.assistantMessage;
        if (!assistant || assistant.tools.length === 0) return;

        const tool =
            this.findToolForEvent(assistant.tools, event) ||
            assistant.tools[assistant.tools.length - 1];
        if (event.result !== undefined) {
            tool.result = String(event.result);
            tool.partialResult = undefined;
        }
        const isError = (event as unknown as Record<string, unknown>).is_error;
        if (isError !== undefined) {
            tool.isError = Boolean(isError);
        }
    }

    private findToolForEvent(
        tools: ToolExecution[],
        event: { tool_call_id?: string },
    ): ToolExecution | undefined {
        return event.tool_call_id
            ? [...tools]
                  .reverse()
                  .find((tool) => tool.tool_call_id === event.tool_call_id)
            : undefined;
    }

    private handleTurnEnd(event: TurnEndEvent): void {
        const turn = this.getTurnById(event.turn_id);
        if (turn) {
            if (turn.assistantMessage) {
                turn.assistantMessage.isComplete = true;
            }
            turn.isComplete = true;
        }
    }

    // ── Turn management ──────────────────────────────────────────────────

    addTurn(userContent: string): Turn {
        const turn: Turn = {
            id: `turn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
            userMessage: { type: "user", content: userContent },
            assistantMessage: null,
            isComplete: false,
        };
        this.state.turns.push(turn);
        this.state.currentTurnId = turn.id;
        this.notify();
        return turn;
    }

    private getCurrentTurn(): Turn | undefined {
        if (!this.state.currentTurnId) return undefined;
        return this.state.turns.find((t) => t.id === this.state.currentTurnId);
    }

    private getTurnById(id: string): Turn | undefined {
        return this.state.turns.find((t) => t.id === id);
    }

    clear(): void {
        this.state = { ...DEFAULT_STATE };
        this.notify();
    }

    addSystemMessage(content: string): void {
        this.state.turns.push({
            id: `sys_${Date.now()}`,
            userMessage: {
                type: "user" as const,
                content: `[System] ${content}`,
            },
            assistantMessage: null,
            isComplete: true,
        });
        this.notify();
    }

    // ── Thinking display ─────────────────────────────────────────────────

    getThinkingDisplayMode(): ThinkingDisplayStyle {
        return this.state.thinkingDisplayMode;
    }

    cycleThinkingDisplayMode(): void {
        const modes: ThinkingDisplayStyle[] = [
            "collapsed",
            "summary",
            "expanded",
        ];
        const current = this.state.thinkingDisplayMode;
        const next = modes[(modes.indexOf(current) + 1) % modes.length];
        this.state.thinkingDisplayMode = next;
        this.notify();
    }

    setThinkingDisplayMode(mode: ThinkingDisplayStyle): void {
        this.state.thinkingDisplayMode = mode;
        this.notify();
    }

    setThinkingLevel(level: string): void {
        this.state.thinkingLevel = level;
        this.notify();
    }

    getThinkingLevel(): string {
        return this.state.thinkingLevel;
    }

    setCacheEnabled(enabled: boolean): void {
        this.state.cacheEnabled = enabled;
        this.notify();
    }

    getCacheEnabled(): boolean {
        return this.state.cacheEnabled;
    }

    // ── Accessors ────────────────────────────────────────────────────────

    getTurns(): Turn[] {
        return this.state.turns;
    }

    getMessageCount(): number {
        return this.state.turns.reduce(
            (count, turn) => count + 1 + (turn.assistantMessage ? 1 : 0),
            0,
        );
    }

    getStreamingContent(): string | null {
        const turn = this.getCurrentTurn();
        if (!turn?.assistantMessage) return null;
        return turn.assistantMessage.content;
    }

    getStreamingThinking(): string[] {
        const turn = this.getCurrentTurn();
        if (!turn?.assistantMessage) return [];
        return turn.assistantMessage.thinkingBlocks;
    }

    hasStreamingContent(): boolean {
        const content = this.getStreamingContent();
        return content !== null && content.length > 0;
    }

    hasStreamingThinking(): boolean {
        const thinking = this.getStreamingThinking();
        return thinking.length > 0 && thinking.some((t) => t.trim().length > 0);
    }

    // ── Rendering helpers ────────────────────────────────────────────────

    getAssistantThinking(turn: Turn): string | null {
        const assistant = turn.assistantMessage;
        if (!assistant) return null;
        const thinking = assistant.thinkingBlocks.filter(
            (t) => t.trim().length > 0,
        );
        if (thinking.length === 0) return null;
        return thinking.join("\n\n");
    }

    getAssistantContent(turn: Turn): string | null {
        const assistant = turn.assistantMessage;
        if (!assistant) return null;
        return assistant.content.length > 0 ? assistant.content : null;
    }

    getAssistantTools(turn: Turn): ToolExecution[] {
        const assistant = turn.assistantMessage;
        if (!assistant) return [];
        return assistant.tools;
    }

    // ── Listener management ────────────────────────────────────────────

    onChange(callback: () => void): () => void {
        this.listeners.push(callback);
        return () => {
            this.listeners = this.listeners.filter((cb) => cb !== callback);
        };
    }

    private notify(): void {
        for (const cb of this.listeners) {
            try {
                cb();
            } catch {
                // Don't crash on bad listeners
            }
        }
    }
}
