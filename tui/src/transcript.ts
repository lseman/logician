// ── Transcript management ──────────────────────────────────────────────────────
// Interleaved chunk model — chunks ordered by arrival time.
// Rendering follows chronological order: thinking → response → tool → thinking …

import type {
	ParsedBridgeEvent,
	ToolEndEvent,
	ToolStartEvent,
	ToolUpdateEvent,
	TurnEndEvent,
} from "./events.ts";

export type ThinkingDisplayStyle = "collapsed" | "summary" | "expanded";

// ── Tool execution (used by tool chunks) ──────────────────────────────────────

export interface ToolExecution {
	tool: string;
	tool_name: string;
	tool_call_id?: string;
	args?: Record<string, unknown>;
	result?: string;
	partialResult?: string;
	isError: boolean;
	isComplete: boolean;
}

// ── Chunk model ───────────────────────────────────────────────────────────────
// Ordered, typed chunks replace the old clustered [thinking[], content, tools[]].

export interface AssistantChunk {
	seq: number; // insertion sequence — defines display order
	type: "thinking" | "content" | "tool";
	// per-type fields
	contentText?: string; // for 'thinking' and 'content'
	tool?: ToolExecution; // for 'tool'
	isComplete: boolean; // true when chunk is finalised
}

// ── Message types ─────────────────────────────────────────────────────────────

export interface AssistantMessage {
	type: "assistant";
	chunks: AssistantChunk[]; // ordered interleaved chunks
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
				this.handleTurnStart(event as TurnEndEvent & { type: "turn_start" });
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
		if (!event.turn_id) return;
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

	// ── Chunk helpers ──────────────────────────────────────────────────────

	/** Get the last chunk of a given type that is still incomplete (streaming). */
	private lastStreamingChunkOfType(
		type: "thinking" | "content" | "tool",
		chunks: AssistantChunk[],
	): AssistantChunk | undefined {
		for (let i = chunks.length - 1; i >= 0; i--) {
			if (chunks[i].type === type && !chunks[i].isComplete) {
				return chunks[i];
			}
		}
		return undefined;
	}

	/**
	 * Find an incomplete tool chunk that a repeated `tool_start` should reuse.
	 * Matches by tool_call_id when both ids are real; falls back to the last
	 * incomplete tool chunk of the same name (covers placeholder ids emitted
	 * during streaming, e.g. `tool_0`).
	 */
	private findReusableToolChunk(
		chunks: AssistantChunk[],
		event: ToolStartEvent,
	): AssistantChunk | undefined {
		for (let i = chunks.length - 1; i >= 0; i--) {
			const c = chunks[i];
			if (c.type !== "tool" || c.isComplete || !c.tool) continue;
			if (event.tool_call_id && c.tool.tool_call_id === event.tool_call_id) {
				return c;
			}
			if (c.tool.tool_name === event.tool_name) return c;
		}
		return undefined;
	}

	private ensureAssistant(turn: Turn): AssistantMessage {
		if (!turn.assistantMessage) {
			turn.assistantMessage = {
				type: "assistant",
				chunks: [],
				isComplete: false,
			};
		}
		return turn.assistantMessage;
	}

	// ── Chunk transition helpers ─────────────────────────────────────────
	// When a new chunk type starts, close any open chunk of a different type.

	private closeStreamingOfType(
		type: "thinking" | "content" | "tool",
		chunks: AssistantChunk[],
	): void {
		for (let i = 0; i < chunks.length; i++) {
			if (chunks[i].type === type && !chunks[i].isComplete) {
				chunks[i].isComplete = true;
			}
		}
	}

	// ── Token handlers ─────────────────────────────────────────────────────

	private handleToken(token: string): void {
		const turn = this.getCurrentTurn();
		if (!turn) return;

		const msg = this.ensureAssistant(turn);
		// Close any open thinking chunk before starting content
		this.closeStreamingOfType("thinking", msg.chunks);

		const lastContent = this.lastStreamingChunkOfType("content", msg.chunks);

		if (lastContent) {
			// Append to existing streaming content chunk
			// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
			lastContent.contentText! += token;
		} else {
			// No streaming content chunk — create a new one
			msg.chunks.push({
				seq: msg.chunks.length,
				type: "content",
				contentText: token,
				isComplete: false,
			});
		}
	}

	private handleThinkingToken(token: string): void {
		const turn = this.getCurrentTurn();
		if (!turn) return;

		const msg = this.ensureAssistant(turn);
		// Close any open content chunk before starting thinking
		this.closeStreamingOfType("content", msg.chunks);

		const lastThinking = this.lastStreamingChunkOfType("thinking", msg.chunks);

		if (lastThinking) {
			// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
			lastThinking.contentText! += token;
		} else {
			msg.chunks.push({
				seq: msg.chunks.length,
				type: "thinking",
				contentText: token,
				isComplete: false,
			});
		}
	}

	private handleToolStart(event: ToolStartEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn) return;

		const msg = this.ensureAssistant(turn);
		// Close any open thinking or content chunk before starting tool
		this.closeStreamingOfType("thinking", msg.chunks);
		this.closeStreamingOfType("content", msg.chunks);

		// A tool can emit `start` twice: once while the model streams the call
		// (placeholder id like `tool_0`) and again at execution time (real id).
		// Reuse the existing streaming chunk instead of pushing a duplicate,
		// otherwise the first chunk is left stuck on "streaming" while a second
		// chunk shows "done".
		const existing = this.findReusableToolChunk(msg.chunks, event);
		if (existing?.tool) {
			existing.tool.tool = event.tool;
			existing.tool.tool_name = event.tool_name;
			existing.tool.tool_call_id = event.tool_call_id;
			if (event.tool_args !== undefined) {
				existing.tool.args = event.tool_args as
					| Record<string, unknown>
					| undefined;
			}
			return;
		}

		msg.chunks.push({
			seq: msg.chunks.length,
			type: "tool",
			tool: {
				tool: event.tool,
				tool_name: event.tool_name,
				tool_call_id: event.tool_call_id,
				args: event.tool_args as Record<string, unknown> | undefined,
				result: undefined,
				partialResult: undefined,
				isError: false,
				isComplete: false,
			},
			isComplete: false,
		});
	}

	private handleToolUpdate(event: ToolUpdateEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		// Find the streaming tool chunk (last incomplete tool chunk)
		const toolChunk = this.lastStreamingChunkOfType(
			"tool",
			turn.assistantMessage.chunks,
		);
		if (!toolChunk?.tool) return;

		if (event.partial_result !== undefined) {
			// Append delta fragments (not overwrite) so partialResult
			// accumulates the full JSON args as the model streams them.
			toolChunk.tool.partialResult =
				(toolChunk.tool.partialResult || "") + String(event.partial_result);
		}
	}

	private handleToolEnd(event: ToolEndEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		const assistant = turn.assistantMessage;

		// Find the tool chunk matching this event
		let toolChunk: AssistantChunk | undefined;
		if (event.tool_call_id) {
			toolChunk = assistant.chunks
				.slice()
				.reverse()
				.find(
					(c) =>
						c.type === "tool" && c.tool?.tool_call_id === event.tool_call_id,
				);
		}
		if (!toolChunk) {
			// Fallback: only mark the last *incomplete* tool chunk as done.
			// Grabbing any tool chunk (including completed ones) could
			// silently mark the wrong chunk or create stale display state.
			toolChunk = assistant.chunks
				.slice()
				.reverse()
				.find((c) => c.type === "tool" && !c.isComplete);
		}
		if (!toolChunk?.tool) return;

		// Mark the tool as finished
		const tool = toolChunk.tool;
		tool.result = event.result !== undefined ? String(event.result) : "";
		tool.partialResult = undefined;
		const isError = (event as unknown as Record<string, unknown>).is_error;
		if (isError !== undefined) {
			tool.isError = Boolean(isError);
		}
		tool.isComplete = true;
		toolChunk.isComplete = true;
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
		const modes: ThinkingDisplayStyle[] = ["collapsed", "summary", "expanded"];
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

	// ── Chunk accessors (for thinking panel & display) ───────────────────

	/** Get thinking chunks from the current (streaming) turn. */
	getThinkingChunks(): AssistantChunk[] {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return [];
		return turn.assistantMessage.chunks.filter((c) => c.type === "thinking");
	}

	/** Get thinking chunks from a completed turn. */
	getTurnThinkingChunks(turn: Turn): AssistantChunk[] {
		const assistant = turn.assistantMessage;
		if (!assistant) return [];
		return assistant.chunks.filter((c) => c.type === "thinking");
	}

	// ── Backward-compatible getters ──────────────────────────────────────

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
		const text = turn.assistantMessage.chunks
			.filter((c) => c.type === "content" && !c.isComplete)
			.map((c) => c.contentText)
			.join("");
		return text || null;
	}

	/** Legacy: thinking blocks — derived from chunks for compatibility. */
	getStreamingThinking(): string[] {
		const chunks = this.getThinkingChunks();
		return chunks.map((c) => c.contentText || "").filter(Boolean);
	}

	hasStreamingContent(): boolean {
		const content = this.getStreamingContent();
		return content !== null && content.length > 0;
	}

	hasStreamingThinking(): boolean {
		const chunks = this.getThinkingChunks();
		return chunks.some((t) => (t.contentText || "").trim().length > 0);
	}

	getAssistantThinking(turn: Turn): string | null {
		const assistant = turn.assistantMessage;
		if (!assistant) return null;
		const thinking = assistant.chunks
			.filter((c) => c.type === "thinking")
			.map((c) => c.contentText || "")
			.filter(Boolean);
		if (thinking.length === 0) return null;
		return thinking.join("\n\n");
	}

	getAssistantContent(turn: Turn): string | null {
		const assistant = turn.assistantMessage;
		if (!assistant) return null;
		const text = assistant.chunks
			.filter((c) => c.type === "content")
			.map((c) => c.contentText)
			.join("");
		return text.length > 0 ? text : null;
	}

	getAssistantTools(turn: Turn): ToolExecution[] {
		const assistant = turn.assistantMessage;
		if (!assistant) return [];
		return (
			assistant.chunks
				.filter((c) => c.type === "tool" && c.tool)
				// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
				.map((c) => c.tool!)
		);
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
