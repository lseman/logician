// ── Transcript management ──────────────────────────────────────────────────────
// Interleaved chunk model — chunks ordered by arrival time.
// Rendering follows chronological order: thinking → response → tool → thinking …

import type {
	ParsedBridgeEvent,
	MessageUpdateEvent,
	SubagentChunkEvent,
	ToolEndEvent,
	ToolStartEvent,
	ToolUpdateEvent,
	TurnEndEvent,
} from "../runtime/events.ts";

export type ThinkingDisplayStyle = "collapsed" | "summary" | "expanded";

// ── Tool execution (used by tool chunks) ──────────────────────────────────────

export interface ToolExecution {
	tool: string;
	tool_name: string;
	tool_call_id?: string;
	args?: Record<string, unknown>;
	result?: string;
	partialResult?: string;
	/** Human-readable progress emitted while the tool executes. */
	streamOutput?: string;
	details?: Record<string, unknown>;
	isError: boolean;
	isComplete: boolean;
	startedAt?: number;
	durationMs?: number;
}

// ── Chunk model ───────────────────────────────────────────────────────────────
// Ordered, typed chunks replace the old clustered [thinking[], content, tools[]].

export type NoticeLevel = "info" | "warn" | "error" | "success";

// ── Subagent child chunks ───────────────────────────────────────────────────
// Mirrors AssistantChunk's interleaving model for a spawn_agent/spawn_agents
// child: thinking/content deltas and tool calls stored in true chronological
// order (by the child's own emit-time seq), instead of the two disjoint
// buckets (tool list + accumulated text string) used before.

export interface ChildToolCall {
	agentId: string;
	toolCallId: string;
	toolName: string;
	args: string;
	status?: "running" | "completed" | "failed";
	isError?: boolean;
	resultPreview?: string;
}

export interface ChildChunk {
	seq: number;
	agentId: string;
	type: "thinking" | "content" | "tool";
	contentText?: string; // for 'thinking' and 'content'
	tool?: ChildToolCall; // for 'tool'
	isComplete: boolean;
}

export interface AssistantChunk {
	seq: number; // insertion sequence — defines display order
	type: "thinking" | "content" | "tool" | "notice";
	// per-type fields
	contentText?: string; // for 'thinking' and 'content'
	tool?: ToolExecution; // for 'tool'
	// for 'notice': a standalone status line (retry / error / model / stopped)
	// rendered with its own icon + colour, not folded into assistant prose.
	notice?: { level: NoticeLevel; label: string; text: string };
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
}

function createDefaultState(): SessionState {
	return {
		turns: [],
		currentTurnId: null,
		thinkingDisplayMode: "expanded",
		thinkingLevel: "off",
	};
}

function containsTextualToolMarkup(content: string): boolean {
	return (
		/<(?:tool\\?_call|function|parameter)\b/i.test(content) ||
		/\[\[\s*tool_call\s*\(/i.test(content)
	);
}

export class Transcript {
	private state: SessionState = createDefaultState();
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
			case "message_update":
				this.handleMessageUpdate(event as MessageUpdateEvent);
				break;
			case "notice":
				this.handleNotice(event as ParsedBridgeEvent & { type: "notice" });
				break;
			case "subagent_chunk":
				this.handleSubagentChunk(event);
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
			.find((t) => !t.isComplete);
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
		const sameName: AssistantChunk[] = [];
		for (let i = chunks.length - 1; i >= 0; i--) {
			const c = chunks[i];
			if (c.type !== "tool" || c.isComplete || !c.tool) continue;
			if (event.tool_call_id && c.tool.tool_call_id === event.tool_call_id) {
				return c;
			}
			if (c.tool.tool_name === event.tool_name) sameName.push(c);
		}
		// A name-only match is safe only when it is unambiguous. Parallel calls
		// to the same tool must remain separate and are reconciled by id.
		if (sameName.length !== 1) return undefined;
		const candidateId = sameName[0].tool?.tool_call_id;
		const candidateIsPlaceholder =
			typeof candidateId === "string" && /^tool_\d+$/.test(candidateId);
		return !event.tool_call_id || !candidateId || candidateIsPlaceholder
			? sameName[0]
			: undefined;
	}

	private findToolChunk(
		chunks: AssistantChunk[],
		toolCallId?: string,
		toolName?: string,
	): AssistantChunk | undefined {
		if (toolCallId) {
			const exact = chunks
				.slice()
				.reverse()
				.find(
					(chunk) =>
						chunk.type === "tool" &&
						chunk.tool?.tool_call_id === toolCallId,
				);
			if (exact) return exact;
		}
		const incomplete = chunks.filter(
			(chunk) =>
				chunk.type === "tool" &&
				!chunk.isComplete &&
				(!toolName || chunk.tool?.tool_name === toolName),
		);
		return incomplete.length === 1 ? incomplete[0] : undefined;
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
			lastContent.contentText = (lastContent.contentText || "") + token;
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

	/** Reconcile a provider's full assistant snapshot with streamed deltas. */
	private handleMessageUpdate(event: MessageUpdateEvent): void {
		if (event.message.role !== "assistant") return;
		const full = event.message.content ?? "";
		const turn = this.getCurrentTurn();
		if (!turn) return;
		const message = this.ensureAssistant(turn);
		const rendered = message.chunks
			.filter((chunk) => chunk.type === "content")
			.map((chunk) => chunk.contentText ?? "")
			.join("");
		let latestContentEnd = message.chunks.length;
		while (
			latestContentEnd > 0 &&
			message.chunks[latestContentEnd - 1].type !== "content"
		) {
			latestContentEnd--;
		}
		let latestContentStart = latestContentEnd;
		while (
			latestContentStart > 0 &&
			message.chunks[latestContentStart - 1].type === "content"
		) {
			latestContentStart--;
		}
		const latestRendered = message.chunks
			.slice(latestContentStart, latestContentEnd)
			.map((chunk) => chunk.contentText ?? "")
			.join("");
		// Textual tool calls stream as ordinary content before agent-core can
		// promote them. The final snapshot contains structured tool_calls and
		// sanitized prose, so replace the streamed content instead of trying to
		// append a suffix to markup that no longer shares the same prefix.
		if (event.message.tool_calls?.length && full !== latestRendered) {
			// Some providers stream useful prose but return an empty content
			// snapshot alongside structured tool calls. Empty is not authoritative
			// unless the streamed run actually contains textual tool-call markup.
			if (!full && !containsTextualToolMarkup(latestRendered)) return;
			// A turn may contain several provider messages separated by tools.
			// Reconcile only the latest contiguous content run, at its original
			// position. Removing every content chunk would pull completed tools
			// above prose that was emitted before those tools.
			if (latestContentStart < latestContentEnd) {
				message.chunks.splice(
					latestContentStart,
					latestContentEnd - latestContentStart,
				);
			}
			if (full) {
				message.chunks.splice(latestContentStart, 0, {
					seq: latestContentStart,
					type: "content",
					contentText: full,
					isComplete: true,
				});
			}
			message.chunks.forEach((chunk, index) => {
				chunk.seq = index;
			});
			return;
		}
		if (!full) return;
		// Streaming backends already delivered this prefix. Append only the
		// missing suffix; non-streaming backends append the complete response.
		if (full.startsWith(rendered)) {
			const missing = full.slice(rendered.length);
			if (missing) this.handleToken(missing);
		}
	}

	// Push a status line as its own chunk so it renders distinctly (iconed,
	// coloured) and never merges into the surrounding assistant prose. Closes any
	// open thinking/content chunk first so the notice lands on its own line.
	private handleNotice(event: {
		level: NoticeLevel;
		label: string;
		text: string;
	}): void {
		if (this.integrateSubagentLifecycleNotice(event)) return;

		// Detect subagent tool call notices (label starts with "↳") and
		// store them on the parent spawn_agent ToolExecution so the expanded
		// view can render the child's tool activity.
		const subagentMatch = event.label.match(/^↳ (.+)$/);
		if (subagentMatch) {
			this.storeChildToolCall(subagentMatch[1], event);
			// Child activity belongs to the integrated spawn_agent card. Avoid
			// duplicating every call as a top-level transcript notice.
			return;
		}

		const turn = this.getCurrentTurn();
		if (!turn) return;
		const msg = this.ensureAssistant(turn);
		this.closeStreamingOfType("thinking", msg.chunks);
		this.closeStreamingOfType("content", msg.chunks);
		msg.chunks.push({
			seq: msg.chunks.length,
			type: "notice",
			notice: { level: event.level, label: event.label, text: event.text },
			isComplete: true,
		});
	}

	private integrateSubagentLifecycleNotice(event: {
		level: NoticeLevel;
		label: string;
		text: string;
	}): boolean {
		const match = /^Subagent (.+)$/.exec(event.label);
		if (!match) return false;
		const turn = this.getCurrentTurn();
		const parent = turn?.assistantMessage?.chunks
			.slice()
			.reverse()
			.find(
				(chunk) =>
					chunk.type === "tool" &&
					["spawn_agent", "spawn_agents"].includes(
						chunk.tool?.tool_name ?? "",
					) &&
					!chunk.isComplete,
			)?.tool;
		if (!parent) return false;
		const details = (parent.details ??= {});
		if (parent.tool_name === "spawn_agents") {
			const lifecycle = (details.lifecycle as Array<{
				agent: string;
				level: NoticeLevel;
				text: string;
			}> | undefined) ?? [];
			lifecycle.push({ agent: match[1], level: event.level, text: event.text });
			details.lifecycle = lifecycle;
			return true;
		}
		details.agent = details.agent || match[1];
		if (event.level === "success") details.status = "completed";
		if (event.level === "warn" || event.level === "error") {
			details.status = "failed";
		}
		details.lifecycleSummary = event.text;
		return true;
	}

	/**
	 * Preserve child-tool notices from older/replayed event streams. New live
	 * streams use subagent_chunk events, but saved sessions may still contain
	 * the former notice representation.
	 */
	private storeChildToolCall(
		agentId: string,
		event: { level: NoticeLevel; label: string; text: string },
	): void {
		const toolChunk = this.findOpenSubagentToolChunk();
		if (!toolChunk?.tool) return;

		const details = (toolChunk.tool.details ??= {});
		const childToolCalls = (details.childToolCalls as ChildToolCall[]) ??
			(details.childToolCalls = []);
		const text = event.text.trim();
		const match = /^([▶✓✗])\s+(\S+)\s+(\S+)(?:\s+([\s\S]*))?$/.exec(text);
		const marker = match?.[1];
		const toolCallId = match?.[2] ?? "";
		const toolName = match?.[3] ?? text.slice(0, 40);
		const payload = match?.[4] ?? "";

		if (marker === "✓" || marker === "✗") {
			const running = childToolCalls.find(
				(call) =>
					call.toolCallId === toolCallId &&
					call.status === "running",
			);
			if (running) {
				running.status = marker === "✗" ? "failed" : "completed";
				running.isError = marker === "✗";
				running.resultPreview = payload;
				return;
			}
		}

		childToolCalls.push({
			agentId,
			toolCallId,
			toolName,
			args: marker === "▶" ? payload : "",
			status:
				marker === "▶"
					? "running"
					: marker === "✗"
						? "failed"
						: "completed",
			isError: marker === "✗" || event.level === "warn",
			resultPreview: marker === "▶" ? undefined : payload,
		});
	}

	/** Find the most recent incomplete singular or batch subagent tool chunk. */
	private findOpenSubagentToolChunk(): AssistantChunk | undefined {
		const turn = this.getCurrentTurn();
		return turn?.assistantMessage?.chunks
			.slice()
			.reverse()
			.find(
				(c) =>
					c.type === "tool" &&
					["spawn_agent", "spawn_agents"].includes(c.tool?.tool_name ?? "") &&
					!c.isComplete,
			);
	}

	/**
	 * Store one ordered chunk (thinking/content delta or tool call) from a
	 * subagent onto its parent spawn_agent/spawn_agents ToolExecution. Chunks
	 * are kept in the child's own emit-time seq order so the expanded view can
	 * interleave text and tool calls exactly as they happened, the same way
	 * the parent agent's own AssistantChunk stream is interleaved.
	 */
	private handleSubagentChunk(event: SubagentChunkEvent): void {
		const toolChunk = this.findOpenSubagentToolChunk();
		if (!toolChunk?.tool) return;

		const details = (toolChunk.tool.details ??= {});
		const childChunks = (details.childChunks as ChildChunk[]) ??
			(details.childChunks = []);

		if (event.kind === "thinking" || event.kind === "content") {
			const last = childChunks[childChunks.length - 1];
			if (last && last.type === event.kind && !last.isComplete && last.agentId === event.agentId) {
				last.contentText = (last.contentText ?? "") + event.delta;
				last.seq = event.seq;
				return;
			}
			childChunks.push({
				seq: event.seq,
				agentId: event.agentId,
				type: event.kind,
				contentText: event.delta,
				isComplete: false,
			});
			return;
		}

		if (event.kind === "tool_start") {
			childChunks.push({
				seq: event.seq,
				agentId: event.agentId,
				type: "tool",
				tool: {
					agentId: event.agentId,
					toolCallId: event.toolCallId,
					toolName: event.toolName,
					args: event.args,
					status: "running",
				},
				isComplete: false,
			});
			return;
		}
		if (event.kind !== "tool_end") return;

		// tool_end
		const running = childChunks
			.slice()
			.reverse()
			.find(
				(c) =>
					c.type === "tool" &&
					c.tool?.toolCallId === event.toolCallId &&
					c.tool?.status === "running",
			);
		if (running?.tool) {
			running.tool.status = event.isError ? "failed" : "completed";
			running.tool.isError = event.isError;
			running.tool.resultPreview = event.result;
			running.isComplete = true;
			running.seq = event.seq;
			return;
		}
		// Completion arrived without a matching start (e.g. truncated replay) —
		// record it standalone rather than dropping the result.
		childChunks.push({
			seq: event.seq,
			agentId: event.agentId,
			type: "tool",
			tool: {
				agentId: event.agentId,
				toolCallId: event.toolCallId,
				toolName: event.toolName,
				args: "",
				status: event.isError ? "failed" : "completed",
				isError: event.isError,
				resultPreview: event.result,
			},
			isComplete: true,
		});
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
				startedAt: Date.now(),
			},
			isComplete: false,
		});
	}

	private handleToolUpdate(event: ToolUpdateEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		const toolChunk = this.findToolChunk(
			turn.assistantMessage.chunks,
			event.tool_call_id,
			event.tool_name || event.tool,
		);
		if (!toolChunk?.tool) return;

		if (event.partial_result !== undefined) {
			if (event.update_kind === "output") {
				toolChunk.tool.streamOutput =
					(toolChunk.tool.streamOutput || "") + String(event.partial_result);
			} else {
				// Argument fragments are kept separately from human-readable progress.
				toolChunk.tool.partialResult =
					(toolChunk.tool.partialResult || "") + String(event.partial_result);
			}
		}
	}

	private handleToolEnd(event: ToolEndEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		const assistant = turn.assistantMessage;

		const toolChunk = this.findToolChunk(
			assistant.chunks,
			event.tool_call_id,
			event.tool_name || event.tool,
		);
		if (!toolChunk?.tool) return;

		// Mark the tool as finished
		const tool = toolChunk.tool;
		if (tool.partialResult) {
			try {
				const streamedArgs = JSON.parse(tool.partialResult) as unknown;
				if (streamedArgs && typeof streamedArgs === "object" && !Array.isArray(streamedArgs)) {
					tool.args = { ...(tool.args ?? {}), ...(streamedArgs as Record<string, unknown>) };
				}
			} catch {
				// A provider may finish without a complete argument stream. The renderer
				// can still recover paths from the textual tool result.
			}
		}
		tool.result = event.result !== undefined ? String(event.result) : "";
		tool.details = {
			...(tool.details ?? {}),
			...(event.details ?? {}),
		};
		if (
			["spawn_agent", "spawn_agents"].includes(tool.tool_name) &&
			tool.streamOutput
		) {
			tool.details.streamTranscript = tool.streamOutput;
		}
		tool.partialResult = undefined;
		tool.streamOutput = undefined;
		const isError = (event as unknown as Record<string, unknown>).is_error;
		if (isError !== undefined) {
			tool.isError = Boolean(isError);
		}
		tool.isComplete = true;
		if (tool.startedAt !== undefined) tool.durationMs = Math.max(0, Date.now() - tool.startedAt);
		toolChunk.isComplete = true;
	}

	private handleTurnEnd(event: TurnEndEvent): void {
		if (event.final_message?.role === "assistant") {
			this.handleMessageUpdate({
				type: "message_update",
				turnId: event.turn_id,
				message: event.final_message,
			});
		}
		const turn = this.getTurnById(event.turn_id);
		if (turn) {
			if (turn.assistantMessage) {
				for (const chunk of turn.assistantMessage.chunks) {
					chunk.isComplete = true;
					if (chunk.type === "tool" && chunk.tool) {
						chunk.tool.isComplete = true;
					}
				}
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
		this.state = createDefaultState();
		this.notify();
	}

	/**
	 * Replace all turns with restored session turns (resume / session switch).
	 * Preserves display settings; only the conversation content is swapped.
	 */
	loadTurns(turns: Turn[]): void {
		this.state.turns = [...turns];
		this.state.currentTurnId = null;
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

	// ── Transcript query helpers ─────────────────────────────────────────

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

	/** Return the current turn's thinking chunks as plain text. */
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
