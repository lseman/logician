// ── Transcript management ──────────────────────────────────────────────────────
// Interleaved chunk model — chunks ordered by arrival time.
// Rendering follows chronological order: thinking → response → tool → thinking …

import type {
	MessageUpdateEvent,
	SubagentChunkEvent,
	TranscriptEvent,
	ToolCallStartEvent,
	ToolCallUpdateEvent,
	ToolEndEvent,
	ToolStartEvent,
	ToolUpdateEvent,
	TurnEndEvent,
} from "../runtime/events.ts";

export type ThinkingDisplayStyle = "collapsed" | "summary" | "expanded";

// ── Tool execution (used by tool chunks) ──────────────────────────────────────

export interface ToolExecution {
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
	/** Position within a spawn_agents batch, if run as part of one. */
	taskIndex?: number;
}

export interface ChildChunk {
	seq: number;
	agentId: string;
	type: "thinking" | "content" | "tool";
	contentText?: string; // for 'thinking' and 'content'
	tool?: ChildToolCall; // for 'tool'
	isComplete: boolean;
	/** Position within a spawn_agents batch, if run as part of one. */
	taskIndex?: number;
}

/** Structured per-task live status for a spawn_agents batch — the single
 * source of truth for status badges, replacing the old streamOutput
 * marker-string protocol. Keyed by task index. */
export interface TaskStatus {
	taskIndex: number;
	agentId: string;
	agent: string;
	status: "running" | "completed" | "failed";
	startedAt: number;
	endedAt?: number;
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
	userMessage: UserMessage | null;
	assistantMessage: AssistantMessage | null;
	isComplete: boolean;
	/** Monotonically increasing content revision. Incremented on every chunk/tool mutation.
	 * The TUI uses this for O(1) prefix-scan comparisons instead of full turnRevisionFor(). */
	contentRevision?: number;
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
	/** Global content revision counter — incremented each time any turn's content changes.
	 * The TUI tracks the last-seen value to skip prefix scans when nothing changed. */
	private _contentRevision = 0;

	// ── Event handling ─────────────────────────────────────────────────────

	handleEvent(event: TranscriptEvent): void {
		const revisionBefore = this._contentRevision;
		const turnCountBefore = this.state.turns.length;
		const currentTurnBefore = this.state.currentTurnId;
		switch (event.type) {
			case "turn_start":
				this.handleTurnStart(event);
				break;
			case "token":
				this.handleToken(String(event.token || ""));
				break;
			case "message_update":
				this.handleMessageUpdate(event);
				break;
			case "notice":
				this.handleNotice(event);
				break;
			case "subagent_chunk":
				this.handleSubagentChunk(event);
				break;
			case "subagent_lifecycle":
				this.handleSubagentLifecycle(event);
				break;
			case "thinking_token":
				this.handleThinkingToken(String(event.token || ""));
				break;
			case "tool_call_start":
				this.handleToolStart(event);
				break;
			case "tool_call_update":
				this.handleToolCallUpdate(event);
				break;
			case "tool_call_id_update":
				this.handleToolCallIdUpdate(event);
				break;
			case "tool_execution_start":
				this.handleToolStart(event);
				break;
			case "tool_execution_update":
				this.handleToolUpdate(event);
				break;
			case "tool_execution_end": {
				this.handleToolEnd(event);
				break;
			}
			case "turn_end":
				this.handleTurnEnd(event);
				break;
		}
		if (
			this._contentRevision !== revisionBefore ||
			this.state.turns.length !== turnCountBefore ||
			this.state.currentTurnId !== currentTurnBefore
		) {
			this.notify();
		}
	}

	private handleTurnStart(event: { turnId: string }): void {
		if (!event.turnId) return;
		// An incomplete turn is always at or near the tail (new turns are only
		// ever appended, and streaming completes roughly in order), so scan
		// backward in place rather than copying + reversing the whole,
		// session-lifetime-unbounded turns array on every turn-start event.
		let pending: Turn | undefined;
		for (let i = this.state.turns.length - 1; i >= 0; i--) {
			if (!this.state.turns[i].isComplete) {
				pending = this.state.turns[i];
				break;
			}
		}
		if (pending) {
			// Reuse the open turn (e.g. slash /spawn after addTurn, or a
			// user message already registered by the TUI).
			pending.id = event.turnId;
			pending.contentRevision = this._contentRevision;
			this.state.currentTurnId = event.turnId;
			return;
		}
		// No open turn: open a fresh one. Never rebind currentTurnId onto a
		// completed prior turn — that split tool_start from lifecycle/stream
		// events and left /spawn cards without their agent output.
		const turn: Turn = {
			id: event.turnId,
			userMessage: null,
			assistantMessage: {
				type: "assistant",
				chunks: [],
				isComplete: false,
			},
			isComplete: false,
			contentRevision: this._contentRevision,
		};
		this.state.turns.push(turn);
		this.state.currentTurnId = event.turnId;
	}

	/** Marks a turn's content as changed: advances the global counter and
	 * stamps it onto the turn, so the TUI's O(1) prefix-scan (comparing a
	 * turn's stored contentRevision against this stamp) correctly detects the
	 * change instead of treating the turn as unchanged. */
	private bumpTurnRevision(turn: Turn): void {
		this._contentRevision++;
		turn.contentRevision = this._contentRevision;
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
	 * Find the tool card created while its call arguments were streaming.
	 * Stable call IDs make name-based reconciliation unnecessary and keep
	 * parallel calls to the same tool independent.
	 */
	private findReusableToolChunk(
		chunks: AssistantChunk[],
		event: ToolCallStartEvent | ToolStartEvent,
	): AssistantChunk | undefined {
		for (let i = chunks.length - 1; i >= 0; i--) {
			const c = chunks[i];
			if (c.type !== "tool" || c.isComplete || !c.tool) continue;
			if (c.tool.tool_call_id === event.toolCallId) {
				return c;
			}
		}
		return undefined;
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
					chunk =>
						chunk.type === "tool" && chunk.tool?.tool_call_id === toolCallId,
				);
			if (exact) return exact;
		}
		const incomplete = chunks.filter(
			chunk =>
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
		this.bumpTurnRevision(turn);
	}

	/** Reconcile a provider's full assistant snapshot with streamed deltas. */
	private handleMessageUpdate(event: MessageUpdateEvent): void {
		if (event.message.role !== "assistant") return;
		const full = event.message.content ?? "";
		const turn = this.getCurrentTurn();
		if (!turn) return;
		const message = this.ensureAssistant(turn);
		const rendered = message.chunks
			.filter(chunk => chunk.type === "content")
			.map(chunk => chunk.contentText ?? "")
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
			.map(chunk => chunk.contentText ?? "")
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
			this.bumpTurnRevision(turn);
			return;
		}
		if (!full) return;
		// Streaming backends already delivered this prefix. Append only the
		// missing suffix; non-streaming backends append the complete response.
		if (full.startsWith(rendered)) {
			const missing = full.slice(rendered.length);
			if (missing) this.handleToken(missing);
		}
		this.bumpTurnRevision(turn);
	}

	// Push a status line as its own chunk so it renders distinctly (iconed,
	// coloured) and never merges into the surrounding assistant prose. Closes any
	// open thinking/content chunk first so the notice lands on its own line.
	private handleNotice(event: {
		level: NoticeLevel;
		label: string;
		text: string;
	}): void {
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

		// MCP load/reload notices only carry a server count — the status bar's
		// mcp indicator (loading… -> N, driven by this same event in
		// bridge-event-handler.ts) already surfaces that. A transcript line
		// per load is redundant clutter, especially since it can fire more
		// than once per session (startup, /mcp refresh, plugin toggle).
		if (event.label === "MCP") return;

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
		this.bumpTurnRevision(turn);
	}

	/**
	 * Structured start/end lifecycle for one subagent run — the single source
	 * of truth for per-task live status in a spawn_agents batch, stored as
	 * `details.taskStatus[taskIndex]`. Replaces the old approach of parsing
	 * `▶/✓/×` markers out of a streamOutput string.
	 */
	private handleSubagentLifecycle(event: {
		phase: "start" | "end";
		agentId: string;
		agent: string;
		task?: string;
		result?: string;
		isError?: boolean;
		turns?: number;
		taskIndex?: number;
	}): void {
		let toolChunk = this.findOpenSubagentToolChunk();
		const turn = this.getCurrentTurn();
		// Direct-mode /spawn: lifecycle can race tool_end. Only create a
		// placeholder on end (never on start) so we don't leave a second
		// incomplete spawn_agent card stuck on "running".
		if (!toolChunk?.tool && event.phase === "end") {
			const turn = this.getCurrentTurn();
			if (!turn?.assistantMessage) return;
			const assistant = turn.assistantMessage;
			const chunk: AssistantChunk = {
				seq: assistant.chunks.length,
				type: "tool",
				tool: {
					tool_name: "spawn_agent",
					tool_call_id: `lifecycle_${Date.now()}`,
					args: event.task
						? { task: event.task, agent: event.agent }
						: undefined,
					result: event.result,
					partialResult: undefined,
					isError: event.isError ?? false,
					isComplete: false,
					startedAt: Date.now(),
				},
				isComplete: false,
			};
			assistant.chunks.push(chunk);
			toolChunk = chunk;
		}
		if (!toolChunk?.tool) return;
		const details = (toolChunk.tool.details ??= {});

		if (toolChunk.tool.tool_name === "spawn_agents") {
			if (event.taskIndex === undefined) return;
			const taskStatus =
				(details.taskStatus as Record<number, TaskStatus>) ??
				(details.taskStatus = {});
			if (event.phase === "start") {
				taskStatus[event.taskIndex] = {
					taskIndex: event.taskIndex,
					agentId: event.agentId,
					agent: event.agent,
					status: "running",
					startedAt: Date.now(),
				};
			} else {
				const existing = taskStatus[event.taskIndex];
				taskStatus[event.taskIndex] = {
					taskIndex: event.taskIndex,
					agentId: event.agentId,
					agent: event.agent,
					status: event.isError ? "failed" : "completed",
					startedAt: existing?.startedAt ?? Date.now(),
					endedAt: Date.now(),
				};
			}
			return;
		}

		// Single spawn_agent: fold into the tool's own details, as before.
		details.agent = details.agent || event.agent;
		if (event.phase === "start") {
			if (details.status !== "completed" && details.status !== "failed") {
				details.status = "running";
			}
			if (event.task && !toolChunk.tool.args) {
				toolChunk.tool.args = { task: event.task, agent: event.agent };
			}
			return;
		}

		details.status = event.isError ? "failed" : "completed";
		details.lifecycleSummary = event.isError
			? event.result
			: `done${event.turns ? ` in ${event.turns} turn(s)` : ""}`;
		// Close the card on lifecycle end so the UI cannot stick on
		// "running" if tool_end is delayed or missed in direct mode.
		toolChunk.tool.isError = event.isError ?? false;
		if (event.result !== undefined && toolChunk.tool.result === undefined) {
			toolChunk.tool.result = event.result;
		}
		toolChunk.tool.isComplete = true;
		toolChunk.isComplete = true;
		if (turn) this.bumpTurnRevision(turn);
		if (
			toolChunk.tool.startedAt !== undefined &&
			toolChunk.tool.durationMs === undefined
		) {
			toolChunk.tool.durationMs = Math.max(
				0,
				Date.now() - toolChunk.tool.startedAt,
			);
		}
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
		const turn = this.getCurrentTurn();
		const childToolCalls =
			(details.childToolCalls as ChildToolCall[]) ??
			(details.childToolCalls = []);
		const text = event.text.trim();
		const match = /^([▶✓✗])\s+(\S+)\s+(\S+)(?:\s+([\s\S]*))?$/.exec(text);
		const marker = match?.[1];
		const toolCallId = match?.[2] ?? "";
		const toolName = match?.[3] ?? text.slice(0, 40);
		const payload = match?.[4] ?? "";

		if (marker === "✓" || marker === "✗") {
			const running = childToolCalls.find(
				call => call.toolCallId === toolCallId && call.status === "running",
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
				marker === "▶" ? "running" : marker === "✗" ? "failed" : "completed",
			isError: marker === "✗" || event.level === "warn",
			resultPreview: marker === "▶" ? undefined : payload,
		});
	}

	/**
	 * Find the tool chunk for a spawn_agent/spawn_agents call. Checks incomplete
	 * chunks first (open stream), then falls back to the most recent completed
	 * spawn chunk so that direct-mode /spawn can still capture lifecycle events
	 * that arrive after the tool_end has already closed the chunk.
	 */
	private findOpenSubagentToolChunk(): AssistantChunk | undefined {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return undefined;
		const chunks = turn.assistantMessage.chunks.slice().reverse();
		// 1. Incomplete (still streaming)
		for (const c of chunks) {
			if (
				c.type === "tool" &&
				["spawn_agent", "spawn_agents"].includes(c.tool?.tool_name ?? "") &&
				!c.isComplete
			) {
				return c;
			}
		}
		// 2. Recently completed (direct-mode /spawn: tool_end fires before lifecycle)
		for (const c of chunks) {
			if (
				c.type === "tool" &&
				["spawn_agent", "spawn_agents"].includes(c.tool?.tool_name ?? "") &&
				c.isComplete
			) {
				return c;
			}
		}
		return undefined;
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

		const turn = this.getCurrentTurn();
		const details = (toolChunk.tool.details ??= {});
		const childChunks =
			(details.childChunks as ChildChunk[]) ?? (details.childChunks = []);

		if (event.kind === "thinking" || event.kind === "content") {
			const last = childChunks[childChunks.length - 1];
			if (
				last &&
				last.type === event.kind &&
				!last.isComplete &&
				last.agentId === event.agentId
			) {
				last.contentText = (last.contentText ?? "") + event.delta;
				last.seq = event.seq;
				if (turn) this.bumpTurnRevision(turn);
				return;
			}
			childChunks.push({
				seq: event.seq,
				agentId: event.agentId,
				type: event.kind,
				contentText: event.delta,
				isComplete: false,
				taskIndex: event.taskIndex,
			});
			if (turn) this.bumpTurnRevision(turn);
			return;
		}

		if (event.kind === "tool_call_id_update") {
			const existing = childChunks
				.slice()
				.reverse()
				.find(
					chunk =>
						chunk.type === "tool" &&
						chunk.tool?.toolCallId === event.previousToolCallId,
				);
			if (existing?.tool) {
				existing.tool.toolCallId = event.toolCallId;
				existing.seq = Math.max(existing.seq, event.seq);
				if (turn) this.bumpTurnRevision(turn);
			}
			return;
		}

		if (event.kind === "tool_execution_start") {
			// Providers emit both tool_call_start and tool_execution_start for
			// the same call. Reuse by toolCallId so each child tool renders once.
			const existingStart = event.toolCallId
				? childChunks
						.slice()
						.reverse()
						.find(
							c => c.type === "tool" && c.tool?.toolCallId === event.toolCallId,
						)
				: undefined;
			if (existingStart?.tool) {
				if (event.args) existingStart.tool.args = event.args;
				if (event.toolName) existingStart.tool.toolName = event.toolName;
				existingStart.seq = Math.max(existingStart.seq, event.seq);
				if (
					existingStart.tool.status !== "completed" &&
					existingStart.tool.status !== "failed"
				) {
					existingStart.tool.status = "running";
					existingStart.isComplete = false;
				}
				if (turn) this.bumpTurnRevision(turn);
				return;
			}
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
					taskIndex: event.taskIndex,
				},
				isComplete: false,
				taskIndex: event.taskIndex,
			});
			if (turn) this.bumpTurnRevision(turn);
			return;
		}
		if (event.kind !== "tool_execution_end") return;

		// tool_end — also dedupe: tool_call_end and tool_execution_end both fire.
		const existingEnd = event.toolCallId
			? childChunks
					.slice()
					.reverse()
					.find(
						c => c.type === "tool" && c.tool?.toolCallId === event.toolCallId,
					)
			: undefined;
		if (existingEnd?.tool) {
			existingEnd.tool.status = event.isError ? "failed" : "completed";
			existingEnd.tool.isError = event.isError;
			if (event.result !== undefined && event.result !== "") {
				existingEnd.tool.resultPreview = event.result;
			}
			existingEnd.isComplete = true;
			existingEnd.seq = Math.max(existingEnd.seq, event.seq);
			return;
		}
		// Completion arrived without a matching start (e.g. truncated replay) —
		// record it standalone rather than dropping the result.
		childChunks.push({
			seq: event.seq,
			agentId: event.agentId,
			type: "tool",
			taskIndex: event.taskIndex,
			tool: {
				agentId: event.agentId,
				toolCallId: event.toolCallId,
				toolName: event.toolName,
				args: "",
				status: event.isError ? "failed" : "completed",
				isError: event.isError,
				resultPreview: event.result,
				taskIndex: event.taskIndex,
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
		this.bumpTurnRevision(turn);
	}

	private handleToolStart(event: ToolCallStartEvent | ToolStartEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn) return;

		const msg = this.ensureAssistant(turn);
		// Close any open thinking or content chunk before starting tool
		this.closeStreamingOfType("thinking", msg.chunks);
		this.closeStreamingOfType("content", msg.chunks);

		// Execution start enriches the card created by call start. These are
		// separate lifecycle phases joined by one stable toolCallId.
		const existing = this.findReusableToolChunk(msg.chunks, event);
		if (existing?.tool) {
			existing.tool.tool_name = event.toolName;
			existing.tool.tool_call_id = event.toolCallId;
			if (event.args !== undefined) {
				existing.tool.args = event.args as Record<string, unknown> | undefined;
			}
			this.bumpTurnRevision(turn);
			return;
		}

		msg.chunks.push({
			seq: msg.chunks.length,
			type: "tool",
			tool: {
				tool_name: event.toolName,
				tool_call_id: event.toolCallId,
				args: event.args as Record<string, unknown> | undefined,
				result: undefined,
				partialResult: undefined,
				isError: false,
				isComplete: false,
				startedAt: Date.now(),
			},
			isComplete: false,
		});
		this.bumpTurnRevision(turn);
	}

	private handleToolCallUpdate(event: ToolCallUpdateEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;
		const toolChunk = this.findToolChunk(
			turn.assistantMessage.chunks,
			event.toolCallId,
		);
		if (!toolChunk?.tool) return;
		toolChunk.tool.partialResult =
			(toolChunk.tool.partialResult || "") + event.delta;
		this.bumpTurnRevision(turn);
	}

	private handleToolCallIdUpdate(event: {
		previousToolCallId: string;
		toolCallId: string;
	}): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;
		const toolChunk = this.findToolChunk(
			turn.assistantMessage.chunks,
			event.previousToolCallId,
		);
		if (!toolChunk?.tool) return;
		toolChunk.tool.tool_call_id = event.toolCallId;
		this.bumpTurnRevision(turn);
	}

	private handleToolUpdate(event: ToolUpdateEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		const toolChunk = this.findToolChunk(
			turn.assistantMessage.chunks,
			event.toolCallId,
			event.toolName,
		);
		if (!toolChunk?.tool) return;

		toolChunk.tool.streamOutput =
			(toolChunk.tool.streamOutput || "") + event.partialResult;
		this.bumpTurnRevision(turn);
	}

	private handleToolEnd(event: ToolEndEvent): void {
		const turn = this.getCurrentTurn();
		if (!turn?.assistantMessage) return;

		const assistant = turn.assistantMessage;

		let toolChunk = this.findToolChunk(
			assistant.chunks,
			event.toolCallId,
			event.toolName,
		);
		// Direct-mode /spawn: tool_start was never emitted, or lifecycle:end
		// already closed a placeholder under a synthetic id. Reuse the most
		// recent spawn chunk before creating another card.
		const isDirectSpawnName = ["spawn_agent", "spawn_agents"].includes(
			event.toolName,
		);
		if (!toolChunk && isDirectSpawnName) {
			toolChunk = [...assistant.chunks]
				.reverse()
				.find(
					c =>
						c.type === "tool" &&
						["spawn_agent", "spawn_agents"].includes(c.tool?.tool_name ?? ""),
				);
			if (toolChunk?.tool && event.toolCallId) {
				toolChunk.tool.tool_call_id = event.toolCallId;
			}
		}
		if (!toolChunk && isDirectSpawnName) {
			const toolName = event.toolName;
			assistant.chunks.push({
				seq: assistant.chunks.length,
				type: "tool",
				tool: {
					tool_name: toolName,
					tool_call_id: event.toolCallId,
					args: undefined,
					result: undefined,
					partialResult: undefined,
					isError: false,
					isComplete: false,
					startedAt: Date.now(),
				},
				isComplete: false,
			});
			toolChunk = assistant.chunks[assistant.chunks.length - 1];
		}
		if (!toolChunk?.tool) return;

		// Mark the tool as finished
		const tool = toolChunk.tool;
		if (tool.partialResult) {
			try {
				const streamedArgs = JSON.parse(tool.partialResult) as unknown;
				if (
					streamedArgs &&
					typeof streamedArgs === "object" &&
					!Array.isArray(streamedArgs)
				) {
					tool.args = {
						...(tool.args ?? {}),
						...(streamedArgs as Record<string, unknown>),
					};
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
		if (event.isError !== undefined) {
			tool.isError = event.isError;
		}
		tool.isComplete = true;
		this.bumpTurnRevision(turn);
		// Prefer the tool's own measured duration (subagent metrics) when
		// present; otherwise fall back to wall-clock from tool_start.
		const metrics = tool.details?.metrics as
			| { durationMs?: number }
			| undefined;
		if (
			typeof metrics?.durationMs === "number" &&
			Number.isFinite(metrics.durationMs)
		) {
			tool.durationMs = Math.max(0, metrics.durationMs);
		} else if (tool.startedAt !== undefined) {
			tool.durationMs = Math.max(0, Date.now() - tool.startedAt);
		}
		toolChunk.isComplete = true;
	}

	private handleTurnEnd(event: TurnEndEvent): void {
		if (event.finalMessage?.role === "assistant") {
			this.handleMessageUpdate({
				type: "message_update",
				turnId: event.turnId,
				message: event.finalMessage,
			});
		}
		const turn = this.getTurnById(event.turnId);
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
			this.bumpTurnRevision(turn);
		}
	}

	// ── Turn management ──────────────────────────────────────────────────

	addTurn(userContent: string): Turn {
		const turn: Turn = {
			id: `turn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
			userMessage: { type: "user", content: userContent },
			assistantMessage: null,
			isComplete: false,
			contentRevision: this._contentRevision,
		};
		this.state.turns.push(turn);
		this.state.currentTurnId = turn.id;
		this.notify();
		return turn;
	}

	private getCurrentTurn(): Turn | undefined {
		if (!this.state.currentTurnId) return undefined;
		return this.state.turns.find(t => t.id === this.state.currentTurnId);
	}

	private getTurnById(id: string): Turn | undefined {
		return this.state.turns.find(t => t.id === id);
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
			contentRevision: this._contentRevision,
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
		return turn.assistantMessage.chunks.filter(c => c.type === "thinking");
	}

	/** Get thinking chunks from a completed turn. */
	getTurnThinkingChunks(turn: Turn): AssistantChunk[] {
		const assistant = turn.assistantMessage;
		if (!assistant) return [];
		return assistant.chunks.filter(c => c.type === "thinking");
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
			.filter(c => c.type === "content" && !c.isComplete)
			.map(c => c.contentText)
			.join("");
		return text || null;
	}

	/** Return the current turn's thinking chunks as plain text. */
	getStreamingThinking(): string[] {
		const chunks = this.getThinkingChunks();
		return chunks.map(c => c.contentText || "").filter(Boolean);
	}

	hasStreamingContent(): boolean {
		const content = this.getStreamingContent();
		return content !== null && content.length > 0;
	}

	hasStreamingThinking(): boolean {
		const chunks = this.getThinkingChunks();
		return chunks.some(t => (t.contentText || "").trim().length > 0);
	}

	getAssistantThinking(turn: Turn): string | null {
		const assistant = turn.assistantMessage;
		if (!assistant) return null;
		const thinking = assistant.chunks
			.filter(c => c.type === "thinking")
			.map(c => c.contentText || "")
			.filter(Boolean);
		if (thinking.length === 0) return null;
		return thinking.join("\n\n");
	}

	getAssistantContent(turn: Turn): string | null {
		const assistant = turn.assistantMessage;
		if (!assistant) return null;
		const text = assistant.chunks
			.filter(c => c.type === "content")
			.map(c => c.contentText)
			.join("");
		return text.length > 0 ? text : null;
	}

	getAssistantTools(turn: Turn): ToolExecution[] {
		const assistant = turn.assistantMessage;
		if (!assistant) return [];
		return (
			assistant.chunks
				.filter(c => c.type === "tool" && c.tool)
				// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
				.map(c => c.tool!)
		);
	}

	// ── Listener management ────────────────────────────────────────────

	onChange(callback: () => void): () => void {
		this.listeners.push(callback);
		return () => {
			this.listeners = this.listeners.filter(cb => cb !== callback);
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
