// ── Agent loop ─────────────────────────────────────────────────────────────────────
// Main ReAct-style loop for the agent. Mirrors Python AgentLoop.

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { withTimeout } from "./async-utils.ts";
import {
	BackendError,
	type BackendErrorCategory,
	type LLMBackend,
} from "./backend.ts";
import { buildBuiltinHooks, composeHooks } from "./builtin-hooks.ts";
import { createDefaultTools } from "./default-tools.ts";
import { createEventEmitter, type EventEmitter } from "./events.ts";
import {
	COMPACTION_TARGET_FRACTION,
	compactToFit,
	convertToChatFormat,
	createAssistantMessage,
	createSystemMessage,
	createToolResultMessage,
	createUserMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "./messages.ts";
import { parseToolCalls, parseToolInput } from "./parser.ts";
import { runHookEvent } from "./plugins.ts";
import { ToolResultCache } from "./tool-cache.ts";
import { ToolRegistry } from "./tools/registry.ts";
import {
	type AgentConfig,
	type AgentError,
	AgentErrorType,
	type AgentEvent,
	type AgentLoopHooks,
	type AgentMessage,
	type EventHandler,
	type Message,
	type StopReason,
	type ToolCall,
} from "./types.ts";

// Cap on pi-style continuations within one run, to bound runaway loops when a
// continuation hook keeps resuming the agent.
const DEFAULT_MAX_CONTINUATIONS = 12;
// Default per-turn timeout (ms). Prevents a single turn from hanging
// indefinitely when the model or tool execution stalls.
const DEFAULT_TURN_TIMEOUT_MS = 300_000; // 5 minutes

interface RunnableToolCall {
	kind: "runnable";
	call: ToolCall;
	args: Record<string, unknown>;
}

interface FinalToolCall {
	kind: "final";
	call: ToolCall;
	args: Record<string, unknown>;
	result: string;
	isError: boolean;
}

type PreparedLoopToolCall = RunnableToolCall | FinalToolCall;

interface ExecutedLoopToolCall {
	call: ToolCall;
	args: Record<string, unknown>;
	result: string;
	isError: boolean;
	details?: Record<string, unknown>;
}

// Outcome of a single turn. `continue` drives the outer loop; `productive`
// marks a turn that called tools (made forward progress) so the unproductive-
// turn safety cap can be reset rather than exhausted by real work.
interface TurnOutcome {
	continue: boolean;
	productive: boolean;
	// Carried into the turn_end event so a consumer can render a completed turn
	// from one event. Set when the turn produced an assistant message.
	stopReason?: StopReason;
	message?: Message;
	toolResults?: Message[];
}

// Result of a single LLM call, including why it stopped. On an unrecoverable
// failure stopReason is "error" and errorMessage holds the provider message so
// the turn can encode it as a visible, recoverable transcript entry.
interface LLMCallResult {
	content: string;
	toolCalls: ToolCall[];
	stopReason: StopReason;
	errorMessage?: string;
}

export interface TurnMetrics {
	turns: number;
	toolCalls: number;
	totalToolTimeMs: number;
	totalLlmTimeMs: number;
	compactions: number;
	retries: number;
	continuations: number;
}

export interface AgentLoopOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
	signal?: AbortSignal;
	// Prior conversation to continue from. When provided, the new user message
	// is appended to this history instead of starting a fresh transcript, so
	// follow-ups like "continue" retain context across turns.
	initialMessages?: Message[];
}

export class AgentLoop {
	private config: AgentConfig;
	private backend: LLMBackend;
	private toolRegistry: ToolRegistry;
	private emitter: EventEmitter;
	private _messages: Message[];
	private _sessionId = "";
	private cwd: string;
	private maxIterations: number;
	private iterationCount: number = 0;
	private onEvent?: EventHandler;
	private signal?: AbortSignal;
	private hooks: AgentLoopHooks;
	private initialMessages?: Message[];
	private continuationCount = 0;
	private maxContinuations: number;
	private _retryAttempt = 0;
	private toolCache: ToolResultCache;
	// Track terminate hint from afterToolCall — when ALL tools in a batch
	// set it, the loop stops after that batch (Pi-style early termination).
	private batchTerminate = false;
	private get maxRetries(): number {
		return this.config.maxRetries ?? 3;
	}
	private get retryBaseDelayMs(): number {
		return this.config.retryBaseDelayMs ?? 1000;
	}
	// Per-turn timeout to prevent runaway turns.
	private get turnTimeoutMs(): number {
		return this.config.turnTimeoutMs ?? DEFAULT_TURN_TIMEOUT_MS;
	}
	// Turn-level metrics collected during run().
	private _turnMetrics: TurnMetrics = {
		turns: 0,
		toolCalls: 0,
		totalToolTimeMs: 0,
		totalLlmTimeMs: 0,
		compactions: 0,
		retries: 0,
		continuations: 0,
	};

	/** Read-only snapshot of turn-level metrics (available after run() completes). */
	get turnMetrics(): TurnMetrics {
		return { ...this._turnMetrics };
	}

	constructor(options: AgentLoopOptions) {
		this.config = options.config;
		this.backend = options.backend;
		this.cwd = options.cwd || process.cwd();
		this.maxIterations = options.maxIterations || 30;
		this.signal = options.signal;
		this.initialMessages = options.initialMessages;
		this.maxContinuations =
			this.config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
		this.iterationCount = 0;
		this._messages = [];
		this.hooks = this.config.hooks || {};
		this.emitter = createEventEmitter();
		this.toolCache = new ToolResultCache(
			this.config.cacheSize ?? 1000,
			this.config.cacheTtlMs ?? 30_000,
		);

		// Set up tool registry
		this.toolRegistry = new ToolRegistry({ cwd: this.cwd });
		this.toolRegistry.registerMany(
			this.config.tools?.length ? this.config.tools : createDefaultTools(),
		);

		// Set up event handler
		this.wrapOnEvent();
	}

	// Capture the config's raw onEvent and replace it with a wrapper that fans
	// every event out through the emitter before the raw handler. Re-run on
	// config refresh so a reused loop keeps emitting through the emitter.
	private wrapOnEvent(): void {
		this.onEvent = this.config.onEvent;
		this.config.onEvent = (event: AgentEvent) => {
			this.emitter.emit(event);
			this.onEvent?.(event);
		};
	}

	get events(): EventEmitter {
		return this.emitter;
	}

	get tools(): ToolRegistry {
		return this.toolRegistry;
	}

	get messages(): Message[] {
		return this._messages;
	}

	/** Replace working messages (used by manual compaction). */
	setMessages(messages: Message[]): void {
		this._messages = messages;
	}

	/** Swap the abort signal for a new run (used by harness loop reuse). */
	updateSignal(signal: AbortSignal | undefined): void {
		this.signal = signal;
	}

	/**
	 * Refresh the loop's config when an existing loop is reused for a new prompt
	 * (harness loop reuse). Without this, runtime changes the harness applies
	 * between prompts — system prompt, temperature, tools, internalHooks — would
	 * not reach the loop, which otherwise keeps its construction-time config.
	 * Re-applies the emitter fan-out wrapper around the incoming raw onEvent,
	 * mirroring the constructor.
	 */
	updateConfig(config: AgentConfig): void {
		this.config = config;
		this.wrapOnEvent();
	}

	async run(userMessage: string): Promise<Message[]> {
		// Compose built-in safeguard hooks (guards, budget stop, proactive
		// compaction) with any user-supplied hooks. Built-in state (failure
		// counts, budget tracker, compaction cooldown) is per-run, so build
		// here rather than in the constructor.
		const builtin = buildBuiltinHooks({
			config: this.config,
			contextWindowTokens: () => this.contextWindowTokens(),
			toolDefs: () => this.toolRegistry.toToolDefinitions(),
		});
		this.hooks = composeHooks(
			[
				{ source: "builtin", hooks: builtin },
				{ source: "harness-queues", hooks: this.config.internalHooks },
				{ source: "user", hooks: this.config.hooks },
			],
			(error, event) => {
				this.emitEvent({
					type: "error",
					message: `${event} hook failed: ${error.message}`,
				});
			},
			this.config.onHookEvent,
		);

		// Initialize with system prompt
		const systemPrompt =
			this.config.systemPrompt || "You are a helpful assistant.";
		const sessionId = this.config.hookSessionId || `tui_${Date.now()}`;
		this._sessionId = sessionId;
		const transcriptPath = this.config.hookTranscriptPath || "";
		const hookBasePayload = {
			session_id: sessionId,
			transcript_path: transcriptPath,
			cwd: this.cwd,
		};
		const hookMessages = await this.userPromptHookMessages(
			userMessage,
			hookBasePayload,
		);
		this.appendTranscript(transcriptPath, {
			type: "user",
			timestamp: new Date().toISOString(),
			message: { role: "user", content: userMessage },
		});
		if (this.initialMessages?.length) {
			// Continue an existing conversation: keep prior history, refresh the
			// system prompt to the current one, append hook context + new turn.
			const priorNonSystem = this.initialMessages.filter(
				(m) => m.role !== "system",
			);
			this._messages = [
				createSystemMessage(systemPrompt),
				...priorNonSystem,
				...hookMessages,
				createUserMessage(userMessage),
			];
		} else {
			this._messages = [
				createSystemMessage(systemPrompt),
				...hookMessages,
				createUserMessage(userMessage),
			];
		}
		this.iterationCount = 0;
		this.continuationCount = 0;
		this._retryAttempt = 0;

		this.emitEvent({ type: "agent_start" });
		this.emitEvent({ type: "phase", phase: "idle" });

		// Safety cap counts only *unproductive* turns (no tool calls). Turns that
		// call tools make forward progress and reset the counter, so a long tool-
		// using task is not silently truncated mid-work. The cap exists solely to
		// bound a model that loops without acting.
		let unproductiveTurns = 0;
		while (true) {
			if (this.signal?.aborted) {
				this.emitEvent({ type: "error", message: "Operation aborted" });
				break;
			}
			if (unproductiveTurns >= this.maxIterations) {
				this.emitEvent({
					type: "max_iterations",
					iterations: this.iterationCount,
					limit: this.maxIterations,
				});
				break;
			}
			this.iterationCount++;
			const turnId = `turn_${this.iterationCount}`;
			const outcome = await this.runTurn(
				turnId,
				transcriptPath,
				hookBasePayload,
			);
			unproductiveTurns = outcome.productive ? 0 : unproductiveTurns + 1;
			if (!outcome.continue) break;
		}

		await this.runHookSafely("Stop", {
			...hookBasePayload,
			stop_hook_active: false,
		});
		this.emitEvent({ type: "phase", phase: "idle" });
		this.emitEvent({ type: "agent_end", messages: this._messages });

		return this._messages;
	}

	/**
	 * Execute one turn: LLM call → tool execution → follow-up check.
	 * Returns true when the loop should continue, false when the turn
	 * completed (no more tools or follow-ups) or the turn was aborted.
	 */
	private async runTurn(
		turnId: string,
		transcriptPath: string,
		hookBasePayload: Record<string, unknown>,
	): Promise<TurnOutcome> {
		this.emitEvent({ type: "turn_start", turnId });
		this.emitEvent({ type: "phase", phase: "thinking" });
		// runTurnInner reports whether the loop should continue and whether the
		// turn was productive. When it stops, turn_end is emitted once here
		// rather than at every early-return.
		const outcome = await this.runTurnInner(
			turnId,
			transcriptPath,
			hookBasePayload,
		);
		if (!outcome.continue) {
			this.emitEvent({
				type: "turn_end",
				turnId,
				stopReason: outcome.stopReason,
				message: outcome.message,
				toolResults: outcome.toolResults,
			});
		}
		return outcome;
	}

	private async runTurnInner(
		turnId: string,
		transcriptPath: string,
		hookBasePayload: Record<string, unknown>,
	): Promise<TurnOutcome> {
		// Drain steering messages respecting queue mode
		const steeringMessages = await this.runGetSteeringMessages();
		if (steeringMessages.length) {
			this.appendInjectedMessages(transcriptPath, steeringMessages);
		}

		const toolDefs = this.toolRegistry.toToolDefinitions();

		// Get LLM response with timeout, auto-retry, and context-full compaction.
		// callLLM emits the context_update right before each provider request, so
		// no separate emit is needed here.
		let result = await this.callLLMGuarded(turnId, toolDefs);
		let { content: assistantContent, toolCalls: assistantToolCalls } = result;

		// Unrecoverable LLM failure: encode it as a visible assistant message so
		// the turn ends cleanly with the error in the transcript and the user (or
		// a follow-up) can recover ("retry", "switch model") rather than facing a
		// silent empty turn.
		if (result.stopReason === "error") {
			const text = `⚠️ Model request failed: ${result.errorMessage ?? "unknown error"}`;
			const message = createAssistantMessage(text);
			this.emitEvent({ type: "message_start", turnId, role: "assistant" });
			this._messages.push(message);
			this.appendTranscript(transcriptPath, {
				type: "assistant",
				timestamp: new Date().toISOString(),
				message: { role: "assistant", content: [{ type: "text", text }] },
			});
			this.emitEvent({ type: "message_end", turnId });
			return {
				continue: false,
				productive: false,
				stopReason: "error",
				message,
			};
		}

		// Empty response (no content, no tools) is ambiguous: a real stop or a
		// transient empty completion. handleEmptyResponse retries once and, if
		// still empty, either nudges-and-continues or ends the turn.
		if (!assistantContent && assistantToolCalls.length === 0) {
			const recovered = await this.handleEmptyResponse(
				turnId,
				toolDefs,
				transcriptPath,
			);
			if (recovered.outcome) return recovered.outcome;
			result = recovered.result;
			assistantContent = result.content;
			assistantToolCalls = result.toolCalls;
		}

		// Emit message_start before assistant response (for steering
		// detection — the bridge can detect when steering messages have
		// been consumed by checking if their text appears in messages).
		this.emitEvent({ type: "message_start", turnId, role: "assistant" });
		// Add assistant message
		const assistantMessage = createAssistantMessage(
			assistantContent,
			assistantToolCalls,
		);
		this._messages.push(assistantMessage);
		this.appendTranscript(transcriptPath, {
			type: "assistant",
			timestamp: new Date().toISOString(),
			message: {
				role: "assistant",
				content: assistantContent
					? [{ type: "text", text: assistantContent }]
					: [],
				tool_calls: assistantToolCalls.map((toolCall) => ({
					id: toolCall.id,
					name: toolCall.name,
					input: parseToolInput(toolCall.arguments),
				})),
			},
		});
		this.emitEvent({ type: "message_end", turnId });

		// A turn that calls tools made forward progress (productive) — even if it
		// then stops — so it never counts against the unproductive-turn cap.
		const hadToolCalls = assistantToolCalls.length > 0;
		const turnStopReason: StopReason = hadToolCalls
			? "tool_calls"
			: result.stopReason;
		// Tool-result messages executeToolCalls appended, for the turn_end payload.
		let toolResults: Message[] = [];

		// Check if we have tool calls
		if (hadToolCalls) {
			this.emitEvent({ type: "phase", phase: "tool" });
			const toolStart = Date.now();
			const beforeTools = this._messages.length;
			await this.executeToolCalls(
				assistantToolCalls,
				turnId,
				transcriptPath,
				hookBasePayload,
			);
			toolResults = this._messages.slice(beforeTools);
			this._turnMetrics.totalToolTimeMs += Date.now() - toolStart;
			this._turnMetrics.toolCalls += assistantToolCalls.length;

			if (this.signal?.aborted) {
				return {
					continue: false,
					productive: true,
					stopReason: "aborted",
					message: assistantMessage,
					toolResults,
				};
			}
			// Pi-style early termination: when ALL tools in the batch
			// set terminate=true, stop after this batch.
			if (this.batchTerminate) {
				this.batchTerminate = false;
				return {
					continue: false,
					productive: true,
					stopReason: turnStopReason,
					message: assistantMessage,
					toolResults,
				};
			}
		}

		// prepareNextTurn / shouldStopAfterTurn contract hooks.
		const isContinuation = this.continuationCount > 0;
		await this.runPrepareNextTurn(hadToolCalls, isContinuation);
		if (await this.runShouldStopAfterTurn(hadToolCalls, isContinuation)) {
			return {
				continue: false,
				productive: hadToolCalls,
				stopReason: turnStopReason,
				message: assistantMessage,
				toolResults,
			};
		}

		// Continue loop: model called tools or follow-ups exist.
		// turn_end stays open — the TUI sees one continuous turn.
		if (hadToolCalls) {
			return { continue: true, productive: true };
		}

		// No tool calls: check follow-ups before ending the turn.
		if (this.continuationCount < this.maxContinuations) {
			const followUps = await this.runGetFollowUpMessages(assistantContent);
			if (followUps.length) {
				this.continuationCount++;
				this._turnMetrics.continuations++;
				this.appendInjectedMessages(transcriptPath, followUps);
				return { continue: true, productive: false };
			}
		}

		// Truly done — no tools, no follow-ups.
		return {
			continue: false,
			productive: hadToolCalls,
			stopReason: turnStopReason,
			message: assistantMessage,
			toolResults,
		};
	}

	/**
	 * Recover from an empty LLM response (no content, no tool calls). Retries the
	 * call once: if the retry produces output, returns it for the turn to use
	 * ({ result }). If it is still empty, returns a terminal { outcome } — a
	 * nudge-and-continue while continuations remain, otherwise a clean stop —
	 * so the caller can return immediately.
	 */
	private async handleEmptyResponse(
		turnId: string,
		toolDefs: Record<string, unknown>[],
		transcriptPath: string,
	): Promise<{ result: LLMCallResult; outcome?: TurnOutcome }> {
		const result = await this.callLLMGuarded(turnId, toolDefs);
		if (result.content || result.toolCalls.length > 0) return { result };

		// Still empty after one retry.
		if (this.continuationCount < this.maxContinuations) {
			this.continuationCount++;
			this._turnMetrics.continuations++;
			this.appendInjectedMessages(transcriptPath, [
				createUserMessage(
					"Your last response was empty. If the task is complete, say so " +
						"explicitly. Otherwise, continue using your tools to make progress.",
				),
			]);
			return { result, outcome: { continue: true, productive: false } };
		}
		return {
			result,
			outcome: { continue: false, productive: false, stopReason: "stop" },
		};
	}

	/**
	 * Run callLLM under the per-turn timeout, recording LLM time and surfacing
	 * timeout / error events. Never throws: on failure returns an empty result
	 * so the caller can decide how to proceed (retry / nudge / stop).
	 */
	private async callLLMGuarded(
		turnId: string,
		toolDefs: Record<string, unknown>[],
	): Promise<LLMCallResult> {
		const llmStart = Date.now();
		try {
			return await withTimeout(
				this.callLLM(turnId, toolDefs),
				this.turnTimeoutMs,
			);
		} catch (e: unknown) {
			const error = e as Error;
			const agentErr = e as AgentError;
			const message =
				agentErr.type === AgentErrorType.TURN_TIMEOUT
					? `Turn ${turnId} timed out after ${this.turnTimeoutMs}ms`
					: error.message;
			this.emitEvent({ type: "error", message });
			return {
				content: "",
				toolCalls: [],
				stopReason: "error",
				errorMessage: message,
			};
		} finally {
			this._turnMetrics.totalLlmTimeMs += Date.now() - llmStart;
		}
	}

	/**
	 * Call the LLM with context-full compaction retry and auto-retry on
	 * transient provider errors. Returns the assistant content and tool calls.
	 */
	private async callLLM(
		turnId: string,
		toolDefs: Record<string, unknown>[],
	): Promise<LLMCallResult> {
		let assistantContent = "";
		let assistantToolCalls: ToolCall[] = [];
		let stopReason: StopReason = "stop";
		let errorMessage: string | undefined;
		let llmSuccess = false;
		let compactionAttempted = false;

		while (!llmSuccess && !compactionAttempted) {
			try {
				const activeToolDefs = this.toolRegistry.toToolDefinitions();
				// Transform context (prune / inject / drain nextTurn) before convert.
				await this.runTransformContext();
				// Apply custom convertToLlm (filters non-LLM messages) then convert to chat format.
				const llmMessages = (this.config.convertToLlm || defaultConvertToLlm)(
					this._messages as AgentMessage[],
				);
				const activeChatMessages = convertToChatFormat(llmMessages);
				this.emitContextUpdate(activeToolDefs);
				// Provider-boundary hooks: per-request headers/timeout + payload
				// rewrite. Resolved once per request, just before sending.
				const reqPatch = await this.runBeforeProviderRequest();
				const response = await this.backend.generate(activeChatMessages, {
					tools: activeToolDefs.length > 0 ? activeToolDefs : undefined,
					temperature: this.config.temperature || 0.5,
					maxTokens: this.config.maxTokens || 4096,
					signal: this.signal,
					thinkingLevel: this.config.thinkingLevel,
					headers: reqPatch?.headers,
					transformPayload: this.hooks.beforeProviderPayload
						? (payload) => this.applyProviderPayload(payload)
						: undefined,
					callbacks: {
						onDelta: (delta: string) => {
							assistantContent += delta;
							this.emitEvent({ type: "text_delta", turnId, delta });
							// Emit full partial message for UI updates
							this.emitEvent({
								type: "message_update",
								turnId,
								message: createAssistantMessage(
									assistantContent,
									assistantToolCalls,
								),
							});
						},
						onThinking: (delta: string) => {
							this.emitEvent({ type: "thinking_delta", delta });
						},
						onTextStart: () => {
							this.emitEvent({ type: "text_start", turnId });
						},
						onTextEnd: () => {
							this.emitEvent({ type: "text_end", turnId });
						},
						// Early "running" state while the model is still streaming the
						// call. The backend only fires this once the name is known, so
						// the UI reuses this chunk (by id/name) when the authoritative
						// tool_call_start is emitted in prepareLoopToolCall — no dup.
						onToolCallStart: (
							toolCallId: string,
							name: string,
							args: string,
						) => {
							this.emitEvent({
								type: "tool_call_start",
								toolName: name,
								toolCallId,
								args,
							});
						},
						onToolCallDelta: (toolCallId: string, delta: string) => {
							this.emitEvent({
								type: "tool_call_delta",
								toolCallId,
								delta,
							});
						},
					},
				});

				assistantContent = response.content || "";
				assistantToolCalls = response.toolCalls;
				if (assistantToolCalls.length === 0 && assistantContent) {
					assistantToolCalls = parseToolCalls(assistantContent);
					if (assistantToolCalls.length > 0) {
						this.emitEvent({
							type: "repair_nudge",
							turnId,
							repairStage: "parse_tool_calls",
							message: "Recovered tool call(s) from textual model output.",
						});
					}
				}
				// tool_calls supersedes the provider stop reason; otherwise carry it
				// through (stop / length).
				stopReason =
					assistantToolCalls.length > 0
						? "tool_calls"
						: response.stopReason === "length"
							? "length"
							: "stop";
				llmSuccess = true;
			} catch (e: unknown) {
				const error = e as Error;
				// Prefer the backend's typed category; fall back to message-string
				// matching for errors thrown outside the backend boundary.
				const category = classifyLoopError(error);

				// 1. Context-full → compact once and retry.
				if (!compactionAttempted && category === "context_full") {
					compactionAttempted = true;
					this._turnMetrics.compactions++;
					const before = this.estimatePayloadTokens(toolDefs);
					const ctxWindow = this.contextWindowTokens();
					// Forced ladder (triggerTokens 0): the provider already rejected
					// the request as too long, so compact regardless of local estimate.
					const compacted = compactToFit(this._messages, {
						triggerTokens: 0,
						targetTokens: ctxWindow
							? Math.floor(ctxWindow * COMPACTION_TARGET_FRACTION)
							: undefined,
						toolDefs,
					});
					if (compacted.changed) {
						this._messages = compacted.messages;
						const after = this.estimatePayloadTokens(toolDefs);
						this.emitEvent({
							type: "compaction",
							reason: "context_full",
							tokensBefore: before,
							tokensAfter: after,
						});
						this.emitContextUpdate(toolDefs, true);
						continue;
					}
					// Compaction didn't help — fall through to error.
				}

				// 2. Auto-retry on retryable provider errors (rate_limit / transient).
				if (
					this.config.autoRetryEnabled !== false &&
					(category === "rate_limit" || category === "transient")
				) {
					const canRetry = this._retryAttempt < this.maxRetries;
					if (canRetry) {
						this._retryAttempt++;
						this._turnMetrics.retries++;
						const delayMs =
							this.retryBaseDelayMs * 2 ** (this._retryAttempt - 1);
						this.emitEvent({
							type: "auto_retry_start",
							attempt: this._retryAttempt,
							maxRetries: this.maxRetries,
							delayMs,
							error: error.message,
						});
						await this._sleep(delayMs);
						this.emitEvent({
							type: "auto_retry_end",
							attempt: this._retryAttempt,
							success: true,
						});
						continue;
					}
				}

				// 3. Give up. Encode the failure so the caller can surface it as a
				// recoverable transcript entry rather than a silent empty turn.
				this.emitEvent({ type: "error", message: error.message });
				assistantContent = "";
				assistantToolCalls = [];
				stopReason = "error";
				errorMessage = error.message;
				break;
			}
		}

		// Reset retry state after each turn (success or failure).
		this._retryAttempt = 0;

		return {
			content: assistantContent,
			toolCalls: assistantToolCalls,
			stopReason,
			errorMessage,
		};
	}

	private emitEvent(event: AgentEvent): void {
		if (event.type === "turn_end" && this.config.turnEndCallback) {
			this.config.turnEndCallback(event.turnId);
		}
		if (this.config.onEvent) {
			this.config.onEvent(event);
		}
	}

	// Memoized payload-token estimate. estimateChatPayloadTokens re-serializes
	// the whole history, and a single turn queries it several times (context
	// update, compaction path) with unchanged messages. Cache by a cheap version
	// signature (count + last-message identity + tool-def count) so repeated
	// queries within a turn are free; recompute only when the history changes.
	private _tokenMemo?: { sig: string; tokens: number };

	private estimatePayloadTokens(tools: Record<string, unknown>[]): number {
		const last = this._messages[this._messages.length - 1];
		const sig = `${this._messages.length}|${last?.role ?? ""}:${
			last?.content?.length ?? 0
		}|${tools.length}`;
		if (this._tokenMemo?.sig === sig) return this._tokenMemo.tokens;
		const tokens = estimateChatPayloadTokens(this._messages, tools);
		this._tokenMemo = { sig, tokens };
		return tokens;
	}

	private emitContextUpdate(
		tools: Record<string, unknown>[] = this.toolRegistry.toToolDefinitions(),
		compacted = false,
	): void {
		this.emitEvent({
			type: "context_update",
			tokens: this.estimatePayloadTokens(tools),
			maxTokens: this.contextWindowTokens(),
			compacted,
		});
	}

	private contextWindowTokens(): number | undefined {
		const configured =
			this.config.contextWindowTokens ||
			envNumber("LOGICIAN_CONTEXT_WINDOW") ||
			envNumber("LOGICIAN_CTX_SIZE");
		return configured && configured > 0 ? configured : undefined;
	}

	private hooksEnabled(): boolean {
		return (
			this.config.runtimeHooksEnabled !== false &&
			process.env.LOGICIAN_HOOKS !== "0"
		);
	}

	private async userPromptHookMessages(
		userMessage: string,
		basePayload: Record<string, unknown>,
	): Promise<Message[]> {
		if (!this.hooksEnabled()) return [];
		try {
			const result = await runHookEvent("UserPromptSubmit", {
				...basePayload,
				prompt: userMessage,
				timeout_seconds: 30,
			});
			const context = (result.additional_contexts || [])
				.map((item) => String(item || "").trim())
				.filter(Boolean)
				.join("\n\n");
			if (!context) return [];
			return [
				createUserMessage(
					`<user-prompt-submit-hook>\n${context}\n</user-prompt-submit-hook>`,
				),
			];
		} catch {
			return [];
		}
	}

	// Hook call sites. Error isolation + reporting is owned by the HookBus
	// (composeHooks wires onError → the loop's error event), so these don't
	// try/catch. A handler that throws is skipped by the bus and reported once.

	private async runBeforeToolCall(
		toolCall: ToolCall,
		args: Record<string, unknown>,
	): Promise<
		| {
				content?: string;
				isError?: boolean;
				args?: Record<string, unknown>;
		  }
		| undefined
	> {
		if (!this.hooks.beforeToolCall) return undefined;
		return (
			(await this.hooks.beforeToolCall({
				toolCall,
				args,
				iteration: this.iterationCount,
			})) || undefined
		);
	}

	private async runAfterToolCall(
		toolCall: ToolCall,
		args: Record<string, unknown>,
		result: string,
		isError: boolean,
	): Promise<
		{ content?: string; isError?: boolean; terminate?: boolean } | undefined
	> {
		if (!this.hooks.afterToolCall) return undefined;
		return (
			(await this.hooks.afterToolCall({
				toolCall,
				args,
				result,
				isError,
				iteration: this.iterationCount,
			})) || undefined
		);
	}

	private async runPrepareNextTurn(
		hadToolCalls: boolean,
		isContinuation: boolean,
	): Promise<void> {
		if (!this.hooks.prepareNextTurn) return;
		const out = await this.hooks.prepareNextTurn({
			messages: this._messages,
			iteration: this.iterationCount,
			hadToolCalls,
			continuationCount: this.continuationCount,
			isContinuation,
		});
		if (out?.messages) this._messages = out.messages;
	}

	private async runShouldStopAfterTurn(
		hadToolCalls: boolean,
		isContinuation: boolean,
	): Promise<boolean> {
		if (!this.hooks.shouldStopAfterTurn) return false;
		return (
			(await this.hooks.shouldStopAfterTurn({
				messages: this._messages,
				iteration: this.iterationCount,
				hadToolCalls,
				continuationCount: this.continuationCount,
				isContinuation,
			})) === true
		);
	}

	// Transform the working context before the LLM call (after steering
	// injection, before convertToLlm). Lets hooks prune/inject at the
	// AgentMessage level — e.g. the harness drains its nextTurn queue here.
	private async runTransformContext(): Promise<void> {
		if (!this.hooks.transformContext) return;
		const out = await this.hooks.transformContext({
			messages: this._messages as AgentMessage[],
			iteration: this.iterationCount,
			signal: this.signal,
		});
		if (out?.messages) this._messages = out.messages as Message[];
	}

	// Resolve per-request headers / timeout from beforeProviderRequest hooks.
	private async runBeforeProviderRequest(): Promise<
		{ headers?: Record<string, string>; timeoutMs?: number } | undefined
	> {
		if (!this.hooks.beforeProviderRequest) return undefined;
		return (
			(await this.hooks.beforeProviderRequest({
				model: this.getModel(),
				sessionId: this._sessionId,
				iteration: this.iterationCount,
			})) ?? undefined
		);
	}

	// Apply beforeProviderPayload hooks to the raw request body. Passed to the
	// backend as transformPayload so the final, backend-built body is rewritten.
	private async applyProviderPayload(
		payload: Record<string, unknown>,
	): Promise<Record<string, unknown>> {
		if (!this.hooks.beforeProviderPayload) return payload;
		const out = await this.hooks.beforeProviderPayload({
			model: this.getModel(),
			payload,
		});
		return out?.payload ?? payload;
	}

	private async runGetSteeringMessages(): Promise<Message[]> {
		if (!this.hooks.getSteeringMessages) return [];
		const r = await this.hooks.getSteeringMessages({
			messages: this._messages,
			iteration: this.iterationCount,
		});
		return r?.length ? r : [];
	}

	private async runGetFollowUpMessages(
		assistantText: string,
	): Promise<Message[]> {
		if (!this.hooks.getFollowUpMessages) return [];
		const r = await this.hooks.getFollowUpMessages({
			messages: this._messages,
			iteration: this.iterationCount,
			assistantText,
			continuationCount: this.continuationCount,
			maxContinuations: this.maxContinuations,
		});
		return r?.length ? r : [];
	}

	private async executeToolCalls(
		toolCalls: ToolCall[],
		turnId: string,
		transcriptPath: string,
		hookBasePayload: Record<string, unknown>,
	): Promise<void> {
		const prepared: PreparedLoopToolCall[] = [];
		for (const toolCall of toolCalls) {
			if (this.signal?.aborted) {
				this.emitEvent({
					type: "error",
					message: "Operation aborted",
				});
				return;
			}
			prepared.push(
				await this.prepareLoopToolCall(toolCall, turnId, hookBasePayload),
			);
		}

		const runnable = prepared.filter(
			(item): item is RunnableToolCall => item.kind === "runnable",
		);
		const executedById = new Map<string, ExecutedLoopToolCall>();
		const parallel = this.shouldExecuteParallel(runnable);

		const executed = parallel
			? await Promise.all(
					runnable.map((item) => this.executePreparedToolCall(item)),
				)
			: await this.executePreparedToolCallsSequentially(runnable);
		for (const item of executed) executedById.set(item.call.id, item);

		// Track terminate hints: the batch terminates only when ALL tools
		// in it set terminate=true (Pi-style early termination).
		const allPrepareFinalized: Array<{
			item: PreparedLoopToolCall;
			executedItem: ExecutedLoopToolCall | undefined;
			terminate: boolean;
		}> = [];

		for (const item of prepared) {
			const executedItem =
				item.kind === "final" ? item : executedById.get(item.call.id);
			if (!executedItem) continue;
			const terminate = await this.finalizeToolCall(
				executedItem,
				transcriptPath,
				hookBasePayload,
			);
			allPrepareFinalized.push({ item, executedItem, terminate });
		}

		// Set batchTerminate when ALL finalized tools set it.
		if (allPrepareFinalized.length > 0) {
			this.batchTerminate = allPrepareFinalized.every((f) => f.terminate);
		}
	}

	private async prepareLoopToolCall(
		toolCall: ToolCall,
		turnId: string,
		hookBasePayload: Record<string, unknown>,
	): Promise<PreparedLoopToolCall> {
		const prepared = this.toolRegistry.prepare(toolCall);
		let toolInput = prepared.args;
		let activeToolCall = prepared.call;

		this.emitEvent({
			type: "tool_call_start",
			toolName: activeToolCall.name,
			toolCallId: activeToolCall.id,
			args: activeToolCall.arguments,
		});

		if (prepared.error) {
			this.emitEvent({
				type: "repair_nudge",
				turnId,
				repairStage: "prepare_arguments",
				toolName: toolCall.name,
				message: prepared.error,
			});
			return {
				kind: "final",
				call: activeToolCall,
				args: toolInput,
				result: prepared.error,
				isError: true,
			};
		}

		const before = await this.runBeforeToolCall(activeToolCall, toolInput);
		if (before?.content !== undefined) {
			return {
				kind: "final",
				call: activeToolCall,
				args: toolInput,
				result: before.content,
				isError: before.isError ?? false,
			};
		}
		if (before?.args !== undefined) {
			toolInput = before.args;
			activeToolCall = {
				...toolCall,
				arguments: JSON.stringify(before.args),
			};
		}

		await this.runHookSafely("PreToolUse", {
			...hookBasePayload,
			matcher_value: this.hookMatcherValue(activeToolCall.name),
			tool_name: activeToolCall.name,
			tool_input: toolInput,
		});

		return { kind: "runnable", call: activeToolCall, args: toolInput };
	}

	private async executePreparedToolCallsSequentially(
		calls: RunnableToolCall[],
	): Promise<ExecutedLoopToolCall[]> {
		const out: ExecutedLoopToolCall[] = [];
		for (const call of calls) {
			if (this.signal?.aborted) break;
			out.push(await this.executePreparedToolCall(call));
		}
		return out;
	}

	private async executePreparedToolCall(
		prepared: RunnableToolCall,
	): Promise<ExecutedLoopToolCall> {
		// ── Cache hit — skip execution ─────────────────────────────────────
		const cached = this.toolCache.get(
			prepared.call.name,
			prepared.call.arguments,
		);
		if (cached) {
			return {
				call: prepared.call,
				args: prepared.args,
				result: cached.result,
				isError: cached.isError,
			};
		}

		const { content: result, details } = await this.toolRegistry.execute(
			prepared.call,
			{
				signal: this.signal,
				onUpdate: (partialResult) => {
					this.emitEvent({
						type: "tool_call_update",
						toolName: prepared.call.name,
						toolCallId: prepared.call.id,
						partialResult,
					});
				},
			},
			prepared.args,
		);
		// ── Cache miss — store result (only successful) ────────────────────
		const isError = result.startsWith("Error:");
		this.toolCache.put(
			prepared.call.name,
			prepared.call.arguments,
			result,
			isError,
		);
		return {
			call: prepared.call,
			args: prepared.args,
			result,
			isError,
			details,
		};
	}

	private async finalizeToolCall(
		executed: ExecutedLoopToolCall,
		transcriptPath: string,
		hookBasePayload: Record<string, unknown>,
	): Promise<boolean> {
		let { result, isError } = executed;
		let terminate = false;
		const after = await this.runAfterToolCall(
			executed.call,
			executed.args,
			result,
			isError,
		);
		if (after) {
			if (after.content !== undefined) result = after.content;
			if (after.isError !== undefined) isError = after.isError;
			if (after.terminate) terminate = true;
		}

		this.emitEvent({
			type: "tool_call_end",
			toolName: executed.call.name,
			toolCallId: executed.call.id,
			result,
			isError,
			details: executed.details,
		});

		await this.recordToolResult(
			transcriptPath,
			hookBasePayload,
			executed.call,
			executed.args,
			result,
			isError,
		);
		return terminate;
	}

	private shouldExecuteParallel(calls: RunnableToolCall[]): boolean {
		if ((this.config.toolExecution ?? "parallel") !== "parallel") return false;
		return calls.every(
			(call) =>
				this.toolRegistry.get(call.call.name)?.executionMode === "parallel",
		);
	}

	private async recordToolResult(
		transcriptPath: string,
		hookBasePayload: Record<string, unknown>,
		toolCall: ToolCall,
		toolInput: Record<string, unknown>,
		result: string,
		isError: boolean,
	): Promise<void> {
		this._messages.push(
			createToolResultMessage(toolCall.id, toolCall.name, result, isError),
		);

		this.appendTranscript(transcriptPath, {
			type: "toolResult",
			timestamp: new Date().toISOString(),
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			content: [{ type: "text", text: result }],
			isError,
		});

		await this.runHookSafely("PostToolUse", {
			...hookBasePayload,
			matcher_value: this.hookMatcherValue(toolCall.name),
			tool_name: toolCall.name,
			tool_input: toolInput,
			tool_response: result,
		});
	}

	private appendInjectedMessages(
		transcriptPath: string,
		messages: Message[],
	): void {
		for (const message of messages) {
			this._messages.push(message);
			if (message.role !== "user") continue;
			this.appendTranscript(transcriptPath, {
				type: "user",
				timestamp: new Date().toISOString(),
				message: { role: "user", content: message.content || "" },
			});
		}
	}

	private async runHookSafely(
		eventType: string,
		payload: Record<string, unknown>,
	): Promise<void> {
		if (!this.hooksEnabled()) return;
		try {
			await runHookEvent(eventType, payload);
		} catch {
			// Hook failures should not break the agent turn.
		}
	}

	private appendTranscript(
		transcriptPath: string,
		entry: Record<string, unknown>,
	): void {
		if (!transcriptPath) return;
		try {
			mkdirSync(dirname(transcriptPath), { recursive: true });
			appendFileSync(transcriptPath, `${JSON.stringify(entry)}\n`, "utf8");
		} catch {
			// Transcript persistence is best-effort for hook integrations.
		}
	}

	private async _sleep(ms: number): Promise<void> {
		// Respect abort signal during backoff sleep.
		if (this.signal?.aborted) return;
		await new Promise<void>((resolve, reject) => {
			const timer = setTimeout(resolve, ms);
			this.signal?.addEventListener(
				"abort",
				() => {
					clearTimeout(timer);
					this.emitEvent({
						type: "error",
						message: "Retry cancelled by abort",
					});
					reject(new Error("Retry cancelled"));
				},
				{ once: true },
			);
		});
	}

	// ── Model cycling ─────────────────────────────────────────────────
	// Pi-style: cycle through configured models (forward or backward).
	// Creates a new backend with the selected model. Emits `model_select`.

	/** Build the model list — primary + alternates. */
	private getModelList(): string[] {
		const models = this.config.models;
		if (!models || models.length === 0) return [this.config.model];
		// Deduplicate while preserving order; primary is always first.
		const seen = new Set<string>();
		const list: string[] = [];
		for (const m of [this.config.model, ...models]) {
			if (!seen.has(m)) {
				seen.add(m);
				list.push(m);
			}
		}
		return list;
	}

	/** Current model index in the model list. */
	private _modelIndex = 0;
	private get modelIndex(): number {
		const list = this.getModelList();
		if (this._modelIndex >= list.length) this._modelIndex = 0;
		return this._modelIndex;
	}
	private get currentModel(): string {
		return this.getModelList()[this.modelIndex] ?? this.config.model;
	}

	/** Cycle to the next model (forward). Returns the new model name. */
	cycleModel(direction: "forward" | "backward" = "forward"): string {
		const list = this.getModelList();
		if (list.length <= 1) return this.config.model;

		const step = direction === "forward" ? 1 : -1;
		this._modelIndex = (this._modelIndex + step + list.length) % list.length;

		// Swap backend model via the backend's own clone method.
		const newModel = list[this._modelIndex];
		this.backend = this.backend.withModel(newModel);

		this.emitEvent({
			type: "model_select",
			model: newModel,
			index: this._modelIndex,
		});
		return newModel;
	}

	/** Get the current model name (for TUI status bar). */
	getModel(): string {
		return this.currentModel;
	}

	/** Get all available models (for TUI status bar). */
	getModels(): string[] {
		return this.getModelList();
	}

	// Build the Claude-Code hook matcher value for a tool: its own name plus any
	// aliases declared on the tool definition (e.g. bash → "bash|Bash"). Aliases
	// live on the tool, not in a loop-level lookup table.
	private hookMatcherValue(toolName: string): string {
		const aliases = this.toolRegistry.get(toolName)?.hookAliases ?? [];
		return [toolName, ...aliases].join("|");
	}
}

function envNumber(name: string): number | undefined {
	const raw = process.env[name];
	if (!raw) return undefined;
	const value = Number(raw);
	return Number.isFinite(value) ? value : undefined;
}

// Resolve an error to a BackendErrorCategory. The backend already classifies
// failures at its boundary (BackendError); this only adds a string-matching
// fallback for errors thrown elsewhere (e.g. a custom backend that throws plain
// Errors, or a wrapped provider SDK error).
function classifyLoopError(error: Error): BackendErrorCategory {
	if (error instanceof BackendError) return error.category;
	const message = `${error.name || ""} ${error.message || ""}`.toLowerCase();
	const contextFull = [
		"context full",
		"context window",
		"exceeds context",
		"exceed context",
		"maximum context",
		"max context",
		"prompt too long",
		"too many tokens",
		"tokens exceed",
		"context size",
		"n_ctx",
	].some((p) => message.includes(p));
	if (contextFull) return "context_full";
	if (/\b429\b/.test(message)) return "rate_limit";
	if (/\b(500|502|503|504)\b/.test(message)) return "transient";
	const network = [
		"econnrefused",
		"econnreset",
		"etimedout",
		"eai-again",
		"socket hang up",
		"connection refused",
		"connection reset",
		"connection timeout",
		"network error",
		"fetch failed",
	].some((p) => message.includes(p));
	if (network) return "transient";
	if (/\b4\d\d\b/.test(message)) return "client";
	return "unknown";
}
