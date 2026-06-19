// ── Agent loop ─────────────────────────────────────────────────────────────────────
// Main ReAct-style loop for the agent. Mirrors Python AgentLoop.

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { withTimeout } from "../tools/shared/async-utils.ts";
import {
	BackendError,
	type BackendErrorCategory,
	type LLMBackend,
} from "./backend.ts";
import { buildBuiltinHooks, composeHooks } from "../hooks/builtin-hooks.ts";
import { createDefaultTools } from "../tools/shared/default-tools.ts";
import { createEventEmitter, type EventEmitter } from "./events.ts";
import {
	convertToChatFormat,
	createAssistantMessage,
	createSystemMessage,
	createToolResultMessage,
	createUserMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "./messages.ts";
import { recoverFromContextFull } from "../compaction/index.ts";
import {
	AgentErrorType,
	type AgentConfig,
	type AgentError,
	type AgentEvent,
	type AgentLoopHooks,
	type AgentMessage,
	type CompactableMessage,
	type EventHandler,
	type Message,
	type StopReason,
	type ToolCall,
} from "./types.ts";
import { parseToolCalls, parseToolInput } from "../tools/shared/parser.ts";
import { createPluginHookLayer } from "../hooks/plugin-hooks.ts";
import { LoopDetector, type TurnSignature } from "./loop-detector.ts";
import { resetTaskStatus } from "../tools/skills/task-status.ts";
import { ToolResultCache } from "./tool-cache.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import {
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	stripAcceptanceReport,
	type ResolvedAcceptance,
	type AcceptanceReport,
	type AcceptanceLedger,
} from "./acceptance-contract.ts";

// Default max outer-loop continuations (follow-up driven). Bounds runaway
// auto-continuation when follow-up messages keep arriving.
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
	// Raw data for loop detection — built once in runTurnInner, consumed in the
	// main loop. Avoids duplicating buildTurnSignature across every return path.
	loopSignatureData?: {
		assistantContent: string;
		toolCalls: ToolCall[];
		toolResults: Message[];
	};
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
	// Sum of provider-reported total tokens across all LLM calls this run.
	usageTokens: number;
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
	/** Per-turn snapshot of harness-level stream options. Updated by harness. */
	streamOptions: import("./types.ts").AgentHarnessStreamOptions = {};
	private _retryAttempt = 0;
	private toolCache: ToolResultCache;
	// Track terminate hint from afterToolCall — when ALL tools in a batch
	// set it, the loop stops after that batch (Pi-style early termination).
	private batchTerminate = false;
	// Mid-stream steering interrupt: the in-flight provider call's controller,
	// and whether the current abort is an interrupt (keep partial output and
	// continue) rather than a cancellation.
	private currentCallController: AbortController | null = null;
	private interruptRequested = false;
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
		usageTokens: 0,
	};
	// Provider-reported tokens consumed this run (for maxTotalTokens budget).
	private _runUsageTokens = 0;
	private loopDetector: LoopDetector;
	// Loop detection: when enabled, watches for repetitive turn patterns and
	// injects a recovery message on first detection. Default OFF — matching
	// pi's trust-model approach. The injected messages often cause more loops.
	private get loopDetectionEnabled(): boolean {
		return this.config.loopDetectionEnabled === true;
	}
	private loopDetectionAttempted = false;

	/** Ledger for acceptance contract self-review/verification. */
	acceptanceLedger: AcceptanceLedger | null = null;

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
		this.iterationCount = 0;
		this._messages = [];
		this.hooks = this.config.hooks || {};
		this.streamOptions = options.config.streamOptions || {};
		this.emitter = createEventEmitter();
		this.toolCache = new ToolResultCache(
			this.config.cacheSize ?? 1000,
			this.config.cacheTtlMs ?? 30_000,
		);

		// Set up tool registry
		this.toolRegistry = new ToolRegistry({
			cwd: this.cwd,
			onQuestionRequest: this.config.onQuestionRequest,
		});
		this.toolRegistry.registerMany(
			this.config.tools?.length ? this.config.tools : createDefaultTools(),
		);

		this.loopDetector = new LoopDetector({
			maxHistory: this.config.loopDetectionWindow ?? 10,
			exactRepeatWindow: this.config.loopDetectionWindow ?? 3,
			degenerateWindow: this.config.degenerateLoopThreshold ?? 4,
			stagnationWindow: this.config.stagnationThreshold ?? 5,
		});
		this.wrapOnEvent();
	}

	// Capture the config's raw onEvent and replace it with a wrapper that
	// stamps the envelope (seq/ts), appends to the JSONL event log, and fans
	// every event out through the emitter before the raw handler. Re-run on
	// config refresh so a reused loop keeps emitting through the emitter.
	private eventSeq = 0;
	private wrapOnEvent(): void {
		this.onEvent = this.config.onEvent;
		this.config.onEvent = (event: AgentEvent) => {
			// Stamp only once: a subagent's events arrive pre-stamped by the
			// child loop and keep the child's ordering.
			if (event.seq === undefined) {
				event.seq = ++this.eventSeq;
				event.ts = Date.now();
			}
			this.appendEventLog(event);
			this.emitter.emit(event);
			this.onEvent?.(event);
		};
	}

	// Best-effort JSONL event log for replay/debugging. message_update is
	// skipped — it re-serializes the whole partial message every delta and the
	// stream is reconstructible from text_delta events.
	private appendEventLog(event: AgentEvent): void {
		const path = this.config.eventLogPath;
		if (!path || event.type === "message_update") return;
		try {
			appendFileSync(path, `${JSON.stringify(event)}\n`, "utf8");
		} catch {
			// Event logging must never break a turn.
		}
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
	 * Refresh the conversation to continue from on the next run() (harness loop
	 * reuse). Without this, run() rebuilds from the construction-time
	 * initialMessages every prompt, replaying only the first prompt's history and
	 * dropping every turn since. Pass the harness's live history before each
	 * reused run.
	 */
	updateInitialMessages(messages: Message[] | undefined): void {
		this.initialMessages = messages?.length ? messages : undefined;
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
		// Initialize with system prompt
		const systemPrompt =
			this.config.systemPrompt || "You are a helpful assistant.";
		// Resolve once and keep it stable across reused runs: a reused loop must
		// not mint a fresh id each prompt or hook session continuity breaks.
		const sessionId =
			this.config.hookSessionId || this._sessionId || `tui_${Date.now()}`;
		this._sessionId = sessionId;
		const transcriptPath = this.config.hookTranscriptPath || "";
		const pluginHooks = createPluginHookLayer({
			enabled: this.hooksEnabled(),
			sessionId,
			transcriptPath,
			cwd: this.cwd,
			getMatcherValue: (toolName) => this.hookMatcherValue(toolName),
			onHookPermissionDenied: (toolCall) => {
				this.emitEvent({
					type: "tool_permission_decision",
					toolName: toolCall.name,
					toolCallId: toolCall.id,
					decision: "deny",
					source: "hook",
				});
			},
		});

		// Compose built-in safeguard hooks (guards, budget stop, proactive
		// compaction), OpenClaude/Claude plugin hooks, and user-supplied hooks.
		// Built-in/plugin state is per-run, so build here rather than in the
		// constructor.
		const builtin = buildBuiltinHooks({
			config: this.config,
			contextWindowTokens: () => this.contextWindowTokens(),
			toolDefs: () => this.toolRegistry.toToolDefinitions(),
			loopDetector: this.loopDetector,
		});
		this.hooks = composeHooks(
			[
				{ source: "builtin", hooks: builtin },
				{ source: "harness-queues", hooks: this.config.internalHooks },
				{ source: "user", hooks: this.config.hooks },
				{ source: "plugins", hooks: pluginHooks.hooks },
			],
			(error, event) => {
				this.emitEvent({
					type: "error",
					message: `${event} hook failed: ${error.message}`,
				});
			},
			this.config.onHookEvent,
		);

		const hookMessages = await pluginHooks.userPromptMessages(userMessage);
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
		this._retryAttempt = 0;
		this._lastUsageTokens = undefined;
		this._runUsageTokens = 0;
		this.loopDetectionAttempted = false;
		// New run = new task: clear any task_status left by the previous one so
		// the continuation logic doesn't honour a stale declaration.
		resetTaskStatus();
		// Reset loop detector for each independent run.
		this.loopDetector.reset();

		this.emitEvent({ type: "agent_start" });
		this.emitEvent({ type: "phase", phase: "idle" });

		// Outer loop: follow-up continuation. After each inner-loop run (no tool
		// calls), check for queued follow-ups. If any exist and we haven't
		// exceeded the safety cap, inject them and re-enter the inner loop.
		// This mirrors Pi's inner/outer pattern: follow-up messages drive a new
		// turn without a hard continuation counter in the turn itself.
		let continuationSafetyCount = 0;
		let unproductiveTurnCount = 0;
		const maxContinuations =
			this.config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
		while (true) {
			// Safety caps — abort signal and iteration limit apply to the whole
			// run, not per-continuation.
			if (this.signal?.aborted) {
				this.emitEvent({ type: "error", message: "Operation aborted" });
				break;
			}
			if (this.iterationCount >= this.maxIterations) {
				this.emitEvent({
					type: "max_iterations",
					iterations: this.iterationCount,
					limit: this.maxIterations,
				});
				break;
			}
			this.iterationCount++;
			const turnId = `turn_${this.iterationCount}`;
			const outcome = await this.runTurn(turnId, transcriptPath);

			// Detect repetitive turn patterns (same content + same tool calls/results).
			if (this.loopDetectionEnabled && outcome.loopSignatureData) {
				const sig = this.buildTurnSignature(
					outcome.loopSignatureData.assistantContent,
					outcome.loopSignatureData.toolCalls,
					outcome.loopSignatureData.toolResults,
				);
				if (
					this.loopDetector.recordAndDetect(sig.assistantContent, sig.toolCalls)
				) {
					outcome.stopReason = "loop_detected";
				}
			}

			if (outcome.stopReason === "loop_detected") {
				// Loop recovery: before hard-stopping, inject a recovery message
				// and give the agent one more chance. This catches recoverable
				// loops (e.g. the agent is circling on a subtask it can't resolve)
				// instead of terminating immediately.
				if (this.loopDetectionEnabled && !this.loopDetectionAttempted) {
					this.loopDetectionAttempted = true;
					this._turnMetrics.continuations++;
					const diagnostic = this.loopDetector.getLoopDiagnostic();
					const recoveryMsg = diagnostic
						? (() => {
								// Build loop-type-specific recovery guidance from the diagnostic.
								let action: string;
								if (diagnostic.startsWith("Exact repeat")) {
									// Exact repeat: the same input produces the same output every time.
									// The agent needs a fundamentally different approach, not just a
									// tweak of the current one.
									action =
										"You are stuck in an exact-repeat loop — the same input produces the " +
										"same output every time. Changing arguments within the same approach " +
										"will not help. You must switch to a completely different strategy: " +
										"use a different tool, re-read the starting context from scratch, or " +
										"break the task into a smaller sub-problem you haven't attempted yet.";
								} else if (diagnostic.startsWith("Degenerate")) {
									// Degenerate loop: same tools, same results, varying args.
									// The agent is producing work but not getting anywhere.
									action =
										"You are in a degenerate loop — you call the same tools and keep " +
										"getting the same results, even though your arguments change slightly. " +
										"Your current tool+strategy combo cannot solve this. Try a different " +
										"tool entirely, or step back and reconsider whether the approach itself " +
										"is correct for this task.";
								} else if (diagnostic.startsWith("Stagnation")) {
									// Stagnation: no new signal across the window.
									// The agent has exhausted known patterns.
									action =
										"You are stagnating — you've exhausted all the (tool, result) shapes " +
										"you know and are cycling through them without introducing anything new. " +
										"At this point you should: (1) re-read your original task to check you " +
										"understand what's actually needed, (2) pick a direction you haven't " +
										"explored, or (3) acknowledge what you've tried and explain why the task " +
										"may be incomplete.";
								} else {
									// Fallback for unexpected diagnostic format.
									action =
										"You are repeating the same pattern without progress. Stop and look " +
										"at your last few turns — they're all producing the same result. " +
										"Pick ONE thing you haven't tried yet and do it. If you genuinely " +
										"have no new ideas, say why and stop.";
								}
								return `Loop detected: ${diagnostic} ${action}`;
							})()
						: "Loop detected: you are repeating the same pattern without progress. " +
							"Stop and look at your last few turns — they're all producing the same result. " +
							"Pick ONE thing you haven't tried yet and do it. " +
							"If you genuinely have no new ideas, say why and stop.";
					this.appendInjectedMessages(transcriptPath, [
						createUserMessage(recoveryMsg),
					]);
					this.emitEvent({
						type: "loop_detected",
						message: "Loop detected — recovery message injected.",
						attempt: 1,
					});
					// Continue the loop with the recovery message in context.
					continue;
				}
				// No recovery or already tried: hard stop.
				this.emitEvent({
					type: this.loopDetectionAttempted ? "loop_detected" : "error",
					message: this.loopDetectionAttempted
						? "Loop detected: recovery attempt did not resolve the pattern. Terminating run."
						: "Loop detected: The agent is repeating itself. Terminating run.",
				});
				break;
			}

			// Track unproductive turns (no tool calls) to bound runaway models.
			// Productive turns reset the counter.
			unproductiveTurnCount = outcome.productive
				? 0
				: unproductiveTurnCount + 1;
			if (unproductiveTurnCount >= this.maxIterations) {
				this.emitEvent({
					type: "max_iterations",
					iterations: this.iterationCount,
					limit: this.maxIterations,
				});
				break;
			}

			// Inner loop continuation: the turn called tools — re-enter the inner
			// loop to process results. No follow-up check here.
			if (outcome.continue) {
				// Run token budget between turns.
				const budget = this.config.maxTotalTokens;
				if (budget && budget > 0 && this._runUsageTokens >= budget) {
					this.emitEvent({
						type: "budget_exhausted",
						usedTokens: this._runUsageTokens,
						limitTokens: budget,
					});
					break;
				}
				continue;
			}

			// Inner loop exited (no tool calls). Check for follow-up messages.
			// Pi-style: follow-ups are injected into the context and re-enter the
			// inner loop. The outer loop owns the safety cap.
			const followUps = await this.runGetFollowUpMessages(
				outcome.message?.content ?? "",
				outcome.stopReason,
			);
			if (followUps.length && continuationSafetyCount < maxContinuations) {
				continuationSafetyCount++;
				this._turnMetrics.continuations++;
				this.appendInjectedMessages(transcriptPath, followUps);
				continue; // re-enter inner loop
			}

			break;
		}

		await pluginHooks.finalStop();
		this.emitEvent({ type: "phase", phase: "idle" });
		this.emitEvent({ type: "agent_end", messages: this._messages });

		// Run acceptance finalization if configured
		await this.runAcceptanceFinalization();

		return this._messages;
	}

	private async runAcceptanceFinalization(): Promise<void> {
		const config = this.config.acceptance;
		if (!config) {
			this.acceptanceLedger = { status: "not-required" };
			return;
		}

		const resolved = resolveEffectiveAcceptance({ explicit: config });
		if (!shouldRunAcceptanceFinalization(resolved)) {
			this.acceptanceLedger = { status: "not-required" };
			return;
		}

		// Get the last assistant message content for report parsing
		let lastContent = "";
		for (let i = this._messages.length - 1; i >= 0; i--) {
			const msg = this._messages[i];
			if (msg.role === "assistant" && msg.content) {
				lastContent = msg.content;
				break;
			}
		}

		const { report, error } = parseAcceptanceReport(lastContent);
		const verification: AcceptanceLedger["verification"] = [];

		// Run verification commands if configured
		if (resolved.verify.length > 0) {
			for (const v of resolved.verify) {
				try {
					const { exec } = await import("node:child_process");
					const result = await new Promise<{
						exitCode: number;
						stdout: string;
						stderr: string;
					}>((resolve) => {
						exec(
							v.command,
							{ cwd: v.cwd, timeout: v.timeoutMs || 30000 },
							(err, stdout, stderr) => {
								resolve({
									exitCode: err ? 1 : 0,
									stdout: stdout?.toString() || "",
									stderr: stderr?.toString() || "",
								});
							},
						);
					});
					verification.push({
						command: v.command,
						result: result.exitCode === 0 ? "passed" : "failed",
						summary: result.stdout?.trim().slice(0, 200),
					});
				} catch {
					verification.push({
						command: v.command,
						result: "failed",
						summary: "command execution error",
					});
				}
			}
		}

		if (error || !report) {
			this.acceptanceLedger = {
				status: lastContent ? "failed" : "timeout",
				config: resolved,
				verification,
			};
		} else {
			const hasFailure = report.criteriaSatisfied?.some(
				(c) => c.status === "failed",
			);
			const verifyFailed = verification.some((v) => v.result === "failed");
			this.acceptanceLedger = {
				status: hasFailure || verifyFailed ? "failed" : "passed",
				report,
				config: resolved,
				verification,
			};
		}
	}

	/**
	 * Execute one turn: LLM call → tool execution → follow-up check.
	 * Returns true when the loop should continue, false when the turn
	 * completed (no more tools or follow-ups) or the turn was aborted.
	 */
	private async runTurn(
		turnId: string,
		transcriptPath: string,
	): Promise<TurnOutcome> {
		this.emitEvent({ type: "turn_start", turnId });
		this.emitEvent({ type: "phase", phase: "thinking" });
		// runTurnInner reports whether the loop should continue and whether the
		// turn was productive. When it stops, turn_end is emitted once here
		// rather than at every early-return.
		const outcome = await this.runTurnInner(
			turnId,
			transcriptPath,
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
	): Promise<TurnOutcome> {
		// Drain steering messages respecting queue mode
		const steeringMessages = await this.runGetSteeringMessages();
		if (steeringMessages.length) {
			this.appendInjectedMessages(transcriptPath, steeringMessages);
		}

		const toolDefs = this.toolRegistry.toToolDefinitions();

		// Emit context state at turn start so the UI always has current token info,
		// not just after the first compaction or provider request.
		this.emitContextUpdate(toolDefs);

		// Get LLM response with timeout, auto-retry, and context-full compaction.
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
				loopSignatureData: {
					assistantContent: text,
					toolCalls: [],
					toolResults: [],
				},
			};
		}

		// Empty response (no content, no tools) is ambiguous: a real stop or a
		// transient empty completion. Retry once; if still empty, nudge-and-
		// continue or stop.
		if (!assistantContent && assistantToolCalls.length === 0) {
			const retryResult = await this.retryEmptyCall(turnId, toolDefs);
			if (retryResult.content || retryResult.toolCalls.length > 0) {
				result = retryResult;
				assistantContent = result.content;
				assistantToolCalls = result.toolCalls;
			} else {
				return await this.handleEmptyResponseAfterRetry(transcriptPath);
			}
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
					loopSignatureData: {
						assistantContent,
						toolCalls: assistantToolCalls,
						toolResults,
					},
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
					loopSignatureData: {
						assistantContent,
						toolCalls: assistantToolCalls,
						toolResults,
					},
				};
			}
		}

		// prepareNextTurn / shouldStopAfterTurn contract hooks.
		await this.runPrepareNextTurn(hadToolCalls);
		if (await this.runShouldStopAfterTurn(hadToolCalls)) {
			return {
				continue: false,
				productive: hadToolCalls,
				stopReason: turnStopReason,
				message: assistantMessage,
				toolResults,
				loopSignatureData: {
					assistantContent,
					toolCalls: assistantToolCalls,
					toolResults,
				},
			};
		}

		// Continue loop: model called tools — re-enter inner loop to process
		// results. No follow-up check here (handled by outer loop).
		// turn_end stays open — the TUI sees one continuous turn.
		if (hadToolCalls) {
			return {
				continue: true,
				productive: true,
				loopSignatureData: {
					assistantContent,
					toolCalls: assistantToolCalls,
					toolResults,
				},
			};
		}

		// Truly done — no tools, no follow-ups (follow-up check moved to outer
		// loop so it doesn't count as a continuation).
		return {
			continue: false,
			productive: hadToolCalls,
			stopReason: turnStopReason,
			message: assistantMessage,
			toolResults,
			loopSignatureData: {
				assistantContent,
				toolCalls: assistantToolCalls,
				toolResults,
			},
		};
	}

	/**
	 * Retry an LLM call that produced no output. Returns the result so the
	 * caller can continue the turn, or a terminal TurnOutcome when the retry
	 * also produced nothing and no further continuations are available.
	 */
	private async retryEmptyCall(
		turnId: string,
		toolDefs: Record<string, unknown>[],
	): Promise<LLMCallResult> {
		return this.callLLMGuarded(turnId, toolDefs);
	}

	/**
	 * Decide what to do when the LLM response is empty (no content, no tool
	 * calls) after one retry. Injects a nudge message and returns a
	 * nudge-and-continue outcome when continuations remain; otherwise returns
	 * a clean stop outcome.
	 */
	private async handleEmptyResponseAfterRetry(
		transcriptPath: string,
	): Promise<TurnOutcome> {
		// Safety cap on empty-response nudges (same outer-loop cap).
		const maxContinuations =
			this.config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
		if (this._turnMetrics.continuations < maxContinuations) {
			this._turnMetrics.continuations++;
			this.appendInjectedMessages(transcriptPath, [
				createUserMessage(
					"Your last response was empty — the model produced no content or tool calls. " +
						"This can happen with transient errors or a stalled generation. " +
						"If the task is complete, say so explicitly and stop. " +
						"Otherwise, pick one specific action from your remaining work and execute it with a tool.",
				),
			]);
			return { continue: true, productive: false };
		}
		return { continue: false, productive: false, stopReason: "stop" };
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
		// Per-call controller linked to the run-level signal. On timeout we abort
		// it so the in-flight provider request is cancelled rather than left
		// streaming into a turn the loop has already abandoned. An external abort
		// (this.signal) is forwarded to it too.
		const callController = new AbortController();
		const forwardAbort = () => callController.abort();
		if (this.signal) {
			if (this.signal.aborted) callController.abort();
			else this.signal.addEventListener("abort", forwardAbort, { once: true });
		}
		this.currentCallController = callController;
		try {
			return await withTimeout(
				this.callLLM(turnId, toolDefs, callController.signal),
				this.turnTimeoutMs,
				() => callController.abort(),
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
			this.currentCallController = null;
			this.signal?.removeEventListener("abort", forwardAbort);
			this._turnMetrics.totalLlmTimeMs += Date.now() - llmStart;
		}
	}

	/**
	 * Interrupt the in-flight LLM call (mid-stream steering): the partial
	 * assistant text streamed so far is kept as the turn's message and the
	 * loop continues, draining steering at the next save point. Returns false
	 * when no call is in flight (nothing to interrupt — queued steering will
	 * drain normally).
	 */
	interruptTurn(): boolean {
		const controller = this.currentCallController;
		if (!controller) return false;
		this.interruptRequested = true;
		controller.abort();
		return true;
	}

	/**
	 * Call the LLM with context-full compaction retry and auto-retry on
	 * transient provider errors. Returns the assistant content and tool calls.
	 * `signal` is the per-call signal (run signal + turn timeout) passed to the
	 * backend so a timeout actually cancels the request.
	 */
	private async callLLM(
		turnId: string,
		toolDefs: Record<string, unknown>[],
		signal: AbortSignal,
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
					signal,
					thinkingLevel: this.config.thinkingLevel,
					headers: reqPatch?.headers && Object.fromEntries(
						Object.entries(reqPatch.headers).filter(([, v]) => v !== undefined),
					),
					transformPayload: this.hooks.beforeProviderPayload
						? (payload) => this.applyProviderPayload(payload)
						: undefined,
					maxRetries: reqPatch?.maxRetries ?? undefined,
					cacheRetention: reqPatch?.cacheRetention,
					metadata: reqPatch?.metadata,
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
							this.emitEvent({ type: "thinking_delta", turnId, delta });
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
				// Prefer the provider's real token count over the local estimate for
				// context reporting; cleared each turn so a missing usage chunk falls
				// back to the estimate rather than a stale value.
				this._lastUsageTokens = response.usage?.totalTokens;
				const consumed = response.usage?.totalTokens ?? 0;
				this._runUsageTokens += consumed;
				this._turnMetrics.usageTokens += consumed;
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
				// Notify observers of the raw provider response (read-only).
				if (this.hooks.afterProviderResponse) {
					await this.hooks.afterProviderResponse({
						model: this.getModel(),
						content: assistantContent,
						toolCallCount: assistantToolCalls.length,
						stopReason,
						usageTokens: response.usage?.totalTokens,
						iteration: this.iterationCount,
					});
				}
				// This request was an auto-retry and it actually succeeded.
				if (this._retryAttempt > 0) {
					this.emitEvent({
						type: "auto_retry_end",
						attempt: this._retryAttempt,
						success: true,
					});
				}
			} catch (e: unknown) {
				const error = e as Error;

				// Mid-stream steering interrupt: not an error. Keep the partial
				// assistant text as this turn's message; the steering message drains
				// at the next save point and the loop continues. A real run-level
				// abort takes precedence and flows through the normal abort path.
				if (this.interruptRequested && !this.signal?.aborted) {
					this.interruptRequested = false;
					assistantToolCalls = [];
					stopReason = "stop";
					llmSuccess = true;
					break;
				}
				this.interruptRequested = false;

				// The failed request was itself an auto-retry: close it out as a
				// failure before deciding whether to retry again or give up.
				if (this._retryAttempt > 0) {
					this.emitEvent({
						type: "auto_retry_end",
						attempt: this._retryAttempt,
						success: false,
					});
				}

				// Prefer the backend's typed category; fall back to message-string
				// matching for errors thrown outside the backend boundary.
				const category = classifyLoopError(error);

				// 1. Context-full → compact once and retry.
				if (!compactionAttempted && category === "context_full") {
					compactionAttempted = true;
					this._turnMetrics.compactions++;
					const ctxWindow = this.contextWindowTokens();
					if (ctxWindow) {
						const recovery = recoverFromContextFull({
							messages: this._messages as CompactableMessage[],
							contextWindowTokens: ctxWindow,
						});
						if (recovery.success) {
							this._messages = recovery.messages as Message[];
							this.emitEvent({
								type: "compaction",
								reason: "context_full",
								tokensBefore: recovery.tokensBefore,
								tokensAfter: recovery.tokensAfter,
							});
							this.emitContextUpdate(toolDefs, true);
							continue;
						}
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
						// Provider-requested delay (Retry-After) wins over exponential
						// backoff; ±20% jitter desynchronizes concurrent retries.
						const base =
							error instanceof BackendError && error.retryAfterMs !== undefined
								? error.retryAfterMs
								: this.retryBaseDelayMs * 2 ** (this._retryAttempt - 1);
						const delayMs = Math.round(base * (0.8 + Math.random() * 0.4));
						this.emitEvent({
							type: "auto_retry_start",
							attempt: this._retryAttempt,
							maxRetries: this.maxRetries,
							delayMs,
							error: error.message,
						});
						await this._sleep(delayMs);
						// No auto_retry_end here: the retry hasn't happened yet. The
						// outcome event fires when the retried request succeeds (success
						// path below) or fails again (top of this catch).
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
	// Provider-reported total tokens from the last LLM response, when available.
	// Used by emitContextUpdate in preference to the local estimate.
	private _lastUsageTokens?: number;

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
			// Provider usage is authoritative when the last response reported it;
			// otherwise fall back to the local payload estimate.
			tokens: this._lastUsageTokens ?? this.estimatePayloadTokens(tools),
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

	private async runPrepareNextTurn(hadToolCalls: boolean): Promise<void> {
		if (!this.hooks.prepareNextTurn) return;
		const out = await this.hooks.prepareNextTurn({
			messages: this._messages,
			iteration: this.iterationCount,
			hadToolCalls,
		});
		if (out?.messages) this._messages = out.messages;
	}

	private async runShouldStopAfterTurn(
		hadToolCalls: boolean,
	): Promise<boolean> {
		if (!this.hooks.shouldStopAfterTurn) return false;
		return (
			(await this.hooks.shouldStopAfterTurn({
				messages: this._messages,
				iteration: this.iterationCount,
				hadToolCalls,
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

	// Resolve per-request headers / timeout / metadata from beforeProviderRequest hooks.
	private async runBeforeProviderRequest(): Promise<
		| {
				headers?: Record<string, string | undefined>;
				timeoutMs?: number;
				maxRetries?: number;
				cacheRetention?: string;
				metadata?: Record<string, unknown>;
				transport?: string;
			}
			| undefined
	> {
		if (!this.hooks.beforeProviderRequest) return undefined;
		return (
			(await this.hooks.beforeProviderRequest({
				model: this.getModel(),
				sessionId: this._sessionId,
				iteration: this.iterationCount,
				streamOptions: this.streamOptions,
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
		stopReason?: StopReason,
	): Promise<Message[]> {
		const messages: Message[] = [];
		const maxContinuations =
			this.config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
		if (this.hooks.getFollowUpMessages) {
			const r = await this.hooks.getFollowUpMessages({
				messages: this._messages,
				iteration: this.iterationCount,
				assistantText,
				maxContinuations,
				stopReason,
			});
			if (r?.length) messages.push(...r);
		}

		return messages;
	}

	private async executeToolCalls(
		toolCalls: ToolCall[],
		turnId: string,
		transcriptPath: string,
	): Promise<void> {
		// Prepare every call up front. On abort, stop preparing but still emit a
		// final aborted result for each remaining call below — the assistant
		// message already carries all tool_calls, and every tool_call needs a
		// matching tool result or the next request is malformed (dangling calls).
		const prepared: PreparedLoopToolCall[] = [];
		let abortedDuringPrepare = false;
		for (const toolCall of toolCalls) {
			if (this.signal?.aborted) {
				this.emitEvent({ type: "error", message: "Operation aborted" });
				abortedDuringPrepare = true;
				break;
			}
			prepared.push(
				await this.prepareLoopToolCall(toolCall, turnId),
			);
		}
		// Calls never prepared (abort cut the loop short) still need a result.
		const preparedIds = new Set(prepared.map((p) => p.call.id));
		const unprepared = abortedDuringPrepare
			? toolCalls.filter((tc) => !preparedIds.has(tc.id))
			: [];

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
			// A runnable item with no executed entry was skipped by an abort mid-
			// batch; synthesize an aborted result so its tool_call is still paired.
			const executedItem =
				item.kind === "final"
					? item
					: (executedById.get(item.call.id) ??
						this.abortedToolResult(item.call, item.args));
			const terminate = await this.finalizeToolCall(
				executedItem,
				transcriptPath,
			);
			allPrepareFinalized.push({ item, executedItem, terminate });
		}

		// Pair any never-prepared calls (raw ToolCall, no args parsed) too.
		for (const call of unprepared) {
			await this.finalizeToolCall(
				this.abortedToolResult(call, {}),
				transcriptPath,
			);
		}

		// Set batchTerminate when ALL finalized tools set it.
		if (allPrepareFinalized.length > 0) {
			this.batchTerminate = allPrepareFinalized.every((f) => f.terminate);
		}
	}

	private async prepareLoopToolCall(
		toolCall: ToolCall,
		turnId: string,
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

		// Permission gate (modes + rules + interactive ask). Runs after
		// beforeToolCall so rules see the final, possibly-rewritten args.
		const permissionBlock = await this.checkPermissions(
			activeToolCall,
			toolInput,
		);
		if (permissionBlock) {
			return {
				kind: "final",
				call: activeToolCall,
				args: toolInput,
				result: permissionBlock,
				isError: true,
			};
		}

		return { kind: "runnable", call: activeToolCall, args: toolInput };
	}

	/**
	 * Evaluate the permission gate for a prepared tool call. Returns the error
	 * text to record as the tool result when the call is denied, or undefined
	 * when it may run. "ask" verdicts route to config.onPermissionRequest; with
	 * no handler installed they fail closed.
	 */
	private async checkPermissions(
		toolCall: ToolCall,
		args: Record<string, unknown>,
	): Promise<string | undefined> {
		const pm = this.config.permissions;
		if (!pm) return undefined;
		const verdict = pm.evaluate(
			toolCall,
			args,
			this.toolRegistry.get(toolCall.name),
		);
		if (verdict.decision === "allow") return undefined;
		if (verdict.decision === "deny") {
			this.emitEvent({
				type: "tool_permission_decision",
				toolName: toolCall.name,
				toolCallId: toolCall.id,
				decision: "deny",
				source: verdict.source,
			});
			return `Permission denied: ${verdict.reason ?? `${toolCall.name} is not allowed in ${pm.getMode()} mode`}.`;
		}

		// "ask" — interactive approval.
		this.emitEvent({
			type: "tool_permission_request",
			toolName: toolCall.name,
			toolCallId: toolCall.id,
			args: toolCall.arguments,
		});
		const handler = this.config.onPermissionRequest;
		if (!handler) {
			this.emitEvent({
				type: "tool_permission_decision",
				toolName: toolCall.name,
				toolCallId: toolCall.id,
				decision: "deny",
				source: "mode",
			});
			return (
				`Permission denied: ${toolCall.name} requires approval in ` +
				`${pm.getMode()} mode and no approval handler is installed.`
			);
		}
		let decision: "allow" | "deny" | "always";
		try {
			decision = await handler({
				toolName: toolCall.name,
				toolCallId: toolCall.id,
				args,
			});
		} catch {
			decision = "deny";
		}
		if (decision === "always") pm.addSessionAllow(toolCall.name);
		this.emitEvent({
			type: "tool_permission_decision",
			toolName: toolCall.name,
			toolCallId: toolCall.id,
			decision,
			source: "user",
		});
		if (decision === "deny") {
			return `Permission denied: the user declined this ${toolCall.name} call.`;
		}
		return undefined;
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
		// Only pure tools (cacheable=true) use the result cache. Most tools observe
		// mutable state the agent itself changes between calls (filesystem, git,
		// shell), so caching them would serve stale results — e.g. re-reading a
		// file the agent just edited.
		const cacheable = this.toolRegistry.get(prepared.call.name)?.cacheable;

		// ── Cache hit — skip execution ─────────────────────────────────────
		if (cacheable) {
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
		// ── Cache miss — store result (only successful, only cacheable) ────
		const isError = result.startsWith("Error:");
		if (cacheable) {
			this.toolCache.put(
				prepared.call.name,
				prepared.call.arguments,
				result,
				isError,
			);
		}
		return {
			call: prepared.call,
			args: prepared.args,
			result,
			isError,
			details,
		};
	}

	// A synthetic result for a tool call the loop never executed (abort mid-batch).
	// Keeps the assistant's tool_call paired with a tool result so the next
	// request is well-formed.
	private abortedToolResult(
		call: ToolCall,
		args: Record<string, unknown>,
	): ExecutedLoopToolCall {
		return {
			call,
			args,
			result: "Error: tool call aborted before execution.",
			isError: true,
		};
	}

	private async finalizeToolCall(
		executed: ExecutedLoopToolCall,
		transcriptPath: string,
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
			executed.call,
			result,
			isError,
		);
		return terminate;
	}

	private shouldExecuteParallel(calls: RunnableToolCall[]): boolean {
		// Global config must allow parallel.
		if ((this.config.toolExecution ?? "parallel") !== "parallel") return false;
		// Parallel by default; only tools that opt into "sequential" block it.
		return !calls.some(
			(call) =>
				this.toolRegistry.get(call.call.name)?.executionMode === "sequential",
		);
	}

	private async recordToolResult(
		transcriptPath: string,
		toolCall: ToolCall,
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

	// Build a TurnSignature from the assistant content and tool result messages.
	// This is used by the main loop to detect repetitive turn patterns.
	private buildTurnSignature(
		assistantContent: string,
		toolCalls: ToolCall[],
		toolResults: Message[],
	): TurnSignature {
		// Build a map from tool_call_id to result text.
		const resultMap = new Map<string, string>();
		for (const msg of toolResults) {
			if (msg.role !== "tool") continue;
			const id = msg.tool_call_id ?? "";
			resultMap.set(id, msg.content ?? "");
		}
		const signatures = toolCalls.map((tc) => {
			const result = resultMap.get(tc.id) ?? "";
			return {
				name: tc.name,
				args: tc.arguments,
				result: result.slice(0, 100), // truncate for signature stability
			};
		});
		return {
			assistantContent,
			toolCalls: signatures,
		};
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
