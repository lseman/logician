// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.
//
// Control flow mirrors earendil-works/pi's agent-loop.ts: an outer loop that
// continues while follow-up messages keep arriving, and an inner loop that
// processes one assistant turn plus its tool calls and any steering
// messages injected before the next turn. Budget/cache/intervention/
// acceptance/adaptive-mode/autonomous-continuation-policy — all real
// features in Logician's prior loop — have no equivalent in pi's loop and
// are not ported here; see the accompanying plan for what that drops.

import type {
	AgentConfig,
	AgentEventSink,
	AgentMessage,
	Message,
	Tool,
	ToolCall,
} from "../../types/index.ts";
import type { RunOutcomeStatus } from "../../types/types-messages.ts";
import {
	resolveAgentSettings,
	resolveExecutionPolicy,
} from "../env/agent-settings.ts";
import {
	assistantText,
	emitMessagePair,
	stopReasonFor,
	withSystemPrompt,
} from "../events.ts";
import {
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "../messages.ts";
import { ToolRegistry } from "../tools/registry.ts";
import type { LLMBackend } from "../utils/backend.ts";
import { processProviderResponse } from "../utils/provider-response.ts";
import {
	createProviderTurnState,
	requestAssistantTurn,
} from "../utils/provider-turn.ts";
import { executeToolBatch } from "../utils/tool-batch.ts";

// A steering interrupt cancels the in-flight provider call to redirect the
// run, not to stop it — the harness auto-continues with the queued steering
// text right after. Matched by exact summary text so both the loop runner
// (which produces it) and the harness (which decides whether to resume as a
// plain turn vs. a fresh next turn) agree on what counts as one.
export const STEERING_INTERRUPT_SUMMARY =
	"Current provider response interrupted to apply steering.";

const STEERING_INTERRUPT_NAME = "SteeringInterruptError";

export function createSteeringInterruptReason(): Error {
	const error = new Error(STEERING_INTERRUPT_SUMMARY);
	error.name = STEERING_INTERRUPT_NAME;
	return error;
}

function isSteeringInterrupt(signal: AbortSignal | undefined): boolean {
	return (
		signal?.aborted === true &&
		signal.reason instanceof Error &&
		signal.reason.name === STEERING_INTERRUPT_NAME
	);
}

function lastAssistantContent(messages: Message[]): string {
	const assistant = [...messages]
		.reverse()
		.find(message => message.role === "assistant");
	return typeof assistant?.content === "string" ? assistant.content : "";
}

/** Resolve a run's terminal outcome from the declared task_status, if any. */
function resolveOutcome(ctx: {
	declared: { status: string; summary: string; ts: number } | null | undefined;
	structuredOutcomeRequired: boolean;
	fallbackStatus?: RunOutcomeStatus;
	fallbackSummary?: string;
}): {
	status: RunOutcomeStatus;
	summary?: string;
	source: "structured" | "heuristic" | "runtime";
} {
	if (ctx.declared) {
		return {
			status: (ctx.declared.status === "done"
				? "completed"
				: ctx.declared.status) as RunOutcomeStatus,
			summary: ctx.declared.summary,
			source: "structured",
		};
	}
	if (ctx.structuredOutcomeRequired) {
		return {
			status: "blocked",
			summary:
				"The run stopped after tool work without a structured task outcome. Resume to verify completion or declare the blocker.",
			source: "runtime",
		};
	}
	return {
		status: ctx.fallbackStatus ?? "completed",
		summary: ctx.fallbackSummary,
		source: "heuristic",
	};
}

export interface RunAgentLoopContext {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
	cwd?: string;
}

export interface RunAgentLoopConfig extends AgentConfig {
	backend: LLMBackend;
	signal?: AbortSignal;
	maxIterations?: number;
	/** Called when in-loop context-full recovery replaces the active transcript. */
	onContextCompacted?: (messages: Message[]) => void;
	/** Refresh mutable harness configuration after a completed turn. */
	refreshNextTurnConfig?: () => Promise<AgentConfig> | AgentConfig;
	// getDeclaredTaskStatus / resetDeclaredTaskStatus are declared on AgentConfig
	// (see types-config.ts) — inherited here, not redeclared.
}

async function runAgentLoopInternal(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	const downstreamEmit = emit;
	let eventSequence = 0;
	emit = event =>
		downstreamEmit({
			...event,
			seq: ++eventSequence,
			ts: Date.now(),
		});
	let messages = [
		...withSystemPrompt(context.systemPrompt, context.messages),
		...prompts,
	];
	const newMessages: Message[] = [...prompts];
	config.resetDeclaredTaskStatus?.();
	const finish = async (outcome: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		await emit({ type: "run_outcome", ...outcome });
		await emit({ type: "agent_end", messages: newMessages });
		return newMessages;
	};
	let settings = resolveAgentSettings(config);
	const maxIterations = settings.maxIterations;
	const executionPolicy = resolveExecutionPolicy(settings.executionProfile);

	const createRegistry = (tools: Tool[]): ToolRegistry => {
		const next = new ToolRegistry({
			cwd: context.cwd ?? config.cwd,
			allowedPaths: config.allowedPaths,
			allowAllPaths: config.allowAllPaths,
			signal: config.signal,
			onQuestionRequest: config.onQuestionRequest,
			maxResultChars: config.truncation?.toolResultMaxChars,
		});
		next.registerMany(tools);
		return next;
	};
	let registry = createRegistry(context.tools ?? config.tools ?? []);

	let iteration = 0;
	let performedToolWork = false;
	let contextWasCompacted = false;
	const providerTurnState = createProviderTurnState();

	async function drainSteering(): Promise<Message[]> {
		return (
			(await config.hooks?.getSteeringMessages?.({ messages, iteration })) ?? []
		);
	}

	async function drainFollowUps(): Promise<Message[]> {
		return (
			(await config.hooks?.getFollowUpMessages?.({
				messages,
				iteration,
				assistantText: assistantText(newMessages.at(-1)),
				stopReason: "stop",
			})) ?? []
		);
	}

	let pendingMessages = await drainSteering();

	// Apply beforeAgentStart hook
	const beforeAgentStartResult = await config.hooks?.beforeAgentStart?.({
		prompt: prompts.map(p => p.content).join("\n"),
		systemPrompt: context.systemPrompt ?? "",
		messages: messages as AgentMessage[],
	});

	await emit({ type: "agent_start" });
	const promptTurnId = "turn_0";
	for (const prompt of prompts) {
		await emitMessagePair(emit, promptTurnId, prompt);
	}

	// Apply beforeAgentStart hook results to messages and system prompt
	if (beforeAgentStartResult?.messages) {
		for (const msg of beforeAgentStartResult.messages) {
			messages.push(msg as Message);
			newMessages.push(msg as Message);
		}
	}
	if (beforeAgentStartResult?.systemPrompt) {
		context.systemPrompt = beforeAgentStartResult.systemPrompt;
	}

	// Outer loop: continues while follow-up messages keep arriving after the
	// agent would otherwise stop (mirrors pi's runLoop outer while(true)).
	while (iteration < maxIterations) {
		if (config.signal?.aborted) {
			const steeringInterrupt = isSteeringInterrupt(config.signal);
			if (!steeringInterrupt) {
				await emit({ type: "error", message: "Operation aborted" });
			}
			return finish({
				status: "cancelled",
				summary: steeringInterrupt
					? STEERING_INTERRUPT_SUMMARY
					: "Operation aborted before the provider request.",
				source: "runtime",
			});
		}

		let hasMoreToolCalls = true;
		// Inner loop: process one assistant turn plus its tool calls, and any
		// steering messages injected before the next turn.
		while (
			(hasMoreToolCalls || pendingMessages.length > 0) &&
			iteration < maxIterations
		) {
			iteration++;
			const turnId = `turn_${iteration}`;
			await emit({ type: "turn_start", turnId });

			if (pendingMessages.length > 0) {
				for (const pending of pendingMessages) {
					messages.push(pending);
					newMessages.push(pending);
					await emitMessagePair(emit, turnId, pending);
				}
				pendingMessages = [];
			}

			const transformResult = await config.hooks?.transformContext?.({
				messages: messages as AgentMessage[],
				iteration,
				signal: config.signal,
			});
			const transformed = transformResult?.messages;
			if (transformed) {
				messages = transformed as Message[];
				if (contextWasCompacted) config.onContextCompacted?.(messages);
			}

			const turnResult = await requestAssistantTurn({
				state: providerTurnState,
				messages,
				config,
				settings,
				registry,
				turnId,
				iteration,
				contextWasCompacted,
				convertToLlm: config.convertToLlm ?? defaultConvertToLlm,
				emit,
				isSteeringInterrupt,
				steeringInterruptSummary: STEERING_INTERRUPT_SUMMARY,
			});
			if (turnResult.kind === "finish") {
				return finish(turnResult.outcome);
			}
			const response = turnResult.response;
			messages = turnResult.messages;
			contextWasCompacted = turnResult.contextWasCompacted;

			const processResult = processProviderResponse({
				response,
				registry,
				messages,
				newMessages,
				turnId,
				iteration,
				emit,
				config,
			});

			let toolCalls: ToolCall[];
			let assistant: Message;
			if (processResult.success) {
				toolCalls = processResult.toolCalls;
				assistant = processResult.assistant;
				if (toolCalls.some(call => call.name !== "task_status")) {
					performedToolWork = true;
				}
			} else {
				return finish({
					status: "failed",
					summary:
						processResult.errorMessage ?? "Model returned empty response.",
					source: "runtime",
				});
			}
			const rawStopReason =
				(response?.stopReason as "stop" | "length" | "error") ?? "stop";
			const stopReason = stopReasonFor(rawStopReason, toolCalls);

			hasMoreToolCalls = false;
			const batch = await executeToolBatch({
				registry,
				toolCalls,
				rawStopReason,
				toolExecution: settings.toolExecution,
				iteration,
				signal: config.signal,
				hooks: config.hooks,
				permissions: config.permissions,
				onPermissionRequest: config.onPermissionRequest,
				emit,
			});
			const toolResults = batch.messages;
			const toolTerminated = batch.terminated;
			for (const toolResult of toolResults) {
				messages.push(toolResult);
				newMessages.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = true;
			}

			// The final usage-only SSE chunk is optional and many local providers
			// omit it. Estimate the serialized conversation as a reliable fallback
			// so context usage never remains stuck at zero.
			const contextTokens = Math.max(
				estimateChatPayloadTokens(messages, registry.toToolDefinitions()),
				response?.usage?.totalTokens ?? 0,
			);
			await emit({
				type: "context_update",
				tokens: contextTokens,
				maxTokens: config.contextWindowTokens,
				cachedTokens: response?.usage?.cachedTokens ?? null,
				promptTokens: response?.usage?.promptTokens ?? null,
				completionTokens: response?.usage?.completionTokens ?? null,
			});
			await emit({
				type: "turn_end",
				turnId,
				stopReason,
				message: assistant,
				toolResults,
			});

			const refreshedConfig = await config.refreshNextTurnConfig?.();
			if (refreshedConfig) {
				Object.assign(config, refreshedConfig);
				settings = resolveAgentSettings(config);
				context.systemPrompt = refreshedConfig.systemPrompt;
				messages = [
					createSystemMessage(
						refreshedConfig.systemPrompt ?? "You are a helpful assistant.",
					),
					...messages.filter(message => message.role !== "system"),
				];
				registry = createRegistry(refreshedConfig.tools ?? []);
			}

			const prepareResult = await config.hooks?.prepareNextTurn?.({
				messages,
				iteration,
				hadToolCalls: toolCalls.length > 0,
			});
			const prepared = prepareResult?.messages;
			if (prepared) {
				messages = prepared;
				if (contextWasCompacted) config.onContextCompacted?.(messages);
			}

			// When a tool signals terminate, still drain follow-ups before exiting
			// — this prevents skipping queued follow-up messages (e.g. steering
			// injected mid-turn) just because a tool requested termination.
			if (toolTerminated) {
				const followUpsOnTerminate = await drainFollowUps();
				if (followUpsOnTerminate.length > 0) {
					pendingMessages = followUpsOnTerminate;
					hasMoreToolCalls = false;
					// Re-enter inner loop with follow-up messages
					continue;
				}
				return finish(
					resolveOutcome({
						declared: config.getDeclaredTaskStatus?.() ?? null,
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
			}

			// Only invoke shouldStopAfterTurn when no tool calls ran. Tool turns
			// always continue unless the hook is explicitly wired to stop on tool
			// turns — checking it unconditionally causes premature exits when
			// hooks have stale state from a previous no-tool turn.
			const stop =
				toolCalls.length === 0
					? ((await config.hooks?.shouldStopAfterTurn?.({
							messages,
							iteration,
							hadToolCalls: false,
						})) ?? false)
					: false;
			if (stop) {
				return finish(
					resolveOutcome({
						declared: config.getDeclaredTaskStatus?.() ?? null,
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
			}

			pendingMessages = await drainSteering();
		}

		// Agent would stop here. Check for follow-up messages (pi-style outer
		// loop). There is no built-in autonomous-continuation policy — the
		// harness populates getFollowUpMessages / shouldStopAfterTurn itself
		// (e.g. builtin-hooks' todo/length-truncation nudges) if it wants
		// anything beyond "stop when there's nothing pending."
		const followUpMessages = await drainFollowUps();
		if (followUpMessages.length > 0) {
			pendingMessages = followUpMessages;
			continue;
		}
		break;
	}

	if (iteration >= maxIterations) {
		await emit({
			type: "max_iterations",
			iterations: iteration,
			limit: maxIterations,
		});
	}

	const declared = config.getDeclaredTaskStatus?.() ?? null;
	if (
		(declared || (performedToolWork && registry.has("task_status"))) &&
		executionPolicy.embeddedPoliciesEnabled
	) {
		return finish(
			resolveOutcome({
				declared,
				structuredOutcomeRequired:
					performedToolWork && registry.has("task_status"),
			}),
		);
	}
	if (config.signal?.aborted) {
		return finish({
			status: "cancelled",
			summary: isSteeringInterrupt(config.signal)
				? STEERING_INTERRUPT_SUMMARY
				: "Operation aborted.",
			source: "runtime",
		});
	}

	const finalText = lastAssistantContent(newMessages);
	return finish({
		status: iteration >= maxIterations ? "failed" : "completed",
		summary: finalText || undefined,
		source:
			iteration >= maxIterations || !executionPolicy.embeddedPoliciesEnabled
				? "runtime"
				: "heuristic",
	});
}

export function runAgentLoop(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	return runAgentLoopInternal(context, prompts, config, emit);
}
