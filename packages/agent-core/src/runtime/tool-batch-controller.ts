import { createToolResultMessage } from "../agent/messages.ts";
import type {
	AgentEvent,
	AgentHooks,
	Message,
	ToolCall,
} from "../agent/types.ts";
import type { ExtensionEvent as TypedExtensionEvent } from "../hooks/extensions/events.ts";
import type { PermissionManager } from "../tools/shared/permissions.ts";
import type { ToolRegistry } from "../tools/shared/registry.ts";

type Emit = (event: AgentEvent) => void | Promise<void>;
type EmitExtension = (event: TypedExtensionEvent) => Promise<void>;
type OnPermissionRequest = (ctx: {
	toolName: string;
	toolCallId: string;
	args: Record<string, unknown>;
}) => Promise<"allow" | "deny" | "always">;
export interface PermissionDecisionRecord {
	toolCallId: string;
	toolName: string;
	decision: "allow" | "deny";
	source: "rule" | "mode" | "user" | "fail_closed";
	scope?: "once" | "session";
}

export interface ToolBatchControllerOptions {
	registry: ToolRegistry;
	toolCalls: ToolCall[];
	rawStopReason: "stop" | "length" | "error";
	toolExecution?: "parallel" | "sequential";
	iteration: number;
	signal?: AbortSignal;
	internalHooks?: AgentHooks;
	hooks?: AgentHooks;
	permissions?: PermissionManager;
	onPermissionRequest?: OnPermissionRequest;
	onPermissionDecision?: (
		decision: PermissionDecisionRecord,
	) => void | Promise<void>;
	onToolIntent?: (input: {
		toolCallId: string;
		toolName: string;
		args: Record<string, unknown>;
		recovery:
			| "pure"
			| "idempotent"
			| "receipt_recoverable"
			| "at_most_once_unknown";
	}) =>
		| { operationId: string; idempotencyKey: string }
		| undefined
		| Promise<{ operationId: string; idempotencyKey: string } | undefined>;
	onToolResult?: (input: {
		toolCallId: string;
		toolName: string;
		result: string;
		isError: boolean;
		receipt?: string;
	}) => void | Promise<void>;
	emit: Emit;
	emitExtension: EmitExtension;
}

const PERMISSION_DENIED_PREFIX = "Tool call denied";

export interface ToolBatchResult {
	messages: Message[];
	terminated: boolean;
	executedToolCallIds: string[];
}

const CANCELLED_TOOL_RESULT =
	"Tool call was not executed because the operation was cancelled.";

/**
 * Gate a single prepared call behind the permission engine. Returns denial
 * text to short-circuit execution, or undefined to let it proceed.
 *
 * "ask" verdicts need `onPermissionRequest` to resolve interactively; with no
 * handler wired up (e.g. headless run) an "ask" fails closed as a denial
 * rather than silently executing.
 */
async function evaluatePermission(
	permissions: PermissionManager,
	onPermissionRequest: OnPermissionRequest | undefined,
	call: ToolCall,
	args: Record<string, unknown>,
	tool: ReturnType<ToolRegistry["get"]>,
	onDecision: ToolBatchControllerOptions["onPermissionDecision"],
	emit: Emit,
): Promise<string | undefined> {
	const verdict = permissions.evaluate(call, args, tool);
	if (verdict.decision === "allow") {
		await onDecision?.({
			toolCallId: call.id,
			toolName: call.name,
			decision: "allow",
			source: verdict.source,
		});
		await emit({
			type: "tool_permission_decision",
			toolCallId: call.id,
			toolName: call.name,
			decision: "allow",
			source: verdict.source,
		});
		return undefined;
	}
	if (verdict.decision === "deny") {
		await onDecision?.({
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: verdict.source,
		});
		await emit({
			type: "tool_permission_decision",
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: verdict.source,
		});
		return `${PERMISSION_DENIED_PREFIX}: ${verdict.reason ?? `"${call.name}" is not permitted in the current mode.`}`;
	}

	// decision === "ask"
	if (!onPermissionRequest) {
		await onDecision?.({
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: "fail_closed",
		});
		await emit({
			type: "tool_permission_decision",
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: "fail_closed",
		});
		return `${PERMISSION_DENIED_PREFIX}: "${call.name}" requires approval, but no interactive handler is available.`;
	}
	await emit({
		type: "tool_permission_request",
		toolCallId: call.id,
		toolName: call.name,
		args: JSON.stringify(args),
	});
	const answer = await onPermissionRequest({
		toolName: call.name,
		toolCallId: call.id,
		args,
	});
	if (answer === "deny") {
		await onDecision?.({
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: "user",
			scope: "once",
		});
		await emit({
			type: "tool_permission_decision",
			toolCallId: call.id,
			toolName: call.name,
			decision: "deny",
			source: "user",
		});
		return `${PERMISSION_DENIED_PREFIX}: the user denied "${call.name}".`;
	}
	if (answer === "always") {
		permissions.addSessionAllow(call.name);
	}
	await onDecision?.({
		toolCallId: call.id,
		toolName: call.name,
		decision: "allow",
		source: "user",
		scope: answer === "always" ? "session" : "once",
	});
	await emit({
		type: "tool_permission_decision",
		toolCallId: call.id,
		toolName: call.name,
		decision: answer,
		source: "user",
	});
	return undefined;
}

export async function executeToolBatch(
	options: ToolBatchControllerOptions,
): Promise<ToolBatchResult> {
	const {
		registry,
		toolCalls,
		rawStopReason,
		iteration,
		signal,
		emit,
		emitExtension,
	} = options;
	if (rawStopReason === "length") {
		const messages: Message[] = [];
		for (const call of toolCalls) {
			const prepared = registry.prepare(call);
			await emit({
				type: "tool_execution_start",
				toolCallId: prepared.call.id,
				toolName: prepared.call.name,
				args: prepared.args,
			});
			await emitExtension({
				type: "tool_execution_start",
				toolCallId: prepared.call.id,
				toolName: prepared.call.name,
				args: prepared.args,
			});
			const text =
				call.name === "write_file"
					? `Tool call "${call.name}" was not executed because the assistant response hit the output token limit; its arguments may be truncated. ` +
						"The content is too large for a single call. Split it into smaller chunks and use write_file_append repeatedly (same path, in order) instead of retrying write_file with the full content."
					: `Tool call "${call.name}" was not executed because the assistant response hit the output token limit; its arguments may be truncated. Re-issue the tool call with complete arguments.`;
			await emit({
				type: "tool_call_end",
				toolName: call.name,
				toolCallId: call.id,
				result: text,
				isError: true,
			});
			await emit({
				type: "tool_execution_end",
				toolCallId: call.id,
				toolName: call.name,
				result: text,
				isError: true,
			});
			await emitExtension({
				type: "tool_execution_end",
				toolCallId: call.id,
				toolName: call.name,
				result: text,
				isError: true,
			});
			messages.push(createToolResultMessage(call.id, call.name, text, true));
		}
		return { messages, terminated: false, executedToolCallIds: [] };
	}

	type Plan = {
		prepared: ReturnType<ToolRegistry["prepare"]>;
		args: Record<string, unknown>;
		immediateContent?: string;
		immediateError: boolean;
	};
	const plans: Plan[] = [];
	for (const toolCall of toolCalls) {
		const prepared = registry.prepare(toolCall);
		await emit({
			type: "tool_execution_start",
			toolCallId: prepared.call.id,
			toolName: prepared.call.name,
			args: prepared.args,
		});
		await emitExtension({
			type: "tool_execution_start",
			toolCallId: prepared.call.id,
			toolName: prepared.call.name,
			args: prepared.args,
		});

		let permissionDenial: string | undefined;
		if (
			!signal?.aborted &&
			prepared.error === undefined &&
			options.permissions
		) {
			permissionDenial = await evaluatePermission(
				options.permissions,
				options.onPermissionRequest,
				prepared.call,
				prepared.args,
				registry.get(prepared.call.name),
				options.onPermissionDecision,
				emit,
			);
		}

		const context = { toolCall: prepared.call, args: prepared.args, iteration };
		let before =
			signal?.aborted || permissionDenial !== undefined
				? undefined
				: await options.internalHooks?.beforeToolCall?.(context, signal);
		if (
			!signal?.aborted &&
			permissionDenial === undefined &&
			before === undefined
		) {
			before = await options.hooks?.beforeToolCall?.(context, signal);
		}
		const immediateContent =
			before?.content ??
			prepared.error ??
			permissionDenial ??
			(signal?.aborted ? CANCELLED_TOOL_RESULT : undefined);
		plans.push({
			prepared,
			args: before?.args ?? prepared.args,
			immediateContent,
			immediateError:
				before?.isError === true ||
				prepared.error !== undefined ||
				permissionDenial !== undefined ||
				immediateContent === CANCELLED_TOOL_RESULT,
		});
	}

	const execute = async (
		plan: Plan,
	): Promise<{
		message: Message;
		terminate: boolean;
		executed: boolean;
		toolCallId: string;
	}> => {
		const { prepared, args } = plan;
		let resultText = plan.immediateContent;
		let isError = plan.immediateError;
		let terminate = false;
		let accepting = true;
		let executed = false;
		if (resultText === undefined) {
			const tool = registry.get(prepared.call.name);
			const durableIntent = await options.onToolIntent?.({
				toolCallId: prepared.call.id,
				toolName: prepared.call.name,
				args,
				recovery:
					tool?.recoverySemantics ??
					(tool?.readOnly === true || tool?.cacheable === true
						? "pure"
						: "at_most_once_unknown"),
			});
			const result = await registry.execute(
				prepared.call,
				{
					signal,
					operationId: durableIntent?.operationId,
					idempotencyKey: durableIntent?.idempotencyKey,
					onUpdate: async partialResult => {
						if (!accepting) return;
						await emit({
							type: "tool_execution_update",
							toolCallId: prepared.call.id,
							toolName: prepared.call.name,
							args,
							partialResult,
						});
					},
				},
				args,
			);
			accepting = false;
			resultText = result.content;
			isError = result.isError === true;
			terminate = result.terminate === true;
			executed = true;
			await options.onToolResult?.({
				toolCallId: prepared.call.id,
				toolName: prepared.call.name,
				result: resultText,
				isError,
				receipt: result.recoveryReceipt,
			});
		}
		accepting = false;
		const context = {
			toolCall: prepared.call,
			args,
			result: resultText,
			isError,
			iteration,
		};
		let after = await options.internalHooks?.afterToolCall?.(context, signal);
		if (after === undefined) {
			after = await options.hooks?.afterToolCall?.(context, signal);
		}
		if (after?.content !== undefined) resultText = after.content;
		if (after?.isError !== undefined) isError = after.isError;
		await emit({
			type: "tool_call_end",
			toolName: prepared.call.name,
			toolCallId: prepared.call.id,
			result: resultText,
			isError,
		});
		await emit({
			type: "tool_execution_end",
			toolCallId: prepared.call.id,
			toolName: prepared.call.name,
			result: resultText,
			isError,
		});
		await emitExtension({
			type: "tool_execution_end",
			toolCallId: prepared.call.id,
			toolName: prepared.call.name,
			result: resultText,
			isError,
		});
		return {
			message: createToolResultMessage(
				prepared.call.id,
				prepared.call.name,
				resultText,
				isError,
			),
			terminate: after?.terminate ?? terminate,
			executed,
			toolCallId: prepared.call.id,
		};
	};

	const outcomes: Array<{
		message: Message;
		terminate: boolean;
		executed: boolean;
		toolCallId: string;
	}> = [];
	if (options.toolExecution !== "parallel") {
		for (let index = 0; index < plans.length; index++) {
			const plan = plans[index];
			if (signal?.aborted && plan.immediateContent === undefined) {
				plan.immediateContent = CANCELLED_TOOL_RESULT;
				plan.immediateError = true;
			}
			outcomes.push(await execute(plan));
		}
	} else {
		// Sequential tools are ordering barriers, not a reason to serialize the
		// entire model-issued batch. Parallel-safe calls on either side still run
		// concurrently, while their results remain in the original call order.
		let parallelStage: Plan[] = [];
		const flushParallelStage = async () => {
			if (parallelStage.length === 0) return;
			outcomes.push(...(await Promise.all(parallelStage.map(execute))));
			parallelStage = [];
		};
		for (let index = 0; index < plans.length; index++) {
			const plan = plans[index];
			const mode = registry.get(toolCalls[index].name)?.executionMode;
			if (mode !== "sequential") {
				parallelStage.push(plan);
				continue;
			}
			await flushParallelStage();
			outcomes.push(await execute(plan));
		}
		await flushParallelStage();
	}
	return {
		messages: outcomes.map(outcome => outcome.message),
		terminated:
			outcomes.length > 0 && outcomes.every(outcome => outcome.terminate),
		executedToolCallIds: outcomes
			.filter(outcome => outcome.executed)
			.map(outcome => outcome.toolCallId),
	};
}
