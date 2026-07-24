import { createToolResultMessage } from "../core/messages.ts";
import type { AgentEvent, AgentHooks, Message, ToolCall } from "../core/types.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";

type Emit = (event: AgentEvent) => void | Promise<void>;
type EmitExtension = (event: { type: string; [key: string]: unknown }) => Promise<void>;

export interface ToolBatchControllerOptions {
	registry: ToolRegistry;
	toolCalls: ToolCall[];
	rawStopReason: "stop" | "length" | "error";
	toolExecution?: "parallel" | "sequential";
	iteration: number;
	signal?: AbortSignal;
	internalHooks?: AgentHooks;
	hooks?: AgentHooks;
	emit: Emit;
	emitExtension: EmitExtension;
}

export interface ToolBatchResult {
	messages: Message[];
	terminated: boolean;
}

const CANCELLED_TOOL_RESULT =
	"Tool call was not executed because the operation was cancelled.";

export async function executeToolBatch(options: ToolBatchControllerOptions): Promise<ToolBatchResult> {
	const { registry, toolCalls, rawStopReason, iteration, signal, emit, emitExtension } = options;
	if (rawStopReason === "length") {
		const messages: Message[] = [];
		for (const call of toolCalls) {
			const prepared = registry.prepare(call);
			await emit({ type: "tool_execution_start", toolCallId: prepared.call.id, toolName: prepared.call.name, args: prepared.args });
			await emitExtension({ type: "tool_execution_start", toolCallId: prepared.call.id, toolName: prepared.call.name, args: prepared.args });
			const text = `Tool call "${call.name}" was not executed because the assistant response hit the output token limit; its arguments may be truncated. Re-issue the tool call with complete arguments.`;
			await emit({ type: "tool_call_end", toolName: call.name, toolCallId: call.id, result: text, isError: true });
			await emit({ type: "tool_execution_end", toolCallId: call.id, toolName: call.name, result: text, isError: true });
			await emitExtension({ type: "tool_execution_end", toolCallId: call.id, toolName: call.name, result: text, isError: true });
			messages.push(createToolResultMessage(call.id, call.name, text, true));
		}
		return { messages, terminated: false };
	}

	type Plan = { prepared: ReturnType<ToolRegistry["prepare"]>; args: Record<string, unknown>; immediateContent?: string; immediateError: boolean };
	const plans: Plan[] = [];
	for (const toolCall of toolCalls) {
		const prepared = registry.prepare(toolCall);
		await emit({ type: "tool_execution_start", toolCallId: prepared.call.id, toolName: prepared.call.name, args: prepared.args });
		await emitExtension({ type: "tool_execution_start", toolCallId: prepared.call.id, toolName: prepared.call.name, args: prepared.args });
		const context = { toolCall: prepared.call, args: prepared.args, iteration };
		let before = signal?.aborted
			? undefined
			: await options.internalHooks?.beforeToolCall?.(context, signal);
		if (!signal?.aborted && before === undefined) {
			before = await options.hooks?.beforeToolCall?.(context, signal);
		}
		const immediateContent =
			before?.content ?? prepared.error ?? (signal?.aborted ? CANCELLED_TOOL_RESULT : undefined);
		plans.push({
			prepared,
			args: before?.args ?? prepared.args,
			immediateContent,
			immediateError:
				before?.isError === true ||
				prepared.error !== undefined ||
				immediateContent === CANCELLED_TOOL_RESULT,
		});
	}

	const execute = async (plan: Plan): Promise<{ message: Message; terminate: boolean }> => {
		const { prepared, args } = plan;
		let resultText = plan.immediateContent;
		let isError = plan.immediateError;
		let terminate = false;
		const updates: Promise<void>[] = [];
		let accepting = true;
		if (resultText === undefined) {
			const result = await registry.execute(prepared.call, { signal, onUpdate: (partialResult) => {
				if (!accepting) return;
				updates.push(Promise.resolve(emit({ type: "tool_execution_update", toolCallId: prepared.call.id, toolName: prepared.call.name, args, partialResult })));
				updates.push(Promise.resolve(emit({ type: "tool_call_update", toolCallId: prepared.call.id, toolName: prepared.call.name, partialResult })));
			} }, args);
			accepting = false;
			await Promise.all(updates);
			resultText = result.content;
			isError = result.isError === true;
			terminate = result.terminate === true;
		}
		accepting = false;
		await Promise.all(updates);
		const context = { toolCall: prepared.call, args, result: resultText, isError, iteration };
		let after = await options.internalHooks?.afterToolCall?.(context, signal);
		if (after === undefined) {
			after = await options.hooks?.afterToolCall?.(context, signal);
		}
		if (after?.content !== undefined) resultText = after.content;
		if (after?.isError !== undefined) isError = after.isError;
		await emit({ type: "tool_call_end", toolName: prepared.call.name, toolCallId: prepared.call.id, result: resultText, isError });
		await emit({ type: "tool_execution_end", toolCallId: prepared.call.id, toolName: prepared.call.name, result: resultText, isError });
		await emitExtension({ type: "tool_execution_end", toolCallId: prepared.call.id, toolName: prepared.call.name, result: resultText, isError });
		return { message: createToolResultMessage(prepared.call.id, prepared.call.name, resultText, isError), terminate: after?.terminate ?? terminate };
	};

	const outcomes: Array<{ message: Message; terminate: boolean }> = [];
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
			outcomes.push(...await Promise.all(parallelStage.map(execute)));
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
	return { messages: outcomes.map((outcome) => outcome.message), terminated: outcomes.length > 0 && outcomes.every((outcome) => outcome.terminate) };
}
