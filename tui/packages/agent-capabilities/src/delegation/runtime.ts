import type { LLMBackend } from "@logician/agent-core/agent/backend.ts";
import { runAgentLoop } from "@logician/agent-core/agent/agent-loop-runner.ts";
import {
	stripAcceptanceReport,
	type AcceptanceConfig,
	type AcceptanceLedger,
} from "@logician/agent-core/agent/guards/acceptance-contract.ts";
import type {
	AgentConfig,
	AgentEvent,
	Message,
	Tool,
} from "@logician/agent-core";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
} from "@logician/agent-core/agent/types/types-hooks.ts";

export interface DelegationContract {
	expectedOutput?: string;
	successCriteria?: string[];
	maxValidationRetries?: number;
}

export interface DelegationBudget {
	timeoutMs?: number;
	maxToolCalls?: number;
	toolLimits?: Record<string, number>;
}

/** Per-task input shape for the spawn_agents tool. */
export interface SpawnAgentsTask {
	task: string;
	agent?: string;
	expected_output?: string;
	success_criteria?: string[];
	max_validation_retries?: number;
	timeout_ms?: number;
	max_tool_calls?: number;
}

export interface DelegatedRunResult {
	content: string;
	messages: Message[];
	status: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
	turns: number;
	durationMs: number;
	toolCalls: number;
	toolCallsByName: Record<string, number>;
	validationAttempts: number;
	acceptance?: AcceptanceLedger;
}

function positiveInt(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value > 0
		? Math.floor(value)
		: undefined;
}

function nonNegativeInt(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value >= 0
		? Math.floor(value)
		: undefined;
}

export function contractFromArgs(
	args: Record<string, unknown>,
): DelegationContract {
	return {
		expectedOutput:
			typeof args.expected_output === "string" && args.expected_output.trim()
				? args.expected_output.trim()
				: undefined,
		successCriteria: Array.isArray(args.success_criteria)
			? args.success_criteria.map(String).map((v) => v.trim()).filter(Boolean)
			: undefined,
		maxValidationRetries: nonNegativeInt(args.max_validation_retries),
	};
}

export function budgetFromArgs(
	args: Record<string, unknown>,
	fallback: DelegationBudget,
): DelegationBudget {
	return {
		timeoutMs: positiveInt(args.timeout_ms) ?? fallback.timeoutMs,
		maxToolCalls: positiveInt(args.max_tool_calls) ?? fallback.maxToolCalls,
		toolLimits: fallback.toolLimits,
	};
}

export const DELEGATION_CONTRACT_PROPERTIES = {
	expected_output: {
		type: "string",
		description: "Concrete shape and contents required in the final result.",
	},
	success_criteria: {
		type: "array",
		items: { type: "string" },
		description: "Criteria the subagent must explicitly satisfy with evidence.",
	},
	max_validation_retries: {
		type: "integer",
		minimum: 0,
		maximum: 5,
		description: "Correction attempts after contract validation fails (default: 2).",
	},
	timeout_ms: {
		type: "integer",
		minimum: 1000,
		description: "Whole-task deadline including tools and validation retries.",
	},
	max_tool_calls: {
		type: "integer",
		minimum: 1,
		description: "Maximum total tool calls allowed for this delegated task.",
	},
} as const;

function acceptanceFor(contract: DelegationContract): AcceptanceConfig | undefined {
	const criteria = [
		...(contract.expectedOutput
			? [`Final result matches this expected output: ${contract.expectedOutput}`]
			: []),
		...(contract.successCriteria ?? []),
	];
	return criteria.length
		? { criteria, maxFinalizationTurns: contract.maxValidationRetries ?? 2 }
		: undefined;
}

function combineSignal(parent: AbortSignal | undefined, timeoutMs?: number): AbortSignal | undefined {
	const timeout = timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined;
	if (parent && timeout) return AbortSignal.any([parent, timeout]);
	return parent ?? timeout;
}

function budgetTools(
	tools: Tool[],
	budget: DelegationBudget,
	counters: {
		total: number;
		byName: Record<string, number>;
		violation?: string;
	},
): Tool[] {
	// No budget active — return tools unmodified (no wrapper overhead).
	if (!budget.maxToolCalls && !budget.toolLimits) return tools;
	return tools.map((tool) => ({
		...tool,
		execute: async (args, ctx) => {
			const nextTotal = counters.total + 1;
			const nextForTool = (counters.byName[tool.name] ?? 0) + 1;
			if (budget.maxToolCalls && nextTotal > budget.maxToolCalls) {
				counters.violation =
					`Delegated task exceeded its ${budget.maxToolCalls}-call tool budget.`;
				throw new Error(counters.violation);
			}
			const toolLimit = budget.toolLimits?.[tool.name];
			if (toolLimit && nextForTool > toolLimit) {
				counters.violation =
					`Tool "${tool.name}" exceeded its delegated-task limit of ${toolLimit}.`;
				throw new Error(counters.violation);
			}
			counters.total = nextTotal;
			counters.byName[tool.name] = nextForTool;
			return tool.execute(args, ctx);
		},
	}));
}

export async function runDelegatedAgent(params: {
	task: string;
	config: AgentConfig;
	backend: LLMBackend;
	tools: Tool[];
	maxIterations: number;
	signal?: AbortSignal;
	contract?: DelegationContract;
	budget?: DelegationBudget;
	onEvent: (event: AgentEvent) => void;
}): Promise<DelegatedRunResult> {
	const startedAt = Date.now();
	const counters: {
		total: number;
		byName: Record<string, number>;
		violation?: string;
	} = { total: 0, byName: {} };
	const contract = params.contract ?? {};
	const acceptance = acceptanceFor(contract);
	const signal = combineSignal(params.signal, params.budget?.timeoutMs);
	const tools = budgetTools(params.tools, params.budget ?? {}, counters);
	let status: DelegatedRunResult["status"] = "completed";
	let ledger: AcceptanceLedger | undefined;
	let turns = 0;
	let validationAttempts = 0;

	// Subagents run runAgentLoop directly, bypassing the harness's
	// buildBuiltinHooks — so without this, task_status's "done" declaration
	// never sets `terminate: true` and the child loops on runner nudges
	// ("call task_status with the accurate status") until maxIterations,
	// surfacing as a spurious isError result even though the model finished.
	const priorAfterToolCall = params.config.internalHooks?.afterToolCall;
	const afterToolCall = async (
		ctx: AfterToolCallContext,
		signal?: AbortSignal,
	): Promise<AfterToolCallResult | undefined> => {
		const prior = await priorAfterToolCall?.(ctx, signal);
		if (prior?.terminate) return prior;
		if (ctx.toolCall.name === "task_status" && !ctx.isError) {
			return { ...prior, terminate: true };
		}
		return prior;
	};
	const configWithTermination: AgentConfig = {
		...params.config,
		internalHooks: { ...params.config.internalHooks, afterToolCall },
	};

	let messages: Message[] = [];
	let prompts: Message[] = [
		{ role: "user", content: params.task } satisfies Message,
	];
	const maxAttempts = acceptance ? (contract.maxValidationRetries ?? 2) + 1 : 1;
	for (let attempt = 0; attempt < maxAttempts; attempt++) {
		let acceptancePassed = !acceptance;
		const remainingIterations = Math.max(1, params.maxIterations - turns);
		const produced = await runAgentLoop(
			{
				systemPrompt: params.config.systemPrompt,
				messages,
				tools,
				cwd: params.config.cwd,
			},
			prompts,
			{
				...configWithTermination,
				backend: params.backend,
				tools,
				maxIterations: remainingIterations,
				signal,
				// Delegation owns its cross-run retry loop so it can keep aggregate
				// budgets and return only the clean final result. Disable the runner's
				// in-run corrective turn here to avoid stacking both retry mechanisms.
				acceptance: acceptance
					? { ...acceptance, maxFinalizationTurns: 0 }
					: undefined,
			},
			(event) => {
				if (event.type === "turn_start") turns++;
				if (event.type === "run_outcome") status = event.status;
				if (event.type === "acceptance_complete") {
					validationAttempts++;
					acceptancePassed = event.status === "passed";
					ledger = event.report as AcceptanceLedger | undefined;
					if (!acceptancePassed) status = "failed";
				}
				params.onEvent(event);
			},
		);
		messages = [...messages, ...produced];
		if (acceptancePassed || !acceptance || signal?.aborted) break;
		if (attempt + 1 >= maxAttempts || turns >= params.maxIterations) break;
		prompts = [{
			role: "user",
			content:
				"The delegated result failed its acceptance contract. Correct the result, " +
				"provide concrete evidence for every required criterion, and emit a new " +
				"acceptance-report. Do not repeat work that is already complete.",
			timestamp: Date.now(),
		}];
	}

	const final = [...messages]
		.reverse()
		.find((message) => message.role === "assistant" && message.content?.trim());
	const raw = final?.content?.trim() || "(subagent produced no final message)";
	if (counters.violation) status = "failed";
	return {
		content: acceptance ? stripAcceptanceReport(raw).trim() || raw : raw,
		messages,
		status,
		turns,
		durationMs: Date.now() - startedAt,
		toolCalls: counters.total,
		toolCallsByName: counters.byName,
		validationAttempts,
		acceptance: ledger,
	};
}
