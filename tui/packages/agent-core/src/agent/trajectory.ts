import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type { GenerateOptions, LLMBackend, LLMResponse } from "./backend.ts";
import { BackendError } from "./backend.ts";
import type { AgentConfig, AgentEvent } from "./types.ts";

export interface TrajectoryMetadata {
	harnessVersion: string;
	model: string;
	baseUrl: string;
	config: Record<string, unknown>;
	tools: Array<{ name: string; description?: string }>;
}

export interface TrajectoryEntry {
	version: 1;
	sequence: number;
	timestamp: number;
	sessionId: string;
	runId: string;
	operationId: string;
	kind: "run_start" | "agent_event" | "run_finish";
	payload: Record<string, unknown>;
}

export interface TrajectoryReport {
	events: number;
	durationMs: number;
	providerRetries: number;
	toolCalls: number;
	toolFailures: number;
	loopEscapes: number;
	compactions: number;
	acceptancePassed: boolean;
	prematureStop: boolean;
	replayComplete: boolean;
}

function safeId(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function serializableConfig(config: AgentConfig): Record<string, unknown> {
	return {
		model: config.model,
		models: config.models,
		temperature: config.temperature,
		maxTokens: config.maxTokens,
		maxIterations: config.maxIterations,
		executionProfile: config.executionProfile,
		inferenceMode: config.inferenceMode,
		thinkingLevel: config.thinkingLevel,
		contextWindowTokens: config.contextWindowTokens,
		runBudget: config.runBudget,
		acceptance: config.acceptance,
		reflectionConfig: config.reflectionConfig,
		stopPolicies: config.stopPolicies,
		streamOptions: config.streamOptions
			? { ...config.streamOptions, headers: undefined }
			: undefined,
	};
}

/** Append-only trajectory recorder plus deterministic replay evaluator. */
export class TrajectoryRecorder {
	private sequence = 0;
	private sessionId: string;

	constructor(
		private readonly cwd: string,
		sessionId: string,
	) {
		this.sessionId = sessionId;
		this.sequence = this.load().at(-1)?.sequence ?? 0;
	}

	useSession(sessionId: string): void {
		this.sessionId = sessionId;
		this.sequence = this.load().at(-1)?.sequence ?? 0;
	}

	start(
		runId: string,
		operationId: string,
		config: AgentConfig,
		cause: "prompt" | "continue",
	): void {
		const metadata: TrajectoryMetadata = {
			harnessVersion: "0.2.0",
			model: config.model,
			baseUrl: config.baseUrl,
			config: serializableConfig(config),
			tools: (config.tools ?? []).map(tool => ({
				name: tool.name,
				description: tool.description,
			})),
		};
		this.append(runId, operationId, "run_start", { cause, metadata });
	}

	record(runId: string, operationId: string, event: AgentEvent): void {
		this.append(
			runId,
			operationId,
			"agent_event",
			event as unknown as Record<string, unknown>,
		);
	}

	finish(runId: string, operationId: string, status: string): void {
		this.append(runId, operationId, "run_finish", { status });
	}

	load(): TrajectoryEntry[] {
		const file = this.filePath();
		if (!existsSync(file)) return [];
		const entries: TrajectoryEntry[] = [];
		for (const line of readFileSync(file, "utf8").split("\n")) {
			if (!line.trim()) continue;
			try {
				const entry = JSON.parse(line) as TrajectoryEntry;
				if (entry.version === 1 && entry.sessionId === this.sessionId)
					entries.push(entry);
			} catch {
				// A torn final append does not invalidate the replayable prefix.
			}
		}
		return entries;
	}

	evaluate(): TrajectoryReport {
		return evaluateTrajectory(this.load());
	}

	private filePath(): string {
		return path.join(
			this.cwd,
			".logician",
			"trajectories",
			`${safeId(this.sessionId)}.jsonl`,
		);
	}

	private append(
		runId: string,
		operationId: string,
		kind: TrajectoryEntry["kind"],
		payload: Record<string, unknown>,
	): void {
		if (this.sessionId.startsWith("tui_")) return;
		const entry: TrajectoryEntry = {
			version: 1,
			sequence: ++this.sequence,
			timestamp: Date.now(),
			sessionId: this.sessionId,
			runId,
			operationId,
			kind,
			payload,
		};
		try {
			const file = this.filePath();
			mkdirSync(path.dirname(file), { recursive: true });
			// A leading newline isolates this record from a previously torn final write.
			appendFileSync(file, `\n${JSON.stringify(entry)}\n`, "utf8");
		} catch {
			// Observability persistence must never break the agent operation itself.
			this.sequence--;
		}
	}
}

export function evaluateTrajectory(
	entries: TrajectoryEntry[],
): TrajectoryReport {
	const events = entries.filter(entry => entry.kind === "agent_event");
	const eventTypes = events.map(entry => entry.payload.type);
	const outcome = [...events]
		.reverse()
		.find(entry => entry.payload.type === "run_outcome")?.payload;
	const taskState = [...events]
		.reverse()
		.find(entry => entry.payload.type === "task_state_update")?.payload.state as
		| {
				phase?: string;
				blockers?: string[];
				verification?: Array<{ passed: boolean }>;
		  }
		| undefined;
	const first = entries[0]?.timestamp ?? 0;
	const last = entries.at(-1)?.timestamp ?? first;
	const finished = entries.some(entry => entry.kind === "run_finish");
	const acceptancePassed =
		outcome?.status === "completed" &&
		(taskState?.blockers?.length ?? 0) === 0 &&
		(taskState?.verification?.every(item => item.passed) ?? true);
	return {
		events: events.length,
		durationMs: Math.max(0, last - first),
		providerRetries: eventTypes.filter(type => type === "agent_retry_start")
			.length,
		toolCalls: eventTypes.filter(type => type === "tool_execution_start")
			.length,
		toolFailures: events.filter(
			entry =>
				entry.payload.type === "tool_execution_end" &&
				entry.payload.isError === true,
		).length,
		loopEscapes: eventTypes.filter(
			type => type === "loop_detected" || type === "harness_intervention",
		).length,
		compactions: eventTypes.filter(type => type === "compaction").length,
		acceptancePassed,
		prematureStop:
			outcome?.status === "completed" &&
			Boolean(taskState) &&
			taskState?.phase !== "handoff" &&
			!acceptancePassed,
		replayComplete: entries.length === 0 || finished,
	};
}

export type InjectedFault =
	| "rate_limit"
	| "timeout"
	| "context_full"
	| "malformed_response";

/** Deterministic backend adapter for exercising the real recovery loop. */
export class FaultInjectingBackend implements LLMBackend {
	private cursor = 0;
	readonly model: string;

	constructor(
		private readonly backend: LLMBackend,
		private readonly faults: InjectedFault[],
	) {
		this.model = backend.model;
	}

	async generate(
		messages: Record<string, unknown>[],
		options?: GenerateOptions,
	): Promise<LLMResponse> {
		const fault = this.faults[this.cursor++];
		if (fault === "rate_limit")
			throw new BackendError({
				category: "rate_limit",
				message: "injected rate limit",
				status: 429,
			});
		if (fault === "timeout")
			throw new BackendError({
				category: "transient",
				message: "injected timeout",
			});
		if (fault === "context_full")
			throw new BackendError({
				category: "context_full",
				message: "injected context overflow",
			});
		if (fault === "malformed_response")
			return {
				content: null,
				toolCalls: [],
				stopReason: "error",
				errorMessage: "injected malformed response",
			};
		return this.backend.generate(messages, options);
	}

	withModel(model: string): LLMBackend {
		return new FaultInjectingBackend(
			this.backend.withModel(model),
			this.faults.slice(this.cursor),
		);
	}

	withEndpoint(model: string, baseUrl: string): LLMBackend {
		const backend =
			this.backend.withEndpoint?.(model, baseUrl) ??
			this.backend.withModel(model);
		return new FaultInjectingBackend(backend, this.faults.slice(this.cursor));
	}
}
