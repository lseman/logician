import type { AgentEvent } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import { mapAgentEvent } from "../../events/event-mapping.ts";

export interface RuntimeActivityDependencies {
	emit: (event: RuntimeEvent) => void;
	runPhase: () => string | undefined;
}

/** Owns transient runtime activity and projects core events for clients. */
export class RuntimeActivity {
	private readonly dependencies: RuntimeActivityDependencies;
	private retry?: string;
	private repair?: string;
	private readonly activeSubagents = new Set<string>();
	private tokens = 0;
	private maxTokens?: number;

	constructor(dependencies: RuntimeActivityDependencies) {
		this.dependencies = dependencies;
	}

	handle(event: AgentEvent): void {
		if (event.type === "context_update") {
			this.tokens = event.tokens;
			this.maxTokens = event.maxTokens;
		}
		if (event.type === "agent_retry_start") {
			this.retry = `${event.attempt}/${event.maxRetries}`;
		} else if (event.type === "agent_retry_end") {
			this.retry = undefined;
		}
		if (event.type === "repair_nudge") this.repair = event.repairStage;
		if (event.type === "turn_start") this.repair = undefined;
		if (event.type === "subagent_start") {
			this.activeSubagents.add(event.agentId);
		} else if (event.type === "subagent_end") {
			this.activeSubagents.delete(event.agentId);
		}

		const mapped = mapAgentEvent(event);
		if (mapped) this.dependencies.emit(mapped);
		this.emitStatus();
	}

	setContext(
		tokens: number,
		maxTokens?: number,
	): { tokens: number; maxTokens?: number } {
		this.tokens = tokens;
		this.maxTokens = maxTokens ?? this.maxTokens;
		return this.context();
	}

	context(): { tokens: number; maxTokens?: number } {
		return { tokens: this.tokens, maxTokens: this.maxTokens };
	}

	resetContext(): void {
		this.tokens = 0;
	}

	private emitStatus(): void {
		const runPhase = this.dependencies.runPhase();
		if (!runPhase) return;
		this.dependencies.emit({
			type: "runtime_status",
			runPhase,
			retry: this.retry,
			repair: this.repair,
			activeSubagents: this.activeSubagents.size,
		});
	}
}
