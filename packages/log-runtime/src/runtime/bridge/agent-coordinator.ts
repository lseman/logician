/** Coordinates optional reasoners, subagents, and heuristic evolution. */

import type { LLMBackend, Tool } from "@logician/log-core";
import type { AgentSession } from "@logician/log-core/harness";
import type { RuntimeEvent } from "@logician/log-protocol";
import {
	get_reasoner,
	getReasonerMeta,
	type ReasonerConfig,
} from "../../capabilities/reasoning/index.ts";
import { EohController } from "../eoh/controller.ts";
import { SubagentCoordinator } from "../subagent-coordinator.ts";

// ── Dependencies ───────────────────────────────────────────────────────────────

export interface AgentCoordinatorDeps {
	emit: (event: RuntimeEvent) => void;
	getBackend: () => LLMBackend | null;
	getBaseUrl: () => string;
	getCurrentModel: () => string;
	harness: AgentSession | null;
	cwd: string;
	projectTrusted: boolean;
	maxParallelAgents?: number;
	getEnabledPluginRoots: () => Array<{ name: string; installPath: string }>;
	getDefaultTools: () => Tool[];
	ensureSession: () => AgentSession;
	reportError: (error: unknown) => void;
}

// ── AgentCoordinator class ────────────────────────────────────────────────────

export class AgentCoordinator {
	private reasonerId: string;
	private reasonerConfig: ReasonerConfig;
	private readonly eohController: EohController;
	private readonly deps: AgentCoordinatorDeps;
	private readonly subagents: SubagentCoordinator;
	private injected = false;

	constructor(
		deps: AgentCoordinatorDeps,
		reasonerId?: string,
		reasonerConfig?: ReasonerConfig,
	) {
		this.deps = deps;
		this.reasonerId = reasonerId?.trim().toLowerCase() || "none";
		this.reasonerConfig = reasonerConfig ?? {};

		this.eohController = new EohController({
			cwd: deps.cwd,
			emit: event => deps.emit(event),
			getBaseUrl: deps.getBaseUrl,
			getCurrentModel: deps.getCurrentModel,
		});

		this.subagents = new SubagentCoordinator({
			config: () => ({ systemPrompt: "", tools: [] }) as any,
			backend: deps.getBackend()!,
			cwd: deps.cwd,
			projectTrusted: deps.projectTrusted,
			maxParallelAgents: deps.maxParallelAgents,
			getEnabledPluginRoots: deps.getEnabledPluginRoots,
			getDefaultTools: deps.getDefaultTools,
			onToolAdded: () => {},
			ensureSession: deps.ensureSession,
			emit: event => deps.emit(event),
			reportError: deps.reportError,
		});
	}

	// ── Reasoner ─────────────────────────────────────────────────────────────

	/**
	 * Run the configured pre-turn reasoner on a message.
	 * Returns the advisory text to inject into the system prompt, or empty string.
	 * Throws if the reasoner fails.
	 */
	async runReasoner(message: string, backend: LLMBackend): Promise<string> {
		if (this.reasonerId === "none") return "";

		const meta = getReasonerMeta(this.reasonerId);
		if (!meta) {
			throw new Error(`Unknown reasoner '${this.reasonerId}'.`);
		}

		this.emitNotice("info", "Reasoner", `Running ${meta.name} pre-reasoning`);

		const reasoner = get_reasoner(this.reasonerId, backend, {
			...meta.defaultConfig,
			...this.reasonerConfig,
		});

		const startedAt = Date.now();
		let trace: Awaited<ReturnType<typeof reasoner.solve>>;
		try {
			trace = await reasoner.solve(message);
		} catch (error) {
			this.emitNotice(
				"error",
				"Reasoner",
				`${meta.name} failed after ${Date.now() - startedAt}ms: ${error instanceof Error ? error.message : String(error)}`,
			);
			throw error;
		}

		this.emitNotice(
			"success",
			"Reasoner",
			`${meta.name} completed in ${Date.now() - startedAt}ms.`,
		);

		const advisory = [trace.reasoning, trace.answer]
			.map(part => part?.trim())
			.filter(Boolean)
			.join("\n\nProposed answer:\n");

		return advisory || "";
	}

	emitNotice(
		level: "info" | "error" | "success" | "warn",
		label: string,
		text: string,
	): void {
		this.deps.emit({
			type: "notice",
			level,
			label,
			text,
		} as RuntimeEvent);
	}

	/** Set the reasoner ID. */
	setReasonerId(reasonerId: string): void {
		const normalized = reasonerId.trim().toLowerCase();
		if (normalized !== "none" && !getReasonerMeta(normalized)) {
			throw new Error(`Unknown reasoner '${reasonerId}'.`);
		}
		this.reasonerId = normalized || "none";
	}

	/** Get the current reasoner ID. */
	getReasonerStatus(): string {
		return this.reasonerId;
	}

	// ── EoH ──────────────────────────────────────────────────────────────────

	/** EoH command: /eoh <file.py> [generations] | stop | status | best | reset */
	eohCommand(raw: string): string {
		return this.eohController.command(raw);
	}

	// ── Subagents ────────────────────────────────────────────────────────────

	/** Register the spawn_agent and spawn_agents tools bound to discovered definitions. */
	async injectSubagents(): Promise<void> {
		await this.subagents.inject();
		this.injected = true;
	}

	/** Directly invoke the spawn_agent tool without going through the LLM. */
	spawnAgentDirectly(task: string, agent?: string): void {
		this.subagents.spawnDirectly(task, agent);
	}

	/** Whether subagents have been injected. */
	isInjected(): boolean {
		return this.injected;
	}
}
