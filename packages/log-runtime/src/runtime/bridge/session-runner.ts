/** SessionRunner: stable API for agent turn orchestration.

 * Extracts the runMessage/continuation lifecycle from AgentBridge behind
 * a clean interface. The runner owns the turn lifecycle (begin/end),
 * reasoner integration and session management —
 * exposing only the public operations AgentBridge needs.
 *
 * Public API:
 *   runner.submit(message)   → starts a full turn (runMessage path)
 *   runner.continue()         → runs the queued continuation
 *
 * The runner coordinates with AgentBridge through callback slots
 * so bridge remains the owner of session/config/events state.
 */

import { randomUUID } from "node:crypto";
import type { LLMBackend } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import type { AgentSession } from "@logician/log-core/session";
import type { RepositoryMap } from "../../capabilities/repository-map/repository-map.ts";
import {
	formatSkillActivationNotice,
	SkillActivationSession,
} from "../../capabilities/skills/activation.ts";
import {
	formatSkillInvocation,
	type Skill,
} from "../../capabilities/skills/loader.ts";
import type { RuntimeEventBus } from "../events/runtime-event-bus.ts";
import type { AgentCoordinator } from "./application/agent-coordinator.ts";

// ── Callback slots — filled by AgentBridge at construction ────────────────

/**
 * Callbacks injected by AgentBridge to give the runner access to
 * bridge-owned state (session, config, event bus, etc.).
 */
export interface SessionRunnerCallbacks {
	/** Emit a RuntimeEvent through the bridge's event bus. */
	emit(event: RuntimeEvent): void;
	/** Report an error through the bridge's diagnostic bus. */
	reportError(
		error: unknown,
		context: { component: string; operation: string; recoverable: boolean },
	): void;
	/** Get the current AgentSession (null if not yet created). */
	getSession(): AgentSession | null;
	/** Get or create the AgentSession. */
	ensureSession(): AgentSession;
	/** Get the bridge-owned session id used for run correlation. */
	getSessionId(): string;
	/** Get the current AgentConfig system prompt. */
	getSystemPrompt(): string | undefined;
	/** Get trusted skills discovered by the runtime. */
	getSkills(): Skill[];
	/** Render repository-map context for the given message, if enabled. */
	renderRepositoryContext(message: string): string | undefined;
	/** Publish current context-token usage (bridge owns the measurement). */
	publishUsage(): void;
}

// ── SessionRunner ─────────────────────────────────────────────────────────

/**
 * Orchestrates one agent turn: reasoner → prompt → continuation. Encapsulates
 * the full runMessage flow and the continuation
 * flow behind stable methods.
 *
 * All persistent state (session, config, events) lives in AgentBridge and is
 * accessed via the callback slots.
 */
export class SessionRunner {
	private readonly callbacks: SessionRunnerCallbacks;
	private readonly events: RuntimeEventBus;
	private readonly backend: LLMBackend;
	/** Live accessor — AgentCoordinator is assigned after async bridge init. */
	private readonly getAgentCoordinator: () => AgentCoordinator | null;
	/** Live accessor — RepositoryMap is a runtime-context getter on the bridge. */
	private readonly getRepositoryMap: () => RepositoryMap | undefined;
	private readonly skillActivations = new SkillActivationSession();

	constructor(deps: {
		callbacks: SessionRunnerCallbacks;
		events: RuntimeEventBus;
		backend: LLMBackend;
		getAgentCoordinator: () => AgentCoordinator | null;
		getRepositoryMap: () => RepositoryMap | undefined;
	}) {
		this.callbacks = deps.callbacks;
		this.events = deps.events;
		this.backend = deps.backend;
		this.getAgentCoordinator = deps.getAgentCoordinator;
		this.getRepositoryMap = deps.getRepositoryMap;
	}

	/**
	 * Submit a user message for a full agent turn.
	 *
	 * This is the core runMessage flow: repository-map context, reasoner
	 * advisory, session management, prompt, and continuation if queued.
	 * Startup hooks and MCP background loading are
	 * delegated to the bridge (which owns those concerns and runs them
	 * before calling submit).
	 */
	async submit(message: string): Promise<void> {
		const runId = `run_${randomUUID()}`;
		const turnId = `turn_${randomUUID()}`;
		this.events.beginRun({
			sessionId: this.callbacks.getSessionId(),
			runId,
			turnId,
		});

		let persistentSystemPrompt: string | undefined;
		let turnSystemPrompt = this.callbacks.getSystemPrompt();
		let turnSucceeded = false;
		const activations = isFormattedSkillInvocation(message)
			? []
			: this.skillActivations.select(this.callbacks.getSkills(), message);

		try {
			const session = this.callbacks.ensureSession();

			// Repository-map context for the initial message.
			const repositoryContext = this.callbacks.renderRepositoryContext(message);
			if (repositoryContext) {
				persistentSystemPrompt = this.callbacks.getSystemPrompt();
				turnSystemPrompt = `${persistentSystemPrompt}\n\n${repositoryContext}`;
				session.configure({ systemPrompt: turnSystemPrompt });
			}

			// Reasoner advisory.
			const agentCoordinator = this.getAgentCoordinator();
			if (agentCoordinator) {
				const advisory = await agentCoordinator.runReasoner(
					message,
					this.backend,
				);
				if (advisory) {
					persistentSystemPrompt ??= this.callbacks.getSystemPrompt();
					turnSystemPrompt = `${turnSystemPrompt}\n\nA structured reasoner produced the following advisory analysis for this turn. Verify it, use tools as needed, and do not mention this internal advisory unless useful:\n\n${advisory}`;
					session.configure({ systemPrompt: turnSystemPrompt });
				}
			}

			if (activations.length) {
				this.callbacks.emit({
					type: "notice",
					level: "info",
					label: "Skills",
					text: formatSkillActivationNotice(activations),
				});
			}

			this.callbacks.emit({ type: "turn_start", turnId });
			await session.prompt(message, {
				contextContributions: activations.map(activation => ({
					source: `skill:${activation.skill.name}`,
					priority: activation.score,
					messages: [
						{
							role: "system",
							content: formatSkillInvocation(activation.skill),
						},
					],
				})),
			});
			turnSucceeded = true;
		} catch (err: unknown) {
			const error = err as Error;
			this.callbacks.reportError(error, {
				component: "agent-runtime",
				operation: "run-message",
				recoverable: false,
			});
			throw error;
		} finally {
			try {
				if (persistentSystemPrompt !== undefined) {
					this.callbacks
						.getSession()
						?.configure({ systemPrompt: persistentSystemPrompt });
				}
				this.callbacks.publishUsage();
				this.callbacks.emit({ type: "turn_end", turnId });

				const session = this.callbacks.getSession();
				const hasQueuedContinuation =
					turnSucceeded && (session?.getQueues().nextTurn.length ?? 0) > 0;
				if (hasQueuedContinuation) {
					this.skillActivations.continueWith(activations);
					await this.continue();
				} else {
					this.callbacks.emit({ type: "phase", state: "ready" });
				}
			} finally {
				if (this.callbacks.getSession()) {
					this.events.endRun();
				}
			}
		}
	}

	/**
	 * Run a queued continuation.
	 *
	 * Renders an empty repository query and runs the queued next-turn message.
	 * The continuation owns the terminal ready event (no READY is emitted
	 * between an interrupted turn and the queued replacement).
	 */
	async continue(): Promise<void> {
		const session = this.callbacks.getSession();
		if (!session) return;

		const repoQuery = this.getRepositoryMap()?.render("");
		session.setRepositoryQuery(repoQuery);
		const activations = this.skillActivations.select(
			this.callbacks.getSkills(),
			"continue",
		);
		await session.runQueuedContinuation(undefined, repoQuery, {
			contextContributions: activations.map(activation => ({
				source: `skill:${activation.skill.name}`,
				priority: activation.score,
				messages: [
					{
						role: "system",
						content: formatSkillInvocation(activation.skill),
					},
				],
			})),
		});
	}
}

function isFormattedSkillInvocation(message: string): boolean {
	return /^\s*<skill\s+name="[^"]+"/u.test(message);
}
