import type {
	AgentConfig,
	LLMBackend,
	Message,
	QueueMode,
} from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	type AbortResult,
	estimateChatPayloadTokens,
	type SessionStore,
} from "@logician/log-core/runtime";
import { AgentSession } from "@logician/log-core/session";
import {
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
} from "../../../adapters/claude-code/hook-layer.ts";
import {
	runHookEvent,
	runSessionStartHooks,
} from "../../../adapters/claude-code/plugin-runtime.ts";
import type { ExtensionRegistry } from "../../../capabilities/extensions/extensions.ts";
import { getTasks } from "../../../capabilities/tasks/todo.ts";
import type { AgentBridgeOptions } from "../types.ts";

export interface ConversationSessionDependencies {
	config: () => AgentConfig;
	backend: LLMBackend;
	extensions: () => ExtensionRegistry;
	emit: (event: RuntimeEvent) => void;
	contextChanged: () => void;
	contextCompacted: (tokens: number) => void;
	compaction?: AgentBridgeOptions["compaction"];
}

/** Owns the live AgentSession and every invariant required to construct it. */
export class ConversationSession {
	private currentSession: AgentSession | null = null;
	private sessionId: string;
	private durableSession?: SessionStore;

	constructor(
		private readonly dependencies: ConversationSessionDependencies,
		initialSessionId: string,
	) {
		this.sessionId = initialSessionId;
	}

	get current(): AgentSession | null {
		return this.currentSession;
	}

	/** Internal migration seam for bridge tests that provide an in-memory session adapter. */
	replace(session: AgentSession | null): void {
		this.currentSession = session;
	}

	ensure(): AgentSession {
		if (this.currentSession) return this.currentSession;

		const { backend, emit, extensions } = this.dependencies;
		const config = this.dependencies.config();
		const session = new AgentSession({
			config: { ...config, taskLedger: { snapshot: getTasks } },
			backend,
			cwd: config.cwd,
			maxIterations: config.maxIterations,
			onEvent: event => emit(event as RuntimeEvent),
			extensionRunner: extensions().runner ?? undefined,
			pluginHookFactory: context =>
				createClaudeCodeHookLayer({
					enabled: context.enabled,
					sessionId: context.sessionId,
					transcriptPath: context.transcriptPath,
					cwd: context.cwd,
					getMatcherValue: toolName => {
						const tool = context.tools.find(item => item.name === toolName);
						return (
							tool?.hookAliases?.join("|") || claudeToolMatcherName(toolName)
						);
					},
				}),
			pluginLifecycle: {
				sessionStart: async (context, source) => {
					await runSessionStartHooks({
						source,
						session_id: context.sessionId,
						transcript_path: context.transcriptPath,
						cwd: context.cwd,
					});
				},
				sessionEnd: async (context, reason) => {
					await runHookEvent("SessionEnd", {
						session_id: context.sessionId,
						transcript_path: context.transcriptPath,
						cwd: context.cwd,
						reason,
					});
				},
				preCompact: async context => {
					await runHookEvent("PreCompact", {
						session_id: context.sessionId,
						transcript_path: context.transcriptPath,
						cwd: context.cwd,
					});
				},
				postCompact: async context => {
					await runHookEvent("PostCompact", {
						session_id: context.sessionId,
						transcript_path: context.transcriptPath,
						cwd: context.cwd,
					});
				},
			},
		});

		session.setSessionId(this.sessionId);
		if (this.durableSession) session.attachSession(this.durableSession);
		if (this.dependencies.compaction) {
			session.setAutoCompactionSettings(this.dependencies.compaction);
		}
		session.observe({
			settled: nextTurnCount => {
				if (nextTurnCount === 0) return;
				emit({
					type: "notice",
					level: "info",
					label: "Continuation",
					text: `${nextTurnCount} next-turn message(s) queued; continuation will start after settlement.`,
				});
			},
		});

		this.currentSession = session;
		return session;
	}

	use(sessionId: string, durableSession?: SessionStore): void {
		this.sessionId = sessionId;
		this.durableSession = durableSession;
		this.currentSession?.setSessionId(sessionId);
		if (durableSession) this.currentSession?.attachSession(durableSession);
	}

	restoreHistory(messages: Message[]): boolean {
		try {
			this.ensure().setHistory(messages);
			this.dependencies.contextChanged();
			return true;
		} catch {
			return false;
		}
	}

	steer(message: string): void {
		this.currentSession?.steer(message);
	}

	steerQueue(message: string): void {
		this.currentSession?.steerQueue(message);
	}

	steerNow(message: string): void {
		this.currentSession?.steerNow(message);
	}

	followUp(message: string): void {
		this.currentSession?.followUp(message);
	}

	nextTurn(message: string): void {
		this.currentSession?.nextTurn(message);
	}

	setSteeringMode(mode: QueueMode): void {
		this.currentSession?.setSteeringMode(mode);
	}

	setFollowUpMode(mode: QueueMode): void {
		this.currentSession?.setFollowUpMode(mode);
	}

	queues(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return (
			this.currentSession?.getQueues() ?? {
				steering: [],
				followUp: [],
				nextTurn: [],
			}
		);
	}

	flushSteeringNow(): number {
		return this.currentSession?.flushSteeringNow() ?? 0;
	}

	clearQueues(): ReturnType<ConversationSession["queues"]> {
		return this.currentSession?.clearQueues() ?? this.queues();
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.currentSession?.dropQueuedMessage(displayIndex);
	}

	async abort(): Promise<AbortResult | null> {
		return (await this.currentSession?.abort()) ?? null;
	}

	async compact(): Promise<{
		tokensSaved: number;
		tokensBefore: number;
		tokensAfter: number;
	} | null> {
		if (!this.currentSession) return null;
		const tokensSaved = await this.currentSession.compact();
		if (tokensSaved === null) return null;
		const tokensAfter = estimateChatPayloadTokens(this.currentSession.messages);
		const tokensBefore = tokensAfter + tokensSaved;
		this.dependencies.emit({
			type: "compaction",
			reason: "manual",
			tokensBefore,
			tokensAfter,
		});
		this.dependencies.contextCompacted(tokensAfter);
		return { tokensSaved, tokensBefore, tokensAfter };
	}

	fork(): string | null {
		return this.currentSession?.fork() ?? null;
	}

	async branchSummary(): Promise<string | null> {
		if (!this.currentSession) return null;
		const summary = await this.currentSession.branchSummary();
		this.dependencies.contextChanged();
		return summary;
	}

	rewind(): { messages: number; filesRestored: number } | null {
		try {
			const restored = this.currentSession?.rewind() ?? null;
			if (restored) this.dependencies.contextChanged();
			return restored;
		} catch {
			return null;
		}
	}

	discardBranch(): boolean {
		const discarded = this.currentSession?.discardBranch() ?? false;
		if (discarded) this.dependencies.contextChanged();
		return discarded;
	}

	clearAndDrop(): void {
		this.currentSession?.clearHistory();
		this.drop();
	}

	drop(): void {
		this.replace(null);
	}
}
