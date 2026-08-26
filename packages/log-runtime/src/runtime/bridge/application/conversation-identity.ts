import type { AgentConfig } from "@logician/log-core";
import type { SessionStore } from "@logician/log-core/runtime";
import { createHookTranscriptPath, eventLogPathFor } from "../environment.ts";

interface IdentitySessionPort {
	use(sessionId: string, durableSession?: SessionStore): void;
}

interface IdentityEventPort {
	setSessionId(sessionId: string): void;
}

interface IdentityMemoryPort {
	onSessionChanged(sessionId: string, previousSessionId: string): void;
	resetSession(sessionId: string): void;
}

export interface ConversationIdentityDependencies {
	cwd: string;
	config: () => AgentConfig | undefined;
	sessions: () => IdentitySessionPort | undefined;
	events: () => IdentityEventPort | undefined;
	memory: () => IdentityMemoryPort | undefined;
}

/** Owns the identity shared by sessions, hooks, events, and memory. */
export class ConversationIdentity {
	private sessionId: string;
	private transcriptPath: string;
	private readonly dependencies: ConversationIdentityDependencies;

	constructor(
		initialSessionId: string,
		dependencies: ConversationIdentityDependencies,
	) {
		this.sessionId = initialSessionId;
		this.dependencies = dependencies;
		this.transcriptPath = createHookTranscriptPath(
			dependencies.cwd,
			initialSessionId,
		);
	}

	get id(): string {
		return this.sessionId;
	}

	get transcript(): string {
		return this.transcriptPath;
	}

	use(sessionId: string, durableSession?: SessionStore): boolean {
		if (!sessionId.trim()) return false;
		const previousSessionId = this.sessionId;
		this.sessionId = sessionId;
		this.refreshPaths();
		this.dependencies.events()?.setSessionId(sessionId);
		this.dependencies.sessions()?.use(sessionId, durableSession);
		this.dependencies.memory()?.onSessionChanged(sessionId, previousSessionId);
		return true;
	}

	reset(): void {
		this.refreshPaths();
		this.dependencies.memory()?.resetSession(this.sessionId);
	}

	private refreshPaths(): void {
		this.transcriptPath = createHookTranscriptPath(
			this.dependencies.cwd,
			this.sessionId,
		);
		const config = this.dependencies.config();
		if (!config) return;
		config.hookSessionId = this.sessionId;
		config.hookTranscriptPath = this.transcriptPath;
		config.eventLogPath = eventLogPathFor(this.transcriptPath);
	}
}
