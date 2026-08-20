import type { ExtensionRunner } from "../extension/runner.ts";
import type { LLMBackend } from "../provider/backend.ts";
import type { HarnessPhase } from "../state/runtime-state.ts";
import type {
	AgentConfig,
	AgentHarnessStreamOptions,
} from "../types/types-config.ts";
import type {
	AgentHooks,
	EventHandler,
	Message,
	Tool,
} from "../types/types-messages.ts";

export interface HarnessObserver {
	event?: EventHandler;
	phaseChange?: (phase: HarnessPhase, previous: HarnessPhase) => void;
	settled?: (nextTurnCount: number) => void;
	queueChange?: (queues: HarnessQueues) => void;
}

/** Inert configuration and observation bundle composed before construction. */
export interface HarnessModule {
	name: string;
	config?: Partial<AgentConfig>;
	observers?: HarnessObserver[];
}

export interface HarnessCompatibilityHookContext {
	enabled: boolean;
	sessionId: string;
	transcriptPath: string;
	cwd: string;
	tools: Tool[];
}

export interface HarnessCompatibilityHookLayer {
	hooks?: AgentHooks;
	userPromptMessages(prompt: string): Promise<Message[]>;
}

export type HarnessCompatibilityHookFactory = (
	context: HarnessCompatibilityHookContext,
) => HarnessCompatibilityHookLayer;

export interface HarnessCompatibilityLifecycle {
	sessionStart(
		context: HarnessCompatibilityHookContext,
		source: string,
	): Promise<void>;
	sessionEnd(
		context: HarnessCompatibilityHookContext,
		reason: string,
	): Promise<void>;
	preCompact(context: HarnessCompatibilityHookContext): Promise<void>;
	postCompact(context: HarnessCompatibilityHookContext): Promise<void>;
}

export function defineHarnessModule(module: HarnessModule): HarnessModule {
	return module;
}

export interface AgentHarnessOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
	extensionRunner?: ExtensionRunner;
	modules?: HarnessModule[];
	compatibilityHookFactory?: HarnessCompatibilityHookFactory;
	compatibilityLifecycle?: HarnessCompatibilityLifecycle;
}

export interface HarnessTurnSnapshot {
	promptText: string;
	initialMessages: Message[];
	config: AgentConfig;
	streamOptions: AgentHarnessStreamOptions;
	signal: AbortSignal;
}

export interface HarnessQueues {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}

export interface AbortResult {
	clearedSteering: string[];
	clearedFollowUp: string[];
	clearedNextTurn: string[];
}
