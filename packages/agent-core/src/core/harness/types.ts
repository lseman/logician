import type { ExtensionRunner } from "../extension/index.ts";
import type { LLMBackend } from "../provider/backend.ts";
import type { HarnessPhase } from "../state/runtime-state.ts";
import type {
	AgentConfig,
	AgentHarnessStreamOptions,
	EventHandler,
	Message,
} from "../types/index.ts";

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
