import type { ExtensionRunner } from "../../extensions/index.ts";
import type { LLMBackend } from "../core/backend.ts";
import type {
	AgentConfig,
	AgentHarnessStreamOptions,
	Message,
} from "../types/index.ts";

export interface AgentHarnessOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
	extensionRunner?: ExtensionRunner;
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
