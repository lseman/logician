import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import type { ExtensionRunner } from "../../system/extension/runner.ts";
import type {
	AgentConfig,
	AgentHarnessStreamOptions,
} from "../../system/types/types-config.ts";
import type {
	AgentHooks,
	EventHandler,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import type { HarnessPhase } from "../state/runtime-state.ts";

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

export interface HarnessPluginHookContext {
	enabled: boolean;
	sessionId: string;
	transcriptPath: string;
	cwd: string;
	tools: Tool[];
}

export interface HarnessPluginHookLayer {
	hooks?: AgentHooks;
	userPromptMessages(prompt: string): Promise<Message[]>;
}

export type HarnessPluginHookFactory = (
	context: HarnessPluginHookContext,
) => HarnessPluginHookLayer;

export interface HarnessPluginLifecycle {
	sessionStart(
		context: HarnessPluginHookContext,
		source: string,
	): Promise<void>;
	sessionEnd(context: HarnessPluginHookContext, reason: string): Promise<void>;
	preCompact(context: HarnessPluginHookContext): Promise<void>;
	postCompact(context: HarnessPluginHookContext): Promise<void>;
}

export function defineHarnessModule(module: HarnessModule): HarnessModule {
	return module;
}

export interface AgentSessionOptions {
	config: AgentConfig;
	backend: LLMBackend;
	cwd?: string;
	maxIterations?: number;
	extensionRunner?: ExtensionRunner;
	modules?: HarnessModule[];
	pluginHookFactory?: HarnessPluginHookFactory;
	pluginLifecycle?: HarnessPluginLifecycle;
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
