import type { AdaptiveContextLearningState } from "../context/adaptive-context-controller.ts";
import type { ContextContribution } from "../context/context-engine.ts";
import type { HarnessPhase } from "../events/runtime-state.ts";
import type { ExtensionRunner } from "../extensions/runner.ts";
import type { LLMBackend } from "../provider/backend.ts";
import type {
	AgentConfig,
	AgentHarnessStreamOptions,
} from "../types/config.ts";
import type {
	AgentHooks,
	EventHandler,
	Message,
	Tool,
} from "../types/messages.ts";

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

interface HarnessPluginHookContext {
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
	cwd?: string | undefined;
	maxIterations?: number | undefined;
	extensionRunner?: ExtensionRunner | undefined;
	modules?: HarnessModule[] | undefined;
	pluginHookFactory?: HarnessPluginHookFactory | undefined;
	pluginLifecycle?: HarnessPluginLifecycle | undefined;
	contextLearning?: {
		initialState?: AdaptiveContextLearningState | undefined;
		onStateChange?: ((state: AdaptiveContextLearningState) => void) | undefined;
	};
}

/** Request-scoped context supplied by the host for one user-initiated turn. */
export interface HarnessPromptOptions {
	contextContributions?: readonly ContextContribution[];
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
