import type {
	ExtensionAPI,
	ExtensionDefinition,
	ExtensionEvent,
} from "./types.ts";

export interface CompatibilityAdapter {
	hasHandlers(event: string): boolean;
	emitToolCall(input: {
		toolCallId: string;
		toolName: string;
		input: Record<string, unknown>;
	}): Promise<{
		input: Record<string, unknown>;
		block?: boolean;
		reason?: string;
	}>;
	emitToolResult(input: {
		toolCallId: string;
		toolName: string;
		input: Record<string, unknown>;
		content: Array<{ type: string; text: string }>;
		isError: boolean;
		details?: Record<string, unknown>;
	}): Promise<{
		toolCallId: string;
		toolName: string;
		input: Record<string, unknown>;
		content: Array<{ type: string; text: string }>;
		isError: boolean;
		details?: Record<string, unknown>;
	}>;
	emitFromLogician(event: ExtensionEvent): Promise<{
		messages?: unknown[];
		systemPrompt?: string;
	}>;
	emitInputEvent(
		text: string,
		images: unknown[],
		source: "interactive" | "rpc" | "extension",
	): Promise<{
		action: "continue" | "transform" | "handled";
		text?: string;
		images?: unknown[];
	} | null>;
	emitUserBashEvent(
		command: string,
		excludeFromContext: boolean,
	): Promise<{
		action: "continue" | "intercept" | "replace";
		result?: { output: string; exitCode: number; cancelled: boolean };
		operations?: unknown;
	} | null>;
	emitProjectTrustEvent(cwd: string): Promise<{
		trusted: "yes" | "no" | "undecided";
		remember?: boolean;
	} | null>;
	getRegisteredTools(): Array<{ name: string; description: string }>;
	getRegisteredCommands(): Array<{ name: string; description?: string }>;
	getApi(): unknown;
}

export interface CompatibilityAdapterFactory {
	matches(definition: ExtensionDefinition, source: string | null): boolean;
	create(input: {
		api: ExtensionAPI;
		cwd: string;
		sessionId: string;
		runtime?: unknown;
	}): CompatibilityAdapter;
}
