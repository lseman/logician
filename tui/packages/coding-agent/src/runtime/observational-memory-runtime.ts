import type { Message, Tool } from "@logician/agent-core";
import type { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import {
	createMemorySearchTool,
	createMemorySystem,
	createRecallTool,
	hashId,
	MEMORY_SEARCH_TOOL_NAME,
	type MemoryStore,
	type MemoryStoreEvent,
	RECALL_TOOL_NAME,
} from "@logician/observational-memory";
import path from "node:path";
import { estimateTokens } from "@logician/agent-core/core/messages.ts";

export interface ObservationalMemoryRuntimeOptions {
	model: string;
	apiKey: string;
	baseUrl: string;
	cwd: string;
	eventBus: ExtensionEventBus;
	getMessages: () => Message[];
	onStoreEvent: (event: MemoryStoreEvent) => void;
}

export interface ObservationalMemoryRuntime {
	store: MemoryStore;
	tools: Tool[];
	dispose: () => void;
}

export function createObservationalMemoryRuntime(
	options: ObservationalMemoryRuntimeOptions,
): ObservationalMemoryRuntime {
	const memorySystem = createMemorySystem({
		model: options.model || "gpt-4o",
		apiKey: options.apiKey,
		baseUrl: options.baseUrl,
		persistencePath: path.join(
			options.cwd,
			".logician/observational-memory",
			"memory.json",
		),
	});
	const store = memorySystem.store;
	store.load();
	const unsubscribeStore = store.subscribe(options.onStoreEvent);
	const sourceEntries = () =>
		options
			.getMessages()
			.filter(
				(message) =>
					typeof message.content === "string" && message.content.trim(),
			)
			.map((message, index) => ({
				id: hashId(`${index}:${message.role}:${message.content}`),
				role: message.role,
				content: String(message.content),
				tokenCount: estimateTokens(String(message.content)),
			}));
	const unsubHooks = memorySystem.registerHooks(options.eventBus, {
		getSourceEntries: sourceEntries,
		getRetrievalContext: () =>
			options
				.getMessages()
				.filter(
					(message) =>
						(message.role === "user" || message.role === "assistant") &&
						typeof message.content === "string",
				)
				.slice(-4)
				.map((message) => String(message.content).slice(0, 1_000))
				.join("\n"),
	});
	const recallHandler = createRecallTool({
		memoryStore: store,
		sourceEntries: () =>
			options
				.getMessages()
				.filter(
					(message) =>
						typeof message.content === "string" && message.content.trim(),
				)
				.map((message, index) => ({
					id: hashId(`${index}:${message.role}:${message.content}`),
					type: "message",
					origin: message.role,
					timestamp: new Date(
						message.timestamp ?? Date.now(),
					).toISOString(),
					content: String(message.content),
				})),
	});
	const memorySearchHandler = createMemorySearchTool(store);
	const tools: Tool[] = [
		{
			name: MEMORY_SEARCH_TOOL_NAME,
			readOnly: true,
			executionMode: "parallel",
			description: "Search active observational memories and reflections by topic",
			parameters: {
				type: "object",
				properties: {
					query: { type: "string", minLength: 1 },
					limit: { type: "number", minimum: 1, maximum: 20 },
				},
				required: ["query"],
			},
			execute: async (args: Record<string, unknown>) =>
				JSON.stringify(
					memorySearchHandler(
						typeof args.query === "string" ? args.query : "",
						typeof args.limit === "number" ? args.limit : undefined,
					),
				),
		},
		{
			name: RECALL_TOOL_NAME,
			readOnly: true,
			executionMode: "parallel",
			description: "Recover exact evidence for an observational memory ID",
			parameters: {
				type: "object",
				properties: { id: { type: "string", pattern: "^[a-f0-9]{12}$" } },
				required: ["id"],
			},
			execute: async (args: Record<string, unknown>) =>
				JSON.stringify(
					recallHandler(typeof args.id === "string" ? args.id : ""),
				),
		},
	];
	return {
		store,
		tools,
		dispose: () => {
			store.save();
			unsubHooks();
			unsubscribeStore();
		},
	};
}
