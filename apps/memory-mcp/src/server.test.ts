import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createMemoryStore, type MemoryStore } from "@logician/memory";
import { createMemoryMcpServer, MEMORY_MCP_TOOLS } from "./server.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
	for (const directory of temporaryDirectories.splice(0)) {
		rmSync(directory, { recursive: true, force: true });
	}
});

function fixture(workspace = "/workspace-a"): {
	store: MemoryStore;
	server: ReturnType<typeof createMemoryMcpServer>;
	dbPath: string;
} {
	const directory = mkdtempSync(join(tmpdir(), "logician-memory-mcp-"));
	temporaryDirectories.push(directory);
	const dbPath = join(directory, "memory.db");
	const store = createMemoryStore(dbPath);
	store.setCurrentWorkspace(workspace);
	return { store, server: createMemoryMcpServer(store), dbPath };
}

async function call(
	server: ReturnType<typeof createMemoryMcpServer>,
	name: string,
	args: Record<string, unknown>,
) {
	const response = await server.handle({
		jsonrpc: "2.0",
		id: 1,
		method: "tools/call",
		params: { name, arguments: args },
	});
	return response?.result as {
		isError?: boolean;
		content: Array<{ type: string; text: string }>;
		structuredContent?: Record<string, unknown>;
	};
}

describe("memory MCP protocol", () => {
	test("initializes and exposes only the five intentional tools", async () => {
		const { store, server } = fixture();
		const initialized = await server.handle({
			jsonrpc: "2.0",
			id: 1,
			method: "initialize",
		});
		const listed = await server.handle({
			jsonrpc: "2.0",
			id: 2,
			method: "tools/list",
		});

		expect(initialized?.result).toMatchObject({
			serverInfo: { name: "logician-memory" },
			capabilities: { tools: { listChanged: false } },
		});
		expect(MEMORY_MCP_TOOLS.map(tool => tool.name)).toEqual([
			"memory_search",
			"memory_get",
			"memory_save",
			"memory_observe",
			"memory_feedback",
		]);
		expect(
			((listed?.result as { tools?: unknown[] } | undefined)?.tools ?? [])
				.length,
		).toBe(5);
		store.close();
	});

	test("memory_save is durable and idempotent", async () => {
		const { store, server } = fixture();
		const args = {
			content: "Retries use bounded exponential backoff.",
			idempotencyKey: "agent-a:turn-4:decision-1",
			type: "architecture",
			strength: 8,
		};
		const first = await call(server, "memory_save", args);
		const second = await call(server, "memory_save", args);

		expect(first.isError).toBeUndefined();
		expect(store.list()).toHaveLength(1);
		const firstMemory = first.structuredContent?.memory as
			| { id: string }
			| undefined;
		const secondMemory = second.structuredContent?.memory as
			| { id: string }
			| undefined;
		expect(secondMemory?.id).toBe(firstMemory?.id);
		store.close();
	});

	test("memory_observe is idempotent across repeated deliveries", async () => {
		const { store, server } = fixture();
		const args = {
			sessionId: "session-1",
			idempotencyKey: "tool-call-9:post",
			hookType: "post_tool_use",
			toolName: "read_file",
			toolInput: { path: "src/auth.ts" },
			toolOutput: "Uses bounded retries",
		};
		await call(server, "memory_observe", args);
		await call(server, "memory_observe", args);

		expect(store.listObservations("session-1")).toHaveLength(1);
		store.close();
	});

	test("get and search never cross the configured workspace", async () => {
		const { store, server, dbPath } = fixture("/workspace-a");
		const saved = await call(server, "memory_save", {
			content: "Workspace A authentication convention",
			idempotencyKey: "auth-convention",
		});
		const id = (saved.structuredContent?.memory as { id?: string } | undefined)
			?.id;
		expect(id).toBeString();
		store.close();

		const foreignStore = createMemoryStore(dbPath);
		foreignStore.setCurrentWorkspace("/workspace-b");
		const foreignServer = createMemoryMcpServer(foreignStore);
		const get = await call(foreignServer, "memory_get", { ids: [id] });
		const search = await call(foreignServer, "memory_search", {
			query: "authentication",
		});

		expect(
			(get.structuredContent?.entries as unknown[] | undefined) ?? [],
		).toHaveLength(0);
		expect(
			(search.structuredContent?.entries as unknown[] | undefined) ?? [],
		).toHaveLength(0);
		foreignStore.close();
	});

	test("search uses canonical traced retrieval and feedback joins its trace", async () => {
		const { store, server } = fixture();
		for (let index = 0; index < 3; index++) {
			await call(server, "memory_save", {
				content: `Authentication retries use bounded backoff variant ${index}.`,
				idempotencyKey: `auth-${index}`,
				strength: 8,
			});
		}
		const search = await call(server, "memory_search", {
			query: "authentication retries bounded backoff",
			limit: 2,
		});
		const traceId = search.structuredContent?.traceId as string | undefined;
		const entries = search.structuredContent?.entries as unknown[] | undefined;
		expect(traceId).toBeString();
		if (!traceId) throw new Error("search did not return a retrieval trace");
		expect(entries).toHaveLength(2);
		expect(store.listRetrievalTraces(1)[0]?.id).toBe(traceId);

		const feedback = await call(server, "memory_feedback", {
			retrievalTraceId: traceId,
			taskId: "task-auth",
			idempotencyKey: "feedback-auth",
			outcome: { environmentPassed: true },
		});
		expect(feedback.isError).toBeUndefined();
		expect(store.listOutcomeReceipts(1)[0]?.retrievalTraceId).toBe(traceId);
		store.close();
	});

	test("tool validation is returned as an MCP tool error", async () => {
		const { store, server } = fixture();
		const result = await call(server, "memory_feedback", {
			retrievalTraceId: "missing",
			taskId: "task-1",
			idempotencyKey: "feedback-1",
			outcome: { environmentPassed: "yes" },
		});

		expect(result.isError).toBe(true);
		expect(result.content[0]?.text).toContain(
			"outcome.environmentPassed must be a boolean",
		);
		store.close();
	});
});
