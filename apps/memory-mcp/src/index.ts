#!/usr/bin/env bun
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { createMemoryStore } from "@logician/memory";
import { createMemoryMcpServer } from "./server.js";

function option(name: string): string | undefined {
	const index = process.argv.indexOf(name);
	return index >= 0 ? process.argv[index + 1] : undefined;
}

const workspace =
	option("--workspace") || process.env.LOGICIAN_MEMORY_WORKSPACE;
if (!workspace) {
	console.error(
		"logician-memory-mcp requires --workspace <path> or LOGICIAN_MEMORY_WORKSPACE",
	);
	process.exit(2);
}

const resolvedWorkspace = resolve(workspace);
const dbPath = resolve(
	option("--db") ||
		process.env.LOGICIAN_MEMORY_DB ||
		resolve(resolvedWorkspace, ".logician", "memory.db"),
);
const store = createMemoryStore(dbPath);
store.setCurrentWorkspace(resolvedWorkspace);
const server = createMemoryMcpServer(store);

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", async line => {
	if (!line.trim()) return;
	try {
		const response = await server.handle(JSON.parse(line));
		if (response) process.stdout.write(`${JSON.stringify(response)}\n`);
	} catch (error) {
		process.stdout.write(
			`${JSON.stringify({
				jsonrpc: "2.0",
				id: null,
				error: {
					code: -32700,
					message: error instanceof Error ? error.message : String(error),
				},
			})}\n`,
		);
	}
});

function close(): void {
	store.close();
}

lines.on("close", close);
process.once("SIGINT", () => {
	close();
	process.exit(0);
});
process.once("SIGTERM", () => {
	close();
	process.exit(0);
});
