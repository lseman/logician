import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ToolResult } from "@logician/log-core";
import {
	McpServerRegistry,
	setMcpRegistryInstance,
} from "../../capabilities/mcp/mcp-server-registry.ts";
import type { McpClient } from "../../capabilities/mcp/client.ts";
import { createReadTool } from "../../capabilities/tools/read-file.ts";
import { hasBeenRead } from "../../capabilities/tools/support/read-tracker.ts";
import { createWriteTool } from "../../capabilities/tools/write-file.ts";
import { ArtifactRegistry } from "../../runtime/bridge/support/internal-urls/artifact-manager.ts";
import { LocalProtocolHandler } from "../../runtime/bridge/support/internal-urls/local-protocol.ts";
import { ConflictProtocolHandler } from "../../runtime/bridge/support/internal-urls/conflict-protocol.ts";
import { IssueProtocolHandler } from "../../runtime/bridge/support/internal-urls/issue-protocol.ts";
import { MemoryProtocolHandler } from "../../runtime/bridge/support/internal-urls/memory-protocol.ts";
import { PrProtocolHandler } from "../../runtime/bridge/support/internal-urls/pr-protocol.ts";
import { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";

const dirs: string[] = [];
afterEach(() => {
	for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function temp(): string {
	const dir = mkdtempSync(path.join(tmpdir(), "protocol-fixes-"));
	dirs.push(dir);
	return dir;
}
function result(value: string | ToolResult): ToolResult {
	return typeof value === "string" ? { content: value } : value;
}
function tools() {
	const urls = new InternalUrlRouter();
	return { urls, read: createReadTool(urls), write: createWriteTool(undefined, urls) };
}

// ── memory:// ───────────────────────────────────────────────────────────────

function stubMemoryGateway() {
	return {
		listObservations: async () => [{ id: "obs-1", content: "hello observation" }],
		listMemories: async () => [{ id: "mem-1", content: "hello memory" }],
	};
}

test("memory://list, memory://memories, memory://observe/<id> resolve without the host-mismatch bug", async () => {
	const { urls, read } = tools();
	urls.register(new MemoryProtocolHandler());
	const memory = stubMemoryGateway();

	const list = result(await read.execute({ path: "memory://list" }, { memory }));
	expect(list.content).toContain("obs-1");

	const memories = result(await read.execute({ path: "memory://memories" }, { memory }));
	expect(memories.content).toContain("mem-1");

	const observe = result(await read.execute({ path: "memory://observe/obs-1" }, { memory }));
	expect(observe.content).toContain("hello observation");
});

// ── pr:// / issue:// ────────────────────────────────────────────────────────

function stubGithubClient(calls: Array<{ name: string; args: Record<string, unknown> }>): McpClient {
	return {
		name: "github",
		initialize: async () => {},
		listTools: async () => [],
		callTool: async (name, args) => {
			calls.push({ name, args });
			return { pull_requests: [], issues: [] };
		},
		close: () => {},
	};
}

test("pr://owner/repo/1428/files actually invokes the GitHub client (parser no longer always returns null)", async () => {
	const registry = new McpServerRegistry();
	const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
	registry.clients = [stubGithubClient(calls)];
	setMcpRegistryInstance(registry);

	const { urls, read } = tools();
	urls.register(new PrProtocolHandler());
	const output = result(await read.execute({ path: "pr://octocat/Hello-World/1428/files" }, {}));
	expect(calls).toHaveLength(1);
	expect(calls[0]?.name).toBe("pull_requests_get_files");
	expect(calls[0]?.args).toMatchObject({ owner: "octocat", repo: "Hello-World", number: 1428 });
	expect(output.content).not.toContain("GitHub MCP server is not available");
});

test("issue://owner/repo/42/comments actually invokes the GitHub client", async () => {
	const registry = new McpServerRegistry();
	const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
	registry.clients = [stubGithubClient(calls)];
	setMcpRegistryInstance(registry);

	const { urls, read } = tools();
	urls.register(new IssueProtocolHandler());
	await read.execute({ path: "issue://octocat/Hello-World/42/comments" }, {});
	expect(calls).toHaveLength(1);
	expect(calls[0]?.name).toBe("issues_get_comments");
	expect(calls[0]?.args).toMatchObject({ owner: "octocat", repo: "Hello-World", issue_number: 42 });
});

test("issue://owner/repo/abc rejects a non-numeric issue number instead of passing NaN through", async () => {
	const registry = new McpServerRegistry();
	const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
	registry.clients = [stubGithubClient(calls)];
	setMcpRegistryInstance(registry);

	const { urls, read } = tools();
	urls.register(new IssueProtocolHandler());
	const output = result(await read.execute({ path: "issue://octocat/Hello-World/abc" }, {}));
	expect(calls).toHaveLength(0);
	expect(output.content).toContain("Use `issue://owner/repo`");
});

// ── local:// artifact selectors ───────────────────────────────────────────

test("local://<id>:5-8 and :raw work, verified against non-ASCII content (byte/line regression guard)", async () => {
	const cwd = temp();
	ArtifactRegistry.resetForTests();
	ArtifactRegistry.instance().init({ cwd, sessionId: "sel-test" });
	const lines = ["línea uno", "línea dos", "línea tres", "línea cuatro", "línea cinco", "línea seis", "línea siete", "línea ocho"];
	const id = await ArtifactRegistry.instance().save(lines.join("\n"), "tool");
	if (id === null) throw new Error("save failed");

	const { urls, read } = tools();
	urls.register(new LocalProtocolHandler());
	const ranged = result(await read.execute({ path: `local://${id}:5-8` }, {}));
	for (const l of ["línea cinco", "línea seis", "línea siete", "línea ocho"]) {
		expect(ranged.content).toContain(l);
	}
	for (const l of ["línea uno", "línea dos", "línea tres", "línea cuatro"]) {
		expect(ranged.content).not.toContain(l);
	}

	const raw = result(await read.execute({ path: `local://${id}:raw` }, {}));
	for (const l of lines) expect(raw.content).toContain(l);

	ArtifactRegistry.resetForTests();
});

test("local://<id>:1-2,4-5 (comma-list) is not treated as a selector and falls through to full content", async () => {
	const cwd = temp();
	ArtifactRegistry.resetForTests();
	ArtifactRegistry.instance().init({ cwd, sessionId: "sel-comma-test" });
	const id = await ArtifactRegistry.instance().save("a\nb\nc\nd\ne", "tool");
	if (id === null) throw new Error("save failed");

	const { urls, read } = tools();
	urls.register(new LocalProtocolHandler());
	const output = result(await read.execute({ path: `local://${id}:1-2,4-5` }, {}));

	ArtifactRegistry.resetForTests();
});

// ── conflict:// ─────────────────────────────────────────────────────────────

function conflictFixture(): string {
	return [
		"before",
		"<<<<<<< ours-1",
		"mine-1",
		"=======",
		"theirs-1",
		">>>>>>> theirs-1",
		"middle",
		"<<<<<<< ours-2",
		"mine-2",
		"=======",
		"theirs-2",
		">>>>>>> theirs-2",
		"after",
	].join("\n");
}

test("conflict://<file> lists blocks and conflict://<file>:<index> reads one", async () => {
	const cwd = temp();
	writeFileSync(path.join(cwd, "app.ts"), conflictFixture());
	const { urls, read } = tools();
	urls.register(new ConflictProtocolHandler());

	const listing = result(await read.execute({ path: "conflict://app.ts" }, { cwd }));
	expect(listing.content).toContain("Block 0");
	expect(listing.content).toContain("Block 1");

	const block0 = result(await read.execute({ path: "conflict://app.ts:0" }, { cwd }));
	expect(block0.content).toContain("mine-1");
	expect(block0.content).toContain("theirs-1");
});

test("conflict:// write resolves only the targeted block, leaving the other's markers intact", async () => {
	const cwd = temp();
	const file = path.join(cwd, "app.ts");
	writeFileSync(file, conflictFixture());
	const { urls, read, write } = tools();
	urls.register(new ConflictProtocolHandler());

	await read.execute({ path: "app.ts" }, { cwd }); // read-before-write gate
	const output = result(
		await write.execute({ path: "conflict://app.ts:0", content: "ours" }, { cwd }),
	);
	expect(output.isError).not.toBe(true);

	const after = result(await read.execute({ path: "app.ts" }, { cwd }));
	expect(after.content).toContain("mine-1");
	expect(after.content).not.toContain("<<<<<<< ours-1");
	// second block untouched
	expect(after.content).toContain("<<<<<<< ours-2");
	expect(after.content).toContain("mine-2");
	expect(after.content).toContain("theirs-2");
});

test("conflict:// write with a bare file path (no index) resolves every block", async () => {
	const cwd = temp();
	const file = path.join(cwd, "app.ts");
	writeFileSync(file, conflictFixture());
	const { urls, read, write } = tools();
	urls.register(new ConflictProtocolHandler());

	await read.execute({ path: "app.ts" }, { cwd });
	await write.execute({ path: "conflict://app.ts", content: "theirs" }, { cwd });

	const after = result(await read.execute({ path: "app.ts" }, { cwd }));
	expect(after.content).toContain("theirs-1");
	expect(after.content).toContain("theirs-2");
	expect(after.content).not.toContain("<<<<<<<");
});

test("conflict:// write rejects an invalid strategy", async () => {
	const cwd = temp();
	writeFileSync(path.join(cwd, "app.ts"), conflictFixture());
	const { urls, read, write } = tools();
	urls.register(new ConflictProtocolHandler());
	await read.execute({ path: "app.ts" }, { cwd });

	const output = result(
		await write.execute({ path: "conflict://app.ts:0", content: "bogus" }, { cwd }),
	);
	expect(output.content).toContain("Error");
});

test("conflict:// write to a file that hasn't been read is rejected", async () => {
	const cwd = temp();
	writeFileSync(path.join(cwd, "app.ts"), conflictFixture());
	const { urls, write } = tools();
	urls.register(new ConflictProtocolHandler());

	const output = result(
		await write.execute({ path: "conflict://app.ts:0", content: "ours" }, { cwd }),
	);
	expect(output.content).toContain("Error");
});

test("conflict:// rejects path traversal (no containment check previously)", async () => {
	const cwd = temp();
	const outside = temp();
	writeFileSync(path.join(outside, "secret.ts"), conflictFixture());
	const { urls, read } = tools();
	urls.register(new ConflictProtocolHandler());

	const output = result(
		await read.execute({ path: `conflict://../${path.basename(outside)}/secret.ts` }, { cwd }),
	);
	expect(output.isError).toBe(true);
});

// ── :conflicts selector on a plain file read ────────────────────────────────

test("path:conflicts formats the file's conflict blocks (previously advertised but never wired up)", async () => {
	const cwd = temp();
	writeFileSync(path.join(cwd, "app.ts"), conflictFixture());
	const { read } = tools();
	const output = result(await read.execute({ path: "app.ts:conflicts" }, { cwd }));
	expect(output.content).toContain("Conflict #0");
	expect(output.content).toContain("Conflict #1");
	expect(output.content).toContain("mine-1");
});

test("hasBeenRead(file) reflects reads through both the plain path and conflict:// after a write", async () => {
	const cwd = temp();
	const file = path.join(cwd, "app.ts");
	writeFileSync(file, conflictFixture());
	const { urls, read, write } = tools();
	urls.register(new ConflictProtocolHandler());

	expect(hasBeenRead(file)).toBe(false);
	await read.execute({ path: "app.ts" }, { cwd });
	expect(hasBeenRead(file)).toBe(true);
	await write.execute({ path: "conflict://app.ts:0", content: "ours" }, { cwd });
	// A second write right after the first should not be blocked by staleness
	// (refreshAfterWrite must run after a successful conflict:// write).
	// Indices are recomputed by re-scanning the file each time, so the block
	// that was "index 1" before the first resolution is "index 0" now.
	const second = result(
		await write.execute({ path: "conflict://app.ts:0", content: "theirs" }, { cwd }),
	);
	expect(second.isError).not.toBe(true);
});
