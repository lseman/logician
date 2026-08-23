import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { McpClient } from "../../capabilities/mcp/client.ts";
import { McpServerRegistry } from "../../capabilities/mcp/mcp-server-registry.ts";

function withIsolatedMcpEnvironment(
	run: (home: string, workspace: string) => Promise<void>,
): Promise<void> {
	const root = mkdtempSync(path.join(tmpdir(), "logician-mcp-manager-"));
	const home = path.join(root, "home");
	const workspace = path.join(root, "workspace");
	mkdirSync(path.join(home, ".logician"), { recursive: true });
	mkdirSync(workspace, { recursive: true });

	const previous = {
		HOME: process.env.HOME,
		LOGICIAN_CONFIG: process.env.LOGICIAN_CONFIG,
		LOGICIAN_MCP_CONFIG: process.env.LOGICIAN_MCP_CONFIG,
		MCP_CONFIG: process.env.MCP_CONFIG,
	};
	process.env.HOME = home;
	delete process.env.LOGICIAN_CONFIG;
	delete process.env.LOGICIAN_MCP_CONFIG;
	delete process.env.MCP_CONFIG;

	return run(home, workspace).finally(() => {
		for (const [name, value] of Object.entries(previous)) {
			if (value === undefined) delete process.env[name];
			else process.env[name] = value;
		}
		rmSync(root, { recursive: true, force: true });
	});
}

void test("MCP snapshot merges plugin, global, and project servers with project precedence", async () => {
	await withIsolatedMcpEnvironment(async (home, workspace) => {
		const globalPath = path.join(home, ".logician", "settings.json");
		const projectPath = path.join(workspace, ".mcp.json");
		writeFileSync(
			globalPath,
			JSON.stringify({
				mcpServers: {
					global: { command: "global-mcp" },
					shared: { command: "global-shared" },
				},
			}),
		);
		writeFileSync(
			projectPath,
			JSON.stringify({
				mcpServers: {
					project: { command: "project-mcp" },
					shared: { command: "project-shared" },
				},
			}),
		);

		const manager = new McpServerRegistry({
			loadPluginConfigs: async () => ({
				plugin: { command: "plugin-mcp" },
				shared: { command: "plugin-shared" },
			}),
		});
		const snapshot = await manager.getSnapshot(workspace);
		const byName = Object.fromEntries(
			snapshot.servers.map(server => [server.serverName, server]),
		);

		assert.deepEqual(Object.keys(byName).sort(), [
			"global",
			"plugin",
			"project",
			"shared",
		]);
		assert.equal(byName.shared.server.command, "project-shared");
		assert.equal(byName.plugin.configPath, undefined);
		assert.equal(byName.global.configPath, globalPath);
		assert.equal(byName.project.configPath, projectPath);
		assert.equal(snapshot.configPath, projectPath);
	});
});

void test("MCP snapshot includes loaded plugin-provided servers", async () => {
	await withIsolatedMcpEnvironment(async (_home, workspace) => {
		const manager = new McpServerRegistry({
			loadPluginConfigs: async () => ({
				plugin_example_server: { command: "plugin-mcp" },
			}),
		});
		const client: McpClient = {
			name: "plugin_example_server",
			initialize: async () => {},
			listTools: async () => [
				{
					name: "search",
					description: "Search",
					inputSchema: {},
				},
			],
			callTool: async () => ({}),
			close: () => {},
		};
		(manager as unknown as { clients: McpClient[] }).clients = [client];

		const snapshot = await manager.getSnapshot(workspace);

		assert.equal(snapshot.servers.length, 1);
		assert.equal(snapshot.servers[0]?.serverName, "plugin_example_server");
		assert.equal(snapshot.servers[0]?.loaded, true);
		assert.equal(snapshot.servers[0]?.toolCount, 1);
		assert.deepEqual(snapshot.loadedServers, {
			plugin_example_server: { toolCount: 1 },
		});
	});
});

void test("MCP snapshot asks live servers for tools concurrently", async () => {
	await withIsolatedMcpEnvironment(async (_home, workspace) => {
		let started = 0;
		let release!: () => void;
		const gate = new Promise<void>(resolve => {
			release = resolve;
		});
		const client = (name: string): McpClient => ({
			name,
			initialize: async () => {},
			listTools: async () => {
				started++;
				await gate;
				return [];
			},
			callTool: async () => ({}),
			close: () => {},
		});
		const manager = new McpServerRegistry({
			loadPluginConfigs: async () => ({
				first: { command: "first" },
				second: { command: "second" },
			}),
		});
		(manager as unknown as { clients: McpClient[] }).clients = [
			client("first"),
			client("second"),
		];

		const snapshotPromise = manager.getSnapshot(workspace);
		await new Promise<void>(resolve => setImmediate(resolve));
		assert.equal(started, 2);
		release();
		await snapshotPromise;
	});
});

void test("toggling a global MCP from a project workspace updates its defining file", async () => {
	await withIsolatedMcpEnvironment(async (home, workspace) => {
		const globalPath = path.join(home, ".logician", "settings.json");
		const projectPath = path.join(workspace, ".mcp.json");
		writeFileSync(
			globalPath,
			`{
				// Explicit inference choices must survive an MCP toggle.
				"inferenceMode": "thinking-coding",
				"thinkingLevel": "xhigh",
				"mcpServers": { "global": { "command": "global-mcp" } }
			}`,
		);
		writeFileSync(
			projectPath,
			JSON.stringify({
				mcpServers: { project: { command: "project-mcp" } },
			}),
		);
		const manager = new McpServerRegistry({
			loadPluginConfigs: async () => ({}),
		});

		await manager.setServerEnabled("global", false, workspace);

		const global = JSON.parse(readFileSync(globalPath, "utf8"));
		const project = JSON.parse(readFileSync(projectPath, "utf8"));
		assert.equal(global.mcpServers.global.enabled, false);
		assert.equal(global.inferenceMode, "thinking-coding");
		assert.equal(global.thinkingLevel, "xhigh");
		assert.equal(project.mcpServers.global, undefined);
	});
});
