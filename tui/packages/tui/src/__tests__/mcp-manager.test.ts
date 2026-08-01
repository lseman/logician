import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { McpManagerOverlay } from "../overlays/mcp-manager.ts";
import { PluginManagerOverlay } from "../overlays/plugin-manager.ts";
import { initTheme } from "../terminal/theme.ts";

function createOverlay(): McpManagerOverlay {
	try {
		initTheme("dark");
	} catch {
		// Theme may already be initialized by another test.
	}
	const overlay = new McpManagerOverlay();
	overlay.setSnapshot({
		servers: [
			{
				server_name: "agentmemory",
				command: "npx",
				type: "stdio",
				enabled: true,
			},
		],
	});
	return overlay;
}

void describe("McpManagerOverlay", () => {
	void it("can be reopened after closing with q", () => {
		const overlay = createOverlay();

		overlay.show();
		assert.deepEqual(overlay.handleInput("q"), { type: "close" });
		overlay.hide();
		assert.equal(overlay.isVisibleOverlay(), false);

		overlay.show();
		assert.equal(overlay.isVisibleOverlay(), true);
		assert.ok(overlay.render(80).join("\n").includes("agentmemory"));
		assert.deepEqual(overlay.handleInput("q"), { type: "close" });
	});
});

void describe("PluginManagerOverlay", () => {
	void it("can be reopened after closing with q", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [
				{
					plugin_id: "example@local",
					name: "Example",
					enabled: true,
				},
			],
		});

		overlay.show();
		assert.deepEqual(overlay.handleInput("q"), { type: "close" });
		overlay.hide();
		assert.equal(overlay.isVisibleOverlay(), false);

		overlay.show();
		assert.equal(overlay.isVisibleOverlay(), true);
		assert.ok(overlay.render(80).join("\n").includes("example@local"));
		assert.deepEqual(overlay.handleInput("q"), { type: "close" });
	});
});

void describe("McpManagerOverlay rendering", () => {
	void it("renders server list with correct format", () => {
		const overlay = createOverlay();
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("MCP Servers"));
		assert.ok(text.includes("agentmemory"));
		assert.ok(text.includes("(cmd)"));
		assert.ok(text.includes("0 tools"));
	});

	void it("renders multiple servers", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [
				{ server_name: "server1", command: "cmd1", type: "stdio", enabled: true },
				{ server_name: "server2", command: "cmd2", type: "http", enabled: false },
				{ server_name: "server3", command: "cmd3", type: "streamable-http", enabled: true },
			],
			loadedServers: {
				server1: { toolCount: 5 },
				server2: { toolCount: 0 },
				server3: { toolCount: 3 },
			},
		});
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("server1"));
		assert.ok(text.includes("server2"));
		assert.ok(text.includes("server3"));
		assert.ok(text.includes("5 tool(s)"));
		assert.ok(text.includes("3 tool(s)"));
		assert.ok(text.includes("(cmd)"));
		assert.ok(text.includes("(http)"));
	});

	void it("handles navigation with arrow keys", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [
				{ server_name: "a", command: "a", type: "stdio", enabled: true },
				{ server_name: "b", command: "b", type: "stdio", enabled: true },
				{ server_name: "c", command: "c", type: "stdio", enabled: true },
			],
		});
		overlay.show();
		assert.equal(overlay["selection"].index, 0);

		// Move down
		overlay.handleInput("\x1b[B");
		assert.equal(overlay["selection"].index, 1);

		// Move up
		overlay.handleInput("\x1b[A");
		assert.equal(overlay["selection"].index, 0);

		// Wrap to bottom (2 down from 0 = index 2)
		overlay.handleInput("\x1b[B");
		overlay.handleInput("\x1b[B");
		assert.equal(overlay["selection"].index, 2);

		// Wrap to top (1 up from 2 = index 1)
		overlay.handleInput("\x1b[A");
		assert.equal(overlay["selection"].index, 1);

		// One more up = wrap to 0
		overlay.handleInput("\x1b[A");
		assert.equal(overlay["selection"].index, 0);
	});

	void it("handles page up/down", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: Array.from({ length: 20 }, (_, i) => ({
				server_name: `s${i}`,
				command: `cmd${i}`,
				type: "stdio",
				enabled: true,
			})),
		});
		overlay.show();
		assert.equal(overlay["selection"].index, 0);

		// Page down
		overlay.handleInput("\x1b[6~");
		assert.ok(overlay["selection"].index >= 8);

		// Page up
		overlay.handleInput("\x1b[5~");
		assert.ok(overlay["selection"].index < 8);
	});

	void it("handles j/k navigation", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [
				{ server_name: "a", command: "a", type: "stdio", enabled: true },
				{ server_name: "b", command: "b", type: "stdio", enabled: true },
				{ server_name: "c", command: "c", type: "stdio", enabled: true },
			],
		});
		overlay.show();

		// j moves down
		overlay.handleInput("j");
		assert.equal(overlay["selection"].index, 1);

		// k moves up
		overlay.handleInput("k");
		assert.equal(overlay["selection"].index, 0);
	});

	void it("handles refresh action", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		assert.deepEqual(overlay.handleInput("r"), { type: "refresh" });
		assert.deepEqual(overlay.handleInput("R"), { type: "refresh" });
	});

	void it("handles enter to close", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		assert.deepEqual(overlay.handleInput("\r"), { type: "close" });
		assert.deepEqual(overlay.handleInput("\n"), { type: "close" });
	});

	void it("handles escape and ctrl-c", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		assert.deepEqual(overlay.handleInput("\x1b"), { type: "close" });
		assert.deepEqual(overlay.handleInput("\x03"), { type: "close" });
	});

	void it("renders empty state", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({ servers: [] });
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("No MCP servers configured"));
	});

	void it("renders busy indicator", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		overlay.setBusy("a");
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("updating"));
	});

	void it("renders custom message", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		overlay.setMessage("Custom message");
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("Custom message"));
	});

	void it("returns null for input when not visible", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		assert.equal(overlay.handleInput("q"), null);
		assert.equal(overlay.handleInput("\r"), null);
	});

	void it("renders with config path", () => {
		const overlay = new McpManagerOverlay();
		overlay.setSnapshot({
			configPath: "/home/user/.logician/mcp.json",
			servers: [{ server_name: "a", command: "a", type: "stdio", enabled: true }],
		});
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("Config:"));
		assert.ok(text.includes(".logician/mcp.json"));
	});
});

void describe("PluginManagerOverlay rendering", () => {
	void it("renders plugin list with correct format", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [
				{ plugin_id: "test@local", name: "Test Plugin", enabled: true },
			],
		});
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("Plugins"));
		assert.ok(text.includes("test@local"));
	});

	void it("renders multiple plugins", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [
				{ plugin_id: "a@local", name: "Plugin A", enabled: true },
				{ plugin_id: "b@local", name: "Plugin B", enabled: false },
			],
		});
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("a@local"));
		assert.ok(text.includes("b@local"));
	});

	void it("handles navigation with arrow keys", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [
				{ plugin_id: "a@local", name: "A", enabled: true },
				{ plugin_id: "b@local", name: "B", enabled: true },
				{ plugin_id: "c@local", name: "C", enabled: true },
			],
		});
		overlay.show();
		assert.equal(overlay["selection"].index, 0);

		overlay.handleInput("\x1b[B");
		assert.equal(overlay["selection"].index, 1);

		overlay.handleInput("\x1b[A");
		assert.equal(overlay["selection"].index, 0);
	});

	void it("handles refresh action", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [{ plugin_id: "a@local", name: "A", enabled: true }],
		});
		overlay.show();
		assert.deepEqual(overlay.handleInput("r"), { type: "refresh" });
		assert.deepEqual(overlay.handleInput("R"), { type: "refresh" });
	});

	void it("renders empty state", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({ plugins: [] });
		overlay.show();
		const lines = overlay.render(80);
		const text = lines.join("\n");
		assert.ok(text.includes("No plugins installed"));
	});

	void it("returns null for input when not visible", () => {
		const overlay = new PluginManagerOverlay();
		overlay.setSnapshot({
			plugins: [{ plugin_id: "a@local", name: "A", enabled: true }],
		});
		assert.equal(overlay.handleInput("q"), null);
	});
});
