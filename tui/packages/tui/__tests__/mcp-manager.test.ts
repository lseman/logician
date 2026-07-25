import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { McpManagerOverlay } from "../src/components/mcp-manager.ts";
import { PluginManagerOverlay } from "../src/components/plugin-manager.ts";
import { initTheme } from "../src/layers/theme/theme.ts";

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
