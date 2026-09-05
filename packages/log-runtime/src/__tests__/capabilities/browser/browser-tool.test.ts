// ── Browser tool tests ───────────────────────────────────────────────────────

import { test, describe } from "bun:test";
import assert from "node:assert/strict";

// ── Stub BrowserManager ───────────────────────────────────────────────────────

class StubBrowserManager {
	tabs = new Map<string, { url: string }>();
	clicked: string[] = [];
	typed: Array<[string, string]> = [];
	filled: Array<[string, string]> = [];
	keyboard: string[] = [];
	screenshots = 0;
	evaluations: string[] = [];

	async open(name: string, url?: string) {
		this.tabs.set(name, { url: url ?? "" });
		return { name, url: url ?? "", title: "", closed: false };
	}

	async goto(name: string, _url: string) {
		const t = this.tabs.get(name);
		if (t) t.url = _url;
	}

	async getUrl(name: string) {
		return this.tabs.get(name)?.url ?? "";
	}

	async getTitle(_name: string) {
		return "";
	}

	async click(_name: string, selector: string) {
		this.clicked.push(selector);
	}

	async type(_name: string, selector: string, text: string) {
		this.typed.push([selector, text]);
	}

	async fill(_name: string, selector: string, value: string) {
		this.filled.push([selector, value]);
	}

	async press(_name: string, key: string) {
		this.keyboard.push(key);
	}

	async screenshot(_name: string) {
		this.screenshots++;
		return Buffer.from("fake");
	}

	async evaluate(_name: string, expression: string) {
		this.evaluations.push(expression);
		return null;
	}

	async ariaSnapshot(_name: string) {
		return "{}";
	}

	async waitForSelector(_name: string, _selector: string) {
		return true;
	}

	async waitForUrl(_name: string, _url: string) {
		return true;
	}

	listTabs(): string[] {
		return [...this.tabs.keys()];
	}

	async closeTab(name: string) {
		this.tabs.delete(name);
	}

	async close() {
		this.tabs.clear();
	}
}

// Create the tool with a stub (type-cast for test isolation).
function makeTool(manager: StubBrowserManager) {
	return import(
		"../../../capabilities/browser/browser-tool.ts"
	).then((m) => {
		// @ts-expect-error stub satisfies BrowserManager interface
		return m.createBrowserTool({ manager });
	});
}

function makeCtx() {
	return {} as import("@logician/log-core").ToolContext;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("browser tool", () => {
	test("open creates a tab", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "test" }, makeCtx());
		assert.ok(manager.tabs.has("test"));
	});

	test("open defaults to 'default'", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open" }, makeCtx());
		assert.ok(manager.tabs.has("default"));
	});

	test("goto navigates tab", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "goto", name: "t", url: "https://example.com" },
			makeCtx(),
		);
		assert.equal(manager.tabs.get("t")?.url, "https://example.com");
	});

	test("click records selector", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "click", name: "t", selector: "#submit" },
			makeCtx(),
		);
		assert.equal(manager.clicked[0], "#submit");
	});

	test("type records selector and text", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "type", name: "t", selector: "#q", text: "hello" },
			makeCtx(),
		);
		assert.equal(manager.typed[0][0], "#q");
		assert.equal(manager.typed[0][1], "hello");
	});

	test("fill records selector and value", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "fill", name: "t", selector: "#email", value: "a@b.com" },
			makeCtx(),
		);
		assert.equal(manager.filled[0][0], "#email");
		assert.equal(manager.filled[0][1], "a@b.com");
	});

	test("press records key", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "press", name: "t", key: "Enter" },
			makeCtx(),
		);
		assert.equal(manager.keyboard[0], "Enter");
	});

	test("screenshot increments counter", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute({ operation: "screenshot", name: "t" }, makeCtx());
		assert.equal(manager.screenshots, 1);
	});

	test("evaluate records expression", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute(
			{ operation: "evaluate", name: "t", expression: "1 + 1" },
			makeCtx(),
		);
		assert.equal(manager.evaluations[0], "1 + 1");
	});

	test("listTabs returns tabs", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "a" }, makeCtx());
		await tool.execute({ operation: "open", name: "b" }, makeCtx());
		assert.ok(manager.listTabs().includes("a"));
		assert.ok(manager.listTabs().includes("b"));
	});

	test("close removes tab", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "t" }, makeCtx());
		await tool.execute({ operation: "close", name: "t" }, makeCtx());
		assert.ok(!manager.tabs.has("t"));
	});

	test("closeAll clears all tabs", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		await tool.execute({ operation: "open", name: "a" }, makeCtx());
		await tool.execute({ operation: "open", name: "b" }, makeCtx());
		await tool.execute({ operation: "closeAll" }, makeCtx());
		assert.equal(manager.tabs.size, 0);
	});

	test("unknown operation returns error string", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		const result = await tool.execute(
			{ operation: "nonexistent" },
			makeCtx(),
		);
		assert.ok(typeof result === "string");
		assert.ok((result as string).includes("Unknown operation"));
	});

	test("missing args returns error", async () => {
		const manager = new StubBrowserManager();
		const tool = await makeTool(manager);
		const result = await tool.execute(
			{ operation: "click", name: "t" },
			makeCtx(),
		);
		assert.ok(typeof result === "string");
		assert.ok((result as string).includes("click requires"));
	});
});
