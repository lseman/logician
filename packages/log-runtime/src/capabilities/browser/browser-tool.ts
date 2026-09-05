// ── Browser tool ─────────────────────────────────────────────────────────────
// Drives Chromium via Puppeteer with tab lifecycle, navigation, and
// interaction helpers. Models OMP's browser automation pattern.

import type { Tool, ToolContext } from "@logician/log-core";
import type { BrowserManager } from "./browser-manager.ts";

export interface BrowserToolDeps {
	manager: BrowserManager;
}

const browserSchema = {
	type: "object",
	properties: {
		operation: {
			type: "string",
			description:
				"Browser operation: open, close, goto, click, type, fill, press, screenshot, evaluate, ariaSnapshot, getUrl, getTitle, waitForSelector, waitForUrl, listTabs, closeAll.",
			enum: [
				"open", "close", "goto", "click", "type", "fill",
				"press", "screenshot", "evaluate", "ariaSnapshot",
				"getUrl", "getTitle", "waitForSelector", "waitForUrl",
				"listTabs", "closeAll",
			],
		},
		name: {
			type: "string",
			description: "Tab name (required for most operations).",
		},
		url: {
			type: "string",
			description: "URL to navigate to (used by open, goto, waitForUrl).",
		},
		selector: {
			type: "string",
			description: "CSS selector (used by click, type, fill, waitForSelector).",
		},
		text: {
			type: "string",
			description: "Text to type (used by type).",
		},
		value: {
			type: "string",
			description: "Value to fill (used by fill).",
		},
		key: {
			type: "string",
			description: "Keyboard key to press (used by press).",
		},
		expression: {
			type: "string",
			description: "JavaScript expression to evaluate in page context (used by evaluate).",
		},
		timeout_ms: {
			type: "integer",
			description: "Timeout in milliseconds (used by waitForSelector, waitForUrl).",
		},
		type_format: {
			type: "string",
			description: "Screenshot format: jpeg or png (used by screenshot).",
		},
		quality: {
			type: "integer",
			description: "Screenshot quality 0-100 (used by screenshot).",
		},
		delay: {
			type: "integer",
			description: "Typing delay in ms between characters (used by type).",
		},
	},
	required: ["operation"],
} as const;

/**
 * Drive a Chromium browser via Puppeteer.
 *
 * Tab names are stable identifiers. A tab persists across calls until
 * explicitly closed. Use `open` to create/reuse a tab, then interact.
 */
export function createBrowserTool(deps: BrowserToolDeps): Tool {
	const manager = deps.manager;

	return {
		name: "browser",
		label: "Browser",
		description:
			"Drive a Chromium browser via Puppeteer. Supports tab lifecycle, " +
			"navigation, element interaction, screenshots, JavaScript evaluation, " +
			"and accessibility snapshots.",
		promptSnippet: "Drive a Chromium browser for testing or scraping",
		promptGuidelines: [
			"Use open(name?, url?) to create or reuse a named tab",
			"Use goto(name, url) to navigate",
			"Use click(name, selector) to interact with elements",
			"Use type(name, selector, text) to input text",
			"Use screenshot(name) to capture the viewport",
			"Use evaluate(name, expression) to run JavaScript",
			"Use ariaSnapshot(name) for accessibility tree inspection",
			"Use close(name) or closeAll() to release resources",
			"Tabs persist across calls — open once, reuse many times",
		],
		readOnly: false,
		executionMode: "sequential",
		parameters: browserSchema,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<string> => {
			const operation = args.operation as string;
			const name = args.name as string | undefined;
			const url = args.url as string | undefined;
			const selector = args.selector as string | undefined;
			const text = args.text as string | undefined;
			const value = args.value as string | undefined;
			const key = args.key as string | undefined;
			const expression = args.expression as string | undefined;
			const timeout_ms = args.timeout_ms as number | undefined;
			const type_format = args.type_format as string | undefined;
			const quality = args.quality as number | undefined;
			const delay = args.delay as number | undefined;

			switch (operation) {
				case "open": {
					const tab = await manager.open(name ?? "default", url);
					return `Opened tab "${tab.name}" at ${tab.url || "blank"}`;
				}

				case "goto": {
					if (!name || !url) return 'goto requires "name" and "url"';
					await manager.goto(name, url);
					const curUrl = await manager.getUrl(name);
					const curTitle = await manager.getTitle(name);
					return `Navigated tab "${name}" to ${url} (now: ${curUrl}, title: ${curTitle})`;
				}

				case "click": {
					if (!name || !selector) return 'click requires "name" and "selector"';
					await manager.click(name, selector);
					return `Clicked "${selector}" on tab "${name}"`;
				}

				case "type": {
					if (!name || !selector || text === undefined) {
						return 'type requires "name", "selector", and "text"';
					}
					await manager.type(name, selector, text, { delay });
					return `Typed "${text}" into "${selector}" on tab "${name}"`;
				}

				case "fill": {
					if (!name || !selector || value === undefined) {
						return 'fill requires "name", "selector", and "value"';
					}
					await manager.fill(name, selector, value);
					return `Filled "${selector}" on tab "${name}"`;
				}

				case "press": {
					if (!name || !key) return 'press requires "name" and "key"';
					await manager.press(name, key);
					return `Pressed "${key}" on tab "${name}"`;
				}

				case "screenshot": {
					if (!name) return 'screenshot requires "name"';
					const buf = await manager.screenshot(name, {
						type: type_format as "jpeg" | "png" | undefined,
						quality,
					});
					const base64 = buf.toString("base64");
					const fmt = type_format || "jpeg";
					return `Screenshot of tab "${name}" (${buf.length} bytes, ${fmt}) [base64: ${base64.slice(0, 100)}...]`;
				}

				case "evaluate": {
					if (!name || !expression) return 'evaluate requires "name" and "expression"';
					const result = await manager.evaluate(name, expression);
					return `evaluated expression on tab "${name}": ${JSON.stringify(result)}`;
				}

				case "ariaSnapshot": {
					if (!name) return 'ariaSnapshot requires "name"';
					const snapshot = await manager.ariaSnapshot(name);
					return `Accessibility snapshot of tab "${name}": ${snapshot.slice(0, 200)}...`;
				}

				case "getUrl": {
					if (!name) return 'getUrl requires "name"';
					return `Tab "${name}" URL: ${await manager.getUrl(name)}`;
				}

				case "getTitle": {
					if (!name) return 'getTitle requires "name"';
					return `Tab "${name}" title: ${await manager.getTitle(name)}`;
				}

				case "waitForSelector": {
					if (!name || !selector) return 'waitForSelector requires "name" and "selector"';
					const found = await manager.waitForSelector(name, selector, timeout_ms);
					return found
						? `Selector "${selector}" found on tab "${name}"`
						: `Selector "${selector}" not found on tab "${name}"`;
				}

				case "waitForUrl": {
					if (!name || !url) return 'waitForUrl requires "name" and "url"';
					const matched = await manager.waitForUrl(name, url, timeout_ms);
					return matched
						? `URL pattern "${url}" matched on tab "${name}"`
						: `URL pattern "${url}" not matched on tab "${name}"`;
				}

				case "listTabs": {
					const tabs = manager.listTabs();
					return `Open tabs: ${tabs.join(", ") || "none"}`;
				}

				case "close": {
					if (!name) return 'close requires "name"';
					await manager.closeTab(name);
					return `Closed tab "${name}"`;
				}

				case "closeAll": {
					await manager.close();
					return "Closed all tabs and browser";
				}

				default:
					return `Unknown operation: "${operation}". Supported: open, close, goto, click, type, fill, press, screenshot, evaluate, ariaSnapshot, getUrl, getTitle, waitForSelector, waitForUrl, listTabs, closeAll.`;
			}
		},
	};
}
