// ── Browser Manager ──────────────────────────────────────────────────────────
// Manages Chromium tabs via Puppeteer with tab lifecycle, navigation, and
// interaction helpers. Models OMP's browser automation pattern.

import puppeteer, { type KeyInput } from "puppeteer";
import type { Browser, Page } from "puppeteer";

export interface BrowserTab {
	/** Stable name for the tab, scoped to this manager instance. */
	name: string;
	/** Underlying Puppeteer page. */
	page: Page;
	/** Current URL. */
	url: string;
	/** Current page title. */
	title: string;
	/** Whether the tab is still open. */
	closed: boolean;
}

export interface BrowserManagerConfig {
	/** Launch Chromium with headless mode. */
	headless?: boolean;
	/** Maximum wait for page loads in ms. */
	navigationTimeoutMs?: number;
	/** Screenshot quality (0-100, JPEG). */
	screenshotQuality?: number;
}

export interface BrowserManagerDeps {
	config?: BrowserManagerConfig;
}

interface _TabState {
	tab: BrowserTab;
}

/**
 * Manages a set of Chromium browser tabs. Creates one browser instance shared
 * across all tabs, with tab names providing stable addresses.
 */
export class BrowserManager {
	private readonly config: Required<BrowserManagerConfig>;
	private browser: Browser | null = null;
	private tabs: Map<string, _TabState> = new Map();
	private openPromise: Promise<Browser> | null = null;

	constructor(deps: BrowserManagerDeps = {}) {
		this.config = {
			headless: deps.config?.headless ?? true,
			navigationTimeoutMs: deps.config?.navigationTimeoutMs ?? 30_000,
			screenshotQuality: deps.config?.screenshotQuality ?? 80,
		};
	}

	private async ensureBrowser(): Promise<Browser> {
		if (this.browser) return this.browser;
		if (this.openPromise) {
			await this.openPromise;
			if (!this.browser) throw new Error("Browser failed to open");
			return this.browser;
		}
		this.openPromise = this._openBrowser();
		this.browser = await this.openPromise;
		this.openPromise = null;
		return this.browser!;
	}

	private async _openBrowser(): Promise<Browser> {
		return puppeteer.launch({
			headless: this.config.headless,
			args: [
				"--no-sandbox",
				"--disable-setuid-sandbox",
				"--disable-dev-shm-usage",
				"--disable-gpu",
			],
		});
	}

	async open(name: string, url?: string): Promise<BrowserTab> {
		const existing = this.tabs.get(name);
		if (existing && !existing.tab.closed) {
			if (url) await this.goto(name, url);
			return existing.tab;
		}

		const browser = await this.ensureBrowser();
		const page = await browser.newPage();
		await page.setViewport({ width: 1280, height: 800 });

		const tab: BrowserTab = {
			name,
			page,
			url: "",
			title: "",
			closed: false,
		};

		page.on("framenavigated", () => {
			tab.url = page.url();
			page.title().then((t) => (tab.title = t));
		});

		this.tabs.set(name, { tab });

		if (url) await this.goto(name, url);
		return tab;
	}

	async goto(name: string, url: string): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		await state.tab.page.goto(url, {
			waitUntil: "domcontentloaded",
			timeout: this.config.navigationTimeoutMs,
		});
		state.tab.url = state.tab.page.url();
		state.tab.title = await state.tab.page.title();
	}

	async click(name: string, selector: string): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		await state.tab.page.waitForSelector(selector, { timeout: 5_000 });
		await state.tab.page.click(selector);
	}

	async type(
		name: string,
		selector: string,
		text: string,
		options?: { delay?: number },
	): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		await state.tab.page.waitForSelector(selector, { timeout: 5_000 });
		await state.tab.page.type(selector, text, { delay: options?.delay });
	}

	async fill(name: string, selector: string, value: string): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		await state.tab.page.waitForSelector(selector, { timeout: 5_000 });
		await state.tab.page.$eval(
			selector,
			(el: Element, val: string) => {
				// Handle contenteditable elements.
				if (
					"contentEditable" in el &&
					(el as HTMLElement).contentEditable === "true"
				) {
					(el as HTMLElement).innerText = val;
				} else if (el instanceof HTMLInputElement) {
					el.value = val;
					el.dispatchEvent(new Event("input", { bubbles: true }));
					el.dispatchEvent(new Event("change", { bubbles: true }));
				}
			},
			value,
		);
	}

	async press(name: string, key: string): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		await state.tab.page.keyboard.press(key as KeyInput);
	}

	async screenshot(
		name: string,
		options?: { path?: string; type?: "jpeg" | "png"; quality?: number },
	): Promise<Buffer> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		const type = options?.type ?? "jpeg";
		const quality = options?.quality ?? this.config.screenshotQuality;
		return state.tab.page.screenshot({
			type,
			quality: type === "jpeg" ? quality : undefined,
		}) as Promise<Buffer>;
	}

	async ariaSnapshot(name: string): Promise<string> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		const snapshot = await state.tab.page.accessibility.snapshot();
		return JSON.stringify(snapshot, null, 2);
	}

	async evaluate<T>(name: string, expression: string): Promise<T> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		return state.tab.page.evaluate(expression) as Promise<T>;
	}

	async getUrl(name: string): Promise<string> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);
		return state.tab.page.url();
	}

	async getTitle(name: string): Promise<string> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);
		return state.tab.page.title();
	}

	async waitForSelector(
		name: string,
		selector: string,
		timeoutMs?: number,
	): Promise<boolean> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		const timeout = timeoutMs ?? 5_000;
		try {
			await state.tab.page.waitForSelector(selector, { timeout });
			return true;
		} catch {
			return false;
		}
	}

	async waitForUrl(
		name: string,
		pattern: string,
		timeoutMs?: number,
	): Promise<boolean> {
		const state = this.tabs.get(name);
		if (!state) throw new Error(`Tab "${name}" not found`);
		if (state.tab.closed) throw new Error(`Tab "${name}" is closed`);

		const timeout = timeoutMs ?? this.config.navigationTimeoutMs;
		// Convert wildcard pattern to regex.
		const regexSource = pattern.includes("*")
			? `^${pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`
			: pattern;

		try {
			await state.tab.page.waitForFunction(
				((url: string, regex: string) => {
					const re = new RegExp(regex);
					return re.test(url);
				}) as (...args: unknown[]) => boolean,
				{ timeout },
				regexSource,
			);
			return true;
		} catch {
			return false;
		}
	}

	listTabs(): string[] {
		return [...this.tabs.entries()]
			.filter(([, s]) => !s.tab.closed)
			.map(([name]) => name);
	}

	async closeTab(name: string): Promise<void> {
		const state = this.tabs.get(name);
		if (!state) return;
		if (state.tab.closed) return;

		await state.tab.page.close();
		state.tab.closed = true;
		this.tabs.delete(name);
	}

	async close(): Promise<void> {
		const names = this.listTabs();
		for (const name of names) {
			await this.closeTab(name).catch(() => {});
		}

		if (this.browser) {
			await this.browser.close();
			this.browser = null;
		}
	}
}
