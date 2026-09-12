import { expect, spyOn, test } from "bun:test";
import type { Browser } from "puppeteer";
import puppeteer from "puppeteer";
import { BrowserManager } from "../../../capabilities/browser/browser-manager.ts";

test("browser launch failure does not prevent retry", async () => {
	let launches = 0;
	const browser = {
		newPage: async () => {
			throw new Error("page reached");
		},
		close: async () => {},
	} as unknown as Browser;
	const launch = spyOn(puppeteer, "launch").mockImplementation(async () => {
		launches++;
		if (launches === 1) throw new Error("launch failed");
		return browser;
	});
	const manager = new BrowserManager();
	try {
		await expect(manager.open("first")).rejects.toThrow("launch failed");
		await expect(manager.open("second")).rejects.toThrow("page reached");
		expect(launches).toBe(2);
	} finally {
		await manager.close();
		launch.mockRestore();
	}
});
