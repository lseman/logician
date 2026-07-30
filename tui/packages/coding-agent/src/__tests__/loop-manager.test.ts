import assert from "node:assert/strict";
import { test } from "node:test";
import { LoopManager } from "../application/loop-manager.ts";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

void test("parseInterval validates units and bounds", () => {
	assert.equal(LoopManager.parseInterval("30s"), 30_000);
	assert.equal(LoopManager.parseInterval("2H"), 7_200_000);
	assert.equal(LoopManager.parseInterval("100ms"), 100);
	assert.equal(LoopManager.parseInterval("0s"), null);
	assert.equal(LoopManager.parseInterval("soon"), null);
});

void test("loop callbacks never overlap", async () => {
	const manager = new LoopManager();
	let active = 0;
	let maxActive = 0;
	let calls = 0;
	manager.setOnTick(async () => {
		active++;
		maxActive = Math.max(maxActive, active);
		calls++;
		await delay(140);
		active--;
	});
	manager.start("work", 100);
	await delay(370);
	manager.stop();
	assert.equal(maxActive, 1);
	assert.equal(calls, 2);
});

void test("stop aborts an active callback and prevents rescheduling", async () => {
	const manager = new LoopManager();
	let aborted = false;
	manager.setOnTick((_iteration, _prompt, signal) =>
		new Promise<void>((resolve) => {
			signal.addEventListener("abort", () => {
				aborted = true;
				resolve();
			});
		}),
	);
	manager.start("work", 100);
	await delay(120);
	manager.stop();
	await delay(20);
	assert.equal(aborted, true);
	assert.equal(manager.isActive(), false);
});

void test("state snapshots cannot mutate manager state", () => {
	const manager = new LoopManager();
	manager.start("work", 1000);
	const snapshot = manager.getState();
	assert.ok(snapshot);
	assert.throws(() => {
		(snapshot as { iteration: number }).iteration = 99;
	});
	assert.equal(manager.getState()?.iteration, 0);
	manager.stop();
});
