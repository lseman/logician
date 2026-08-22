import { test } from "bun:test";
import assert from "node:assert/strict";
import { LoopRunner } from "../../capabilities/commands/loop-runner.ts";

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

void test("parseInterval validates units and bounds", () => {
	assert.equal(LoopRunner.parseInterval("30s"), 30_000);
	assert.equal(LoopRunner.parseInterval("2H"), 7_200_000);
	assert.equal(LoopRunner.parseInterval("100ms"), 100);
	assert.equal(LoopRunner.parseInterval("0s"), null);
	assert.equal(LoopRunner.parseInterval("soon"), null);
});

void test("loop callbacks never overlap", async () => {
	const manager = new LoopRunner();
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
	const manager = new LoopRunner();
	let aborted = false;
	manager.setOnTick(
		(_iteration, _prompt, signal) =>
			new Promise<void>(resolve => {
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
	const manager = new LoopRunner();
	manager.start("work", 1000);
	const snapshot = manager.getState();
	assert.ok(snapshot);
	assert.throws(() => {
		(snapshot as { iteration: number }).iteration = 99;
	});
	assert.equal(manager.getState()?.iteration, 0);
	manager.stop();
});

void test("repeated callback failures open the circuit breaker", async () => {
	const manager = new LoopRunner();
	manager.setOnTick(() => {
		throw new Error("still broken");
	});
	manager.start("work", 100);
	await delay(750);
	const state = manager.getState();
	assert.equal(state?.status, "stopped");
	assert.equal(state?.consecutiveFailures, 3);
	assert.match(state?.lastError ?? "", /still broken/);
	manager.stop();
});
