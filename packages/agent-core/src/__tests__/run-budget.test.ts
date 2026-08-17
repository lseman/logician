import { test } from "bun:test";
import assert from "node:assert/strict";
import { RunBudgetController } from "../agent/core/run-budget.ts";

void test("provider budget honors an explicit limit", () => {
	const budget = new RunBudgetController({ maxProviderCalls: 2 });
	assert.equal(budget.requestProviderCall().allowed, true);
	assert.equal(budget.requestProviderCall().allowed, true);
	assert.equal(budget.requestProviderCall().allowed, false);
});

void test("tool and elapsed budgets fail closed", () => {
	let now = 0;
	const budget = new RunBudgetController(
		{ maxToolCalls: 2, maxElapsedMs: 10 },
		() => now,
	);
	assert.equal(budget.requestToolBatch(2).allowed, true);
	assert.equal(budget.requestToolBatch(1).allowed, false);
	now = 10;
	assert.match(budget.requestProviderCall().reason ?? "", /elapsed-time/);
});

void test("token accounting restores task-spanning consumption", () => {
	const consumed: Array<{ resource: string; amount: number }> = [];
	const first = new RunBudgetController(
		{ maxTokens: 100 },
		Date.now,
		{ tokens: 70 },
		item => consumed.push(item),
	);
	assert.equal(first.recordTokens(20).allowed, true);
	assert.equal(first.snapshot().tokens, 90);
	assert.equal(first.recordTokens(11).allowed, false);
	assert.equal(first.snapshot().tokens, 101);
	assert.deepEqual(consumed, [
		{ resource: "token", amount: 20 },
		{ resource: "token", amount: 11 },
	]);
});
