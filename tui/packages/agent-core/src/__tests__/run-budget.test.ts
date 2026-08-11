import assert from "node:assert/strict";
import { test } from "node:test";
import { RunBudgetController } from "../agent/run-budget.ts";

void test("provider budget preserves finalization reserve", () => {
	const budget = new RunBudgetController({
		maxProviderCalls: 3,
		reserveFinalizationCalls: 1,
	});
	assert.equal(budget.requestProviderCall().allowed, true);
	assert.equal(budget.requestProviderCall().allowed, true);
	assert.equal(budget.requestProviderCall().allowed, false);
	assert.equal(budget.requestProviderCall(true).allowed, true);
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
