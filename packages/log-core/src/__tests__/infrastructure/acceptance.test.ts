import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	validateAcceptanceInput,
} from "../../control/guards/acceptance-contract.ts";
import type { AcceptanceConfig } from "../../system/types/acceptance.ts";

// ── resolveEffectiveAcceptance ──────────────────────────────────────────────

void test("resolve with no config returns none", () => {
	const resolved = resolveEffectiveAcceptance({ explicit: undefined });
	assert.equal(resolved.level, "none");
	assert.equal(resolved.explicit, false);
	assert.equal(resolved.criteria.length, 0);
	assert.equal(resolved.verify.length, 0);
});

void test("resolve with verify returns verified", () => {
	const resolved = resolveEffectiveAcceptance({
		explicit: { verify: [{ id: "test", command: "echo ok" }] },
	});
	assert.equal(resolved.level, "verified");
	assert.equal(resolved.explicit, true);
	assert.equal(resolved.verify.length, 1);
});

void test("resolve with criteria returns none (no verify)", () => {
	const resolved = resolveEffectiveAcceptance({
		explicit: { criteria: ["must pass"] },
	});
	assert.equal(resolved.level, "none");
	assert.equal(resolved.criteria.length, 1);
});

void test("resolve with criteria and verify returns verified", () => {
	const resolved = resolveEffectiveAcceptance({
		explicit: {
			criteria: ["must pass"],
			verify: [{ id: "test", command: "echo ok" }],
		},
	});
	assert.equal(resolved.level, "verified");
	assert.equal(resolved.criteria.length, 1);
	assert.equal(resolved.verify.length, 1);
});

// ── shouldRunAcceptanceFinalization ─────────────────────────────────────────

void test("shouldRun with no verify returns false", () => {
	const resolved = resolveEffectiveAcceptance({ explicit: undefined });
	assert.equal(shouldRunAcceptanceFinalization(resolved), false);
});

void test("shouldRun with verify returns true", () => {
	const resolved = resolveEffectiveAcceptance({
		explicit: { verify: [{ id: "test", command: "echo ok" }] },
	});
	assert.equal(shouldRunAcceptanceFinalization(resolved), true);
});

// ── validateAcceptanceInput ────────────────────────────────────────────────

void test("validate rejects empty criteria", () => {
	const errors = validateAcceptanceInput({ criteria: [""] });
	assert.equal(errors.length > 0, true);
});

void test("validate rejects unknown keys", () => {
	const errors = validateAcceptanceInput({
		criteria: ["ok"],
		
		foo: "bar",
	} as AcceptanceConfig);
	assert.equal(errors.length > 0, true);
});

void test("validate accepts minimal config with verify", () => {
	const errors = validateAcceptanceInput({
		verify: [{ id: "test", command: "echo ok" }],
	});
	assert.deepEqual(errors, []);
});

void test("validate rejects config with neither criteria nor verify", () => {
	const errors = validateAcceptanceInput({});
	assert.equal(errors.length > 0, true);
});
