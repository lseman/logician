// ── Compaction Methods Tests ─────────────────────────────────────────────────
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
	COMPACTION_METHOD_CHOICES,
	DEFAULT_COMPACTION_METHOD_ORDER,
	getCompactionMethodDescription,
	getCompactionMethodLabel,
	isCompactionMethod,
	resolveCompactionMethodOrder,
} from "../src/compaction-methods/index.ts";
import {
	getFallbackChain,
	isMethodAvailable,
	resolveCompactionMethod,
	shouldUseServerCompaction,
} from "../src/compaction-methods/resolver.ts";

describe("Compaction Method Types", () => {
	it("has exactly 5 compaction methods", () => {
		assert.equal(COMPACTION_METHOD_CHOICES.length, 5);
	});

	it("includes all expected method values", () => {
		const values = COMPACTION_METHOD_CHOICES.map(c => c.value);
		assert.ok(values.includes("remote"));
		assert.ok(values.includes("snapcompact"));
		assert.ok(values.includes("handoff"));
		assert.ok(values.includes("shake"));
		assert.ok(values.includes("soft"));
	});

	it("has labels for all methods", () => {
		for (const choice of COMPACTION_METHOD_CHOICES) {
			assert.ok(choice.label.length > 0, `label missing for ${choice.value}`);
		}
	});

	it("has descriptions for all methods", () => {
		for (const choice of COMPACTION_METHOD_CHOICES) {
			assert.ok(choice.description.length > 0, `description missing for ${choice.value}`);
		}
	});

	it("default order has all 5 methods", () => {
		assert.equal(DEFAULT_COMPACTION_METHOD_ORDER.length, 5);
		// Default: remote first (server-native), shake before soft
		assert.equal(DEFAULT_COMPACTION_METHOD_ORDER[0], "remote");
		assert.equal(DEFAULT_COMPACTION_METHOD_ORDER[4], "soft");
	});
});

describe("isCompactionMethod", () => {
	it("returns true for valid methods", () => {
		assert.ok(isCompactionMethod("remote"));
		assert.ok(isCompactionMethod("snapcompact"));
		assert.ok(isCompactionMethod("handoff"));
		assert.ok(isCompactionMethod("shake"));
		assert.ok(isCompactionMethod("soft"));
	});

	it("returns false for invalid values", () => {
		assert.equal(isCompactionMethod("invalid"), false);
		assert.equal(isCompactionMethod(""), false);
		assert.equal(isCompactionMethod(null), false);
		assert.equal(isCompactionMethod(undefined), false);
		assert.equal(isCompactionMethod(123), false);
	});
});

describe("resolveCompactionMethodOrder", () => {
	it("filters malformed entries", () => {
		const result = resolveCompactionMethodOrder(["remote", "invalid", "shake"]);
		assert.deepEqual(result, ["remote", "shake"]);
	});

	it("preserves first-occurrence order", () => {
		const result = resolveCompactionMethodOrder(["soft", "shake", "handoff"]);
		assert.deepEqual(result, ["soft", "shake", "handoff"]);
	});

	it("removes duplicates", () => {
		const result = resolveCompactionMethodOrder(["shake", "remote", "shake"]);
		assert.deepEqual(result, ["shake", "remote"]);
	});

	it("returns empty for non-array input", () => {
		assert.deepEqual(resolveCompactionMethodOrder(null), []);
		assert.deepEqual(resolveCompactionMethodOrder(undefined), []);
		assert.deepEqual(resolveCompactionMethodOrder("shake"), []);
	});
});

describe("getCompactionMethodLabel", () => {
	it("returns label for valid method", () => {
		assert.equal(getCompactionMethodLabel("shake"), "Shake");
		assert.equal(getCompactionMethodLabel("snapcompact"), "Snapcompact");
	});

	it("falls back to method name for invalid", () => {
		assert.equal(getCompactionMethodLabel("invalid"), "invalid");
	});
});

describe("getCompactionMethodDescription", () => {
	it("returns description for valid method", () => {
		const desc = getCompactionMethodDescription("shake");
		assert.ok(desc.length > 0);
		assert.ok(desc.includes("heavy content"));
	});

	it("returns empty string for invalid method", () => {
		assert.equal(getCompactionMethodDescription("invalid"), "");
	});
});

describe("resolveCompactionMethod", () => {
	it("returns snapcompact when mode is snapcompact", () => {
		const result = resolveCompactionMethod({ mode: "snapcompact" }, {
			serverCompactionAvailable: false,
			snapcompactAvailable: false, // even if unavailable, mode override wins
			llmAvailable: true,
		});
		assert.equal(result, "snapcompact");
	});

	it("uses configured method order when available", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: false,
			llmAvailable: true,
		};
		const result = resolveCompactionMethod(
			{ methodOrder: ["shake", "soft"] },
			caps,
		);
		assert.equal(result, "shake"); // shake is always available
	});

	it("skips unavailable methods and falls back", () => {
		const caps = {
			serverCompactionAvailable: true,
			snapcompactAvailable: false,
			llmAvailable: true,
		};
		const result = resolveCompactionMethod(
			{ methodOrder: ["snapcompact", "shake"] },
			caps,
		);
		assert.equal(result, "shake"); // snapcompact unavailable, falls to shake
	});

	it("returns null when no methods available", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: false,
			llmAvailable: false,
		};
		const result = resolveCompactionMethod(
			{ methodOrder: ["snapcompact", "handoff"] },
			caps,
		);
		assert.equal(result, null);
	});

	it("uses single method override when provided", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: true,
			llmAvailable: true,
		};
		const result = resolveCompactionMethod({ method: "snapcompact" }, caps);
		assert.equal(result, "snapcompact");
	});

	it("uses default order when no settings provided", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: true,
			llmAvailable: true,
		};
		const result = resolveCompactionMethod({}, caps);
		assert.equal(result, "snapcompact"); // snapcompact is 2nd in default order and available
	});

	it("prefers remote when server compaction available", () => {
		const caps = {
			serverCompactionAvailable: true,
			snapcompactAvailable: false,
			llmAvailable: true,
		};
		const result = resolveCompactionMethod({}, caps);
		assert.equal(result, "remote");
	});
});

describe("isMethodAvailable", () => {
	it("shake is always available", () => {
		assert.ok(isMethodAvailable("shake"));
		assert.ok(
			isMethodAvailable("shake", {
				serverCompactionAvailable: false,
				snapcompactAvailable: false,
				llmAvailable: false,
			}),
		);
	});

	it("snapcompact requires snapcompactAvailable", () => {
		assert.ok(
			isMethodAvailable("snapcompact", {
				serverCompactionAvailable: false,
				snapcompactAvailable: true,
				llmAvailable: false,
			}),
		);
		assert.equal(
			isMethodAvailable("snapcompact", {
				serverCompactionAvailable: false,
				snapcompactAvailable: false,
				llmAvailable: true,
			}),
			false,
		);
	});

	it("remote requires serverCompactionAvailable", () => {
		assert.ok(
			isMethodAvailable("remote", {
				serverCompactionAvailable: true,
				snapcompactAvailable: false,
				llmAvailable: false,
			}),
		);
		assert.equal(
			isMethodAvailable("remote", {
				serverCompactionAvailable: false,
				snapcompactAvailable: false,
				llmAvailable: false,
			}),
			false,
		);
	});

	it("handoff and soft require llmAvailable", () => {
		assert.ok(isMethodAvailable("handoff", { llmAvailable: true }));
		assert.ok(isMethodAvailable("soft", { llmAvailable: true }));
		assert.equal(
			isMethodAvailable("handoff", { llmAvailable: false }),
			false,
		);
	});
});

describe("getFallbackChain", () => {
	it("returns filtered available methods in order", () => {
		const caps = {
			serverCompactionAvailable: true,
			snapcompactAvailable: false,
			llmAvailable: true,
		};
		const chain = getFallbackChain({ methodOrder: ["remote", "snapcompact", "shake"] }, caps);
		assert.deepEqual(chain, ["remote", "shake"]); // snapcompact filtered out
	});

	it("returns empty when no methods available", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: false,
			llmAvailable: false,
		};
		const chain = getFallbackChain(
			{ methodOrder: ["remote", "handoff"] },
			caps,
		);
		assert.deepEqual(chain, []);
	});

	it("handles single method override", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: true,
			llmAvailable: true,
		};
		const chain = getFallbackChain({ method: "snapcompact" }, caps);
		assert.deepEqual(chain, ["snapcompact"]);
	});
});

describe("shouldUseServerCompaction", () => {
	it("returns true when remote is resolved", () => {
		const caps = {
			serverCompactionAvailable: true,
			snapcompactAvailable: false,
			llmAvailable: false,
		};
		assert.ok(shouldUseServerCompaction({ methodOrder: ["remote"] }, caps));
	});

	it("returns false when remote is not resolved", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: true,
			llmAvailable: true,
		};
		assert.equal(shouldUseServerCompaction({}, caps), false);
	});

	it("returns false when remote unavailable", () => {
		const caps = {
			serverCompactionAvailable: false,
			snapcompactAvailable: false,
			llmAvailable: true,
		};
		assert.equal(shouldUseServerCompaction({ methodOrder: ["remote", "shake"] }, caps), false);
	});
});
