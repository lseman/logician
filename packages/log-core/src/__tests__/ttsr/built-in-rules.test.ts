// -- TTSR Built-In Rules Tests -------------------------------------------------

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	BUILTIN_RULES,
	getBuiltInRuleByName,
	isBuiltInRule,
	RULE_DOC_GAP,
	RULE_MISSING_ERROR_HANDLING,
	RULE_SECRET_EXPOSURE,
	RULE_SECURITY_ANTIPATTERN,
	RULE_TEST_SKIP,
	RULE_UNSAFE_SHELL,
} from "../../ttsr/built-in-rules.ts";
import { RuleLoader } from "../../ttsr/rule-loader.ts";
import { TtsrManager } from "../../ttsr/ttsr-manager.ts";
import type { TtsrMatchContext, TtsrRule } from "../../types/ttsr.ts";

function textCtx(source: "text" | "thinking" = "text"): TtsrMatchContext {
	return { source };
}

function toolCtx(toolName?: string, filePaths?: string[]): TtsrMatchContext {
	return { source: "tool", toolName, filePaths };
}

// ── Rule Count & Registry ─────────────────────────────────────────────────────

describe("Built-in rule registry", () => {
	it("has exactly 7 built-in rules", () => {
		assert.equal(BUILTIN_RULES.length, 7);
	});

	it("exports all expected rule names", () => {
		const names = BUILTIN_RULES.map(r => r.name);
		assert.ok(names.includes("secret-exposure"));
		assert.ok(names.includes("test-skip"));
		assert.ok(names.includes("unsafe-shell"));
		assert.ok(names.includes("broad-file-write"));
		assert.ok(names.includes("missing-error-handling"));
		assert.ok(names.includes("security-antipattern"));
		assert.ok(names.includes("doc-gap"));
	});

	it("getBuiltInRuleByName returns correct rule", () => {
		const rule = getBuiltInRuleByName("secret-exposure");
		assert.equal(rule?.name, "secret-exposure");
		assert.ok(rule?.content.length > 0);
	});

	it("getBuiltInRuleByName returns undefined for unknown name", () => {
		assert.equal(getBuiltInRuleByName("nonexistent"), undefined);
	});

	it("isBuiltInRule returns true for known names", () => {
		assert.equal(isBuiltInRule("secret-exposure"), true);
		assert.equal(isBuiltInRule("test-skip"), true);
	});

	it("isBuiltInRule returns false for unknown names", () => {
		assert.equal(isBuiltInRule("fake-rule"), false);
	});
});

// ── Secret Exposure Rule ──────────────────────────────────────────────────────

describe("RULE_SECRET_EXPOSURE", () => {
	const rule = RULE_SECRET_EXPOSURE;

	it("matches hardcoded password", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta('password: "mySuperSecret123!"', textCtx());
		assert.equal(matches.length, 1);
	});

	it("matches API key assignment", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			'api_key = "AKIAIOSFODNN7EXAMPLE"',
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("matches access token", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			'access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"',
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("does not match empty password", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta('password: ""', textCtx());
		assert.equal(matches.length, 0);
	});

	it("respects tool scope when set to text only", () => {
		const ruleTextOnly: TtsrRule = { ...rule, scope: ["text"] };
		const mgr = new TtsrManager();
		mgr.addRule(ruleTextOnly);
		const matches = mgr.checkDelta(
			'password: "secret12345678"',
			toolCtx("bash"),
		);
		assert.equal(matches.length, 0);
	});
});

// ── Test Skip Rule ────────────────────────────────────────────────────────────

describe("RULE_TEST_SKIP", () => {
	const rule = RULE_TEST_SKIP;

	it("matches xit", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("xit('should work')", textCtx());
		assert.equal(matches.length, 1);
	});

	it("matches .skip(", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"describe.skip('test', () => {})",
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("matches xdescribe", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("xdescribe('suite', fn)", textCtx());
		assert.equal(matches.length, 1);
	});

	it("does not match normal test names", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"test('it should handle errors', fn)",
			textCtx(),
		);
		assert.equal(matches.length, 0);
	});
});

// ── Unsafe Shell Rule ─────────────────────────────────────────────────────────

describe("RULE_UNSAFE_SHELL", () => {
	const rule = RULE_UNSAFE_SHELL;

	it("matches rm -rf /", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("rm -rf /", toolCtx("bash"));
		assert.equal(matches.length, 1);
	});

	it("matches rm -rf ../", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("rm -rf ../", toolCtx("bash"));
		assert.equal(matches.length, 1);
	});

	it("matches sudo rm -rf", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("sudo rm -rf /tmp/data", toolCtx("bash"));
		assert.equal(matches.length, 1);
	});

	it("does not match safe rm", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("rm -rf /tmp/my-file.log", toolCtx("bash"));
		assert.equal(matches.length, 0);
	});

	it("does not match rm without -rf", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("rm file.txt", toolCtx("bash"));
		assert.equal(matches.length, 0);
	});
});

// ── Security Anti-Pattern Rule ────────────────────────────────────────────────

describe("RULE_SECURITY_ANTIPATTERN", () => {
	const rule = RULE_SECURITY_ANTIPATTERN;

	it("matches eval()", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("eval(userInput)", textCtx());
		assert.equal(matches.length, 1);
	});

	it("matches innerHTML assignment", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("element.innerHTML = data;", textCtx());
		assert.equal(matches.length, 1);
	});

	it("matches SQL string concatenation", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"query('SELECT * FROM users WHERE id=' + userId)",
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("does not match safe code", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"const result = fetch('/api/data').then(res => res.json());",
			textCtx(),
		);
		assert.equal(matches.length, 0);
	});
});

// ── Missing Error Handling Rule ───────────────────────────────────────────────

describe("RULE_MISSING_ERROR_HANDLING", () => {
	const rule = RULE_MISSING_ERROR_HANDLING;

	it("matches bare fetch await", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("await fetch(url)", textCtx());
		assert.equal(matches.length, 1);
	});

	it("matches .then() without catch (heuristic)", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"promise.then(result => console.log(result))",
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("does not match code without fetch/then patterns", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("const x = 42;", textCtx());
		assert.equal(matches.length, 0);
	});
});

// ── Doc Gap Rule ──────────────────────────────────────────────────────────────

describe("RULE_DOC_GAP", () => {
	const rule = RULE_DOC_GAP;

	it("matches TODO with document keyword", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"// TODO: document this function",
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("matches promise to add documentation", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta(
			"I will add documentation for this API",
			textCtx(),
		);
		assert.equal(matches.length, 1);
	});

	it("does not match normal TODO", () => {
		const mgr = new TtsrManager();
		mgr.addRule(rule);
		const matches = mgr.checkDelta("// TODO: fix this bug later", textCtx());
		assert.equal(matches.length, 0);
	});
});

// ── Rule Loader Tests ─────────────────────────────────────────────────────────

describe("RuleLoader", () => {
	it("loads all built-in rules when builtinRules is true", async () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		const count = await loader.loadAll();
		assert.equal(count, 7);
	});

	it("loads zero rules when builtinRules is false", async () => {
		const mgr = new TtsrManager({ builtinRules: false });
		const loader = new RuleLoader(mgr, { builtinRules: false });
		const count = await loader.loadAll();
		assert.equal(count, 0);
	});

	it("respects disabledRules list", async () => {
		const mgr = new TtsrManager({
			builtinRules: true,
			disabledRules: ["secret-exposure", "test-skip"],
		});
		const loader = new RuleLoader(mgr);
		const count = await loader.loadAll();
		assert.equal(count, 5); // 7 - 2 disabled
	});

	it("loads user rules alongside built-ins", async () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		const userRule: TtsrRule = {
			name: "custom-rule",
			path: "user.md",
			description: "custom",
			content: "custom content",
			conditions: ["custom"],
			scope: ["text"],
			interruptMode: "always",
		};
		const count = await loader.loadAll([userRule]);
		assert.equal(count, 8); // 7 built-in + 1 user
	});

	it("tracks rule sources correctly", () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		const userRule: TtsrRule = {
			name: "user-rule",
			path: "user.md",
			description: "u",
			content: "u",
			conditions: ["x"],
			scope: ["text"],
			interruptMode: "always",
		};
		loader.loadAll([userRule]);

		assert.equal(loader.getSource("secret-exposure"), "builtin");
		assert.equal(loader.getSource("user-rule"), "user");
		assert.equal(loader.getSource("nonexistent"), undefined);
	});

	it("unloads user rules but keeps built-ins", () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		const userRule: TtsrRule = {
			name: "unloadable",
			path: "u.md",
			description: "d",
			content: "c",
			conditions: ["x"],
			scope: ["text"],
			interruptMode: "always",
		};
		loader.loadAll([userRule]);

		assert.equal(loader.getLoadedRules().length, 8);
		loader.unloadUserRule("unloadable");
		assert.equal(loader.getLoadedRules().length, 7);
	});

	it("clears all user rules", () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		loader.loadAll([
			{
				name: "a",
				path: "a.md",
				description: "d",
				content: "c",
				conditions: ["x"],
				scope: ["text"] as const,
				interruptMode: "always" as const,
			} as TtsrRule,
			{
				name: "b",
				path: "b.md",
				description: "d",
				content: "c",
				conditions: ["y"],
				scope: ["text"] as const,
				interruptMode: "always" as const,
			} as TtsrRule,
		]);

		assert.equal(loader.getLoadedRules().length, 9);
		loader.clearUserRules();
		assert.equal(loader.getLoadedRules().length, 7);
	});

	it("duplicate rule names are not reloaded", async () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr);
		const count1 = await loader.loadAll();
		const count2 = await loader.loadAll(); // second load should be no-op for built-ins
		assert.equal(count1, 7);
		assert.equal(count2, 0); // already loaded
	});

	it("returns settings", () => {
		const mgr = new TtsrManager({ builtinRules: true });
		const loader = new RuleLoader(mgr, { repeatMode: "gap", repeatGap: 5 });
		const s = loader.getSettings();
		assert.equal(s.builtinRules, true);
		assert.equal(s.repeatMode, "gap");
		assert.equal(s.repeatGap, 5);
	});
});
