// -- TTSR Manager Tests --------------------------------------------------------
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { TtsrManager } from "../ttsr-manager.ts";
import type { TtsrMatchContext } from "../../types/ttsr-types.ts";

function textCtx(source: "text" | "thinking" = "text"): TtsrMatchContext {
  return { source };
}

function toolCtx(toolName = "bash", filePaths?: string[]): TtsrMatchContext {
  return { source: "tool", toolName, filePaths };
}

function makeRule(overrides?: Record<string, unknown>): any {
  return {
    name: "test-rule",
    path: "test.md",
    description: "test",
    content: "test content",
    conditions: ["\\bfoo\\b"],
    interruptMode: "always" as const,
    scope: ["text"],
    ...overrides,
  };
}

function mgrWithSettings(overrides: Record<string, unknown>): TtsrManager {
  return new TtsrManager(overrides as any);
}

describe("TtsrManager", () => {
  it("matches basic regex in text", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["secret"] }));
    const matches = mgr.checkDelta("hello secret world", textCtx());
    assert.equal(matches.length, 1);
    assert.equal(matches[0].name, "test-rule");
  });

  it("does not match when pattern absent", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["secret"] }));
    const matches = mgr.checkDelta("hello world", textCtx());
    assert.equal(matches.length, 0);
  });

  it("accumulates stream buffer across deltas", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["secret word"] }));
    const ctx = textCtx();
    assert.equal(mgr.checkDelta("secret", ctx).length, 0);
    const matches = mgr.checkDelta(" word here", ctx);
    assert.equal(matches.length, 1);
  });

  it("respects text scope", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["foo"], scope: ["text"] }));
    const matches = mgr.checkDelta("hello foo world", toolCtx());
    assert.equal(matches.length, 0);
  });

  it("respects tool:bash scope", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["history"], scope: ["tool:bash"] }));
    const matches = mgr.checkDelta("using history", toolCtx("bash"));
    assert.equal(matches.length, 1);
  });

  it("tool:bash scope does not match tool:eval", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["history"], scope: ["tool:bash"] }));
    const matches = mgr.checkDelta("using history", toolCtx("eval"));
    assert.equal(matches.length, 0);
  });

  it("repeatMode:once blocks second trigger", () => {
    const mgr = mgrWithSettings({ repeatMode: "once" });
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    assert.equal(mgr.checkDelta("hello foo", textCtx()).length, 1);
    assert.equal(mgr.checkDelta("hello foo again", textCtx()).length, 0);
  });

  it("repeatMode:gap allows retrigger after N messages", () => {
    const mgr = mgrWithSettings({ repeatMode: "gap", repeatGap: 2 });
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    assert.equal(mgr.checkDelta("hello foo", textCtx()).length, 1);
    mgr.incrementMessageCount();
    assert.equal(mgr.checkDelta("hello foo", textCtx()).length, 0);
    mgr.incrementMessageCount();
    assert.equal(mgr.checkDelta("hello foo", textCtx()).length, 1);
  });

  it("disabled rules cannot be added", () => {
    const mgr = mgrWithSettings({ disabledRules: ["test-rule"] });
    assert.equal(mgr.addRule(makeRule({ conditions: ["foo"] })), false);
  });

  it("enabled:false disables matching", () => {
    const mgr = mgrWithSettings({ enabled: false });
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    assert.equal(mgr.checkDelta("hello foo world", textCtx()).length, 0);
  });

  it("getRules returns all registered rules", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ name: "a", conditions: ["foo"] }));
    mgr.addRule(makeRule({ name: "b", conditions: ["bar"] }));
    assert.equal(mgr.getRules().length, 2);
  });

  it("markInjected tracks injected rules", () => {
    const mgr = mgrWithSettings({ repeatMode: "once" });
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    const ctx = textCtx();
    assert.equal(mgr.checkDelta("hello foo", ctx).length, 1);
    assert.equal(mgr.getInjectedRuleNames().length, 1);
    assert.equal(mgr.checkDelta("hello foo again", ctx).length, 0);
  });

  it("restoreInjected marks rules as already injected", () => {
    const mgr = mgrWithSettings({ repeatMode: "once" });
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    mgr.restoreInjected(["test-rule"]);
    assert.equal(mgr.checkDelta("hello foo", textCtx()).length, 0);
  });

  it("resetBuffer clears stream accumulation", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["secret word"] }));
    const ctx = textCtx();
    mgr.checkDelta("secret", ctx);
    mgr.resetBuffer();
    assert.equal(mgr.checkDelta(" word here", ctx).length, 0);
  });

  it("hasRules returns correct state", () => {
    const mgr = new TtsrManager();
    assert.equal(mgr.hasRules(), false);
    mgr.addRule(makeRule({ conditions: ["foo"] }));
    assert.equal(mgr.hasRules(), true);
  });

  it("hasAstRules returns true when AST conditions present", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: [], astConditions: ["{ name: $_ }"], scope: ["tool"] }));
    assert.equal(mgr.hasAstRules(), true);
  });

  it("scope:thinking matches thinking tokens", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["bad idea"], scope: ["thinking"] }));
    assert.equal(mgr.checkDelta("that is a bad idea", textCtx("thinking")).length, 1);
    assert.equal(mgr.checkDelta("that is a bad idea", textCtx("text")).length, 0);
  });

  it("scope:tool matches any tool", () => {
    const mgr = new TtsrManager();
    mgr.addRule(makeRule({ conditions: ["history"], scope: ["tool"] }));
    assert.equal(mgr.checkDelta("using history", toolCtx("bash")).length, 1);
    assert.equal(mgr.checkDelta("using history", toolCtx("eval")).length, 0);
  });
});
