// ── hub process manager tests ────────────────────────────────────────────────

import { test, expect } from "bun:test";
import { HubProcessManager, type HubConfig } from "../process-manager.ts";

test("creates manager with default config", () => {
	const mgr = new HubProcessManager();
	expect(mgr).toBeDefined();
});

test("creates manager with custom config", () => {
	const config: HubConfig = {
		defaultTimeoutMs: 5000,
		readinessTimeoutMs: 10000,
		stateDir: "/tmp/test-hub",
	};
	const mgr = new HubProcessManager(config);
	expect(mgr).toBeDefined();
});

test("ps returns empty list initially", () => {
	const mgr = new HubProcessManager();
	const states = mgr.ps();
	expect(states).toEqual([]);
});

test("describe returns null for unknown process", () => {
	const mgr = new HubProcessManager();
	const state = mgr.describe("nonexistent");
	expect(state).toBeNull();
});

test("logs returns null for unknown process", () => {
	const mgr = new HubProcessManager();
	const result = mgr.logs("nonexistent");
	expect(result).toBeNull();
});

test("stop returns error for unknown process", async () => {
	const mgr = new HubProcessManager();
	const res = await mgr.stop("nonexistent");
	expect(res.success).toBe(false);
	expect(res.message).toContain("not found");
});

test("restart throws for unknown process", async () => {
	const mgr = new HubProcessManager();
	await expect(mgr.restart("nonexistent")).rejects.toThrow();
});

test("send returns error for unknown process", () => {
	const mgr = new HubProcessManager();
	const res = mgr.send("nonexistent", "hello");
	expect(res.success).toBe(false);
	expect(res.message).toContain("not found");
});

test("wait returns error for unknown process", async () => {
	const mgr = new HubProcessManager();
	const res = await mgr.wait("nonexistent");
	expect(res.success).toBe(false);
	expect(res.message).toContain("not found");
});

test("cleanupAll works with no processes", () => {
	const mgr = new HubProcessManager();
	mgr.cleanupAll(); // Should not throw
	expect(true).toBe(true);
});

test("ps list after stop is empty", async () => {
	const mgr = new HubProcessManager();
	// Stop non-existent process
	await mgr.stop("test");
	const states = mgr.ps();
	expect(states).toEqual([]);
});
